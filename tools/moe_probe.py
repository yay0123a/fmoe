"""Compare learned MoE routing against deterministic runtime ablations."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tfs_moe_fusion.config import load_config
from tfs_moe_fusion.losses import LossContext, MultiTaskLossManager
from tfs_moe_fusion.model import build_model
from tfs_moe_fusion.moe import (
    clear_moe_routing_override,
    set_moe_routing_override,
)
from tfs_moe_fusion.trainer import (
    active_phase,
    build_batch_provider,
    load_checkpoint,
)
from tfs_moe_fusion.types import ExpertType, FusionBatch, RouterDiagnostics, TaskType
from tfs_moe_fusion.utils import resolve_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True, choices=[item.value for item in TaskType])
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--single-expert",
        action="append",
        choices=[item.value for item in ExpertType],
        default=[],
        help="Also probe single:<expert>; repeat to select multiple experts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON destination (default: moe_probe_<task>.json).",
    )
    return parser


def _autocast_context(precision: str, device: torch.device):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    if device.type == "cpu" and dtype is torch.float16:
        raise RuntimeError("FP16 MoE probing is not supported on CPU")
    return torch.autocast(device.type, dtype=dtype)


def _add_vector(target: list[float], value: torch.Tensor, weight: int) -> None:
    detached = value.detach().float().cpu().tolist()
    if not target:
        target.extend(0.0 for _ in detached)
    if len(target) != len(detached):
        raise RuntimeError("Router diagnostic vector size changed between batches")
    for index, item in enumerate(detached):
        target[index] += float(item) * weight


def _update_scalar_map(
    sums: dict[str, float],
    counts: dict[str, int],
    values: dict[str, Any],
    weight: int,
) -> None:
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        sums[name] = sums.get(name, 0.0) + float(value.detach().float()) * weight
        counts[name] = counts.get(name, 0) + weight


def _update_block(
    states: dict[str, dict[str, Any]], diagnostic: RouterDiagnostics
) -> None:
    batch_size = diagnostic.probabilities.shape[0]
    expert_names = tuple(diagnostic.auxiliary.get("expert_names", ()))
    if not expert_names:
        expert_names = tuple(str(index) for index in range(diagnostic.logits.shape[1]))
    state = states.setdefault(
        diagnostic.block_id,
        {
            "samples": 0,
            "expert_names": expert_names,
            "importance": [],
            "hard_load": [],
            "entropy": 0.0,
            "top1_margin": 0.0,
            "raw_sums": {},
            "raw_counts": {},
            "contribution_sums": {},
            "contribution_counts": {},
        },
    )
    if state["expert_names"] != expert_names:
        raise RuntimeError(f"Expert order changed for block {diagnostic.block_id!r}")
    state["samples"] += batch_size
    importance = (
        diagnostic.importance
        if diagnostic.importance is not None
        else diagnostic.probabilities.detach().mean(
            tuple(index for index in range(diagnostic.probabilities.ndim) if index != 1)
        )
    )
    hard_load = (
        diagnostic.hard_load
        if diagnostic.hard_load is not None
        else (
            torch.nn.functional.one_hot(
                diagnostic.topk_indices, diagnostic.probabilities.shape[1]
            )
            .float()
            .mean((0, 1))
            if diagnostic.topk_indices is not None
            else torch.nn.functional.one_hot(
                diagnostic.probabilities.argmax(1), diagnostic.probabilities.shape[1]
            )
            .float()
            .mean((0, 1, 2))
        )
    )
    _add_vector(state["importance"], importance, batch_size)
    _add_vector(state["hard_load"], hard_load, batch_size)
    entropy = diagnostic.auxiliary.get("router/entropy")
    if entropy is None:
        entropy = diagnostic.entropy.detach().float().mean()
    margin = diagnostic.auxiliary.get("router/top1_margin")
    if margin is None:
        top = diagnostic.probabilities.detach().float().topk(2, dim=1).values
        margin = (top[:, 0] - top[:, 1]).mean()
    state["entropy"] += float(torch.as_tensor(entropy).detach().float()) * batch_size
    state["top1_margin"] += (
        float(torch.as_tensor(margin).detach().float()) * batch_size
    )
    _update_scalar_map(
        state["raw_sums"],
        state["raw_counts"],
        diagnostic.auxiliary.get("expert_residual_rms", {}),
        batch_size,
    )
    _update_scalar_map(
        state["contribution_sums"],
        state["contribution_counts"],
        diagnostic.auxiliary.get("expert_weighted_contribution_rms", {}),
        batch_size,
    )


def _finalize_blocks(states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for block_id, state in states.items():
        samples = state["samples"]
        result[block_id] = {
            "expert_names": list(state["expert_names"]),
            "importance": [value / samples for value in state["importance"]],
            "hard_load": [value / samples for value in state["hard_load"]],
            "entropy": state["entropy"] / samples,
            "top1_margin": state["top1_margin"] / samples,
            "expert_residual_rms": {
                name: value / state["raw_counts"][name]
                for name, value in state["raw_sums"].items()
            },
            "expert_weighted_contribution_rms": {
                name: value / state["contribution_counts"][name]
                for name, value in state["contribution_sums"].items()
            },
        }
    return result


@torch.no_grad()
def evaluate_mode(
    model: torch.nn.Module,
    loss_manager: MultiTaskLossManager,
    batches: Iterable[FusionBatch],
    *,
    mode: str,
    seed: int,
    device: torch.device,
    precision: str,
    epoch: int,
    global_step: int,
    phase_name: str,
    loss_multipliers: dict[str, float],
) -> dict[str, Any]:
    set_moe_routing_override(model, mode, seed)
    total_sum = 0.0
    sample_count = 0
    component_sums: dict[str, float] = {}
    component_counts: dict[str, int] = {}
    weighted_sums: dict[str, float] = {}
    weighted_counts: dict[str, int] = {}
    block_states: dict[str, dict[str, Any]] = {}

    for batch in batches:
        batch_size = len(batch.sample_ids)
        device_batch = batch.to(device, non_blocking=device.type == "cuda")
        with _autocast_context(precision, device):
            output = model(device_batch)
            loss = loss_manager(
                LossContext(
                    device_batch,
                    output,
                    device_batch.task,
                    epoch,
                    global_step,
                    model,
                    phase=phase_name,
                    loss_multipliers=loss_multipliers,
                )
            )
        total_sum += float(loss.total.detach().float()) * batch_size
        sample_count += batch_size
        _update_scalar_map(
            component_sums, component_counts, loss.components, batch_size
        )
        _update_scalar_map(
            weighted_sums, weighted_counts, loss.weighted_components, batch_size
        )
        for diagnostic in output.router_diagnostics:
            _update_block(block_states, diagnostic)

    if sample_count == 0:
        raise RuntimeError("MoE probe received no samples")
    return {
        "mode": mode,
        "total_loss": total_sum / sample_count,
        "loss_components": {
            name: value / component_counts[name]
            for name, value in component_sums.items()
        },
        "weighted_loss_components": {
            name: value / weighted_counts[name]
            for name, value in weighted_sums.items()
        },
        "blocks": _finalize_blocks(block_states),
    }


def relative_difference(value: float, reference: float, denominator: float) -> float:
    if denominator == 0:
        raise ZeroDivisionError("Cannot compute routing metric with a zero loss")
    return (value - reference) / abs(denominator)


def _print_summary(report: dict[str, Any], destination: Path) -> None:
    print(f"MoE routing probe: task={report['task']} batches={report['batches']}")
    for mode, result in report["modes"].items():
        print(f"  {mode:<26} total_loss={result['total_loss']:.8f}")
    print(f"  routing_advantage={report['routing_advantage']:.8f}")
    print(f"  shuffle_penalty={report['shuffle_penalty']:.8f}")
    print(f"JSON: {destination}")


def main() -> None:
    args = build_parser().parse_args()
    if args.batches <= 0:
        raise ValueError("--batches must be positive")
    config = load_config(args.config)
    device = resolve_device(args.device or config.training.device)
    seed_everything(args.seed, config.experiment.deterministic)
    task = TaskType.parse(args.task)
    model = build_model(config).to(device).eval()
    checkpoint = load_checkpoint(
        args.checkpoint,
        model,
        strict=True,
        map_location=device,
    )
    loss_manager = MultiTaskLossManager(config.training.losses).to(device).eval()
    provider = build_batch_provider(config)
    try:
        batches = [provider.next_batch(task) for _ in range(args.batches)]
    finally:
        provider.close()

    phase = active_phase(config.training.phases.phases, checkpoint.epoch)
    modes = ["learned", "uniform", "shuffled"]
    modes.extend(f"single:{expert}" for expert in args.single_expert)
    results: dict[str, Any] = {}
    try:
        for mode in modes:
            results[mode] = evaluate_mode(
                model,
                loss_manager,
                batches,
                mode=mode,
                seed=args.seed,
                device=device,
                precision=config.training.precision,
                epoch=checkpoint.epoch,
                global_step=checkpoint.global_step,
                phase_name=phase.name,
                loss_multipliers=dict(phase.loss_multipliers),
            )
    finally:
        clear_moe_routing_override(model)

    learned_loss = results["learned"]["total_loss"]
    uniform_loss = results["uniform"]["total_loss"]
    shuffled_loss = results["shuffled"]["total_loss"]
    report = {
        "task": task.value,
        "batches": args.batches,
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "modes": results,
        "routing_advantage": relative_difference(
            uniform_loss, learned_loss, uniform_loss
        ),
        "shuffle_penalty": relative_difference(
            shuffled_loss, learned_loss, learned_loss
        ),
    }
    destination = args.output or Path(f"moe_probe_{task.value}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _print_summary(report, destination)


if __name__ == "__main__":
    main()
