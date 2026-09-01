"""Benchmark dense-reference and direct-sparse MoE training execution."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tfs_moe_fusion.config import load_config
from tfs_moe_fusion.moe import (
    ExpertContext,
    FunctionalMoEBlock,
    RouterContext,
)
from tfs_moe_fusion.types import ModalityType, TaskType


@dataclass(slots=True)
class Result:
    mode: str
    batch_size: int
    mean_step_ms: float
    samples_per_second: float
    peak_memory_mib: float | None
    expert_sample_assignments: int


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def contexts(feature: torch.Tensor) -> tuple[RouterContext, ExpertContext]:
    batch = feature.shape[0]
    source_a = torch.randn_like(feature)
    source_b = torch.randn_like(feature)
    router = RouterContext(
        feature=feature,
        task=TaskType.VIF,
        modality_a=ModalityType.VISIBLE_RGB,
        modality_b=ModalityType.INFRARED_GRAY,
        low_energy=torch.rand(batch, device=feature.device),
        high_energy=torch.rand(batch, device=feature.device),
    )
    expert = ExpertContext(
        task=TaskType.VIF,
        modality_a=ModalityType.VISIBLE_RGB,
        modality_b=ModalityType.INFRARED_GRAY,
        source_a_feature=source_a,
        source_b_feature=source_b,
    )
    return router, expert


def benchmark(
    block: FunctionalMoEBlock,
    feature: torch.Tensor,
    *,
    sparse: bool,
    warmup: int,
    iterations: int,
) -> Result:
    device = feature.device
    router_context, expert_context = contexts(feature)

    def step() -> int:
        block.zero_grad(set_to_none=True)
        feature.grad = None
        output = block(
            feature,
            router_context,
            expert_context,
            sparse_execution=sparse,
        )
        output.feature.square().mean().backward()
        return int(output.diagnostics.auxiliary["expert_sample_assignments"])

    for _ in range(warmup):
        step()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    assignments = 0
    for _ in range(iterations):
        assignments = step()
    synchronize(device)
    elapsed = time.perf_counter() - started
    mean_seconds = elapsed / iterations
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else None
    )
    return Result(
        mode="sparse_batch" if sparse else "dense_masked_reference",
        batch_size=feature.shape[0],
        mean_step_ms=mean_seconds * 1000,
        samples_per_second=feature.shape[0] / mean_seconds,
        peak_memory_mib=peak,
        expert_sample_assignments=assignments,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    config = load_config(ROOT / "configs/default.yaml")
    config.model.moe.noisy_topk = False
    block = (
        FunctionalMoEBlock(args.channels, config.model.moe, "benchmark.moe0")
        .to(device)
        .train()
    )
    feature = torch.randn(
        args.batch_size,
        args.channels,
        args.height,
        args.width,
        device=device,
        requires_grad=True,
    )

    dense = benchmark(
        block,
        feature,
        sparse=False,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    sparse = benchmark(
        block,
        feature,
        sparse=True,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    payload = {
        "device": str(device),
        "dense": asdict(dense),
        "sparse": asdict(sparse),
        "speedup": dense.mean_step_ms / sparse.mean_step_ms,
        "assignment_reduction": (
            1 - sparse.expert_sample_assignments / dense.expert_sample_assignments
        ),
        "memory_reduction": (
            1 - sparse.peak_memory_mib / dense.peak_memory_mib
            if dense.peak_memory_mib and sparse.peak_memory_mib
            else None
        ),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
