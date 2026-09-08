"""TFS-MoE-Fusion consolidated implementation."""

from __future__ import annotations

import os
import random
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from tfs_moe_fusion.config import ProjectConfig

CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True, slots=True)
class CheckpointLoadReport:
    path: Path
    epoch: int
    global_step: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    sampler_state: dict[str, Any] | None
    scheduler_state: dict[str, Any] | None
    scaler_state: dict[str, Any] | None
    ema_state: dict[str, Any] | None
    engine_state: dict[str, Any] | None
    metadata: dict[str, Any]


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: ProjectConfig,
    *,
    epoch: int,
    global_step: int,
    optimizer: Optimizer | None = None,
    scheduler_state: dict[str, Any] | None = None,
    scaler_state: dict[str, Any] | None = None,
    ema_state: dict[str, Any] | None = None,
    sampler_state: dict[str, Any] | None = None,
    engine_state: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler_state,
        "scaler": scaler_state,
        "ema": ema_state,
        "sampler": sampler_state,
        "engine": engine_state,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": asdict(config),
        "metadata": metadata or {},
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        },
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: Optimizer | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = False,
) -> CheckpointLoadReport:
    source = Path(path)
    payload = torch.load(source, map_location=map_location, weights_only=False)
    version = payload.get("format_version")
    if version not in {1, CHECKPOINT_FORMAT_VERSION}:
        raise RuntimeError(
            f"Unsupported checkpoint format {version}; expected {CHECKPOINT_FORMAT_VERSION}"
        )

    incompatible = model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        # ``map_location`` applies to every tensor in the checkpoint, including
        # RNG states.  CPU RNG state restoration only accepts a CPU ByteTensor,
        # so move device-mapped states back before handing them to PyTorch.
        torch.set_rng_state(payload["rng"]["torch"].detach().cpu())
        if payload["rng"].get("python") is not None:
            random.setstate(payload["rng"]["python"])
        if payload["rng"].get("numpy") is not None:
            np.random.set_state(payload["rng"]["numpy"])
        cuda_state = payload["rng"].get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            # A checkpoint may have been saved while more GPUs were visible.
            # ``set_rng_state_all`` indexes the current CUDA generators, so
            # passing surplus states raises once it reaches a hidden device.
            visible_cuda_count = torch.cuda.device_count()
            torch.cuda.set_rng_state_all(
                [
                    state.detach().cpu()
                    for state in cuda_state[:visible_cuda_count]
                ]
            )

    return CheckpointLoadReport(
        path=source,
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
        sampler_state=payload.get("sampler"),
        scheduler_state=payload.get("scheduler"),
        scaler_state=payload.get("scaler"),
        ema_state=payload.get("ema"),
        engine_state=payload.get("engine"),
        metadata=dict(payload.get("metadata", {})),
    )


from dataclasses import dataclass, field

from tfs_moe_fusion.types import TaskType


@dataclass(slots=True)
class StatefulTaskSampler:
    weights: dict[TaskType, float]
    seed: int = 3407
    _random: random.Random = field(init=False, repr=False)
    _draws: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if not self.weights or not set(self.weights) <= set(TaskType):
            raise ValueError("weights must define a non-empty subset of TaskType")
        if any(value <= 0 for value in self.weights.values()):
            raise ValueError("all task weights must be positive")
        self._random = random.Random(self.seed)
        self._draws = 0

    @classmethod
    def from_strings(
        cls, weights: dict[str, float], seed: int = 3407
    ) -> StatefulTaskSampler:
        return cls({TaskType.parse(key): value for key, value in weights.items()}, seed)

    def next_task(self) -> TaskType:
        tasks = list(self.weights)
        selected = self._random.choices(
            tasks, weights=[self.weights[task] for task in tasks], k=1
        )[0]
        self._draws += 1
        return selected

    def state_dict(self) -> dict[str, Any]:
        return {
            "weights": {task.value: value for task, value in self.weights.items()},
            "seed": self.seed,
            "draws": self._draws,
            "random_state": self._random.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        restored = {
            TaskType.parse(key): float(value) for key, value in state["weights"].items()
        }
        if restored != self.weights:
            raise ValueError("Task sampler weights differ from checkpoint")
        self.seed = int(state["seed"])
        self._draws = int(state["draws"])
        self._random.setstate(state["random_state"])


from contextlib import nullcontext


class AMPController:
    def __init__(self, precision: str, device: torch.device) -> None:
        if (
            precision == "bf16"
            and device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "This CUDA device does not support BF16; set training.precision=fp16"
            )
        self.precision, self.device = precision, device
        self.dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        self.enabled = precision != "fp32"
        self.scaler = torch.amp.GradScaler(
            device.type, enabled=precision == "fp16" and device.type == "cuda"
        )

    def autocast(self):
        if not self.enabled:
            return nullcontext()
        return torch.autocast(self.device.type, dtype=self.dtype)

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def unscale_(self, optimizer) -> None:
        self.scaler.unscale_(optimizer)

    def step(self, optimizer) -> None:
        self.scaler.step(optimizer)
        self.scaler.update()


from torch import Tensor
from torch.nn.parallel import DistributedDataParallel


def distributed_available() -> bool:
    return (
        torch.distributed.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1
    )


def initialize_distributed(device: torch.device) -> tuple[int, int]:
    if not distributed_available():
        return 0, 1
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo"
        )
    return torch.distributed.get_rank(), torch.distributed.get_world_size()


def wrap_ddp(model: nn.Module, device: torch.device, find_unused: bool) -> nn.Module:
    if not distributed_available():
        return model
    kwargs = {"find_unused_parameters": find_unused}
    if device.type == "cuda":
        kwargs["device_ids"] = [device.index]
    return DistributedDataParallel(model, **kwargs)


def reduce_mean(value: Tensor) -> Tensor:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value
    result = value.detach().clone()
    torch.distributed.all_reduce(result)
    return result / torch.distributed.get_world_size()


from collections.abc import Iterator
from contextlib import contextmanager


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        for name, value in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].lerp_(value.detach(), 1 - self.decay)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.decay = float(state["decay"])
        self.updates = int(state["updates"])
        self.shadow = dict(state["shadow"])  # type: ignore[arg-type]

    @contextmanager
    def apply(self, model: nn.Module) -> Iterator[None]:
        backup: dict[str, Tensor] = {}
        state = model.state_dict()
        with torch.no_grad():
            for name, value in self.shadow.items():
                backup[name] = state[name].detach().clone()
                state[name].copy_(value)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, value in backup.items():
                    state[name].copy_(value)


from dataclasses import dataclass

from torch.optim import AdamW

from tfs_moe_fusion.config import OptimizerConfig

GROUP_NAMES = (
    "shared_backbone",
    "common_experts",
    "low_frequency_experts",
    "detail_experts",
    "semantic_experts",
    "ir_experts",
    "coarse_routers",
    "focus_head",
    "guidance_pyramid",
    "feedback_experts",
    "feedback_routers",
    "shared_common",
    "shared_low",
    "shared_detail",
    "shared_semantic",
    "shared_ir",
    "shared_focus",
    "core_site_adapters",
    "feedback_site_adapters",
    "core_routers",
    "rgb_stem",
    "ir_stem",
    "gray_stem",
    "source_backbone",
    "fused_backbone",
    "cross_modal_fusion",
    "coarse_decoder_trunk",
    "vif_coarse_head",
    "mfif_coarse_head",
    "mfif_interactions",
    "refinement_decoder_trunk",
    "mfif_residual_head",
    "vif_residual_head",
    "core_common_scale",
    "feedback_common_scale",
    "core_specialist_scale",
    "feedback_specialist_scale",
)


def parameter_group_name(name: str, shared_pool_enabled: bool = False) -> str:
    if name.startswith("core.moe_blocks") and name.endswith(".common_scale"):
        return "core_common_scale"
    if name.startswith("core.moe_blocks") and name.endswith(".specialist_scale"):
        return "core_specialist_scale"
    if name.startswith("feedback.feedback_moe") and name.endswith(".common_scale"):
        return "feedback_common_scale"
    if name.startswith("feedback.feedback_moe") and name.endswith(
        ".specialist_scale"
    ):
        return "feedback_specialist_scale"
    if shared_pool_enabled:
        shared_experts = {
            "shared_expert_bank.common.": "shared_common",
            "shared_expert_bank.specialists.low_frequency.": "shared_low",
            "shared_expert_bank.specialists.detail.": "shared_detail",
            "shared_expert_bank.specialists.semantic.": "shared_semantic",
            "shared_expert_bank.specialists.infrared_saliency.": "shared_ir",
            "shared_expert_bank.specialists.focus.": "shared_focus",
        }
        for prefix, group in shared_experts.items():
            if name.startswith(prefix):
                return group
        if name.startswith("core.task_embedding") or (
            name.startswith("core.moe_blocks") and ".router." in name
        ):
            return "core_routers"
        if name.startswith("core.moe_blocks"):
            return "core_site_adapters"
        if name.startswith("feedback.feedback_moe"):
            return (
                "feedback_routers"
                if ".router." in name
                else "feedback_site_adapters"
            )
    structural_groups = (
        ("core.stems.rgb.", "rgb_stem"),
        ("core.stems.infrared.", "ir_stem"),
        ("core.stems.gray.", "gray_stem"),
        ("core.source_stages.", "source_backbone"),
        ("core.source_downsamples.", "source_backbone"),
        ("core.fused_stages.", "fused_backbone"),
        ("core.fused_downsamples.", "fused_backbone"),
        ("core.cross_modal_fusions.", "cross_modal_fusion"),
        ("core.decoder_stages.", "coarse_decoder_trunk"),
        ("core.fusion_y_head.", "vif_coarse_head"),
        ("core.mfif_rgb_head.", "mfif_coarse_head"),
        ("feedback.mfif_interactions.", "mfif_interactions"),
        ("feedback.decoder.residual_head.", "mfif_residual_head"),
        ("feedback.decoder.", "refinement_decoder_trunk"),
        ("feedback.y_residual_head.", "vif_residual_head"),
    )
    for prefix, group in structural_groups:
        if name.startswith(prefix):
            return group
    if ".expert_pool.modules_by_name.common" in name:
        return "common_experts" if name.startswith("core.") else "feedback_experts"
    if ".common_expert." in name:
        return "common_experts" if name.startswith("core.") else "feedback_experts"
    if ".expert_pool.modules_by_name.low_frequency" in name:
        return (
            "low_frequency_experts" if name.startswith("core.") else "feedback_experts"
        )
    if ".expert_pool.modules_by_name.detail" in name:
        return "detail_experts" if name.startswith("core.") else "feedback_experts"
    if ".expert_pool.modules_by_name.semantic" in name:
        return "semantic_experts" if name.startswith("core.") else "feedback_experts"
    if ".expert_pool.modules_by_name.infrared_saliency" in name:
        return "ir_experts" if name.startswith("core.") else "feedback_experts"
    if name.startswith(("core.moe_blocks", "core.task_embedding")):
        return "coarse_routers"
    if name.startswith("feedback.focus_head"):
        return "focus_head"
    if name.startswith(("feedback.guidance_builder", "feedback.feedback_conditioners")):
        return "guidance_pyramid"
    if name.startswith("feedback.feedback_moe"):
        return (
            "feedback_routers"
            if ".router." in name or name.endswith("residual_scale")
            else "feedback_experts"
        )
    return "shared_backbone"


@dataclass(slots=True)
class ParameterGroupRegistry:
    groups: dict[str, tuple[nn.Parameter, ...]]
    names: dict[str, str]

    @classmethod
    def from_model(cls, model: nn.Module) -> ParameterGroupRegistry:
        groups: dict[str, list[nn.Parameter]] = {name: [] for name in GROUP_NAMES}
        names: dict[str, str] = {}
        seen: set[int] = set()
        shared_pool_enabled = getattr(model, "shared_expert_bank", None) is not None
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            group = parameter_group_name(name, shared_pool_enabled)
            groups[group].append(parameter)
            names[name] = group
            seen.add(id(parameter))
        expected = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        if seen != expected:
            raise RuntimeError(
                "Parameter group registry did not cover every trainable parameter"
            )
        return cls({name: tuple(values) for name, values in groups.items()}, names)

    def parameters(self, group: str) -> tuple[nn.Parameter, ...]:
        return self.groups[group]


def build_optimizer(
    model: nn.Module, config: OptimizerConfig
) -> tuple[Optimizer, ParameterGroupRegistry]:
    registry = ParameterGroupRegistry.from_model(model)
    decay_exempt = {
        id(parameter)
        for module in model.modules()
        if isinstance(
            module, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d, nn.Embedding)
        )
        for parameter in module.parameters(recurse=False)
    }
    groups = []
    for group_name, parameters in registry.groups.items():
        multiplier = config.lr_multipliers.get(group_name, 1.0)
        for no_decay in (False, True):
            selected = [
                parameter
                for parameter in parameters
                if (id(parameter) in decay_exempt or parameter.ndim <= 1) is no_decay
            ]
            if selected:
                groups.append(
                    {
                        "params": selected,
                        "lr": config.learning_rate * multiplier,
                        "initial_lr": config.learning_rate * multiplier,
                        "weight_decay": 0.0 if no_decay else config.weight_decay,
                        "group_name": group_name,
                        "no_decay": no_decay,
                    }
                )
    optimizer = AdamW(
        groups, lr=config.learning_rate, betas=tuple(config.betas), eps=config.epsilon
    )
    return optimizer, registry


import math

from torch.optim.lr_scheduler import LambdaLR

from tfs_moe_fusion.config import (
    RouterTemperatureScheduleConfig,
    SchedulerConfig,
    TaskSchedulePhaseConfig,
    TrainingPhaseConfig,
)


def build_scheduler(optimizer: Optimizer, config: SchedulerConfig, total_steps: int):
    if config.name == "constant":
        return LambdaLR(optimizer, lambda _: 1.0)
    if config.name == "step":
        return LambdaLR(
            optimizer, lambda step: config.gamma ** (step // config.step_size)
        )

    maximum_lr = max(float(group["initial_lr"]) for group in optimizer.param_groups)
    minimum_factor = min(1.0, config.minimum_learning_rate / maximum_lr)

    def factor(step: int) -> float:
        if config.warmup_steps and step < config.warmup_steps:
            return max(1e-8, (step + 1) / config.warmup_steps)
        progress = (step - config.warmup_steps) / max(
            1, total_steps - config.warmup_steps
        )
        cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))
        return minimum_factor + (1 - minimum_factor) * cosine

    return LambdaLR(optimizer, factor)


def active_phase(phases: list[TrainingPhaseConfig], epoch: int) -> TrainingPhaseConfig:
    for phase in phases:
        if phase.start <= epoch < phase.end:
            return phase
    return phases[-1]


def scheduled_task(
    phases: list[TaskSchedulePhaseConfig],
    global_step: int,
    steps_per_epoch: int,
) -> TaskType:
    epoch = global_step // steps_per_epoch
    for phase in phases:
        if phase.start_epoch <= epoch < phase.end_epoch:
            phase_step = global_step - phase.start_epoch * steps_per_epoch
            return TaskType.parse(phase.pattern[phase_step % len(phase.pattern)])
    raise ValueError(f"No task schedule phase covers epoch {epoch}")


def router_temperature(
    config: RouterTemperatureScheduleConfig, step: int, total: int
) -> float:
    progress = min(max(step / max(1, total - 1), 0), 1)
    if config.schedule == "constant":
        value = config.start
    elif config.schedule == "linear":
        value = config.start + (config.end - config.start) * progress
    else:
        value = config.end + (config.start - config.end) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )
    return max(config.minimum, value)


from contextlib import ExitStack, contextmanager

from tfs_moe_fusion.config import (
    MoEExecutionScheduleConfig,
    MoEStarvationLossConfig,
    TaskUpdatePolicyConfig,
)
from tfs_moe_fusion.moe import (
    FunctionalMoEBlock,
    MoEExecutionPolicy,
    TaskEmbedding,
    iter_moe_blocks,
)


def _cosine_progress(step: int, start: int, end: int) -> float:
    if end <= start:
        return float(step >= end)
    progress = min(max((step - start) / (end - start), 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * progress)


class RouterLoadMonitor:
    """Checkpointable block-level EMA load monitor and recovery controller."""

    def __init__(
        self,
        config: MoEExecutionScheduleConfig,
        starvation: MoEStarvationLossConfig | None = None,
    ) -> None:
        self.config = config
        self.starvation = starvation or MoEStarvationLossConfig()
        self.usage_ema: dict[str, list[float]] = {}
        self.top2_mass_ema: dict[str, float] = {}
        self.starvation_counters: dict[str, list[int]] = {}
        self.overload_counters: dict[str, int] = {}
        self.recovery_until_step = 0
        self.floor_usage_ema: dict[str, list[float]] = {}
        self.floor_low_counters: dict[str, list[int]] = {}
        self.floor_release_counters: dict[str, list[int]] = {}
        self.floor_active: dict[str, list[bool]] = {}
        self.floor_active_steps: dict[str, list[int]] = {}
        self.floor_expert_names: dict[str, list[str]] = {}

    @property
    def recovery_active(self) -> bool:
        return not self.starvation.enabled and self.recovery_until_step > 0

    def in_recovery(self, step: int) -> bool:
        return not self.starvation.enabled and step < self.recovery_until_step

    @torch.no_grad()
    def update(
        self,
        diagnostics: tuple[RouterDiagnostics, ...],
        step: int,
        task: TaskType | None = None,
    ) -> dict[str, float]:
        if self.starvation.enabled:
            return self._update_starvation_floor(
                diagnostics, task or TaskType.VIF
            )
        decay = self.config.monitor.ema_decay
        minimum = self.config.monitor.starvation_threshold
        maximum = self.config.monitor.overload_threshold
        patience = self.config.monitor.patience_steps
        result: dict[str, float] = {}
        triggered = False
        # Reduce all blocks together, then transfer once. Validity stays local
        # to this rank, as before; only load and top-2 mass are averaged by DDP.
        statistics, validity = [], []
        for item in diagnostics:
            experts = item.probabilities.shape[1]
            load = (
                item.hard_load
                if item.hard_load is not None
                else torch.nn.functional.one_hot(item.topk_indices, experts)
                .amax(1)
                .float()
                .mean(0)
            )
            valid = item.valid_expert_mask.any(
                tuple(index for index in range(item.valid_expert_mask.ndim) if index != 1)
            )
            top2_mass = item.probabilities.topk(min(2, experts), dim=1).values.sum(1).mean()
            statistics.extend((load.float(), top2_mass.float().reshape(1)))
            validity.append(valid.float())
        if diagnostics:
            reduced = reduce_mean(torch.cat(statistics))
            packed = torch.cat((reduced, *validity)).cpu().tolist()
            offset, valid_offset = 0, reduced.numel()
        for item in diagnostics:
            experts = item.probabilities.shape[1]
            load = packed[offset : offset + experts]
            top2_mass = packed[offset + experts]
            valid = packed[valid_offset : valid_offset + experts]
            offset += experts + 1
            valid_offset += experts
            name = item.block_id
            previous = self.usage_ema.get(name, load)
            usage = [
                (decay * old + (1 - decay) * float(current) if bool(is_valid) else old)
                for old, current, is_valid in zip(
                    previous, load, valid
                )
            ]
            self.usage_ema[name] = usage
            previous_mass = self.top2_mass_ema.get(name, top2_mass)
            self.top2_mass_ema[name] = decay * previous_mass + (1 - decay) * top2_mass
            counters = self.starvation_counters.setdefault(name, [0] * experts)
            for index, is_valid in enumerate(valid):
                if not is_valid:
                    continue
                counters[index] = counters[index] + 1 if usage[index] < minimum else 0
                triggered = triggered or counters[index] >= patience
            overload = self.overload_counters.get(name, 0)
            overload = overload + 1 if max(usage) > maximum else 0
            self.overload_counters[name] = overload
            triggered = triggered or overload >= patience
            result[f"router_top2_mass/{name}"] = self.top2_mass_ema[name]
            result[f"router_max_load/{name}"] = max(usage)
        if triggered:
            self.recovery_until_step = max(
                self.recovery_until_step,
                step + 1 + self.config.recovery.steps,
            )
            self.starvation_counters = {
                name: [0] * len(values)
                for name, values in self.starvation_counters.items()
            }
            self.overload_counters = {name: 0 for name in self.overload_counters}
        result["router_recovery_active"] = float(self.in_recovery(step + 1))
        if self.top2_mass_ema:
            result["router_top2_mass"] = sum(self.top2_mass_ema.values()) / len(
                self.top2_mass_ema
            )
        return result

    @torch.no_grad()
    def _update_starvation_floor(
        self, diagnostics: tuple[RouterDiagnostics, ...], task: TaskType
    ) -> dict[str, float]:
        config = self.starvation
        packed_parts: list[torch.Tensor] = []
        layouts: list[tuple[RouterDiagnostics, int]] = []
        for item in diagnostics:
            probabilities = item.probabilities.detach().float()
            experts = probabilities.shape[1]
            spatial_dims = probabilities.ndim - 2
            valid = item.valid_expert_mask.reshape(
                *item.valid_expert_mask.shape, *((1,) * spatial_dims)
            ).to(probabilities)
            valid = valid.expand_as(probabilities)
            opportunity = item.auxiliary.get("starvation_opportunity")
            if isinstance(opportunity, torch.Tensor):
                if opportunity.shape != probabilities.shape:
                    raise ValueError(
                        "Router starvation opportunity weights must match probabilities"
                    )
                opportunity = opportunity.detach().float().to(probabilities) * valid
            else:
                opportunity = valid
            reduce_dims = tuple(
                index for index in range(probabilities.ndim) if index != 1
            )
            numerator = (probabilities * opportunity).sum(reduce_dims)
            denominator = opportunity.sum(reduce_dims)
            possible = valid.sum(reduce_dims)
            top2_mass = probabilities.topk(min(2, experts), dim=1).values.sum(1).mean()
            packed_parts.extend(
                (numerator, denominator, possible, top2_mass.reshape(1))
            )
            layouts.append((item, experts))
        if not packed_parts:
            return {
                "router_recovery_active": 0.0,
                "router_starvation_active": 0.0,
            }
        packed = reduce_mean(torch.cat(packed_parts)).cpu().tolist()
        offset = 0
        result: dict[str, float] = {}
        active_count = 0
        for item, experts in layouts:
            numerator = packed[offset : offset + experts]
            offset += experts
            denominator = packed[offset : offset + experts]
            offset += experts
            possible = packed[offset : offset + experts]
            offset += experts
            top2_mass = float(packed[offset])
            offset += 1
            usage = [
                float(value) / max(float(weight), 1e-8)
                for value, weight in zip(numerator, denominator)
            ]
            support = [
                float(weight) / max(float(count), 1.0)
                for weight, count in zip(denominator, possible)
            ]
            task_key = task.value if config.per_task else "all"
            key = f"{task_key}|{item.block_id}"
            names = [
                str(name)
                for name in item.auxiliary.get(
                    "expert_names", tuple(str(index) for index in range(experts))
                )
            ]
            if len(names) != experts:
                names = [str(index) for index in range(experts)]
            self.floor_expert_names[key] = names
            previous = self.floor_usage_ema.get(key, usage)
            ema = list(previous)
            low = self.floor_low_counters.setdefault(key, [0] * experts)
            release = self.floor_release_counters.setdefault(key, [0] * experts)
            active = self.floor_active.setdefault(key, [False] * experts)
            ages = self.floor_active_steps.setdefault(key, [0] * experts)
            for index in range(experts):
                eligible = support[index] >= config.evidence_threshold
                if not eligible:
                    continue
                ema[index] = (
                    config.ema_decay * previous[index]
                    + (1 - config.ema_decay) * usage[index]
                )
                if active[index]:
                    ages[index] += 1
                    release[index] = (
                        release[index] + 1
                        if ema[index] >= config.release_threshold
                        else 0
                    )
                    if release[index] >= config.release_patience_steps:
                        active[index] = False
                        ages[index] = 0
                        release[index] = 0
                else:
                    low[index] = (
                        low[index] + 1 if ema[index] < config.threshold else 0
                    )
                    if low[index] >= config.patience_steps:
                        active[index] = True
                        ages[index] = 1
                        low[index] = 0
                active_count += int(active[index])
                result[
                    f"router_usage_ema/{task_key}/{item.block_id}/{names[index]}"
                ] = ema[index]
                result[
                    f"router_evidence_support/{task_key}/{item.block_id}/{names[index]}"
                ] = support[index]
                result[
                    f"router_starvation_counter/{task_key}/{item.block_id}/{names[index]}"
                ] = float(low[index])
                result[
                    f"router_starvation_active/{task_key}/{item.block_id}/{names[index]}"
                ] = float(active[index])
            self.floor_usage_ema[key] = ema
            result[f"router_top2_mass/{item.block_id}"] = top2_mass
            result[f"router_max_load/{item.block_id}"] = max(usage)
        result["router_recovery_active"] = 0.0
        result["router_starvation_active"] = float(active_count)
        return result

    def starvation_strengths(self, task: TaskType) -> dict[str, list[float]]:
        if not self.starvation.enabled:
            return {}
        task_key = task.value if self.starvation.per_task else "all"
        prefix = f"{task_key}|"
        strengths: dict[str, list[float]] = {}
        for key, active in self.floor_active.items():
            if not key.startswith(prefix):
                continue
            ages = self.floor_active_steps[key]
            strengths[key[len(prefix) :]] = [
                min(1.0, age / self.starvation.ramp_steps) if enabled else 0.0
                for enabled, age in zip(active, ages)
            ]
        return strengths

    def state_dict(self) -> dict[str, Any]:
        return {
            "usage_ema": self.usage_ema,
            "top2_mass_ema": self.top2_mass_ema,
            "starvation_counters": self.starvation_counters,
            "overload_counters": self.overload_counters,
            "recovery_until_step": self.recovery_until_step,
            "starvation_floor": {
                "usage_ema": self.floor_usage_ema,
                "low_counters": self.floor_low_counters,
                "release_counters": self.floor_release_counters,
                "active": self.floor_active,
                "active_steps": self.floor_active_steps,
                "expert_names": self.floor_expert_names,
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.starvation.enabled:
            floor = state.get("starvation_floor", {})
            self.floor_usage_ema = {
                str(name): [float(value) for value in values]
                for name, values in floor.get("usage_ema", {}).items()
            }
            self.floor_low_counters = {
                str(name): [int(value) for value in values]
                for name, values in floor.get("low_counters", {}).items()
            }
            self.floor_release_counters = {
                str(name): [int(value) for value in values]
                for name, values in floor.get("release_counters", {}).items()
            }
            self.floor_active = {
                str(name): [bool(value) for value in values]
                for name, values in floor.get("active", {}).items()
            }
            self.floor_active_steps = {
                str(name): [int(value) for value in values]
                for name, values in floor.get("active_steps", {}).items()
            }
            self.floor_expert_names = {
                str(name): [str(value) for value in values]
                for name, values in floor.get("expert_names", {}).items()
            }
            # Legacy recovery was global and included overload triggers.  It is
            # intentionally not restored when the one-sided floor is enabled.
            self.recovery_until_step = 0
            return
        self.usage_ema = {
            str(name): [float(value) for value in values]
            for name, values in state.get("usage_ema", {}).items()
        }
        self.top2_mass_ema = {
            str(name): float(value)
            for name, value in state.get("top2_mass_ema", {}).items()
        }
        self.starvation_counters = {
            str(name): [int(value) for value in values]
            for name, values in state.get("starvation_counters", {}).items()
        }
        self.overload_counters = {
            str(name): int(value)
            for name, value in state.get("overload_counters", {}).items()
        }
        self.recovery_until_step = int(state.get("recovery_until_step", 0))


class MoEExecutionScheduler:
    def __init__(
        self, config: MoEExecutionScheduleConfig, monitor: RouterLoadMonitor
    ) -> None:
        self.config, self.monitor = config, monitor

    def resolve(self, step: int) -> MoEExecutionPolicy | None:
        if not self.config.enabled:
            return None
        warmup = self.config.warmup
        recovery = self.monitor.in_recovery(step)
        temperature = self._temperature(step)
        noise = self._noise(step)
        if recovery:
            temperature = max(temperature, self.config.recovery.temperature_floor)
            noise = max(noise, self.config.recovery.noise_std)

        if step < warmup.uniform_steps:
            return MoEExecutionPolicy(
                "dense_uniform",
                uniform_to_soft=0.0,
                spatial_gate_scale=0.0,
                detach_router=True,
                temperature=temperature,
                noise_std=noise,
            )
        if step < warmup.uniform_to_soft_end:
            progress = _cosine_progress(
                step, warmup.uniform_steps, warmup.uniform_to_soft_end
            )
            return MoEExecutionPolicy(
                "dense_annealed",
                uniform_to_soft=progress,
                spatial_gate_scale=progress,
                temperature=temperature,
                noise_std=noise,
            )
        if step < warmup.soft_to_topk_end:
            progress = _cosine_progress(
                step, warmup.uniform_to_soft_end, warmup.soft_to_topk_end
            )
            return MoEExecutionPolicy(
                "dense_annealed",
                uniform_to_soft=1.0,
                soft_to_topk=progress,
                temperature=temperature,
                noise_std=noise,
            )

        interval = (
            self.config.recovery.refresh_interval
            if recovery
            else self.config.refresh.interval
        )
        elapsed = step - warmup.soft_to_topk_end
        if elapsed > 0 and elapsed % interval == 0:
            return MoEExecutionPolicy(
                "expert_refresh",
                refresh_routed_fraction=self.config.refresh.routed_fraction,
                detach_router=True,
                expert_only=self.config.refresh.expert_only,
                spatial_gate_scale=0.0,
                temperature=temperature,
                noise_std=noise,
            )
        return MoEExecutionPolicy(
            "sparse_batch", temperature=temperature, noise_std=noise
        )

    def _temperature(self, step: int) -> float:
        warmup, routing = self.config.warmup, self.config.routing
        if step <= warmup.uniform_steps:
            return routing.initial_temperature
        if step < warmup.soft_to_topk_end:
            progress = _cosine_progress(
                step, warmup.uniform_steps, warmup.soft_to_topk_end
            )
            return routing.initial_temperature + progress * (
                routing.sparse_start_temperature - routing.initial_temperature
            )
        progress = _cosine_progress(step, warmup.soft_to_topk_end, routing.final_step)
        return routing.sparse_start_temperature + progress * (
            routing.final_temperature - routing.sparse_start_temperature
        )

    def _noise(self, step: int) -> float:
        routing = self.config.routing
        progress = _cosine_progress(step, 0, routing.noise_end_step)
        return routing.initial_noise_std * (1 - progress)


def expert_regularizer_flags(
    loss_config: Any,
    loss_multipliers: dict[str, float],
    task: TaskType,
) -> tuple[bool, bool]:
    """Resolve independent frequency and infrared MoE auxiliary work."""
    return (
        loss_config.frequency.enabled
        and loss_multipliers.get("frequency", 1.0) > 0,
        loss_config.infrared.enabled
        and task is TaskType.VIF
        and loss_multipliers.get("infrared", 1.0) > 0,
    )


class TaskParameterPolicy:
    def __init__(
        self, registry: ParameterGroupRegistry, config: TaskUpdatePolicyConfig
    ) -> None:
        self.registry, self.config = registry, config

    @contextmanager
    def apply(self, task: TaskType) -> Iterator[None]:
        frozen = set(self.config.freeze[task.value])
        changed: list[nn.Parameter] = []
        for group in frozen:
            for parameter in self.registry.parameters(group):
                if parameter.requires_grad:
                    parameter.requires_grad_(False)
                    changed.append(parameter)
        try:
            yield
        finally:
            for parameter in changed:
                parameter.requires_grad_(True)

    @torch.no_grad()
    def scale_gradients(self, task: TaskType) -> None:
        """Scale selected parameter-group gradients for one task update."""

        for group, scale in self.config.gradient_scales.get(
            task.value, {}
        ).items():
            for parameter in self.registry.parameters(group):
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)


class ExpertOnlyParameterPolicy:
    expert_groups = frozenset(
        {
            "common_experts",
            "low_frequency_experts",
            "detail_experts",
            "semantic_experts",
            "ir_experts",
            "feedback_experts",
            "shared_common",
            "shared_low",
            "shared_detail",
            "shared_semantic",
            "shared_ir",
            "shared_focus",
        }
    )

    def __init__(self, registry: ParameterGroupRegistry) -> None:
        self.registry = registry

    @contextmanager
    def apply(self, enabled: bool) -> Iterator[None]:
        if not enabled:
            yield
            return
        changed: list[nn.Parameter] = []
        for group, parameters in self.registry.groups.items():
            if group in self.expert_groups:
                continue
            for parameter in parameters:
                if parameter.requires_grad:
                    parameter.requires_grad_(False)
                    changed.append(parameter)
        try:
            yield
        finally:
            for parameter in changed:
                parameter.requires_grad_(True)


import torch

from tfs_moe_fusion.types import RouterDiagnostics


class GradientConflictMonitor:
    """Compare shared update directions observed on homogeneous task steps."""

    def __init__(self) -> None:
        self.previous: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def observe(self, model: nn.Module, task: TaskType) -> dict[str, float]:
        gradients = [
            (
                parameter.grad.detach().float().flatten().cpu()
                if parameter.grad is not None
                else torch.zeros(parameter.numel(), dtype=torch.float32)
            )
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if not gradients:
            return {}
        current = torch.cat(gradients)
        result = {}
        for other, value in self.previous.items():
            common = min(current.numel(), value.numel())
            cosine = torch.nn.functional.cosine_similarity(
                current[:common], value[:common], dim=0
            )
            result[f"gradient_cosine/{task.value}_{other}"] = float(cosine)
        self.previous[task.value] = current
        return result

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.previous

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.previous = dict(state)


def gradient_statistics(
    model: nn.Module, *, max_norm: float | None = None
) -> dict[str, float]:
    if max_norm is not None:
        # Clipping already computes the global norm. Materialize it once and
        # keep the per-step non-finite guard before the optimizer update.
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
        return {
            "gradient_norm": norm,
            "nonfinite_gradients": float(not math.isfinite(norm)),
        }
    gradients = [
        parameter.grad.detach().float().norm()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return {"gradient_norm": 0.0, "nonfinite_gradients": 0.0}
    stacked = torch.stack(gradients)
    norm, nonfinite = torch.stack(
        (torch.linalg.vector_norm(stacked), (~torch.isfinite(stacked)).sum())
    ).cpu().tolist()
    return {"gradient_norm": norm, "nonfinite_gradients": nonfinite}


def expert_gradient_statistics(
    registry: ParameterGroupRegistry,
) -> dict[str, float]:
    expert_groups = ExpertOnlyParameterPolicy.expert_groups
    result: dict[str, float] = {}
    for group in sorted(expert_groups):
        gradients = [
            parameter.grad.detach().float().norm()
            for parameter in registry.parameters(group)
            if parameter.grad is not None
        ]
        norm = (
            torch.linalg.vector_norm(torch.stack(gradients))
            if gradients
            else torch.zeros(())
        )
        result[f"expert_grad_norm/{group}"] = float(norm)
    return result


def moe_gradient_statistics(model: nn.Module) -> dict[str, float]:
    """Summarize router and per-expert gradients without counting shared weights twice."""

    def norm(parameters: dict[int, nn.Parameter]) -> float:
        gradients = [
            parameter.grad.detach().float().norm()
            for parameter in parameters.values()
            if parameter.grad is not None
        ]
        if not gradients:
            return 0.0
        return float(torch.linalg.vector_norm(torch.stack(gradients)))

    router_parameters: dict[int, nn.Parameter] = {}
    expert_parameters: dict[str, dict[int, nn.Parameter]] = {}
    for block in iter_moe_blocks(model):
        for parameter in block.router.parameters():
            router_parameters[id(parameter)] = parameter
        shared_bank = block.shared_expert_bank
        if shared_bank is not None:
            selected = expert_parameters.setdefault("common", {})
            for parameter in shared_bank.common.parameters():
                selected[id(parameter)] = parameter
            experts = shared_bank.specialists.items()
        else:
            common_expert = getattr(block, "common_expert", None)
            if common_expert is not None:
                selected = expert_parameters.setdefault("common", {})
                for parameter in common_expert.parameters():
                    selected[id(parameter)] = parameter
            experts = block.expert_pool.modules_by_name.items()
        for expert_name, expert in experts:
            selected = expert_parameters.setdefault(expert_name, {})
            for parameter in expert.parameters():
                selected[id(parameter)] = parameter
    for module in model.modules():
        if isinstance(module, TaskEmbedding):
            for parameter in module.parameters():
                router_parameters[id(parameter)] = parameter

    result = {"router_grad_norm": norm(router_parameters)}
    result.update(
        {
            f"expert_grad_norm/{expert_name}": norm(parameters)
            for expert_name, parameters in sorted(expert_parameters.items())
        }
    )
    return result


def moe_residual_statistics(
    values: tuple[RouterDiagnostics, ...],
) -> dict[str, float]:
    ratios = {
        item.block_id: item.auxiliary.get("moe_residual_to_input_ratio")
        for item in values
        if item.auxiliary.get("moe_residual_to_input_ratio") is not None
    }
    if not ratios:
        return {"moe_residual_to_input_ratio": 0.0}
    result = {
        f"moe_residual_to_input_ratio/{block_id}": float(value)
        for block_id, value in ratios.items()
    }
    result["moe_residual_to_input_ratio"] = sum(result.values()) / len(result)
    return result


def loss_group_gradient_statistics(
    result: LossOutput, parameters: tuple[nn.Parameter, ...]
) -> dict[str, float]:
    """Compare effective SEG fusion/semantic gradients on representative weights."""
    totals: dict[str, Tensor] = {}
    for namespace in ("seg_fusion", "semantic"):
        values = [
            value
            for name, value in result.weighted_components.items()
            if name.split("/", 1)[0] == namespace and value.requires_grad
        ]
        if values:
            totals[namespace] = torch.stack(values).sum()
    if len(totals) != 2 or not parameters:
        return {}

    norms: dict[str, float] = {}
    for namespace, total in totals.items():
        gradients = torch.autograd.grad(
            total,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        squared = [
            gradient.detach().float().square().sum()
            for gradient in gradients
            if gradient is not None
        ]
        norm = torch.stack(squared).sum().sqrt() if squared else total.new_zeros(())
        norms[namespace] = float(norm)
    fusion = max(norms["seg_fusion"], 1e-12)
    return {
        "grad_norm/seg_fusion": norms["seg_fusion"],
        "grad_norm/semantic": norms["semantic"],
        "grad_ratio/semantic_to_fusion": norms["semantic"] / fusion,
    }


def _scalar_metrics_to_cpu(metrics: dict[str, Any]) -> dict[str, Any]:
    """Materialize scalar metrics with one transfer per device, without graphs."""
    result = dict(metrics)
    groups: dict[torch.device, list[tuple[str, Tensor]]] = {}
    for name, value in metrics.items():
        if isinstance(value, Tensor):
            groups.setdefault(value.device, []).append(
                (name, value.detach().reshape(()))
            )
    for entries in groups.values():
        scalars = torch.stack([value for _, value in entries]).cpu().tolist()
        result.update(zip((name for name, _ in entries), scalars))
    return result


def router_statistics(values: tuple[RouterDiagnostics, ...]) -> dict[str, Any]:
    if not values:
        return {
            "router_entropy": 0.0,
            "router_max_load": 0.0,
            "router/top1_margin": 0.0,
            "router/probability_std": 0.0,
            "router/entropy": 0.0,
            "routing_override": "learned",
        }
    entropy = torch.stack(
        [item.entropy.mean() for item in values if item.entropy is not None]
    ).mean()
    load = torch.stack(
        [
            item.hard_load
            if item.hard_load is not None
            else torch.nn.functional.one_hot(
                item.topk_indices, item.probabilities.shape[1]
            )
            .amax(1)
            .float()
            .mean(0)
            for item in values
        ]
    ).mean(0)
    result = {
        "router_entropy": entropy.detach(),
        "router_max_load": load.max().detach(),
    }
    router_metrics = {
        name: [item.auxiliary[name] for item in values if name in item.auxiliary]
        for name in (
            "router/top1_margin",
            "router/probability_std",
            "router/entropy",
            "router/spatial_variance",
        )
    }
    for name, metric_values in router_metrics.items():
        if metric_values:
            result[name] = (
                torch.stack([torch.as_tensor(value).detach() for value in metric_values])
                .float()
                .mean()
            )
    overrides = {
        str(item.auxiliary.get("routing_override", "learned")) for item in values
    }
    result["routing_override"] = (
        next(iter(overrides)) if len(overrides) == 1 else "mixed"
    )
    for item in values:
        residuals = item.auxiliary.get("expert_residual_rms", {})
        for expert, value in residuals.items():
            result[f"expert_residual_rms/{item.block_id}/{expert}"] = torch.as_tensor(value)
        contributions = item.auxiliary.get("expert_weighted_contribution_rms", {})
        for expert, value in contributions.items():
            result[
                f"expert_weighted_contribution_rms/{item.block_id}/{expert}"
            ] = torch.as_tensor(value)
        effective_ratios = item.auxiliary.get(
            "expert_effective_contribution_ratio", {}
        )
        for expert, value in effective_ratios.items():
            result[
                f"expert_effective_contribution_ratio/{item.block_id}/{expert}"
            ] = torch.as_tensor(value)
        for name in (
            "common_effective_contribution_ratio",
            "specialist_effective_contribution_ratio",
            "common_scale_rms",
            "specialist_scale_rms",
        ):
            value = item.auxiliary.get(name)
            if value is not None:
                result[f"{name}/{item.block_id}"] = torch.as_tensor(value)
        for name in (
            "router/top1_margin",
            "router/probability_std",
            "router/entropy",
            "router/spatial_variance",
        ):
            value = item.auxiliary.get(name)
            if value is not None:
                result[f"{name}/{item.block_id}"] = torch.as_tensor(value)
        result[f"routing_override/{item.block_id}"] = str(
            item.auxiliary.get("routing_override", "learned")
        )
        scale = item.auxiliary.get("residual_scale_rms")
        if scale is not None:
            result[f"moe_residual_scale/{item.block_id}"] = torch.as_tensor(scale)
    return _scalar_metrics_to_cpu(result)


import logging
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tfs_moe_fusion.data import (
    SemanticRTFusionDataset,
    SynchronizedAugmentationConfig,
    SynchronizedImageAugmentation,
    collate_fusion_samples,
)
from tfs_moe_fusion.losses import (
    LossContext,
    LossOutput,
    MultiTaskLossManager,
    moe_starvation_floor_loss,
)
from tfs_moe_fusion.types import FusionBatch


@dataclass(slots=True)
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    micro_step: int = 0
    phase: str = "stabilization"
    router_temperature: float = 1.0
    collapse_count: int = 0
    last_loss_gradient_bucket: int = -1


class _InfiniteBatchSampler:
    """Reproduce one task's resumable order while DataLoader prefetches ahead."""

    def __init__(
        self,
        order: list[int],
        cursor: int,
        batch_size: int,
        random_state: object,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        usable = len(order) // world_size * world_size
        if usable == 0:
            raise ValueError("Dataset must contain at least one sample per rank")
        self.order = list(order)
        self.cursor = cursor
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.local_length = usable // world_size
        self.random = random.Random()
        self.random.setstate(random_state)

    def __iter__(self):
        while True:
            indices: list[int] = []
            while len(indices) < self.batch_size:
                if self.cursor == self.local_length:
                    self.random.shuffle(self.order)
                    self.cursor = 0
                take = min(self.batch_size - len(indices), self.local_length - self.cursor)
                begin = self.rank + self.cursor * self.world_size
                end = begin + take * self.world_size
                indices.extend(self.order[begin:end:self.world_size])
                self.cursor += take
            yield indices


class SemanticRTBatchProvider:
    """Prefetched, checkpointable provider for the configured task subset.

    The shared order is deterministically partitioned by rank.  Every rank has
    the same number of samples per cycle, so sampler state remains resumable
    from the primary-process checkpoint.
    """

    def __init__(self, config: ProjectConfig, *, rank: int = 0, world_size: int = 1) -> None:
        self.batch_size = config.training.batch_size
        self.num_workers = config.data.num_workers
        self.pin_memory = config.data.pin_memory
        self.seed = config.experiment.seed
        self.rank = rank
        self.world_size = world_size
        self.tasks = tuple(
            TaskType.parse(task) for task in config.training.task_sampling.weights
        )
        if not self.tasks:
            raise ValueError("At least one active task is required")
        augmentation = SynchronizedImageAugmentation(
            SynchronizedAugmentationConfig(
                crop_size=config.data.crop_size,
                horizontal_flip_probability=(config.data.horizontal_flip_probability),
                rotation_degrees=config.data.rotation_degrees,
                rotation_probability=config.data.rotation_probability,
                segmentation_min_valid_pixels=(
                    config.data.segmentation_min_valid_pixels
                ),
                segmentation_crop_attempts=config.data.segmentation_crop_attempts,
            )
        )
        root = self._project_path(config.data.root)
        mfif_root = self._project_path(config.data.mfif_root)
        if config.data.dataset == "semantic_rt":
            manifest = self._project_path(config.data.manifest)
            self.datasets = {
                task: SemanticRTFusionDataset(
                    task, root, mfif_root, manifest, augmentation=augmentation
                )
                for task in self.tasks
            }
        elif config.data.dataset == "msrs":
            from tfs_moe_fusion.data import MSRSFusionDataset

            self.datasets = {
                task: MSRSFusionDataset(task, root, mfif_root, augmentation=augmentation)
                for task in self.tasks
            }
        else:
            raise ValueError(f"No batch provider is configured for {config.data.dataset!r}")
        if any(len(dataset) < self.world_size for dataset in self.datasets.values()):
            raise ValueError("Each active task dataset must contain at least world_size samples")
        self.provider_name = config.data.dataset
        self.manifest_digest = self._manifest_digest(
            tuple(
                f"{task.value}:{sample_id}"
                for task, dataset in self.datasets.items()
                for sample_id in dataset.sample_ids
            )
        )
        self.randoms = {
            task: random.Random(config.experiment.seed + 1009 * (task.index + 1))
            for task in self.tasks
        }
        self.orders = {task: list(range(len(self.datasets[task]))) for task in self.tasks}
        for task in self.tasks:
            self.randoms[task].shuffle(self.orders[task])
        self.cursors = {task: 0 for task in self.tasks}
        self.cycles = {task: 0 for task in self.tasks}
        self.loaders: dict[TaskType, DataLoader] = {}
        self.iterators: dict[TaskType, Any] = {}
        self._rebuild_loaders()

    def next_batch(self, task: TaskType) -> FusionBatch:
        if task not in self.datasets:
            raise ValueError(f"Task {task.value} is not active for this provider")
        if task not in self.iterators:
            self.iterators[task] = iter(self.loaders[task])
        batch = next(self.iterators[task])
        self._advance_consumed_order(task)
        return batch

    def _advance_consumed_order(self, task: TaskType) -> None:
        indices: list[int] = []
        while len(indices) < self.batch_size:
            cursor = self.cursors[task]
            order = self.orders[task]
            local_length = len(order) // self.world_size
            if cursor == local_length:
                self.randoms[task].shuffle(order)
                self.cursors[task] = 0
                self.cycles[task] += 1
                cursor = 0
            take = min(self.batch_size - len(indices), local_length - cursor)
            indices.extend(
                order[
                    self.rank + cursor * self.world_size : self.rank
                    + (cursor + take) * self.world_size : self.world_size
                ]
            )
            self.cursors[task] = cursor + take

    def _rebuild_loaders(self) -> None:
        self._shutdown_loaders()
        for task in self.tasks:
            sampler = _InfiniteBatchSampler(
                self.orders[task],
                self.cursors[task],
                self.batch_size,
                self.randoms[task].getstate(),
                self.rank,
                self.world_size,
            )
            generator = torch.Generator().manual_seed(
                self.seed + 7919 * (task.index + 1) + self.cycles[task]
            )
            loader = DataLoader(
                self.datasets[task],
                batch_sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fusion_samples,
                persistent_workers=self.num_workers > 0,
                generator=generator,
            )
            self.loaders[task] = loader

    def _shutdown_loaders(self) -> None:
        for iterator in getattr(self, "iterators", {}).values():
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
        self.iterators = {}
        self.loaders = {}

    def close(self) -> None:
        self._shutdown_loaders()

    def state_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "batch_size": self.batch_size,
            "world_size": self.world_size,
            "manifest_digest": self.manifest_digest,
            "orders": {task.value: list(order) for task, order in self.orders.items()},
            "cursors": {task.value: cursor for task, cursor in self.cursors.items()},
            "cycles": {task.value: cycle for task, cycle in self.cycles.items()},
            "random_states": {
                task.value: generator.getstate()
                for task, generator in self.randoms.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("provider") != self.provider_name:
            raise ValueError(f"Checkpoint data provider is not {self.provider_name}")
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError("Checkpoint batch size differs from config")
        if int(state.get("world_size", 1)) != self.world_size:
            raise ValueError("Checkpoint world size differs from config")
        if state["manifest_digest"] != self.manifest_digest:
            raise ValueError("Checkpoint dataset contents differ from config")
        restored_orders = {
            TaskType.parse(key): [int(index) for index in value]
            for key, value in state["orders"].items()
        }
        for task, order in restored_orders.items():
            if sorted(order) != list(range(len(self.datasets[task]))):
                raise ValueError(
                    f"Checkpoint has an invalid {task.value} order"
                )
        self.orders = restored_orders
        self.cursors = {
            TaskType.parse(key): int(value) for key, value in state["cursors"].items()
        }
        self.cycles = {
            TaskType.parse(key): int(value) for key, value in state["cycles"].items()
        }
        for key, value in state["random_states"].items():
            self.randoms[TaskType.parse(key)].setstate(value)
        self._rebuild_loaders()

    @staticmethod
    def _project_path(value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[1] / path

    @staticmethod
    def _manifest_digest(values: tuple[str, ...]) -> str:
        import hashlib

        return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def build_batch_provider(
    config: ProjectConfig, *, rank: int = 0, world_size: int = 1
) -> SemanticRTBatchProvider:
    return SemanticRTBatchProvider(config, rank=rank, world_size=world_size)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: ProjectConfig,
        device: torch.device,
        run_dir: str | Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self.raw_model, self.config, self.device = model.to(device), config, device
        self.rank, self.world_size = (
            initialize_distributed(device)
            if config.training.distributed.enabled
            else (0, 1)
        )
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.logger = logger or logging.getLogger(__name__)
        self.loss_manager = MultiTaskLossManager(config.training.losses).to(device)
        self.optimizer, self.registry = build_optimizer(
            model, config.training.optimizer
        )
        self.model = (
            wrap_ddp(
                self.raw_model,
                device,
                config.training.distributed.find_unused_parameters,
            )
            if config.training.distributed.enabled
            else self.raw_model
        )
        self.total_steps = (
            config.training.max_steps
            or config.training.epochs * config.training.steps_per_epoch
        )
        self.scheduler = build_scheduler(
            self.optimizer, config.training.scheduler, self.total_steps
        )
        self.amp = AMPController(config.training.precision, device)
        self.ema = (
            ModelEMA(self.raw_model, config.training.ema.decay)
            if config.training.ema.enabled
            else None
        )
        self.policy = TaskParameterPolicy(
            self.registry, config.training.task_update_policy
        )
        self.expert_only_policy = ExpertOnlyParameterPolicy(self.registry)
        self.router_monitor = RouterLoadMonitor(
            config.training.moe_execution,
            config.training.losses.moe_starvation,
        )
        self.moe_scheduler = MoEExecutionScheduler(
            config.training.moe_execution, self.router_monitor
        )
        self.task_sampler = StatefulTaskSampler.from_strings(
            config.training.task_sampling.weights, config.experiment.seed + 17
        )
        self.provider = build_batch_provider(
            config, rank=self.rank, world_size=self.world_size
        )
        self.state = TrainerState()
        self.gradient_conflicts = GradientConflictMonitor()
        self.last_loss: LossOutput | None = None

    def train(self, max_steps: int | None = None) -> TrainerState:
        stop = min(max_steps or self.total_steps, self.total_steps)
        self.model.train()
        steps_per_epoch = self.config.training.steps_per_epoch
        displayed_epochs = (stop + steps_per_epoch - 1) // steps_per_epoch
        progress = None
        progress_epoch = -1
        epoch_loss_sums = {task: 0.0 for task in TaskType}
        epoch_task_counts = {task: 0 for task in TaskType}
        try:
            while self.state.global_step < stop:
                epoch_index = self.state.global_step // steps_per_epoch
                if progress is None or epoch_index != progress_epoch:
                    if progress is not None:
                        progress.close()
                    progress_epoch = epoch_index
                    epoch_loss_sums = {task: 0.0 for task in TaskType}
                    epoch_task_counts = {task: 0 for task in TaskType}
                    phase = active_phase(
                        self.config.training.phases.phases, epoch_index
                    )
                    progress = tqdm(
                        total=steps_per_epoch,
                        initial=self.state.global_step % steps_per_epoch,
                        desc=(
                            f"Epoch {epoch_index + 1}/{displayed_epochs} [{phase.name}]"
                        ),
                        unit="step",
                        dynamic_ncols=True,
                        disable=self.rank != 0,
                        leave=True,
                    )

                task = self._next_task()
                result = self.train_step(task)
                total_loss, task_loss = self._progress_losses(result, task)
                auxiliary_loss = total_loss - task_loss
                epoch_loss_sums[task] += total_loss
                epoch_task_counts[task] += 1
                progress.update(1)
                progress.set_postfix(
                    task=task.value,
                    loss=f"{total_loss:.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    refresh=bool(result.diagnostics.get("moe_refresh", 0.0)),
                )

                if self.state.global_step % self.config.training.log_every_steps == 0:
                    y_only_metrics = {
                        name: result.diagnostics[name]
                        for name in (
                            "chroma_cb_error",
                            "chroma_cr_error",
                            "y_gamut_clip_ratio",
                            "coarse_final_y_mae",
                            "y_residual_rms",
                            "y_residual_to_coarse_ratio",
                            "y_residual_scale",
                            "y_gradient_loss",
                            "ir_intensity_weight_mean",
                            "ir_intensity_weight_max",
                            "ir_intensity_weight_active_ratio",
                            "ir_hotness_mean",
                            "ir_hotness_active_ratio",
                            "router_ir_importance",
                            "router_ir_hard_load",
                            "router_ir_weighted_contribution_rms",
                            "router_starvation_active",
                            "router_recovery_active",
                            "cross_modal_ir_weight/s1",
                            "cross_modal_ir_weight/s2",
                            "cross_modal_ir_weight/s3",
                            "cross_modal_ir_weight/s4",
                        )
                        if name in result.diagnostics
                    }
                    log_metrics = _scalar_metrics_to_cpu(
                        {
                            **{
                                f"component/{name}": value
                                for name, value in result.weighted_components.items()
                            },
                            **{
                                f"diagnostic/{name}": value
                                for name, value in y_only_metrics.items()
                            },
                        }
                    )
                    values = " ".join(
                        f"{name}={log_metrics['component/' + name]:.5f}"
                        for name in result.weighted_components
                    )
                    y_only_values = " ".join(
                        f"{name}={log_metrics['diagnostic/' + name]:.7f}"
                        for name in y_only_metrics
                    )
                    values = " ".join(
                        value for value in (values, y_only_values) if value
                    )
                    self.logger.info(
                        "step=%d epoch=%d/%d task=%s phase=%s total=%.5f "
                        "task_loss=%.5f aux_loss=%.5f lr=%.3e %s",
                        self.state.global_step,
                        self.state.epoch + 1,
                        displayed_epochs,
                        task.value,
                        self.state.phase,
                        total_loss,
                        task_loss,
                        auxiliary_loss,
                        self.optimizer.param_groups[0]["lr"],
                        values,
                        extra={"terminal": False},
                    )
                if (
                    self.state.global_step
                    % self.config.training.diagnostics.interval
                    == 0
                ):
                    starvation_metrics = _scalar_metrics_to_cpu(
                        {
                            name: value
                            for name, value in result.diagnostics.items()
                            if name.startswith(
                                (
                                    "router_usage_ema/",
                                    "router_evidence_support/",
                                    "router_starvation_active/",
                                )
                            )
                        }
                    )
                    if starvation_metrics:
                        self.logger.info(
                            "step=%d task=%s router_starvation %s",
                            self.state.global_step,
                            task.value,
                            " ".join(
                                f"{name}={value:.5f}"
                                for name, value in starvation_metrics.items()
                            ),
                            extra={"terminal": False},
                        )
                    contribution_metrics = {
                        name: value
                        for name, value in result.diagnostics.items()
                        if name.startswith(
                            (
                                "expert_effective_contribution_ratio/",
                                "common_effective_contribution_ratio/",
                                "specialist_effective_contribution_ratio/",
                            )
                        )
                    }
                    if contribution_metrics:
                        self.logger.info(
                            "step=%d task=%s effective_contribution %s",
                            self.state.global_step,
                            task.value,
                            " ".join(
                                f"{name}={value:.7f}"
                                for name, value in contribution_metrics.items()
                            ),
                            extra={"terminal": False},
                        )
                loss_gradient_values = " ".join(
                    f"{name}={value:.5f}"
                    for name, value in result.diagnostics.items()
                    if name.startswith(
                        ("grad_norm/seg_fusion", "grad_norm/semantic", "grad_ratio/")
                    )
                )
                if loss_gradient_values:
                    self.logger.info(
                        "step=%d task=%s loss_group_gradients %s",
                        self.state.global_step,
                        task.value,
                        loss_gradient_values,
                        extra={"terminal": False},
                    )
                if (
                    self.rank == 0
                    and self.state.global_step
                    % self.config.training.checkpoint.every_steps
                    == 0
                ):
                    self.save(
                        self.checkpoint_dir / f"step_{self.state.global_step:08d}.pt"
                    )
                    self._prune_checkpoints()
                epoch_finished = self.state.global_step % steps_per_epoch == 0
                epoch_number = self.state.global_step // steps_per_epoch
                if epoch_finished:
                    self.state.epoch = epoch_number
                    progress.close()
                    progress = None
                    averages = " ".join(
                        (
                            f"{item.value}_avg="
                            f"{epoch_loss_sums[item] / epoch_task_counts[item]:.5f}"
                            f"({epoch_task_counts[item]} steps)"
                        )
                        for item in TaskType
                        if epoch_task_counts[item]
                    )
                    self.logger.info(
                        "epoch=%d/%d completed phase=%s %s",
                        epoch_number,
                        displayed_epochs,
                        self.state.phase,
                        averages,
                        extra={"terminal": False},
                    )
                if (
                    self.rank == 0
                    and epoch_finished
                    and epoch_number % self.config.training.checkpoint.every_epochs == 0
                ):
                    self.save(self.checkpoint_dir / f"epoch_{epoch_number:04d}.pt")
        finally:
            if progress is not None:
                progress.close()
        if self.rank == 0:
            self.save(self.checkpoint_dir / "latest.pt")
            if self.state.global_step >= self.total_steps:
                self.save(
                    self.checkpoint_dir / "final.pt",
                    metadata={"weights": "raw", "resumable": True},
                )
                if self.ema is not None:
                    self.save_ema(self.checkpoint_dir / "final_ema.pt")
        return self.state

    @staticmethod
    def _progress_losses(result: LossOutput, task: TaskType) -> tuple[float, float]:
        task_namespaces = {
            TaskType.VIF: {"fusion"},
            TaskType.MFIF: {"fusion", "focus"},
            TaskType.SEG: {"seg_fusion", "semantic"},
        }[task]
        task_components = [
            value.detach()
            for name, value in result.weighted_components.items()
            if name.split("/", 1)[0] in task_namespaces
        ]
        task_total = (
            torch.stack(task_components).sum()
            if task_components
            else result.total.detach().new_zeros(())
        )
        values = torch.stack((result.total.detach(), task_total)).float().cpu().tolist()
        return float(values[0]), float(values[1])

    def train_step(self, task: TaskType) -> LossOutput:
        config = self.config.training
        self.state.epoch = self.state.global_step // config.steps_per_epoch
        phase = active_phase(config.phases.phases, self.state.epoch)
        self.state.phase = phase.name
        loss_multipliers = dict(phase.loss_multipliers)
        execution_policy = self.moe_scheduler.resolve(self.state.global_step)
        if execution_policy is None:
            execution_policy = MoEExecutionPolicy.legacy(
                self.config.model.moe.train_execution == "sparse_batch"
            )
        compute_frequency, compute_infrared = expert_regularizer_flags(
            config.losses, loss_multipliers, task
        )
        execution_policy = replace(
            execution_policy,
            compute_frequency_regularizers=compute_frequency,
            compute_infrared_regularizers=compute_infrared,
        )
        self._set_moe_execution_policy(execution_policy)
        self.state.router_temperature = (
            execution_policy.temperature
            if execution_policy is not None and execution_policy.temperature is not None
            else router_temperature(
                config.router_temperature, self.state.global_step, self.total_steps
            )
        )
        self._set_router_temperature(self.state.router_temperature)
        if execution_policy is not None:
            self._set_router_noise(execution_policy.noise_std)
        self.optimizer.zero_grad(set_to_none=True)
        aggregate: LossOutput | None = None
        loss_gradient_info: dict[str, float] = {}
        loss_gradient_bucket = (
            self.state.global_step // config.diagnostics.loss_gradient_interval
        )
        if execution_policy is not None and execution_policy.expert_only:
            loss_multipliers.update({"moe": 0.0, "consistency": 0.0})
        with ExitStack() as stack:
            stack.enter_context(self.policy.apply(task))
            stack.enter_context(
                self.expert_only_policy.apply(
                    execution_policy is not None and execution_policy.expert_only
                )
            )
            for micro_index in range(config.gradient_accumulation_steps):
                batch = self.provider.next_batch(task).to(
                    self.device,
                    non_blocking=(
                        self.device.type == "cuda" and self.config.data.pin_memory
                    ),
                )
                with self.amp.autocast():
                    output = self.model(batch)
                    auxiliary: dict[str, Any] = {}
                    if not (
                        execution_policy is not None and execution_policy.expert_only
                    ) and self._use_consistency(task):
                        auxiliary["paired_output"] = self.model(
                            self._paired_batch(batch)
                        )
                    result = self.loss_manager(
                        LossContext(
                            batch,
                            output,
                            task,
                            self.state.epoch,
                            self.state.global_step,
                            self.model,
                            auxiliary,
                            phase.name,
                            loss_multipliers,
                        )
                    )
                    starvation = config.losses.moe_starvation
                    if starvation.enabled and not (
                        execution_policy is not None
                        and execution_policy.expert_only
                    ):
                        floor_loss = moe_starvation_floor_loss(
                            output.router_balance_states,
                            self.router_monitor.starvation_strengths(task),
                            threshold=starvation.threshold,
                            evidence_threshold=starvation.evidence_threshold,
                        )
                        weighted_floor = (starvation.weight * floor_loss).clamp_max(
                            starvation.max_total_weight
                        )
                        result.components["moe_starvation/floor"] = floor_loss
                        result.weighted_components[
                            "moe_starvation/floor"
                        ] = weighted_floor
                        result.total = result.total + weighted_floor
                    if (
                        micro_index == 0
                        and task is TaskType.SEG
                        and self.state.global_step
                        < config.diagnostics.loss_gradient_until_step
                        and loss_gradient_bucket > self.state.last_loss_gradient_bucket
                    ):
                        loss_gradient_info = loss_group_gradient_statistics(
                            result, self._loss_gradient_parameters()
                        )
                        if loss_gradient_info:
                            self.state.last_loss_gradient_bucket = loss_gradient_bucket
                    scaled_loss = result.total / config.gradient_accumulation_steps
                self.amp.backward(scaled_loss)
                aggregate = result
                self.state.micro_step += 1
        assert aggregate is not None
        self.amp.unscale_(self.optimizer)
        self.policy.scale_gradients(task)
        diagnostics_due = self.state.global_step % config.diagnostics.interval == 0
        gradient_info = {}
        if diagnostics_due:
            gradient_info.update(expert_gradient_statistics(self.registry))
            gradient_info.update(moe_gradient_statistics(self.model))
        gradient_info.update(
            gradient_statistics(
                self.model,
                max_norm=config.gradient_clip.max_norm
                if config.gradient_clip.enabled
                else None,
            )
        )
        if gradient_info["nonfinite_gradients"]:
            raise FloatingPointError("Training produced non-finite gradients")
        self.amp.step(self.optimizer)
        self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.raw_model)
        aggregate.diagnostics.update(gradient_info)
        aggregate.diagnostics.update(loss_gradient_info)
        diagnostics_config = config.diagnostics
        if (
            diagnostics_config.gradient_conflict_enabled
            and self.state.global_step % diagnostics_config.gradient_conflict_interval
            == 0
        ):
            aggregate.diagnostics.update(
                self.gradient_conflicts.observe(self.model, task)
            )
        aggregate.diagnostics.update(
            router_statistics(
                aggregate_context := tuple(
                    # The output diagnostics remain reachable from the loss components only
                    # during this method; get the latest forward via the local output.
                    output.router_diagnostics
                )
            )
        )
        if diagnostics_due:
            aggregate.diagnostics.update(
                moe_residual_statistics(tuple(output.router_diagnostics))
            )
        aggregate.diagnostics.update(
            self.router_monitor.update(
                tuple(output.router_diagnostics),
                self.state.global_step,
                task,
            )
        )
        if execution_policy is not None:
            aggregate.diagnostics.update(
                {
                    "moe_execution": execution_policy.mode,
                    "moe_uniform_to_soft": execution_policy.uniform_to_soft,
                    "moe_soft_to_topk": execution_policy.soft_to_topk,
                    "moe_refresh": float(execution_policy.expert_only),
                }
            )
        del aggregate_context
        self._monitor_collapse(aggregate.diagnostics)
        self.state.global_step += 1
        self.last_loss = aggregate
        return aggregate

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        return save_checkpoint(
            path,
            self.raw_model,
            self.config,
            epoch=self.state.epoch,
            global_step=self.state.global_step,
            optimizer=self.optimizer,
            scheduler_state=self.scheduler.state_dict(),
            scaler_state=self.amp.scaler.state_dict(),
            ema_state=self.ema.state_dict() if self.ema is not None else None,
            sampler_state=self.task_sampler.state_dict(),
            engine_state={
                "trainer": self.state.__dict__
                if hasattr(self.state, "__dict__")
                else {
                    field: getattr(self.state, field) for field in self.state.__slots__
                },
                "provider": self.provider.state_dict(),
                "gradient_conflicts": self.gradient_conflicts.state_dict(),
                "router_monitor": self.router_monitor.state_dict(),
            },
            metadata=metadata,
        )

    def save_ema(self, path: str | Path) -> Path:
        if self.ema is None:
            raise RuntimeError("Cannot export EMA weights when EMA is disabled")
        with self.ema.apply(self.raw_model):
            return save_checkpoint(
                path,
                self.raw_model,
                self.config,
                epoch=self.state.epoch,
                global_step=self.state.global_step,
                ema_state=self.ema.state_dict(),
                metadata={"weights": "ema", "resumable": False},
            )

    def resume(self, path: str | Path) -> None:
        report = load_checkpoint(
            path,
            self.raw_model,
            optimizer=self.optimizer,
            map_location=self.device,
            restore_rng=True,
        )
        if report.metadata.get("resumable") is False:
            raise RuntimeError(
                f"Checkpoint {report.path} contains evaluation-only weights and "
                "cannot resume training"
            )
        if report.scheduler_state is not None:
            self.scheduler.load_state_dict(report.scheduler_state)
        if report.scaler_state is not None:
            self.amp.scaler.load_state_dict(report.scaler_state)
        if self.ema is not None and report.ema_state is not None:
            self.ema.load_state_dict(report.ema_state)
        if report.sampler_state is not None:
            self.task_sampler.load_state_dict(report.sampler_state)
        state = report.engine_state or {}
        if "trainer" in state:
            self.state = TrainerState(**state["trainer"])
        else:
            self.state.epoch, self.state.global_step = report.epoch, report.global_step
        if "provider" in state:
            self.provider.load_state_dict(state["provider"])
        if "gradient_conflicts" in state:
            self.gradient_conflicts.load_state_dict(state["gradient_conflicts"])
        if "router_monitor" in state:
            self.router_monitor.load_state_dict(state["router_monitor"])

    def _next_task(self) -> TaskType:
        strategy = self.config.training.task_sampling.strategy
        if strategy == "scheduled":
            return scheduled_task(
                self.config.training.task_schedule,
                self.state.global_step,
                self.config.training.steps_per_epoch,
            )
        if strategy == "alternating":
            return tuple(TaskType)[self.state.global_step % len(TaskType)]
        return self.task_sampler.next_task()

    def _use_consistency(self, task: TaskType) -> bool:
        config = self.config.training.losses.consistency
        return (
            config.enabled
            and task in {TaskType.VIF, TaskType.SEG}
            and random.random() < config.probability
        )

    @staticmethod
    def _paired_batch(batch: FusionBatch) -> FusionBatch:
        task = TaskType.SEG if batch.task is TaskType.VIF else TaskType.VIF
        return FusionBatch(
            batch.source_a,
            batch.source_b,
            task,
            batch.sample_ids,
            batch.target,
            batch.focus_target,
            batch.segmentation_target,
            batch.metadata,
        )

    def _set_router_temperature(self, temperature: float) -> None:
        for module in self.model.modules():
            router = getattr(module, "router", None)
            if router is not None and hasattr(router, "temperature"):
                router.temperature = temperature

    def _loss_gradient_parameters(self) -> tuple[nn.Parameter, ...]:
        selected: list[nn.Parameter] = []
        for group in ("refinement_decoder_trunk", "shared_backbone"):
            selected.extend(
                parameter
                for parameter in self.registry.parameters(group)
                if parameter.requires_grad
            )
            if selected:
                break
        return tuple(selected[:8])

    def _set_router_noise(self, standard_deviation: float) -> None:
        for module in self.model.modules():
            router = getattr(module, "router", None)
            if router is not None and hasattr(router, "noisy_topk"):
                router.noisy_topk = standard_deviation > 0
                router.noisy_topk_std = standard_deviation

    def _set_moe_execution_policy(self, policy: MoEExecutionPolicy | None) -> None:
        for module in self.model.modules():
            if isinstance(module, FunctionalMoEBlock):
                module.set_execution_policy(policy)

    def _monitor_collapse(self, diagnostics: dict[str, Any]) -> None:
        threshold = self.config.training.diagnostics.router_collapse_threshold
        self.state.collapse_count = (
            self.state.collapse_count + 1
            if diagnostics.get("router_max_load", 0) > threshold
            else 0
        )
        if (
            self.state.collapse_count
            == self.config.training.diagnostics.collapse_patience
        ):
            self.logger.warning(
                "Router collapse persisted for %d steps", self.state.collapse_count
            )

    def _prune_checkpoints(self) -> None:
        checkpoints = sorted(self.checkpoint_dir.glob("step_*.pt"))
        for path in checkpoints[: -self.config.training.checkpoint.keep_last]:
            path.unlink()
