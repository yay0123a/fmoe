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
        torch.set_rng_state(payload["rng"]["torch"])
        if payload["rng"].get("python") is not None:
            random.setstate(payload["rng"]["python"])
        if payload["rng"].get("numpy") is not None:
            np.random.set_state(payload["rng"]["numpy"])
        cuda_state = payload["rng"].get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)

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
        if set(self.weights) != set(TaskType):
            raise ValueError("weights must define every TaskType")
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
        tasks = list(TaskType)
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
    "coarse_decoder",
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
    "refinement_decoder",
    "residual_head",
    "shared_common",
    "shared_low",
    "shared_detail",
    "shared_semantic",
    "shared_ir",
    "core_site_adapters",
    "feedback_site_adapters",
    "core_routers",
)


def parameter_group_name(name: str, shared_pool_enabled: bool = False) -> str:
    if shared_pool_enabled:
        shared_experts = {
            "shared_expert_bank.common.": "shared_common",
            "shared_expert_bank.specialists.low_frequency.": "shared_low",
            "shared_expert_bank.specialists.detail.": "shared_detail",
            "shared_expert_bank.specialists.semantic.": "shared_semantic",
            "shared_expert_bank.specialists.infrared_saliency.": "shared_ir",
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
    if name.startswith(
        ("core.decoder_stages", "core.fusion_y_head", "core.mfif_rgb_head")
    ):
        return "coarse_decoder"
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
    if name.startswith(
        (
            "feedback.guidance_builder",
            "feedback.feedback_conditioners",
            "feedback.mfif_interactions",
        )
    ):
        return "guidance_pyramid"
    if name.startswith("feedback.feedback_moe"):
        return (
            "feedback_routers"
            if ".router." in name or name.endswith("residual_scale")
            else "feedback_experts"
        )
    if name.startswith("feedback.decoder.residual_head"):
        return "residual_head"
    if name.startswith("feedback.decoder"):
        return "refinement_decoder"
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

from tfs_moe_fusion.config import MoEExecutionScheduleConfig, TaskUpdatePolicyConfig
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

    def __init__(self, config: MoEExecutionScheduleConfig) -> None:
        self.config = config
        self.usage_ema: dict[str, list[float]] = {}
        self.top2_mass_ema: dict[str, float] = {}
        self.starvation_counters: dict[str, list[int]] = {}
        self.overload_counters: dict[str, int] = {}
        self.recovery_until_step = 0

    @property
    def recovery_active(self) -> bool:
        return self.recovery_until_step > 0

    def in_recovery(self, step: int) -> bool:
        return step < self.recovery_until_step

    @torch.no_grad()
    def update(
        self, diagnostics: tuple[RouterDiagnostics, ...], step: int
    ) -> dict[str, float]:
        decay = self.config.monitor.ema_decay
        minimum = self.config.monitor.starvation_threshold
        maximum = self.config.monitor.overload_threshold
        patience = self.config.monitor.patience_steps
        result: dict[str, float] = {}
        triggered = False
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
            load = reduce_mean(load).cpu()
            valid = item.valid_expert_mask.any(
                tuple(index for index in range(item.valid_expert_mask.ndim) if index != 1)
            ).cpu()
            top2_mass = item.probabilities.topk(min(2, experts), dim=1).values.sum(1).mean()
            top2_mass = float(reduce_mean(top2_mass).cpu())
            name = item.block_id
            previous = self.usage_ema.get(name, load.tolist())
            usage = [
                (decay * old + (1 - decay) * float(current) if bool(is_valid) else old)
                for old, current, is_valid in zip(
                    previous, load.tolist(), valid.tolist()
                )
            ]
            self.usage_ema[name] = usage
            previous_mass = self.top2_mass_ema.get(name, top2_mass)
            self.top2_mass_ema[name] = decay * previous_mass + (1 - decay) * top2_mass
            counters = self.starvation_counters.setdefault(name, [0] * experts)
            for index, is_valid in enumerate(valid.tolist()):
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

    def state_dict(self) -> dict[str, Any]:
        return {
            "usage_ema": self.usage_ema,
            "top2_mass_ema": self.top2_mass_ema,
            "starvation_counters": self.starvation_counters,
            "overload_counters": self.overload_counters,
            "recovery_until_step": self.recovery_until_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
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


def gradient_statistics(model: nn.Module) -> dict[str, float]:
    gradients = [
        parameter.grad.detach().float().norm()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return {"gradient_norm": 0.0, "nonfinite_gradients": 0.0}
    stacked = torch.stack(gradients)
    return {
        "gradient_norm": float(torch.linalg.vector_norm(stacked)),
        "nonfinite_gradients": float((~torch.isfinite(stacked)).sum()),
    }


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
        "router_entropy": float(entropy.detach()),
        "router_max_load": float(load.max().detach()),
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
            result[name] = float(
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
            result[f"expert_residual_rms/{item.block_id}/{expert}"] = float(value)
        contributions = item.auxiliary.get("expert_weighted_contribution_rms", {})
        for expert, value in contributions.items():
            result[
                f"expert_weighted_contribution_rms/{item.block_id}/{expert}"
            ] = float(value)
        for name in (
            "router/top1_margin",
            "router/probability_std",
            "router/entropy",
            "router/spatial_variance",
        ):
            value = item.auxiliary.get(name)
            if value is not None:
                result[f"{name}/{item.block_id}"] = float(value)
        result[f"routing_override/{item.block_id}"] = str(
            item.auxiliary.get("routing_override", "learned")
        )
        scale = item.auxiliary.get("residual_scale_rms")
        if scale is not None:
            result[f"moe_residual_scale/{item.block_id}"] = float(scale)
    return result


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
from tfs_moe_fusion.losses import LossContext, LossOutput, MultiTaskLossManager
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
    ) -> None:
        self.order = list(order)
        self.cursor = cursor
        self.batch_size = batch_size
        self.random = random.Random()
        self.random.setstate(random_state)

    def __iter__(self):
        while True:
            indices: list[int] = []
            while len(indices) < self.batch_size:
                if self.cursor == len(self.order):
                    self.random.shuffle(self.order)
                    self.cursor = 0
                take = min(
                    self.batch_size - len(indices), len(self.order) - self.cursor
                )
                indices.extend(self.order[self.cursor : self.cursor + take])
                self.cursor += take
            yield indices


class SemanticRTBatchProvider:
    """Prefetched provider with checkpointable consumed order for all three tasks."""

    def __init__(self, config: ProjectConfig) -> None:
        self.batch_size = config.training.batch_size
        self.num_workers = config.data.num_workers
        self.pin_memory = config.data.pin_memory
        self.seed = config.experiment.seed
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
        manifest = self._project_path(config.data.manifest)
        self.datasets = {
            task: SemanticRTFusionDataset(
                task,
                root,
                mfif_root,
                manifest,
                augmentation=augmentation,
            )
            for task in TaskType
        }
        lengths = {len(dataset) for dataset in self.datasets.values()}
        if len(lengths) != 1:
            raise ValueError(
                f"SemanticRT task datasets must have equal lengths, got {lengths}"
            )
        self.manifest_digest = self._manifest_digest(
            next(iter(self.datasets.values())).sample_ids
        )
        self.randoms = {
            task: random.Random(config.experiment.seed + 1009 * (task.index + 1))
            for task in TaskType
        }
        self.orders = {task: list(range(len(self.datasets[task]))) for task in TaskType}
        for task in TaskType:
            self.randoms[task].shuffle(self.orders[task])
        self.cursors = {task: 0 for task in TaskType}
        self.cycles = {task: 0 for task in TaskType}
        self.loaders: dict[TaskType, DataLoader] = {}
        self.iterators: dict[TaskType, Any] = {}
        self._rebuild_loaders()

    def next_batch(self, task: TaskType) -> FusionBatch:
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
            if cursor == len(order):
                self.randoms[task].shuffle(order)
                self.cursors[task] = 0
                self.cycles[task] += 1
                cursor = 0
            take = min(self.batch_size - len(indices), len(order) - cursor)
            indices.extend(order[cursor : cursor + take])
            self.cursors[task] = cursor + take

    def _rebuild_loaders(self) -> None:
        self._shutdown_loaders()
        for task in TaskType:
            sampler = _InfiniteBatchSampler(
                self.orders[task],
                self.cursors[task],
                self.batch_size,
                self.randoms[task].getstate(),
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
            "provider": "semantic_rt",
            "batch_size": self.batch_size,
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
        if state.get("provider") != "semantic_rt":
            raise ValueError("Checkpoint data provider is not SemanticRT")
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError("SemanticRT checkpoint batch size differs from config")
        if state["manifest_digest"] != self.manifest_digest:
            raise ValueError("SemanticRT checkpoint manifest differs from config")
        restored_orders = {
            TaskType.parse(key): [int(index) for index in value]
            for key, value in state["orders"].items()
        }
        for task, order in restored_orders.items():
            if sorted(order) != list(range(len(self.datasets[task]))):
                raise ValueError(
                    f"SemanticRT checkpoint has an invalid {task.value} order"
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
    config: ProjectConfig,
) -> SemanticRTBatchProvider:
    if config.data.dataset == "semantic_rt":
        return SemanticRTBatchProvider(config)
    raise ValueError(f"No batch provider is configured for {config.data.dataset!r}")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: ProjectConfig,
        device: torch.device,
        run_dir: str | Path,
        logger: logging.Logger | None = None,
    ) -> None:
        if config.data.dataset == "semantic_rt" and config.training.distributed.enabled:
            raise ValueError("The SemanticRT training profile is single-GPU only")
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
        self.router_monitor = RouterLoadMonitor(config.training.moe_execution)
        self.moe_scheduler = MoEExecutionScheduler(
            config.training.moe_execution, self.router_monitor
        )
        self.task_sampler = StatefulTaskSampler.from_strings(
            config.training.task_sampling.weights, config.experiment.seed + 17
        )
        self.provider = build_batch_provider(config)
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
                    values = " ".join(
                        f"{name}={float(value.detach()):.5f}"
                        for name, value in result.weighted_components.items()
                    )
                    y_only_values = " ".join(
                        f"{name}={float(result.diagnostics[name]):.7f}"
                        for name in (
                            "chroma_cb_error",
                            "chroma_cr_error",
                            "y_gamut_clip_ratio",
                            "coarse_final_y_mae",
                            "y_gradient_loss",
                            "ir_intensity_weight_mean",
                            "ir_intensity_weight_max",
                            "ir_intensity_weight_active_ratio",
                        )
                        if name in result.diagnostics
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
        gradient_info = gradient_statistics(self.model)
        diagnostics_due = self.state.global_step % config.diagnostics.interval == 0
        if diagnostics_due:
            gradient_info.update(expert_gradient_statistics(self.registry))
            gradient_info.update(moe_gradient_statistics(self.model))
        if gradient_info["nonfinite_gradients"]:
            raise FloatingPointError("Training produced non-finite gradients")
        if config.gradient_clip.enabled:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), config.gradient_clip.max_norm
            )
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
                tuple(output.router_diagnostics), self.state.global_step
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
        for group in ("refinement_decoder", "shared_backbone"):
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
