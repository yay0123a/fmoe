"""TFS-MoE-Fusion consolidated implementation."""

from __future__ import annotations

import logging
from pathlib import Path


class _TerminalLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "terminal", True))


def configure_logging(
    level: int = logging.INFO, log_file: Path | None = None
) -> logging.Logger:
    """Configure the package logger without mutating unrelated root handlers."""

    logger = logging.getLogger("tfs_moe_fusion")
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    stream.addFilter(_TerminalLogFilter())
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


from torch import nn


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def module_parameter_report(modules: dict[str, nn.Module]) -> dict[str, dict[str, int]]:
    return {
        name: {
            "total": count_parameters(module),
            "trainable": count_trainable_parameters(module),
        }
        for name, module in modules.items()
    }


from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

from tfs_moe_fusion.types import ConfigurationError

T = TypeVar("T")


class Registry(Generic[T]):
    """Map stable configuration names to constructors or component objects."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        key = self._normalize(name)

        def decorator(item: T) -> T:
            if key in self._items:
                raise ConfigurationError(
                    f"{self.name} registry already contains {key!r}"
                )
            self._items[key] = item
            return item

        return decorator

    def add(self, name: str, item: T) -> None:
        self.register(name)(item)

    def get(self, name: str) -> T:
        key = self._normalize(name)
        try:
            return self._items[key]
        except KeyError as error:
            legal = ", ".join(sorted(self._items)) or "<empty>"
            raise ConfigurationError(
                f"Unknown {self.name} {name!r}. Legal names: {legal}"
            ) from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._normalize(name) in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    @staticmethod
    def _normalize(name: str) -> str:
        value = name.strip().lower()
        if not value:
            raise ConfigurationError("Registry names cannot be empty")
        return value


import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


from tfs_moe_fusion.config import ProjectConfig, load_config, save_resolved_config
from tfs_moe_fusion.data import (
    DeterministicDummyFusionDataset,
    collate_fusion_samples,
)
from tfs_moe_fusion.types import FusionBatch, TaskType


def prepare_run(config_path: str | Path) -> tuple[ProjectConfig, Path]:
    config = load_config(config_path)
    run_dir = Path(config.experiment.output_dir) / config.experiment.name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    return config, run_dir


def make_dummy_batch(
    config: ProjectConfig, task: TaskType, batch_size: int | None = None
) -> FusionBatch:
    size = batch_size or config.training.batch_size
    dataset = DeterministicDummyFusionDataset(
        task=task,
        length=max(size, config.data.dummy_length),
        height=config.data.height,
        width=config.data.width,
        seed=config.experiment.seed,
    )
    return collate_fusion_samples([dataset[index] for index in range(size)])


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device
