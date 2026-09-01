from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from tfs_moe_fusion.config import load_config
from tfs_moe_fusion.model import build_model
from tfs_moe_fusion.trainer import (
    StatefulTaskSampler,
    load_checkpoint,
    save_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def _smoke_config(assets: tuple[Path, Path, Path] | None = None):
    config = load_config(ROOT / "configs/default.yaml")
    config.model.backbone.channels = [8, 16, 32, 64]
    config.model.backbone.depths = [1, 1, 1, 1]
    config.model.frequency.fdconv_kernel_num = 4
    config.model.moe.router_hidden_channels = 16
    config.model.moe.expert_expansion = 1
    config.model.guidance.focus.hidden_channels = 8
    config.model.guidance.semantic.input_size = 32
    config.model.guidance.semantic.enabled = False
    config.model.feedback.guide_channels = 8
    config.data.crop_size = 32
    config.data.num_workers = 0
    config.data.horizontal_flip_probability = 0.0
    config.data.rotation_probability = 0.0
    if assets is not None:
        config.data.root, config.data.mfif_root, config.data.manifest = map(str, assets)
    config.training.epochs = 1
    config.training.steps_per_epoch = 20
    config.training.max_steps = 20
    config.training.batch_size = 1
    config.training.precision = "fp32"
    config.training.task_sampling.strategy = "alternating"
    config.training.scheduler.warmup_steps = 2
    config.training.ema.enabled = True
    config.training.losses.strict_targets = False
    return config


def test_checkpoint_round_trip_and_sampler_resume(tmp_path: Path) -> None:
    config = _smoke_config()
    model = build_model(config)
    sampler = StatefulTaskSampler.from_strings(
        config.training.task_sampling.weights, seed=config.experiment.seed
    )
    sampler.next_task()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model,
        config,
        epoch=3,
        global_step=17,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        metadata={"purpose": "test"},
    )

    restored_model = build_model(config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-4)
    report = load_checkpoint(
        checkpoint, restored_model, optimizer=restored_optimizer, strict=True
    )
    assert report.epoch == 3
    assert report.global_step == 17
    assert report.missing_keys == ()
    assert report.unexpected_keys == ()
    assert report.metadata == {"purpose": "test"}

    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(expected, actual)

    assert report.sampler_state is not None
    restored_sampler = StatefulTaskSampler.from_strings(
        config.training.task_sampling.weights, seed=config.experiment.seed
    )
    restored_sampler.load_state_dict(report.sampler_state)
    assert sampler.next_task() is restored_sampler.next_task()


from pathlib import Path

from tfs_moe_fusion.trainer import Trainer
from tfs_moe_fusion.types import TaskType
from tfs_moe_fusion.utils import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def test_trainer_exports_applied_ema_weights(
    tmp_path: Path, semantic_rt_assets: tuple[Path, Path, Path]
) -> None:
    config = _smoke_config(semantic_rt_assets)
    trainer = Trainer(
        build_model(config), config, torch.device("cpu"), tmp_path / "trainer"
    )
    assert trainer.ema is not None
    with torch.no_grad():
        parameter = next(
            value
            for value in trainer.raw_model.parameters()
            if value.is_floating_point()
        )
        parameter.add_(1.0)
    trainer.ema.update(trainer.raw_model)

    destination = trainer.save_ema(tmp_path / "final_ema.pt")
    restored = build_model(config)
    report = load_checkpoint(destination, restored)

    assert report.metadata == {"weights": "ema", "resumable": False}
    restored_state = restored.state_dict()
    for name, value in trainer.ema.shadow.items():
        torch.testing.assert_close(restored_state[name], value)

    resumed = Trainer(
        build_model(config), config, torch.device("cpu"), tmp_path / "resumed"
    )
    with pytest.raises(RuntimeError, match="evaluation-only"):
        resumed.resume(destination)


def test_resume_reproduces_the_next_optimizer_update(
    tmp_path: Path, semantic_rt_assets: tuple[Path, Path, Path]
) -> None:
    config = _smoke_config(semantic_rt_assets)
    seed_everything(config.experiment.seed, True)
    baseline = Trainer(
        build_model(config), config, torch.device("cpu"), tmp_path / "baseline"
    )
    baseline.train_step(TaskType.VIF)
    checkpoint = baseline.save(tmp_path / "exact.pt")
    expected_monitor = deepcopy(baseline.router_monitor.state_dict())
    baseline.train_step(TaskType.MFIF)
    expected = {
        name: value.detach().clone()
        for name, value in baseline.model.state_dict().items()
    }

    resumed = Trainer(
        build_model(config), config, torch.device("cpu"), tmp_path / "resumed"
    )
    resumed.resume(checkpoint)
    assert resumed.router_monitor.state_dict() == expected_monitor
    resumed.train_step(TaskType.MFIF)
    for name, value in resumed.model.state_dict().items():
        # Optimizer/RNG/sampler states are exact; parallel CPU kernels may differ
        # at the last floating-point bit across separate module allocations.
        torch.testing.assert_close(value, expected[name], rtol=1e-6, atol=1e-8)
