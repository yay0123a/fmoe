from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tfs_moe_fusion.backbone import CustomMultiscaleBackbone
from tfs_moe_fusion.config import load_config
from tfs_moe_fusion.model import build_model
from tfs_moe_fusion.types import (
    ConfigurationError,
    TaskType,
)
from tfs_moe_fusion.utils import configure_logging, make_probe_batch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/default.yaml"


def test_default_config_selects_real_semantic_rt_training() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config.data.dataset == "semantic_rt"
    assert config.data.root == "data/semantic_rt"
    assert config.data.mfif_root == "data/mfif/semantic_rt"
    assert config.data.manifest == (
        "data/splits/semantic_rt_test_uniform_2000_seed3407.txt"
    )
    assert config.training.device == "cuda:0"
    assert config.model.guidance.semantic.model_dir == (
        "weights/segformer_b0_cityscapes"
    )
    assert config.model.moe.block_counts == {"s1": 0, "s2": 1, "s3": 1, "s4": 1}
    assert config.model.moe.top_k == 2
    assert config.model.guidance.semantic.detach_guidance_input is True
    assert config.data.num_workers == 4
    assert config.data.pin_memory is True
    warmup = config.training.moe_execution.warmup
    assert (
        warmup.uniform_steps,
        warmup.uniform_to_soft_end,
        warmup.soft_to_topk_end,
    ) == (50, 150, 300)


def test_default_config_uses_minimal_seg_loss_schedule() -> None:
    config = load_config(DEFAULT_CONFIG)
    losses = config.training.losses
    assert losses.seg_fusion.enabled
    assert (losses.seg_fusion.intensity, losses.seg_fusion.gradient) == (1.0, 1.0)
    assert losses.semantic.cross_entropy == 1.0
    assert losses.semantic.dice == 0.5
    assert losses.semantic.coarse_supervision == 0.0
    assert losses.semantic.boundary_alignment == 0.0
    assert not losses.frequency.enabled
    assert not losses.consistency.enabled

    phases = config.training.phases.phases
    assert [(phase.start, phase.end) for phase in phases] == [
        (0, 5),
        (5, 15),
        (15, 44),
        (44, 50),
    ]
    assert phases[0].loss_multipliers == {"seg_fusion": 1.0, "semantic": 0.25}
    assert phases[1].loss_multipliers == {"seg_fusion": 1.0, "semantic": 0.5}
    assert phases[2].loss_multipliers == {"seg_fusion": 0.75, "semantic": 1.0}
    assert phases[3].loss_multipliers == {
        "seg_fusion": 0.75,
        "semantic": 1.0,
        "moe": 1.5,
    }


def test_detailed_training_logs_can_be_hidden_from_terminal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_path = tmp_path / "train.log"
    logger = configure_logging(log_file=log_path)
    try:
        logger.info("visible message")
        logger.info("detailed message", extra={"terminal": False})
        for handler in logger.handlers:
            handler.flush()
        terminal = capsys.readouterr().err
        file_text = log_path.read_text(encoding="utf-8")
        assert "visible message" in terminal
        assert "detailed message" not in terminal
        assert "visible message" in file_text
        assert "detailed message" in file_text
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


def test_default_config_builds_final_model() -> None:
    config = load_config(DEFAULT_CONFIG)
    config.model.backbone.channels = [8, 16, 32, 64]
    config.model.backbone.depths = [1, 1, 1, 1]
    config.model.frequency.fdconv_kernel_num = 4
    config.model.moe.router_hidden_channels = 16
    config.model.moe.expert_expansion = 1
    config.model.guidance.focus.hidden_channels = 8
    config.model.guidance.semantic.input_size = 32
    config.model.guidance.semantic.enabled = False
    config.model.feedback.guide_channels = 8
    model = build_model(config)
    assert isinstance(model.core, CustomMultiscaleBackbone)
    assert model.core.is_scientific_model is True


@pytest.mark.integration
def test_default_config_builds_with_local_semantic_assets() -> None:
    config = load_config(DEFAULT_CONFIG)
    model_dir = ROOT / config.model.guidance.semantic.model_dir
    required = ("config.json", "preprocessor_config.json", "pytorch_model.bin")
    if not all((model_dir / name).is_file() for name in required):
        pytest.skip("local SegFormer integration assets are not installed")
    model = build_model(config)
    assert model.feedback.semantic_backend is not None


def test_unknown_config_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "schema_version: 1\nexperiment:\n  unknown_switch: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="unknown_switch"):
        load_config(path)


def test_ground_truth_router_guidance_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad_guidance.yaml"
    path.write_text(
        f"_base_: {DEFAULT_CONFIG}\n"
        "model:\n"
        "  guidance:\n"
        "    focus:\n"
        "      predicted_guidance_only: false\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="Ground-truth guidance"):
        load_config(path)


@pytest.mark.parametrize("task", list(TaskType))
def test_engineering_probe_is_deterministic_and_task_valid(task: TaskType) -> None:
    config = load_config(DEFAULT_CONFIG)
    first = make_probe_batch(config, task, batch_size=2, spatial_size=(19, 21))
    repeated = make_probe_batch(config, task, batch_size=2, spatial_size=(19, 21))
    assert torch.equal(first.source_a.image, repeated.source_a.image)
    assert torch.equal(first.source_b.image, repeated.source_b.image)
    assert first.task is task
    assert first.batch_size == 2
