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
    assert config.experiment.name == "semantic_rt_vif_curriculum_2000_single_gpu"
    assert config.data.dataset == "semantic_rt"
    assert config.data.root == "data/semantic_rt"
    assert config.data.mfif_root == "data/mfif/semantic_rt"
    assert config.data.manifest == (
        "data/splits/semantic_rt_test_uniform_2000_seed3407.txt"
    )
    assert config.training.device.startswith("cuda:")
    assert config.model.guidance.semantic.model_dir == (
        "weights/segformer_b0_cityscapes"
    )
    assert config.model.moe.block_counts == {"s1": 0, "s2": 1, "s3": 1, "s4": 1}
    assert config.model.moe.top_k == 2
    assert config.model.guidance.semantic.detach_guidance_input is True
    assert config.model.feedback.vif_seg_refinement_enabled is False
    assert config.data.num_workers == 4
    assert config.data.pin_memory is True
    warmup = config.training.moe_execution.warmup
    assert (
        warmup.uniform_steps,
        warmup.uniform_to_soft_end,
        warmup.soft_to_topk_end,
    ) == (50, 150, 300)
    assert config.training.task_sampling.strategy == "scheduled"
    assert [phase.pattern for phase in config.training.task_schedule] == [
        ["vif"],
        ["seg", "vif"],
        ["seg", "seg", "vif"],
        ["seg", "seg", "seg", "vif"],
        ["vif"],
        ["mfif", "mfif", "mfif", "vif"],
        ["vif"],
        ["vif", "seg", "vif", "mfif"],
    ]
    assert config.training.diagnostics.loss_gradient_interval == 250
    assert config.training.diagnostics.loss_gradient_until_step == 20_000
    assert config.training.checkpoint.every_epochs == 5
    assert config.training.losses.vif.color == 0.0


def test_stage1_y_only_config_is_a_five_epoch_vif_probe() -> None:
    config = load_config(ROOT / "configs/semantic_rt_y_only_stage1.yaml")
    assert config.experiment.name == "semantic_rt_y_only_stage1_vif_5epoch"
    assert config.training.epochs == 5
    assert config.training.task_sampling.strategy == "scheduled"
    assert len(config.training.task_schedule) == 1
    assert config.training.task_schedule[0].pattern == ["vif"]
    assert config.training.task_schedule[0].end_epoch == 5
    assert config.training.phases.phases[0].end == 5
    assert config.training.losses.vif.color == 0.0
    assert config.training.checkpoint.every_epochs == 1


def test_experiment_b_changes_only_vif_gradient_family_and_weights() -> None:
    config = load_config(ROOT / "configs/semantic_rt_y_only_experiment_b.yaml")
    vif = config.training.losses.vif
    assert config.experiment.name == (
        "semantic_rt_y_only_experiment_b_directional_gradient"
    )
    assert config.training.epochs == 5
    assert len(config.training.task_schedule) == 1
    assert config.training.task_schedule[0].pattern == ["vif"]
    assert (vif.intensity, vif.gradient, vif.ssim) == (1.0, 0.5, 0.5)
    assert (vif.color, vif.coarse_supervision) == (0.0, 0.1)
    assert vif.gradient_mode == "directional_visible_anchor"
    assert vif.ir_gradient_dominance_ratio == 1.2
    assert vif.visible_gradient_support_kernel == 3
    assert config.model.feedback.vif_seg_refinement_enabled is False


def test_experiment_c_adds_only_visible_anchored_weighted_intensity() -> None:
    config = load_config(ROOT / "configs/semantic_rt_y_only_experiment_c.yaml")
    vif = config.training.losses.vif
    assert config.experiment.name == (
        "semantic_rt_y_only_experiment_c_weighted_intensity"
    )
    assert config.training.epochs == 5
    assert config.training.task_schedule[0].pattern == ["vif"]
    assert vif.intensity_mode == "gradient_weighted_visible_anchor"
    assert vif.intensity_energy_normalization == "per_sample_mean"
    assert vif.ir_intensity_max_weight == 0.3
    assert vif.intensity_visible_support_kernel == 3
    assert vif.intensity_weight_smoothing_kernel == 3
    assert vif.gradient_mode == "directional_visible_anchor"
    assert (vif.intensity, vif.gradient, vif.ssim) == (1.0, 0.5, 0.5)
    assert (vif.color, vif.coarse_supervision) == (0.0, 0.1)
    assert config.model.feedback.vif_seg_refinement_enabled is False


def test_experiment_d0_is_a_500_step_no_ssim_probe() -> None:
    config = load_config(ROOT / "configs/semantic_rt_y_only_experiment_d0_no_ssim.yaml")
    vif = config.training.losses.vif

    assert config.experiment.name == (
        "semantic_rt_y_only_experiment_d0_no_ssim_500step"
    )
    assert config.training.epochs == 1
    assert config.training.steps_per_epoch == 500
    assert config.training.scheduler.warmup_steps == 100
    assert config.training.task_schedule[0].pattern == ["vif"]
    assert config.training.task_schedule[0].end_epoch == 1
    assert config.training.phases.phases[0].end == 1
    assert vif.ssim == 0.0
    assert vif.ssim_mode == "visible_anchor"


def test_experiment_d_uses_low_weight_visible_anchored_y_ssim() -> None:
    config = load_config(
        ROOT / "configs/semantic_rt_y_only_experiment_d_stable_ssim.yaml"
    )
    vif = config.training.losses.vif

    assert config.experiment.name == "semantic_rt_y_only_experiment_d_stable_y_ssim"
    assert config.training.epochs == 5
    assert config.training.steps_per_epoch == 750
    assert vif.ssim == 0.1
    assert vif.ssim_mode == "visible_anchor"
    assert vif.intensity_mode == "gradient_weighted_visible_anchor"
    assert vif.gradient_mode == "directional_visible_anchor"


def test_simple_vif_config_uses_classic_two_source_targets() -> None:
    config = load_config(ROOT / "configs/semantic_rt_y_only_simple_vif.yaml")
    vif = config.training.losses.vif

    assert config.experiment.name == "semantic_rt_y_only_simple_vif"
    assert config.training.epochs == 5
    assert config.training.steps_per_epoch == 750
    assert config.training.task_schedule[0].pattern == ["vif"]
    assert (vif.intensity, vif.gradient, vif.ssim) == (1.0, 1.0, 0.1)
    assert (vif.color, vif.coarse_supervision) == (0.0, 0.1)
    assert vif.intensity_mode == "pixel_max"
    assert vif.gradient_mode == "magnitude_max"
    assert vif.ssim_mode == "source_max"
    assert config.model.feedback.vif_seg_refinement_enabled is False


def test_default_config_uses_vif_anchored_loss_schedule() -> None:
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
        (0, 20),
        (20, 21),
        (21, 23),
        (23, 25),
        (25, 30),
        (30, 35),
        (35, 40),
        (40, 50),
    ]
    assert phases[0].loss_multipliers == {"seg_fusion": 1.0, "semantic": 0.25}
    assert phases[1].loss_multipliers == {"seg_fusion": 1.0, "semantic": 0.25}
    assert phases[2].loss_multipliers == {"seg_fusion": 1.0, "semantic": 0.5}
    assert phases[3].loss_multipliers == {"seg_fusion": 1.0, "semantic": 1.0}
    assert all(not phase.loss_multipliers for phase in phases[4:7])
    assert phases[7].loss_multipliers == {"seg_fusion": 1.0, "semantic": 1.0}


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
