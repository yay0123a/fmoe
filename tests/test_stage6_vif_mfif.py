from __future__ import annotations

from pathlib import Path

import torch

from tfs_moe_fusion.config import load_config
from tfs_moe_fusion.losses import LossContext, MultiTaskLossManager
from tfs_moe_fusion.model import build_model
from tfs_moe_fusion.types import TaskType
from tfs_moe_fusion.utils import make_probe_batch

ROOT = Path(__file__).resolve().parents[1]


def _config(config_name: str = "stage6_vif_mfif.yaml"):
    config = load_config(ROOT / "configs" / config_name)
    config.model.backbone.channels = [8, 16, 32, 64]
    config.model.backbone.depths = [1, 1, 1, 1]
    config.model.frequency.fdconv_kernel_num = 4
    config.model.moe.expert_dim = 8
    config.model.moe.router_hidden_channels = 8
    config.model.moe.expert_expansion = 1
    config.model.guidance.focus.hidden_channels = 8
    config.model.feedback.guide_channels = 8
    config.training.ema.enabled = False
    return config


def test_stage6_config_trains_only_vif_and_mfif_without_semantic_modules() -> None:
    config = _config()
    assert config.training.epochs == 50
    assert config.training.task_sampling.weights == {"vif": 1.0, "mfif": 1.0}
    assert all(
        set(phase.pattern) <= {"vif", "mfif"}
        for phase in config.training.task_schedule
    )
    assert config.model.moe.experts == [
        "common",
        "low_frequency",
        "detail",
        "infrared_saliency",
        "focus",
    ]
    assert not config.model.guidance.semantic.enabled

    model = build_model(config).train()
    assert model.feedback.semantic_backend is None
    assert "semantic" not in model.shared_expert_bank.specialists

    for task in (TaskType.VIF, TaskType.MFIF):
        batch = make_probe_batch(config, task, spatial_size=(31, 37))
        output = model(batch)
        loss = MultiTaskLossManager(config.training.losses)(
            LossContext(batch, output, task, 0, 0, model)
        ).total
        loss.backward()
        assert torch.isfinite(loss)
        model.zero_grad(set_to_none=True)


def test_stage7_adaptive_ir_profile_enables_ir_and_moe_supervision() -> None:
    config = _config("stage7_adaptive_ir.yaml")

    assert config.data.dataset == "msrs"
    assert config.training.losses.vif.intensity_mode == "adaptive_dark_ir"
    assert config.training.losses.vif.gradient_mode == "adaptive_directional"
    assert config.training.losses.vif.ssim_mode == "adaptive_source"
    assert config.training.losses.vif.ir_intensity_max_weight == 0.55
    assert config.training.losses.vif.coarse_supervision == 0.1
    assert config.training.losses.infrared.weight == 0.2
    assert config.training.losses.moe.enabled

    model = build_model(config).train()
    batch = make_probe_batch(config, TaskType.VIF, spatial_size=(31, 37))
    output = model(batch)
    result = MultiTaskLossManager(config.training.losses)(
        LossContext(batch, output, TaskType.VIF, 0, 0, model)
    )
    result.total.backward()
    assert torch.isfinite(result.total)
    assert "cross_modal_ir_weight/s1" in result.diagnostics
    assert "router_ir_importance" in result.diagnostics
