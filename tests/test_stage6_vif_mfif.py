from __future__ import annotations

from pathlib import Path

import torch

from tfs_moe_fusion.config import load_config
from tfs_moe_fusion.ir_evidence import build_router_ir_evidence
from tfs_moe_fusion.losses import LossContext, MultiTaskLossManager
from tfs_moe_fusion.model import build_model
from tfs_moe_fusion.moe import MoESiteAdapter, SiteSpatialRouter
from tfs_moe_fusion.trainer import ParameterGroupRegistry, TaskParameterPolicy
from tfs_moe_fusion.types import ModalityType, TaskType
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


def test_stage7_adaptive_ir_profile_disables_balance_but_keeps_moe() -> None:
    config = _config("stage7_adaptive_ir.yaml")

    assert config.data.dataset == "msrs"
    assert config.training.losses.vif.intensity_mode == "hot_object_aware"
    assert config.training.losses.vif.gradient_mode == "independent_directional"
    assert config.training.losses.vif.ssim_mode == "structure_adaptive"
    assert config.training.losses.vif.ir_intensity_max_weight == 0.40
    assert config.training.losses.vif.ir_hot_weight == 0.40
    assert config.training.losses.vif.hot_underexposure_weight == 0.08
    assert config.training.losses.vif.ir_structure_decoupled
    assert config.training.losses.vif.highlight_saturation_threshold == 0.90
    assert config.training.losses.vif.highlight_rgb_clip_threshold == 0.98
    assert config.training.losses.vif.highlight_tone_strength == 6.0
    assert config.training.losses.vif.highlight_ir_detail_scale == 0.06
    assert config.training.losses.vif.highlight_reconstruction_weight == 0.04
    assert config.training.losses.vif.highlight_gradient_weight == 0.02
    assert config.training.losses.vif.ir_structure_max_weight == 0.80
    assert config.training.losses.vif.structure_ssim_scale == 0.40
    assert config.training.losses.vif.coarse_supervision == 0.1
    assert config.training.losses.infrared.weight == 0.12
    assert not config.training.losses.moe.enabled
    assert config.training.losses.moe_starvation.enabled
    assert config.training.losses.moe_starvation.threshold == 0.02
    assert config.training.losses.moe_starvation.patience_steps == 500
    assert config.model.moe.enabled

    model = build_model(config).train()
    batch = make_probe_batch(config, TaskType.VIF, spatial_size=(31, 37))
    output = model(batch)
    result = MultiTaskLossManager(config.training.losses)(
        LossContext(batch, output, TaskType.VIF, 0, 0, model)
    )
    assert output.router_balance_states
    assert all(state.probabilities.requires_grad for state in output.router_balance_states)
    assert all(
        state.opportunity_weights is not None
        and state.opportunity_weights.shape == state.probabilities.shape
        for state in output.router_balance_states
    )
    assert all(
        not diagnostic.probabilities.requires_grad
        for diagnostic in output.router_diagnostics
    )
    assert "moe/soft_balance" not in result.components
    assert "moe/switch_balance" not in result.components
    result.total.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for name, parameter in model.named_parameters()
        if ".router." in name
    )
    assert torch.isfinite(result.total)
    assert "fusion/hot_underexposure" in result.components
    assert "fusion/highlight_reconstruction" in result.components
    assert "fusion/highlight_gradient" in result.components
    assert "ir_hotness_mean" in result.diagnostics
    assert "ir_structure_weight_mean" in result.diagnostics
    assert "highlight_saturation_ratio" in result.diagnostics
    assert "highlight_bloom_ratio" in result.diagnostics
    assert "highlight_target_reduction" in result.diagnostics
    assert "cross_modal_ir_weight/s1" in result.diagnostics
    assert "router_ir_importance" in result.diagnostics


def test_router_ir_evidence_is_directional_order_invariant_and_task_gated() -> None:
    height = width = 33
    ramp = torch.linspace(0.15, 0.75, width).view(1, 1, 1, width)
    visible = ramp.expand(1, 3, height, width).clone()
    infrared = visible.mean(1, keepdim=True)
    infrared[..., 11:22, 11:22] += 0.35
    infrared.clamp_(0, 1)

    evidence = build_router_ir_evidence(
        visible,
        infrared,
        ModalityType.VISIBLE_RGB,
        ModalityType.INFRARED_GRAY,
        TaskType.VIF,
    )
    swapped = build_router_ir_evidence(
        infrared,
        visible,
        ModalityType.INFRARED_GRAY,
        ModalityType.VISIBLE_RGB,
        TaskType.VIF,
    )

    assert evidence.shape == (1, 3, height, width)
    assert torch.isfinite(evidence).all()
    assert evidence[:, :1].amin() >= -1 and evidence[:, :1].amax() <= 1
    assert evidence[:, 1:].amin() >= 0 and evidence[:, 1:].amax() <= 1
    assert torch.allclose(evidence, swapped)
    assert evidence[:, 0, 13:20, 13:20].mean() > evidence[:, 0, :5, :5].mean()
    assert evidence[:, 1, 13:20, 13:20].mean() > evidence[:, 1, :5, :5].mean()
    assert evidence[:, 2, 10:23, 10:23].mean() > evidence[:, 2, :5, :5].mean()

    mfif = build_router_ir_evidence(
        visible,
        infrared,
        ModalityType.VISIBLE_RGB,
        ModalityType.INFRARED_GRAY,
        TaskType.MFIF,
    )
    assert torch.count_nonzero(mfif) == 0


def test_stage8_routes_with_ir_evidence_without_growing_router_input() -> None:
    config = _config("stage8_router_ir_evidence.yaml")
    assert config.experiment.name == "stage8_msrs_router_ir_evidence"
    assert config.model.moe.router_ir_evidence_enabled
    assert config.model.moe.source_aware_expert_evidence_enabled
    assert not config.training.losses.moe.enabled
    assert config.training.losses.moe_starvation.enabled

    model = build_model(config).eval()
    routers = [
        module for module in model.modules() if isinstance(module, SiteSpatialRouter)
    ]
    assert routers
    assert all(router.ir_evidence_enabled for router in routers)
    expected_channels = max(8, config.model.moe.router_hidden_channels) * 4 + 7
    assert all(router.body[0].in_channels == expected_channels for router in routers)
    adapters = [
        module for module in model.modules() if isinstance(module, MoESiteAdapter)
    ]
    assert adapters and all(adapter.source_in is None for adapter in adapters)
    adapter = adapters[0]
    native = torch.randn(1, adapter.feature_in.in_channels, 5, 7)
    torch.testing.assert_close(
        adapter.project_feature(native), adapter.project_source(native)
    )
    registry = ParameterGroupRegistry.from_model(model)
    policy = TaskParameterPolicy(registry, config.training.task_update_policy)
    mfif_trainable = {
        "focus_head",
        "feedback_routers",
        "shared_low",
        "shared_detail",
        "shared_focus",
        "core_site_adapters",
        "feedback_site_adapters",
        "core_routers",
        "mfif_coarse_head",
        "mfif_interactions",
        "mfif_residual_head",
    }
    with policy.apply(TaskType.MFIF):
        for group, parameters in registry.groups.items():
            if parameters:
                assert all(
                    parameter.requires_grad is (group in mfif_trainable)
                    for parameter in parameters
                ), group

    batch = make_probe_batch(config, TaskType.VIF, spatial_size=(31, 37))
    batch.source_b.image[..., 9:22, 12:25] = 1.0
    with torch.no_grad():
        output = model(batch)
    assert output.router_diagnostics
    assert all(
        diagnostic.auxiliary["router/ir_evidence_enabled"]
        for diagnostic in output.router_diagnostics
    )
    assert any(
        diagnostic.auxiliary["router/ir_saliency_mean"] > 0
        for diagnostic in output.router_diagnostics
    )


def test_stage9_gives_specialists_and_feedback_more_training_space() -> None:
    config = _config("stage9_specialist_feedback_space.yaml")
    assert config.experiment.name == "stage9_msrs_specialist_feedback_space"
    assert config.model.moe.specialist_scale_init == 0.05
    assert config.training.losses.vif.coarse_supervision == 0.05
    final_phase = config.training.phases.phases[-1]
    assert final_phase.name == "vif_convergence"
    assert final_phase.loss_multipliers == {"fusion/coarse": 0.0}
