from __future__ import annotations

from pathlib import Path

import pytest

from tfs_moe_fusion.config import load_config

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


import torch

from tfs_moe_fusion.trainer import AMPController


def test_cpu_bf16_amp_performs_optimizer_step() -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    amp = AMPController("bf16", torch.device("cpu"))
    with amp.autocast():
        loss = model(torch.rand(3, 4)).square().mean()
    amp.backward(loss)
    amp.unscale_(optimizer)
    amp.step(optimizer)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


from tfs_moe_fusion.trainer import ModelEMA


def test_ema_updates_and_restores_live_weights() -> None:
    model = torch.nn.Linear(2, 1)
    ema = ModelEMA(model, 0.5)
    before = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(2)
    live = model.weight.detach().clone()
    ema.update(model)
    with ema.apply(model):
        assert not torch.equal(model.weight, live)
    torch.testing.assert_close(model.weight, live)
    assert not torch.equal(before, live)


from tfs_moe_fusion.losses import focus_losses


def test_focus_loss_supervises_selection_boundary_and_confidence() -> None:
    logits = torch.randn(2, 2, 11, 13, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 11, 13)).float()
    values = focus_losses(logits, logits.softmax(1)[:, :1], target)
    assert set(values) == {"focus/selection", "focus/boundary", "focus/confidence"}
    sum(values.values()).backward()
    assert logits.grad is not None


from tfs_moe_fusion.losses import frequency_specialization
from tfs_moe_fusion.types import RouterBalanceState, RouterDiagnostics


def _diagnostic() -> RouterDiagnostics:
    probabilities = torch.tensor([[0.6, 0.4]], requires_grad=True)
    return RouterDiagnostics(
        "test",
        probabilities.log(),
        probabilities,
        torch.tensor([[0, 1]]),
        probabilities,
        torch.ones(1, 2, dtype=torch.bool),
        auxiliary={
            "expert_residuals": {
                "low_frequency": torch.rand(1, 3, 8, 8, requires_grad=True),
                "detail": torch.rand(1, 3, 8, 8, requires_grad=True),
            }
        },
    )


def _balance_state() -> RouterBalanceState:
    probabilities = torch.tensor([[0.6, 0.4]], requires_grad=True)
    return RouterBalanceState(
        probabilities,
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([1.0, 0.0]),
    )


def test_frequency_specialization_reports_both_leakages() -> None:
    values = frequency_specialization((_diagnostic(),))
    assert values["frequency/low_leakage"] > 0
    assert values["frequency/detail_leakage"] > 0


from tfs_moe_fusion.losses import (
    adaptive_ir_blend_weight,
    adaptive_tone_aware_intensity_target,
    align_infrared_luminance,
    directional_gradient_loss,
    directional_gradient_targets,
    highlight_reconstruction_losses,
    highlight_reconstruction_target,
    hot_object_ir_blend_weight,
    hot_underexposure_loss,
    luminance,
    mfif_losses,
    normalized_edge_energy,
    seg_fusion_anchor_losses,
    soft_directional_gradient_loss,
    soft_directional_gradient_targets,
    ssim,
    vif_intensity_target,
    vif_losses,
)


@pytest.mark.parametrize("tone_enabled", [True, False])
def test_tone_aware_target_preserves_normal_regions_and_prioritizes_highlights(
    tone_enabled: bool,
) -> None:
    visible = torch.full((1, 3, 64, 64), 0.2)
    infrared = torch.full((1, 1, 64, 64), 0.2)
    # Equal-area bright regions keep global VIS/IR statistics matched, making
    # the behavior assertions independent of the luminance alignment itself.
    visible[..., 8:24, 8:24] = 1.0
    infrared[..., 32:48, 8:24] = 1.0
    visible[..., 8:24, 40:56] = 1.0
    infrared[..., 8:24, 40:56] = 1.0

    target, ir_weight, hotness, highlight, core, aligned_ir = (
        adaptive_tone_aware_intensity_target(
            visible,
            infrared,
            hot_low=0.08,
            hot_high=0.3,
            hot_weight=0.4,
            max_weight=0.4,
            smoothing_kernel=3,
            saturation_threshold=0.9,
            saturation_transition=0.04,
            rgb_clip_threshold=0.98,
            local_std_threshold=0.025,
            tone_knee=0.75,
            tone_strength=6.0,
            highlight_core_threshold=0.5,
            highlight_tone_enabled=tone_enabled,
        )
    )
    visible_y = luminance(visible)
    normal = (..., slice(52, 60), slice(8, 24))
    visible_highlight = (..., slice(12, 20), slice(12, 20))
    thermal_hot = (..., slice(36, 44), slice(12, 20))
    overlap = (..., slice(12, 20), slice(44, 52))

    torch.testing.assert_close(target[normal], visible_y[normal])
    assert target[thermal_hot].mean() > visible_y[thermal_hot].mean()
    assert hotness[thermal_hot].mean() > 0.9
    for region in (visible_highlight, overlap):
        if tone_enabled:
            assert target[region].mean() < visible_y[region].mean()
        else:
            torch.testing.assert_close(target[region], visible_y[region])
    assert highlight[visible_highlight].mean() > 0.9
    assert core[overlap].min() == 1
    assert ir_weight[overlap].max() == 0
    assert torch.isfinite(aligned_ir).all()


def test_vif_and_mfif_losses_are_finite_and_differentiable() -> None:
    fused = torch.rand(2, 3, 15, 17, requires_grad=True)
    visible = torch.rand_like(fused)
    infrared = torch.rand(2, 1, 15, 17)
    values = {
        **vif_losses(fused, visible, infrared),
        **mfif_losses(fused, visible, False),
    }
    assert values
    sum(values.values()).backward()
    assert fused.grad is not None and torch.isfinite(fused.grad).all()


def test_adaptive_ir_target_prefers_dark_ir_salient_regions() -> None:
    visible = torch.full((1, 1, 24, 24), 0.8)
    visible[..., :12] = 0.15
    infrared = torch.full_like(visible, 0.2)
    infrared[..., 3:9, 3:9] = 0.9

    aligned, weight = adaptive_ir_blend_weight(
        visible,
        infrared,
        max_weight=0.55,
        darkness_threshold=0.38,
        darkness_transition=0.12,
        saliency_weight=0.65,
        visible_support_kernel=3,
        smoothing_kernel=3,
        energy_normalization="per_sample_mean",
    )

    assert aligned.min() >= 0 and aligned.max() <= 1
    assert weight.max() <= 0.55
    assert weight[..., 3:9, 3:9].mean() > weight[..., 3:9, 15:21].mean()


def test_hot_object_ir_admits_thermal_targets_in_bright_and_dark_regions() -> None:
    visible = torch.full((1, 1, 32, 32), 0.72)
    visible[..., :16] = 0.20
    infrared = torch.full_like(visible, 0.08)
    infrared[..., 4:12, 4:12] = 0.95
    infrared[..., 20:28, 20:28] = 0.95

    aligned, weight, hotness = hot_object_ir_blend_weight(
        visible,
        infrared,
        max_weight=0.85,
        darkness_threshold=0.38,
        darkness_transition=0.12,
        saliency_weight=0.65,
        hot_contrast_low=0.08,
        hot_contrast_high=0.30,
        hot_weight=0.80,
        dark_context_weight=0.45,
        edge_weight=0.08,
        visible_support_kernel=3,
        smoothing_kernel=3,
        energy_normalization="per_sample_mean",
    )

    bright_hot = weight[..., 20:28, 20:28].mean()
    bright_background = weight[..., 20:28, 16:20].mean()
    assert aligned[..., 20:28, 20:28].mean() > visible[..., 20:28, 20:28].mean()
    assert hotness[..., 4:12, 4:12].mean() > 0.9
    assert hotness[..., 20:28, 20:28].mean() > 0.9
    assert bright_hot > 0.7
    assert bright_hot > bright_background + 0.6


def test_highlight_reconstruction_compresses_only_clipped_flat_regions() -> None:
    visible = torch.full((1, 3, 40, 56), 0.2)
    visible[..., 8:24, 5:19] = 0.95
    visible[..., 8:32, 32:52] = 1.0
    infrared = torch.full((1, 1, 40, 56), 0.5)
    infrared[..., 8:32, 32:42] = 0.25
    infrared[..., 8:32, 42:52] = 0.75
    normal = luminance(visible)
    structure = torch.ones_like(normal)
    kwargs = {
        "saturation_threshold": 0.9,
        "saturation_transition": 0.04,
        "rgb_clip_threshold": 0.98,
        "local_std_threshold": 0.025,
        "tone_knee": 0.75,
        "tone_strength": 6.0,
    }

    target, saturation, bloom = highlight_reconstruction_target(
        normal, visible, infrared, structure, ir_detail_scale=0.06, **kwargs
    )
    base_target, _, _ = highlight_reconstruction_target(
        normal, visible, infrared, structure, ir_detail_scale=0.0, **kwargs
    )

    assert saturation[..., 12:20, 9:15].max() == 0
    assert saturation[..., 12:28, 36:48].mean() > 0.7
    torch.testing.assert_close(target[..., 12:20, 9:15], normal[..., 12:20, 9:15])
    assert target[..., 12:28, 36:48].mean() < 0.93
    detail_delta = target[..., 12:28, 36:48] - base_target[..., 12:28, 36:48]
    assert detail_delta.abs().mean() > 1e-3
    assert detail_delta.mean().abs() < 0.02
    assert bloom.mean() > saturation.mean()


def test_highlight_reconstruction_loss_prefers_compressed_target() -> None:
    visible = torch.ones(1, 1, 16, 16, requires_grad=True)
    target = torch.full_like(visible, 0.85)
    bloom = torch.ones_like(visible)
    bad = highlight_reconstruction_losses(visible, target, bloom)
    good = highlight_reconstruction_losses(target, target, bloom)

    assert bad[0] > 0.1
    assert good[0] == 0
    assert good[1] < bad[1]
    sum(bad).backward()
    assert visible.grad is not None and torch.isfinite(visible.grad).all()


def test_hot_underexposure_loss_is_region_normalized_and_differentiable() -> None:
    visible = torch.full((1, 1, 24, 24), 0.2)
    infrared = visible.clone()
    infrared[..., 8:16, 8:16] = 0.9
    hotness = torch.zeros_like(visible)
    hotness[..., 8:16, 8:16] = 1

    fused = visible.clone().requires_grad_()
    loss = hot_underexposure_loss(
        fused,
        visible,
        infrared,
        hotness,
        minimum_contrast_retention=0.65,
    )
    assert loss > 0.4
    loss.backward()
    assert fused.grad is not None and torch.isfinite(fused.grad).all()

    sufficient = visible + 0.65 * (infrared - visible)
    satisfied_loss = hot_underexposure_loss(
        sufficient,
        visible,
        infrared,
        hotness,
        minimum_contrast_retention=0.65,
    )
    torch.testing.assert_close(satisfied_loss, torch.zeros_like(satisfied_loss))


def test_adaptive_vif_losses_are_finite_and_differentiable() -> None:
    fused_y = torch.rand(2, 1, 19, 23, requires_grad=True)
    visible = torch.rand(2, 3, 19, 23)
    infrared = torch.rand(2, 1, 19, 23)
    values = vif_losses(
        fused_y.expand(-1, 3, -1, -1),
        visible,
        infrared,
        fused_y,
        intensity_mode="adaptive_dark_ir",
        ir_intensity_max_weight=0.55,
        ir_darkness_threshold=0.38,
        ir_darkness_transition=0.12,
        ir_saliency_weight=0.65,
        gradient_mode="adaptive_directional",
        ssim_mode="adaptive_source",
    )
    sum(values.values()).backward()
    assert fused_y.grad is not None and torch.isfinite(fused_y.grad).all()


def test_structure_adaptive_ssim_cannot_escape_with_visible_only_output() -> None:
    visible = torch.linspace(0, 1, 19 * 23).reshape(1, 1, 19, 23)
    infrared = torch.flip(visible, dims=(-1,))
    fused = visible.clone().requires_grad_()
    values = vif_losses(
        fused,
        visible,
        infrared,
        fused,
        target_intensity=visible,
        intensity_weight=torch.zeros_like(visible),
        structure_weight=torch.ones_like(visible),
        structure_ssim_scale=0.4,
        gradient_mode="independent_directional",
        ssim_mode="structure_adaptive",
    )

    assert values["fusion/ssim"] > 0
    sum(values.values()).backward()
    assert fused.grad is not None and torch.isfinite(fused.grad).all()


def test_aligned_ir_matches_visible_global_luminance_statistics() -> None:
    visible = torch.rand(2, 1, 31, 29)
    infrared = torch.rand(2, 1, 31, 29) * 0.2 + 0.1
    aligned = align_infrared_luminance(visible, infrared)
    torch.testing.assert_close(
        aligned.mean((-2, -1)), visible.mean((-2, -1)), atol=2e-3, rtol=0
    )


def test_ssim_stays_bounded_under_bf16_autocast() -> None:
    vertical = torch.linspace(0, 1, 64).view(1, 1, 64, 1)
    horizontal = torch.linspace(0, 1, 64).view(1, 1, 1, 64)
    left = 0.4 + 0.005 * (horizontal + vertical - 1)
    right = 0.4 + 0.005 * (horizontal - vertical)

    expected = ssim(left, right)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = ssim(left, right)

    assert torch.isfinite(actual)
    assert -1 <= actual <= 1
    torch.testing.assert_close(actual, expected)


def test_ssim_is_one_for_identical_constant_images() -> None:
    image = torch.full((2, 3, 17, 19), 0.4, requires_grad=True)
    score = ssim(image, image)

    torch.testing.assert_close(score, torch.tensor(1.0), atol=5e-4, rtol=0)
    (1 - score).backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


def test_vif_visible_anchor_ssim_uses_y_not_free_rgb_chroma() -> None:
    predicted_y = torch.rand(1, 1, 17, 19, requires_grad=True)
    visible = torch.rand(1, 3, 17, 19)
    infrared = torch.rand(1, 1, 17, 19)
    fused_with_unrelated_chroma = torch.rand_like(visible)

    values = vif_losses(
        fused_with_unrelated_chroma,
        visible,
        infrared,
        predicted_y,
        ssim_mode="visible_anchor",
    )

    expected = 1 - ssim(predicted_y, luminance(visible))
    torch.testing.assert_close(values["fusion/ssim"], expected)
    values["fusion/ssim"].backward()
    assert predicted_y.grad is not None and torch.isfinite(predicted_y.grad).all()


def test_directional_gradient_protects_nearby_visible_edges() -> None:
    visible = torch.zeros(2, 1, 15, 15)
    infrared = torch.zeros_like(visible)
    visible[0, :, :, 5:] = 0.5
    infrared[0, :, :, 6:] = 0.55
    infrared[1, :, :, 9:] = 0.8

    _, _, choose_ir = directional_gradient_targets(
        visible,
        infrared,
        ir_dominance_ratio=1.2,
        visible_support_kernel=3,
    )

    assert not choose_ir[0, :, :, 4:8].any()
    assert choose_ir[1, :, :, 8:11].any()
    torch.testing.assert_close(
        directional_gradient_loss(
            visible[:1],
            visible[:1],
            infrared[:1],
            ir_dominance_ratio=1.2,
            visible_support_kernel=3,
        ),
        torch.tensor(0.0),
    )


def test_directional_gradient_loss_is_differentiable() -> None:
    fused_y = torch.rand(2, 1, 15, 17, requires_grad=True)
    loss = directional_gradient_loss(
        fused_y,
        torch.rand_like(fused_y),
        torch.rand_like(fused_y),
    )
    loss.backward()
    assert fused_y.grad is not None and torch.isfinite(fused_y.grad).all()


def test_soft_directional_gradient_is_continuous_and_protects_visible_edges() -> None:
    visible = torch.zeros(2, 1, 15, 15)
    infrared = torch.zeros_like(visible)
    visible[0, :, :, 5:] = 0.5
    infrared[0, :, :, 6:] = 0.55
    infrared[1, :, :, 9:] = 0.8

    _, _, ir_weight = soft_directional_gradient_targets(
        visible,
        infrared,
        ir_dominance_ratio=1.2,
        visible_support_kernel=3,
        transition=0.15,
        min_magnitude=0.02,
    )

    assert ir_weight[0, :, :, 4:8].max() < 0.5
    assert ir_weight[1, :, :, 8:11].max() > 0.5
    fused_y = torch.rand(2, 1, 15, 17, requires_grad=True)
    loss = soft_directional_gradient_loss(
        fused_y,
        torch.rand_like(fused_y),
        torch.rand_like(fused_y),
    )
    loss.backward()
    assert fused_y.grad is not None and torch.isfinite(fused_y.grad).all()


def test_soft_directional_gradient_stays_fp32_under_bf16_autocast() -> None:
    fused_y = torch.rand(1, 1, 15, 17, requires_grad=True)
    visible_y = torch.rand_like(fused_y)
    infrared = torch.rand_like(fused_y)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = soft_directional_gradient_loss(fused_y, visible_y, infrared)

    assert loss.dtype is torch.float32
    loss.backward()
    assert fused_y.grad is not None and torch.isfinite(fused_y.grad).all()


def test_normalized_edge_energy_has_no_flat_region_floor() -> None:
    flat = torch.full((2, 1, 15, 17), 0.4)
    energy = normalized_edge_energy(flat)
    assert torch.count_nonzero(energy[..., 2:-2, 2:-2]) == 0


def test_weighted_intensity_is_visible_anchored_and_bounded() -> None:
    visible = torch.full((1, 1, 17, 19), 0.4)
    infrared = visible.clone()
    infrared[..., 5:12, 6:13] = 0.9
    target, weight = vif_intensity_target(
        visible,
        infrared,
        mode="gradient_weighted_visible_anchor",
        energy_normalization="per_sample_mean",
        ir_max_weight=0.3,
        visible_support_kernel=3,
        weight_smoothing_kernel=3,
    )

    assert weight is not None
    assert weight.min() >= 0
    assert weight.max() <= 0.3 + 1e-7
    assert torch.count_nonzero(weight[..., 4:13, 5:14]) > 0
    assert torch.all(target >= torch.minimum(visible, infrared))
    assert torch.all(target <= torch.maximum(visible, infrared))
    torch.testing.assert_close(target[weight == 0], visible[weight == 0])


def test_weighted_intensity_keeps_flat_modalities_at_visible_y() -> None:
    visible = torch.full((1, 1, 15, 17), 0.2)
    infrared = torch.full_like(visible, 0.9)
    target, weight = vif_intensity_target(
        visible,
        infrared,
        mode="gradient_weighted_visible_anchor",
    )

    assert weight is not None
    torch.testing.assert_close(weight, torch.zeros_like(weight), atol=1e-6, rtol=0)
    torch.testing.assert_close(target, visible, atol=1e-6, rtol=0)


def test_seg_fusion_anchor_is_minimal_and_differentiable() -> None:
    fused = torch.rand(2, 3, 15, 17, requires_grad=True)
    visible = torch.rand_like(fused)
    infrared = torch.rand(2, 1, 15, 17)
    values = seg_fusion_anchor_losses(fused, visible, infrared)
    assert set(values) == {"seg_fusion/intensity", "seg_fusion/gradient"}
    sum(values.values()).backward()
    assert fused.grad is not None and torch.isfinite(fused.grad).all()


from tfs_moe_fusion.losses import semantic_losses


def test_semantic_loss_handles_ignore_index() -> None:
    logits = torch.randn(1, 4, 9, 7, requires_grad=True)
    target = torch.randint(0, 4, (1, 9, 7))
    target[:, 0] = 255
    values = semantic_losses(
        logits, target, torch.rand(1, 1, 9, 7), torch.rand(1, 3, 9, 7), 255, None
    )
    sum(values.values()).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_semantic_loss_handles_fully_ignored_crop() -> None:
    logits = torch.randn(1, 19, 9, 7, requires_grad=True)
    target = torch.full((1, 9, 7), 255)
    values = semantic_losses(
        logits,
        target,
        torch.rand(1, 1, 9, 7),
        torch.rand(1, 3, 9, 7),
        255,
        None,
    )
    assert set(values) == {"semantic/ce", "semantic/dice"}
    total = sum(values.values())
    assert torch.isfinite(total)
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


from pathlib import Path

from tfs_moe_fusion.losses import LossContext, LossOutput, MultiTaskLossManager
from tfs_moe_fusion.model import build_model
from tfs_moe_fusion.types import TaskType
from tfs_moe_fusion.utils import make_probe_batch

ROOT = Path(__file__).resolve().parents[1]


def _enable_step1_vif(config) -> None:
    vif = config.training.losses.vif
    vif.objective_mode = "adaptive_three_term"
    vif.active_terms = ["intensity"]
    vif.intensity_mode = "tone_aware_hot_object"


def test_step1_vif_has_only_adaptive_intensity_loss() -> None:
    config = _smoke_config()
    _enable_step1_vif(config)
    model = build_model(config).train()
    batch = make_probe_batch(config, TaskType.VIF)
    manager = MultiTaskLossManager(config.training.losses)
    result = manager(LossContext(batch, model(batch), TaskType.VIF, 0, 0, model))

    assert set(result.components) == {"fusion/intensity"}
    assert set(result.weighted_components) == {"fusion/intensity"}
    assert result.diagnostics["component_count"] == 1
    assert result.diagnostics["vif/objective_mode"] == "adaptive_three_term"
    assert manager.loss_log_vars is not None
    expected = -torch.log(torch.tensor([1.0, 1.5, 0.3]))
    torch.testing.assert_close(manager.loss_log_vars.detach(), expected)
    for name in ("intensity", "gradient", "ssim"):
        assert f"loss_weight/{name}" in result.diagnostics
        assert f"loss_log_var/{name}" in result.diagnostics
    result.total.backward()
    assert manager.loss_log_vars.grad is not None
    assert torch.isfinite(manager.loss_log_vars.grad).all()


def test_loss_manager_returns_structured_vif_output() -> None:
    config = _smoke_config()
    model = build_model(config).train()
    batch = make_probe_batch(config, TaskType.VIF)
    value = MultiTaskLossManager(config.training.losses)(
        LossContext(batch, model(batch), TaskType.VIF, 0, 0, model)
    )
    assert value.total.requires_grad and "fusion/intensity" in value.components
    assert value.diagnostics["task"] == "vif" and torch.isfinite(value.total)
    for name in (
        "chroma_cb_error",
        "chroma_cr_error",
        "y_gamut_clip_ratio",
        "coarse_final_y_mae",
        "y_residual_rms",
        "y_residual_to_coarse_ratio",
        "y_residual_scale",
        "y_gradient_loss",
    ):
        assert name in value.diagnostics
        assert torch.isfinite(torch.as_tensor(value.diagnostics[name]))


def test_loss_manager_uses_configured_directional_vif_gradient() -> None:
    config = _smoke_config()
    config.training.losses.vif.gradient_mode = "directional_visible_anchor"
    config.training.losses.vif.ir_gradient_dominance_ratio = 1.2
    config.training.losses.vif.visible_gradient_support_kernel = 3
    model = build_model(config).train()
    batch = make_probe_batch(config, TaskType.VIF)
    output = model(batch)
    value = MultiTaskLossManager(config.training.losses)(
        LossContext(batch, output, TaskType.VIF, 0, 0, model)
    )
    expected = directional_gradient_loss(
        output.fused_y,
        batch.visible_source.image,
        batch.infrared_source.image,
        ir_dominance_ratio=1.2,
        visible_support_kernel=3,
    )

    torch.testing.assert_close(value.components["fusion/gradient"], expected)
    assert value.diagnostics["vif/gradient_mode"] == "directional_visible_anchor"


def test_loss_manager_uses_configured_weighted_intensity_target() -> None:
    config = _smoke_config()
    vif = config.training.losses.vif
    vif.intensity_mode = "gradient_weighted_visible_anchor"
    vif.ir_intensity_max_weight = 0.3
    model = build_model(config).train()
    batch = make_probe_batch(config, TaskType.VIF)
    output = model(batch)
    value = MultiTaskLossManager(config.training.losses)(
        LossContext(batch, output, TaskType.VIF, 0, 0, model)
    )
    target, weight = vif_intensity_target(
        batch.visible_source.image,
        batch.infrared_source.image,
        mode="gradient_weighted_visible_anchor",
        energy_normalization=vif.intensity_energy_normalization,
        ir_max_weight=0.3,
        visible_support_kernel=vif.intensity_visible_support_kernel,
        weight_smoothing_kernel=vif.intensity_weight_smoothing_kernel,
    )

    assert output.fused_y is not None and weight is not None
    torch.testing.assert_close(
        value.components["fusion/intensity"],
        (output.fused_y - target).abs().mean(),
    )
    assert value.diagnostics["vif/intensity_mode"] == (
        "gradient_weighted_visible_anchor"
    )
    torch.testing.assert_close(
        value.diagnostics["ir_intensity_weight_mean"], weight.mean()
    )
    assert value.diagnostics["ir_intensity_weight_max"] <= 0.3 + 1e-7


def test_seg_loss_has_fusion_anchor_without_vif_ssim_or_color() -> None:
    config = _smoke_config()
    model = build_model(config).train()
    batch = make_probe_batch(config, TaskType.SEG)
    value = MultiTaskLossManager(config.training.losses)(
        LossContext(
            batch,
            model(batch),
            TaskType.SEG,
            0,
            0,
            model,
            phase="stabilization",
            loss_multipliers={"seg_fusion": 1.0, "semantic": 0.25},
        )
    )
    assert "seg_fusion/intensity" in value.components
    assert "seg_fusion/gradient" in value.components
    assert "fusion/ssim" not in value.components
    assert "fusion/color" not in value.components
    assert "semantic/boundary" not in value.components
    assert "semantic/coarse" not in value.components
    assert not any(name.startswith("frequency/") for name in value.components)
    assert not any(name.startswith("consistency/") for name in value.components)
    assert "infrared/preservation" not in value.components
    torch.testing.assert_close(
        value.weighted_components["seg_fusion/intensity"],
        value.components["seg_fusion/intensity"],
    )


from tfs_moe_fusion.losses import moe_balance_loss


def test_moe_balance_is_availability_aware_and_finite() -> None:
    importance, load, entropy = moe_balance_loss((_balance_state(),))
    assert importance.isfinite() and load.isfinite() and entropy.isfinite()


def test_switch_balance_routes_gradient_through_probabilities() -> None:
    state = _balance_state()
    soft, switch, _ = moe_balance_loss((state,))
    (soft + switch).backward()
    assert state.probabilities.grad is not None
    assert torch.isfinite(state.probabilities.grad).all()


from tfs_moe_fusion.trainer import (
    ExpertOnlyParameterPolicy,
    MoEExecutionScheduler,
    RouterLoadMonitor,
    expert_regularizer_flags,
    loss_group_gradient_statistics,
    scheduled_task,
)


def test_seg_loss_group_gradient_ratio_is_reported() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    fusion = parameter.square()
    semantic = (3 * parameter).square()
    result = LossOutput(
        fusion + semantic,
        {},
        {"seg_fusion/intensity": fusion, "semantic/ce": semantic},
        {},
        {},
    )
    values = loss_group_gradient_statistics(result, (parameter,))
    assert values["grad_norm/seg_fusion"] == pytest.approx(4.0)
    assert values["grad_norm/semantic"] == pytest.approx(36.0)
    assert values["grad_ratio/semantic_to_fusion"] == pytest.approx(9.0)


def test_scheduled_task_resets_each_phase_pattern() -> None:
    config = load_config(ROOT / "configs/default.yaml")
    phases = config.training.task_schedule
    steps = config.training.steps_per_epoch

    assert scheduled_task(phases, 0, steps) is TaskType.VIF
    assert scheduled_task(phases, 20 * steps, steps) is TaskType.SEG
    assert scheduled_task(phases, 20 * steps + 1, steps) is TaskType.VIF
    assert scheduled_task(phases, 21 * steps, steps) is TaskType.SEG
    assert scheduled_task(phases, 21 * steps + 2, steps) is TaskType.VIF
    assert scheduled_task(phases, 25 * steps, steps) is TaskType.VIF
    assert scheduled_task(phases, 30 * steps, steps) is TaskType.MFIF
    assert scheduled_task(phases, 30 * steps + 3, steps) is TaskType.VIF
    assert scheduled_task(phases, 40 * steps, steps) is TaskType.VIF
    assert scheduled_task(phases, 40 * steps + 1, steps) is TaskType.SEG


def test_loss_group_gradient_statistics_skips_detached_namespace() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    fusion = parameter.square()
    result = LossOutput(
        fusion,
        {},
        {
            "seg_fusion/intensity": fusion,
            "semantic/ce": torch.tensor(1.0),
        },
        {},
        {},
    )
    assert loss_group_gradient_statistics(result, (parameter,)) == {}


def test_moe_execution_schedule_is_continuous_and_refreshes() -> None:
    config = _smoke_config().training.moe_execution
    monitor = RouterLoadMonitor(config)
    scheduler = MoEExecutionScheduler(config, monitor)

    uniform = scheduler.resolve(49)
    uniform_boundary = scheduler.resolve(50)
    uniform_to_soft_last = scheduler.resolve(149)
    soft_boundary = scheduler.resolve(150)
    soft_to_topk_last = scheduler.resolve(299)
    sparse_boundary = scheduler.resolve(300)
    refresh = scheduler.resolve(500)

    assert uniform is not None and uniform.mode == "dense_uniform"
    assert uniform_boundary is not None
    assert uniform_boundary.mode == "dense_annealed"
    assert uniform_boundary.uniform_to_soft == pytest.approx(0.0)
    assert uniform_to_soft_last is not None
    assert uniform_to_soft_last.mode == "dense_annealed"
    assert soft_boundary is not None
    assert soft_boundary.mode == "dense_annealed"
    assert soft_boundary.uniform_to_soft == pytest.approx(1.0)
    assert soft_boundary.soft_to_topk == pytest.approx(0.0)
    assert soft_to_topk_last is not None
    assert soft_to_topk_last.mode == "dense_annealed"
    assert sparse_boundary is not None and sparse_boundary.mode == "sparse_batch"
    assert refresh is not None and refresh.mode == "expert_refresh"
    assert refresh.expert_only and refresh.detach_router


def test_zero_frequency_multiplier_skips_only_frequency_regularizers() -> None:
    config = _smoke_config().training.losses
    config.frequency.enabled = True
    config.infrared.enabled = True
    frequency, infrared = expert_regularizer_flags(
        config, {"frequency": 0.0}, TaskType.VIF
    )
    assert frequency is False
    assert infrared is True


def test_router_monitor_recovery_state_round_trips() -> None:
    config = _smoke_config().training.moe_execution
    config.monitor.patience_steps = 1
    config.monitor.starvation_threshold = 0.2
    monitor = RouterLoadMonitor(config)
    probabilities = torch.tensor([[0.9, 0.1]])
    diagnostic = RouterDiagnostics(
        "s2.moe0",
        probabilities.log(),
        probabilities,
        torch.tensor([[0]]),
        torch.ones(1, 1),
        torch.ones(1, 2, dtype=torch.bool),
    )
    monitor.update((diagnostic,), step=10)
    assert monitor.in_recovery(11)

    restored = RouterLoadMonitor(config)
    restored.load_state_dict(monitor.state_dict())
    assert restored.state_dict() == monitor.state_dict()
    policy = MoEExecutionScheduler(config, restored).resolve(11)
    assert policy is not None
    assert policy.temperature >= config.recovery.temperature_floor
    assert policy.noise_std >= config.recovery.noise_std


def test_availability_starvation_floor_is_task_local_and_one_sided() -> None:
    from tfs_moe_fusion.losses import moe_starvation_floor_loss
    from tfs_moe_fusion.trainer import RouterLoadMonitor
    from tfs_moe_fusion.types import RouterBalanceState, RouterDiagnostics, TaskType

    project = _smoke_config()
    config = project.training.losses.moe_starvation
    config.enabled = True
    config.ema_decay = 0.5
    config.patience_steps = 2
    config.release_patience_steps = 2
    config.ramp_steps = 1
    monitor = RouterLoadMonitor(project.training.moe_execution, config)
    probabilities = torch.tensor([[0.99, 0.01]])
    opportunity = torch.ones_like(probabilities)
    diagnostic = RouterDiagnostics(
        "s2.moe0",
        probabilities.log(),
        probabilities,
        torch.tensor([[0]]),
        torch.ones(1, 1),
        torch.ones(1, 2, dtype=torch.bool),
        auxiliary={
            "expert_names": ("detail", "infrared_saliency"),
            "starvation_opportunity": opportunity,
        },
    )
    monitor.update((diagnostic,), 10, TaskType.VIF)
    assert monitor.starvation_strengths(TaskType.VIF) == {
        "s2.moe0": [0.0, 0.0]
    }
    metrics = monitor.update((diagnostic,), 11, TaskType.VIF)
    assert monitor.starvation_strengths(TaskType.VIF) == {
        "s2.moe0": [0.0, 1.0]
    }
    assert monitor.starvation_strengths(TaskType.MFIF) == {}
    assert metrics["router_recovery_active"] == 0.0

    live = probabilities.clone().requires_grad_()
    state = RouterBalanceState(
        live,
        torch.ones(1, 2, dtype=torch.bool),
        block_id="s2.moe0",
        expert_names=("detail", "infrared_saliency"),
        opportunity_weights=opportunity,
    )
    loss = moe_starvation_floor_loss(
        (state,),
        monitor.starvation_strengths(TaskType.VIF),
        threshold=config.threshold,
        evidence_threshold=config.evidence_threshold,
    )
    assert loss.detach().item() == pytest.approx(0.25)
    loss.backward()
    assert live.grad is not None and live.grad[0, 1] < 0

    restored = RouterLoadMonitor(project.training.moe_execution, config)
    restored.load_state_dict(monitor.state_dict())
    assert restored.starvation_strengths(TaskType.VIF) == {
        "s2.moe0": [0.0, 1.0]
    }


def test_starvation_floor_ignores_insufficient_physical_evidence() -> None:
    from tfs_moe_fusion.losses import moe_starvation_floor_loss
    from tfs_moe_fusion.types import RouterBalanceState

    probabilities = torch.tensor([[0.99, 0.01]], requires_grad=True)
    state = RouterBalanceState(
        probabilities,
        torch.ones(1, 2, dtype=torch.bool),
        block_id="s2.moe0",
        opportunity_weights=torch.tensor([[1.0, 0.01]]),
    )
    loss = moe_starvation_floor_loss(
        (state,),
        {"s2.moe0": [0.0, 1.0]},
        threshold=0.02,
        evidence_threshold=0.05,
    )
    assert loss == 0


def test_router_monitor_batches_blocks_without_reducing_local_validity(monkeypatch) -> None:
    import tfs_moe_fusion.trainer as trainer_module

    config = _smoke_config().training.moe_execution
    config.monitor.ema_decay = 0.5
    config.monitor.patience_steps = 2
    config.monitor.starvation_threshold = 0.2
    monitor = RouterLoadMonitor(config)
    calls = []

    def fake_reduce(value):
        calls.append(value.clone())
        return value  # One packed collective per step, excluding validity.

    monkeypatch.setattr(trainer_module, "reduce_mean", fake_reduce)
    diagnostics = []
    for name, probs in (("a", [0.75, 0.25]), ("b", [0.6, 0.4, 0.0])):
        probabilities = torch.tensor([probs])
        diagnostics.append(RouterDiagnostics(
            name, probabilities.log(), probabilities,
            torch.tensor([[0]]), torch.ones(1, 1), probabilities > 0,
        ))
    first = monitor.update(tuple(diagnostics), 10)
    assert len(calls) == 1 and calls[0].numel() == 7
    assert monitor.usage_ema == {"a": [1.0, 0.0], "b": [1.0, 0.0, 0.0]}
    assert monitor.starvation_counters["b"] == [0, 1, 0]
    assert first["router_recovery_active"] == 0
    second = monitor.update(tuple(diagnostics), 11)
    assert len(calls) == 2
    assert second["router_recovery_active"] == 1
    assert monitor.recovery_until_step == 12 + config.recovery.steps
    assert monitor.starvation_counters["b"] == [0, 0, 0]


@pytest.mark.parametrize("max_norm", [None, 1.0])
def test_gradient_statistics_preserves_norm_and_clipping(max_norm) -> None:
    from tfs_moe_fusion.trainer import gradient_statistics

    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0, 4.0]])
    result = gradient_statistics(model, max_norm=max_norm)
    assert result == {"gradient_norm": 5.0, "nonfinite_gradients": 0.0}
    expected = torch.tensor([[3.0, 4.0]])
    if max_norm is not None:
        expected *= max_norm / (5.0 + 1e-6)
    torch.testing.assert_close(model.weight.grad, expected)


def test_router_statistics_preserves_scalar_values_and_detaches_graphs() -> None:
    from tfs_moe_fusion.trainer import router_statistics

    probabilities = torch.tensor([[0.75, 0.25]], requires_grad=True)
    entropy = -(probabilities * probabilities.log()).sum(1)
    diagnostic = RouterDiagnostics(
        "block", probabilities.log(), probabilities,
        torch.tensor([[0]]), torch.ones(1, 1), torch.ones(1, 2, dtype=torch.bool),
        entropy=entropy,
        auxiliary={
            "expert_residual_rms": {"common": probabilities[0, 0]},
            "expert_weighted_contribution_rms": {"common": 0.5},
            "expert_effective_contribution_ratio": {"common": 0.05},
            "common_effective_contribution_ratio": 0.05,
            "specialist_effective_contribution_ratio": 0.025,
            "common_scale_rms": 0.1,
            "specialist_scale_rms": 0.03,
            "router/top1_margin": probabilities[0, 0] - probabilities[0, 1],
            "residual_scale_rms": 0.125,
        },
    )
    result = router_statistics((diagnostic,))
    assert result["router_entropy"] == pytest.approx(entropy.item())
    assert result["router_max_load"] == 1.0
    assert result["expert_residual_rms/block/common"] == 0.75
    assert result["expert_weighted_contribution_rms/block/common"] == 0.5
    assert result["expert_effective_contribution_ratio/block/common"] == 0.05
    assert result["common_effective_contribution_ratio/block"] == 0.05
    assert result["specialist_effective_contribution_ratio/block"] == 0.025
    assert result["common_scale_rms/block"] == 0.1
    assert result["specialist_scale_rms/block"] == 0.03
    assert result["router/top1_margin/block"] == 0.5
    assert result["moe_residual_scale/block"] == 0.125
    assert all(isinstance(value, (float, str)) for value in result.values())


@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_trainer_rejects_nonfinite_gradients_before_optimizer(
    tmp_path, semantic_rt_assets, monkeypatch, clip, bad_value
) -> None:
    config = _smoke_config(semantic_rt_assets)
    config.training.gradient_clip.enabled = clip
    trainer = Trainer(build_model(config), config, torch.device("cpu"), tmp_path)
    parameter = next(trainer.model.parameters())
    parameter.register_hook(lambda grad: torch.full_like(grad, bad_value))
    stepped = []
    monkeypatch.setattr(trainer.optimizer, "step", lambda *a, **k: stepped.append(True))
    with pytest.raises(FloatingPointError, match="non-finite gradients"):
        trainer.train_step(TaskType.VIF)
    assert not stepped and trainer.state.global_step == 0
    trainer.provider.close()


def test_expert_only_policy_freezes_every_nonexpert_group() -> None:
    config = _smoke_config()
    registry = ParameterGroupRegistry.from_model(build_model(config))
    policy = ExpertOnlyParameterPolicy(registry)
    with policy.apply(True):
        for group, parameters in registry.groups.items():
            expected = group in policy.expert_groups
            assert all(parameter.requires_grad is expected for parameter in parameters)
    assert all(
        parameter.requires_grad
        for parameters in registry.groups.values()
        for parameter in parameters
    )


from pathlib import Path

from tfs_moe_fusion.trainer import build_optimizer

ROOT = Path(__file__).resolve().parents[1]


def test_optimizer_groups_cover_trainable_parameters_exactly_once() -> None:
    config = _smoke_config()
    model = build_model(config)
    optimizer, registry = build_optimizer(model, config.training.optimizer)
    registered = [
        parameter for values in registry.groups.values() for parameter in values
    ]
    optimized = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    assert len({id(parameter) for parameter in registered}) == len(registered)
    assert {id(parameter) for parameter in optimized} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert registry.names["core.fusion_y_head.0.weight"] == "vif_coarse_head"
    assert registry.names["core.mfif_rgb_head.0.weight"] == "mfif_coarse_head"


from pathlib import Path

from tfs_moe_fusion.trainer import ParameterGroupRegistry, TaskParameterPolicy

ROOT = Path(__file__).resolve().parents[1]


def test_mfif_policy_temporarily_freezes_ir_experts() -> None:
    config = _smoke_config()
    registry = ParameterGroupRegistry.from_model(build_model(config))
    policy = TaskParameterPolicy(registry, config.training.task_update_policy)
    parameters = registry.parameters("ir_experts")
    assert parameters
    with policy.apply(TaskType.MFIF):
        assert all(not parameter.requires_grad for parameter in parameters)
    assert all(parameter.requires_grad for parameter in parameters)


def test_task_policy_scales_only_configured_group_gradients() -> None:
    config = _smoke_config()
    registry = ParameterGroupRegistry.from_model(build_model(config))
    config.training.task_update_policy.gradient_scales = {
        "mfif": {"ir_experts": 0.25}
    }
    policy = TaskParameterPolicy(registry, config.training.task_update_policy)
    infrared = registry.parameters("ir_experts")
    detail = registry.parameters("detail_experts")
    assert infrared and detail
    for parameter in (*infrared, *detail):
        parameter.grad = torch.ones_like(parameter)

    policy.scale_gradients(TaskType.MFIF)

    assert all(torch.all(parameter.grad == 0.25) for parameter in infrared)
    assert all(torch.all(parameter.grad == 1.0) for parameter in detail)


from tfs_moe_fusion.losses import low_frequency_consistency


def test_low_frequency_consistency_is_zero_for_equal_images() -> None:
    image = torch.rand(1, 3, 13, 15)
    torch.testing.assert_close(
        low_frequency_consistency(image, image, 5, 1.0), torch.tensor(0.0)
    )


from pathlib import Path

from tfs_moe_fusion.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]


def test_step1_trainer_updates_and_restores_loss_weights(
    tmp_path: Path, semantic_rt_assets: tuple[Path, Path, Path]
) -> None:
    config = _smoke_config(semantic_rt_assets)
    _enable_step1_vif(config)
    config.training.ema.enabled = False
    trainer = Trainer(build_model(config), config, torch.device("cpu"), tmp_path)

    assert trainer.loss_manager.loss_log_vars is not None
    groups = [
        group
        for group in trainer.optimizer.param_groups
        if group.get("group_name") == "vif_loss_weights"
    ]
    assert len(groups) == 1
    assert groups[0]["weight_decay"] == 0
    assert groups[0]["initial_lr"] == pytest.approx(
        config.training.optimizer.learning_rate
        * config.training.losses.vif.adaptive_weight_lr_multiplier
    )
    before = trainer.loss_manager.loss_log_vars.detach().clone()
    result = trainer.train_step(TaskType.VIF)
    after = trainer.loss_manager.loss_log_vars.detach().clone()
    assert set(result.components) == {"fusion/intensity"}
    assert not torch.equal(after, before)
    torch.testing.assert_close(after[1:], before[1:])

    checkpoint = trainer.save(tmp_path / "step1.pt")
    trainer.provider.close()
    restored = Trainer(
        build_model(config), config, torch.device("cpu"), tmp_path / "restored"
    )
    restored.resume(checkpoint)
    assert restored.loss_manager.loss_log_vars is not None
    torch.testing.assert_close(restored.loss_manager.loss_log_vars.detach(), after)
    restored.provider.close()


def test_step1_trainer_accepts_legacy_model_only_optimizer_checkpoint(
    tmp_path: Path, semantic_rt_assets: tuple[Path, Path, Path]
) -> None:
    legacy_config = _smoke_config(semantic_rt_assets)
    legacy_config.training.ema.enabled = False
    legacy = Trainer(
        build_model(legacy_config),
        legacy_config,
        torch.device("cpu"),
        tmp_path / "legacy",
    )
    legacy.train_step(TaskType.VIF)
    checkpoint = legacy.save(tmp_path / "legacy.pt")
    legacy.provider.close()

    step1_config = _smoke_config(semantic_rt_assets)
    _enable_step1_vif(step1_config)
    step1_config.training.ema.enabled = False
    step1 = Trainer(
        build_model(step1_config),
        step1_config,
        torch.device("cpu"),
        tmp_path / "step1",
    )
    step1.resume(checkpoint)
    assert step1.loss_manager.loss_log_vars is not None
    assert len(step1.scheduler.base_lrs) == len(step1.optimizer.param_groups)
    result = step1.train_step(TaskType.VIF)
    assert set(result.components) == {"fusion/intensity"}
    step1.provider.close()


def test_trainer_performs_real_updates_for_all_tasks(
    tmp_path: Path, semantic_rt_assets: tuple[Path, Path, Path]
) -> None:
    config = _smoke_config(semantic_rt_assets)
    config.training.diagnostics.interval = 2
    trainer = Trainer(build_model(config), config, torch.device("cpu"), tmp_path)
    for index, task in enumerate(TaskType):
        result = trainer.train_step(task)
        assert torch.isfinite(result.total)
        if index % config.training.diagnostics.interval == 0:
            assert "router_grad_norm" in result.diagnostics
            assert "expert_grad_norm/common" in result.diagnostics
            assert "moe_residual_to_input_ratio" in result.diagnostics
        else:
            assert "router_grad_norm" not in result.diagnostics
            assert "expert_grad_norm/common" not in result.diagnostics
            assert not any(
                name.startswith("expert_grad_norm/")
                for name in result.diagnostics
            )
            assert "moe_residual_to_input_ratio" not in result.diagnostics
    assert trainer.state.global_step == 3


def test_train_displays_one_progress_bar_per_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    semantic_rt_assets: tuple[Path, Path, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    import tfs_moe_fusion.trainer as trainer_module

    config = _smoke_config(semantic_rt_assets)
    config.training.epochs = 2
    config.training.steps_per_epoch = 2
    config.training.max_steps = None
    config.training.ema.enabled = False
    config.training.task_sampling.strategy = "alternating"
    config.training.log_every_steps = 2
    trainer = Trainer(torch.nn.Linear(2, 1), config, torch.device("cpu"), tmp_path)
    caplog.set_level("INFO", logger=trainer.logger.name)

    progress_bars = []

    class FakeProgress:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.updates = 0
            self.closed = False
            progress_bars.append(self)

        def update(self, value: int) -> None:
            self.updates += value

        def set_postfix(self, refresh: bool = True, **kwargs) -> None:
            self.postfix_refresh = refresh
            self.postfix = kwargs

        def clear(self) -> None:
            pass

        def refresh(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(trainer_module, "tqdm", FakeProgress)
    monkeypatch.setattr(trainer, "save", lambda path, **kwargs: Path(path))

    def fake_train_step(task: TaskType) -> LossOutput:
        namespace = {
            TaskType.VIF: "fusion/intensity",
            TaskType.MFIF: "focus/selection",
            TaskType.SEG: "semantic/ce",
        }[task]
        components = {
            namespace: torch.tensor(1.0),
            "moe/importance": torch.tensor(0.2),
        }
        trainer.state.phase = "stabilization"
        trainer.state.global_step += 1
        diagnostics = {
            "chroma_cb_error": torch.tensor(0.123456749, dtype=torch.float64),
            "chroma_cr_error": 0.25,
            "unlogged": torch.ones(2, 3),
        }
        return LossOutput(
            total=torch.tensor(1.2), components=components,
            weighted_components=components, diagnostics=diagnostics, skipped={},
        )

    monkeypatch.setattr(trainer, "train_step", fake_train_step)
    trainer.train()

    assert [item.kwargs["desc"] for item in progress_bars] == [
        "Epoch 1/2 [vif_base]",
        "Epoch 2/2 [vif_base]",
    ]
    assert [item.updates for item in progress_bars] == [2, 2]
    assert all(set(item.postfix) == {"task", "loss", "lr"} for item in progress_bars)
    assert all(item.closed for item in progress_bars)
    step_logs = [
        record.getMessage() for record in caplog.records
        if record.getMessage().startswith("step=")
    ]
    assert len(step_logs) == 2
    assert step_logs[0].startswith("step=2 ")
    assert step_logs[1].startswith("step=4 ")
    for message in step_logs:
        assert "moe/importance=0.20000 chroma_cb_error=0.1234567 chroma_cr_error=0.2500000" in message
        assert "unlogged" not in message


def test_seed_everything_can_disable_deterministic_mode() -> None:
    from tfs_moe_fusion.utils import seed_everything

    benchmark = torch.backends.cudnn.benchmark
    deterministic = torch.backends.cudnn.deterministic
    algorithms = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        seed_everything(3407, True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        seed_everything(3407, False)
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.benchmark
    finally:
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic
        torch.use_deterministic_algorithms(algorithms, warn_only=warn_only)
