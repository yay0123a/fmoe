from __future__ import annotations

from pathlib import Path

import torch

from tfs_moe_fusion.color import rgb_to_ycbcr
from tfs_moe_fusion.config import load_config
from tfs_moe_fusion.guidance import SemanticGuideOutput
from tfs_moe_fusion.model import build_model
from tfs_moe_fusion.types import TaskType
from tfs_moe_fusion.utils import make_probe_batch

ROOT = Path(__file__).resolve().parents[1]


def _config(*, semantic: bool = False):
    config = load_config(ROOT / "configs/shared_pool_stage5_y_only_feedback.yaml")
    config.model.backbone.channels = [8, 16, 32, 64]
    config.model.backbone.depths = [1, 1, 1, 1]
    config.model.frequency.fdconv_kernel_num = 4
    config.model.moe.expert_dim = 8
    config.model.moe.router_hidden_channels = 8
    config.model.moe.expert_expansion = 1
    config.model.guidance.focus.hidden_channels = 8
    config.model.guidance.semantic.enabled = semantic
    config.model.guidance.semantic.backend = None
    config.model.guidance.semantic.input_size = 32
    config.training.ema.enabled = False
    return config


class _DifferentiableSemanticBackend(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> SemanticGuideOutput:
        evidence = image.mean(1, keepdim=True)
        logits = torch.cat((evidence, -evidence), dim=1)
        probabilities = torch.softmax(logits, dim=1)
        uncertainty = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
            1, keepdim=True
        )
        return SemanticGuideOutput(
            logits,
            probabilities,
            uncertainty,
            probabilities[:, :1],
        )


@torch.no_grad()
def test_stage5_y_only_outputs_and_mfif_rgb_contract() -> None:
    config = _config()
    model = build_model(config).eval()
    for task in (TaskType.VIF, TaskType.SEG):
        output = model(make_probe_batch(config, task, spatial_size=(31, 37)))
        assert output.refinement_y is not None
        assert output.refinement_y.shape == (1, 1, 31, 37)
        assert output.fused_y is not None and output.coarse_y is not None
        torch.testing.assert_close(
            output.fused_y, output.coarse_y + output.refinement_y, rtol=1e-5, atol=1e-6
        )
        assert output.debug["vif_seg_refinement_active"] is True
        assert output.debug["feedback_expert_weighted_contribution_rms"]

    mfif = model(make_probe_batch(config, TaskType.MFIF, spatial_size=(31, 37)))
    assert mfif.fused_y is mfif.coarse_y is mfif.refinement_y is None


@torch.no_grad()
def test_stage5_y_scale_zero_restores_coarse_and_visible_chroma() -> None:
    config = _config()
    model = build_model(config).eval()
    model.feedback.y_residual_head.scale.zero_()
    batch = make_probe_batch(config, TaskType.VIF, spatial_size=(31, 37))
    output = model(batch)

    torch.testing.assert_close(output.fused, output.coarse, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(output.fused_y, output.coarse_y, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(output.refinement_y, torch.zeros_like(output.refinement_y))
    visible_ycc = rgb_to_ycbcr(batch.visible_source.image)
    torch.testing.assert_close(
        rgb_to_ycbcr(output.fused)[:, 1:], visible_ycc[:, 1:], rtol=2e-5, atol=2e-5
    )


def test_stage5_seg_semantic_guidance_and_vif_availability() -> None:
    config = _config(semantic=True)
    model = build_model(config).eval()
    model.feedback.semantic_backend = _DifferentiableSemanticBackend()
    captured = []
    handle = model.shared_expert_bank.specialists["semantic"].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[1])
    )
    try:
        seg = model(make_probe_batch(config, TaskType.SEG, spatial_size=(31, 37)))
    finally:
        handle.remove()
    feedback = [
        item
        for item in seg.router_diagnostics
        if item.block_id.startswith("feedback.")
    ]
    assert captured
    assert all(value.guidance_feature.shape[1] == config.model.moe.expert_dim for value in captured)
    assert all(item.valid_expert_mask[0, 2] for item in feedback)
    assert model.feedback.feedback_moe["s2"].adapter.guidance_in.in_channels == 16
    assert model.feedback.feedback_moe["s3"].adapter.guidance_in.in_channels == 32

    vif = model(make_probe_batch(config, TaskType.VIF, spatial_size=(31, 37)))
    vif_feedback = [
        item
        for item in vif.router_diagnostics
        if item.block_id.startswith("feedback.")
    ]
    assert all(not item.valid_expert_mask[0, 2] for item in vif_feedback)


def test_stage5_final_losses_reach_feedback_router_and_shared_experts() -> None:
    config = _config(semantic=True)
    model = build_model(config).train()
    model.feedback.semantic_backend = _DifferentiableSemanticBackend()
    output = model(make_probe_batch(config, TaskType.SEG, spatial_size=(31, 37)))
    assert output.fused_y is not None
    assert output.segmentation is not None
    (output.fused_y.mean() + output.segmentation.logits[:, :1].mean()).backward()

    router = model.feedback.feedback_moe["s2"].router.body[-1].weight
    shared_low = model.shared_expert_bank.specialists["low_frequency"].delta_gain
    assert router.grad is not None and torch.count_nonzero(router.grad)
    assert shared_low.grad is not None and torch.count_nonzero(shared_low.grad)


def test_stage5_keeps_real_segformer_frozen() -> None:
    config = load_config(ROOT / "configs/shared_pool_stage5_y_only_feedback.yaml")
    model = build_model(config)
    backend = model.feedback.semantic_backend
    assert backend is not None
    assert all(not parameter.requires_grad for parameter in backend.parameters())
