from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tfs_moe_fusion.config import load_config
from tfs_moe_fusion.frequency import local_spectral_evidence
from tfs_moe_fusion.moe import (
    ExpertContext,
    RouterContext,
    SharedExpertBank,
    SharedExpertMoESite,
    SiteSpatialRouter,
    TaskEmbedding,
)
from tfs_moe_fusion.types import ExpertType, ModalityType, TaskType

ROOT = Path(__file__).resolve().parents[1]
SPECIALISTS = ("low_frequency", "detail", "semantic", "infrared_saliency")


def _config():
    config = load_config(ROOT / "configs/shared_pool_stage3_spatial_soft.yaml")
    config.model.moe.expert_dim = 16
    config.model.moe.router_hidden_channels = 8
    config.model.moe.expert_expansion = 1
    return config.model.moe


def _site(stage: str = "s2") -> SharedExpertMoESite:
    config = _config()
    return SharedExpertMoESite(
        8,
        config,
        f"{stage}.moe0",
        TaskEmbedding(8),
        SharedExpertBank(config.expert_dim, config.expert_expansion),
    )


def _contexts(
    feature: torch.Tensor, has_ir: bool = True
) -> tuple[RouterContext, ExpertContext]:
    modality_a = ModalityType.VISIBLE_RGB if has_ir else ModalityType.GENERIC_RGB
    modality_b = ModalityType.INFRARED_GRAY if has_ir else ModalityType.GENERIC_RGB
    task = TaskType.VIF if has_ir else TaskType.MFIF
    source_a, source_b = torch.randn_like(feature), torch.randn_like(feature)
    return (
        RouterContext(
            feature,
            task,
            modality_a,
            modality_b,
            low_energy=torch.rand(feature.shape[0]),
            high_energy=torch.rand(feature.shape[0]),
            source_a=source_a,
            source_b=source_b,
        ),
        ExpertContext(
            task,
            modality_a,
            modality_b,
            source_a_feature=source_a,
            source_b_feature=source_b,
        ),
    )


def test_stage3_config_enables_spatial_soft_patch_grid() -> None:
    config = load_config(ROOT / "configs/shared_pool_stage3_spatial_soft.yaml")
    assert config.model.moe.routing_mode == "spatial_soft"
    assert config.model.moe.patch_size == {"s2": 4, "s3": 2, "s4": 1}
    assert config.model.moe.shared_pool_enabled is True


def test_spatial_router_shape_probability_contract_and_interpolated_mixture() -> None:
    site = _site("s2").eval()
    feature = torch.randn(2, 8, 13, 17)
    output = site(feature, *_contexts(feature))
    assert isinstance(site.router, SiteSpatialRouter)
    assert output.router.logits.shape == output.router.probabilities.shape == (2, 4, 4, 5)
    assert output.router.valid_expert_mask.shape == (2, 4)
    assert output.router.topk_indices is output.router.topk_weights is None
    torch.testing.assert_close(
        output.router.probabilities.sum(1), torch.ones(2, 4, 5)
    )
    weights = output.diagnostics.auxiliary["specialist_mixture_weights"]
    assert weights.shape == (2, 4, 13, 17)
    torch.testing.assert_close(weights.sum(1), torch.ones(2, 13, 17))


@pytest.mark.parametrize("height,width", [(5, 7), (13, 17), (14, 9)])
def test_spatial_routing_and_local_frequency_evidence_are_shape_safe(
    height: int, width: int
) -> None:
    feature = torch.randn(1, 8, height, width)
    low, high = local_spectral_evidence(feature, (3, 4))
    assert low.shape == high.shape == (1, 1, 3, 4)
    assert torch.isfinite(low).all() and torch.isfinite(high).all()
    output = _site("s2")(feature, *_contexts(feature))
    assert output.feature.shape == feature.shape


def test_invalid_infrared_expert_has_zero_probability_everywhere() -> None:
    site = _site().eval()
    feature = torch.randn(2, 8, 13, 17)
    output = site(feature, *_contexts(feature, has_ir=False))
    infrared = SPECIALISTS.index(ExpertType.INFRARED_SALIENCY.value)
    assert torch.count_nonzero(output.router.probabilities[:, infrared]) == 0


def test_zero_initialized_logits_are_uniform_and_spatial_diagnostics_are_finite() -> None:
    site = _site().eval()
    feature = torch.randn(1, 8, 13, 17)
    output = site(feature, *_contexts(feature))
    expected = torch.full_like(output.router.probabilities, 1 / len(SPECIALISTS))
    torch.testing.assert_close(output.router.probabilities, expected)
    auxiliary = output.diagnostics.auxiliary
    assert torch.isfinite(auxiliary["router/spatial_variance"])
    assert torch.isfinite(auxiliary["router/top1_margin"])
    assert auxiliary["router_grid_size"] == (4, 5)


def test_spatial_router_receives_gradient_from_final_loss() -> None:
    torch.manual_seed(229)
    site = _site().train()
    feature = torch.randn(2, 8, 13, 17, requires_grad=True)
    site(feature, *_contexts(feature)).feature.square().mean().backward()
    gradient_sum = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in site.router.parameters()
        if parameter.grad is not None
    )
    assert gradient_sum > 0


def test_spatial_overrides_are_tokenwise_and_clear_to_learned() -> None:
    site = _site().eval()
    feature = torch.randn(1, 8, 13, 17)
    context = _contexts(feature)
    site.set_routing_override("uniform")
    uniform = site(feature, *context).router.probabilities
    torch.testing.assert_close(uniform, torch.full_like(uniform, 1 / len(SPECIALISTS)))
    site.set_routing_override("shuffled", seed=3407)
    first = site(feature, *context).router.probabilities
    second = site(feature, *context).router.probabilities
    torch.testing.assert_close(first, second)
    site.set_routing_override("single:detail")
    single = site(feature, *context).router.probabilities
    assert torch.count_nonzero(single[:, SPECIALISTS.index("detail")]) == single[:, 0].numel()
    site.clear_routing_override()
    assert site.routing_override == "learned"


def test_global_soft_v2_remains_available() -> None:
    config = load_config(ROOT / "configs/shared_pool_stage2_global_soft.yaml").model.moe
    config.expert_dim = 16
    config.router_hidden_channels = 8
    config.expert_expansion = 1
    site = SharedExpertMoESite(
        8,
        config,
        "s2.moe0",
        TaskEmbedding(8),
        SharedExpertBank(config.expert_dim, config.expert_expansion),
    )
    feature = torch.randn(1, 8, 13, 17)
    output = site(feature, *_contexts(feature))
    assert output.router.probabilities.shape == (1, len(SPECIALISTS))
    assert output.router.topk_indices is not None
