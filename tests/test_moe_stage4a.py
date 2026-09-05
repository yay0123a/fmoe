"""Stage 4A must remain numerically equivalent to the Stage 3 spatial path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MethodType

import pytest
import torch
from torch import nn
from torch.nn import functional

from tfs_moe_fusion.config import load_config
from tfs_moe_fusion.frequency import WaveletBands, local_spectral_evidence
from tfs_moe_fusion.model import build_model
from tfs_moe_fusion.moe import (
    DetailExpert,
    ExpertContext,
    FunctionalMoEBlock,
    InfraredSaliencyExpert,
    LowFrequencyExpert,
    RouterContext,
    SemanticExpert,
    SharedExpertBank,
    SharedExpertMoESite,
    TaskEmbedding,
    iter_moe_blocks,
)
from tfs_moe_fusion.types import ModalityType, TaskType
from tfs_moe_fusion.utils import make_probe_batch

ROOT = Path(__file__).resolve().parents[1]
SPECIALISTS = ("low_frequency", "detail", "semantic", "infrared_saliency")


def _site() -> SharedExpertMoESite:
    config = load_config(ROOT / "configs/shared_pool_stage3_spatial_soft.yaml").model.moe
    config.expert_dim = 16
    config.router_hidden_channels = 8
    config.expert_expansion = 1
    return SharedExpertMoESite(
        8,
        config,
        "s2.moe0",
        TaskEmbedding(8),
        SharedExpertBank(config.expert_dim, config.expert_expansion),
    ).eval()


def _contexts(feature: torch.Tensor) -> tuple[RouterContext, ExpertContext]:
    source_a, source_b = torch.randn_like(feature), torch.randn_like(feature)
    boundary, uncertainty = torch.rand_like(feature[:, :1]), torch.rand_like(feature[:, :1])
    return (
        RouterContext(
            feature,
            TaskType.VIF,
            ModalityType.VISIBLE_RGB,
            ModalityType.INFRARED_GRAY,
            low_energy=torch.zeros(feature.shape[0]),
            high_energy=torch.zeros(feature.shape[0]),
            source_a=source_a,
            source_b=source_b,
            focus_confidence=torch.rand_like(feature[:, :1]),
            semantic_boundary=boundary,
            semantic_uncertainty=uncertainty,
        ),
        ExpertContext(
            TaskType.VIF,
            ModalityType.VISIBLE_RGB,
            ModalityType.INFRARED_GRAY,
            source_a_feature=source_a,
            source_b_feature=source_b,
            semantic_boundary=boundary,
            semantic_uncertainty=uncertainty,
        ),
    )


def _stage3_router(site: SharedExpertMoESite, context: RouterContext) -> torch.Tensor:
    """The pre-4A SiteSpatialRouter formula, kept local to this equivalence test."""
    router = site.router
    feature = context.feature
    batch, _, height, width = feature.shape
    grid = (
        (height + router.patch_size - 1) // router.patch_size,
        (width + router.patch_size - 1) // router.patch_size,
    )
    with torch.autocast(device_type=feature.device.type, enabled=False):
        value = feature.float()
        pooled = functional.adaptive_avg_pool2d(value, grid)
        difference = (context.source_a.float() - context.source_b.float()).abs()
        difference = functional.adaptive_avg_pool2d(difference, grid)
        low, high = local_spectral_evidence(value, grid)

        def evidence_map(item: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
            available = value.new_full((batch, 1, *grid), float(item is not None))
            if item is None:
                return value.new_zeros(batch, 1, *grid), available
            return (
                functional.interpolate(
                    item.float(), size=grid, mode="bilinear", align_corners=False
                ).mean(1, keepdim=True),
                available,
            )

        focus, focus_available = evidence_map(context.focus_confidence)
        boundary, boundary_available = evidence_map(context.semantic_boundary)
        uncertainty, _ = evidence_map(context.semantic_uncertainty)
        task = router.task_projection(
            router.task_embedding(context.task, batch, feature.device).float()
        )[:, :, None, None].expand(-1, -1, *grid)
        pair = list(ModalityType).index(context.modality_a) * len(ModalityType)
        pair += list(ModalityType).index(context.modality_b)
        modality = router.modality_projection(
            router.modality_embedding.weight[pair].float()[None].expand(batch, -1)
        )[:, :, None, None].expand(-1, -1, *grid)
        logits = router.body(
            torch.cat(
                (
                    router.feature_projection(pooled),
                    router.difference_projection(difference),
                    task,
                    modality,
                    low,
                    high,
                    focus,
                    focus_available,
                    boundary,
                    boundary_available,
                    uncertainty,
                ),
                dim=1,
            )
        )
    return torch.softmax(logits / router.temperature, dim=1)


def _stage3_residual(expert: nn.Module, feature: torch.Tensor, context: ExpertContext) -> torch.Tensor:
    """The Stage 3 specialist formulas before typed evidence was introduced."""
    if isinstance(expert, LowFrequencyExpert):
        bands = expert.dwt(feature)
        low = expert.low_refine(expert.afno(expert.low_projection(bands.ll))) - bands.ll
        zeros = torch.zeros_like(low)
        return expert.idwt(WaveletBands(low, zeros, zeros, zeros, bands.original_size))
    if isinstance(expert, DetailExpert):
        bands = expert.dwt(feature)
        details = [expert.detail_refine(value) - value for value in (bands.lh, bands.hl, bands.hh)]
        return expert.idwt(
            WaveletBands(
                torch.zeros_like(bands.ll),
                details[0],
                details[1],
                details[2],
                bands.original_size,
            )
        )
    if isinstance(expert, SemanticExpert):
        residual = expert.semantic_refine(feature)
        gate_logits = expert.content_gate(feature)
        if context.semantic_boundary is not None or context.semantic_uncertainty is not None:
            reference = (
                context.semantic_boundary
                if context.semantic_boundary is not None
                else context.semantic_uncertainty
            )
            boundary = (
                context.semantic_boundary
                if context.semantic_boundary is not None
                else torch.zeros_like(reference)
            )
            uncertainty = (
                context.semantic_uncertainty
                if context.semantic_uncertainty is not None
                else torch.zeros_like(reference)
            )
            maps = functional.interpolate(
                torch.cat((boundary, 1.0 - uncertainty), dim=1).float(),
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(feature.dtype)
            gate_logits = gate_logits + expert.external_conditioner(maps)
        gate = torch.sigmoid(gate_logits)
        return residual * gate
    assert isinstance(expert, InfraredSaliencyExpert)
    infrared, other = context.source_b_feature, context.source_a_feature
    difference = (infrared - other).abs()
    saliency = expert.saliency(torch.cat((infrared, difference, feature), dim=1))
    return expert.refine(saliency * (infrared - feature))


def _stage3_canonical_reference(
    site: FunctionalMoEBlock,
    canonical: torch.Tensor,
    canonical_router: RouterContext,
    canonical_expert: ExpertContext,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    common = site.shared_expert_bank.common(canonical).residual
    shared = canonical + site.common_scale * common
    probabilities = _stage3_router(site, replace(canonical_router, feature=shared))
    weights = functional.interpolate(
        probabilities, size=shared.shape[-2:], mode="bilinear", align_corners=False
    )
    weights = weights / weights.sum(1, keepdim=True)
    residuals = []
    specialist = torch.zeros_like(shared)
    for index, name in enumerate(SPECIALISTS):
        residual = _stage3_residual(
            site.shared_expert_bank.specialists[name], shared, canonical_expert
        )
        residuals.append(residual)
        specialist = specialist + residual * weights[:, index : index + 1]
    return shared + site.specialist_scale * specialist, probabilities, torch.stack(
        residuals, dim=1
    )


@torch.no_grad()
def _stage3_reference(
    site: SharedExpertMoESite,
    feature: torch.Tensor,
    router_context: RouterContext,
    expert_context: ExpertContext,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    canonical = site.adapter.feature_in(feature)
    source_a = site.adapter.project_source(expert_context.source_a_feature)
    source_b = site.adapter.project_source(expert_context.source_b_feature)
    canonical_router = replace(
        router_context, feature=canonical, source_a=source_a, source_b=source_b
    )
    canonical_expert = replace(
        expert_context, source_a_feature=source_a, source_b_feature=source_b
    )
    output, probabilities, residuals = _stage3_canonical_reference(
        site, canonical, canonical_router, canonical_expert
    )
    return feature + site.adapter.delta_out(output - canonical), probabilities, residuals


def _assert_equivalent(
    request: pytest.FixtureRequest, name: str, reference: torch.Tensor, actual: torch.Tensor
) -> None:
    difference = (reference - actual).abs()
    maximum = float(difference.detach().max())
    mean = float(difference.detach().mean())
    request.node.user_properties.extend(
        (
            (f"{name}_max_abs_diff", maximum),
            (f"{name}_mean_abs_diff", mean),
        )
    )
    assert maximum <= 1e-5
    torch.testing.assert_close(reference, actual, rtol=1e-5, atol=1e-6)


def test_stage4a_matches_stage3_reference_cpu_fp32(request: pytest.FixtureRequest) -> None:
    torch.manual_seed(3407)
    site = _site()
    with torch.no_grad():
        site.router.body[-1].weight.normal_(std=0.03)
        site.router.body[-1].bias.normal_(std=0.01)
    feature = torch.randn(2, 8, 13, 17)
    router_context, expert_context = _contexts(feature)

    reference_block, reference_probabilities, reference_residuals = _stage3_reference(
        site, feature, router_context, expert_context
    )
    actual = site(
        feature, router_context, expert_context, return_expert_outputs=True
    )
    actual_residuals = torch.stack(
        [actual.expert_outputs[name].residual for name in SPECIALISTS], dim=1
    )
    final_projection = nn.Conv2d(8, 3, 1)
    _assert_equivalent(
        request, "block_output", reference_block, actual.feature
    )
    _assert_equivalent(
        request,
        "router_probabilities",
        reference_probabilities,
        actual.router.probabilities,
    )
    _assert_equivalent(
        request, "expert_residual", reference_residuals, actual_residuals)
    _assert_equivalent(
        request,
        "final_fused_output",
        final_projection(reference_block),
        final_projection(actual.feature),
    )


def test_stage4a_final_fused_output_matches_stage3_reference(
    request: pytest.FixtureRequest,
) -> None:
    torch.manual_seed(3407)
    config = load_config(ROOT / "configs/shared_pool_stage3_spatial_soft.yaml")
    config.model.guidance.semantic.enabled = False
    model = build_model(config).eval()
    batch = make_probe_batch(config, TaskType.VIF)
    with torch.no_grad():
        for block in iter_moe_blocks(model):
            block.router.body[-1].weight.normal_(std=0.03)
            block.router.body[-1].bias.normal_(std=0.01)
        actual = model(batch).fused

    def stage3_forward(
        block: FunctionalMoEBlock,
        tensor: torch.Tensor,
        router_context: RouterContext,
        expert_context: ExpertContext,
        sparse_execution: bool | None,
        return_expert_outputs: bool,
    ):
        current = FunctionalMoEBlock._forward_spatial_v2(
            block,
            tensor,
            router_context,
            expert_context,
            sparse_execution,
            return_expert_outputs,
        )
        reference, _, _ = _stage3_canonical_reference(
            block, tensor, router_context, expert_context
        )
        return replace(current, feature=reference, residual=reference - tensor)

    for block in iter_moe_blocks(model):
        object.__setattr__(block, "_forward_spatial_v2", MethodType(stage3_forward, block))
    with torch.no_grad():
        reference = model(batch).fused
    _assert_equivalent(request, "model_final_fused_output", reference, actual)


def test_stage4a_reuses_one_fused_dwt_and_keeps_availability_site_level() -> None:
    torch.manual_seed(3407)
    site = _site()
    feature = torch.randn(2, 8, 13, 17)
    router_context, expert_context = _contexts(feature)
    low_dwt = site.shared_expert_bank.specialists["low_frequency"].dwt
    detail_dwt = site.shared_expert_bank.specialists["detail"].dwt
    calls = {"low": 0, "detail": 0}
    low_hook = low_dwt.register_forward_hook(lambda *_: calls.__setitem__("low", calls["low"] + 1))
    detail_hook = detail_dwt.register_forward_hook(
        lambda *_: calls.__setitem__("detail", calls["detail"] + 1)
    )
    try:
        output = site(feature, router_context, expert_context)
    finally:
        low_hook.remove()
        detail_hook.remove()
    assert calls == {"low": 1, "detail": 0}
    assert output.router.valid_expert_mask.shape == (2, len(SPECIALISTS))
    assert isinstance(site.adapter.guidance_in, nn.Identity)
    assert not any("site_evidence_builder" in key for key in site.state_dict())
    restored = _site()
    restored.load_state_dict(site.state_dict(), strict=True)
