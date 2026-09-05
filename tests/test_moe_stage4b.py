from __future__ import annotations

from pathlib import Path

import torch

from tfs_moe_fusion.config import load_config
from tfs_moe_fusion.frequency import HaarDWT2D
from tfs_moe_fusion.guidance import FocusGuideOutput
from tfs_moe_fusion.moe import (
    ExpertContext,
    FocusExpert,
    RouterContext,
    SharedExpertBank,
    SharedExpertMoESite,
    Stage4DetailExpert,
    Stage4InfraredSaliencyExpert,
    Stage4LowFrequencyExpert,
    Stage4SemanticExpert,
    TaskEmbedding,
    normalize_relative_delta,
)
from tfs_moe_fusion.types import ExpertType, ModalityType, TaskType

ROOT = Path(__file__).resolve().parents[1]
SPECIALISTS = ("low_frequency", "detail", "semantic", "infrared_saliency", "focus")


def _config():
    config = load_config(ROOT / "configs/shared_pool_stage4b_functional_experts.yaml")
    config.model.moe.expert_dim = 8
    config.model.moe.router_hidden_channels = 8
    config.model.moe.expert_expansion = 1
    return config.model.moe


def _site() -> SharedExpertMoESite:
    config = _config()
    bank = SharedExpertBank(
        config.expert_dim,
        config.expert_expansion,
        tuple(name for name in config.experts if name != "common"),
        config.functional_expert_version,
    )
    return SharedExpertMoESite(8, config, "s2.moe0", TaskEmbedding(8), bank).eval()


def _contexts(
    feature: torch.Tensor,
    task: TaskType,
    source_a: torch.Tensor | None = None,
    source_b: torch.Tensor | None = None,
    focus_a: torch.Tensor | None = None,
    focus_b: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    boundary: torch.Tensor | None = None,
    uncertainty: torch.Tensor | None = None,
    swapped_ir: bool = False,
) -> tuple[RouterContext, ExpertContext]:
    source_a = torch.randn_like(feature) if source_a is None else source_a
    source_b = torch.randn_like(feature) if source_b is None else source_b
    if task is TaskType.MFIF:
        modality_a = modality_b = ModalityType.GENERIC_RGB
    elif swapped_ir:
        modality_a, modality_b = ModalityType.INFRARED_GRAY, ModalityType.VISIBLE_RGB
    else:
        modality_a, modality_b = ModalityType.VISIBLE_RGB, ModalityType.INFRARED_GRAY
    return (
        RouterContext(
            feature,
            task,
            modality_a,
            modality_b,
            low_energy=torch.zeros(feature.shape[0]),
            high_energy=torch.zeros(feature.shape[0]),
            source_a=source_a,
            source_b=source_b,
            focus_a=focus_a,
            focus_b=focus_b,
            focus_confidence=confidence,
            semantic_boundary=boundary,
            semantic_uncertainty=uncertainty,
        ),
        ExpertContext(
            task,
            modality_a,
            modality_b,
            source_a_feature=source_a,
            source_b_feature=source_b,
            focus_a=focus_a,
            focus_b=focus_b,
            focus_confidence=confidence,
            semantic_boundary=boundary,
            semantic_uncertainty=uncertainty,
        ),
    )


def _evidence(
    site: SharedExpertMoESite, feature: torch.Tensor, *contexts
):
    router_context, expert_context = contexts
    evidence = site.site_evidence_builder.build(
        feature, router_context, expert_context, feature.shape[-2:]
    )
    return evidence.expert


def _zero_parameters(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()


def test_stage4b_config_and_relative_normalization_contract() -> None:
    config = load_config(ROOT / "configs/shared_pool_stage4b_functional_experts.yaml")
    assert config.model.moe.functional_expert_version == "stage4b"
    assert config.model.moe.relative_delta_clip == 5.0
    assert ExpertType.FOCUS.value in config.model.moe.experts

    reference = torch.randn(2, 3, 5, 7, requires_grad=True)
    delta = torch.randn_like(reference, requires_grad=True)
    relative = normalize_relative_delta(delta, reference, clip=0.25)
    relative.value.square().mean().backward()
    assert reference.grad is None
    assert delta.grad is not None
    assert relative.value.abs().max() <= 0.25
    assert torch.isfinite(relative.reference_rms).all()


def test_stage4b_availability_and_detached_relative_diagnostics() -> None:
    site = _site()
    feature = torch.randn(2, 8, 13, 17)
    focus = torch.full_like(feature[:, :1], 0.5)
    semantic = torch.ones_like(focus)
    for task, expected in (
        (TaskType.VIF, {"low_frequency", "detail", "infrared_saliency"}),
        (TaskType.MFIF, {"low_frequency", "detail", "focus"}),
        (TaskType.SEG, {"low_frequency", "detail", "infrared_saliency", "semantic"}),
    ):
        context = _contexts(
            feature,
            task,
            focus_a=focus,
            focus_b=focus,
            confidence=focus,
            boundary=semantic,
            uncertainty=torch.zeros_like(semantic),
        )
        output = site(feature, *context)
        valid = {
            name
            for name, enabled in zip(SPECIALISTS, output.router.valid_expert_mask[0])
            if enabled
        }
        assert valid == expected
        assert output.router.valid_expert_mask.ndim == 2
        for name in (
            "low_frequency",
            "detail/contrast/lh",
            "infrared_saliency",
            "infrared_saliency/advantage",
            "focus",
        ):
            prefix = f"evidence/{name}"
            for metric in (
                "reference_rms",
                "raw_delta_rms",
                "delta_to_reference_ratio",
                "normalized_delta_rms",
                "clip_fraction",
            ):
                value = output.diagnostics.auxiliary[f"{prefix}/{metric}"]
                assert not value.requires_grad and torch.isfinite(value)

    missing_focus = site(feature, *_contexts(feature, TaskType.MFIF))
    assert missing_focus.router.valid_expert_mask[0].tolist() == [True, True, False, False, False]
    missing_semantic = site(feature, *_contexts(feature, TaskType.SEG))
    assert missing_semantic.router.valid_expert_mask[0].tolist() == [True, True, False, True, False]


def test_low_and_detail_are_band_limited_and_respond_to_synthetic_patterns() -> None:
    site = _site()
    feature = torch.zeros(1, 8, 16, 16)
    low_source = torch.ones_like(feature)
    low_evidence = _evidence(site, feature, *_contexts(feature, TaskType.MFIF, low_source, low_source))
    low = site.shared_expert_bank.specialists["low_frequency"]
    assert isinstance(low, Stage4LowFrequencyExpert)
    _zero_parameters(low)
    with torch.no_grad():
        low.delta_gain.fill_(1)
    low_output = low(feature, low_evidence.low).residual
    low_bands = HaarDWT2D()(low_output)
    assert low_output.square().mean() > 0
    assert sum(item.abs().max() for item in (low_bands.lh, low_bands.hl, low_bands.hh)) < 1e-5

    checker = torch.tensor(
        [[(-1.0) ** (row + column) for column in range(16)] for row in range(16)]
    )[None, None].expand_as(feature)
    detail_evidence = _evidence(
        site, feature, *_contexts(feature, TaskType.MFIF, checker, checker)
    )
    detail = site.shared_expert_bank.specialists["detail"]
    assert isinstance(detail, Stage4DetailExpert)
    _zero_parameters(detail)
    with torch.no_grad():
        detail.delta_gain.fill_(1)
    detail_output = detail(feature, detail_evidence.detail).residual
    detail_bands = HaarDWT2D()(detail_output)
    assert detail_output.square().mean() > 0
    assert detail_bands.ll.abs().max() < 1e-5


def test_ir_focus_and_semantic_experts_follow_typed_evidence() -> None:
    site = _site()
    feature = torch.ones(1, 8, 16, 16)
    visible = torch.zeros_like(feature)
    infrared = torch.zeros_like(feature)
    infrared[..., 6:10, 6:10] = 3
    ir_evidence = _evidence(
        site, feature, *_contexts(feature, TaskType.VIF, visible, infrared)
    )
    infrared_expert = site.shared_expert_bank.specialists["infrared_saliency"]
    assert isinstance(infrared_expert, Stage4InfraredSaliencyExpert)
    _zero_parameters(infrared_expert)
    with torch.no_grad():
        infrared_expert.delta_gain.fill_(1)
    ir_output = infrared_expert(feature, ir_evidence.infrared).residual.abs().mean(1, keepdim=True)
    assert ir_output[..., 6:10, 6:10].mean() > ir_output[..., :4, :4].mean()

    selection_a = torch.zeros_like(feature[:, :1])
    selection_a[..., :, :8] = 1
    selection_b = 1 - selection_a
    focus_guide = FocusGuideOutput(
        selection_a,
        selection_b,
        torch.cat((selection_a, selection_b), dim=1),
        selection_a,
        selection_b,
        torch.ones_like(selection_a),
    )
    source_a = torch.ones_like(feature)
    source_b = torch.full_like(feature, 3)
    focus_evidence = _evidence(
        site,
        feature,
        *_contexts(
            feature,
            TaskType.MFIF,
            source_a,
            source_b,
            focus_guide.selection_a,
            focus_guide.selection_b,
            focus_guide.confidence,
        ),
    )
    focus = site.shared_expert_bank.specialists["focus"]
    assert isinstance(focus, FocusExpert)
    _zero_parameters(focus)
    with torch.no_grad():
        focus.delta_gain.fill_(1)
    focus_output = focus(feature, focus_evidence.focus).residual.mean(1, keepdim=True)
    assert focus_output[..., :, 8:].mean() > focus_output[..., :, :8].mean()

    semantic = site.shared_expert_bank.specialists["semantic"]
    assert isinstance(semantic, Stage4SemanticExpert)
    _zero_parameters(semantic)
    with torch.no_grad():
        semantic.need_gain.fill_(1)
    need = torch.zeros_like(selection_a)
    need[..., :, :8] = 1
    semantic_evidence = _evidence(
        site,
        feature,
        *_contexts(
            feature,
            TaskType.SEG,
            visible,
            infrared,
            boundary=need,
            uncertainty=torch.zeros_like(need),
        ),
    ).with_semantic_guidance(torch.zeros_like(feature))
    gate = semantic(feature, semantic_evidence.semantic).diagnostics["gate"]
    assert gate[..., :, :8].mean() > gate[..., :, 8:].mean()


def test_mfif_source_swap_preserves_low_detail_and_focus_outputs() -> None:
    torch.manual_seed(3407)
    site = _site()
    feature, source_a, source_b = (torch.randn(1, 8, 14, 18) for _ in range(3))
    selection_a = torch.rand(1, 1, 14, 18)
    selection_b = 1 - selection_a
    confidence = (selection_a - selection_b).abs()
    first = _evidence(
        site,
        feature,
        *_contexts(
            feature,
            TaskType.MFIF,
            source_a,
            source_b,
            selection_a,
            selection_b,
            confidence,
        ),
    )
    second = _evidence(
        site,
        feature,
        *_contexts(
            feature,
            TaskType.MFIF,
            source_b,
            source_a,
            selection_b,
            selection_a,
            confidence,
        ),
    )
    for name in ("low_frequency", "detail", "focus"):
        expert = site.shared_expert_bank.specialists[name]
        expert_type = ExpertType.parse(name)
        output_a = expert(feature, first.for_expert(expert_type)).residual
        output_b = expert(feature, second.for_expert(expert_type)).residual
        torch.testing.assert_close(output_a, output_b, rtol=1e-5, atol=1e-6)
