"""TFS-MoE-Fusion consolidated implementation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

from torch import Tensor

from tfs_moe_fusion.frequency import local_spectral_evidence_from_bands
from tfs_moe_fusion.types import ModalityType, SpectralStatistics, TaskType


@dataclass(slots=True)
class RouterContext:
    feature: Tensor
    task: TaskType
    modality_a: ModalityType
    modality_b: ModalityType
    frequency_stats: SpectralStatistics | None = None
    source_a: Tensor | None = None
    source_b: Tensor | None = None
    focus_a: Tensor | None = None
    focus_b: Tensor | None = None
    focus_confidence: Tensor | None = None
    semantic_boundary: Tensor | None = None
    semantic_uncertainty: Tensor | None = None
    stage_name: str = "unknown"
    aux: dict[str, Any] = field(default_factory=dict)
    low_energy: Tensor | None = None
    high_energy: Tensor | None = None
    focus_reliability: Tensor | None = None

    def __post_init__(self) -> None:
        if self.frequency_stats is None:
            if self.low_energy is None or self.high_energy is None:
                raise ValueError(
                    "RouterContext requires frequency_stats or low/high energy"
                )
            self.frequency_stats = SpectralStatistics(
                self.stage_name, self.low_energy, self.high_energy
            )
        if self.focus_reliability is not None and self.focus_a is None:
            self.focus_a = self.focus_reliability[:, :1]
            self.focus_b = self.focus_reliability[:, 1:2]
            if self.focus_confidence is None:
                self.focus_confidence = (self.focus_a - self.focus_b).abs()


@dataclass(slots=True)
class RouterOutput:
    logits: Tensor
    probabilities: Tensor
    topk_indices: Tensor | None
    topk_weights: Tensor | None
    valid_expert_mask: Tensor
    spatial_gates: Tensor | None = None
    branch_weights: Tensor | None = None
    entropy: Tensor | None = None
    importance: Tensor | None = None
    hard_load: Tensor | None = None
    auxiliary: dict[str, Any] = field(default_factory=dict)


import torch

from tfs_moe_fusion.types import RouterDiagnostics


def summarize_expert_usage(values: tuple[RouterDiagnostics, ...]) -> torch.Tensor:
    if not values:
        return torch.empty(0)
    loads = [value.hard_load for value in values if value.hard_load is not None]
    if not loads:
        loads = [
            value.probabilities.detach().float().mean(
                tuple(index for index in range(value.probabilities.ndim) if index != 1)
            )
            for value in values
        ]
    result = torch.stack(loads).mean(0)
    return result / result.sum().clamp_min(1e-8)


from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from torch import nn

from tfs_moe_fusion.types import ContractError, ExpertType


@dataclass(slots=True)
class ExpertContext:
    task: TaskType
    modality_a: ModalityType
    modality_b: ModalityType
    source_a_feature: Tensor | None = None
    source_b_feature: Tensor | None = None
    focus_reliability: Tensor | None = None
    focus_a: Tensor | None = None
    focus_b: Tensor | None = None
    focus_confidence: Tensor | None = None
    semantic_uncertainty: Tensor | None = None
    semantic_boundary: Tensor | None = None
    stage_name: str = "unknown"
    aux: dict[str, Any] = field(default_factory=dict)
    stage_guidance_feature: Tensor | None = None

    def index_select(self, indices: Tensor) -> ExpertContext:
        def select(value: Tensor | None) -> Tensor | None:
            return value.index_select(0, indices) if value is not None else None

        selected_aux = {
            key: (
                value.index_select(0, indices)
                if isinstance(value, Tensor)
                and value.ndim > 0
                and value.shape[0] >= int(indices.max().item()) + 1
                else value
            )
            for key, value in self.aux.items()
        }
        return ExpertContext(
            self.task,
            self.modality_a,
            self.modality_b,
            select(self.source_a_feature),
            select(self.source_b_feature),
            select(self.focus_reliability),
            select(self.focus_a),
            select(self.focus_b),
            select(self.focus_confidence),
            select(self.semantic_uncertainty),
            select(self.semantic_boundary),
            self.stage_name,
            selected_aux,
            select(self.stage_guidance_feature),
        )


@dataclass(slots=True)
class ExpertOutput:
    residual: Tensor
    expert: ExpertType
    valid_samples: Tensor
    spatial_prior: Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.expert = ExpertType.parse(self.expert)
        if self.residual.ndim != 4:
            raise ContractError("Expert residual must be [B,C,H,W]")
        if self.valid_samples.shape != (self.residual.shape[0],):
            raise ContractError("Expert valid_samples must be [B]")
        if self.valid_samples.dtype is not torch.bool:
            raise ContractError("Expert valid_samples must be boolean")
        if not torch.isfinite(self.residual).all():
            raise ContractError("Expert residual contains NaN or infinity")


class FunctionalExpert(nn.Module, ABC):
    expert_type: ExpertType

    def __call__(self, tensor: Tensor, evidence: ExpertEvidence | ExpertContext = None, *args: Any, **kwargs: Any) -> ExpertOutput:  # type: ignore[override]
        """Keep the former direct ExpertContext call surface as a thin adapter."""
        if isinstance(evidence, ExpertContext):
            dwt = getattr(self, "dwt", HaarDWT2D().to(tensor))
            builder = SiteEvidenceBuilder(
                dwt, ExpertAvailabilityPolicy((self.expert_type.value,))
            )
            evidence = builder.build(
                tensor,
                RouterContext(
                    tensor,
                    evidence.task,
                    evidence.modality_a,
                    evidence.modality_b,
                    low_energy=tensor.new_zeros(tensor.shape[0]),
                    high_energy=tensor.new_zeros(tensor.shape[0]),
                    source_a=evidence.source_a_feature,
                    source_b=evidence.source_b_feature,
                    focus_confidence=evidence.focus_confidence,
                    semantic_boundary=evidence.semantic_boundary,
                    semantic_uncertainty=evidence.semantic_uncertainty,
                ),
                evidence,
                tensor.shape[-2:],
            ).expert
        return super().__call__(tensor, evidence, *args, **kwargs)

    @abstractmethod
    def forward(self, tensor: Tensor, evidence: ExpertEvidence) -> ExpertOutput:
        raise NotImplementedError


import math

from torch import nn
from torch.nn import functional

from tfs_moe_fusion.frequency import (
    AFNO2D,
    FDConv2d,
    HaarDWT2D,
    HaarIDWT2D,
    WaveletBands,
)


@dataclass(frozen=True, slots=True)
class ExpertAvailability:
    """Task/site availability; [B,E] is broadcast over every routing position."""

    valid_mask: Tensor

    def __post_init__(self) -> None:
        if self.valid_mask.ndim != 2 or self.valid_mask.dtype is not torch.bool:
            raise ContractError("Expert availability must be a boolean [B,E] tensor")

    def index_select(self, indices: Tensor) -> ExpertAvailability:
        return ExpertAvailability(self.valid_mask.index_select(0, indices))


class ExpertAvailabilityPolicy:
    """Resolve task/site-level expert availability before expert execution."""

    def __init__(self, expert_names: tuple[str, ...], version: str = "stage3") -> None:
        self.expert_names = expert_names
        self.version = version
        self.infrared_index = (
            expert_names.index(ExpertType.INFRARED_SALIENCY.value)
            if ExpertType.INFRARED_SALIENCY.value in expert_names
            else None
        )

    def resolve(
        self,
        batch: int,
        device: torch.device,
        task: TaskType,
        modality_a: ModalityType,
        modality_b: ModalityType,
        focus_available: bool = False,
        semantic_available: bool = False,
    ) -> ExpertAvailability:
        valid = torch.ones(
            batch, len(self.expert_names), dtype=torch.bool, device=device
        )
        if self.version == "stage4b":
            valid.zero_()
            for name in (ExpertType.LOW_FREQUENCY.value, ExpertType.DETAIL.value):
                valid[:, self.expert_names.index(name)] = True
            if task in {TaskType.VIF, TaskType.SEG}:
                infrared = ExpertType.INFRARED_SALIENCY.value
                if infrared in self.expert_names and (
                    modality_a.is_infrared or modality_b.is_infrared
                ):
                    valid[:, self.expert_names.index(infrared)] = True
            if task is TaskType.MFIF and focus_available:
                focus = ExpertType.FOCUS.value
                if focus in self.expert_names:
                    valid[:, self.expert_names.index(focus)] = True
            if task is TaskType.SEG and semantic_available:
                semantic = ExpertType.SEMANTIC.value
                if semantic in self.expert_names:
                    valid[:, self.expert_names.index(semantic)] = True
            return ExpertAvailability(valid)
        if self.infrared_index is not None and not (
            modality_a.is_infrared or modality_b.is_infrared
        ):
            valid[:, self.infrared_index] = False
        return ExpertAvailability(valid)


@dataclass(frozen=True, slots=True)
class ExpertEvidence:
    """Fully resolved tensors consumed by specialists without optional inputs."""

    fused_bands: WaveletBands
    semantic_maps: Tensor
    semantic_condition_scale: Tensor
    semantic_boundary_available: bool
    infrared_feature: Tensor
    other_feature: Tensor
    infrared_samples_valid: Tensor

    def index_select(self, indices: Tensor) -> ExpertEvidence:
        bands = self.fused_bands
        return ExpertEvidence(
            WaveletBands(
                bands.ll.index_select(0, indices),
                bands.lh.index_select(0, indices),
                bands.hl.index_select(0, indices),
                bands.hh.index_select(0, indices),
                bands.meta,
            ),
            self.semantic_maps.index_select(0, indices),
            self.semantic_condition_scale.index_select(0, indices),
            self.semantic_boundary_available,
            self.infrared_feature.index_select(0, indices),
            self.other_feature.index_select(0, indices),
            self.infrared_samples_valid.index_select(0, indices),
        )


@dataclass(frozen=True, slots=True)
class RelativeDelta:
    value: Tensor
    reference_rms: Tensor
    raw_delta_rms: Tensor
    normalized_delta_rms: Tensor
    delta_to_reference_ratio: Tensor
    clip_fraction: Tensor

    def index_select(self, indices: Tensor) -> RelativeDelta:
        return RelativeDelta(
            self.value.index_select(0, indices),
            self.reference_rms.index_select(0, indices),
            self.raw_delta_rms.index_select(0, indices),
            self.normalized_delta_rms.index_select(0, indices),
            self.delta_to_reference_ratio.index_select(0, indices),
            self.clip_fraction.index_select(0, indices),
        )


def normalize_relative_delta(
    delta: Tensor, reference: Tensor, clip: float = 5.0
) -> RelativeDelta:
    """Normalize a delta by detached per-sample reference RMS, never its own RMS."""
    if delta.shape != reference.shape:
        raise ValueError("relative delta and reference must have equal shapes")
    if clip <= 0:
        raise ValueError("relative delta clip must be positive")
    dims = tuple(range(1, delta.ndim))
    reference_rms = reference.detach().float().square().mean(dims, keepdim=True).sqrt()
    raw_delta_rms = delta.detach().float().square().mean(dims, keepdim=True).sqrt()
    eps = torch.finfo(torch.float32).eps
    scale = reference_rms.clamp_min(eps).to(delta)
    unbounded = delta / scale
    normalized = unbounded.clamp(-clip, clip)
    normalized_rms = normalized.detach().float().square().mean(dims, keepdim=True).sqrt()
    return RelativeDelta(
        normalized,
        reference_rms,
        raw_delta_rms,
        normalized_rms,
        raw_delta_rms / reference_rms.clamp_min(eps),
        (unbounded.detach().abs() > clip).float().mean(dims, keepdim=True),
    )


@dataclass(frozen=True, slots=True)
class LowEvidence:
    delta: RelativeDelta
    contrast: RelativeDelta

    def index_select(self, indices: Tensor) -> LowEvidence:
        return LowEvidence(
            self.delta.index_select(indices), self.contrast.index_select(indices)
        )


@dataclass(frozen=True, slots=True)
class DetailEvidence:
    lh: RelativeDelta
    hl: RelativeDelta
    hh: RelativeDelta
    contrast_lh: RelativeDelta
    contrast_hl: RelativeDelta
    contrast_hh: RelativeDelta

    def index_select(self, indices: Tensor) -> DetailEvidence:
        return DetailEvidence(
            self.lh.index_select(indices),
            self.hl.index_select(indices),
            self.hh.index_select(indices),
            self.contrast_lh.index_select(indices),
            self.contrast_hl.index_select(indices),
            self.contrast_hh.index_select(indices),
        )


@dataclass(frozen=True, slots=True)
class InfraredEvidence:
    delta: RelativeDelta
    advantage: RelativeDelta

    def index_select(self, indices: Tensor) -> InfraredEvidence:
        return InfraredEvidence(
            self.delta.index_select(indices), self.advantage.index_select(indices)
        )


@dataclass(frozen=True, slots=True)
class FocusEvidence:
    delta: RelativeDelta
    confidence: Tensor

    def index_select(self, indices: Tensor) -> FocusEvidence:
        return FocusEvidence(
            self.delta.index_select(indices), self.confidence.index_select(0, indices)
        )


@dataclass(frozen=True, slots=True)
class SemanticEvidence:
    guidance_feature: Tensor
    need: Tensor

    def index_select(self, indices: Tensor) -> SemanticEvidence:
        return SemanticEvidence(
            self.guidance_feature.index_select(0, indices), self.need.index_select(0, indices)
        )


@dataclass(frozen=True, slots=True)
class Stage4ExpertEvidence:
    low: LowEvidence
    detail: DetailEvidence
    infrared: InfraredEvidence
    focus: FocusEvidence
    semantic: SemanticEvidence

    def index_select(self, indices: Tensor) -> Stage4ExpertEvidence:
        return Stage4ExpertEvidence(
            self.low.index_select(indices),
            self.detail.index_select(indices),
            self.infrared.index_select(indices),
            self.focus.index_select(indices),
            self.semantic.index_select(indices),
        )

    def with_semantic_guidance(self, feature: Tensor) -> Stage4ExpertEvidence:
        return replace(self, semantic=SemanticEvidence(feature, self.semantic.need))

    def for_expert(
        self, expert: ExpertType
    ) -> LowEvidence | DetailEvidence | InfraredEvidence | FocusEvidence | SemanticEvidence:
        return {
            ExpertType.LOW_FREQUENCY: self.low,
            ExpertType.DETAIL: self.detail,
            ExpertType.INFRARED_SALIENCY: self.infrared,
            ExpertType.FOCUS: self.focus,
            ExpertType.SEMANTIC: self.semantic,
        }[expert]


@dataclass(frozen=True, slots=True)
class SiteEvidence:
    """Typed, site-local evidence shared by the spatial router and specialists."""

    router_feature: Tensor
    router_difference: Tensor
    router_low_energy: Tensor
    router_high_energy: Tensor
    router_focus: Tensor
    router_focus_available: Tensor
    router_boundary: Tensor
    router_boundary_available: Tensor
    router_uncertainty: Tensor
    task: TaskType
    modality_a: ModalityType
    modality_b: ModalityType
    availability: ExpertAvailability
    expert: ExpertEvidence | Stage4ExpertEvidence
    diagnostics: dict[str, Any]


class SiteEvidenceBuilder:
    """Parameter-free site evidence construction and deterministic alignment."""

    def __init__(
        self,
        dwt: HaarDWT2D,
        availability: ExpertAvailabilityPolicy,
        version: str = "stage3",
        relative_delta_clip: float = 5.0,
        detach_focus_evidence: bool = True,
    ) -> None:
        object.__setattr__(self, "_dwt", dwt)
        self.availability = availability
        self.version = version
        self.relative_delta_clip = relative_delta_clip
        self.detach_focus_evidence = detach_focus_evidence

    @staticmethod
    def project_optional_source(
        projection: nn.Module, source: Tensor | None
    ) -> Tensor | None:
        return projection(source) if source is not None else None

    @staticmethod
    def _align(value: Tensor, size: tuple[int, int]) -> Tensor:
        if value.shape[-2:] == size:
            return value
        return functional.interpolate(value, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def _router_map(
        value: Tensor | None,
        reference: Tensor,
        size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        batch = reference.shape[0]
        available = reference.new_full((batch, 1, *size), float(value is not None))
        if value is None:
            return reference.new_zeros(batch, 1, *size), available
        aligned = functional.interpolate(
            value.float(), size=size, mode="bilinear", align_corners=False
        ).mean(1, keepdim=True)
        return aligned, available

    @staticmethod
    def _zero_bands(reference: WaveletBands) -> WaveletBands:
        return WaveletBands(
            torch.zeros_like(reference.ll),
            torch.zeros_like(reference.lh),
            torch.zeros_like(reference.hl),
            torch.zeros_like(reference.hh),
            reference.meta,
        )

    @staticmethod
    def _average_bands(first: WaveletBands, second: WaveletBands) -> WaveletBands:
        return WaveletBands(
            (first.ll + second.ll) * 0.5,
            (first.lh + second.lh) * 0.5,
            (first.hl + second.hl) * 0.5,
            (first.hh + second.hh) * 0.5,
            first.meta,
        )

    @staticmethod
    def _difference_bands(
        first: WaveletBands, second: WaveletBands, absolute: bool
    ) -> WaveletBands:
        difference = lambda left, right: (left - right).abs() if absolute else left - right
        return WaveletBands(
            difference(first.ll, second.ll),
            difference(first.lh, second.lh),
            difference(first.hl, second.hl),
            difference(first.hh, second.hh),
            first.meta,
        )

    @staticmethod
    def _record_relative(
        diagnostics: dict[str, Any], prefix: str, relative: RelativeDelta
    ) -> None:
        diagnostics.update(
            {
                f"{prefix}/reference_rms": relative.reference_rms.detach().mean(),
                f"{prefix}/raw_delta_rms": relative.raw_delta_rms.detach().mean(),
                f"{prefix}/delta_to_reference_ratio": relative.delta_to_reference_ratio.detach().mean(),
                f"{prefix}/normalized_delta_rms": relative.normalized_delta_rms.detach().mean(),
                f"{prefix}/clip_fraction": relative.clip_fraction.detach().mean(),
            }
        )

    def _stage4_expert_evidence(
        self,
        feature: Tensor,
        bands: WaveletBands,
        aligned_a: Tensor,
        aligned_b: Tensor,
        source_a_available: bool,
        source_b_available: bool,
        infrared_feature: Tensor,
        other_feature: Tensor,
        semantic_maps: Tensor,
        focus_a: Tensor,
        focus_b: Tensor,
        focus_confidence: Tensor,
        focus_available: bool,
        task: TaskType,
        modality_a: ModalityType,
        modality_b: ModalityType,
    ) -> tuple[Stage4ExpertEvidence, bool, dict[str, Any]]:
        source_a_bands = (
            self._dwt(aligned_a) if source_a_available else self._zero_bands(bands)
        )
        source_b_bands = (
            self._dwt(aligned_b) if source_b_available else self._zero_bands(bands)
        )
        source_mean = self._average_bands(source_a_bands, source_b_bands)
        if task is TaskType.MFIF:
            contrast = self._difference_bands(source_a_bands, source_b_bands, True)
        else:
            infrared_bands, other_bands = (
                (source_a_bands, source_b_bands)
                if modality_a is ModalityType.INFRARED_GRAY
                else (source_b_bands, source_a_bands)
                if modality_b is ModalityType.INFRARED_GRAY
                else (self._zero_bands(bands), self._zero_bands(bands))
            )
            contrast = self._difference_bands(infrared_bands, other_bands, False)

        relative = lambda delta, reference: normalize_relative_delta(
            delta, reference, self.relative_delta_clip
        )
        low = LowEvidence(
            relative(source_mean.ll - bands.ll, bands.ll),
            relative(contrast.ll, bands.ll),
        )
        detail = DetailEvidence(
            relative(source_mean.lh - bands.lh, bands.lh),
            relative(source_mean.hl - bands.hl, bands.hl),
            relative(source_mean.hh - bands.hh, bands.hh),
            relative(contrast.lh, bands.lh),
            relative(contrast.hl, bands.hl),
            relative(contrast.hh, bands.hh),
        )
        infrared = InfraredEvidence(
            relative(infrared_feature - feature, feature),
            relative((infrared_feature - other_feature).abs(), feature),
        )
        if focus_available:
            confidence = focus_confidence
            focus_target = focus_a * aligned_a + focus_b * aligned_b
        else:
            focus_target, confidence = torch.zeros_like(feature), feature.new_zeros(
                feature.shape[0], 1, *feature.shape[-2:]
            )
        if self.detach_focus_evidence:
            focus_target, confidence = focus_target.detach(), confidence.detach()
        focus = FocusEvidence(relative(focus_target - feature, feature), confidence)
        semantic = SemanticEvidence(
            semantic_maps,
            # semantic_maps stores boundary and confidence (1 - uncertainty).
            semantic_maps[:, :1] * semantic_maps[:, 1:2],
        )
        diagnostics: dict[str, Any] = {}
        self._record_relative(diagnostics, "evidence/low_frequency", low.delta)
        self._record_relative(diagnostics, "evidence/low_frequency/contrast", low.contrast)
        for name, item in (
            ("lh", detail.lh),
            ("hl", detail.hl),
            ("hh", detail.hh),
        ):
            self._record_relative(diagnostics, f"evidence/detail/{name}", item)
        for name, item in (
            ("lh", detail.contrast_lh),
            ("hl", detail.contrast_hl),
            ("hh", detail.contrast_hh),
        ):
            self._record_relative(diagnostics, f"evidence/detail/contrast/{name}", item)
        self._record_relative(diagnostics, "evidence/infrared_saliency", infrared.delta)
        self._record_relative(
            diagnostics, "evidence/infrared_saliency/advantage", infrared.advantage
        )
        self._record_relative(diagnostics, "evidence/focus", focus.delta)
        return Stage4ExpertEvidence(low, detail, infrared, focus, semantic), focus_available, diagnostics

    def build(
        self,
        feature: Tensor,
        router_context: RouterContext,
        expert_context: ExpertContext,
        grid_size: tuple[int, int],
    ) -> SiteEvidence:
        batch, _, height, width = feature.shape
        size = (height, width)
        source_a = expert_context.source_a_feature
        source_b = expert_context.source_b_feature
        source_a_available, source_b_available = source_a is not None, source_b is not None
        aligned_a = self._align(source_a, size) if source_a_available else torch.zeros_like(feature)
        aligned_b = self._align(source_b, size) if source_b_available else torch.zeros_like(feature)
        difference = (aligned_a - aligned_b).abs() if source_a_available and source_b_available else torch.zeros_like(feature)
        if expert_context.modality_a is ModalityType.INFRARED_GRAY:
            infrared_feature, other_feature = aligned_a, aligned_b
        elif expert_context.modality_b is ModalityType.INFRARED_GRAY:
            infrared_feature, other_feature = aligned_b, aligned_a
        else:
            infrared_feature = other_feature = torch.zeros_like(feature)
        infrared_valid = torch.full(
            (batch,), expert_context.modality_a.is_infrared or expert_context.modality_b.is_infrared,
            dtype=torch.bool,
            device=feature.device,
        )

        boundary = expert_context.semantic_boundary
        uncertainty = expert_context.semantic_uncertainty
        semantic_available = boundary is not None or uncertainty is not None
        semantic_reference = boundary if boundary is not None else uncertainty
        if semantic_reference is None:
            semantic_maps = feature.new_zeros(batch, 2, height, width)
        else:
            semantic_boundary = (
                boundary if boundary is not None else torch.zeros_like(semantic_reference)
            )
            semantic_uncertainty = (
                uncertainty
                if uncertainty is not None
                else torch.zeros_like(semantic_reference)
            )
            semantic_maps = functional.interpolate(
                torch.cat((semantic_boundary, 1.0 - semantic_uncertainty), dim=1).float(),
                size=size,
                mode="bilinear",
                align_corners=False,
            ).to(feature.dtype)
        semantic_scale = feature.new_full(
            (batch, 1, 1, 1), float(semantic_available)
        )
        focus_a_raw, focus_b_raw = expert_context.focus_a, expert_context.focus_b
        focus_confidence_raw = expert_context.focus_confidence
        focus_available_for_expert = all(
            value is not None for value in (focus_a_raw, focus_b_raw, focus_confidence_raw)
        )
        focus_a = self._align(focus_a_raw, size) if focus_a_raw is not None else feature.new_zeros(batch, 1, *size)
        focus_b = self._align(focus_b_raw, size) if focus_b_raw is not None else feature.new_zeros(batch, 1, *size)
        focus_confidence = self._align(focus_confidence_raw, size) if focus_confidence_raw is not None else feature.new_zeros(batch, 1, *size)
        if self.version == "stage4b" and self.detach_focus_evidence:
            focus_a, focus_b, focus_confidence = (
                focus_a.detach(),
                focus_b.detach(),
                focus_confidence.detach(),
            )

        with torch.autocast(device_type=feature.device.type, enabled=False):
            value = feature.float()
            router_feature = functional.adaptive_avg_pool2d(value, grid_size)
            router_difference = functional.adaptive_avg_pool2d(
                difference.float(), grid_size
            )
            bands = self._dwt(feature)
            low, high = local_spectral_evidence_from_bands(
                bands, grid_size, value.dtype
            )
            if self.version == "stage4b":
                focus = functional.adaptive_avg_pool2d(
                    focus_confidence.float(), grid_size
                )
                router_boundary = functional.adaptive_avg_pool2d(
                    semantic_maps[:, :1].float(), grid_size
                )
                router_uncertainty = functional.adaptive_avg_pool2d(
                    (1.0 - semantic_maps[:, 1:2]).float(), grid_size
                )
                focus_available = value.new_full(
                    (batch, 1, *grid_size), float(focus_confidence_raw is not None)
                )
                boundary_available = value.new_full(
                    (batch, 1, *grid_size), float(boundary is not None)
                )
            else:
                focus, focus_available = self._router_map(
                    router_context.focus_confidence, value, grid_size
                )
                router_boundary, boundary_available = self._router_map(
                    router_context.semantic_boundary, value, grid_size
                )
                router_uncertainty, _ = self._router_map(
                    router_context.semantic_uncertainty, value, grid_size
                )
        diagnostics = (
            {"semantic_boundary": boundary.detach()}
            if boundary is not None
            else {}
        )
        expert: ExpertEvidence | Stage4ExpertEvidence = ExpertEvidence(
            bands,
            semantic_maps,
            semantic_scale,
            boundary is not None,
            infrared_feature,
            other_feature,
            infrared_valid,
        )
        if self.version == "stage4b":
            with torch.autocast(device_type=feature.device.type, enabled=False):
                expert, focus_available_for_expert, stage4_diagnostics = (
                    self._stage4_expert_evidence(
                        feature,
                        bands,
                        aligned_a,
                        aligned_b,
                        source_a_available,
                        source_b_available,
                        infrared_feature,
                        other_feature,
                        semantic_maps,
                        focus_a,
                        focus_b,
                        focus_confidence,
                        focus_available_for_expert,
                        router_context.task,
                        router_context.modality_a,
                        router_context.modality_b,
                    )
                )
            diagnostics.update(stage4_diagnostics)
        return SiteEvidence(
            router_feature,
            router_difference,
            low,
            high,
            focus,
            focus_available,
            router_boundary,
            boundary_available,
            router_uncertainty,
            router_context.task,
            router_context.modality_a,
            router_context.modality_b,
            self.availability.resolve(
                batch,
                feature.device,
                router_context.task,
                router_context.modality_a,
                router_context.modality_b,
                focus_available_for_expert,
                semantic_available,
            ),
            expert,
            diagnostics,
        )


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _valid(tensor: Tensor, value: bool = True) -> Tensor:
    return torch.full((tensor.shape[0],), value, dtype=torch.bool, device=tensor.device)


class ResidualConvUnit(nn.Module):
    def __init__(self, channels: int, expansion: int = 2, dilation: int = 1) -> None:
        super().__init__()
        hidden = channels * expansion
        self.body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
            ),
            nn.GroupNorm(_groups(channels), channels),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return self.body(tensor)


class CommonExpert(FunctionalExpert):
    expert_type = ExpertType.COMMON

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(channels, channels, 1)
        self.blocks = nn.Sequential(
            ResidualConvUnit(channels, expansion), ResidualConvUnit(channels, expansion)
        )
        self.output_projection = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, tensor: Tensor, *_: object) -> ExpertOutput:  # type: ignore[override]
        return ExpertOutput(
            self.output_projection(self.blocks(self.input_projection(tensor))),
            self.expert_type,
            _valid(tensor),
        )


class LowFrequencyExpert(FunctionalExpert):
    expert_type = ExpertType.LOW_FREQUENCY

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        self.dwt, self.idwt = HaarDWT2D(), HaarIDWT2D()
        blocks = max(1, math.gcd(channels, 8))
        self.low_projection = nn.Conv2d(channels, channels, 1)
        self.afno = AFNO2D(channels, blocks, residual=False)
        self.low_refine = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, tensor: Tensor, evidence: ExpertEvidence) -> ExpertOutput:
        bands = evidence.fused_bands
        projected = self.low_projection(bands.ll)
        low_prime = self.low_refine(self.afno(projected))
        low = low_prime - bands.ll
        zeros = torch.zeros_like(low)
        residual = self.idwt(
            WaveletBands(low, zeros, zeros, zeros, bands.original_size)
        )
        return ExpertOutput(residual, self.expert_type, _valid(tensor))


class DetailExpert(FunctionalExpert):
    expert_type = ExpertType.DETAIL

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        self.dwt, self.idwt = HaarDWT2D(), HaarIDWT2D()
        del expansion
        self.detail_refine = FDConv2d(
            channels, channels, kernel_size=3, kernel_num=min(8, 9), groups=channels
        )

    def forward(self, tensor: Tensor, evidence: ExpertEvidence) -> ExpertOutput:
        bands = evidence.fused_bands
        details = [
            self.detail_refine(item) - item for item in (bands.lh, bands.hl, bands.hh)
        ]
        residual = self.idwt(
            WaveletBands(
                torch.zeros_like(bands.ll),
                details[0],
                details[1],
                details[2],
                bands.original_size,
            )
        )
        return ExpertOutput(residual, self.expert_type, _valid(tensor))


class SemanticExpert(FunctionalExpert):
    """Large-receptive-field expert; predicted semantic maps can gate it later."""

    expert_type = ExpertType.SEMANTIC

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        self.semantic_refine = nn.Sequential(
            FDConv2d(channels, channels, 3, min(8, 9), groups=channels),
            ResidualConvUnit(channels, expansion),
        )
        self.content_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, 1, 1),
        )
        self.external_conditioner = nn.Conv2d(2, 1, 3, padding=1)

    def forward(self, tensor: Tensor, evidence: ExpertEvidence) -> ExpertOutput:
        residual = self.semantic_refine(tensor)
        gate_logits = self.content_gate(tensor) + self.external_conditioner(
            evidence.semantic_maps
        ) * evidence.semantic_condition_scale
        gate = torch.sigmoid(gate_logits)
        return ExpertOutput(
            residual * gate,
            self.expert_type,
            _valid(tensor),
            diagnostics={"gate": gate, "gate_logits": gate_logits},
        )


class InfraredSaliencyExpert(FunctionalExpert):
    expert_type = ExpertType.INFRARED_SALIENCY

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        self.saliency = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )
        self.refine = ResidualConvUnit(channels, expansion)

    def forward(self, tensor: Tensor, evidence: ExpertEvidence) -> ExpertOutput:
        infrared, other = evidence.infrared_feature, evidence.other_feature
        difference = (infrared - other).abs()
        saliency = self.saliency(torch.cat((infrared, difference, tensor), dim=1))
        residual = self.refine(saliency * (infrared - tensor)) * evidence.infrared_samples_valid[
            :, None, None, None
        ]
        return ExpertOutput(
            residual,
            self.expert_type,
            evidence.infrared_samples_valid,
            diagnostics={"saliency": saliency},
        )


class Stage4LowFrequencyExpert(FunctionalExpert):
    expert_type = ExpertType.LOW_FREQUENCY

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        del expansion
        self.dwt, self.idwt = HaarDWT2D(), HaarIDWT2D()
        self.low_projection = nn.Conv2d(channels, channels, 1)
        self.afno = AFNO2D(channels, max(1, math.gcd(channels, 8)), residual=False)
        self.low_refine = nn.Conv2d(channels, channels, 3, padding=1)
        self.delta_gain = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, tensor: Tensor, evidence: LowEvidence) -> ExpertOutput:
        delta, contrast = evidence.delta.value, evidence.contrast.value
        delta_ll = self.low_refine(self.afno(self.low_projection(delta)))
        delta_ll = delta_ll + self.delta_gain * (delta + 0.25 * contrast)
        zeros = torch.zeros_like(delta_ll)
        return ExpertOutput(
            self.idwt(WaveletBands(delta_ll, zeros, zeros, zeros, tensor.shape[-2:])),
            self.expert_type,
            _valid(tensor),
        )


class Stage4DetailExpert(FunctionalExpert):
    expert_type = ExpertType.DETAIL

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        del expansion
        self.dwt, self.idwt = HaarDWT2D(), HaarIDWT2D()
        self.detail_refine = FDConv2d(
            channels, channels, kernel_size=3, kernel_num=min(8, 9), groups=channels
        )
        self.delta_gain = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, tensor: Tensor, evidence: DetailEvidence) -> ExpertOutput:
        def refine(delta: RelativeDelta, contrast: RelativeDelta) -> Tensor:
            value = delta.value
            return self.detail_refine(value) + self.delta_gain * (
                value + 0.25 * contrast.value
            )

        details = (
            refine(evidence.lh, evidence.contrast_lh),
            refine(evidence.hl, evidence.contrast_hl),
            refine(evidence.hh, evidence.contrast_hh),
        )
        return ExpertOutput(
            self.idwt(
                WaveletBands(
                    torch.zeros_like(details[0]),
                    details[0],
                    details[1],
                    details[2],
                    tensor.shape[-2:],
                )
            ),
            self.expert_type,
            _valid(tensor),
        )


class Stage4InfraredSaliencyExpert(FunctionalExpert):
    expert_type = ExpertType.INFRARED_SALIENCY

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        self.condition = nn.Conv2d(channels * 2, channels, 1)
        self.saliency = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )
        self.refine = ResidualConvUnit(channels, expansion)
        self.delta_gain = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, tensor: Tensor, evidence: InfraredEvidence) -> ExpertOutput:
        delta, advantage = (
            evidence.delta.value,
            evidence.advantage.value,
        )
        condition = self.condition(torch.cat((delta, advantage), dim=1))
        saliency = self.saliency(torch.cat((tensor, delta, advantage), dim=1))
        residual = self.refine(saliency * condition) + self.delta_gain * delta
        return ExpertOutput(
            residual,
            self.expert_type,
            _valid(tensor),
            diagnostics={"saliency": saliency},
        )


class FocusExpert(FunctionalExpert):
    expert_type = ExpertType.FOCUS

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        self.condition = nn.Conv2d(channels, channels, 1)
        self.confidence_projection = nn.Conv2d(1, channels, 1)
        self.refine = ResidualConvUnit(channels, expansion)
        self.delta_gain = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, tensor: Tensor, evidence: FocusEvidence) -> ExpertOutput:
        delta, confidence = evidence.delta.value, evidence.confidence
        condition = self.condition(delta) + self.confidence_projection(confidence)
        residual = self.refine(condition) + self.delta_gain * delta * confidence
        return ExpertOutput(
            residual,
            self.expert_type,
            _valid(tensor),
            diagnostics={"confidence": confidence},
        )


class Stage4SemanticExpert(FunctionalExpert):
    expert_type = ExpertType.SEMANTIC

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        self.semantic_refine = nn.Sequential(
            FDConv2d(channels, channels, 3, min(8, 9), groups=channels),
            ResidualConvUnit(channels, expansion),
        )
        self.content_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, 1, 1),
        )
        self.guidance_gate = nn.Conv2d(channels, 1, 3, padding=1)
        self.need_gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, tensor: Tensor, evidence: SemanticEvidence) -> ExpertOutput:
        gate_logits = (
            self.content_gate(tensor)
            + self.guidance_gate(evidence.guidance_feature)
            + self.need_gain * evidence.need
        )
        gate = torch.sigmoid(gate_logits)
        return ExpertOutput(
            self.semantic_refine(tensor) * gate,
            self.expert_type,
            _valid(tensor),
            diagnostics={"gate": gate, "gate_logits": gate_logits},
        )


EXPERT_CLASSES: dict[ExpertType, type[FunctionalExpert]] = {
    ExpertType.COMMON: CommonExpert,
    ExpertType.LOW_FREQUENCY: LowFrequencyExpert,
    ExpertType.DETAIL: DetailExpert,
    ExpertType.SEMANTIC: SemanticExpert,
    ExpertType.INFRARED_SALIENCY: InfraredSaliencyExpert,
}

STAGE4_EXPERT_CLASSES: dict[ExpertType, type[FunctionalExpert]] = {
    ExpertType.COMMON: CommonExpert,
    ExpertType.LOW_FREQUENCY: Stage4LowFrequencyExpert,
    ExpertType.DETAIL: Stage4DetailExpert,
    ExpertType.SEMANTIC: Stage4SemanticExpert,
    ExpertType.INFRARED_SALIENCY: Stage4InfraredSaliencyExpert,
    ExpertType.FOCUS: FocusExpert,
}


def build_functional_expert(
    expert: ExpertType | str,
    channels: int,
    expansion: int = 2,
    version: str = "stage3",
) -> FunctionalExpert:
    expert_type = ExpertType.parse(expert)
    classes = STAGE4_EXPERT_CLASSES if version == "stage4b" else EXPERT_CLASSES
    return classes[expert_type](channels, expansion)


from torch import nn

from tfs_moe_fusion.types import ExpertType


class FunctionalExpertPool(nn.Module):
    """Own the expert modules and centralize their availability contract."""

    def __init__(
        self,
        channels: int,
        expert_names: list[str] | tuple[str, ...],
        expansion: int = 2,
        version: str = "stage3",
    ) -> None:
        super().__init__()
        self._expert_names = tuple(
            ExpertType.parse(name).value for name in expert_names
        )
        self.modules_by_name = nn.ModuleDict(
            {
                name: build_functional_expert(name, channels, expansion, version)
                for name in self._expert_names
            }
        )

    @property
    def expert_names(self) -> tuple[str, ...]:
        return self._expert_names

    @property
    def num_experts(self) -> int:
        return len(self._expert_names)

    def get_expert(self, index: int | str | ExpertType) -> FunctionalExpert:
        if isinstance(index, int):
            name = self._expert_names[index]
        else:
            name = ExpertType.parse(index).value
        return self.modules_by_name[name]  # type: ignore[return-value]

    def get_validity(self, context: ExpertContext) -> Tensor:
        reference = context.source_a_feature
        if reference is None:
            reference = context.source_b_feature
        if reference is None:
            raise ValueError("Expert validity requires a source feature")
        validity = torch.ones(
            reference.shape[0],
            self.num_experts,
            dtype=torch.bool,
            device=reference.device,
        )
        if not (context.modality_a.is_infrared or context.modality_b.is_infrared):
            infrared = self._expert_names.index(ExpertType.INFRARED_SALIENCY.value)
            validity[:, infrared] = False
        return validity

    def __iter__(self):
        return (self.modules_by_name[name] for name in self._expert_names)

    def __len__(self) -> int:
        return self.num_experts


class SharedExpertBank(nn.Module):
    """One canonical expert set reused by every Stage 2 MoE site."""

    default_specialist_names = (
        ExpertType.LOW_FREQUENCY.value,
        ExpertType.DETAIL.value,
        ExpertType.SEMANTIC.value,
        ExpertType.INFRARED_SALIENCY.value,
    )

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        specialist_names: tuple[str, ...] | None = None,
        version: str = "stage3",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.specialist_names = specialist_names or self.default_specialist_names
        self.common = build_functional_expert(
            ExpertType.COMMON, channels, expansion, version
        )
        self.specialists = nn.ModuleDict(
            {
                name: build_functional_expert(name, channels, expansion, version)
                for name in self.specialist_names
            }
        )


class MoESiteAdapter(nn.Module):
    """Keep a site in its native feature space around canonical MoE deltas."""

    def __init__(
        self,
        stage_channels: int,
        expert_dim: int,
        expert_version: str = "stage3",
        semantic_guidance_source: str = "evidence_maps",
    ) -> None:
        super().__init__()
        self.feature_in = nn.Conv2d(stage_channels, expert_dim, 1)
        self.source_in = nn.Conv2d(stage_channels, expert_dim, 1)
        self.guidance_in = (
            nn.Conv2d(
                stage_channels if semantic_guidance_source == "stage_feature" else 2,
                expert_dim,
                1,
            )
            if expert_version == "stage4b"
            else nn.Identity()
        )
        self.delta_out = nn.Conv2d(expert_dim, stage_channels, 1, bias=False)

    def project_source(self, source: Tensor) -> Tensor:
        return self.source_in(source)

    def project_guidance(self, guidance: Tensor) -> Tensor:
        return self.guidance_in(guidance)


from torch import nn

from tfs_moe_fusion.types import (
    ExpertType,
)


class TaskEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(TaskType), embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(
        self,
        task: TaskType | Tensor,
        batch: int | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        if isinstance(task, Tensor):
            if task.ndim != 1 or task.dtype != torch.long:
                raise ValueError("TaskEmbedding tensor input must be long [B]")
            indices = task
        else:
            if batch is None or device is None:
                raise ValueError("Scalar TaskType input requires batch and device")
            indices = torch.full((batch,), task.index, dtype=torch.long, device=device)
        return self.embedding(indices)


class EvidenceBranch(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, experts: int, norm: bool = False
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        if norm:
            modules.append(nn.LayerNorm(input_size))
        modules.extend(
            (
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, experts),
            )
        )
        self.encoder = nn.Sequential(*modules)
        nn.init.normal_(self.encoder[-1].weight, std=1e-3)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, tensor: Tensor) -> Tensor:
        return self.encoder(tensor)


class FeatureRouter(EvidenceBranch):
    def __init__(self, channels: int, hidden: int, experts: int) -> None:
        super().__init__(channels, hidden, experts)

    def forward(self, feature: Tensor) -> Tensor:
        return super().forward(feature.float().mean(dim=(-2, -1)))


class TaskRouter(EvidenceBranch):
    def __init__(self, embedding_dim: int, hidden: int, experts: int) -> None:
        super().__init__(embedding_dim, hidden, experts)


class SpectralRouter(EvidenceBranch):
    def __init__(self, hidden: int, experts: int) -> None:
        super().__init__(11, hidden, experts, norm=True)

    def forward(self, value: RouterContext | SpectralStatistics) -> Tensor:
        if isinstance(value, RouterContext):
            stats = value.frequency_stats
            assert stats is not None
            reference = value.feature
        else:
            stats = value
            reference = stats.low_energy
        batch = reference.shape[0]
        dwt = (
            stats.dwt_energy
            if stats.dwt_energy is not None
            else reference.new_zeros(batch, 4)
        )
        rings = (
            stats.radial_energy
            if stats.radial_energy is not None
            else reference.new_zeros(batch, 4)
        )
        mid = (
            stats.mid_energy
            if stats.mid_energy is not None
            else (stats.low_energy + stats.high_energy) * 0.5
        )
        descriptor = torch.cat(
            (
                dwt.float(),
                rings.float(),
                torch.stack((stats.low_energy, mid, stats.high_energy), 1).float(),
            ),
            1,
        )
        return super().forward(torch.log1p(descriptor.clamp_min(0)))


class ModalityRouter(nn.Module):
    def __init__(
        self,
        channels: int,
        embedding_dim: int,
        hidden: int,
        experts: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(ModalityType), embedding_dim)
        self.branch = EvidenceBranch(embedding_dim * 2 + channels, hidden, experts)

    def forward(self, context: RouterContext) -> Tensor:
        feature, batch = context.feature, context.feature.shape[0]
        indices = torch.tensor(
            [
                list(ModalityType).index(context.modality_a),
                list(ModalityType).index(context.modality_b),
            ],
            device=feature.device,
        )
        embedding_a, embedding_b = self.embedding(indices).float()
        pair = torch.cat(
            (embedding_a + embedding_b, (embedding_a - embedding_b).abs())
        )[None].expand(batch, -1)
        difference = (
            (context.source_a.float() - context.source_b.float()).abs().mean((-2, -1))
            if context.source_a is not None and context.source_b is not None
            else feature.float().mean((-2, -1)).new_zeros(batch, feature.shape[1])
        )
        return self.branch(torch.cat((pair, difference), 1))


class AuxiliaryRouter(nn.Module):
    def __init__(self, hidden: int, experts: int) -> None:
        super().__init__()
        self.null_focus = nn.Parameter(torch.zeros(5))
        self.null_semantic = nn.Parameter(torch.zeros(5))
        self.branch = EvidenceBranch(12, hidden, experts)

    @staticmethod
    def _summary(value: Tensor, threshold: float) -> Tensor:
        flat = value.float().flatten(1)
        return torch.stack(
            (
                flat.mean(1),
                flat.std(1, unbiased=False),
                (flat > threshold).float().mean(1),
            ),
            dim=1,
        )

    def forward(self, context: RouterContext) -> tuple[Tensor, dict[str, object]]:
        batch, reference = context.feature.shape[0], context.feature
        focus_available = all(
            value is not None
            for value in (context.focus_a, context.focus_b, context.focus_confidence)
        )
        if focus_available:
            assert context.focus_a is not None and context.focus_b is not None
            assert context.focus_confidence is not None
            summary = self._summary(context.focus_confidence, 0.7)
            focus = torch.cat(
                (
                    summary[:, :2],
                    context.focus_a.float().flatten(1).mean(1, keepdim=True),
                    context.focus_b.float().flatten(1).mean(1, keepdim=True),
                    summary[:, 2:3],
                ),
                dim=1,
            )
        else:
            focus = self.null_focus[None].expand(batch, -1)
        semantic_available = (
            context.semantic_boundary is not None
            and context.semantic_uncertainty is not None
        )
        if semantic_available:
            assert (
                context.semantic_boundary is not None
                and context.semantic_uncertainty is not None
            )
            boundary, uncertainty = (
                context.semantic_boundary.float().flatten(1),
                context.semantic_uncertainty.float().flatten(1),
            )
            semantic = torch.stack(
                (
                    boundary.mean(1),
                    (boundary > 0.1).float().mean(1),
                    uncertainty.mean(1),
                    uncertainty.std(1, unbiased=False),
                    (uncertainty > 0.5).float().mean(1),
                ),
                dim=1,
            )
        else:
            semantic = self.null_semantic[None].expand(batch, -1)
        availability = reference.new_tensor(
            [float(focus_available), float(semantic_available)]
        )[None].expand(batch, -1)
        descriptor = torch.cat(
            (focus.to(reference), semantic.to(reference), availability), dim=1
        )
        return self.branch(descriptor.float()), {
            "focus_available": focus_available,
            "semantic_available": semantic_available,
            "focus_descriptor": focus,
            "semantic_descriptor": semantic,
        }


class SpatialRouter(nn.Module):
    def __init__(self, channels: int, experts: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels + 6, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, experts, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, context: RouterContext) -> Tensor:
        x, size = context.feature, context.feature.shape[-2:]

        def resize(value: Tensor | None) -> Tensor:
            if value is None:
                return x.new_zeros(x.shape[0], 1, *size)
            return functional.interpolate(
                value.to(x), size=size, mode="bilinear", align_corners=False
            ).mean(1, keepdim=True)

        difference = None
        if context.source_a is not None and context.source_b is not None:
            difference = (
                (context.source_a - context.source_b).abs().mean(1, keepdim=True)
            )
        maps = (
            resize(difference),
            resize(context.focus_confidence),
            resize(context.semantic_boundary),
            resize(context.semantic_uncertainty),
            x.new_full(
                (x.shape[0], 1, *size), float(context.focus_confidence is not None)
            ),
            x.new_full(
                (x.shape[0], 1, *size), float(context.semantic_boundary is not None)
            ),
        )
        return 2.0 * torch.sigmoid(self.body(torch.cat((x, *maps), dim=1)))


class JointTopKRouter(nn.Module):
    branch_names = ("feature", "task", "spectrum", "modality", "auxiliary")

    def __init__(
        self,
        channels: int,
        experts: list[str],
        top_k: int = 2,
        hidden_channels: int = 64,
        spatial_gating: bool = True,
        temperature: float = 1.0,
        task_embedding: TaskEmbedding | None = None,
        modality_embedding_dim: int = 32,
        branch_dropout: float = 0.0,
        noisy_topk: bool = False,
        noisy_topk_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.expert_types = tuple(ExpertType.parse(item) for item in experts)
        self.expert_count, self.top_k, self.temperature = (
            len(experts),
            top_k,
            temperature,
        )
        if task_embedding is None:
            self.owned_task_embedding = TaskEmbedding(hidden_channels)
            object.__setattr__(self, "_shared_task_embedding", None)
        else:
            self.owned_task_embedding = None
            object.__setattr__(self, "_shared_task_embedding", task_embedding)
        task_dim = self.task_embedding.embedding.embedding_dim
        self.feature_branch = FeatureRouter(
            channels, hidden_channels, self.expert_count
        )
        self.task_branch = TaskRouter(task_dim, hidden_channels, self.expert_count)
        self.spectrum_branch = SpectralRouter(hidden_channels, self.expert_count)
        self.modality_branch = ModalityRouter(
            channels, modality_embedding_dim, hidden_channels, self.expert_count
        )
        self.auxiliary_router = AuxiliaryRouter(hidden_channels, self.expert_count)
        self.branch_logits = nn.Parameter(torch.zeros(len(self.branch_names)))
        self.branch_dropout, self.noisy_topk, self.noisy_topk_std = (
            branch_dropout,
            noisy_topk,
            noisy_topk_std,
        )
        self.spatial_router = (
            SpatialRouter(channels, self.expert_count) if spatial_gating else None
        )

    @property
    def task_embedding(self) -> TaskEmbedding:
        shared = self._shared_task_embedding
        return shared if shared is not None else self.owned_task_embedding

    @property
    def modality_embedding(self) -> nn.Embedding:
        return self.modality_branch.embedding

    def forward(self, context: RouterContext) -> RouterOutput:
        feature, batch = context.feature, context.feature.shape[0]
        with torch.autocast(device_type=feature.device.type, enabled=False):
            feature_logits = self.feature_branch(feature)
            task_logits = self.task_branch(
                self.task_embedding(context.task, batch, feature.device).float()
            )
            spectrum_logits = self.spectrum_branch(context)
            modality_logits = self.modality_branch(context)
            auxiliary_logits, auxiliary = self.auxiliary_router(context)
            branches = torch.stack(
                (
                    feature_logits,
                    task_logits,
                    spectrum_logits,
                    modality_logits,
                    auxiliary_logits,
                ),
                1,
            )
            weights = torch.softmax(self.branch_logits, 0)[None].expand(batch, -1)
            if self.training and self.branch_dropout:
                keep = torch.rand_like(weights) >= self.branch_dropout
                weights = weights * keep
                weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
            logits = (branches * weights[:, :, None]).sum(1)
            if self.training and self.noisy_topk:
                logits = logits + torch.randn_like(logits) * self.noisy_topk_std
            valid = torch.ones(
                batch, self.expert_count, dtype=torch.bool, device=feature.device
            )
            if not (context.modality_a.is_infrared or context.modality_b.is_infrared):
                valid[:, self.expert_types.index(ExpertType.INFRARED_SALIENCY)] = False
            masked = logits.masked_fill(~valid, -torch.inf)
            probabilities = torch.softmax(masked / self.temperature, 1)
            selected, indices = probabilities.topk(self.top_k, 1)
            topk_weights = selected / selected.sum(1, keepdim=True)
            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1)
            importance = probabilities.mean(0)
            hard_load = (
                functional.one_hot(indices, self.expert_count).float().mean((0, 1))
            )
        spatial = (
            self.spatial_router(context) if self.spatial_router is not None else None
        )
        return RouterOutput(
            masked,
            probabilities,
            indices,
            topk_weights,
            valid,
            spatial,
            weights,
            entropy,
            importance,
            hard_load,
            auxiliary,
        )


class SiteSpatialRouter(nn.Module):
    """Site-local v2 router that fuses canonical spatial evidence directly."""

    def __init__(
        self,
        channels: int,
        experts: list[str],
        hidden_channels: int,
        patch_size: int,
        temperature: float,
        task_embedding: TaskEmbedding | None,
        modality_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.expert_types = tuple(ExpertType.parse(item) for item in experts)
        self.expert_count = len(experts)
        self.patch_size, self.temperature = patch_size, temperature
        if task_embedding is None:
            self.owned_task_embedding = TaskEmbedding(hidden_channels)
            object.__setattr__(self, "_shared_task_embedding", None)
        else:
            self.owned_task_embedding = None
            object.__setattr__(self, "_shared_task_embedding", task_embedding)
        width = max(8, hidden_channels)
        self.feature_projection = nn.Conv2d(channels, width, 1)
        self.difference_projection = nn.Conv2d(channels, width, 1)
        self.task_projection = nn.Linear(
            self.task_embedding.embedding.embedding_dim, width
        )
        self.modality_embedding = nn.Embedding(
            len(ModalityType) ** 2, modality_embedding_dim
        )
        self.modality_projection = nn.Linear(modality_embedding_dim, width)
        self.body = nn.Sequential(
            nn.Conv2d(width * 4 + 7, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, self.expert_count, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    @property
    def task_embedding(self) -> TaskEmbedding:
        shared = self._shared_task_embedding
        return shared if shared is not None else self.owned_task_embedding

    def forward(self, evidence: SiteEvidence) -> RouterOutput:
        feature = evidence.router_feature
        batch, _, height, width = feature.shape
        grid_size = (height, width)
        with torch.autocast(device_type=feature.device.type, enabled=False):
            task = self.task_projection(
                self.task_embedding(evidence.task, batch, feature.device).float()
            )[:, :, None, None].expand(-1, -1, *grid_size)
            pair = list(ModalityType).index(evidence.modality_a) * len(ModalityType)
            pair += list(ModalityType).index(evidence.modality_b)
            modality = self.modality_projection(
                self.modality_embedding.weight[pair].float()[None].expand(batch, -1)
            )[:, :, None, None].expand(-1, -1, *grid_size)
            logits = self.body(
                torch.cat(
                    (
                        self.feature_projection(feature),
                        self.difference_projection(evidence.router_difference),
                        task,
                        modality,
                        evidence.router_low_energy,
                        evidence.router_high_energy,
                        evidence.router_focus,
                        evidence.router_focus_available,
                        evidence.router_boundary,
                        evidence.router_boundary_available,
                        evidence.router_uncertainty,
                    ),
                    dim=1,
                )
            )
            valid = evidence.availability.valid_mask
            masked = logits.masked_fill(~valid[:, :, None, None], -torch.inf)
            probabilities = torch.softmax(masked / self.temperature, dim=1)
            entropy_map = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1)
            top = probabilities.topk(min(2, self.expert_count), dim=1).values
            importance = probabilities.mean((0, 2, 3))
            hard_load = functional.one_hot(
                probabilities.argmax(1), self.expert_count
            ).float().mean((0, 1, 2))
        return RouterOutput(
            masked,
            probabilities,
            None,
            None,
            valid,
            entropy=entropy_map.mean(),
            importance=importance,
            hard_load=hard_load,
            auxiliary={
                "entropy_map": entropy_map.detach(),
                "router/top1_margin": (top[:, 0] - top[:, 1]).mean().detach(),
                "router/spatial_variance": probabilities.var(
                    dim=(-2, -1), unbiased=False
                ).mean().detach(),
                "router_grid_size": grid_size,
            },
        )


UnifiedRouter = JointTopKRouter

from dataclasses import dataclass, field

from torch import nn

from tfs_moe_fusion.config import MoEConfig


@dataclass(slots=True)
class MoEOutput:
    feature: Tensor
    residual: Tensor
    router: RouterOutput
    spatial_gates: Tensor | None
    expert_outputs: dict[str, ExpertOutput] | None = None
    diagnostics: RouterDiagnostics | None = None
    aux: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        """Preserve the established ``feature, diagnostics = block(...)`` API."""
        yield self.feature
        yield self.diagnostics


@dataclass(frozen=True, slots=True)
class MoEExecutionPolicy:
    """Resolved execution semantics shared by every MoE forward in one step."""

    mode: str
    uniform_to_soft: float = 1.0
    soft_to_topk: float = 0.0
    spatial_gate_scale: float = 1.0
    refresh_routed_fraction: float = 0.5
    detach_router: bool = False
    expert_only: bool = False
    temperature: float | None = None
    noise_std: float = 0.0
    compute_frequency_regularizers: bool = True
    compute_infrared_regularizers: bool = True

    @property
    def sparse(self) -> bool:
        return self.mode == "sparse_batch"

    @classmethod
    def legacy(cls, sparse: bool) -> MoEExecutionPolicy:
        return cls("sparse_batch" if sparse else "dense_masked")


class FunctionalMoEBlock(nn.Module):
    routing_override_modes = ("learned", "uniform", "shuffled")

    def __init__(
        self,
        channels: int,
        config: MoEConfig,
        block_id: str,
        task_embedding: TaskEmbedding | None = None,
        shared_expert_bank: SharedExpertBank | None = None,
    ) -> None:
        super().__init__()
        self.block_id = block_id
        self.architecture_version = config.architecture_version
        self.routing_mode = config.routing_mode
        self.functional_expert_version = config.functional_expert_version
        self.semantic_guidance_source = config.semantic_guidance_source
        self.common_always_on = config.common_always_on
        self.shared_pool_enabled = shared_expert_bank is not None
        if self.shared_pool_enabled and (
            self.architecture_version != "v2"
            or self.routing_mode not in {"global_soft", "spatial_soft"}
            or not self.common_always_on
        ):
            raise ValueError(
                "Shared expert sites require v2 routing with common always-on"
            )
        object.__setattr__(self, "_shared_expert_bank", shared_expert_bank)
        configured_experts = tuple(config.experts)
        if self.architecture_version == "v2":
            self.expert_names = tuple(
                name
                for name in configured_experts
                if name != ExpertType.COMMON.value
            )
            if self.shared_pool_enabled:
                assert shared_expert_bank is not None
                if self.expert_names != shared_expert_bank.specialist_names:
                    raise ValueError("Shared expert names must match the canonical bank")
                self.adapter = MoESiteAdapter(
                    channels,
                    shared_expert_bank.channels,
                    config.functional_expert_version,
                    config.semantic_guidance_source,
                )
                router_channels = shared_expert_bank.channels
            else:
                self.common_expert = build_functional_expert(
                    ExpertType.COMMON,
                    channels,
                    config.expert_expansion,
                    config.functional_expert_version,
                )
                self.expert_pool = FunctionalExpertPool(
                    channels,
                    self.expert_names,
                    config.expert_expansion,
                    config.functional_expert_version,
                )
                router_channels = channels
        else:
            self.expert_names = configured_experts
            self.expert_pool = FunctionalExpertPool(
                channels,
                self.expert_names,
                config.expert_expansion,
                config.functional_expert_version,
            )
            router_channels = channels
        if self.routing_mode == "spatial_soft":
            stage = next(part for part in block_id.split(".") if part in config.patch_size)
            self.router = SiteSpatialRouter(
                router_channels,
                list(self.expert_names),
                config.router_hidden_channels,
                config.patch_size[stage],
                config.router_temperature,
                task_embedding,
                config.modality_embedding_dim,
            )
        else:
            self.router = JointTopKRouter(
                router_channels,
                list(self.expert_names),
                config.top_k,
                config.router_hidden_channels,
                config.spatial_gating if self.architecture_version == "legacy" else False,
                config.router_temperature,
                task_embedding,
                config.modality_embedding_dim,
                config.branch_dropout,
                config.noisy_topk,
                config.noisy_topk_std,
            )
        low_expert = self._expert_by_name(ExpertType.LOW_FREQUENCY.value)
        if not isinstance(low_expert, (LowFrequencyExpert, Stage4LowFrequencyExpert)):
            raise TypeError("MoE evidence requires the low-frequency expert")
        self.site_evidence_builder = SiteEvidenceBuilder(
            low_expert.dwt,
            ExpertAvailabilityPolicy(self.expert_names, config.functional_expert_version),
            config.functional_expert_version,
            config.relative_delta_clip,
            config.detach_focus_evidence,
        )
        self.sparse_execution = config.sparse_execution
        self.train_execution = config.train_execution
        self.inference_execution = config.inference_execution
        self.execution_policy: MoEExecutionPolicy | None = None
        self.routing_override = "learned"
        self.routing_override_seed = 3407
        if self.architecture_version == "v2":
            self.common_scale = nn.Parameter(
                torch.full((1, router_channels, 1, 1), config.common_scale_init)
            )
            self.specialist_scale = nn.Parameter(
                torch.full(
                    (1, router_channels, 1, 1), config.specialist_scale_init
                )
            )
        else:
            self.residual_scale = nn.Parameter(
                torch.full((1, channels, 1, 1), config.residual_scale)
            )

    def forward(
        self,
        tensor: Tensor,
        router_context: RouterContext,
        expert_context: ExpertContext,
        sparse_execution: bool | None = None,
        return_expert_outputs: bool = False,
    ) -> MoEOutput:
        if self.architecture_version == "v2":
            if self.shared_pool_enabled:
                return self._forward_shared_v2(
                    tensor,
                    router_context,
                    expert_context,
                    sparse_execution,
                    return_expert_outputs,
                )
            return self._forward_v2(
                tensor,
                router_context,
                expert_context,
                sparse_execution,
                return_expert_outputs,
            )
        learned_routing = self.router(router_context)
        routing = self._apply_routing_override(learned_routing)
        policy = self._resolve_policy(sparse_execution)
        evidence = self._build_site_evidence(
            tensor, router_context, expert_context, tensor.shape[-2:]
        )
        if self.routing_override == "uniform" or self.routing_override.startswith(
            "single:"
        ):
            global_weights = routing.probabilities.to(tensor)
            spatial_gates = None
            if policy.detach_router:
                global_weights = global_weights.detach()
        else:
            global_weights, spatial_gates = self._mixture_weights(
                tensor, routing, policy
            )
        mixed_residual = torch.zeros_like(tensor)
        retained_outputs: dict[str, ExpertOutput] | None = (
            {} if return_expert_outputs else None
        )
        regularizers: dict[str, list[Tensor]] = {}
        residual_rms: dict[str, Tensor] = {}
        contribution_rms = {
            name: tensor.detach().float().new_zeros(()) for name in self.expert_names
        }
        expert_sample_assignments = 0
        for index, expert in enumerate(self.experts):
            if self.routing_override == "learned":
                sample_indices = self._expert_indices(tensor, routing, index, policy)
            else:
                selected = (
                    global_weights[:, index].detach() != 0
                ) & routing.valid_expert_mask[:, index]
                sample_indices = torch.nonzero(selected, as_tuple=False).flatten()
            if sample_indices.numel() == 0:
                continue
            expert_sample_assignments += sample_indices.numel()
            selected_tensor = tensor.index_select(0, sample_indices)
            selected_evidence = evidence.expert.index_select(sample_indices)
            expert_output = (
                expert(selected_tensor)
                if self.expert_names[index] == ExpertType.COMMON.value
                else expert(selected_tensor, selected_evidence)
            )
            valid = expert_output.valid_samples[:, None, None, None]
            selected_residual = expert_output.residual * valid
            residual_rms[self.expert_names[index]] = (
                selected_residual.detach().float().square().mean().sqrt()
            )
            weights = global_weights.index_select(0, sample_indices)[
                :, index : index + 1, None, None
            ]
            contribution = selected_residual * weights
            if spatial_gates is not None:
                gates = spatial_gates.index_select(0, sample_indices)[
                    :, index : index + 1
                ]
                contribution = contribution * gates
            contribution_rms[self.expert_names[index]] = (
                contribution.detach().float().square().sum() / tensor.numel()
            ).sqrt()
            mixed_residual = mixed_residual.index_add(0, sample_indices, contribution)
            if self._regularizers_enabled(expert_output.expert, policy):
                for name, value in self._expert_regularizers(
                    expert_output, selected_evidence
                ).items():
                    regularizers.setdefault(name, []).append(value)
            if retained_outputs is not None:
                retained_outputs[self.expert_names[index]] = self._restore_output(
                    tensor, sample_indices, expert_output
                )
        scaled_residual = self.residual_scale * mixed_residual
        output = tensor + scaled_residual
        auxiliary = dict(routing.auxiliary)
        auxiliary["expert_sample_assignments"] = expert_sample_assignments
        auxiliary["expert_residual_rms"] = residual_rms
        auxiliary["expert_weighted_contribution_rms"] = contribution_rms
        auxiliary["residual_scale_rms"] = (
            self.residual_scale.detach().float().square().mean().sqrt()
        )
        input_rms = tensor.detach().float().square().mean().sqrt()
        auxiliary["moe_residual_to_input_ratio"] = (
            scaled_residual.detach().float().square().mean().sqrt()
            / input_rms.clamp_min(torch.finfo(torch.float32).eps)
        )
        probabilities = routing.probabilities.detach().float()
        largest = probabilities.topk(min(2, probabilities.shape[1]), dim=1).values
        top1_margin = (
            (largest[:, 0] - largest[:, 1]).mean()
            if largest.shape[1] == 2
            else largest[:, 0].mean()
        )
        auxiliary.update(
            {
                "expert_names": self.expert_names,
                "router/top1_margin": top1_margin,
                "router/probability_std": probabilities.std(
                    dim=1, unbiased=False
                ).mean(),
                "router/entropy": (
                    routing.entropy.detach().float().mean()
                    if routing.entropy is not None
                    else -(probabilities * probabilities.clamp_min(1e-8).log())
                    .sum(1)
                    .mean()
                ),
                "routing_override": self.routing_override,
            }
        )
        auxiliary.update(evidence.diagnostics)
        if regularizers:
            auxiliary["expert_regularizers"] = {
                name: torch.stack(values).mean()
                for name, values in regularizers.items()
            }
        if retained_outputs is not None:
            auxiliary["expert_residuals"] = {
                name: value.residual for name, value in retained_outputs.items()
            }
        diagnostics = RouterDiagnostics(
            block_id=self.block_id,
            logits=routing.logits,
            probabilities=routing.probabilities,
            topk_indices=routing.topk_indices,
            topk_weights=routing.topk_weights,
            valid_expert_mask=routing.valid_expert_mask,
            spatial_gates=routing.spatial_gates,
            branch_weights=routing.branch_weights,
            entropy=routing.entropy,
            importance=routing.importance,
            hard_load=routing.hard_load,
            auxiliary=auxiliary,
        )
        return MoEOutput(
            output,
            scaled_residual,
            routing,
            routing.spatial_gates,
            retained_outputs,
            diagnostics,
            {
                "execution": policy.mode,
                "uniform_to_soft": policy.uniform_to_soft,
                "soft_to_topk": policy.soft_to_topk,
            },
        )

    def _forward_shared_v2(
        self,
        tensor: Tensor,
        router_context: RouterContext,
        expert_context: ExpertContext,
        sparse_execution: bool | None,
        return_expert_outputs: bool,
    ) -> MoEOutput:
        adapter = self.adapter
        canonical_feature = adapter.feature_in(tensor)
        source_a = self.site_evidence_builder.project_optional_source(
            adapter.source_in, expert_context.source_a_feature
        )
        source_b = self.site_evidence_builder.project_optional_source(
            adapter.source_in, expert_context.source_b_feature
        )
        stage_guidance = None
        if self.semantic_guidance_source == "stage_feature":
            stage_guidance = expert_context.stage_guidance_feature
            if stage_guidance is None:
                raise ValueError(
                    f"{self.block_id} requires stage-aligned guidance in Stage 5 mode"
                )
            if stage_guidance.shape != tensor.shape:
                raise ValueError(
                    f"{self.block_id} guidance must match its native site feature shape"
                )
            stage_guidance = adapter.project_guidance(stage_guidance)
        canonical_router_context = replace(
            router_context,
            feature=canonical_feature,
            source_a=source_a,
            source_b=source_b,
        )
        canonical_expert_context = replace(
            expert_context,
            source_a_feature=source_a,
            source_b_feature=source_b,
            stage_guidance_feature=stage_guidance,
        )
        canonical = self._forward_v2(
            canonical_feature,
            canonical_router_context,
            canonical_expert_context,
            sparse_execution,
            return_expert_outputs,
        )
        stage_residual = adapter.delta_out(canonical.residual)
        diagnostics = canonical.diagnostics
        assert diagnostics is not None
        diagnostics.auxiliary.update(
            {
                "shared_pool_enabled": True,
                "expert_dim": canonical_feature.shape[1],
                "moe_residual_to_input_ratio": self._rms(stage_residual)
                / self._rms(tensor).clamp_min(torch.finfo(torch.float32).eps),
            }
        )
        return MoEOutput(
            tensor + stage_residual,
            stage_residual,
            canonical.router,
            None,
            canonical.expert_outputs,
            diagnostics,
            {**canonical.aux, "shared_pool_enabled": True},
        )

    def _forward_v2(
        self,
        tensor: Tensor,
        router_context: RouterContext,
        expert_context: ExpertContext,
        sparse_execution: bool | None,
        return_expert_outputs: bool,
    ) -> MoEOutput:
        if self.routing_mode == "spatial_soft":
            return self._forward_spatial_v2(
                tensor,
                router_context,
                expert_context,
                sparse_execution,
                return_expert_outputs,
            )
        policy = self._resolve_policy(sparse_execution)
        retained = {} if return_expert_outputs else None
        residual_rms: dict[str, Tensor] = {}
        contribution_rms = {
            name: tensor.detach().float().new_zeros(())
            for name in (ExpertType.COMMON.value, *self.expert_names)
        }
        regularizers: dict[str, list[Tensor]] = {}

        common_residual = torch.zeros_like(tensor)
        assignments = 0
        if self.common_always_on:
            common = self._common_expert(tensor)
            if not common.valid_samples.all():
                raise RuntimeError("The v2 always-on common expert became unavailable")
            common_residual = common.residual
            assignments += tensor.shape[0]
            residual_rms[ExpertType.COMMON.value] = self._rms(common_residual)
            contribution_rms[ExpertType.COMMON.value] = residual_rms[
                ExpertType.COMMON.value
            ]
            if retained is not None:
                retained[ExpertType.COMMON.value] = common
        shared = tensor + self.common_scale * common_residual

        learned_routing = self.router(replace(router_context, feature=shared))
        routing = self._apply_routing_override(learned_routing)
        evidence = self._build_site_evidence(
            shared,
            replace(router_context, feature=shared),
            expert_context,
            shared.shape[-2:],
        )
        weights = routing.probabilities.to(shared)
        if self.routing_override == "single:common":
            weights = torch.zeros_like(weights)

        specialist = torch.zeros_like(shared)
        for index, expert in enumerate(self.experts):
            selected = (weights[:, index].detach() != 0) & (
                routing.valid_expert_mask[:, index]
            )
            indices = torch.nonzero(selected, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            assignments += indices.numel()
            selected_evidence = evidence.expert.index_select(indices)
            expert_evidence = (
                selected_evidence.for_expert(ExpertType.parse(self.expert_names[index]))
                if isinstance(selected_evidence, Stage4ExpertEvidence)
                else selected_evidence
            )
            expert_output = expert(
                shared.index_select(0, indices), expert_evidence
            )
            residual = expert_output.residual * expert_output.valid_samples[
                :, None, None, None
            ]
            name = self.expert_names[index]
            residual_rms[name] = self._rms(residual)
            contribution = residual * weights.index_select(0, indices)[
                :, index : index + 1, None, None
            ]
            contribution_rms[name] = (
                contribution.detach().float().square().sum() / tensor.numel()
            ).sqrt()
            specialist = specialist.index_add(0, indices, contribution)
            if self._regularizers_enabled(expert_output.expert, policy):
                for key, value in self._expert_regularizers(
                    expert_output, selected_evidence
                ).items():
                    regularizers.setdefault(key, []).append(value)
            if retained is not None:
                retained[name] = self._restore_output(shared, indices, expert_output)

        output = shared + self.specialist_scale * specialist
        total_residual = output - tensor
        probabilities = routing.probabilities.detach().float()
        top = probabilities.topk(min(2, probabilities.shape[1]), dim=1).values
        auxiliary = {
            **routing.auxiliary,
            **evidence.diagnostics,
            "architecture_version": self.architecture_version,
            "routing_mode": self.routing_mode,
            "common_always_on": self.common_always_on,
            "expert_names": self.expert_names,
            "expert_sample_assignments": assignments,
            "expert_residual_rms": residual_rms,
            "expert_weighted_contribution_rms": contribution_rms,
            "common_scale_rms": self._rms(self.common_scale),
            "specialist_scale_rms": self._rms(self.specialist_scale),
            "specialist_mixture_weights": weights.detach(),
            "moe_residual_to_input_ratio": self._rms(total_residual)
            / self._rms(tensor).clamp_min(torch.finfo(torch.float32).eps),
            "router/top1_margin": (top[:, 0] - top[:, 1]).mean(),
            "router/probability_std": probabilities.std(1, unbiased=False).mean(),
            "router/entropy": routing.entropy.detach().float().mean(),
            "routing_override": self.routing_override,
        }
        if regularizers:
            auxiliary["expert_regularizers"] = {
                name: torch.stack(values).mean()
                for name, values in regularizers.items()
            }
        if retained is not None:
            auxiliary["expert_residuals"] = {
                name: value.residual for name, value in retained.items()
            }
        diagnostics = RouterDiagnostics(
            self.block_id,
            routing.logits,
            routing.probabilities,
            routing.topk_indices,
            routing.topk_weights,
            routing.valid_expert_mask,
            branch_weights=routing.branch_weights,
            entropy=routing.entropy,
            importance=routing.importance,
            hard_load=routing.hard_load,
            auxiliary=auxiliary,
        )
        return MoEOutput(
            output,
            total_residual,
            routing,
            None,
            retained,
            diagnostics,
            {"architecture_version": "v2", "execution": self.routing_mode},
        )

    def _forward_spatial_v2(
        self,
        tensor: Tensor,
        router_context: RouterContext,
        expert_context: ExpertContext,
        sparse_execution: bool | None,
        return_expert_outputs: bool,
    ) -> MoEOutput:
        policy = self._resolve_policy(sparse_execution)
        retained = {} if return_expert_outputs else None
        residual_rms: dict[str, Tensor] = {}
        contribution_rms = {
            name: tensor.detach().float().new_zeros(())
            for name in (ExpertType.COMMON.value, *self.expert_names)
        }
        regularizers: dict[str, list[Tensor]] = {}

        common = self._common_expert(tensor)
        if not common.valid_samples.all():
            raise RuntimeError("The v2 always-on common expert became unavailable")
        common_residual = common.residual
        residual_rms[ExpertType.COMMON.value] = self._rms(common_residual)
        contribution_rms[ExpertType.COMMON.value] = residual_rms[ExpertType.COMMON.value]
        if retained is not None:
            retained[ExpertType.COMMON.value] = common
        shared = tensor + self.common_scale * common_residual

        evidence = self._build_site_evidence(
            shared,
            replace(router_context, feature=shared),
            expert_context,
            self._spatial_grid_size(shared),
        )
        if self.functional_expert_version == "stage4b":
            if not isinstance(evidence.expert, Stage4ExpertEvidence):
                raise TypeError("Stage 4B requires typed Stage4ExpertEvidence")
            semantic_guidance = (
                expert_context.stage_guidance_feature
                if self.semantic_guidance_source == "stage_feature"
                else self.adapter.project_guidance(
                    evidence.expert.semantic.guidance_feature
                )
            )
            if semantic_guidance is None:
                raise ValueError(
                    f"{self.block_id} requires canonical stage guidance in Stage 5 mode"
                )
            evidence = replace(
                evidence,
                expert=evidence.expert.with_semantic_guidance(
                    semantic_guidance
                ),
            )
        routing = self._apply_routing_override(self.router(evidence))
        weights = functional.interpolate(
            routing.probabilities.to(shared),
            size=shared.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        weights = weights / weights.sum(1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        if self.routing_override == "single:common":
            weights = torch.zeros_like(weights)

        specialist = torch.zeros_like(shared)
        assignments = tensor.shape[0]
        for index, expert in enumerate(self.experts):
            selected = (weights[:, index].detach().amax((-2, -1)) != 0) & (
                routing.valid_expert_mask[:, index]
            )
            indices = torch.nonzero(selected, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            assignments += indices.numel()
            selected_evidence = evidence.expert.index_select(indices)
            expert_evidence = (
                selected_evidence.for_expert(ExpertType.parse(self.expert_names[index]))
                if isinstance(selected_evidence, Stage4ExpertEvidence)
                else selected_evidence
            )
            expert_output = expert(
                shared.index_select(0, indices), expert_evidence
            )
            residual = expert_output.residual * expert_output.valid_samples[
                :, None, None, None
            ]
            name = self.expert_names[index]
            residual_rms[name] = self._rms(residual)
            contribution = residual * weights.index_select(0, indices)[
                :, index : index + 1
            ]
            contribution_rms[name] = (
                contribution.detach().float().square().sum() / tensor.numel()
            ).sqrt()
            specialist = specialist.index_add(0, indices, contribution)
            if self._regularizers_enabled(expert_output.expert, policy):
                for key, value in self._expert_regularizers(
                    expert_output, selected_evidence
                ).items():
                    regularizers.setdefault(key, []).append(value)
            if retained is not None:
                retained[name] = self._restore_output(shared, indices, expert_output)

        output = shared + self.specialist_scale * specialist
        total_residual = output - tensor
        probabilities = routing.probabilities.detach().float()
        top = probabilities.topk(min(2, probabilities.shape[1]), dim=1).values
        auxiliary = {
            **routing.auxiliary,
            **evidence.diagnostics,
            "architecture_version": self.architecture_version,
            "routing_mode": self.routing_mode,
            "common_always_on": self.common_always_on,
            "expert_names": self.expert_names,
            "expert_sample_assignments": assignments,
            "expert_residual_rms": residual_rms,
            "expert_weighted_contribution_rms": contribution_rms,
            "common_scale_rms": self._rms(self.common_scale),
            "specialist_scale_rms": self._rms(self.specialist_scale),
            "specialist_mixture_weights": weights.detach(),
            "moe_residual_to_input_ratio": self._rms(total_residual)
            / self._rms(tensor).clamp_min(torch.finfo(torch.float32).eps),
            "router/top1_margin": (top[:, 0] - top[:, 1]).mean(),
            "router/probability_std": probabilities.std(1, unbiased=False).mean(),
            "router/entropy": routing.entropy.detach().float().mean(),
            "routing_override": self.routing_override,
        }
        if regularizers:
            auxiliary["expert_regularizers"] = {
                name: torch.stack(values).mean()
                for name, values in regularizers.items()
            }
        if retained is not None:
            auxiliary["expert_residuals"] = {
                name: value.residual.detach() for name, value in retained.items()
            }
        diagnostics = RouterDiagnostics(
            self.block_id,
            routing.logits.detach(),
            routing.probabilities.detach(),
            None,
            None,
            routing.valid_expert_mask,
            entropy=routing.entropy.detach(),
            importance=(
                routing.importance.detach()
                if routing.importance is not None
                else None
            ),
            hard_load=(
                routing.hard_load.detach() if routing.hard_load is not None else None
            ),
            auxiliary=auxiliary,
        )
        return MoEOutput(
            output,
            total_residual,
            routing,
            None,
            retained,
            diagnostics,
            {"architecture_version": "v2", "execution": self.routing_mode},
        )

    @staticmethod
    def _rms(value: Tensor) -> Tensor:
        return value.detach().float().square().mean().sqrt()

    def _spatial_grid_size(self, feature: Tensor) -> tuple[int, int]:
        if not isinstance(self.router, SiteSpatialRouter):
            raise TypeError("Spatial grid size is only defined for SiteSpatialRouter")
        height, width = feature.shape[-2:]
        return (
            (height + self.router.patch_size - 1) // self.router.patch_size,
            (width + self.router.patch_size - 1) // self.router.patch_size,
        )

    def _build_site_evidence(
        self,
        feature: Tensor,
        router_context: RouterContext,
        expert_context: ExpertContext,
        grid_size: tuple[int, int],
    ) -> SiteEvidence:
        return self.site_evidence_builder.build(
            feature, router_context, expert_context, grid_size
        )

    def set_execution_policy(self, policy: MoEExecutionPolicy | None) -> None:
        self.execution_policy = policy

    def set_routing_override(self, mode: str, seed: int = 3407) -> None:
        normalized = self._normalize_routing_override(mode)
        if normalized.startswith("single:"):
            expert_name = normalized.split(":", 1)[1]
            if not self._supports_single_expert(expert_name):
                available_names = list(self.expert_names)
                if self.architecture_version == "v2" and self.common_always_on:
                    available_names.insert(0, ExpertType.COMMON.value)
                available = ", ".join(available_names)
                raise ValueError(
                    f"MoE block {self.block_id!r} does not contain expert "
                    f"{expert_name!r}; available experts: {available}"
                )
        self.routing_override = normalized
        self.routing_override_seed = int(seed)

    def _supports_single_expert(self, expert_name: str) -> bool:
        return expert_name in self.expert_names or (
            expert_name == ExpertType.COMMON.value
            and self.architecture_version == "v2"
            and self.common_always_on
        )

    def clear_routing_override(self) -> None:
        self.routing_override = "learned"

    @classmethod
    def _normalize_routing_override(cls, mode: str) -> str:
        normalized = mode.strip().lower()
        if normalized in cls.routing_override_modes:
            return normalized
        if normalized.startswith("single:"):
            expert_name = normalized.split(":", 1)[1]
            try:
                return f"single:{ExpertType.parse(expert_name).value}"
            except ContractError as error:
                raise ValueError(str(error)) from error
        choices = ", ".join((*cls.routing_override_modes, "single:<expert>"))
        raise ValueError(f"Unknown routing override {mode!r}; expected one of {choices}")

    def _apply_routing_override(self, routing: RouterOutput) -> RouterOutput:
        mode = self.routing_override
        if mode == "learned":
            return routing
        if mode == "single:common" and self.architecture_version == "v2":
            return routing
        if routing.probabilities.ndim == 4:
            return self._apply_spatial_routing_override(routing)

        valid = routing.valid_expert_mask
        auxiliary = dict(routing.auxiliary)
        if mode == "uniform":
            probabilities = valid.to(routing.probabilities)
            probabilities = probabilities / probabilities.sum(1, keepdim=True)
            logits = torch.zeros_like(routing.logits).masked_fill(~valid, -torch.inf)
            spatial_gates = None
        elif mode == "shuffled":
            if not torch.equal(valid, valid[:1].expand_as(valid)):
                raise RuntimeError(
                    "Shuffled routing requires a consistent expert validity mask "
                    "within each batch"
                )
            valid_indices = torch.nonzero(valid[0], as_tuple=False).flatten()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.routing_override_seed)
            order = torch.randperm(valid_indices.numel(), generator=generator).to(
                valid_indices.device
            )
            permutation = torch.arange(
                valid.shape[1], device=valid.device, dtype=torch.long
            )
            permutation[valid_indices] = valid_indices.index_select(0, order)
            probabilities = routing.probabilities.index_select(1, permutation)
            logits = routing.logits.index_select(1, permutation)
            spatial_gates = (
                routing.spatial_gates.index_select(1, permutation)
                if routing.spatial_gates is not None
                else None
            )
            auxiliary["routing_permutation"] = permutation.detach()
        else:
            expert_name = mode.split(":", 1)[1]
            expert_index = self.expert_names.index(expert_name)
            invalid_samples = torch.nonzero(
                ~valid[:, expert_index], as_tuple=False
            ).flatten()
            if invalid_samples.numel():
                sample_list = ", ".join(
                    str(index) for index in invalid_samples.detach().cpu().tolist()
                )
                raise ValueError(
                    f"Routing override {mode!r} is invalid for MoE block "
                    f"{self.block_id!r} on sample(s) {sample_list}"
                )
            probabilities = torch.zeros_like(routing.probabilities)
            probabilities[:, expert_index] = 1
            logits = torch.full_like(routing.logits, -torch.inf)
            logits[:, expert_index] = 0
            spatial_gates = None

        active = (probabilities > 0) & valid
        top_k = min(routing.topk_indices.shape[1], int(active.sum(1).min().item()))
        selected, indices = probabilities.masked_fill(~valid, -torch.inf).topk(
            top_k, dim=1
        )
        topk_weights = selected / selected.sum(1, keepdim=True).clamp_min(1e-8)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1)
        importance = probabilities.mean(0)
        hard_load = (
            importance
            if mode == "uniform" or mode.startswith("single:")
            else functional.one_hot(indices, probabilities.shape[1])
            .float()
            .mean((0, 1))
        )
        return RouterOutput(
            logits=logits,
            probabilities=probabilities,
            topk_indices=indices,
            topk_weights=topk_weights,
            valid_expert_mask=valid,
            spatial_gates=spatial_gates,
            branch_weights=routing.branch_weights,
            entropy=entropy,
            importance=importance,
            hard_load=hard_load,
            auxiliary=auxiliary,
        )

    def _apply_spatial_routing_override(self, routing: RouterOutput) -> RouterOutput:
        mode, valid = self.routing_override, routing.valid_expert_mask
        auxiliary = dict(routing.auxiliary)
        if mode == "uniform":
            probabilities = valid.to(routing.probabilities)
            probabilities = probabilities / probabilities.sum(1, keepdim=True)
            probabilities = probabilities[:, :, None, None].expand_as(
                routing.probabilities
            )
            logits = torch.zeros_like(routing.logits).masked_fill(
                ~valid[:, :, None, None], -torch.inf
            )
        elif mode == "shuffled":
            if not torch.equal(valid, valid[:1].expand_as(valid)):
                raise RuntimeError(
                    "Shuffled routing requires a consistent expert validity mask within each batch"
                )
            valid_indices = torch.nonzero(valid[0], as_tuple=False).flatten()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.routing_override_seed)
            permutation = torch.arange(valid.shape[1], device=valid.device)
            permutation[valid_indices] = valid_indices.index_select(
                0, torch.randperm(valid_indices.numel(), generator=generator).to(valid.device)
            )
            probabilities = routing.probabilities.index_select(1, permutation)
            logits = routing.logits.index_select(1, permutation)
            auxiliary["routing_permutation"] = permutation.detach()
        else:
            expert_name = mode.split(":", 1)[1]
            expert_index = self.expert_names.index(expert_name)
            invalid = torch.nonzero(~valid[:, expert_index], as_tuple=False).flatten()
            if invalid.numel():
                sample_list = ", ".join(map(str, invalid.detach().cpu().tolist()))
                raise ValueError(
                    f"Routing override {mode!r} is invalid for MoE block "
                    f"{self.block_id!r} on sample(s) {sample_list}"
                )
            probabilities = torch.zeros_like(routing.probabilities)
            probabilities[:, expert_index] = 1
            logits = torch.full_like(routing.logits, -torch.inf)
            logits[:, expert_index] = 0

        entropy_map = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1)
        top = probabilities.topk(min(2, probabilities.shape[1]), dim=1).values
        importance = probabilities.mean((0, 2, 3))
        hard_load = functional.one_hot(
            probabilities.argmax(1), probabilities.shape[1]
        ).float().mean((0, 1, 2))
        auxiliary.update(
            {
                "entropy_map": entropy_map.detach(),
                "router/top1_margin": (top[:, 0] - top[:, 1]).mean().detach(),
                "router/spatial_variance": probabilities.var(
                    dim=(-2, -1), unbiased=False
                ).mean().detach(),
            }
        )
        return RouterOutput(
            logits,
            probabilities,
            None,
            None,
            valid,
            entropy=entropy_map.mean(),
            importance=importance,
            hard_load=hard_load,
            auxiliary=auxiliary,
        )

    def _resolve_policy(self, sparse_execution: bool | None) -> MoEExecutionPolicy:
        if sparse_execution is not None:
            return MoEExecutionPolicy.legacy(sparse_execution)
        if self.execution_policy is not None and self.training:
            return self.execution_policy
        execution = self.train_execution if self.training else self.inference_execution
        return MoEExecutionPolicy.legacy(execution == "sparse_batch")

    def _mixture_weights(
        self,
        tensor: Tensor,
        routing: RouterOutput,
        policy: MoEExecutionPolicy,
    ) -> tuple[Tensor, Tensor | None]:
        valid = routing.valid_expert_mask.to(tensor)
        uniform = valid / valid.sum(1, keepdim=True).clamp_min(1)
        topk = tensor.new_zeros(tensor.shape[0], len(self.experts))
        topk.scatter_(1, routing.topk_indices, routing.topk_weights)
        soft = routing.probabilities.to(tensor)

        if policy.mode == "dense_uniform":
            weights, gates = uniform, None
        elif policy.mode == "dense_annealed":
            beta = min(max(policy.uniform_to_soft, 0.0), 1.0)
            alpha = min(max(policy.soft_to_topk, 0.0), 1.0)
            weights = (1 - beta) * uniform + beta * soft
            weights = (1 - alpha) * weights + alpha * topk
            gates = routing.spatial_gates
            if gates is not None and policy.spatial_gate_scale < 1:
                scale = max(policy.spatial_gate_scale, 0.0)
                gates = 1 + scale * (gates - 1)
        elif policy.mode == "expert_refresh":
            routed = min(max(policy.refresh_routed_fraction, 0.0), 1.0)
            weights = routed * soft.detach() + (1 - routed) * uniform
            gates = None
        else:
            weights, gates = topk, routing.spatial_gates

        if policy.detach_router:
            weights = weights.detach()
            gates = gates.detach() if gates is not None else None
        return weights, gates

    @staticmethod
    def _expert_indices(
        tensor: Tensor,
        routing: RouterOutput,
        expert_index: int,
        policy: MoEExecutionPolicy,
    ) -> Tensor:
        if policy.sparse:
            selected = (routing.topk_indices == expert_index).any(dim=1)
        else:
            selected = routing.valid_expert_mask[:, expert_index]
        return torch.nonzero(selected, as_tuple=False).flatten()

    @staticmethod
    def _restore_output(
        tensor: Tensor, indices: Tensor, output: ExpertOutput
    ) -> ExpertOutput:
        residual = torch.zeros_like(tensor).index_copy(0, indices, output.residual)
        valid = torch.zeros(
            tensor.shape[0], dtype=torch.bool, device=tensor.device
        ).index_copy(0, indices, output.valid_samples)
        return ExpertOutput(
            residual,
            output.expert,
            valid,
            diagnostics=output.diagnostics,
        )

    @staticmethod
    def _regularizers_enabled(
        expert: ExpertType, policy: MoEExecutionPolicy
    ) -> bool:
        if expert in {
            ExpertType.LOW_FREQUENCY,
            ExpertType.DETAIL,
            ExpertType.SEMANTIC,
        }:
            return policy.compute_frequency_regularizers
        if expert is ExpertType.INFRARED_SALIENCY:
            return policy.compute_infrared_regularizers
        return False

    @staticmethod
    def _expert_regularizers(
        output: ExpertOutput,
        evidence: ExpertEvidence | Stage4ExpertEvidence | ExpertContext,
    ) -> dict[str, Tensor]:
        if isinstance(evidence, ExpertContext):
            tensor = output.residual
            evidence = SiteEvidenceBuilder(
                HaarDWT2D().to(tensor), ExpertAvailabilityPolicy((output.expert.value,))
            ).build(
                tensor,
                RouterContext(
                    tensor,
                    evidence.task,
                    evidence.modality_a,
                    evidence.modality_b,
                    low_energy=tensor.new_zeros(tensor.shape[0]),
                    high_energy=tensor.new_zeros(tensor.shape[0]),
                    source_a=evidence.source_a_feature,
                    source_b=evidence.source_b_feature,
                    semantic_boundary=evidence.semantic_boundary,
                    semantic_uncertainty=evidence.semantic_uncertainty,
                ),
                evidence,
                tensor.shape[-2:],
            ).expert
        residual = output.residual
        if output.expert is ExpertType.LOW_FREQUENCY:
            smooth = functional.avg_pool2d(residual, 5, stride=1, padding=2)
            return {"frequency/low_leakage": (residual - smooth).abs().mean()}
        if output.expert is ExpertType.DETAIL:
            smooth = functional.avg_pool2d(residual, 5, stride=1, padding=2)
            return {"frequency/detail_leakage": smooth.abs().mean()}
        if output.expert is ExpertType.SEMANTIC:
            gate_logits = output.diagnostics.get("gate_logits")
            if isinstance(evidence, Stage4ExpertEvidence) and isinstance(
                gate_logits, Tensor
            ):
                return {
                    "frequency/semantic_boundary": functional.binary_cross_entropy_with_logits(
                        gate_logits, evidence.semantic.need.to(gate_logits).clamp(0, 1)
                    )
                }
            if isinstance(gate_logits, Tensor) and evidence.semantic_boundary_available:
                target = evidence.semantic_maps[:, :1].to(gate_logits).clamp(0, 1)
                return {
                    "frequency/semantic_boundary": (
                        functional.binary_cross_entropy_with_logits(
                            gate_logits, target
                        )
                    )
                }
        if output.expert is ExpertType.INFRARED_SALIENCY:
            saliency = output.diagnostics.get("saliency")
            if isinstance(evidence, Stage4ExpertEvidence) and isinstance(saliency, Tensor):
                target = (
                    evidence.infrared.advantage.value.detach()
                    .abs()
                    .mean(1, keepdim=True)
                )
                target = target / target.flatten(1).amax(1).clamp_min(1e-6)[
                    :, None, None, None
                ]
                return {"infrared/saliency_alignment": functional.l1_loss(saliency, target)}
            if isinstance(saliency, Tensor) and evidence.infrared_samples_valid.any():
                target = (evidence.infrared_feature - evidence.other_feature).abs().mean(
                    1, keepdim=True
                )
                maximum = target.flatten(1).amax(1).clamp_min(1e-6)
                target = (target / maximum[:, None, None, None]).detach()
                return {
                    "infrared/saliency_alignment": functional.l1_loss(saliency, target)
                }
        return {}

    @property
    def experts(self) -> tuple[nn.Module, ...]:
        if self.shared_pool_enabled:
            bank = self.shared_expert_bank
            assert bank is not None
            return tuple(bank.specialists[name] for name in self.expert_names)
        return tuple(self.expert_pool)

    def _expert_by_name(self, name: str) -> FunctionalExpert:
        if self.shared_pool_enabled:
            bank = self.shared_expert_bank
            assert bank is not None
            return bank.specialists[name]  # type: ignore[return-value]
        return self.expert_pool.get_expert(name)

    @property
    def shared_expert_bank(self) -> SharedExpertBank | None:
        return self._shared_expert_bank

    def _common_expert(self, tensor: Tensor) -> ExpertOutput:
        bank = self.shared_expert_bank
        if bank is not None:
            return bank.common(tensor)
        return self.common_expert(tensor)


class SharedExpertMoESite(FunctionalMoEBlock):
    """A v2 global-soft MoE site backed by a root-owned shared expert bank."""

    def __init__(
        self,
        stage_channels: int,
        config: MoEConfig,
        block_id: str,
        task_embedding: TaskEmbedding,
        shared_expert_bank: SharedExpertBank,
    ) -> None:
        super().__init__(
            stage_channels,
            config,
            block_id,
            task_embedding,
            shared_expert_bank,
        )


# The design document's canonical name; the older name remains supported.
TaskFrequencyMoEBlock = FunctionalMoEBlock


def iter_moe_blocks(model: nn.Module) -> Iterator[FunctionalMoEBlock]:
    """Yield every unique core and feedback MoE block in a model or DDP wrapper."""
    return (
        module for module in model.modules() if isinstance(module, FunctionalMoEBlock)
    )


def set_moe_routing_override(
    model: nn.Module, mode: str, seed: int = 3407
) -> None:
    """Apply one validated runtime routing override to every MoE block."""
    normalized = FunctionalMoEBlock._normalize_routing_override(mode)
    blocks = tuple(iter_moe_blocks(model))
    if normalized.startswith("single:"):
        expert_name = normalized.split(":", 1)[1]
        missing = [
            block.block_id
            for block in blocks
            if not block._supports_single_expert(expert_name)
        ]
        if missing:
            raise ValueError(
                f"Expert {expert_name!r} is missing from MoE block(s): "
                + ", ".join(missing)
            )
    for block in blocks:
        block.set_routing_override(normalized, seed)


def clear_moe_routing_override(model: nn.Module) -> None:
    """Restore learned routing for every MoE block."""
    for block in iter_moe_blocks(model):
        block.clear_routing_override()
