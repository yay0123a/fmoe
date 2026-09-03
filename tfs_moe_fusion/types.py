"""TFS-MoE-Fusion consolidated implementation."""

from __future__ import annotations


class TFSFusionError(RuntimeError):
    """Base error for the project."""


class ConfigurationError(TFSFusionError, ValueError):
    """Raised when a configuration violates the permanent schema."""


class ContractError(TFSFusionError, ValueError):
    """Raised when data does not satisfy a typed interface contract."""


from enum import Enum


class _StringEnum(str, Enum):
    @classmethod
    def parse(cls, value: str | _StringEnum):
        if isinstance(value, cls):
            return value
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError as error:
            legal = ", ".join(item.value for item in cls)
            raise ContractError(
                f"Unknown {cls.__name__} {value!r}. Legal values: {legal}"
            ) from error


class TaskType(_StringEnum):
    VIF = "vif"
    MFIF = "mfif"
    SEG = "seg"

    @property
    def index(self) -> int:
        return {TaskType.VIF: 0, TaskType.MFIF: 1, TaskType.SEG: 2}[self]


class ModalityType(_StringEnum):
    VISIBLE_RGB = "visible_rgb"
    GENERIC_RGB = "generic_rgb"
    INFRARED_GRAY = "infrared_gray"
    GENERIC_GRAY = "generic_gray"

    @property
    def channels(self) -> int:
        if self in {ModalityType.VISIBLE_RGB, ModalityType.GENERIC_RGB}:
            return 3
        return 1

    @property
    def is_infrared(self) -> bool:
        return self is ModalityType.INFRARED_GRAY


class ExpertType(_StringEnum):
    COMMON = "common"
    LOW_FREQUENCY = "low_frequency"
    DETAIL = "detail"
    SEMANTIC = "semantic"
    INFRARED_SALIENCY = "infrared_saliency"


from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass(slots=True)
class SourceBatch:
    image: Tensor
    modality: ModalityType

    def __post_init__(self) -> None:
        self.modality = ModalityType.parse(self.modality)
        _validate_image(self.image, "source image", batched=True)
        if self.image.shape[1] != self.modality.channels:
            raise ContractError(
                f"{self.modality.value} requires {self.modality.channels} channel(s), "
                f"got {self.image.shape[1]}"
            )

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> SourceBatch:
        return SourceBatch(
            self.image.to(device, non_blocking=non_blocking), self.modality
        )


@dataclass(slots=True)
class FusionBatch:
    source_a: SourceBatch
    source_b: SourceBatch
    task: TaskType
    sample_ids: tuple[str, ...]
    target: Tensor | None = None
    focus_target: Tensor | None = None
    segmentation_target: Tensor | None = None
    metadata: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.task = TaskType.parse(self.task)
        shape_a = self.source_a.image.shape
        shape_b = self.source_b.image.shape
        if shape_a[0] != shape_b[0] or shape_a[-2:] != shape_b[-2:]:
            raise ContractError(
                "Both sources must have equal batch and spatial dimensions; "
                f"got {tuple(shape_a)} and {tuple(shape_b)}"
            )
        batch_size = shape_a[0]
        if len(self.sample_ids) != batch_size:
            raise ContractError(
                f"sample_ids has {len(self.sample_ids)} entries for batch size {batch_size}"
            )
        if self.metadata and len(self.metadata) != batch_size:
            raise ContractError("metadata must be empty or have one item per sample")

        for name, value in (
            ("target", self.target),
            ("focus_target", self.focus_target),
            ("segmentation_target", self.segmentation_target),
        ):
            if value is not None:
                _validate_supervision(value, name, batch_size, shape_a[-2:])

        modalities = (self.source_a.modality, self.source_b.modality)
        has_ir = ModalityType.INFRARED_GRAY in modalities
        if self.task in {TaskType.VIF, TaskType.SEG} and not has_ir:
            raise ContractError(f"{self.task.value} requires one real infrared source")
        if self.task in {TaskType.VIF, TaskType.SEG} and (
            modalities.count(ModalityType.VISIBLE_RGB) != 1
            or modalities.count(ModalityType.INFRARED_GRAY) != 1
        ):
            raise ContractError(
                f"{self.task.value} requires exactly one visible RGB and one real "
                "infrared source"
            )
        if self.task is TaskType.MFIF and has_ir:
            raise ContractError("MFIF cannot use an infrared source")

    @property
    def batch_size(self) -> int:
        return int(self.source_a.image.shape[0])

    @property
    def spatial_size(self) -> tuple[int, int]:
        return tuple(self.source_a.image.shape[-2:])  # type: ignore[return-value]

    @property
    def has_infrared(self) -> bool:
        return self.source_a.modality.is_infrared or self.source_b.modality.is_infrared

    @property
    def visible_source(self) -> SourceBatch:
        for source in (self.source_a, self.source_b):
            if source.modality is ModalityType.VISIBLE_RGB:
                return source
        raise ContractError(f"{self.task.value} batch has no visible RGB source")

    @property
    def infrared_source(self) -> SourceBatch:
        for source in (self.source_a, self.source_b):
            if source.modality is ModalityType.INFRARED_GRAY:
                return source
        raise ContractError(f"{self.task.value} batch has no real infrared source")

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> FusionBatch:
        move = (
            lambda value: value.to(device, non_blocking=non_blocking)
            if value is not None
            else None
        )
        return FusionBatch(
            source_a=self.source_a.to(device, non_blocking=non_blocking),
            source_b=self.source_b.to(device, non_blocking=non_blocking),
            task=self.task,
            sample_ids=self.sample_ids,
            target=move(self.target),
            focus_target=move(self.focus_target),
            segmentation_target=move(self.segmentation_target),
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class FusionSample:
    source_a: Tensor
    modality_a: ModalityType
    source_b: Tensor
    modality_b: ModalityType
    task: TaskType
    sample_id: str
    target: Tensor | None = None
    focus_target: Tensor | None = None
    segmentation_target: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_image(self.source_a, "sample source_a", batched=False)
        _validate_image(self.source_b, "sample source_b", batched=False)
        if self.source_a.shape[-2:] != self.source_b.shape[-2:]:
            raise ContractError("Sample source images must have equal spatial size")
        if self.source_a.shape[0] != self.modality_a.channels:
            raise ContractError("sample source_a channel/modality mismatch")
        if self.source_b.shape[0] != self.modality_b.channels:
            raise ContractError("sample source_b channel/modality mismatch")


def _validate_image(tensor: Tensor, name: str, batched: bool) -> None:
    expected_rank = 4 if batched else 3
    if not isinstance(tensor, Tensor) or tensor.ndim != expected_rank:
        raise ContractError(f"{name} must be a rank-{expected_rank} torch.Tensor")
    if not tensor.is_floating_point():
        raise ContractError(f"{name} must be floating point")
    if tensor.shape[-2] <= 0 or tensor.shape[-1] <= 0:
        raise ContractError(f"{name} must have non-empty spatial dimensions")
    if not torch.isfinite(tensor).all():
        raise ContractError(f"{name} contains NaN or infinity")


def _validate_supervision(
    tensor: Tensor, name: str, batch_size: int, spatial_size: tuple[int, int]
) -> None:
    if not isinstance(tensor, Tensor) or tensor.ndim not in {3, 4}:
        raise ContractError(f"{name} must be rank 3 or 4")
    if tensor.shape[0] != batch_size or tuple(tensor.shape[-2:]) != tuple(spatial_size):
        raise ContractError(f"{name} does not match batch/spatial dimensions")


from dataclasses import dataclass, field


@dataclass(slots=True)
class SpectralStatistics:
    stage: str
    low_energy: Tensor
    high_energy: Tensor
    radial_energy: Tensor | None = None
    dwt_energy: Tensor | None = None
    mid_energy: Tensor | None = None

    @property
    def fft_ring_energy(self) -> Tensor | None:
        return self.radial_energy


@dataclass(slots=True)
class RouterDiagnostics:
    block_id: str
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

    def __post_init__(self) -> None:
        if self.logits.ndim not in {2, 4} or self.probabilities.shape != self.logits.shape:
            raise ContractError(
                "Router logits/probabilities must be equal [B,E] or [B,E,H,W] tensors"
            )
        batch, experts = self.logits.shape[:2]
        expected_mask = (
            (batch, experts)
            if self.logits.ndim == 2
            else (batch, experts, 1, 1)
        )
        if self.valid_expert_mask.shape != expected_mask:
            raise ContractError("Router valid_expert_mask shape does not match routing")
        if self.valid_expert_mask.dtype is not torch.bool:
            raise ContractError("Router valid_expert_mask must be boolean")
        if (self.topk_indices is None) != (self.topk_weights is None):
            raise ContractError("Router Top-k indices and weights must be both set or none")
        if self.logits.ndim == 2:
            if self.topk_indices is None or self.topk_weights is None:
                raise ContractError("Global router diagnostics require Top-k values")
            if (
                self.topk_indices.ndim != 2
                or self.topk_weights.shape != self.topk_indices.shape
                or self.topk_indices.shape[0] != batch
            ):
                raise ContractError(
                    "Router Top-k indices/weights must be equal [B,K] tensors"
                )
        elif self.topk_indices is not None:
            raise ContractError("Spatial router diagnostics must not carry Top-k values")
        if torch.isnan(self.logits).any() or torch.isposinf(self.logits).any():
            raise ContractError("Router logits contain NaN or positive infinity")
        if not torch.isfinite(self.probabilities).all() or (
            self.topk_weights is not None and not torch.isfinite(self.topk_weights).all()
        ):
            raise ContractError("Router probabilities or Top-k weights are non-finite")
        if self.topk_indices is not None and not self.valid_expert_mask.gather(
            1, self.topk_indices
        ).all():
            raise ContractError("Router selected an unavailable expert")
        ones = torch.ones_like(self.probabilities.select(1, 0))
        if not torch.allclose(
            self.probabilities.sum(dim=1), ones, atol=1e-5, rtol=1e-5
        ):
            raise ContractError("Router probabilities must sum to one")
        if self.topk_weights is not None and not torch.allclose(
            self.topk_weights.sum(dim=1),
            torch.ones(batch, device=self.probabilities.device, dtype=self.probabilities.dtype),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ContractError("Router Top-k weights must sum to one")
        if self.spatial_gates is not None:
            if self.spatial_gates.ndim != 4 or self.spatial_gates.shape[:2] != (
                batch,
                experts,
            ):
                raise ContractError("Router spatial_gates must be [B,E,H,W]")
            if not torch.isfinite(self.spatial_gates).all():
                raise ContractError("Router spatial_gates are non-finite")
        if self.branch_weights is not None:
            if self.branch_weights.ndim != 2 or self.branch_weights.shape[0] != batch:
                raise ContractError("Router branch_weights must be [B,num_branches]")
            if not torch.isfinite(self.branch_weights).all():
                raise ContractError("Router branch_weights are non-finite")


@dataclass(slots=True)
class AuxiliaryOutputs:
    focus_reliability: Tensor | None = None
    focus_selection: Tensor | None = None
    focus_confidence: Tensor | None = None
    focus_boundary: Tensor | None = None
    semantic_probabilities: Tensor | None = None
    semantic_logits: Tensor | None = None
    semantic_uncertainty: Tensor | None = None
    semantic_boundary: Tensor | None = None

    def __post_init__(self) -> None:
        values = (
            self.focus_reliability,
            self.focus_selection,
            self.focus_confidence,
            self.focus_boundary,
            self.semantic_probabilities,
            self.semantic_logits,
            self.semantic_uncertainty,
            self.semantic_boundary,
        )
        for value in values:
            if value is not None and (
                value.ndim != 4 or not torch.isfinite(value).all()
            ):
                raise ContractError(
                    "Auxiliary predictions must be finite [B,C,H,W] tensors"
                )
        if self.focus_selection is not None:
            sums = self.focus_selection.sum(dim=1)
            if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5):
                raise ContractError("focus_selection must sum to one across sources")
        if self.focus_reliability is not None and self.focus_reliability.shape[1] != 2:
            raise ContractError("focus_reliability must contain two source channels")
        if self.semantic_probabilities is not None:
            batch, _, height, width = self.semantic_probabilities.shape
            sums = self.semantic_probabilities.sum(dim=1)
            if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4):
                raise ContractError("semantic probabilities must sum to one")
            for name, value in (
                ("semantic_uncertainty", self.semantic_uncertainty),
                ("semantic_boundary", self.semantic_boundary),
            ):
                if value is None or value.shape != (batch, 1, height, width):
                    raise ContractError(
                        f"{name} must be [B,1,H,W] with semantic probabilities"
                    )


@dataclass(slots=True)
class FusionOutput:
    fused: Tensor
    task: TaskType
    coarse: Tensor | None = None
    refinement: Tensor | None = None
    fused_y: Tensor | None = None
    coarse_y: Tensor | None = None
    refinement_y: Tensor | None = None
    spectral_statistics: tuple[SpectralStatistics, ...] = field(default_factory=tuple)
    router_diagnostics: tuple[RouterDiagnostics, ...] = field(default_factory=tuple)
    auxiliary: AuxiliaryOutputs | None = None
    focus: Any | None = None
    coarse_segmentation: Any | None = None
    segmentation: Any | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def focus_a(self) -> Tensor | None:
        return (
            self.auxiliary.focus_reliability[:, :1]
            if self.auxiliary and self.auxiliary.focus_reliability is not None
            else None
        )

    @property
    def focus_b(self) -> Tensor | None:
        return (
            self.auxiliary.focus_reliability[:, 1:]
            if self.auxiliary and self.auxiliary.focus_reliability is not None
            else None
        )

    @property
    def focus_confidence(self) -> Tensor | None:
        return self.auxiliary.focus_confidence if self.auxiliary else None

    def __post_init__(self) -> None:
        self.task = TaskType.parse(self.task)
        if not isinstance(self.fused, Tensor) or self.fused.ndim != 4:
            raise ContractError("FusionOutput.fused must be [B,C,H,W]")
        if not torch.isfinite(self.fused).all():
            raise ContractError("FusionOutput.fused contains NaN or infinity")
        if self.coarse is not None and self.coarse.shape != self.fused.shape:
            raise ContractError("FusionOutput.coarse must match fused shape")
        if self.refinement is not None and self.refinement.shape != self.fused.shape:
            raise ContractError("FusionOutput.refinement must match fused shape")
        for name, value in (
            ("fused_y", self.fused_y),
            ("coarse_y", self.coarse_y),
            ("refinement_y", self.refinement_y),
        ):
            if value is not None and (
                value.ndim != 4
                or value.shape[1] != 1
                or value.shape[0] != self.fused.shape[0]
                or value.shape[-2:] != self.fused.shape[-2:]
                or not torch.isfinite(value).all()
            ):
                raise ContractError(
                    f"FusionOutput.{name} must be finite [B,1,H,W] matching fused"
                )
        if self.task in {TaskType.VIF, TaskType.SEG} and any(
            value is None for value in (self.fused_y, self.coarse_y, self.refinement_y)
        ):
            raise ContractError("VIF/SEG FusionOutput requires typed Y predictions")
        if self.task is TaskType.MFIF and any(
            value is not None
            for value in (self.fused_y, self.coarse_y, self.refinement_y)
        ):
            raise ContractError("MFIF FusionOutput must remain an RGB-only prediction")

    @property
    def coarse_fused(self) -> Tensor | None:
        return self.coarse

    @property
    def router(self) -> tuple[RouterDiagnostics, ...]:
        return self.router_diagnostics

    @property
    def frequency(self) -> tuple[SpectralStatistics, ...]:
        return self.spectral_statistics

    @property
    def aux(self) -> dict[str, Any]:
        """Compatibility view for scalar/tensor debug diagnostics.

        Structured predicted maps live in ``auxiliary``; unclamped
        image and clamp ratios intentionally remain in this free-form mapping.
        """
        return self.debug
