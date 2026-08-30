"""TFS-MoE-Fusion consolidated implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

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
    topk_indices: Tensor
    topk_weights: Tensor
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
        experts = values[0].probabilities.shape[1]
        loads = [
            torch.nn.functional.one_hot(value.topk_indices, experts)
            .float()
            .mean((0, 1))
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

    @abstractmethod
    def forward(self, tensor: Tensor, context: ExpertContext) -> ExpertOutput:
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

    def forward(self, tensor: Tensor, context: ExpertContext) -> ExpertOutput:
        del context
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

    def forward(self, tensor: Tensor, context: ExpertContext) -> ExpertOutput:
        del context
        bands = self.dwt(tensor)
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

    def forward(self, tensor: Tensor, context: ExpertContext) -> ExpertOutput:
        del context
        bands = self.dwt(tensor)
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

    def forward(self, tensor: Tensor, context: ExpertContext) -> ExpertOutput:
        residual = self.semantic_refine(tensor)
        gate_logits = self.content_gate(tensor)
        if (
            context.semantic_boundary is not None
            or context.semantic_uncertainty is not None
        ):
            boundary = context.semantic_boundary
            uncertainty = context.semantic_uncertainty
            if boundary is None:
                boundary = tensor.new_zeros(tensor.shape[0], 1, *tensor.shape[-2:])
            if uncertainty is None:
                uncertainty = tensor.new_zeros(tensor.shape[0], 1, *tensor.shape[-2:])
            maps = torch.cat((boundary, 1.0 - uncertainty), dim=1)
            maps = functional.interpolate(
                maps.float(),
                size=tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(tensor.dtype)
            gate_logits = gate_logits + self.external_conditioner(maps)
        gate = torch.sigmoid(gate_logits)
        return ExpertOutput(
            residual * gate,
            self.expert_type,
            _valid(tensor),
            diagnostics={"gate": gate},
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

    def forward(self, tensor: Tensor, context: ExpertContext) -> ExpertOutput:
        infrared = other = None
        if context.modality_a is ModalityType.INFRARED_GRAY:
            infrared, other = context.source_a_feature, context.source_b_feature
        elif context.modality_b is ModalityType.INFRARED_GRAY:
            infrared, other = context.source_b_feature, context.source_a_feature
        if infrared is None:
            return ExpertOutput(
                torch.zeros_like(tensor), self.expert_type, _valid(tensor, False)
            )
        if other is None:
            other = torch.zeros_like(infrared)
        difference = (infrared - other).abs()
        saliency = self.saliency(torch.cat((infrared, difference, tensor), dim=1))
        residual = self.refine(saliency * (infrared - tensor))
        return ExpertOutput(
            residual,
            self.expert_type,
            _valid(tensor),
            diagnostics={"saliency": saliency},
        )


EXPERT_CLASSES: dict[ExpertType, type[FunctionalExpert]] = {
    ExpertType.COMMON: CommonExpert,
    ExpertType.LOW_FREQUENCY: LowFrequencyExpert,
    ExpertType.DETAIL: DetailExpert,
    ExpertType.SEMANTIC: SemanticExpert,
    ExpertType.INFRARED_SALIENCY: InfraredSaliencyExpert,
}


def build_functional_expert(
    expert: ExpertType | str, channels: int, expansion: int = 2
) -> FunctionalExpert:
    expert_type = ExpertType.parse(expert)
    return EXPERT_CLASSES[expert_type](channels, expansion)


from torch import nn

from tfs_moe_fusion.types import ExpertType


class FunctionalExpertPool(nn.Module):
    """Own the expert modules and centralize their availability contract."""

    def __init__(
        self,
        channels: int,
        expert_names: list[str] | tuple[str, ...],
        expansion: int = 2,
    ) -> None:
        super().__init__()
        self._expert_names = tuple(
            ExpertType.parse(name).value for name in expert_names
        )
        self.modules_by_name = nn.ModuleDict(
            {
                name: build_functional_expert(name, channels, expansion)
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

    @property
    def sparse(self) -> bool:
        return self.mode == "sparse_batch"

    @classmethod
    def legacy(cls, sparse: bool) -> MoEExecutionPolicy:
        return cls("sparse_batch" if sparse else "dense_masked")


class FunctionalMoEBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        config: MoEConfig,
        block_id: str,
        task_embedding: TaskEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.block_id = block_id
        self.expert_names = tuple(config.experts)
        self.expert_pool = FunctionalExpertPool(
            channels, self.expert_names, config.expert_expansion
        )
        self.router = JointTopKRouter(
            channels,
            config.experts,
            config.top_k,
            config.router_hidden_channels,
            config.spatial_gating,
            config.router_temperature,
            task_embedding,
            config.modality_embedding_dim,
            config.branch_dropout,
            config.noisy_topk,
            config.noisy_topk_std,
        )
        self.sparse_execution = config.sparse_execution
        self.train_execution = config.train_execution
        self.inference_execution = config.inference_execution
        self.execution_policy: MoEExecutionPolicy | None = None
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
        routing = self.router(router_context)
        policy = self._resolve_policy(sparse_execution)
        global_weights, spatial_gates = self._mixture_weights(tensor, routing, policy)
        mixed_residual = torch.zeros_like(tensor)
        retained_outputs: dict[str, ExpertOutput] | None = (
            {} if return_expert_outputs else None
        )
        regularizers: dict[str, list[Tensor]] = {}
        residual_rms: dict[str, Tensor] = {}
        for index, expert in enumerate(self.experts):
            sample_indices = self._expert_indices(tensor, routing, index, policy)
            if sample_indices.numel() == 0:
                continue
            selected_tensor = tensor.index_select(0, sample_indices)
            selected_context = self._select_context(expert_context, sample_indices)
            expert_output = expert(selected_tensor, selected_context)
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
            mixed_residual = mixed_residual.index_add(0, sample_indices, contribution)
            for name, value in self._expert_regularizers(
                expert_output, selected_context
            ).items():
                regularizers.setdefault(name, []).append(value)
            if retained_outputs is not None:
                retained_outputs[self.expert_names[index]] = self._restore_output(
                    tensor, sample_indices, expert_output
                )
        scaled_residual = self.residual_scale * mixed_residual
        output = tensor + scaled_residual
        auxiliary = dict(routing.auxiliary)
        auxiliary["expert_residual_rms"] = residual_rms
        auxiliary["residual_scale_rms"] = (
            self.residual_scale.detach().float().square().mean().sqrt()
        )
        if regularizers:
            auxiliary["expert_regularizers"] = {
                name: torch.stack(values).mean()
                for name, values in regularizers.items()
            }
        if retained_outputs is not None:
            auxiliary["expert_residuals"] = {
                name: value.residual for name, value in retained_outputs.items()
            }
        if expert_context.semantic_boundary is not None:
            auxiliary["semantic_boundary"] = expert_context.semantic_boundary
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

    def set_execution_policy(self, policy: MoEExecutionPolicy | None) -> None:
        self.execution_policy = policy

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
    def _expert_regularizers(
        output: ExpertOutput, context: ExpertContext
    ) -> dict[str, Tensor]:
        residual = output.residual
        if output.expert is ExpertType.LOW_FREQUENCY:
            smooth = functional.avg_pool2d(residual, 5, stride=1, padding=2)
            return {"frequency/low_leakage": (residual - smooth).abs().mean()}
        if output.expert is ExpertType.DETAIL:
            smooth = functional.avg_pool2d(residual, 5, stride=1, padding=2)
            return {"frequency/detail_leakage": smooth.abs().mean()}
        if output.expert is ExpertType.SEMANTIC:
            gate = output.diagnostics.get("gate")
            boundary = context.semantic_boundary
            if isinstance(gate, Tensor) and boundary is not None:
                target = functional.interpolate(
                    boundary.to(gate),
                    gate.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0, 1)
                return {
                    "frequency/semantic_boundary": functional.binary_cross_entropy(
                        gate.clamp(1e-6, 1 - 1e-6), target
                    )
                }
        if output.expert is ExpertType.INFRARED_SALIENCY:
            saliency = output.diagnostics.get("saliency")
            infrared = other = None
            if context.modality_a is ModalityType.INFRARED_GRAY:
                infrared, other = context.source_a_feature, context.source_b_feature
            elif context.modality_b is ModalityType.INFRARED_GRAY:
                infrared, other = context.source_b_feature, context.source_a_feature
            if isinstance(saliency, Tensor) and infrared is not None:
                other = torch.zeros_like(infrared) if other is None else other
                target = (infrared - other).abs().mean(1, keepdim=True)
                maximum = target.flatten(1).amax(1).clamp_min(1e-6)
                target = (target / maximum[:, None, None, None]).detach()
                return {
                    "infrared/saliency_alignment": functional.l1_loss(saliency, target)
                }
        return {}

    @staticmethod
    def _select_context(context: ExpertContext, indices: Tensor) -> ExpertContext:
        return context.index_select(indices)

    @property
    def experts(self) -> tuple[nn.Module, ...]:
        return tuple(self.expert_pool)


# The design document's canonical name; the older name remains supported.
TaskFrequencyMoEBlock = FunctionalMoEBlock
