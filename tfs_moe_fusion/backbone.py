"""TFS-MoE-Fusion consolidated implementation."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional


@dataclass(frozen=True, slots=True)
class PaddingInfo:
    original_height: int
    original_width: int
    pad_right: int
    pad_bottom: int
    mode: str


class ArbitrarySizePadder:
    def __init__(self, multiple: int = 8) -> None:
        if multiple <= 0:
            raise ValueError("multiple must be positive")
        self.multiple = multiple

    def pad(self, tensor: Tensor) -> tuple[Tensor, PaddingInfo]:
        if tensor.ndim != 4:
            raise ValueError("pad expects [B,C,H,W]")
        height, width = tensor.shape[-2:]
        pad_bottom = (-height) % self.multiple
        pad_right = (-width) % self.multiple
        mode = "reflect"
        if (pad_bottom and pad_bottom >= height) or (pad_right and pad_right >= width):
            mode = "replicate"
        info = PaddingInfo(height, width, pad_right, pad_bottom, mode)
        if pad_bottom == 0 and pad_right == 0:
            return tensor, info
        padded = functional.pad(tensor, (0, pad_right, 0, pad_bottom), mode=mode)
        return padded, info

    @staticmethod
    def unpad(tensor: Tensor, info: PaddingInfo) -> Tensor:
        if tensor.ndim != 4:
            raise ValueError("unpad expects [B,C,H,W]")
        if (
            tensor.shape[-2] < info.original_height
            or tensor.shape[-1] < info.original_width
        ):
            raise ValueError("tensor is smaller than the requested original size")
        return tensor[..., : info.original_height, : info.original_width]


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import pairwise

from torch import nn

from tfs_moe_fusion.types import FusionBatch, RouterDiagnostics, SpectralStatistics


@dataclass(slots=True)
class FeaturePyramid:
    s1: Tensor
    s2: Tensor
    s3: Tensor
    s4: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {"s1": self.s1, "s2": self.s2, "s3": self.s3, "s4": self.s4}

    def validate(self) -> None:
        values = tuple(self)
        if any(value.ndim != 4 for value in values):
            raise ValueError("Feature pyramid entries must be [B,C,H,W]")
        if len({value.shape[0] for value in values}) != 1:
            raise ValueError("Feature pyramid batch sizes must match")
        for shallow, deep in pairwise(values):
            expected = ((shallow.shape[-2] + 1) // 2, (shallow.shape[-1] + 1) // 2)
            if deep.shape[-2:] != expected:
                raise ValueError("Feature pyramid must downsample by two at each stage")

    def __iter__(self):
        return iter((self.s1, self.s2, self.s3, self.s4))

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int | str) -> Tensor:
        if isinstance(index, str):
            return self.as_dict()[index]
        return tuple(self)[index]


@dataclass(slots=True)
class BackboneOutput:
    source_a: FeaturePyramid
    source_b: FeaturePyramid
    fused: FeaturePyramid
    decoder_feature: Tensor
    fused_image: Tensor
    padded_fused_image: Tensor
    fused_y: Tensor | None
    padded_fused_y: Tensor | None
    gamut_projection_mask: Tensor | None
    spectral_statistics: tuple[SpectralStatistics, ...]
    router_diagnostics: tuple[RouterDiagnostics, ...]
    padding: object
    debug: dict[str, object] = field(default_factory=dict)

    @property
    def source_a_pyramid(self) -> FeaturePyramid:
        return self.source_a

    @property
    def source_b_pyramid(self) -> FeaturePyramid:
        return self.source_b

    @property
    def fused_pyramid(self) -> FeaturePyramid:
        return self.fused


class FusionBackbone(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: FusionBatch) -> BackboneOutput:
        raise NotImplementedError


import torch
from torch import nn

from tfs_moe_fusion.color import compose_luminance_with_visible_chroma
from tfs_moe_fusion.config import BackboneConfig, FrequencyConfig, MoEConfig
from tfs_moe_fusion.frequency import (
    FrequencyFoundationBlock,
    SpectralStatsExtractor,
)
from tfs_moe_fusion.moe import (
    ExpertContext,
    FunctionalMoEBlock,
    RouterContext,
    SharedExpertBank,
    SharedExpertMoESite,
    TaskEmbedding,
)
from tfs_moe_fusion.types import ModalityType, TaskType


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight, self.bias, self.epsilon = (
            nn.Parameter(torch.ones(channels)),
            nn.Parameter(torch.zeros(channels)),
            epsilon,
        )

    def forward(self, tensor: Tensor) -> Tensor:
        mean = tensor.mean(1, keepdim=True)
        variance = (tensor - mean).square().mean(1, keepdim=True)
        return (tensor - mean) * torch.rsqrt(variance + self.epsilon) * self.weight[
            :, None, None
        ] + self.bias[:, None, None]


def build_norm(name: str, channels: int) -> nn.Module:
    if name == "group_norm":
        return nn.GroupNorm(_groups(channels), channels)
    if name == "layer_norm_2d":
        return LayerNorm2d(channels)
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported normalization: {name}")


class ConvNeXtLikeBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 2.0,
        kernel_size: int = 3,
        layer_scale_init: float = 1e-6,
        drop_path: float = 0.0,
        normalization: str = "group_norm",
    ) -> None:
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size, padding=kernel_size // 2, groups=channels
        )
        self.norm = build_norm(normalization, channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, channels, 1)
        )
        self.layer_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), layer_scale_init)
        )
        self.drop_path = drop_path

    def forward(self, tensor: Tensor) -> Tensor:
        update = self.layer_scale * self.mlp(self.norm(self.depthwise(tensor)))
        if self.training and self.drop_path:
            keep = 1.0 - self.drop_path
            mask = torch.empty(
                tensor.shape[0], 1, 1, 1, device=tensor.device
            ).bernoulli_(keep)
            update = update * mask / keep
        return tensor + update


# Compatibility name used by earlier tests/importers.
LocalFeatureBlock = ConvNeXtLikeBlock


class ModalityStem(nn.Module):
    def __init__(self, input_channels: int, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.projection = nn.Conv2d(input_channels, channels, 1)

    def forward(self, tensor: Tensor) -> Tensor:
        return functional.gelu(self.body(tensor) + self.projection(tensor))


class ModalityStemBank(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.rgb, self.infrared, self.gray = (
            ModalityStem(3, channels),
            ModalityStem(1, channels),
            ModalityStem(1, channels),
        )

    def forward(self, tensor: Tensor, modality: ModalityType | Tensor) -> Tensor:
        if isinstance(modality, Tensor):
            if modality.shape != (tensor.shape[0],) or modality.dtype != torch.long:
                raise ValueError("modality ids must be long [B]")
            output = tensor.new_empty(
                tensor.shape[0], self.rgb.body[0].out_channels, *tensor.shape[-2:]
            )
            for value in modality.unique():
                indices = torch.nonzero(modality == value, as_tuple=False).flatten()
                kind = list(ModalityType)[int(value)]
                selected = tensor.index_select(0, indices)
                stem = (
                    self.rgb
                    if kind in {ModalityType.VISIBLE_RGB, ModalityType.GENERIC_RGB}
                    else self.infrared
                    if kind is ModalityType.INFRARED_GRAY
                    else self.gray
                )
                if selected.shape[1] != kind.channels:
                    raise ValueError(
                        "Mixed-modality tensors must have a compatible channel layout"
                    )
                output.index_copy_(0, indices, stem(selected))
            return output
        if modality in {ModalityType.VISIBLE_RGB, ModalityType.GENERIC_RGB}:
            return self.rgb(tensor)
        if modality is ModalityType.INFRARED_GRAY:
            return self.infrared(tensor)
        if modality is ModalityType.GENERIC_GRAY:
            return self.gray(tensor)
        raise ValueError(f"Unsupported modality: {modality}")


class DownsampleBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return self.body(tensor)


class CrossModalFusionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.source_projection = nn.Conv2d(channels, channels, 1)
        hidden = max(4, channels // 4)
        self.shared_score = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=False),
        )
        self.previous_projection = nn.Conv2d(channels, channels, 1)
        self.refine = nn.Sequential(
            ConvNeXtLikeBlock(channels), ConvNeXtLikeBlock(channels)
        )

    def forward(
        self, source_a: Tensor, source_b: Tensor, previous: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        pa, pb = self.source_projection(source_a), self.source_projection(source_b)
        common, difference = (pa + pb) * 0.5, (pa - pb).abs()
        score_a = self.shared_score(torch.cat((pa, common, difference), 1))
        score_b = self.shared_score(torch.cat((pb, common, difference), 1))
        weights = torch.softmax(torch.cat((score_a, score_b), 1), 1)
        fused = weights[:, :1] * pa + weights[:, 1:] * pb
        if previous is not None:
            fused = fused + self.previous_projection(previous)
        fused = self.refine(fused)
        return fused, {
            "weight_a": weights[:, :1],
            "weight_b": weights[:, 1:],
            "difference": difference,
        }


SymmetricAdaptiveFusion = CrossModalFusionBlock


class SkipFusionBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.up_projection = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.skip_projection = nn.Conv2d(output_channels, output_channels, 1)
        self.compression = nn.Conv2d(output_channels * 2, output_channels, 1)
        self.blocks = nn.Sequential(
            ConvNeXtLikeBlock(output_channels), ConvNeXtLikeBlock(output_channels)
        )

    def forward(self, tensor: Tensor, skip: Tensor) -> Tensor:
        tensor = functional.interpolate(
            tensor, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        tensor = self.up_projection(tensor)
        return self.blocks(
            self.compression(torch.cat((tensor, self.skip_projection(skip)), 1))
        )


DecoderStage = SkipFusionBlock


class CustomMultiscaleBackbone(FusionBackbone):
    is_scientific_model = True

    def __init__(
        self,
        channels: list[int],
        depths: list[int],
        frequency: FrequencyConfig,
        output_channels: int = 3,
        pad_multiple: int = 8,
        moe: MoEConfig | None = None,
        backbone_config: BackboneConfig | None = None,
        shared_expert_bank: SharedExpertBank | None = None,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or len(depths) != 4:
            raise ValueError("The fusion backbone requires exactly four stages")
        settings = backbone_config or BackboneConfig(channels=channels, depths=depths)
        self.channels, self.depths = tuple(channels), tuple(depths)
        self.padder = ArbitrarySizePadder(pad_multiple)
        self.stats_extractor = SpectralStatsExtractor(frequency.statistics_detach)
        self.stems = ModalityStemBank(channels[0])
        # Stable compatibility handles.
        self.rgb_stem, self.infrared_stem, self.gray_stem = (
            self.stems.rgb,
            self.stems.infrared,
            self.stems.gray,
        )
        block = lambda c: ConvNeXtLikeBlock(
            c,
            settings.mlp_ratio,
            settings.dw_kernel_size,
            settings.layer_scale_init,
            settings.drop_path,
            settings.normalization,
        )
        self.source_stages = nn.ModuleList(
            [
                nn.Sequential(*[block(c) for _ in range(d)])
                for c, d in zip(channels, depths, strict=True)
            ]
        )
        self.fused_stages = nn.ModuleList(
            [
                nn.Sequential(*[block(c) for _ in range(d)])
                for c, d in zip(channels, depths, strict=True)
            ]
        )
        self.source_downsamples = nn.ModuleList(
            [DownsampleBlock(channels[i], channels[i + 1]) for i in range(3)]
        )
        self.fused_downsamples = nn.ModuleList(
            [DownsampleBlock(channels[i], channels[i + 1]) for i in range(3)]
        )
        self.cross_modal_fusions = nn.ModuleList(
            [CrossModalFusionBlock(c) for c in channels]
        )
        self.frequency_blocks = nn.ModuleDict()
        uses_moe_slots = moe is not None and moe.enabled
        if frequency.enabled and not uses_moe_slots:
            for index, count in enumerate(channels):
                stage = f"s{index + 1}"
                if stage in frequency.placements:
                    self.frequency_blocks[stage] = FrequencyFoundationBlock(
                        count,
                        frequency.afno_num_blocks,
                        frequency.afno_sparsity_threshold,
                        frequency.afno_hard_thresholding_fraction,
                        frequency.fdconv_kernel_size,
                        frequency.fdconv_kernel_num,
                        frequency.residual_scale,
                        frequency.afno,
                        frequency.fdconv,
                        frequency.cross_frequency,
                    )
        self.task_embedding = (
            TaskEmbedding(moe.task_embedding_dim) if moe is not None else None
        )
        self.moe_blocks = nn.ModuleDict()
        if moe is not None and moe.enabled:
            if moe.shared_pool_enabled != (shared_expert_bank is not None):
                raise ValueError("MoE shared-pool config and bank must agree")
            for index, count in enumerate(channels):
                stage = f"s{index + 1}"
                blocks = nn.ModuleList(
                    [
                        (
                            SharedExpertMoESite(
                                count,
                                moe,
                                f"{stage}.moe{block_index}",
                                self.task_embedding,
                                shared_expert_bank,
                            )
                            if shared_expert_bank is not None
                            else FunctionalMoEBlock(
                                count,
                                moe,
                                f"{stage}.moe{block_index}",
                                self.task_embedding,
                            )
                        )
                        for block_index in range(moe.block_counts.get(stage, 0))
                        if stage in moe.placements
                    ]
                )
                if blocks:
                    self.moe_blocks[stage] = blocks
        self.decoder_stages = nn.ModuleList(
            [SkipFusionBlock(channels[i], channels[i - 1]) for i in range(3, 0, -1)]
        )
        self.fusion_y_head = nn.Sequential(
            nn.Conv2d(channels[0], 1, 3, padding=1), nn.Sigmoid()
        )
        self.mfif_rgb_head = nn.Sequential(
            nn.Conv2d(channels[0], output_channels, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, batch: FusionBatch) -> BackboneOutput:
        image_a, padding = self.padder.pad(batch.source_a.image)
        image_b, padding_b = self.padder.pad(batch.source_b.image)
        if padding != padding_b:
            raise RuntimeError("Source padding unexpectedly differs")
        feature_a = self.stems(image_a, batch.source_a.modality)
        feature_b = self.stems(image_b, batch.source_b.modality)
        source_a, source_b, fused_values = [], [], []
        diagnostics, cross_modal = [], []
        spectral = [
            self.stats_extractor(image_a, "input_a"),
            self.stats_extractor(image_b, "input_b"),
        ]
        fused: Tensor | None = None
        for index, (source_stage, fused_stage, fusion) in enumerate(
            zip(
                self.source_stages,
                self.fused_stages,
                self.cross_modal_fusions,
                strict=True,
            )
        ):
            if index:
                feature_a, feature_b = (
                    self.source_downsamples[index - 1](feature_a),
                    self.source_downsamples[index - 1](feature_b),
                )
            feature_a, feature_b = source_stage(feature_a), source_stage(feature_b)
            source_a.append(feature_a)
            source_b.append(feature_b)
            previous = (
                self.fused_downsamples[index - 1](fused) if fused is not None else None
            )
            fused, fusion_diagnostics = fusion(feature_a, feature_b, previous)
            fused = fused_stage(fused)
            cross_modal.append(fusion_diagnostics)
            stage = f"s{index + 1}"
            if stage in self.moe_blocks:
                for block_index, moe_block in enumerate(self.moe_blocks[stage]):
                    stats = self.stats_extractor(
                        fused, f"{stage}.moe{block_index}.input"
                    )
                    spectral.append(stats)
                    router_context = RouterContext(
                        fused,
                        batch.task,
                        batch.source_a.modality,
                        batch.source_b.modality,
                        stats,
                        feature_a,
                        feature_b,
                        stage_name=f"{stage}.moe{block_index}",
                    )
                    expert_context = ExpertContext(
                        batch.task,
                        batch.source_a.modality,
                        batch.source_b.modality,
                        feature_a,
                        feature_b,
                        stage_name=f"{stage}.moe{block_index}",
                        stage_guidance_feature=fused.new_zeros(fused.shape),
                    )
                    fused, router_diagnostics = moe_block(
                        fused, router_context, expert_context
                    )
                    diagnostics.append(router_diagnostics)
            elif stage in self.frequency_blocks:
                fused, _ = self.frequency_blocks[stage](fused, f"fused_{stage}")
                stats = self.stats_extractor(fused, f"fused_{stage}")
                spectral.append(stats)
            fused_values.append(fused)
        assert fused is not None
        decoded = fused
        for decoder, skip in zip(
            self.decoder_stages, reversed(fused_values[:-1]), strict=True
        ):
            decoded = decoder(decoded, skip)
        padded_y: Tensor | None = None
        gamut_projection_mask: Tensor | None = None
        if batch.task in {TaskType.VIF, TaskType.SEG}:
            predicted_y = self.fusion_y_head(decoded)
            visible = (
                image_a
                if batch.source_a.modality is ModalityType.VISIBLE_RGB
                else image_b
            )
            padded_image, padded_y, gamut_projection_mask = (
                compose_luminance_with_visible_chroma(predicted_y, visible)
            )
            head_name = "fusion_y"
        else:
            padded_image = self.mfif_rgb_head(decoded)
            head_name = "mfif_rgb"
        pyramids = (
            FeaturePyramid(*source_a),
            FeaturePyramid(*source_b),
            FeaturePyramid(*fused_values),
        )
        for pyramid in pyramids:
            pyramid.validate()
        debug: dict[str, object] = {
            "scientific_model": True,
            "padding": padding,
            "source_encoder_shared": True,
            "coarse_head": head_name,
            "cross_modal": tuple(cross_modal),
            "frequency_placements": tuple(sorted(self.frequency_blocks)),
            "moe_placements": tuple(sorted(self.moe_blocks)),
        }
        if gamut_projection_mask is not None:
            debug["y_gamut_clip_ratio"] = self.padder.unpad(
                gamut_projection_mask, padding
            ).mean()
        return BackboneOutput(
            source_a=pyramids[0],
            source_b=pyramids[1],
            fused=pyramids[2],
            decoder_feature=decoded,
            fused_image=self.padder.unpad(padded_image, padding),
            padded_fused_image=padded_image,
            fused_y=(
                self.padder.unpad(padded_y, padding) if padded_y is not None else None
            ),
            padded_fused_y=padded_y,
            gamut_projection_mask=(
                self.padder.unpad(gamut_projection_mask, padding)
                if gamut_projection_mask is not None
                else None
            ),
            spectral_statistics=tuple(spectral),
            router_diagnostics=tuple(diagnostics),
            padding=padding,
            debug=debug,
        )
