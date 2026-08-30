"""TFS-MoE-Fusion consolidated implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class FocusGuideOutput:
    reliability_a: Tensor
    reliability_b: Tensor
    selection_logits: Tensor
    selection_a: Tensor
    selection_b: Tensor
    confidence: Tensor
    boundary: Tensor | None = None
    aux: dict[str, Any] = field(default_factory=dict)

    @property
    def reliability(self) -> Tensor:
        # Legacy callers used this name for the normalized source selection.
        # The scientifically distinct sigmoid reliabilities remain explicit above.
        return self.selection

    @property
    def selection(self) -> Tensor:
        return torch.cat((self.selection_a, self.selection_b), dim=1)

    @property
    def logits(self) -> Tensor:
        return self.selection_logits


@dataclass(slots=True)
class SemanticBackendOutput:
    logits: Tensor
    features: tuple[Tensor, ...] = ()
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SemanticGuideOutput:
    logits: Tensor
    probabilities: Tensor
    uncertainty: Tensor
    boundary: Tensor
    features: tuple[Tensor, ...] = ()
    available: bool = True
    aux: dict[str, Any] = field(default_factory=dict)


class FocusGuideBackend(nn.Module, ABC):
    @abstractmethod
    def forward(self, *args, **kwargs) -> FocusGuideOutput:
        raise NotImplementedError


class SemanticBackendBase(nn.Module, ABC):
    @abstractmethod
    def forward(self, image: Tensor) -> SemanticBackendOutput:
        raise NotImplementedError


# Compatibility alias.
SemanticGuideBackend = SemanticBackendBase


class PretrainedSemanticAdapter(nn.Module):
    def __init__(
        self,
        backend: nn.Module,
        freeze_backend: bool = True,
        detach_input: bool = False,
    ) -> None:
        super().__init__()
        self.backend, self.freeze_backend, self.detach_input = (
            backend,
            freeze_backend,
            detach_input,
        )
        if freeze_backend:
            for parameter in backend.parameters():
                parameter.requires_grad_(False)
            backend.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backend:
            self.backend.eval()
        return self

    def forward(self, image: Tensor) -> SemanticGuideOutput:
        value = self.backend(image.detach() if self.detach_input else image)
        if isinstance(value, SemanticGuideOutput):
            return value
        if not isinstance(value, SemanticBackendOutput):
            raise TypeError("Semantic backend must return SemanticBackendOutput")
        from tfs_moe_fusion.models.guidance.adapter import semantic_guide_from_logits

        return semantic_guide_from_logits(
            value.logits, image.shape[-2:], value.features
        )


class NullSemanticGuide(nn.Module):
    def forward(self, image: Tensor) -> None:
        del image


from torch import nn
from torch.nn import functional

from tfs_moe_fusion.frequency import FDConv2d


def _luminance(image: Tensor) -> Tensor:
    if image.shape[1] == 1:
        return image
    weights = image.new_tensor((0.299, 0.587, 0.114))[None, :, None, None]
    return (image * weights).sum(1, keepdim=True)


class ImageDetailExtractor(nn.Module):
    def __init__(
        self,
        normalize: str = "per_sample",
        use_sobel: bool = True,
        use_laplacian: bool = True,
    ) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        laplacian = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        )
        self.register_buffer("sobel_x", sobel_x[None, None], persistent=True)
        self.register_buffer("sobel_y", sobel_x.t()[None, None], persistent=True)
        self.register_buffer("laplacian", laplacian[None, None], persistent=True)
        self.normalize = normalize
        self.use_sobel, self.use_laplacian = use_sobel, use_laplacian

    def _normal(self, value: Tensor) -> Tensor:
        if self.normalize == "none":
            return value
        mean = value.mean((-2, -1), keepdim=True)
        std = value.std((-2, -1), keepdim=True, unbiased=False)
        return (value - mean) / (std + 1e-6)

    def forward(self, image: Tensor) -> Tensor:
        gray = _luminance(image).float()
        gx = functional.conv2d(gray, self.sobel_x, padding=1)
        gy = functional.conv2d(gray, self.sobel_y, padding=1)
        gradient = torch.sqrt(gx.square() + gy.square() + 1e-8)
        laplacian = functional.conv2d(gray, self.laplacian, padding=1).abs()
        if not self.use_sobel:
            gradient = torch.zeros_like(gradient)
        if not self.use_laplacian:
            laplacian = torch.zeros_like(laplacian)
        return torch.cat((self._normal(gradient), self._normal(laplacian)), 1).to(
            image.dtype
        )


class FocusReliabilityEstimator(nn.Module):
    def __init__(
        self,
        s1_channels: int,
        s2_channels: int,
        hidden: int,
        use_fdconv: bool = True,
    ) -> None:
        super().__init__()
        self.s2_projection = nn.Conv2d(s2_channels, hidden, 1)
        self.input_projection = nn.Conv2d(s1_channels + hidden + 2, hidden, 1)
        self.local = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GroupNorm(1, hidden),
            nn.Conv2d(hidden, hidden, 1),
            nn.GELU(),
        )
        self.fdconv = (
            FDConv2d(hidden, hidden, 3, min(8, 9), groups=hidden)
            if use_fdconv
            else nn.Identity()
        )
        self.output = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )

    def forward(self, s1: Tensor, s2: Tensor, detail: Tensor) -> Tensor:
        s2 = functional.interpolate(
            self.s2_projection(s2), s1.shape[-2:], mode="bilinear", align_corners=False
        )
        if detail.shape[-2:] != s1.shape[-2:]:
            detail = functional.interpolate(
                detail, s1.shape[-2:], mode="bilinear", align_corners=False
            )
        value = self.input_projection(torch.cat((s1, s2, detail), 1))
        return self.output(self.fdconv(self.local(value)))


class LightweightFocusHead(FocusGuideBackend):
    def __init__(
        self,
        channels: list[int],
        hidden_channels: int = 32,
        detach_backbone_features: bool = False,
        gradient_normalization: str = "per_sample",
        use_sobel: bool = True,
        use_laplacian: bool = True,
        use_fdconv: bool = True,
    ) -> None:
        super().__init__()
        self.detail = ImageDetailExtractor(
            gradient_normalization, use_sobel, use_laplacian
        )
        self.estimator = FocusReliabilityEstimator(
            channels[0], channels[1], hidden_channels, use_fdconv
        )
        self.detach_backbone_features = detach_backbone_features

    def forward(self, *args) -> FocusGuideOutput:
        if len(args) == 4:
            image_a, image_b, source_a_features, source_b_features = args
        elif len(args) == 3:
            # Compatibility with the earlier (coarse, pyramid_a, pyramid_b) API.
            image_a, source_a_features, source_b_features = args
            image_b = image_a
        else:
            raise TypeError("Focus head expects two images and two source pyramids")
        a1, a2 = source_a_features[0], source_a_features[1]
        b1, b2 = source_b_features[0], source_b_features[1]
        if self.detach_backbone_features:
            a1, a2, b1, b2 = (value.detach() for value in (a1, a2, b1, b2))
        logit_a = self.estimator(a1, a2, self.detail(image_a))
        logit_b = self.estimator(b1, b2, self.detail(image_b))
        target_size = image_a.shape[-2:]
        if logit_a.shape[-2:] != target_size:
            logit_a = functional.interpolate(
                logit_a, target_size, mode="bilinear", align_corners=False
            )
            logit_b = functional.interpolate(
                logit_b, target_size, mode="bilinear", align_corners=False
            )
        logits = torch.cat((logit_a, logit_b), 1)
        selection = torch.softmax(logits, 1)
        confidence = (selection[:, :1] - selection[:, 1:]).abs()
        difference = selection[:, :1] - selection[:, 1:]
        dx = functional.pad(
            (difference[..., 1:] - difference[..., :-1]).abs(), (0, 1, 0, 0)
        )
        dy = functional.pad(
            (difference[..., 1:, :] - difference[..., :-1, :]).abs(), (0, 0, 0, 1)
        )
        boundary = (dx + dy).clamp(0, 1)
        return FocusGuideOutput(
            torch.sigmoid(logit_a),
            torch.sigmoid(logit_b),
            logits,
            selection[:, :1],
            selection[:, 1:],
            confidence,
            boundary,
        )


import json
import math
from pathlib import Path

from torch import nn

from tfs_moe_fusion.types import ConfigurationError


class OverlapPatchEmbedding(nn.Module):
    def __init__(
        self, input_channels: int, channels: int, kernel: int, stride: int
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            input_channels, channels, kernel, stride=stride, padding=kernel // 2
        )
        self.layer_norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, tensor: Tensor) -> tuple[Tensor, int, int]:
        tensor = self.proj(tensor)
        height, width = tensor.shape[-2:]
        tokens = tensor.flatten(2).transpose(1, 2)
        return self.layer_norm(tokens), height, width


class EfficientSelfAttention(nn.Module):
    def __init__(self, channels: int, heads: int, reduction: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("SegFormer channels must be divisible by attention heads")
        self.channels = channels
        self.heads = heads
        self.head_channels = channels // heads
        self.reduction = reduction
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        if reduction > 1:
            self.sr = nn.Conv2d(channels, channels, reduction, stride=reduction)
            self.layer_norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, tokens: Tensor, height: int, width: int) -> Tensor:
        batch, length, _ = tokens.shape
        query = (
            self.query(tokens)
            .view(batch, length, self.heads, self.head_channels)
            .transpose(1, 2)
        )
        if self.reduction > 1:
            spatial = tokens.transpose(1, 2).reshape(
                batch, self.channels, height, width
            )
            reduced = self.sr(spatial).flatten(2).transpose(1, 2)
            key_value_tokens = self.layer_norm(reduced)
        else:
            key_value_tokens = tokens
        reduced_length = key_value_tokens.shape[1]
        key = (
            self.key(key_value_tokens)
            .view(batch, reduced_length, self.heads, self.head_channels)
            .transpose(1, 2)
        )
        value = (
            self.value(key_value_tokens)
            .view(batch, reduced_length, self.heads, self.head_channels)
            .transpose(1, 2)
        )
        attention = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_channels),
            dim=-1,
        )
        context = (
            torch.matmul(attention, value)
            .transpose(1, 2)
            .reshape(batch, length, self.channels)
        )
        return context


class AttentionOutput(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dense = nn.Linear(channels, channels)

    def forward(self, tensor: Tensor) -> Tensor:
        return self.dense(tensor)


class SegformerAttention(nn.Module):
    def __init__(self, channels: int, heads: int, reduction: int) -> None:
        super().__init__()
        self.self = EfficientSelfAttention(channels, heads, reduction)
        self.output = AttentionOutput(channels)

    def forward(self, tokens: Tensor, height: int, width: int) -> Tensor:
        attention = self.self(tokens, height, width)
        return self.output(attention)


class DepthwiseMixing(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, tokens: Tensor, height: int, width: int) -> Tensor:
        batch, _, channels = tokens.shape
        spatial = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return self.dwconv(spatial).flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, channels: int, ratio: int) -> None:
        super().__init__()
        hidden = channels * ratio
        self.dense1 = nn.Linear(channels, hidden)
        self.dwconv = DepthwiseMixing(hidden)
        self.dense2 = nn.Linear(hidden, channels)

    def forward(self, tokens: Tensor, height: int, width: int) -> Tensor:
        tokens = self.dense1(tokens)
        tokens = functional.gelu(self.dwconv(tokens, height, width))
        return self.dense2(tokens)


class SegformerBlock(nn.Module):
    def __init__(
        self, channels: int, heads: int, reduction: int, mlp_ratio: int
    ) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(channels, eps=1e-6)
        self.attention = SegformerAttention(channels, heads, reduction)
        self.layer_norm_2 = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = MixFFN(channels, mlp_ratio)

    def forward(self, tokens: Tensor, height: int, width: int) -> Tensor:
        tokens = tokens + self.attention(self.layer_norm_1(tokens), height, width)
        return tokens + self.mlp(self.layer_norm_2(tokens), height, width)


class SegformerEncoder(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        channels = [int(value) for value in config["hidden_sizes"]]  # type: ignore[index]
        depths = [int(value) for value in config["depths"]]  # type: ignore[index]
        heads = [int(value) for value in config["num_attention_heads"]]  # type: ignore[index]
        reductions = [int(value) for value in config["sr_ratios"]]  # type: ignore[index]
        ratios = [int(value) for value in config["mlp_ratios"]]  # type: ignore[index]
        kernels = [int(value) for value in config["patch_sizes"]]  # type: ignore[index]
        strides = [int(value) for value in config["strides"]]  # type: ignore[index]
        inputs = [int(config.get("num_channels", 3)), *channels[:-1]]
        self.patch_embeddings = nn.ModuleList(
            [
                OverlapPatchEmbedding(i, c, k, s)
                for i, c, k, s in zip(inputs, channels, kernels, strides, strict=True)
            ]
        )
        self.block = nn.ModuleList(
            [
                nn.ModuleList([SegformerBlock(c, h, r, m) for _ in range(d)])
                for c, d, h, r, m in zip(
                    channels, depths, heads, reductions, ratios, strict=True
                )
            ]
        )
        self.layer_norm = nn.ModuleList(
            [
                nn.LayerNorm(value, eps=float(config.get("layer_norm_eps", 1e-6)))
                for value in channels
            ]
        )

    def forward(self, tensor: Tensor) -> tuple[Tensor, ...]:
        features = []
        for patch, blocks, norm in zip(
            self.patch_embeddings, self.block, self.layer_norm, strict=True
        ):
            tokens, height, width = patch(tensor)
            for block in blocks:
                tokens = block(tokens, height, width)
            tokens = norm(tokens)
            tensor = tokens.transpose(1, 2).reshape(
                tokens.shape[0], tokens.shape[2], height, width
            )
            features.append(tensor)
        return tuple(features)


class SegformerModel(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        self.encoder = SegformerEncoder(config)

    def forward(self, tensor: Tensor) -> tuple[Tensor, ...]:
        return self.encoder(tensor)


class LinearProjection(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_channels, output_channels)

    def forward(self, tensor: Tensor) -> Tensor:
        return self.proj(tensor.flatten(2).transpose(1, 2))


class SegformerDecodeHead(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        channels = [int(value) for value in config["hidden_sizes"]]  # type: ignore[index]
        hidden = int(config["decoder_hidden_size"])
        classes = len(config["id2label"])  # type: ignore[arg-type]
        self.linear_c = nn.ModuleList(
            [LinearProjection(value, hidden) for value in channels]
        )
        self.linear_fuse = nn.Conv2d(hidden * len(channels), hidden, 1, bias=False)
        self.batch_norm = nn.BatchNorm2d(hidden)
        self.classifier = nn.Conv2d(hidden, classes, 1)

    def forward(self, features: tuple[Tensor, ...]) -> Tensor:
        target_size = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.linear_c, strict=True):
            tokens = projection(feature)
            spatial = tokens.transpose(1, 2).reshape(
                feature.shape[0], tokens.shape[-1], *feature.shape[-2:]
            )
            spatial = functional.interpolate(
                spatial, size=target_size, mode="bilinear", align_corners=False
            )
            projected.append(spatial)
        fused = self.linear_fuse(torch.cat(tuple(reversed(projected)), dim=1))
        fused = functional.relu(self.batch_norm(fused), inplace=False)
        return self.classifier(fused)


class SegformerForSemanticSegmentation(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)

    def forward(self, tensor: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        features = self.segformer(tensor)
        return self.decode_head(features), features


class FrozenSegformerBackend(SemanticGuideBackend):
    """Load and freeze the user-provided SegFormer checkpoint without Transformers."""

    def __init__(
        self,
        model_dir: str | Path,
        input_size: int = 256,
        expected_classes: int = 19,
        strict_loading: bool = True,
    ) -> None:
        super().__init__()
        directory = Path(model_dir)
        if not directory.is_absolute() and not directory.exists():
            project_root = Path(__file__).resolve().parents[1]
            directory = project_root / directory
        config_path = directory / "config.json"
        processor_path = directory / "preprocessor_config.json"
        weights_path = directory / "pytorch_model.bin"
        missing = [
            str(path)
            for path in (config_path, processor_path, weights_path)
            if not path.is_file()
        ]
        if missing:
            raise ConfigurationError(f"Incomplete local SegFormer assets: {missing}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        processor = json.loads(processor_path.read_text(encoding="utf-8"))
        classes = len(self.config["id2label"])
        if classes != expected_classes:
            raise ConfigurationError(
                f"Semantic checkpoint has {classes} classes, expected {expected_classes}"
            )
        self.model = SegformerForSemanticSegmentation(self.config)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        incompatible = self.model.load_state_dict(state, strict=strict_loading)
        if strict_loading and (
            incompatible.missing_keys or incompatible.unexpected_keys
        ):
            raise ConfigurationError(f"SegFormer state mismatch: {incompatible}")
        self.input_size = input_size
        self.register_buffer(
            "image_mean",
            torch.tensor(processor["image_mean"]).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(processor["image_std"]).view(1, 3, 1, 1),
            persistent=True,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> FrozenSegformerBackend:
        super().train(False)
        self.model.eval()
        return self

    def forward(self, coarse: Tensor) -> SemanticGuideOutput:
        if coarse.ndim != 4 or coarse.shape[1] != 3:
            raise ValueError("FrozenSegformerBackend expects RGB [B,3,H,W]")
        original_size = coarse.shape[-2:]
        resized = functional.interpolate(
            coarse,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        normalized = (resized.float() - self.image_mean) / self.image_std
        # Parameters are frozen, but this graph deliberately remains
        # differentiable with respect to the coarse image for training losses.
        logits, features = self.model(normalized)
        # Keep probability normalization and the derived entropy/boundary maps
        # in FP32. Under BF16 autocast, a 19-class softmax can accumulate an
        # error of several 1e-3 when summed across classes; converting those
        # already-rounded probabilities to FP32 afterwards cannot recover it.
        with torch.autocast(device_type=coarse.device.type, enabled=False):
            logits = functional.interpolate(
                logits.float(),
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
            probabilities = torch.softmax(logits, dim=1)
            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
                dim=1, keepdim=True
            ) / math.log(probabilities.shape[1])
            horizontal = functional.pad(
                (probabilities[..., :, 1:] - probabilities[..., :, :-1])
                .abs()
                .mean(1, keepdim=True),
                (0, 1, 0, 0),
            )
            vertical = functional.pad(
                (probabilities[..., 1:, :] - probabilities[..., :-1, :])
                .abs()
                .mean(1, keepdim=True),
                (0, 0, 0, 1),
            )
            boundary = horizontal + vertical
            boundary = boundary / (boundary.flatten(1).amax(1).view(-1, 1, 1, 1) + 1e-8)
        return SemanticGuideOutput(
            logits,
            probabilities,
            entropy,
            boundary,
            tuple(features),
        )


from torch import nn


def semantic_guide_from_logits(
    logits: Tensor, size: tuple[int, int], features: tuple[Tensor, ...] = ()
) -> SemanticGuideOutput:
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = functional.interpolate(
            logits.float(), size=size, mode="bilinear", align_corners=False
        )
        probabilities = torch.softmax(logits, 1)
        uncertainty = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
            1, keepdim=True
        ) / math.log(probabilities.shape[1])
        gx = functional.pad(
            probabilities[..., 1:] - probabilities[..., :-1], (0, 1, 0, 0)
        )
        gy = functional.pad(
            probabilities[..., 1:, :] - probabilities[..., :-1, :], (0, 0, 0, 1)
        )
        boundary = torch.sqrt(
            gx.square().sum(1, keepdim=True) + gy.square().sum(1, keepdim=True) + 1e-12
        )
        maximum = boundary.flatten(1).amax(1).view(-1, 1, 1, 1)
        boundary = boundary / (maximum + 1e-8)
    return SemanticGuideOutput(
        logits,
        probabilities,
        uncertainty,
        boundary,
        features,
        aux={"confidence": 1.0 - uncertainty},
    )


class GuidancePyramidAdapter(nn.Module):
    def __init__(
        self,
        channels: list[int],
        semantic_classes: int | None = None,
        guide_channels: int = 32,
    ) -> None:
        super().__init__()
        del semantic_classes, guide_channels
        self.stage_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(8, value, 3, padding=1),
                    nn.GroupNorm(1, value),
                    nn.GELU(),
                    nn.Conv2d(value, value, 3, padding=1),
                )
                for value in channels
            ]
        )

    def forward(
        self,
        focus: FocusGuideOutput | None,
        semantic: SemanticGuideOutput | None,
        target_features,
    ) -> tuple[Tensor, ...]:
        reference = target_features[0]
        full_size = reference.shape[-2:]

        def map_or_zero(value: Tensor | None) -> Tensor:
            if value is None:
                return reference.new_zeros(reference.shape[0], 1, *full_size)
            return functional.interpolate(
                value.to(reference), full_size, mode="bilinear", align_corners=False
            )

        maps = (
            map_or_zero(focus.reliability_a if focus else None),
            map_or_zero(focus.reliability_b if focus else None),
            map_or_zero(focus.confidence if focus else None),
            map_or_zero(semantic.boundary if semantic else None),
            map_or_zero(semantic.uncertainty if semantic else None),
            map_or_zero(1.0 - semantic.uncertainty if semantic else None),
            reference.new_full(
                (reference.shape[0], 1, *full_size), float(focus is not None)
            ),
            reference.new_full(
                (reference.shape[0], 1, *full_size), float(semantic is not None)
            ),
        )
        base = torch.cat(maps, 1)
        return tuple(
            projection(
                functional.interpolate(
                    base, target.shape[-2:], mode="bilinear", align_corners=False
                )
            )
            for projection, target in zip(
                self.stage_projections, target_features, strict=True
            )
        )


class GuidanceConditioner(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Conv2d(channels, channels, 1)
        self.projection = nn.Conv2d(channels, channels, 1)
        nn.init.normal_(self.projection.weight, std=1e-3)
        nn.init.zeros_(self.projection.bias)

    def forward(self, feature: Tensor, guidance: Tensor) -> Tensor:
        return feature + torch.sigmoid(self.gate(guidance)) * self.projection(guidance)
