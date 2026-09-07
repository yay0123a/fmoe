"""TFS-MoE-Fusion consolidated implementation."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from tfs_moe_fusion.color import luminance, rgb_to_ycbcr


def _depthwise(image: Tensor, kernel: Tensor) -> Tensor:
    channels = image.shape[1]
    weight = kernel.to(device=image.device, dtype=image.dtype).expand(
        channels, 1, -1, -1
    )
    return F.conv2d(image, weight, padding=kernel.shape[-1] // 2, groups=channels)


def sobel(image: Tensor) -> tuple[Tensor, Tensor]:
    gray = luminance(image)
    kx = gray.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))).view(1, 1, 3, 3) / 4
    return _depthwise(gray, kx), _depthwise(gray, kx.transpose(-1, -2))


def gradient_magnitude(image: Tensor, epsilon: float = 1e-6) -> Tensor:
    gx, gy = sobel(image)
    return torch.sqrt(gx.square() + gy.square() + epsilon)


def laplacian(image: Tensor) -> Tensor:
    kernel = image.new_tensor(((0, 1, 0), (1, -4, 1), (0, 1, 0))).view(1, 1, 3, 3)
    return _depthwise(luminance(image), kernel)


def gaussian_kernel(size: int, sigma: float, reference: Tensor) -> Tensor:
    coordinates = (
        torch.arange(size, device=reference.device, dtype=reference.dtype)
        - (size - 1) / 2
    )
    vector = torch.exp(-coordinates.square() / (2 * sigma * sigma))
    vector = vector / vector.sum()
    return (vector[:, None] * vector[None, :]).view(1, 1, size, size)


def gaussian_low_pass(image: Tensor, size: int = 9, sigma: float = 2.0) -> Tensor:
    return _depthwise(image, gaussian_kernel(size, sigma, image))


def high_pass(image: Tensor, size: int = 9, sigma: float = 2.0) -> Tensor:
    return image - gaussian_low_pass(image, size, sigma)


def charbonnier(left: Tensor, right: Tensor, epsilon: float = 1e-3) -> Tensor:
    return torch.sqrt((left - right).square() + epsilon * epsilon).mean()


def ssim_map(left: Tensor, right: Tensor, size: int = 7, sigma: float = 1.5) -> Tensor:
    """Return a numerically stable local SSIM map in ``[-1, 1]``.

    SSIM's local variance is a subtraction of two nearly equal values. Running
    that calculation under BF16 autocast can make the variance negative and
    collapse the denominator, producing scores in the hundreds or thousands.
    Keep the local statistics in FP32 and explicitly enforce their mathematical
    bounds so mixed-precision training cannot optimize that numerical failure.
    """
    if left.shape[0] != right.shape[0] or left.shape[-2:] != right.shape[-2:]:
        raise ValueError("SSIM inputs must share batch and spatial dimensions")
    if size <= 0 or size % 2 == 0:
        raise ValueError("SSIM window size must be positive and odd")
    if right.shape[1] == 1 and left.shape[1] == 3:
        right = right.expand(-1, 3, -1, -1)
    if left.shape[1] == 1 and right.shape[1] == 3:
        left = left.expand(-1, 3, -1, -1)
    if left.shape[1] != right.shape[1]:
        raise ValueError("SSIM inputs must have equal channels or a gray/RGB pair")

    with torch.autocast(device_type=left.device.type, enabled=False):
        left_fp32 = left.float()
        right_fp32 = right.float()
        channels = left_fp32.shape[1]
        window = gaussian_kernel(size, sigma, left_fp32).expand(channels, 1, size, size)
        padding = size // 2
        left_padded = F.pad(
            left_fp32, (padding, padding, padding, padding), mode="replicate"
        )
        right_padded = F.pad(
            right_fp32, (padding, padding, padding, padding), mode="replicate"
        )
        mu_x = F.conv2d(left_padded, window, groups=channels)
        mu_y = F.conv2d(right_padded, window, groups=channels)
        sigma_x = (
            F.conv2d(left_padded.square(), window, groups=channels) - mu_x.square()
        ).clamp_min(0.0)
        sigma_y = (
            F.conv2d(right_padded.square(), window, groups=channels) - mu_y.square()
        ).clamp_min(0.0)
        sigma_xy = (
            F.conv2d(left_padded * right_padded, window, groups=channels) - mu_x * mu_y
        )
        c1, c2 = 0.01**2, 0.03**2
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
        score = numerator / denominator.clamp_min(torch.finfo(torch.float32).eps)
        if not torch.isfinite(score).all():
            raise FloatingPointError("SSIM produced a non-finite local score")
        return score.clamp(-1.0, 1.0)


def ssim(left: Tensor, right: Tensor, size: int = 7, sigma: float = 1.5) -> Tensor:
    """Return a numerically stable mean SSIM score in ``[-1, 1]``."""
    return ssim_map(left, right, size, sigma).mean()


def entropy(probabilities: Tensor) -> Tensor:
    count = probabilities.shape[1]
    return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
        1
    ).mean() / math.log(count)


from dataclasses import dataclass, field
from typing import Any

from torch import nn

from tfs_moe_fusion.types import FusionBatch, FusionOutput, TaskType


@dataclass(slots=True)
class LossContext:
    batch: FusionBatch
    output: FusionOutput
    task: TaskType
    epoch: int
    global_step: int
    model: nn.Module | None = None
    aux: dict[str, Any] = field(default_factory=dict)
    phase: str = "joint"
    loss_multipliers: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class LossOutput:
    total: Tensor
    components: dict[str, Tensor]
    weighted_components: dict[str, Tensor]
    diagnostics: dict[str, Any]
    skipped: dict[str, str]


def low_frequency_consistency(
    left: Tensor, right: Tensor, size: int, sigma: float
) -> Tensor:
    return (
        (gaussian_low_pass(left, size, sigma) - gaussian_low_pass(right, size, sigma))
        .abs()
        .mean()
    )


def focus_losses(
    logits: Tensor, confidence: Tensor, target_a: Tensor
) -> dict[str, Tensor]:
    if target_a.ndim == 4:
        target_a = target_a[:, 0]
    target_a = target_a.float().clamp(0, 1)
    labels = (target_a < 0.5).long()  # one means source A; CE class zero means A
    target_boundary = gradient_magnitude(target_a[:, None]).clamp(0, 1)
    probabilities = torch.softmax(logits, 1)
    predicted_boundary = gradient_magnitude(
        probabilities[:, :1] - probabilities[:, 1:]
    ).clamp(0, 1)
    target_confidence = 1 - gaussian_low_pass(target_boundary, 9, 2.0).clamp(0, 1)
    return {
        "focus/selection": F.cross_entropy(logits, labels),
        "focus/boundary": F.l1_loss(predicted_boundary, target_boundary),
        "focus/confidence": F.l1_loss(confidence, target_confidence),
    }


from tfs_moe_fusion.types import RouterDiagnostics


def frequency_specialization(
    diagnostics: tuple[RouterDiagnostics, ...],
) -> dict[str, Tensor]:
    low_terms: list[Tensor] = []
    detail_terms: list[Tensor] = []
    semantic_terms: list[Tensor] = []
    infrared_terms: list[Tensor] = []
    for item in diagnostics:
        regularizers = item.auxiliary.get("expert_regularizers", {})
        if regularizers:
            if (value := regularizers.get("frequency/low_leakage")) is not None:
                low_terms.append(value)
            if (value := regularizers.get("frequency/detail_leakage")) is not None:
                detail_terms.append(value)
            if (value := regularizers.get("frequency/semantic_boundary")) is not None:
                semantic_terms.append(value)
            if (value := regularizers.get("infrared/saliency_alignment")) is not None:
                infrared_terms.append(value)
            continue
        residuals = item.auxiliary.get("expert_residuals", {})
        low, detail, semantic = (
            residuals.get("low_frequency"),
            residuals.get("detail"),
            residuals.get("semantic"),
        )
        if low is not None:
            low_terms.append(high_pass(low).abs().mean())
        if detail is not None:
            detail_terms.append(gaussian_low_pass(detail).abs().mean())
        boundary = item.auxiliary.get("semantic_boundary")
        if semantic is not None and boundary is not None:
            boundary = torch.nn.functional.interpolate(
                boundary, semantic.shape[-2:], mode="bilinear", align_corners=False
            )
            semantic_terms.append(
                (semantic.abs().mean(1, keepdim=True) * (1 - boundary)).mean()
            )
    zero = diagnostics[0].probabilities.sum() * 0

    def mean(values: list[Tensor]) -> Tensor:
        return torch.stack(values).mean() if values else zero

    return {
        "frequency/low_leakage": mean(low_terms),
        "frequency/detail_leakage": mean(detail_terms),
        "frequency/semantic_boundary": mean(semantic_terms),
        "infrared/saliency_alignment": mean(infrared_terms),
    }


def directional_gradient_targets(
    visible_y: Tensor,
    infrared: Tensor,
    *,
    ir_dominance_ratio: float = 1.2,
    visible_support_kernel: int = 3,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build signed targets, admitting IR only beyond nearby visible support."""
    if ir_dominance_ratio < 1.0:
        raise ValueError("ir_dominance_ratio must be at least 1")
    if visible_support_kernel <= 0 or visible_support_kernel % 2 == 0:
        raise ValueError("visible_support_kernel must be positive and odd")
    visible_y, infrared = luminance(visible_y), luminance(infrared)
    gx_v, gy_v = sobel(visible_y)
    gx_i, gy_i = sobel(infrared)
    magnitude_v = torch.sqrt(gx_v.square() + gy_v.square() + 1e-6)
    magnitude_i = torch.sqrt(gx_i.square() + gy_i.square() + 1e-6)
    visible_support = F.max_pool2d(
        magnitude_v,
        kernel_size=visible_support_kernel,
        stride=1,
        padding=visible_support_kernel // 2,
    )
    choose_ir = magnitude_i > visible_support * ir_dominance_ratio
    return (
        torch.where(choose_ir, gx_i, gx_v),
        torch.where(choose_ir, gy_i, gy_v),
        choose_ir,
    )


def directional_gradient_loss(
    fused_y: Tensor,
    visible_y: Tensor,
    infrared: Tensor,
    *,
    ir_dominance_ratio: float = 1.2,
    visible_support_kernel: int = 3,
) -> Tensor:
    """Match signed Sobel components to a visible-anchored multimodal target."""
    target_gx, target_gy, _ = directional_gradient_targets(
        visible_y,
        infrared,
        ir_dominance_ratio=ir_dominance_ratio,
        visible_support_kernel=visible_support_kernel,
    )
    fused_gx, fused_gy = sobel(fused_y)
    return F.l1_loss(fused_gx, target_gx) + F.l1_loss(fused_gy, target_gy)


def soft_directional_gradient_targets(
    visible_y: Tensor,
    infrared: Tensor,
    *,
    ir_dominance_ratio: float = 1.2,
    visible_support_kernel: int = 3,
    transition: float = 0.15,
    min_magnitude: float = 0.02,
) -> tuple[Tensor, Tensor, Tensor]:
    """Blend signed gradients only for confident IR-only structure.

    Nearby visible support suppresses an IR edge, which avoids widening a
    slightly misregistered visible edge. The continuous gate makes the target
    stable around the admission threshold.
    """
    if ir_dominance_ratio < 1.0:
        raise ValueError("ir_dominance_ratio must be at least 1")
    if visible_support_kernel <= 0 or visible_support_kernel % 2 == 0:
        raise ValueError("visible_support_kernel must be positive and odd")
    if transition <= 0 or min_magnitude <= 0:
        raise ValueError("soft directional gradient parameters must be positive")
    with torch.autocast(device_type=visible_y.device.type, enabled=False):
        visible_y, infrared = luminance(visible_y.float()), luminance(infrared.float())
        gx_v, gy_v = sobel(visible_y)
        gx_i, gy_i = sobel(infrared)
        magnitude_v = torch.sqrt(gx_v.square() + gy_v.square() + 1e-6)
        magnitude_i = torch.sqrt(gx_i.square() + gy_i.square() + 1e-6)
        visible_support = F.max_pool2d(
            magnitude_v,
            kernel_size=visible_support_kernel,
            stride=1,
            padding=visible_support_kernel // 2,
        )
        dominance = magnitude_i / visible_support.clamp_min(min_magnitude)
        dominance_gate = torch.sigmoid(
            (dominance - ir_dominance_ratio) / transition
        )
        edge_gate = torch.sigmoid((magnitude_i - min_magnitude) / min_magnitude)
        ir_weight = dominance_gate * edge_gate
        return (
            torch.lerp(gx_v, gx_i, ir_weight),
            torch.lerp(gy_v, gy_i, ir_weight),
            ir_weight,
        )


def soft_directional_gradient_loss(
    fused_y: Tensor,
    visible_y: Tensor,
    infrared: Tensor,
    *,
    ir_dominance_ratio: float = 1.2,
    visible_support_kernel: int = 3,
    transition: float = 0.15,
    min_magnitude: float = 0.02,
    charbonnier_epsilon: float = 1e-3,
) -> Tensor:
    """Robustly match a softly gated, visible-anchored signed gradient target."""
    if charbonnier_epsilon <= 0:
        raise ValueError("charbonnier_epsilon must be positive")
    with torch.autocast(device_type=fused_y.device.type, enabled=False):
        target_gx, target_gy, _ = soft_directional_gradient_targets(
            visible_y,
            infrared,
            ir_dominance_ratio=ir_dominance_ratio,
            visible_support_kernel=visible_support_kernel,
            transition=transition,
            min_magnitude=min_magnitude,
        )
        fused_gx, fused_gy = sobel(fused_y.float())
        return charbonnier(fused_gx, target_gx, charbonnier_epsilon) + charbonnier(
            fused_gy, target_gy, charbonnier_epsilon
        )


def adaptive_directional_gradient_loss(
    fused_y: Tensor,
    visible_y: Tensor,
    infrared_aligned: Tensor,
    infrared_weight: Tensor,
    *,
    charbonnier_epsilon: float = 1e-3,
) -> Tensor:
    """Preserve VI edges while admitting aligned IR detail where IR is trusted."""
    if charbonnier_epsilon <= 0:
        raise ValueError("charbonnier_epsilon must be positive")
    with torch.autocast(device_type=fused_y.device.type, enabled=False):
        gx_v, gy_v = sobel(visible_y.float())
        gx_i, gy_i = sobel(infrared_aligned.float())
        magnitude_v = torch.sqrt(gx_v.square() + gy_v.square() + 1e-6)
        magnitude_i = torch.sqrt(gx_i.square() + gy_i.square() + 1e-6)
        edge_reliability = magnitude_i / (magnitude_i + magnitude_v + 1e-6)
        edge_weight = (infrared_weight.float() * (0.25 + 0.75 * edge_reliability)).clamp(
            0, 1
        )
        target_gx = torch.lerp(gx_v, gx_i, edge_weight)
        target_gy = torch.lerp(gy_v, gy_i, edge_weight)
        fused_gx, fused_gy = sobel(fused_y.float())
        return charbonnier(fused_gx, target_gx, charbonnier_epsilon) + charbonnier(
            fused_gy, target_gy, charbonnier_epsilon
        )


def normalized_edge_energy(
    image: Tensor,
    *,
    normalization: str = "per_sample_mean",
    epsilon: float = 1e-6,
    scale_floor: float = 1e-4,
) -> Tensor:
    """Return edge energy without the positive epsilon floor in flat regions."""
    # Replicated padding prevents a constant image from becoming a strong edge at
    # the frame boundary through the zero padding used by the general Sobel helper.
    padded = F.pad(luminance(image), (1, 1, 1, 1), mode="replicate")
    gx, gy = sobel(padded)
    gx, gy = gx[..., 1:-1, 1:-1], gy[..., 1:-1, 1:-1]
    magnitude = (
        torch.sqrt(gx.square() + gy.square() + epsilon) - math.sqrt(epsilon)
    ).clamp_min(0.0)
    if normalization == "none":
        return magnitude
    if normalization != "per_sample_mean":
        raise ValueError(f"Unknown edge-energy normalization: {normalization}")
    scale = magnitude.mean(dim=(-2, -1), keepdim=True).clamp_min(scale_floor)
    return magnitude / scale


def align_infrared_luminance(visible: Tensor, infrared: Tensor) -> Tensor:
    """Match each IR image's global mean and contrast to the visible luminance.

    Visible and thermal sensors do not share an absolute brightness scale.  The
    aligned signal is only a fusion target/reference; it preserves IR structure
    while avoiding a global darkening or brightening caused by raw IR values.
    """
    visible_y, infrared_y = luminance(visible.float()), luminance(infrared.float())
    dims = (-2, -1)
    visible_mean = visible_y.mean(dims, keepdim=True)
    infrared_mean = infrared_y.mean(dims, keepdim=True)
    visible_std = visible_y.std(dims, keepdim=True, unbiased=False)
    infrared_std = infrared_y.std(dims, keepdim=True, unbiased=False)
    scale = (visible_std / infrared_std.clamp_min(1e-3)).clamp(0.25, 4.0)
    return ((infrared_y - infrared_mean) * scale + visible_mean).clamp(0, 1)


def adaptive_ir_blend_weight(
    visible: Tensor,
    infrared: Tensor,
    *,
    max_weight: float,
    darkness_threshold: float,
    darkness_transition: float,
    saliency_weight: float,
    visible_support_kernel: int,
    smoothing_kernel: int,
    energy_normalization: str,
) -> tuple[Tensor, Tensor]:
    """Return aligned IR luminance and a continuous, spatial IR admission map."""
    if not 0.0 <= max_weight <= 1.0:
        raise ValueError("max_weight must be between 0 and 1")
    if darkness_transition <= 0:
        raise ValueError("darkness_transition must be positive")
    if not 0.0 <= saliency_weight <= 1.0:
        raise ValueError("saliency_weight must be between 0 and 1")
    visible_y = luminance(visible.float())
    infrared_aligned = align_infrared_luminance(visible_y, infrared)
    visible_edges = normalized_edge_energy(
        visible_y, normalization=energy_normalization
    )
    infrared_edges = normalized_edge_energy(
        infrared_aligned, normalization=energy_normalization
    )
    visible_support = F.max_pool2d(
        visible_edges,
        kernel_size=visible_support_kernel,
        stride=1,
        padding=visible_support_kernel // 2,
    )
    edge_advantage = infrared_edges / (infrared_edges + visible_support + 1e-6)
    difference = (infrared_aligned - visible_y).abs()
    difference_scale = difference.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
    saliency = 1 - torch.exp(-difference / difference_scale)
    darkness = torch.sigmoid(
        (darkness_threshold - visible_y) / darkness_transition
    )
    reliability = saliency_weight * saliency + (1 - saliency_weight) * edge_advantage
    # IR can still supply a small amount of structure in bright regions, while
    # dark/salient regions receive most of the configured admission budget.
    weight = max_weight * (
        0.15 * edge_advantage + 0.85 * darkness * reliability
    )
    padding = smoothing_kernel // 2
    if padding:
        weight = F.avg_pool2d(
            F.pad(weight, (padding, padding, padding, padding), mode="replicate"),
            kernel_size=smoothing_kernel,
            stride=1,
        )
    return infrared_aligned, weight.clamp(0, max_weight)


def _smoothstep(value: Tensor, low: float, high: float) -> Tensor:
    """Return a cubic transition from zero at ``low`` to one at ``high``."""
    if low < 0 or high <= low:
        raise ValueError("smoothstep requires 0 <= low < high")
    position = ((value - low) / (high - low)).clamp(0, 1)
    return position.square() * (3 - 2 * position)


def hot_object_ir_blend_weight(
    visible: Tensor,
    infrared: Tensor,
    *,
    max_weight: float,
    darkness_threshold: float,
    darkness_transition: float,
    saliency_weight: float,
    hot_contrast_low: float,
    hot_contrast_high: float,
    hot_weight: float,
    dark_context_weight: float,
    edge_weight: float,
    visible_support_kernel: int,
    smoothing_kernel: int,
    energy_normalization: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return aligned IR, its admission map, and a positive thermal-hot map.

    Bright thermal targets are useful in both day and night images.  Unlike the
    dark-region profile, their admission is therefore driven by *positive* IR
    contrast and is not multiplied by the visible-darkness gate.  The existing
    darkness/reliability route remains as a conservative contextual fallback.
    """
    if not 0.0 <= max_weight <= 1.0:
        raise ValueError("max_weight must be between 0 and 1")
    if darkness_transition <= 0:
        raise ValueError("darkness_transition must be positive")
    if not 0.0 <= saliency_weight <= 1.0:
        raise ValueError("saliency_weight must be between 0 and 1")
    if hot_contrast_low < 0 or hot_contrast_high <= hot_contrast_low:
        raise ValueError("hot contrast thresholds must satisfy 0 <= low < high")
    for name, value in (
        ("hot_weight", hot_weight),
        ("dark_context_weight", dark_context_weight),
        ("edge_weight", edge_weight),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    visible_y = luminance(visible.float())
    infrared_aligned = align_infrared_luminance(visible_y, infrared)
    visible_edges = normalized_edge_energy(
        visible_y, normalization=energy_normalization
    )
    infrared_edges = normalized_edge_energy(
        infrared_aligned, normalization=energy_normalization
    )
    visible_support = F.max_pool2d(
        visible_edges,
        kernel_size=visible_support_kernel,
        stride=1,
        padding=visible_support_kernel // 2,
    )
    edge_advantage = infrared_edges / (infrared_edges + visible_support + 1e-6)
    difference = (infrared_aligned - visible_y).abs()
    difference_scale = difference.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
    saliency = 1 - torch.exp(-difference / difference_scale)
    darkness = torch.sigmoid(
        (darkness_threshold - visible_y) / darkness_transition
    )
    reliability = saliency_weight * saliency + (1 - saliency_weight) * edge_advantage

    positive_contrast = (infrared_aligned - visible_y).clamp_min(0)
    hotness = _smoothstep(positive_contrast, hot_contrast_low, hot_contrast_high)
    contextual_weight = (
        dark_context_weight * darkness * reliability + edge_weight * edge_advantage
    )
    padding = smoothing_kernel // 2
    if padding:
        contextual_weight = F.avg_pool2d(
            F.pad(
                contextual_weight,
                (padding, padding, padding, padding),
                mode="replicate",
            ),
            kernel_size=smoothing_kernel,
            stride=1,
        )
    weight = torch.maximum(hot_weight * hotness, contextual_weight)
    return infrared_aligned, weight.clamp(0, max_weight), hotness


def hot_underexposure_loss(
    fused_y: Tensor,
    visible: Tensor,
    infrared_aligned: Tensor,
    hotness: Tensor,
    *,
    minimum_contrast_retention: float,
) -> Tensor:
    """Penalize thermal-hot regions that retain too little positive contrast.

    Normalizing within each image prevents small people and vehicles from being
    diluted by a full-frame mean while leaving non-hot background untouched.
    """
    if not 0.0 <= minimum_contrast_retention <= 1.0:
        raise ValueError("minimum_contrast_retention must be between 0 and 1")
    visible_y = luminance(visible).to(fused_y)
    positive_contrast = (infrared_aligned.to(fused_y) - visible_y).clamp_min(0)
    required = minimum_contrast_retention * positive_contrast
    retained = fused_y - visible_y
    penalty = torch.relu(required - retained) * hotness.to(fused_y)
    numerator = penalty.sum(dim=(-2, -1))
    denominator = hotness.to(fused_y).sum(dim=(-2, -1)).clamp_min(1e-6)
    return (numerator / denominator).mean()


def vif_intensity_target(
    visible: Tensor,
    infrared: Tensor,
    *,
    mode: str = "pixel_max",
    energy_normalization: str = "per_sample_mean",
    ir_max_weight: float = 0.3,
    ir_darkness_threshold: float = 0.35,
    ir_darkness_transition: float = 0.1,
    ir_saliency_weight: float = 0.65,
    ir_hot_contrast_low: float = 0.08,
    ir_hot_contrast_high: float = 0.3,
    ir_hot_weight: float = 0.3,
    ir_dark_context_weight: float = 0.45,
    ir_edge_weight: float = 0.08,
    visible_support_kernel: int = 3,
    weight_smoothing_kernel: int = 3,
) -> tuple[Tensor, Tensor | None]:
    """Construct a VIF Y target and optional per-pixel IR contribution weight."""
    visible_y, infrared_y = luminance(visible), luminance(infrared)
    if mode == "pixel_max":
        return torch.maximum(visible_y, infrared_y), None
    if mode == "adaptive_dark_ir":
        infrared_aligned, weight = adaptive_ir_blend_weight(
            visible_y,
            infrared_y,
            max_weight=ir_max_weight,
            darkness_threshold=ir_darkness_threshold,
            darkness_transition=ir_darkness_transition,
            saliency_weight=ir_saliency_weight,
            visible_support_kernel=visible_support_kernel,
            smoothing_kernel=weight_smoothing_kernel,
            energy_normalization=energy_normalization,
        )
        return visible_y + weight * (infrared_aligned - visible_y), weight
    if mode == "hot_object_aware":
        infrared_aligned, weight, _ = hot_object_ir_blend_weight(
            visible_y,
            infrared_y,
            max_weight=ir_max_weight,
            darkness_threshold=ir_darkness_threshold,
            darkness_transition=ir_darkness_transition,
            saliency_weight=ir_saliency_weight,
            hot_contrast_low=ir_hot_contrast_low,
            hot_contrast_high=ir_hot_contrast_high,
            hot_weight=ir_hot_weight,
            dark_context_weight=ir_dark_context_weight,
            edge_weight=ir_edge_weight,
            visible_support_kernel=visible_support_kernel,
            smoothing_kernel=weight_smoothing_kernel,
            energy_normalization=energy_normalization,
        )
        return visible_y + weight * (infrared_aligned - visible_y).clamp_min(0), weight
    if mode != "gradient_weighted_visible_anchor":
        raise ValueError(f"Unknown VIF intensity mode: {mode}")
    if not 0.0 <= ir_max_weight <= 1.0:
        raise ValueError("ir_max_weight must be between 0 and 1")
    for name, kernel in (
        ("visible_support_kernel", visible_support_kernel),
        ("weight_smoothing_kernel", weight_smoothing_kernel),
    ):
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError(f"{name} must be positive and odd")

    normalized_visible = normalized_edge_energy(
        visible_y, normalization=energy_normalization
    )
    normalized_infrared = normalized_edge_energy(
        infrared_y, normalization=energy_normalization
    )
    visible_support = F.max_pool2d(
        normalized_visible,
        kernel_size=visible_support_kernel,
        stride=1,
        padding=visible_support_kernel // 2,
    )
    ir_share = normalized_infrared / (normalized_infrared + visible_support + 1e-6)
    ir_advantage = (2.0 * ir_share - 1.0).clamp(0.0, 1.0)
    weight = ir_max_weight * ir_advantage
    smoothing_padding = weight_smoothing_kernel // 2
    if smoothing_padding:
        weight = F.avg_pool2d(
            F.pad(
                weight,
                (
                    smoothing_padding,
                    smoothing_padding,
                    smoothing_padding,
                    smoothing_padding,
                ),
                mode="replicate",
            ),
            kernel_size=weight_smoothing_kernel,
            stride=1,
        )
    target = visible_y + weight * (infrared_y - visible_y)
    return target, weight


def vif_losses(
    fused: Tensor,
    visible: Tensor,
    infrared: Tensor,
    fused_y: Tensor | None = None,
    *,
    target_intensity: Tensor | None = None,
    intensity_mode: str = "pixel_max",
    intensity_energy_normalization: str = "per_sample_mean",
    ir_intensity_max_weight: float = 0.3,
    ir_darkness_threshold: float = 0.35,
    ir_darkness_transition: float = 0.1,
    ir_saliency_weight: float = 0.65,
    ir_hot_contrast_low: float = 0.08,
    ir_hot_contrast_high: float = 0.3,
    ir_hot_weight: float = 0.3,
    ir_dark_context_weight: float = 0.45,
    ir_edge_weight: float = 0.08,
    intensity_visible_support_kernel: int = 3,
    intensity_weight_smoothing_kernel: int = 3,
    intensity_weight: Tensor | None = None,
    gradient_mode: str = "magnitude_max",
    ir_gradient_dominance_ratio: float = 1.2,
    visible_gradient_support_kernel: int = 3,
    gradient_transition: float = 0.15,
    gradient_min_magnitude: float = 0.02,
    gradient_charbonnier_epsilon: float = 1e-3,
    ssim_mode: str = "source_max",
) -> dict[str, Tensor]:
    predicted_y = fused_y if fused_y is not None else luminance(fused)
    visible_y = luminance(visible)
    infrared_reference = infrared
    if intensity_mode == "adaptive_dark_ir":
        infrared_reference = align_infrared_luminance(visible_y, infrared)
        if intensity_weight is None:
            _, intensity_weight = adaptive_ir_blend_weight(
                visible_y,
                infrared,
                max_weight=ir_intensity_max_weight,
                darkness_threshold=ir_darkness_threshold,
                darkness_transition=ir_darkness_transition,
                saliency_weight=ir_saliency_weight,
                visible_support_kernel=intensity_visible_support_kernel,
                smoothing_kernel=intensity_weight_smoothing_kernel,
                energy_normalization=intensity_energy_normalization,
            )
    elif intensity_mode == "hot_object_aware":
        infrared_reference, computed_weight, _ = hot_object_ir_blend_weight(
            visible_y,
            infrared,
            max_weight=ir_intensity_max_weight,
            darkness_threshold=ir_darkness_threshold,
            darkness_transition=ir_darkness_transition,
            saliency_weight=ir_saliency_weight,
            hot_contrast_low=ir_hot_contrast_low,
            hot_contrast_high=ir_hot_contrast_high,
            hot_weight=ir_hot_weight,
            dark_context_weight=ir_dark_context_weight,
            edge_weight=ir_edge_weight,
            visible_support_kernel=intensity_visible_support_kernel,
            smoothing_kernel=intensity_weight_smoothing_kernel,
            energy_normalization=intensity_energy_normalization,
        )
        if intensity_weight is None:
            intensity_weight = computed_weight
    if target_intensity is None:
        target_intensity, _ = vif_intensity_target(
            visible_y,
            infrared,
            mode=intensity_mode,
            energy_normalization=intensity_energy_normalization,
            ir_max_weight=ir_intensity_max_weight,
            ir_darkness_threshold=ir_darkness_threshold,
            ir_darkness_transition=ir_darkness_transition,
            ir_saliency_weight=ir_saliency_weight,
            ir_hot_contrast_low=ir_hot_contrast_low,
            ir_hot_contrast_high=ir_hot_contrast_high,
            ir_hot_weight=ir_hot_weight,
            ir_dark_context_weight=ir_dark_context_weight,
            ir_edge_weight=ir_edge_weight,
            visible_support_kernel=intensity_visible_support_kernel,
            weight_smoothing_kernel=intensity_weight_smoothing_kernel,
        )
    if gradient_mode == "magnitude_max":
        target_gradient = torch.maximum(
            gradient_magnitude(visible_y), gradient_magnitude(infrared)
        )
        gradient_loss = (gradient_magnitude(predicted_y) - target_gradient).abs().mean()
    elif gradient_mode == "directional_visible_anchor":
        gradient_loss = directional_gradient_loss(
            predicted_y,
            visible_y,
            infrared,
            ir_dominance_ratio=ir_gradient_dominance_ratio,
            visible_support_kernel=visible_gradient_support_kernel,
        )
    elif gradient_mode == "soft_directional_visible_anchor":
        gradient_loss = soft_directional_gradient_loss(
            predicted_y,
            visible_y,
            infrared,
            ir_dominance_ratio=ir_gradient_dominance_ratio,
            visible_support_kernel=visible_gradient_support_kernel,
            transition=gradient_transition,
            min_magnitude=gradient_min_magnitude,
            charbonnier_epsilon=gradient_charbonnier_epsilon,
        )
    elif gradient_mode == "adaptive_directional":
        if intensity_weight is None:
            raise ValueError(
                "adaptive_directional gradient requires an adaptive IR intensity mode"
            )
        gradient_loss = adaptive_directional_gradient_loss(
            predicted_y,
            visible_y,
            infrared_reference,
            intensity_weight,
            charbonnier_epsilon=gradient_charbonnier_epsilon,
        )
    else:
        raise ValueError(f"Unknown VIF gradient mode: {gradient_mode}")
    color = fused.new_zeros(())
    if fused.shape[1] == visible.shape[1] == 3:
        color = (rgb_to_ycbcr(fused)[:, 1:] - rgb_to_ycbcr(visible)[:, 1:]).abs().mean()
    visible_ssim, infrared_ssim = vif_ssim_scores(
        predicted_y, visible_y, infrared_reference
    )
    if ssim_mode == "source_max":
        selected_ssim = torch.maximum(visible_ssim, infrared_ssim)
    elif ssim_mode == "visible_anchor":
        selected_ssim = visible_ssim
    elif ssim_mode == "adaptive_source":
        if intensity_weight is None:
            raise ValueError(
                "adaptive_source SSIM requires an adaptive IR intensity mode"
            )
        visible_map = ssim_map(predicted_y, visible_y)
        infrared_map = ssim_map(predicted_y, infrared_reference)
        selected_ssim = (
            (1 - intensity_weight.float()) * visible_map
            + intensity_weight.float() * infrared_map
        ).mean()
    else:
        raise ValueError(f"Unknown VIF SSIM mode: {ssim_mode}")
    return {
        "fusion/intensity": (predicted_y - target_intensity).abs().mean(),
        "fusion/gradient": gradient_loss,
        "fusion/ssim": 1 - selected_ssim,
        "fusion/color": color,
    }


def vif_ssim_scores(
    predicted_y: Tensor, visible: Tensor, infrared: Tensor
) -> tuple[Tensor, Tensor]:
    """Compare a VIF prediction to both sources in luminance space."""
    return ssim(predicted_y, luminance(visible)), ssim(predicted_y, luminance(infrared))


def seg_fusion_anchor_losses(
    fused: Tensor,
    visible: Tensor,
    infrared: Tensor,
    fused_y: Tensor | None = None,
) -> dict[str, Tensor]:
    """Minimal fusion-quality anchor for segmentation-conditioned output."""
    predicted_y = fused_y if fused_y is not None else luminance(fused)
    target_intensity = torch.maximum(luminance(visible), luminance(infrared))
    target_gradient = torch.maximum(
        gradient_magnitude(visible), gradient_magnitude(infrared)
    )
    return {
        "seg_fusion/intensity": (predicted_y - target_intensity).abs().mean(),
        "seg_fusion/gradient": (gradient_magnitude(predicted_y) - target_gradient)
        .abs()
        .mean(),
    }


def mfif_losses(
    fused: Tensor, target: Tensor, use_charbonnier: bool
) -> dict[str, Tensor]:
    reconstruction = (
        charbonnier(fused, target) if use_charbonnier else (fused - target).abs().mean()
    )
    return {
        "fusion/reconstruction": reconstruction,
        "fusion/gradient": (gradient_magnitude(fused) - gradient_magnitude(target))
        .abs()
        .mean(),
        "fusion/ssim": 1 - ssim(fused, target),
    }


def moe_balance_loss(
    diagnostics: tuple[RouterDiagnostics, ...],
) -> tuple[Tensor, Tensor, Tensor]:
    if not diagnostics:
        raise ValueError("MoE balance requires router diagnostics")
    importance_losses, load_losses, entropies = [], [], []
    for item in diagnostics:
        valid = item.valid_expert_mask.float()
        reduce_dims = tuple(index for index in range(valid.ndim) if index != 1)
        target = valid.mean(reduce_dims)
        target = target / target.sum().clamp_min(1)
        importance = item.probabilities.mean(
            tuple(index for index in range(item.probabilities.ndim) if index != 1)
        )
        hard_load = (
            item.hard_load.detach()
            if item.hard_load is not None
            else torch.nn.functional.one_hot(
                item.topk_indices, item.probabilities.shape[1]
            )
            .float()
            .mean((0, 1))
            .detach()
        )
        effective_experts = (target > 0).sum().to(importance)
        importance_losses.append((importance - target).square().sum())
        load_losses.append(effective_experts * (importance * hard_load).sum())
        entropies.append(entropy(item.probabilities))
    return (
        torch.stack(importance_losses).mean(),
        torch.stack(load_losses).mean(),
        torch.stack(entropies).mean(),
    )


def residual_magnitude(refinement: Tensor | None, fused: Tensor) -> Tensor:
    return refinement.abs().mean() if refinement is not None else fused.sum() * 0


def range_penalty(unclamped: Tensor) -> Tensor:
    return (-unclamped).relu().mean() + (unclamped - 1).relu().mean()


def dice_loss(logits: Tensor, target: Tensor, ignore_index: int = 255) -> Tensor:
    classes = logits.shape[1]
    valid = target != ignore_index
    safe = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe.long(), classes).permute(0, 3, 1, 2).to(logits.dtype)
    mask = valid[:, None]
    probabilities, one_hot = logits.softmax(1) * mask, one_hot * mask
    intersection = (probabilities * one_hot).sum((0, 2, 3))
    denominator = (probabilities + one_hot).sum((0, 2, 3))
    present = one_hot.sum((0, 2, 3)) > 0
    scores = (2 * intersection + 1e-6) / (denominator + 1e-6)
    return (1 - scores[present]).mean() if present.any() else logits.sum() * 0


def semantic_losses(
    logits: Tensor,
    target: Tensor,
    boundary: Tensor | None,
    fused: Tensor,
    ignore_index: int,
    class_weights: Tensor | None,
) -> dict[str, Tensor]:
    valid = target != ignore_index
    cross_entropy = (
        F.cross_entropy(
            logits, target.long(), weight=class_weights, ignore_index=ignore_index
        )
        if valid.any()
        else logits.sum() * 0
    )
    losses = {
        "semantic/ce": cross_entropy,
        "semantic/dice": dice_loss(logits, target, ignore_index),
    }
    if boundary is not None and valid.any():
        valid_mask = valid.float()[:, None]
        label_boundary = gradient_magnitude(target.float()[:, None]) * valid_mask
        losses["semantic/boundary"] = (
            F.l1_loss(boundary * valid_mask, label_boundary.clamp(0, 1))
            + 0.1 * (boundary * (1 - gradient_magnitude(fused).clamp(0, 1))).mean()
        )
    return losses


from tfs_moe_fusion.config import LossConfig
from tfs_moe_fusion.types import ContractError


class MultiTaskLossManager(nn.Module):
    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, context: LossContext) -> LossOutput:
        if (
            context.task is not context.batch.task
            or context.task is not context.output.task
        ):
            raise ContractError("Loss task must match both batch and output")
        components: dict[str, Tensor] = {}
        weights: dict[str, float] = {}
        skipped: dict[str, str] = {}
        intensity_weight: Tensor | None = None
        hotness: Tensor | None = None
        hot_infrared_reference: Tensor | None = None
        hot_target_reference: Tensor | None = None
        output, batch = context.output, context.batch
        if context.task is TaskType.VIF:
            visible, infrared = self._visible_ir(batch)
            vif_config = self.config.vif
            target_intensity, intensity_weight = vif_intensity_target(
                visible,
                infrared,
                mode=vif_config.intensity_mode,
                energy_normalization=vif_config.intensity_energy_normalization,
                ir_max_weight=vif_config.ir_intensity_max_weight,
                ir_darkness_threshold=vif_config.ir_darkness_threshold,
                ir_darkness_transition=vif_config.ir_darkness_transition,
                ir_saliency_weight=vif_config.ir_saliency_weight,
                ir_hot_contrast_low=vif_config.ir_hot_contrast_low,
                ir_hot_contrast_high=vif_config.ir_hot_contrast_high,
                ir_hot_weight=vif_config.ir_hot_weight,
                ir_dark_context_weight=vif_config.ir_dark_context_weight,
                ir_edge_weight=vif_config.ir_edge_weight,
                visible_support_kernel=vif_config.intensity_visible_support_kernel,
                weight_smoothing_kernel=(vif_config.intensity_weight_smoothing_kernel),
            )
            if vif_config.intensity_mode == "hot_object_aware":
                hot_infrared_reference, intensity_weight, hotness = (
                    hot_object_ir_blend_weight(
                        visible,
                        infrared,
                        max_weight=vif_config.ir_intensity_max_weight,
                        darkness_threshold=vif_config.ir_darkness_threshold,
                        darkness_transition=vif_config.ir_darkness_transition,
                        saliency_weight=vif_config.ir_saliency_weight,
                        hot_contrast_low=vif_config.ir_hot_contrast_low,
                        hot_contrast_high=vif_config.ir_hot_contrast_high,
                        hot_weight=vif_config.ir_hot_weight,
                        dark_context_weight=vif_config.ir_dark_context_weight,
                        edge_weight=vif_config.ir_edge_weight,
                        visible_support_kernel=(
                            vif_config.intensity_visible_support_kernel
                        ),
                        smoothing_kernel=(
                            vif_config.intensity_weight_smoothing_kernel
                        ),
                        energy_normalization=(
                            vif_config.intensity_energy_normalization
                        ),
                    )
                )
                hot_target_reference = target_intensity
            final_vif = vif_losses(
                output.fused,
                visible,
                infrared,
                output.fused_y,
                target_intensity=target_intensity,
                intensity_mode=vif_config.intensity_mode,
                intensity_energy_normalization=vif_config.intensity_energy_normalization,
                ir_intensity_max_weight=vif_config.ir_intensity_max_weight,
                ir_darkness_threshold=vif_config.ir_darkness_threshold,
                ir_darkness_transition=vif_config.ir_darkness_transition,
                ir_saliency_weight=vif_config.ir_saliency_weight,
                ir_hot_contrast_low=vif_config.ir_hot_contrast_low,
                ir_hot_contrast_high=vif_config.ir_hot_contrast_high,
                ir_hot_weight=vif_config.ir_hot_weight,
                ir_dark_context_weight=vif_config.ir_dark_context_weight,
                ir_edge_weight=vif_config.ir_edge_weight,
                intensity_visible_support_kernel=(
                    vif_config.intensity_visible_support_kernel
                ),
                intensity_weight_smoothing_kernel=(
                    vif_config.intensity_weight_smoothing_kernel
                ),
                intensity_weight=intensity_weight,
                gradient_mode=vif_config.gradient_mode,
                ir_gradient_dominance_ratio=vif_config.ir_gradient_dominance_ratio,
                visible_gradient_support_kernel=(
                    vif_config.visible_gradient_support_kernel
                ),
                gradient_transition=vif_config.gradient_transition,
                gradient_min_magnitude=vif_config.gradient_min_magnitude,
                gradient_charbonnier_epsilon=(
                    vif_config.gradient_charbonnier_epsilon
                ),
                ssim_mode=vif_config.ssim_mode,
            )
            components.update(final_vif)
            weights.update(
                {
                    "fusion/intensity": vif_config.intensity,
                    "fusion/gradient": vif_config.gradient,
                    "fusion/ssim": vif_config.ssim,
                    "fusion/color": vif_config.color,
                }
            )
            if (
                hotness is not None
                and hot_infrared_reference is not None
                and vif_config.hot_underexposure_weight > 0
            ):
                predicted_y = (
                    output.fused_y
                    if output.fused_y is not None
                    else luminance(output.fused)
                )
                components["fusion/hot_underexposure"] = hot_underexposure_loss(
                    predicted_y,
                    visible,
                    hot_infrared_reference,
                    hotness,
                    minimum_contrast_retention=(
                        vif_config.hot_minimum_contrast_retention
                    ),
                )
                weights["fusion/hot_underexposure"] = (
                    vif_config.hot_underexposure_weight
                )
            if output.coarse is not None:
                coarse = (
                    final_vif
                    if output.coarse is output.fused
                    and output.coarse_y is output.fused_y
                    else vif_losses(
                        output.coarse,
                        visible,
                        infrared,
                        output.coarse_y,
                        target_intensity=target_intensity,
                        intensity_mode=vif_config.intensity_mode,
                        intensity_energy_normalization=(
                            vif_config.intensity_energy_normalization
                        ),
                        ir_intensity_max_weight=vif_config.ir_intensity_max_weight,
                        ir_darkness_threshold=vif_config.ir_darkness_threshold,
                        ir_darkness_transition=vif_config.ir_darkness_transition,
                        ir_saliency_weight=vif_config.ir_saliency_weight,
                        ir_hot_contrast_low=vif_config.ir_hot_contrast_low,
                        ir_hot_contrast_high=vif_config.ir_hot_contrast_high,
                        ir_hot_weight=vif_config.ir_hot_weight,
                        ir_dark_context_weight=vif_config.ir_dark_context_weight,
                        ir_edge_weight=vif_config.ir_edge_weight,
                        intensity_visible_support_kernel=(
                            vif_config.intensity_visible_support_kernel
                        ),
                        intensity_weight_smoothing_kernel=(
                            vif_config.intensity_weight_smoothing_kernel
                        ),
                        intensity_weight=intensity_weight,
                        gradient_mode=vif_config.gradient_mode,
                        ir_gradient_dominance_ratio=(
                            vif_config.ir_gradient_dominance_ratio
                        ),
                        visible_gradient_support_kernel=(
                            vif_config.visible_gradient_support_kernel
                        ),
                        gradient_transition=vif_config.gradient_transition,
                        gradient_min_magnitude=vif_config.gradient_min_magnitude,
                        gradient_charbonnier_epsilon=(
                            vif_config.gradient_charbonnier_epsilon
                        ),
                        ssim_mode=vif_config.ssim_mode,
                    )
                )
                components["fusion/coarse"] = (
                    coarse["fusion/intensity"] + coarse["fusion/gradient"]
                )
                weights["fusion/coarse"] = vif_config.coarse_supervision
        elif context.task is TaskType.MFIF:
            if batch.target is None:
                self._missing("fusion/mfif", "MFIF fused target is missing", skipped)
            else:
                components.update(
                    mfif_losses(
                        output.fused, batch.target, self.config.mfif.use_charbonnier
                    )
                )
                weights.update(
                    {
                        "fusion/reconstruction": self.config.mfif.reconstruction,
                        "fusion/gradient": self.config.mfif.gradient,
                        "fusion/ssim": self.config.mfif.ssim,
                    }
                )
                if output.coarse is not None:
                    components["fusion/coarse"] = mfif_losses(
                        output.coarse, batch.target, self.config.mfif.use_charbonnier
                    )["fusion/reconstruction"]
                    weights["fusion/coarse"] = self.config.mfif.coarse_supervision
            self._focus(context, components, weights, skipped)
        else:
            self._semantic(context, components, weights, skipped)
        self._shared(
            context,
            components,
            weights,
            skipped,
            hotness=hotness,
            hot_target_reference=hot_target_reference,
        )
        weighted = {
            name: value * weight * self._phase_multiplier(name, context)
            for name, value in components.items()
            if (weight := weights.get(name, 1.0)) != 0
        }
        if not weighted:
            raise RuntimeError("No active differentiable losses were produced")
        total = torch.stack(tuple(weighted.values())).sum()
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite total loss")
        diagnostics = {
            "task": context.task.value,
            "phase": context.phase,
            "component_count": len(components),
            "router_blocks": len(output.router_diagnostics),
        }
        for name in (
            "chroma_cb_error",
            "chroma_cr_error",
            "y_gamut_clip_ratio",
            "coarse_final_y_mae",
        ):
            if name in output.debug:
                value = output.debug[name]
                diagnostics[name] = (
                    value.detach() if isinstance(value, Tensor) else value
                )
        if context.task is TaskType.VIF:
            visible, infrared = self._visible_ir(batch)
            visible_ssim, infrared_ssim = vif_ssim_scores(
                output.fused_y if output.fused_y is not None else output.fused,
                visible,
                infrared,
            )
            diagnostics.update(
                {
                    "vif/ssim_visible": visible_ssim.detach(),
                    "vif/ssim_infrared": infrared_ssim.detach(),
                    "vif/ssim_mode": self.config.vif.ssim_mode,
                    "vif/gradient_mode": self.config.vif.gradient_mode,
                    "vif/intensity_mode": self.config.vif.intensity_mode,
                    "y_gradient_loss": components["fusion/gradient"].detach(),
                }
            )
            if intensity_weight is not None:
                diagnostics.update(
                    {
                        "ir_intensity_weight_mean": intensity_weight.mean().detach(),
                        "ir_intensity_weight_max": intensity_weight.amax().detach(),
                        "ir_intensity_weight_active_ratio": (
                            (intensity_weight > 0).float().mean().detach()
                        ),
                    }
                )
            if hotness is not None:
                diagnostics.update(
                    {
                        "ir_hotness_mean": hotness.mean().detach(),
                        "ir_hotness_active_ratio": (
                            (hotness > 0).float().mean().detach()
                        ),
                    }
                )
            cross_modal = output.debug.get("cross_modal", ())
            for index, values in enumerate(cross_modal, start=1):
                infrared_weight = values.get("weight_b")
                if isinstance(infrared_weight, Tensor):
                    diagnostics[f"cross_modal_ir_weight/s{index}"] = (
                        infrared_weight.detach().mean()
                    )
            ir_importance, ir_hard_load, ir_contribution = [], [], []
            for item in output.router_diagnostics:
                names = item.auxiliary.get("expert_names", ())
                if "infrared_saliency" not in names:
                    continue
                expert_index = names.index("infrared_saliency")
                if item.importance is not None:
                    ir_importance.append(item.importance[expert_index].detach())
                if item.hard_load is not None:
                    ir_hard_load.append(item.hard_load[expert_index].detach())
                contributions = item.auxiliary.get(
                    "expert_weighted_contribution_rms", {}
                )
                contribution = contributions.get("infrared_saliency")
                if isinstance(contribution, Tensor):
                    ir_contribution.append(contribution.detach())
            if ir_importance:
                diagnostics["router_ir_importance"] = torch.stack(ir_importance).mean()
            if ir_hard_load:
                diagnostics["router_ir_hard_load"] = torch.stack(ir_hard_load).mean()
            if ir_contribution:
                diagnostics["router_ir_weighted_contribution_rms"] = torch.stack(
                    ir_contribution
                ).mean()
        return LossOutput(total, components, weighted, diagnostics, skipped)

    def _focus(self, context, components, weights, skipped) -> None:
        focus, target = context.output.focus, context.batch.focus_target
        if focus is None or target is None:
            self._missing(
                "focus", "MFIF focus logits or focus target are missing", skipped
            )
            return
        components.update(
            focus_losses(focus.selection_logits, focus.confidence, target)
        )
        weights.update(
            {
                "focus/selection": self.config.focus.selection,
                "focus/boundary": self.config.focus.boundary,
                "focus/confidence": self.config.focus.confidence,
            }
        )

    def _semantic(self, context, components, weights, skipped) -> None:
        if self.config.seg_fusion.enabled:
            visible, infrared = self._visible_ir(context.batch)
            components.update(
                seg_fusion_anchor_losses(
                    context.output.fused,
                    visible,
                    infrared,
                    context.output.fused_y,
                )
            )
            weights.update(
                {
                    "seg_fusion/intensity": self.config.seg_fusion.intensity,
                    "seg_fusion/gradient": self.config.seg_fusion.gradient,
                }
            )
        target, segmentation = (
            context.batch.segmentation_target,
            context.output.segmentation,
        )
        if target is None or segmentation is None or not segmentation.available:
            self._missing(
                "semantic", "SEG label or final semantic prediction is missing", skipped
            )
            return
        class_weights = None
        if self.config.semantic.class_weights is not None:
            class_weights = segmentation.logits.new_tensor(
                self.config.semantic.class_weights
            )
        components.update(
            semantic_losses(
                segmentation.logits,
                target,
                (
                    segmentation.boundary
                    if self.config.semantic.boundary_alignment > 0
                    else None
                ),
                context.output.fused,
                self.config.semantic.ignore_index,
                class_weights,
            )
        )
        weights.update(
            {
                "semantic/ce": self.config.semantic.cross_entropy,
                "semantic/dice": self.config.semantic.dice,
                "semantic/boundary": self.config.semantic.boundary_alignment,
            }
        )
        coarse = context.output.coarse_segmentation
        if (
            self.config.semantic.coarse_supervision > 0
            and coarse is not None
            and context.output.coarse is not None
        ):
            components["semantic/coarse"] = semantic_losses(
                coarse.logits,
                target,
                coarse.boundary,
                context.output.coarse,
                self.config.semantic.ignore_index,
                class_weights,
            )["semantic/ce"]
            weights["semantic/coarse"] = self.config.semantic.coarse_supervision
        if self.config.semantic.improvement_enabled and "semantic/coarse" in components:
            components["semantic/improvement"] = torch.relu(
                components["semantic/ce"]
                - components["semantic/coarse"].detach()
                + self.config.semantic.improvement_margin
            )
            weights["semantic/improvement"] = self.config.semantic.improvement_weight

    def _shared(
        self,
        context,
        components,
        weights,
        skipped,
        *,
        hotness: Tensor | None = None,
        hot_target_reference: Tensor | None = None,
    ) -> None:
        output = context.output
        if output.router_diagnostics and (
            self.config.frequency.enabled
            or (self.config.infrared.enabled and context.task is TaskType.VIF)
        ):
            specialization = frequency_specialization(output.router_diagnostics)
            if self.config.frequency.enabled:
                for name in (
                    "frequency/low_leakage",
                    "frequency/detail_leakage",
                    "frequency/semantic_boundary",
                ):
                    components[name] = specialization[name]
                weights.update(
                    {
                        "frequency/low_leakage": self.config.frequency.weight
                        * self.config.frequency.low_leakage,
                        "frequency/detail_leakage": self.config.frequency.weight
                        * self.config.frequency.detail_leakage,
                        "frequency/semantic_boundary": self.config.frequency.weight
                        * self.config.frequency.semantic_boundary,
                    }
                )
            if self.config.infrared.enabled and context.task is TaskType.VIF:
                components["infrared/saliency_alignment"] = specialization[
                    "infrared/saliency_alignment"
                ]
                weights["infrared/saliency_alignment"] = (
                    self.config.infrared.weight
                    * self.config.infrared.saliency_alignment
                )
        elif self.config.frequency.enabled:
            skipped["frequency"] = "No router diagnostics"
        if self.config.moe.enabled and output.router_diagnostics:
            soft_balance, switch_balance, router_entropy = moe_balance_loss(
                output.router_diagnostics
            )
            components.update(
                {
                    "moe/soft_balance": soft_balance,
                    "moe/switch_balance": switch_balance,
                }
            )
            weights.update(
                {
                    "moe/soft_balance": self.config.moe.weight
                    * self.config.moe.soft_balance_weight,
                    "moe/switch_balance": self.config.moe.weight
                    * self.config.moe.switch_balance_weight,
                }
            )
            if self.config.moe.entropy_enabled:
                components["moe/entropy_target"] = (
                    router_entropy - self.config.moe.entropy_target
                ).square()
                weights["moe/entropy_target"] = (
                    self.config.moe.weight * self.config.moe.entropy_weight
                )
        if self.config.infrared.enabled and context.task is TaskType.VIF:
            predicted_y = (
                output.fused_y
                if output.fused_y is not None
                else luminance(output.fused)
            )
            if hotness is not None and hot_target_reference is not None:
                pixel_error = (
                    predicted_y - hot_target_reference.to(predicted_y)
                ).abs()
                numerator = (pixel_error * hotness.to(predicted_y)).sum(dim=(-2, -1))
                denominator = hotness.to(predicted_y).sum(dim=(-2, -1)).clamp_min(1e-6)
                components["infrared/preservation"] = (numerator / denominator).mean()
            else:
                _, infrared = self._visible_ir(context.batch)
                saliency = next(
                    (
                        item.auxiliary.get("ir_saliency")
                        for item in output.router_diagnostics
                        if item.auxiliary.get("ir_saliency") is not None
                    ),
                    None,
                )
                if saliency is None:
                    saliency = luminance(infrared)
                if self.config.infrared.detach_saliency:
                    saliency = saliency.detach()
                saliency = torch.nn.functional.interpolate(
                    saliency,
                    output.fused.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                components["infrared/preservation"] = (
                    (predicted_y - luminance(infrared)).abs() * saliency
                ).mean()
            weights["infrared/preservation"] = self.config.infrared.weight
        paired = context.aux.get("paired_output")
        if self.config.consistency.enabled and paired is not None:
            components["consistency/task_lowfreq"] = low_frequency_consistency(
                output.fused,
                paired.fused,
                self.config.consistency.gaussian_kernel_size,
                self.config.consistency.gaussian_sigma,
            )
            weights["consistency/task_lowfreq"] = self.config.consistency.weight
        components["regularization/residual"] = residual_magnitude(
            output.refinement, output.fused
        )
        weights["regularization/residual"] = (
            self.config.regularization.residual_magnitude
        )
        components["regularization/range"] = range_penalty(
            output.debug.get("final_preclamp", output.fused)
        )
        weights["regularization/range"] = self.config.regularization.range_penalty

    def _missing(self, key: str, message: str, skipped: dict[str, str]) -> None:
        if self.config.strict_targets:
            raise ContractError(message)
        skipped[key] = message

    @staticmethod
    def _visible_ir(batch):
        return batch.visible_source.image, batch.infrared_source.image

    @staticmethod
    def _phase_multiplier(name: str, context: LossContext) -> float:
        namespace = name.split("/", 1)[0]
        return context.loss_multipliers.get(
            name, context.loss_multipliers.get(namespace, 1.0)
        )
