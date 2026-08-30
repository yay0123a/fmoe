"""TFS-MoE-Fusion consolidated implementation."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def luminance(image: Tensor) -> Tensor:
    if image.shape[1] == 1:
        return image
    weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (image[:, :3] * weights).sum(1, keepdim=True)


def rgb_to_ycbcr(image: Tensor) -> Tensor:
    if image.shape[1] != 3:
        raise ValueError("YCbCr conversion requires RGB input")
    r, g, b = image.unbind(1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return torch.stack((y, cb, cr), 1)


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


def ssim(left: Tensor, right: Tensor, size: int = 7, sigma: float = 1.5) -> Tensor:
    if right.shape[1] == 1 and left.shape[1] == 3:
        right = right.expand(-1, 3, -1, -1)
    if left.shape[1] == 1 and right.shape[1] == 3:
        left = left.expand(-1, 3, -1, -1)
    channels = left.shape[1]
    window = gaussian_kernel(size, sigma, left).expand(channels, 1, size, size)
    mu_x = F.conv2d(left, window, padding=size // 2, groups=channels)
    mu_y = F.conv2d(right, window, padding=size // 2, groups=channels)
    sigma_x = (
        F.conv2d(left * left, window, padding=size // 2, groups=channels)
        - mu_x.square()
    )
    sigma_y = (
        F.conv2d(right * right, window, padding=size // 2, groups=channels)
        - mu_y.square()
    )
    sigma_xy = (
        F.conv2d(left * right, window, padding=size // 2, groups=channels) - mu_x * mu_y
    )
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(torch.finfo(left.dtype).eps)
    return score.mean()


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


def vif_losses(fused: Tensor, visible: Tensor, infrared: Tensor) -> dict[str, Tensor]:
    target_intensity = torch.maximum(luminance(visible), luminance(infrared))
    target_gradient = torch.maximum(
        gradient_magnitude(visible), gradient_magnitude(infrared)
    )
    color = fused.new_zeros(())
    if fused.shape[1] == visible.shape[1] == 3:
        color = (rgb_to_ycbcr(fused)[:, 1:] - rgb_to_ycbcr(visible)[:, 1:]).abs().mean()
    return {
        "fusion/intensity": (luminance(fused) - target_intensity).abs().mean(),
        "fusion/gradient": (gradient_magnitude(fused) - target_gradient).abs().mean(),
        "fusion/ssim": 1 - torch.maximum(ssim(fused, visible), ssim(fused, infrared)),
        "fusion/color": color,
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
        target = valid.sum(0) / valid.sum().clamp_min(1)
        importance = item.probabilities.mean(0)
        hard_load = (
            torch.nn.functional.one_hot(item.topk_indices, item.probabilities.shape[1])
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
from tfs_moe_fusion.types import ContractError, ModalityType


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
        output, batch = context.output, context.batch
        if context.task is TaskType.VIF:
            visible, infrared = self._visible_ir(batch)
            components.update(vif_losses(output.fused, visible, infrared))
            weights.update(
                {
                    "fusion/intensity": self.config.vif.intensity,
                    "fusion/gradient": self.config.vif.gradient,
                    "fusion/ssim": self.config.vif.ssim,
                    "fusion/color": self.config.vif.color,
                }
            )
            if output.coarse is not None:
                coarse = vif_losses(output.coarse, visible, infrared)
                components["fusion/coarse"] = (
                    coarse["fusion/intensity"] + coarse["fusion/gradient"]
                )
                weights["fusion/coarse"] = self.config.vif.coarse_supervision
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
        self._shared(context, components, weights, skipped)
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
        if context.task is TaskType.VIF:
            visible, infrared = self._visible_ir(batch)
            diagnostics.update(
                {
                    "vif/ssim_visible": ssim(output.fused, visible).detach(),
                    "vif/ssim_infrared": ssim(output.fused, infrared).detach(),
                }
            )
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
                segmentation.boundary,
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
        if coarse is not None and context.output.coarse is not None:
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

    def _shared(self, context, components, weights, skipped) -> None:
        output = context.output
        if output.router_diagnostics and (
            self.config.frequency.enabled or self.config.infrared.enabled
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
            if self.config.infrared.enabled:
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
        if self.config.infrared.enabled and context.batch.has_infrared:
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
                saliency, output.fused.shape[-2:], mode="bilinear", align_corners=False
            )
            components["infrared/preservation"] = (
                (luminance(output.fused) - luminance(infrared)).abs() * saliency
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
        if batch.source_a.modality is ModalityType.INFRARED_GRAY:
            return batch.source_b.image, batch.source_a.image
        if batch.source_b.modality is ModalityType.INFRARED_GRAY:
            return batch.source_a.image, batch.source_b.image
        raise ContractError("VIF/SEG loss requires a real infrared source")

    @staticmethod
    def _phase_multiplier(name: str, context: LossContext) -> float:
        namespace = name.split("/", 1)[0]
        return context.loss_multipliers.get(
            name, context.loss_multipliers.get(namespace, 1.0)
        )
