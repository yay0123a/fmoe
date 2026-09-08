"""Deterministic image-space evidence for thermal-aware spatial routing."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from tfs_moe_fusion.color import luminance
from tfs_moe_fusion.types import ModalityType, TaskType


def align_infrared_luminance(visible: Tensor, infrared: Tensor) -> Tensor:
    """Match IR mean and contrast to visible luminance per sample."""

    visible_y, infrared_y = luminance(visible.float()), luminance(infrared.float())
    dims = (-2, -1)
    visible_mean = visible_y.mean(dims, keepdim=True)
    infrared_mean = infrared_y.mean(dims, keepdim=True)
    visible_std = visible_y.std(dims, keepdim=True, unbiased=False)
    infrared_std = infrared_y.std(dims, keepdim=True, unbiased=False)
    scale = (visible_std / infrared_std.clamp_min(1e-3)).clamp(0.25, 4.0)
    return ((infrared_y - infrared_mean) * scale + visible_mean).clamp(0, 1)


def _replicated_sobel_magnitude(image: Tensor) -> Tensor:
    kernel_x = image.new_tensor(
        ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))
    ).view(1, 1, 3, 3) / 4
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kernel_x)
    gy = F.conv2d(padded, kernel_x.transpose(-1, -2))
    return torch.sqrt(gx.square() + gy.square() + 1e-6) - 1e-3


def _soft_saliency(value: Tensor, floor: float = 1e-3) -> Tensor:
    scale = value.mean((-2, -1), keepdim=True).clamp_min(floor)
    return 1 - torch.exp(-value / scale)


def build_router_ir_evidence(
    source_a: Tensor,
    source_b: Tensor,
    modality_a: ModalityType,
    modality_b: ModalityType,
    task: TaskType,
    *,
    local_contrast_kernel: int = 9,
    edge_dominance_ratio: float = 1.2,
    edge_transition: float = 0.15,
    edge_min_magnitude: float = 0.02,
) -> Tensor:
    """Return signed advantage, thermal saliency, and IR-only edge maps."""

    batch, _, height, width = source_a.shape
    if task not in {TaskType.VIF, TaskType.SEG}:
        return source_a.new_zeros(batch, 3, height, width)
    if modality_a is ModalityType.INFRARED_GRAY:
        infrared, visible = source_a, source_b
    elif modality_b is ModalityType.INFRARED_GRAY:
        infrared, visible = source_b, source_a
    else:
        return source_a.new_zeros(batch, 3, height, width)

    with torch.autocast(device_type=source_a.device.type, enabled=False):
        visible_y = luminance(visible.detach().float())
        infrared_y = align_infrared_luminance(visible_y, infrared.detach().float())
        difference = infrared_y - visible_y
        difference_scale = difference.abs().mean(
            (-2, -1), keepdim=True
        ).clamp_min(1e-3)
        signed_advantage = torch.tanh(difference / difference_scale)

        padding = local_contrast_kernel // 2
        local_mean = F.avg_pool2d(
            F.pad(
                infrared_y,
                (padding, padding, padding, padding),
                mode="replicate",
            ),
            local_contrast_kernel,
            stride=1,
        )
        local_contrast = _soft_saliency((infrared_y - local_mean).abs())
        novelty = _soft_saliency(difference.abs())
        thermal_saliency = torch.sqrt(
            (local_contrast * novelty).clamp_min(0) + 1e-8
        ).clamp(0, 1)

        visible_edge = _replicated_sobel_magnitude(visible_y).clamp_min(0)
        infrared_edge = _replicated_sobel_magnitude(infrared_y).clamp_min(0)
        visible_support = F.max_pool2d(
            visible_edge, kernel_size=3, stride=1, padding=1
        )
        dominance = infrared_edge / visible_support.clamp_min(edge_min_magnitude)
        dominance_gate = torch.sigmoid(
            (dominance - edge_dominance_ratio) / edge_transition
        )
        magnitude_gate = torch.sigmoid(
            (infrared_edge - edge_min_magnitude) / edge_min_magnitude
        )
        ir_only_edge = (dominance_gate * magnitude_gate).clamp(0, 1)
        return torch.cat(
            (signed_advantage, thermal_saliency, ir_only_edge), dim=1
        ).to(source_a)
