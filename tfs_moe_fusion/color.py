"""Differentiable color transforms used by fusion heads and losses."""

from __future__ import annotations

import torch
from torch import Tensor


def luminance(image: Tensor) -> Tensor:
    """Return BT.601 luminance while preserving ``[B,1,H,W]`` inputs."""
    if image.ndim != 4:
        raise ValueError("luminance expects [B,C,H,W]")
    if image.shape[1] == 1:
        return image
    if image.shape[1] != 3:
        raise ValueError("luminance requires one or three channels")
    weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (image * weights).sum(1, keepdim=True)


def rgb_to_ycbcr(image: Tensor) -> Tensor:
    """Convert full-range RGB in channel-first layout to BT.601 YCbCr."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("YCbCr conversion requires RGB [B,3,H,W] input")
    r, g, b = image.unbind(1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return torch.stack((y, cb, cr), 1)


def ycbcr_to_rgb(image: Tensor) -> Tensor:
    """Convert full-range BT.601 YCbCr to RGB without clipping channels."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("RGB conversion requires YCbCr [B,3,H,W] input")
    y, cb, cr = image.unbind(1)
    cb = cb - 0.5
    cr = cr - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack((r, g, b), 1)


def compose_luminance_with_visible_chroma(
    predicted_y: Tensor, visible_rgb: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Compose predicted Y with visible chroma using a gamut-safe Y projection.

    Projecting Y onto the legal interval for the fixed visible Cb/Cr keeps RGB in
    gamut without independently clipping R/G/B, which would alter chroma.

    Returns ``(rgb, safe_y, projection_mask)``.
    """
    if predicted_y.ndim != 4 or predicted_y.shape[1] != 1:
        raise ValueError("predicted_y must be [B,1,H,W]")
    if visible_rgb.ndim != 4 or visible_rgb.shape[1] != 3:
        raise ValueError("visible_rgb must be [B,3,H,W]")
    if (
        predicted_y.shape[0] != visible_rgb.shape[0]
        or predicted_y.shape[-2:] != visible_rgb.shape[-2:]
    ):
        raise ValueError("predicted_y and visible_rgb must share batch/spatial shape")

    visible_ycc = rgb_to_ycbcr(visible_rgb)
    cb = visible_ycc[:, 1:2] - 0.5
    cr = visible_ycc[:, 2:3] - 0.5
    offsets = torch.cat(
        (
            1.402 * cr,
            -0.344136 * cb - 0.714136 * cr,
            1.772 * cb,
        ),
        dim=1,
    )
    lower = (-offsets).amax(dim=1, keepdim=True).clamp_min(0.0)
    upper = (1.0 - offsets).amin(dim=1, keepdim=True).clamp_max(1.0)
    # Visible chroma comes from an in-gamut RGB image, so the interval exists.
    # Avoid a tensor-to-host assertion here because this runs on every GPU forward.
    upper = torch.maximum(upper, lower)
    safe_y = torch.maximum(torch.minimum(predicted_y, upper), lower)
    projection_mask = ((predicted_y < lower) | (predicted_y > upper)).to(
        predicted_y.dtype
    )
    rgb = safe_y + offsets
    return rgb, safe_y, projection_mask
