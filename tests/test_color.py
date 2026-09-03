from __future__ import annotations

import torch

from tfs_moe_fusion.color import (
    compose_luminance_with_visible_chroma,
    luminance,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)


def test_ycbcr_round_trip_is_numerically_stable() -> None:
    rgb = torch.rand(2, 3, 17, 19)
    restored = ycbcr_to_rgb(rgb_to_ycbcr(rgb))
    torch.testing.assert_close(restored, rgb, rtol=2e-5, atol=2e-5)


def test_gamut_safe_composition_preserves_visible_chroma() -> None:
    visible = torch.rand(2, 3, 17, 19)
    predicted_y = torch.rand(2, 1, 17, 19)
    rgb, safe_y, projection_mask = compose_luminance_with_visible_chroma(
        predicted_y, visible
    )

    assert rgb.min() >= -2e-6
    assert rgb.max() <= 1 + 2e-6
    assert safe_y.shape == projection_mask.shape == predicted_y.shape
    torch.testing.assert_close(luminance(rgb), safe_y, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        rgb_to_ycbcr(rgb)[:, 1:],
        rgb_to_ycbcr(visible)[:, 1:],
        rtol=2e-5,
        atol=2e-5,
    )


def test_gamut_safe_composition_backpropagates_to_unprojected_y() -> None:
    visible = torch.full((1, 3, 5, 7), 0.5)
    predicted_y = torch.full((1, 1, 5, 7), 0.4, requires_grad=True)
    rgb, safe_y, projection_mask = compose_luminance_with_visible_chroma(
        predicted_y, visible
    )

    assert torch.count_nonzero(projection_mask) == 0
    (rgb.mean() + safe_y.mean()).backward()
    assert predicted_y.grad is not None
    assert torch.count_nonzero(predicted_y.grad) == predicted_y.numel()
