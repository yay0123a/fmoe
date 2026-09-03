"""Recompose a fused RGB image from its luminance and visible-image chroma."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--visible", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--comparison",
        type=Path,
        help="Optional left-to-right visible/fused/recomposed comparison image",
    )
    return parser


def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    red, green, blue = np.moveaxis(image, -1, 0)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = 0.5 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 0.5 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    return np.stack((luminance, cb, cr), axis=-1)


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    luminance, cb, cr = np.moveaxis(image, -1, 0)
    red = luminance + 1.402 * (cr - 0.5)
    green = luminance - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)
    blue = luminance + 1.772 * (cb - 0.5)
    return np.stack((red, green, blue), axis=-1)


def recompose_visible_chroma(
    fused: np.ndarray, visible: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    fused_ycbcr = rgb_to_ycbcr(fused)
    visible_ycbcr = rgb_to_ycbcr(visible)
    recomposed_ycbcr = np.concatenate(
        (fused_ycbcr[..., :1], visible_ycbcr[..., 1:]), axis=-1
    )
    unclipped = ycbcr_to_rgb(recomposed_ycbcr)
    return np.clip(unclipped, 0.0, 1.0), unclipped


def diagnostics(
    fused: np.ndarray,
    visible: np.ndarray,
    recomposed: np.ndarray,
    unclipped: np.ndarray,
) -> dict[str, float]:
    fused_ycbcr = rgb_to_ycbcr(fused)
    visible_ycbcr = rgb_to_ycbcr(visible)
    recomposed_ycbcr = rgb_to_ycbcr(recomposed)
    gradient_y, gradient_x = np.gradient(fused_ycbcr[..., 0])
    magnitude = np.hypot(gradient_x, gradient_y)
    threshold = float(np.quantile(magnitude, 0.9))
    edge_mask = magnitude >= threshold

    fused_chroma_error = np.abs(
        fused_ycbcr[..., 1:] - visible_ycbcr[..., 1:]
    ).mean(axis=-1)
    recomposed_chroma_error = np.abs(
        recomposed_ycbcr[..., 1:] - visible_ycbcr[..., 1:]
    ).mean(axis=-1)

    def purple_excess(image: np.ndarray) -> np.ndarray:
        return np.maximum(np.minimum(image[..., 0], image[..., 2]) - image[..., 1], 0)

    return {
        "fused_visible_chroma_mae": float(fused_chroma_error.mean()),
        "recomposed_visible_chroma_mae": float(recomposed_chroma_error.mean()),
        "fused_visible_edge_chroma_mae": float(fused_chroma_error[edge_mask].mean()),
        "recomposed_visible_edge_chroma_mae": float(
            recomposed_chroma_error[edge_mask].mean()
        ),
        "recomposed_fused_luminance_mae": float(
            np.abs(recomposed_ycbcr[..., 0] - fused_ycbcr[..., 0]).mean()
        ),
        "out_of_range_pixel_fraction": float(
            np.any((unclipped < 0.0) | (unclipped > 1.0), axis=-1).mean()
        ),
        "fused_edge_purple_excess": float(purple_excess(fused)[edge_mask].mean()),
        "recomposed_edge_purple_excess": float(
            purple_excess(recomposed)[edge_mask].mean()
        ),
    }


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(values, mode="RGB").save(path)


def main() -> None:
    args = build_parser().parse_args()
    fused, visible = load_rgb(args.fused), load_rgb(args.visible)
    if fused.shape != visible.shape:
        raise ValueError(
            f"Fused and visible images must have equal shapes, got "
            f"{fused.shape} and {visible.shape}"
        )
    recomposed, unclipped = recompose_visible_chroma(fused, visible)
    save_rgb(recomposed, args.output)
    if args.comparison is not None:
        comparison = np.concatenate((visible, fused, recomposed), axis=1)
        save_rgb(comparison, args.comparison)
    print(json.dumps(diagnostics(fused, visible, recomposed, unclipped), indent=2))
    if args.comparison is not None:
        print("comparison_order=visible,fused,recomposed")


if __name__ == "__main__":
    main()
