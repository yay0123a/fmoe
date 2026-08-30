from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def semantic_rt_assets(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create the real SemanticRT on-disk contract for trainer unit tests."""

    root = tmp_path / "semantic_rt"
    mfif_root = tmp_path / "mfif" / "semantic_rt"
    sample_id = "img_00001"
    for directory in (
        root / "rgb",
        root / "thermal",
        root / "labels",
        mfif_root / "AiF",
        mfif_root / "dof_stack" / sample_id,
        mfif_root / "quantized_depth",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    size = 40
    y, x = np.mgrid[:size, :size]
    rgb = np.stack(
        ((x * 7) % 256, (y * 9) % 256, ((x + y) * 5) % 256), axis=-1
    ).astype(np.uint8)
    thermal = ((x + 2 * y) * 3 % 256).astype(np.uint8)
    labels = np.resize(np.arange(13, dtype=np.uint8), (size, size))
    focus = np.zeros((size, size), dtype=np.uint8)
    focus[:, size // 2 :] = 255

    Image.fromarray(rgb).save(root / "rgb" / f"{sample_id}.jpg")
    Image.fromarray(thermal).save(root / "thermal" / f"{sample_id}.jpg")
    Image.fromarray(labels).save(root / "labels" / f"{sample_id}.png")
    Image.fromarray(rgb).save(mfif_root / "AiF" / f"{sample_id}.jpg")
    Image.fromarray(rgb).save(mfif_root / "dof_stack" / sample_id / "0.jpg")
    Image.fromarray(rgb).save(mfif_root / "dof_stack" / sample_id / "1.jpg")
    Image.fromarray(focus).save(
        mfif_root / "quantized_depth" / f"{sample_id}.png"
    )
    manifest = tmp_path / "semantic_rt_manifest.txt"
    manifest.write_text(f"{sample_id}\n", encoding="utf-8")
    return root, mfif_root, manifest
