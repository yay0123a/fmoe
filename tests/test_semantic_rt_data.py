from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tfs_moe_fusion.config import load_config
from tfs_moe_fusion.data import (
    SemanticRTFusionDataset,
    SynchronizedAugmentationConfig,
    SynchronizedImageAugmentation,
)
from tfs_moe_fusion.trainer import SemanticRTBatchProvider, build_batch_provider
from tfs_moe_fusion.types import ModalityType, TaskType

ROOT = Path(__file__).resolve().parents[1]


def _write_fixture(tmp_path: Path, size: int = 32) -> tuple[Path, Path, Path]:
    root = tmp_path / "semantic_rt"
    mfif = tmp_path / "mfif" / "semantic_rt"
    sample_id = "img_00001"
    for directory in (
        root / "rgb",
        root / "thermal",
        root / "labels",
        mfif / "AiF",
        mfif / "dof_stack" / sample_id,
        mfif / "quantized_depth",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    y, x = np.mgrid[:size, :size]
    rgb = np.stack(((x * 7) % 256, (y * 9) % 256, ((x + y) * 5) % 256), -1).astype(
        np.uint8
    )
    thermal = ((x + 2 * y) * 3 % 256).astype(np.uint8)
    labels = np.resize(np.arange(13, dtype=np.uint8), (size, size))
    focus = np.zeros((size, size), dtype=np.uint8)
    focus[:, size // 2 :] = 255

    Image.fromarray(rgb).save(root / "rgb" / f"{sample_id}.jpg")
    Image.fromarray(thermal).save(root / "thermal" / f"{sample_id}.jpg")
    Image.fromarray(labels).save(root / "labels" / f"{sample_id}.png")
    Image.fromarray(rgb).save(mfif / "AiF" / f"{sample_id}.jpg")
    Image.fromarray(rgb).save(mfif / "dof_stack" / sample_id / "0.jpg")
    Image.fromarray(rgb).save(mfif / "dof_stack" / sample_id / "1.jpg")
    Image.fromarray(focus).save(mfif / "quantized_depth" / f"{sample_id}.png")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"{sample_id}\n", encoding="utf-8")
    return root, mfif, manifest


def test_semantic_rt_vif_modalities_and_shapes(tmp_path: Path) -> None:
    root, mfif, manifest = _write_fixture(tmp_path)
    dataset = SemanticRTFusionDataset(TaskType.VIF, root, mfif, manifest)

    sample = dataset[0]

    assert sample.source_a.shape == (3, 32, 32)
    assert sample.source_b.shape == (1, 32, 32)
    assert sample.modality_a is ModalityType.VISIBLE_RGB
    assert sample.modality_b is ModalityType.INFRARED_GRAY
    assert sample.segmentation_target is None


def test_semantic_rt_segmentation_is_mapped_to_cityscapes(tmp_path: Path) -> None:
    root, mfif, manifest = _write_fixture(tmp_path)
    dataset = SemanticRTFusionDataset(TaskType.SEG, root, mfif, manifest)

    target = dataset[0].segmentation_target

    assert target is not None
    assert target.dtype is torch.long
    assert set(target.unique().tolist()) == {5, 6, 11, 12, 13, 17, 18, 255}


def test_semantic_rt_mfif_inverts_generated_depth_mask(tmp_path: Path) -> None:
    root, mfif, manifest = _write_fixture(tmp_path)
    dataset = SemanticRTFusionDataset(TaskType.MFIF, root, mfif, manifest)

    sample = dataset[0]

    assert sample.target is not None
    assert sample.focus_target is not None
    assert torch.equal(sample.focus_target[:, :, :16], torch.ones(1, 32, 16))
    assert torch.equal(sample.focus_target[:, :, 16:], torch.zeros(1, 32, 16))


def test_semantic_rt_augmentation_is_synchronized_and_categorical(
    tmp_path: Path,
) -> None:
    root, mfif, manifest = _write_fixture(tmp_path, size=48)
    augmentation = SynchronizedImageAugmentation(
        SynchronizedAugmentationConfig(
            crop_size=24,
            horizontal_flip_probability=1.0,
            rotation_degrees=10.0,
            rotation_probability=1.0,
        )
    )
    dataset = SemanticRTFusionDataset(
        TaskType.MFIF, root, mfif, manifest, augmentation=augmentation
    )

    sample = dataset[0]

    assert sample.source_a.shape == (3, 24, 24)
    assert sample.target is not None and sample.target.shape == (3, 24, 24)
    assert torch.equal(sample.source_a, sample.source_b)
    assert torch.equal(sample.source_a, sample.target)
    assert sample.focus_target is not None
    assert set(sample.focus_target.unique().tolist()) <= {0.0, 1.0}


def test_semantic_rt_batch_provider_builds_all_tasks_and_restores_order(
    tmp_path: Path,
) -> None:
    root, mfif, manifest = _write_fixture(tmp_path)
    config = load_config(ROOT / "configs/default.yaml")
    config.data.dataset = "semantic_rt"
    config.data.root = str(root)
    config.data.mfif_root = str(mfif)
    config.data.manifest = str(manifest)
    config.data.crop_size = 16
    config.data.num_workers = 0
    config.training.batch_size = 1

    provider = build_batch_provider(config)

    assert isinstance(provider, SemanticRTBatchProvider)
    for task in TaskType:
        batch = provider.next_batch(task)
        assert batch.task is task
        assert batch.batch_size == 1
        assert batch.spatial_size == (16, 16)

    state = provider.state_dict()
    restored = SemanticRTBatchProvider(config)
    restored.load_state_dict(state)
    for task in TaskType:
        assert (
            restored.next_batch(task).sample_ids == provider.next_batch(task).sample_ids
        )


def test_semantic_rt_provider_uses_dataloader_worker_and_pin_memory_settings(
    tmp_path: Path,
) -> None:
    root, mfif, manifest = _write_fixture(tmp_path)
    config = load_config(ROOT / "configs/default.yaml")
    config.data.root = str(root)
    config.data.mfif_root = str(mfif)
    config.data.manifest = str(manifest)
    config.data.crop_size = 16
    config.data.num_workers = 1
    config.data.pin_memory = True
    config.training.batch_size = 1
    provider = SemanticRTBatchProvider(config)
    try:
        batch = provider.next_batch(TaskType.VIF)
        loader = provider.loaders[TaskType.VIF]
        assert loader.num_workers == 1
        assert loader.pin_memory is True
        assert batch.batch_size == 1
    finally:
        provider.close()


def test_semantic_rt_single_gpu_config_is_valid() -> None:
    config = load_config(ROOT / "configs/semantic_rt_single_gpu.yaml")
    assert config.data.dataset == "semantic_rt"
    assert config.training.distributed.enabled is False
    assert config.training.epochs == 50
    assert config.training.steps_per_epoch == 750
    assert [
        (phase.name, phase.start, phase.end) for phase in config.training.phases.phases
    ] == [
        ("stabilization", 0, 5),
        ("semantic_ramp", 5, 15),
        ("joint", 15, 44),
        ("routing_finetune", 44, 50),
    ]
