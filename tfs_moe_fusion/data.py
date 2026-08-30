"""TFS-MoE-Fusion consolidated implementation."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from tfs_moe_fusion.types import FusionSample, TaskType


class FusionDatasetAdapter(Dataset[FusionSample], ABC):
    task: TaskType

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> FusionSample:
        raise NotImplementedError


import torch
from torch import Tensor

from tfs_moe_fusion.types import FusionSample, ModalityType


class DeterministicDummyFusionDataset(FusionDatasetAdapter):
    """Generate reproducible tensors solely for infrastructure tests."""

    def __init__(
        self,
        task: TaskType | str,
        length: int = 16,
        height: int = 128,
        width: int = 128,
        seed: int = 3407,
    ) -> None:
        self.task = TaskType.parse(task)
        if length <= 0 or height <= 0 or width <= 0:
            raise ValueError("length, height, and width must be positive")
        self.length = length
        self.height = height
        self.width = width
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> FusionSample:
        if not 0 <= index < self.length:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(
            self.seed + self.task.index * 1_000_003 + index
        )
        base = torch.rand((3, self.height, self.width), generator=generator)

        if self.task is TaskType.MFIF:
            mask = self._focus_mask()
            shifted = torch.roll(base, shifts=1, dims=-1)
            source_a = mask * base + (1.0 - mask) * shifted
            source_b = (1.0 - mask) * base + mask * shifted
            return FusionSample(
                source_a=source_a,
                modality_a=ModalityType.GENERIC_RGB,
                source_b=source_b,
                modality_b=ModalityType.GENERIC_RGB,
                task=self.task,
                sample_id=f"dummy-mfif-{index:05d}",
                target=base,
                focus_target=mask,
                metadata={"fixture": True},
            )

        infrared = base.mean(dim=0, keepdim=True)
        infrared = (
            0.8 * infrared + 0.2 * torch.rand(infrared.shape, generator=generator)
        ).clamp(0.0, 1.0)
        target = 0.5 * base + 0.5 * infrared.expand_as(base)
        segmentation = None
        if self.task is TaskType.SEG:
            segmentation = (infrared[0] > 0.5).to(torch.long)
        return FusionSample(
            source_a=base,
            modality_a=ModalityType.VISIBLE_RGB,
            source_b=infrared,
            modality_b=ModalityType.INFRARED_GRAY,
            task=self.task,
            sample_id=f"dummy-{self.task.value}-{index:05d}",
            target=target,
            segmentation_target=segmentation,
            metadata={"fixture": True},
        )

    def _focus_mask(self) -> Tensor:
        split = max(1, self.width // 2)
        mask = torch.zeros((1, self.height, self.width))
        mask[..., :split] = 1.0
        return mask


SEMANTIC_RT_TO_CITYSCAPES = {
    0: 255,  # background
    1: 255,  # car stop
    2: 18,  # bike -> bicycle (dataset-level approximation)
    3: 12,  # bicyclist -> rider
    4: 17,  # motorcycle
    5: 12,  # motorcyclist -> rider
    6: 13,  # car
    7: 255,  # tricycle
    8: 6,  # traffic light
    9: 255,  # box
    10: 5,  # pole
    11: 255,  # curve
    12: 11,  # person
}
SEMANTIC_RT_MAPPED_RAW_IDS = frozenset(
    raw_id
    for raw_id, cityscapes_id in SEMANTIC_RT_TO_CITYSCAPES.items()
    if cityscapes_id != 255
)


@dataclass(frozen=True, slots=True)
class SynchronizedAugmentationConfig:
    crop_size: int = 256
    horizontal_flip_probability: float = 0.5
    rotation_degrees: float = 10.0
    rotation_probability: float = 0.5
    segmentation_min_valid_pixels: int = 64
    segmentation_crop_attempts: int = 10

    def __post_init__(self) -> None:
        if self.crop_size <= 0:
            raise ValueError("crop_size must be positive")
        for name, value in (
            ("horizontal_flip_probability", self.horizontal_flip_probability),
            ("rotation_probability", self.rotation_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= self.rotation_degrees < 45.0:
            raise ValueError("rotation_degrees must be in [0, 45)")
        if self.segmentation_min_valid_pixels < 0:
            raise ValueError("segmentation_min_valid_pixels cannot be negative")
        if self.segmentation_crop_attempts <= 0:
            raise ValueError("segmentation_crop_attempts must be positive")


class SynchronizedImageAugmentation:
    """Apply one geometric transform to every member of a supervised sample."""

    def __init__(self, config: SynchronizedAugmentationConfig) -> None:
        self.config = config

    def __call__(
        self,
        images: dict[str, Image.Image],
        *,
        categorical: frozenset[str] = frozenset(),
        segmentation_key: str | None = None,
    ) -> dict[str, Image.Image]:
        if not images:
            raise ValueError("Synchronized augmentation requires at least one image")
        sizes = {image.size for image in images.values()}
        if len(sizes) != 1:
            raise ContractError(
                f"All augmented sample members must have one size, got {sizes}"
            )

        canvas_size = self._rotation_canvas_size()
        prepared = {
            name: self._pad_to_canvas(image, canvas_size, name)
            for name, image in images.items()
        }
        width, height = next(iter(prepared.values())).size
        selection_label = (
            prepared[segmentation_key] if segmentation_key is not None else None
        )
        left, top = self._sample_crop(width, height, canvas_size, selection_label)
        box = (left, top, left + canvas_size, top + canvas_size)
        transformed = {name: image.crop(box) for name, image in prepared.items()}

        angle = 0.0
        if (
            self.config.rotation_degrees > 0
            and random.random() < self.config.rotation_probability
        ):
            angle = random.uniform(
                -self.config.rotation_degrees, self.config.rotation_degrees
            )
        if angle:
            transformed = {
                name: image.rotate(
                    angle,
                    resample=(
                        Image.Resampling.NEAREST
                        if name in categorical
                        else Image.Resampling.BILINEAR
                    ),
                    fillcolor=self._fill_value(name),
                )
                for name, image in transformed.items()
            }

        crop_size = self.config.crop_size
        margin = (canvas_size - crop_size) // 2
        final_box = (margin, margin, margin + crop_size, margin + crop_size)
        transformed = {
            name: image.crop(final_box) for name, image in transformed.items()
        }
        if random.random() < self.config.horizontal_flip_probability:
            transformed = {
                name: image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                for name, image in transformed.items()
            }
        return transformed

    def _rotation_canvas_size(self) -> int:
        radians = math.radians(self.config.rotation_degrees)
        scale = abs(math.cos(radians)) + abs(math.sin(radians))
        return max(
            self.config.crop_size,
            math.ceil(self.config.crop_size * scale) + 2,
        )

    def _sample_crop(
        self,
        width: int,
        height: int,
        size: int,
        segmentation: Image.Image | None,
    ) -> tuple[int, int]:
        attempts = (
            self.config.segmentation_crop_attempts if segmentation is not None else 1
        )
        last = (0, 0)
        for _ in range(attempts):
            left = random.randint(0, width - size)
            top = random.randint(0, height - size)
            last = (left, top)
            if segmentation is None:
                return last
            label = np.asarray(segmentation.crop((left, top, left + size, top + size)))
            valid = np.isin(label, tuple(SEMANTIC_RT_MAPPED_RAW_IDS)).sum()
            if valid >= self.config.segmentation_min_valid_pixels:
                return last
        return last

    @staticmethod
    def _fill_value(name: str) -> int | tuple[int, int, int]:
        if name == "segmentation":
            return 255
        return 0

    @classmethod
    def _pad_to_canvas(cls, image: Image.Image, size: int, name: str) -> Image.Image:
        width, height = image.size
        if width >= size and height >= size:
            return image
        target_width, target_height = max(width, size), max(height, size)
        output = Image.new(
            image.mode, (target_width, target_height), cls._fill_value(name)
        )
        output.paste(
            image, ((target_width - width) // 2, (target_height - height) // 2)
        )
        return output


class SemanticRTFusionDataset(FusionDatasetAdapter):
    """SemanticRT adapter shared by VIF, SEG, and generated MFIF tasks."""

    def __init__(
        self,
        task: TaskType | str,
        root: str | Path,
        mfif_root: str | Path,
        manifest: str | Path,
        *,
        augmentation: SynchronizedImageAugmentation | None = None,
        strict_files: bool = True,
    ) -> None:
        self.task = TaskType.parse(task)
        self.root = Path(root)
        self.mfif_root = Path(mfif_root)
        self.manifest = Path(manifest)
        self.augmentation = augmentation
        self.sample_ids = self._load_manifest(self.manifest)
        if strict_files:
            missing = [
                str(path)
                for sample_id in self.sample_ids
                for path in self._required_paths(sample_id)
                if not path.is_file()
            ]
            if missing:
                preview = ", ".join(missing[:5])
                raise FileNotFoundError(
                    f"SemanticRT {self.task.value} has {len(missing)} missing files: "
                    f"{preview}"
                )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> FusionSample:
        sample_id = self.sample_ids[index]
        if self.task is TaskType.MFIF:
            return self._mfif_sample(sample_id)
        return self._infrared_visible_sample(sample_id)

    def _infrared_visible_sample(self, sample_id: str) -> FusionSample:
        images = {
            "source_a": Image.open(self.root / "rgb" / f"{sample_id}.jpg").convert(
                "RGB"
            ),
            "source_b": Image.open(self.root / "thermal" / f"{sample_id}.jpg").convert(
                "L"
            ),
        }
        categorical = frozenset()
        segmentation_key = None
        if self.task is TaskType.SEG:
            images["segmentation"] = Image.open(
                self.root / "labels" / f"{sample_id}.png"
            ).convert("L")
            categorical = frozenset({"segmentation"})
            segmentation_key = "segmentation"
        if self.augmentation is not None:
            images = self.augmentation(
                images,
                categorical=categorical,
                segmentation_key=segmentation_key,
            )
        segmentation = (
            self._map_segmentation(images["segmentation"])
            if self.task is TaskType.SEG
            else None
        )
        return FusionSample(
            source_a=self._image_tensor(images["source_a"]),
            modality_a=ModalityType.VISIBLE_RGB,
            source_b=self._image_tensor(images["source_b"]),
            modality_b=ModalityType.INFRARED_GRAY,
            task=self.task,
            sample_id=sample_id,
            segmentation_target=segmentation,
            metadata={"dataset": "semantic_rt", "manifest": str(self.manifest)},
        )

    def _mfif_sample(self, sample_id: str) -> FusionSample:
        stack = self.mfif_root / "dof_stack" / sample_id
        images = {
            "source_a": Image.open(stack / "0.jpg").convert("RGB"),
            "source_b": Image.open(stack / "1.jpg").convert("RGB"),
            "target": Image.open(self.mfif_root / "AiF" / f"{sample_id}.jpg").convert(
                "RGB"
            ),
            "focus": Image.open(
                self.mfif_root / "quantized_depth" / f"{sample_id}.png"
            ).convert("L"),
        }
        if self.augmentation is not None:
            images = self.augmentation(images, categorical=frozenset({"focus"}))
        focus = torch.from_numpy(
            1.0 - np.asarray(images["focus"], dtype=np.float32).copy() / 255.0
        ).unsqueeze(0)
        return FusionSample(
            source_a=self._image_tensor(images["source_a"]),
            modality_a=ModalityType.GENERIC_RGB,
            source_b=self._image_tensor(images["source_b"]),
            modality_b=ModalityType.GENERIC_RGB,
            task=self.task,
            sample_id=sample_id,
            target=self._image_tensor(images["target"]),
            focus_target=focus.clamp(0.0, 1.0),
            metadata={"dataset": "semantic_rt_mfif", "manifest": str(self.manifest)},
        )

    def _required_paths(self, sample_id: str) -> tuple[Path, ...]:
        if self.task is TaskType.MFIF:
            return (
                self.mfif_root / "dof_stack" / sample_id / "0.jpg",
                self.mfif_root / "dof_stack" / sample_id / "1.jpg",
                self.mfif_root / "AiF" / f"{sample_id}.jpg",
                self.mfif_root / "quantized_depth" / f"{sample_id}.png",
            )
        paths = (
            self.root / "rgb" / f"{sample_id}.jpg",
            self.root / "thermal" / f"{sample_id}.jpg",
        )
        if self.task is TaskType.SEG:
            paths += (self.root / "labels" / f"{sample_id}.png",)
        return paths

    @staticmethod
    def _load_manifest(path: Path) -> tuple[str, ...]:
        if not path.is_file():
            raise FileNotFoundError(f"SemanticRT manifest does not exist: {path}")
        values = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if not values:
            raise ValueError(f"SemanticRT manifest is empty: {path}")
        if len(values) != len(set(values)):
            raise ValueError(f"SemanticRT manifest contains duplicate IDs: {path}")
        return values

    @staticmethod
    def _image_tensor(image: Image.Image) -> Tensor:
        array = np.asarray(image, dtype=np.float32).copy()
        if array.ndim == 2:
            array = array[..., None]
        return torch.from_numpy(array).permute(2, 0, 1) / 255.0

    @staticmethod
    def _map_segmentation(image: Image.Image) -> Tensor:
        raw = np.asarray(image, dtype=np.uint8)
        mapped = np.full(raw.shape, 255, dtype=np.int64)
        for raw_id, cityscapes_id in SEMANTIC_RT_TO_CITYSCAPES.items():
            mapped[raw == raw_id] = cityscapes_id
        return torch.from_numpy(mapped)


from tfs_moe_fusion.types import ContractError, FusionBatch, FusionSample, SourceBatch


def collate_fusion_samples(samples: list[FusionSample]) -> FusionBatch:
    if not samples:
        raise ContractError("Cannot collate an empty sample list")
    task = samples[0].task
    modality_a = samples[0].modality_a
    modality_b = samples[0].modality_b
    if any(sample.task is not task for sample in samples):
        raise ContractError("A training batch must contain exactly one task")
    if any(sample.modality_a is not modality_a for sample in samples):
        raise ContractError("source_a modality must be homogeneous within a batch")
    if any(sample.modality_b is not modality_b for sample in samples):
        raise ContractError("source_b modality must be homogeneous within a batch")

    return FusionBatch(
        source_a=SourceBatch(
            torch.stack([sample.source_a for sample in samples]), modality_a
        ),
        source_b=SourceBatch(
            torch.stack([sample.source_b for sample in samples]), modality_b
        ),
        task=task,
        sample_ids=tuple(sample.sample_id for sample in samples),
        target=_stack_optional([sample.target for sample in samples], "target"),
        focus_target=_stack_optional(
            [sample.focus_target for sample in samples], "focus_target"
        ),
        segmentation_target=_stack_optional(
            [sample.segmentation_target for sample in samples],
            "segmentation_target",
        ),
        metadata=tuple(sample.metadata for sample in samples),
    )


def _stack_optional(
    values: list[torch.Tensor | None], name: str
) -> torch.Tensor | None:
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise ContractError(f"{name} must be present for every sample or none")
    return torch.stack([value for value in values if value is not None])
