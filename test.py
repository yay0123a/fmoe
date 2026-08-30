"""Fuse paired images with a trained TFS-MoE-Fusion checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tfs_moe_fusion.config import load_config
from tfs_moe_fusion.model import build_model
from tfs_moe_fusion.trainer import load_checkpoint
from tfs_moe_fusion.types import FusionBatch, ModalityType, SourceBatch, TaskType
from tfs_moe_fusion.utils import configure_logging, make_probe_batch, resolve_device

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input-a", type=Path)
    parser.add_argument("--input-b", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/test"))
    parser.add_argument(
        "--task", required=True, choices=[item.value for item in TaskType]
    )
    parser.add_argument("--modality-a", choices=[item.value for item in ModalityType])
    parser.add_argument("--modality-b", choices=[item.value for item in ModalityType])
    parser.add_argument(
        "--device",
        default=None,
        help="Override training.device from the config (for example cpu or cuda:1)",
    )
    parser.add_argument("--save-coarse", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device or config.training.device)
    logger = configure_logging()
    if args.dry_run:
        config.model.guidance.semantic.enabled = False
        logger.info(
            "Dry-run uses an engineering probe with semantic weights disabled"
        )
    model = build_model(config).to(device).eval()

    if args.checkpoint is not None:
        report = load_checkpoint(args.checkpoint, model, map_location=device)
        logger.info(
            "Loaded checkpoint epoch=%d step=%d", report.epoch, report.global_step
        )
    elif not args.dry_run:
        raise ValueError("--checkpoint is required unless --dry-run is used")

    task = TaskType.parse(args.task)
    if args.dry_run:
        batch = make_probe_batch(config, task).to(device)
        with torch.no_grad():
            output = model(batch)
        logger.info(
            "Dry-run succeeded task=%s output=%s", task.value, tuple(output.fused.shape)
        )
        return

    if args.input_a is None or args.input_b is None:
        raise ValueError("--input-a and --input-b are required")
    modality_a, modality_b = _modalities(task, args.modality_a, args.modality_b)
    pairs = _paired_paths(args.input_a, args.input_b)
    single_output = len(pairs) == 1 and args.output.suffix.lower() in IMAGE_SUFFIXES

    for path_a, path_b in pairs:
        image_a = _load_image(path_a, modality_a)
        image_b = _load_image(path_b, modality_b)
        batch = FusionBatch(
            SourceBatch(image_a, modality_a),
            SourceBatch(image_b, modality_b),
            task,
            (path_a.stem,),
        ).to(device)
        with torch.no_grad():
            output = model(batch)
        destination = (
            args.output if single_output else args.output / f"{path_a.stem}.png"
        )
        _save_image(output.fused, destination)
        if args.save_coarse and output.coarse is not None:
            coarse = destination.with_name(f"{destination.stem}_coarse.png")
            _save_image(output.coarse, coarse)
        logger.info("Saved %s", destination)


def _modalities(
    task: TaskType, first: str | None, second: str | None
) -> tuple[ModalityType, ModalityType]:
    if first and second:
        return ModalityType.parse(first), ModalityType.parse(second)
    if first or second:
        raise ValueError("Specify both --modality-a and --modality-b, or neither")
    if task is TaskType.MFIF:
        return ModalityType.GENERIC_RGB, ModalityType.GENERIC_RGB
    return ModalityType.VISIBLE_RGB, ModalityType.INFRARED_GRAY


def _paired_paths(first: Path, second: Path) -> list[tuple[Path, Path]]:
    if first.is_file() and second.is_file():
        return [(first, second)]
    if not first.is_dir() or not second.is_dir():
        raise ValueError("Inputs must both be files or both be directories")
    left = {
        path.stem: path
        for path in first.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    }
    right = {
        path.stem: path
        for path in second.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    }
    names = sorted(left.keys() & right.keys())
    if not names:
        raise ValueError("Input directories contain no images with matching stems")
    return [(left[name], right[name]) for name in names]


def _load_image(path: Path, modality: ModalityType) -> torch.Tensor:
    mode = "RGB" if modality.channels == 3 else "L"
    array = np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    image = tensor.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    values = np.rint(image * 255.0).astype(np.uint8)
    if values.shape[-1] == 1:
        values = values[..., 0]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values).save(path)


if __name__ == "__main__":
    main()
