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
CITYSCAPES_COLORS = np.array(
    [
        [128, 64, 128],  # road
        [244, 35, 232],  # sidewalk
        [70, 70, 70],  # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],  # traffic light
        [220, 220, 0],  # traffic sign
        [107, 142, 35],  # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],  # sky
        [220, 20, 60],  # person
        [255, 0, 0],  # rider
        [0, 0, 142],  # car
        [0, 0, 70],  # truck
        [0, 60, 100],  # bus
        [0, 80, 100],  # train
        [0, 0, 230],  # motorcycle
        [119, 11, 32],  # bicycle
    ],
    dtype=np.uint8,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage7_adaptive_ir.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/stage7_msrs_adaptive_ir/checkpoints/latest.pt"),
    )
    parser.add_argument("--input-a", type=Path,default=Path("data/msrs/test/vi/00918N.png")) #vi
    parser.add_argument("--input-b", type=Path,default=Path("data/msrs/test/ir/00918N.png")) #ir
    #parser.add_argument("--input-a", type=Path,default=Path("data/mfif/semantic_rt/dof_stack/img_00125/0.jpg")) #n
    #parser.add_argument("--input-b", type=Path,default=Path("data/mfif/semantic_rt/dof_stack/img_00125/1.jpg")) #f
    parser.add_argument("--output", type=Path, default=Path("runs/stage7"))
    parser.add_argument(
        "--task", required=True, choices=[item.value for item in TaskType]
    )
    parser.add_argument("--modality-a", choices=[item.value for item in ModalityType])
    parser.add_argument("--modality-b", choices=[item.value for item in ModalityType])
    parser.add_argument(
        "--device",
        #default=None,
        default="cuda:5",
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
        if task is TaskType.SEG:
            if output.segmentation is None or not output.segmentation.available:
                raise RuntimeError(
                    "SEG task did not produce segmentation output; enable "
                    "model.guidance.semantic.enabled and set final_pass_policy "
                    "to 'seg_only' or 'all'"
                )
            probabilities = output.segmentation.probabilities
            if probabilities.shape[1] != len(CITYSCAPES_COLORS):
                raise ValueError("Segmentation export requires 19 Cityscapes classes")
            pred = probabilities.argmax(dim=1)
            mask = pred[0].detach().cpu().numpy().astype(np.uint8)
            seg_path = destination.with_name(f"{destination.stem}_seg.png")
            Image.fromarray(mask).save(seg_path)
            color_path = destination.with_name(f"{destination.stem}_seg_color.png")
            Image.fromarray(CITYSCAPES_COLORS[mask]).save(color_path)
            logger.info("Saved segmentation %s", seg_path)
            logger.info("Saved segmentation visualization %s", color_path)
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
