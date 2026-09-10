"""Train the final TFS-MoE-Fusion model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from tfs_moe_fusion.losses import LossContext, MultiTaskLossManager
from tfs_moe_fusion.model import build_model
from tfs_moe_fusion.trainer import Trainer, load_checkpoint
from tfs_moe_fusion.types import TaskType
from tfs_moe_fusion.utils import (
    configure_logging,
    make_probe_batch,
    prepare_run,
    resolve_distributed_device,
    seed_everything,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage15_dark_ir_three_stage.yaml"))
    parser.add_argument(
        "--device",
        default=None,
        help="Override training.device from the config (for example cpu or cuda:1)",
    )
    parser.add_argument("--max-steps", type=int)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", type=Path)
    checkpoint_group.add_argument(
        "--init-checkpoint", type=Path,
        help="Initialize model weights only; start optimizer, loss weights and steps fresh",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--task", default="vif", choices=[task.value for task in TaskType]
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rank = int(os.environ.get("RANK", "0"))
    config, run_dir = prepare_run(args.config, save_resolved=rank == 0)
    if args.init_checkpoint and config.training.checkpoint.resume:
        raise ValueError("--init-checkpoint cannot be combined with checkpoint.resume")
    logger = configure_logging(log_file=run_dir / "train.log" if rank == 0 else None)
    if rank != 0:
        logger.disabled = True
    seed_everything(config.experiment.seed, config.experiment.deterministic)
    device = resolve_distributed_device(
        args.device or config.training.device, config.training.distributed.enabled
    )
    if args.dry_run:
        config.model.guidance.semantic.enabled = False
        config.training.losses.strict_targets = False
        logger.info(
            "Dry-run uses an engineering probe with semantic weights disabled"
        )

    model = build_model(config).to(device)
    if args.init_checkpoint:
        load_checkpoint(args.init_checkpoint, model, map_location=device)
        logger.info("Initialized model weights from %s (fresh training state)", args.init_checkpoint)

    if args.dry_run:
        task = TaskType.parse(args.task)
        batch = make_probe_batch(config, task).to(device)
        model.train()
        output = model(batch)
        loss_manager = MultiTaskLossManager(config.training.losses).to(device)
        loss = loss_manager(
            LossContext(batch, output, task, 0, 0, model)
        ).total
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in (*model.parameters(), *loss_manager.parameters())
        ):
            raise RuntimeError("Dry-run produced a non-finite gradient")
        logger.info(
            "Dry-run succeeded task=%s input=%s output=%s device=%s loss=%.6f",
            task.value,
            tuple(batch.source_a.image.shape),
            tuple(output.fused.shape),
            device,
            float(loss.detach()),
        )
        return

    trainer = Trainer(model, config, device, run_dir, logger)
    resume = args.resume or config.training.checkpoint.resume
    if resume:
        trainer.resume(resume)
        logger.info(
            "Resumed training from %s at step=%d", resume, trainer.state.global_step
        )
    state = trainer.train(args.max_steps)
    logger.info(
        "Training completed at epoch=%d step=%d", state.epoch, state.global_step
    )


if __name__ == "__main__":
    main()
