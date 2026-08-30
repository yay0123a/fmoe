from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tfs_moe_fusion.backbone import CustomMultiscaleBackbone
from tfs_moe_fusion.config import load_config
from tfs_moe_fusion.data import (
    DeterministicDummyFusionDataset,
    collate_fusion_samples,
)
from tfs_moe_fusion.model import build_model
from tfs_moe_fusion.types import (
    ConfigurationError,
    ContractError,
    ModalityType,
    TaskType,
)
from tfs_moe_fusion.utils import configure_logging

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/default.yaml"


def test_detailed_training_logs_can_be_hidden_from_terminal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_path = tmp_path / "train.log"
    logger = configure_logging(log_file=log_path)
    try:
        logger.info("visible message")
        logger.info("detailed message", extra={"terminal": False})
        for handler in logger.handlers:
            handler.flush()
        terminal = capsys.readouterr().err
        file_text = log_path.read_text(encoding="utf-8")
        assert "visible message" in terminal
        assert "detailed message" not in terminal
        assert "visible message" in file_text
        assert "detailed message" in file_text
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


def test_default_config_builds_final_model() -> None:
    config = load_config(DEFAULT_CONFIG)
    config.model.backbone.channels = [8, 16, 32, 64]
    config.model.backbone.depths = [1, 1, 1, 1]
    config.model.frequency.fdconv_kernel_num = 4
    config.model.moe.router_hidden_channels = 16
    config.model.moe.expert_expansion = 1
    config.model.guidance.focus.hidden_channels = 8
    config.model.guidance.semantic.input_size = 32
    config.model.feedback.guide_channels = 8
    model = build_model(config)
    assert isinstance(model.core, CustomMultiscaleBackbone)
    assert model.core.is_scientific_model is True


def test_unknown_config_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "schema_version: 1\nexperiment:\n  unknown_switch: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="unknown_switch"):
        load_config(path)


def test_ground_truth_router_guidance_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad_guidance.yaml"
    path.write_text(
        f"_base_: {DEFAULT_CONFIG}\n"
        "model:\n"
        "  guidance:\n"
        "    focus:\n"
        "      predicted_guidance_only: false\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="Ground-truth guidance"):
        load_config(path)


@pytest.mark.parametrize("task", list(TaskType))
def test_dummy_dataset_is_deterministic(task: TaskType) -> None:
    dataset = DeterministicDummyFusionDataset(task, length=4, height=15, width=17)
    first = dataset[2]
    repeated = dataset[2]
    assert torch.equal(first.source_a, repeated.source_a)
    assert torch.equal(first.source_b, repeated.source_b)
    assert first.sample_id == repeated.sample_id


def test_dummy_modalities_match_task() -> None:
    vif = DeterministicDummyFusionDataset(TaskType.VIF)[0]
    mfif = DeterministicDummyFusionDataset(TaskType.MFIF)[0]
    assert vif.modality_b is ModalityType.INFRARED_GRAY
    assert not mfif.modality_a.is_infrared
    assert not mfif.modality_b.is_infrared


def test_collate_produces_task_homogeneous_batch() -> None:
    dataset = DeterministicDummyFusionDataset(
        TaskType.SEG, length=3, height=19, width=21
    )
    batch = collate_fusion_samples([dataset[0], dataset[1]])
    assert batch.task is TaskType.SEG
    assert batch.batch_size == 2
    assert batch.segmentation_target is not None
    assert batch.segmentation_target.shape == (2, 19, 21)


def test_collate_rejects_mixed_tasks() -> None:
    vif = DeterministicDummyFusionDataset(TaskType.VIF)[0]
    seg = DeterministicDummyFusionDataset(TaskType.SEG)[0]
    with pytest.raises(ContractError, match="exactly one task"):
        collate_fusion_samples([vif, seg])
