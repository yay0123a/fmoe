from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from tfs_moe_fusion.backbone import FeaturePyramid
from tfs_moe_fusion.config import load_config
from tfs_moe_fusion.losses import LossContext, MultiTaskLossManager
from tfs_moe_fusion.model import build_model
from tfs_moe_fusion.trainer import build_optimizer
from tfs_moe_fusion.types import ConfigurationError, TaskType
from tfs_moe_fusion.utils import make_probe_batch

ROOT = Path(__file__).resolve().parents[1]
THREE_STAGE = ROOT / "configs/stage15_dark_ir_three_stage.yaml"


def small_config(stages=3):
    path = THREE_STAGE if stages == 3 else ROOT / "configs/stage15_dark_ir.yaml"
    config = load_config(path)
    config.model.backbone.channels = [8, 16, 32, 64][:stages]
    config.model.backbone.depths = [1] * stages
    # Two channels per GroupNorm group also support a 1x1 deepest feature map.
    config.model.moe.expert_dim = 16
    config.model.moe.router_hidden_channels = 8
    config.model.moe.expert_expansion = 1
    config.model.guidance.focus.hidden_channels = 8
    config.validate()
    return config


@pytest.mark.parametrize("stages", [3, 4])
def test_pyramid_exposes_only_active_scales(stages):
    features = [torch.rand(1, 8, 16 // 2**i, 16 // 2**i) for i in range(stages)]
    pyramid = FeaturePyramid(*features)
    pyramid.validate()
    assert len(pyramid) == stages
    assert list(pyramid.as_dict()) == [f"s{i + 1}" for i in range(stages)]
    assert pyramid[-1] is features[-1]
    assert pyramid[f"s{stages}"] is features[-1]
    if stages == 3:
        assert pyramid.s4 is None
        with pytest.raises(KeyError):
            pyramid["s4"]
        with pytest.raises(IndexError):
            pyramid[3]


@pytest.mark.parametrize("stages", [3, 4])
@pytest.mark.parametrize("task", [TaskType.VIF, TaskType.MFIF])
@pytest.mark.parametrize("shape", [(17, 19), (1, 2)])
def test_fusion_forward_loss_and_optimizer_step(stages, task, shape):
    config = small_config(stages)
    model = build_model(config).train()
    optimizer, _ = build_optimizer(model, config.training.optimizer)
    batch = make_probe_batch(config, task, spatial_size=shape)
    output = model(batch)
    assert output.fused.shape == (1, 3, *shape)
    assert torch.isfinite(output.fused).all()
    expected = [f"s{i}.moe0" for i in range(2, stages + 1)]
    expected += ["feedback.s3.moe0", "feedback.s2.moe0"]
    assert [d.block_id for d in output.router_diagnostics] == expected
    assert model.feedback.semantic_backend is None
    if task is TaskType.VIF:
        assert output.fused_y.shape == (1, 1, *shape)
        assert output.debug["vif_seg_refinement_active"]

    loss = MultiTaskLossManager(config.training.losses)(
        LossContext(batch, output, task, 0, 0, model)
    ).total
    assert torch.isfinite(loss)
    loss.backward()
    for module in (
        model.core.source_stages[-1],
        model.core.moe_blocks["s3"],
        model.shared_expert_bank,
        model.feedback.decoder,
    ):
        assert any(
            p.grad is not None and torch.count_nonzero(p.grad)
            for p in module.parameters()
        )
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    parameter = model.core.source_stages[-1].blocks[0].in_proj.weight
    before = parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(before, parameter)


def test_three_stage_frequency_fallback_uses_only_existing_scales():
    config = small_config()
    config.model.moe.enabled = False
    model = build_model(config)
    assert list(model.core.frequency_blocks) == ["s2", "s3"]
    batch = make_probe_batch(config, TaskType.VIF, spatial_size=(17, 19))
    output = model.core(batch)
    assert len(output.fused) == 3
    assert output.fused_image.shape == (1, 3, 17, 19)
    output.fused_image.mean().backward()


@pytest.mark.parametrize("invalid", ["downsample", "moe", "frequency", "patch", "counts"])
def test_three_stage_config_rejects_stale_four_stage_settings(invalid):
    config = small_config()
    if invalid == "downsample":
        config.model.backbone.max_downsample = 8
    elif invalid == "moe":
        config.model.moe.placements.append("s4")
    elif invalid == "frequency":
        config.model.frequency.placements.append("s4")
    elif invalid == "patch":
        config.model.moe.patch_size["s4"] = 1
    else:
        config.model.moe.block_counts["s4"] = 1
    with pytest.raises(ConfigurationError):
        config.validate()


def test_three_stage_default_is_fresh_training_without_semantics():
    from train import build_parser

    args = build_parser().parse_args([])
    assert args.config.name == THREE_STAGE.name
    config = load_config(THREE_STAGE)
    assert config.training.checkpoint.resume is None
    assert config.training.task_sampling.weights == {"vif": 1.0, "mfif": 1.0}
    assert not config.model.guidance.semantic.enabled
    model = build_model(config)
    assert config.model.backbone.cross_modal.enabled
    baseline_config = load_config(THREE_STAGE)
    baseline_config.model.backbone.cross_modal.enabled = False
    baseline = build_model(baseline_config)
    assert sum(p.numel() for p in baseline.parameters()) == 7_719_172
    extra = sum(
        p.numel() for fusion in model.core.cross_modal_fusions
        for p in fusion.interaction.parameters()
    )
    assert sum(p.numel() for p in model.parameters()) == 7_719_172 + extra
    assert all(p.requires_grad for p in model.parameters())


def test_three_stage_preserves_stage15_training_settings():
    baseline = load_config(ROOT / "configs/stage15_dark_ir.yaml")
    config = load_config(THREE_STAGE)
    assert asdict(config.training) == asdict(baseline.training)
    assert asdict(config.data) == asdict(baseline.data)
    assert config.training.losses.vif.dark_ir_blend == 0.6
    assert config.training.losses.vif.dark_ir_max_gain == 0.3
    assert config.training.losses.vif.glare_ir_weight == 0.2
    assert not config.training.losses.vif.highlight_tone_enabled
