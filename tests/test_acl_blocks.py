from pathlib import Path

import pytest
import torch
from torch.nn import functional

from tfs_moe_fusion.acl_blocks import LAMA2D, ACLStage, LinearAttention2D, MDCBlock
from tfs_moe_fusion.backbone import ConvNeXtLikeBlock, CustomMultiscaleBackbone
from tfs_moe_fusion.config import ConfigurationError, load_config
from tfs_moe_fusion.types import TaskType
from tfs_moe_fusion.utils import make_probe_batch

ROOT = Path(__file__).resolve().parents[1]


def test_linear_attention_matches_explicit_kernel_attention() -> None:
    torch.manual_seed(7)
    module = LinearAttention2D(8, 2).eval()
    tensor = torch.randn(1, 8, 3, 5)
    actual = module(tensor)

    tokens = tensor.flatten(2).transpose(1, 2)
    query, key = module.qk(tokens).chunk(2, dim=-1)
    query, key = functional.elu(query) + 1, functional.elu(key) + 1
    query = query.reshape(1, 15, 2, 4).transpose(1, 2)
    key = key.reshape(1, 15, 2, 4).transpose(1, 2)
    values = tokens.reshape(1, 15, 2, 4).transpose(1, 2)
    scores = query @ key.transpose(-2, -1)
    expected = (scores @ values) / scores.sum(dim=-1, keepdim=True)
    expected = expected.transpose(1, 2).reshape(1, 15, 8)
    expected = expected.transpose(1, 2).reshape_as(tensor) + module.lepe(tensor)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "module",
    [LAMA2D(8, 2), MDCBlock(8), ACLStage(8, 2, 2, use_mdc=True)],
)
def test_acl_blocks_preserve_odd_shapes_and_backpropagate(
    module: torch.nn.Module,
) -> None:
    tensor = torch.randn(2, 8, 5, 7, requires_grad=True)
    output = module(tensor)
    assert output.shape == tensor.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_invalid_lama_head_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        LinearAttention2D(10, 3)

    config = load_config(ROOT / "configs/default.yaml")
    config.model.backbone.lama_heads = [2, 4, 6, 8]
    config.model.backbone.channels = [8, 16, 32, 64]
    with pytest.raises(ConfigurationError, match="divisible"):
        config.validate()


def test_backbone_replaces_only_source_stages() -> None:
    config = load_config(ROOT / "configs/default.yaml")
    config.model.backbone.channels = [8, 16, 32, 64]
    config.model.backbone.depths = [1, 1, 1, 1]
    config.model.frequency.enabled = False
    config.model.moe.enabled = False
    backbone = CustomMultiscaleBackbone(
        config.model.backbone.channels,
        config.model.backbone.depths,
        config.model.frequency,
        config.model.output_channels,
        config.model.pad_multiple,
        config.model.moe,
        config.model.backbone,
    )

    assert all(isinstance(stage, ACLStage) for stage in backbone.source_stages)
    assert all(
        isinstance(stage.mdc, MDCBlock) for stage in backbone.source_stages[:2]
    )
    assert all(
        isinstance(stage.mdc, torch.nn.Identity)
        for stage in backbone.source_stages[2:]
    )
    assert all(
        isinstance(stage[0], ConvNeXtLikeBlock) for stage in backbone.fused_stages
    )
    batch = make_probe_batch(config, TaskType.MFIF, spatial_size=(17, 19))
    output = backbone(batch)
    assert output.fused_image.shape == (1, 3, 17, 19)
    assert [feature.shape[-2:] for feature in output.source_a] == [
        (24, 24),
        (12, 12),
        (6, 6),
        (3, 3),
    ]
    output.fused_image.mean().backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_linear_attention_is_finite_under_cuda_bf16_autocast() -> None:
    module = LinearAttention2D(16, 4).cuda().train()
    tensor = torch.randn(1, 16, 9, 11, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = module(tensor)
        loss = output.float().square().mean()
    assert torch.isfinite(output).all()
    loss.backward()
    assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
