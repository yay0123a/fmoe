from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from tfs_moe_fusion.backbone import CrossModalFusionBlock
from tfs_moe_fusion.config import CrossModalConfig, load_config
from tfs_moe_fusion.cross_modal import (
    BidirectionalWindowCrossAttention,
    InteractionAdaptiveFusion,
    OAFBlockLite,
)
from tfs_moe_fusion.model import build_model
from tfs_moe_fusion.types import ConfigurationError, TaskType
from tfs_moe_fusion.utils import make_probe_batch

ROOT = Path(__file__).resolve().parents[1]


def small_config():
    config = load_config(ROOT / "configs/stage15_dark_ir_three_stage.yaml")
    config.model.backbone.channels = [8, 16, 32]
    config.model.backbone.depths = [1, 1, 1]
    config.model.moe.expert_dim = 16
    config.model.moe.router_hidden_channels = 8
    config.model.moe.expert_expansion = 1
    config.model.guidance.focus.hidden_channels = 8
    config.validate()
    return config


@pytest.mark.parametrize("shape", [(2, 8, 9, 13), (1, 8, 1, 1), (1, 8, 8, 8)])
@pytest.mark.parametrize("heads", [None, 2])
def test_iacf_forward_backward_and_scalar_diagnostics(shape, heads):
    torch.manual_seed(7)
    module = InteractionAdaptiveFusion(8, heads)
    a, b = (torch.randn(shape, requires_grad=True) for _ in range(2))
    result, diagnostics = module(a, b, (a + b) / 2)
    assert result.shape == a.shape and torch.isfinite(result).all()
    assert all(
        v.ndim == 0 and not v.requires_grad and torch.isfinite(v)
        for v in diagnostics.values()
    )
    weights = [
        diagnostics[f"oaf_weight_{name}"] for name in OAFBlockLite.operation_names
    ]
    torch.testing.assert_close(sum(weights), torch.tensor(1.0))
    if heads is None:
        assert module.relation is None and module.cross_logit is None
        assert "relation_mean" not in diagnostics  # Disabled, not fabricated zeros.
    else:
        assert 0 < diagnostics["relation_mean"] < 1
        assert diagnostics["cross_update_a_rms"] > 0
        assert diagnostics["cross_update_b_rms"] > 0
        torch.testing.assert_close(diagnostics["cross_scale"], torch.tensor(0.1))
    torch.testing.assert_close(diagnostics["oaf_scale"], torch.tensor(0.1))
    result.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()
    )
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (a, b))


def test_padded_window_attention_matches_unpadded_reference():
    torch.manual_seed(3)
    module = BidirectionalWindowCrossAttention(8, 2, 8).eval()
    a, b = torch.randn(2, 2, 8, 3, 5).unbind(0)
    actual = module(a, b)
    qkv = [
        module.qkv(module.norm(x.flatten(2).transpose(1, 2)))
        .reshape(2, 15, 3, 2, 4)
        .permute(2, 0, 3, 1, 4)
        .unbind(0)
        for x in (a, b)
    ]
    for i in range(2):
        q = qkv[i][0]
        k, v = qkv[1 - i][1:]
        result = F.scaled_dot_product_attention(q, k, v)
        result = module.projection(result.transpose(1, 2).reshape(2, 15, 8))
        torch.testing.assert_close(
            actual[i], result.transpose(1, 2).reshape(2, 8, 3, 5)
        )


def test_windows_and_batch_are_independent():
    module = BidirectionalWindowCrossAttention(8, 2, 4).eval()
    a, b = torch.randn(2, 2, 8, 8, 8).unbind(0)
    da, db = module(a, b)
    for sample in range(2):
        for row in (0, 4):
            for col in (0, 4):
                region = (
                    slice(sample, sample + 1),
                    slice(None),
                    slice(row, row + 4),
                    slice(col, col + 4),
                )
                expected = module(a[region], b[region])
                torch.testing.assert_close(da[region], expected[0])
                torch.testing.assert_close(db[region], expected[1])


def test_oaf_constant_hpf_and_feature_dependent_six_way_mixture():
    torch.manual_seed(4)
    module = OAFBlockLite(8)
    a = torch.full((2, 8, 3, 5), 2.0)
    b = torch.randn_like(a)
    assert torch.count_nonzero(module.high_pass(a)) == 0
    output, weights = module(a, b)
    candidates = [module.high_pass(x) for x in (a, b)]
    expected = sum(
        weights[:, i * 3 + j, None, None, None] * value
        for i, x in enumerate((a, b))
        for j, value in enumerate(
            (candidates[i], x + module.add(x), x * module.multiply(x).sigmoid())
        )
    )
    torch.testing.assert_close(output, expected)
    _, other_weights = module(b, a * 3)
    assert not torch.allclose(weights, other_weights)


def test_anchor_and_previous_path_are_preserved():
    baseline = CrossModalFusionBlock(8)
    module = CrossModalFusionBlock(8, CrossModalConfig(enabled=True), "s2")
    missing, unexpected = module.load_state_dict(baseline.state_dict(), strict=False)
    assert missing and all(name.startswith("interaction.") for name in missing)
    assert not unexpected
    a, b, previous = (torch.randn(2, 8, 9, 11) for _ in range(3))
    # Limit beta -> 0 must recover the exact historical fusion, including previous/refine.
    with torch.no_grad():
        module.interaction.oaf_logit.fill_(-100)
    torch.testing.assert_close(module(a, b, previous)[0], baseline(a, b, previous)[0])
    previous.requires_grad_()
    module(a, b, previous)[0].mean().backward()
    assert torch.count_nonzero(previous.grad)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_forward_backward(device):
    if device == "cuda" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("CUDA BF16 is unavailable")
    module = InteractionAdaptiveFusion(8, 2).to(device)
    a, b = (
        torch.randn(2, 8, 9, 11, device=device, requires_grad=True) for _ in range(2)
    )
    with torch.autocast(device, dtype=torch.bfloat16):
        output, diagnostics = module(a, b, (a + b) / 2)
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(value) for value in diagnostics.values())
    output.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("heads", {"s4": 8}),
        ("heads", {"s2": 5}),
        ("window_size", 0),
        ("alpha_init", 0),
        ("beta_init", 1),
    ],
)
def test_invalid_iacf_config(field, value):
    config = load_config(ROOT / "configs/stage15_dark_ir_three_stage.yaml")
    setattr(config.model.backbone.cross_modal, field, value)
    with pytest.raises(ConfigurationError, match="cross_modal"):
        config.validate()


@pytest.mark.parametrize("task", list(TaskType))
def test_three_stage_iacf_task_smoke_preserves_source_evidence(task):
    config = small_config()
    model = build_model(config).train()
    captured = {}

    def capture(_module, args):
        captured["a"], captured["b"] = args[:2]

    handle = model.core.cross_modal_fusions[-1].register_forward_pre_hook(capture)
    batch = make_probe_batch(config, task, spatial_size=(17, 19))
    core = model.core(batch)
    handle.remove()
    assert core.source_a.s3 is captured["a"] and core.source_b.s3 is captured["b"]
    output = model(batch)
    assert output.fused.shape == (1, 3, 17, 19) and torch.isfinite(output.fused).all()
    diagnostics = output.debug["cross_modal"]
    assert "relation_mean" not in diagnostics[0]
    assert all("relation_mean" in stage for stage in diagnostics[1:])
    output.fused.square().mean().backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    for fusion in model.core.cross_modal_fusions:
        assert fusion.interaction.oaf_logit.grad.abs() > 0
        if fusion.interaction.attention is not None:
            assert fusion.interaction.cross_logit.grad.abs() > 0


def test_trainer_updates_iacf_and_emits_scalar_diagnostics(
    tmp_path, semantic_rt_assets
):
    from tfs_moe_fusion.trainer import Trainer

    config = small_config()
    config.data.dataset = "semantic_rt"
    config.data.root, config.data.mfif_root, config.data.manifest = map(
        str, semantic_rt_assets
    )
    config.data.crop_size = 16
    config.data.num_workers = 0
    config.data.pin_memory = False
    config.training.batch_size = 1
    config.training.precision = "bf16"
    # The real 2000-step warmup starts below the FP32 ULP of a ~-2.2 logit.
    config.training.scheduler.warmup_steps = 0
    config.training.ema.enabled = False
    config.training.losses.strict_targets = False
    trainer = Trainer(build_model(config), config, torch.device("cpu"), tmp_path)
    parameter = trainer.raw_model.core.cross_modal_fusions[1].interaction.oaf_logit
    before = parameter.detach().clone()
    try:
        result = trainer.train_step(TaskType.VIF)
        assert torch.isfinite(result.total)
        assert not torch.equal(parameter, before)
        metrics = {k: v for k, v in result.diagnostics.items() if k.startswith("iacf/")}
        assert "iacf/s2/relation_mean" in metrics
        assert "iacf/s1/oaf_weight_b_hpf" in metrics
        assert "iacf/s1/relation_mean" not in metrics
        assert all(
            torch.as_tensor(v).ndim == 0 and torch.isfinite(torch.as_tensor(v))
            for v in metrics.values()
        )
    finally:
        trainer.provider.close()
