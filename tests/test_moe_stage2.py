from __future__ import annotations

from pathlib import Path

import torch

from tfs_moe_fusion.config import load_config
from tfs_moe_fusion.model import build_model
from tfs_moe_fusion.moe import (
    ExpertContext,
    MoESiteAdapter,
    RouterContext,
    SharedExpertBank,
    SharedExpertMoESite,
    TaskEmbedding,
    iter_moe_blocks,
)
from tfs_moe_fusion.trainer import (
    build_optimizer,
    load_checkpoint,
    save_checkpoint,
)
from tfs_moe_fusion.types import ModalityType, TaskType
from tfs_moe_fusion.utils import make_probe_batch

ROOT = Path(__file__).resolve().parents[1]


def _config():
    config = load_config(ROOT / "configs/shared_pool_stage2_global_soft.yaml")
    config.model.backbone.channels = [8, 16, 32, 64]
    config.model.backbone.depths = [1, 1, 1, 1]
    config.model.frequency.fdconv_kernel_num = 4
    config.model.moe.router_hidden_channels = 16
    config.model.moe.expert_expansion = 1
    config.model.guidance.focus.hidden_channels = 8
    config.model.guidance.semantic.enabled = False
    config.model.feedback.guide_channels = 8
    config.training.ema.enabled = False
    return config


def _contexts(
    feature: torch.Tensor,
) -> tuple[RouterContext, ExpertContext]:
    source_a, source_b = torch.randn_like(feature), torch.randn_like(feature)
    return (
        RouterContext(
            feature,
            TaskType.VIF,
            ModalityType.VISIBLE_RGB,
            ModalityType.INFRARED_GRAY,
            low_energy=torch.rand(feature.shape[0]),
            high_energy=torch.rand(feature.shape[0]),
            source_a=source_a,
            source_b=source_b,
        ),
        ExpertContext(
            TaskType.VIF,
            ModalityType.VISIBLE_RGB,
            ModalityType.INFRARED_GRAY,
            source_a_feature=source_a,
            source_b_feature=source_b,
        ),
    )


def test_stage2_config_enables_one_128_channel_shared_pool() -> None:
    config = load_config(ROOT / "configs/shared_pool_stage2_global_soft.yaml")
    assert config.model.moe.shared_pool_enabled is True
    assert config.model.moe.expert_dim == 128
    assert config.model.moe.routing_mode == "global_soft"


def test_shared_bank_has_one_canonical_state_dict_path_per_expert() -> None:
    model = build_model(_config())
    names = tuple(model.state_dict())
    for expert in (
        "common",
        "specialists.low_frequency",
        "specialists.detail",
        "specialists.semantic",
        "specialists.infrared_saliency",
    ):
        paths = [name for name in names if f".{expert}." in name]
        assert paths and all(name.startswith("shared_expert_bank.") for name in paths)


def test_sites_reference_one_bank_but_not_one_router_or_adapter() -> None:
    model = build_model(_config())
    core = model.core.moe_blocks["s2"][0]
    feedback = model.feedback.feedback_moe["s3"]
    assert core.shared_expert_bank is feedback.shared_expert_bank is model.shared_expert_bank
    assert core.router is not feedback.router
    assert core.adapter is not feedback.adapter
    assert {
        id(parameter)
        for parameter in core.router.parameters()
    }.isdisjoint({id(parameter) for parameter in feedback.router.parameters()})
    assert {
        id(parameter)
        for parameter in core.adapter.parameters()
    }.isdisjoint({id(parameter) for parameter in feedback.adapter.parameters()})


def test_shared_detail_parameters_have_one_object_identity() -> None:
    model = build_model(_config())
    detail = model.shared_expert_bank.specialists["detail"]
    assert all(
        block.shared_expert_bank.specialists["detail"] is detail
        for block in iter_moe_blocks(model)
    )
    detail_ids = {id(parameter) for parameter in detail.parameters()}
    assert sum(
        id(parameter) in detail_ids for _, parameter in model.named_parameters()
    ) == len(detail_ids)


def test_site_adapters_project_native_stage_dims_through_128() -> None:
    for channels in (96, 192, 384):
        adapter = MoESiteAdapter(channels, 128)
        feature = torch.randn(1, channels, 3, 5)
        canonical = adapter.feature_in(feature)
        assert canonical.shape == (1, 128, 3, 5)
        assert adapter.source_in(feature).shape == canonical.shape
        assert adapter.delta_out(canonical).shape == feature.shape


def test_zero_scales_make_a_shared_site_identity() -> None:
    config = _config().model.moe
    bank = SharedExpertBank(config.expert_dim, config.expert_expansion)
    site = SharedExpertMoESite(96, config, "s2.moe0", TaskEmbedding(8), bank).eval()
    with torch.no_grad():
        site.common_scale.zero_()
        site.specialist_scale.zero_()
    feature = torch.randn(1, 96, 5, 7)
    output = site(feature, *_contexts(feature))
    assert torch.equal(output.feature, feature)
    assert torch.count_nonzero(output.residual) == 0


def test_shared_sites_keep_v2_routing_overrides() -> None:
    config = _config().model.moe
    bank = SharedExpertBank(config.expert_dim, config.expert_expansion)
    site = SharedExpertMoESite(96, config, "s2.moe0", TaskEmbedding(8), bank).eval()
    feature = torch.randn(1, 96, 5, 7)
    context = _contexts(feature)
    for mode in ("learned", "uniform", "shuffled", "single:detail"):
        site.set_routing_override(mode, seed=3407)
        weights = site(feature, *context).diagnostics.auxiliary[
            "specialist_mixture_weights"
        ]
        torch.testing.assert_close(weights.sum(1), torch.ones(feature.shape[0]))
    site.set_routing_override("single:common")
    assert torch.count_nonzero(
        site(feature, *context).diagnostics.auxiliary["specialist_mixture_weights"]
    ) == 0


def test_shared_optimizer_groups_cover_every_parameter_once() -> None:
    config = _config()
    model = build_model(config)
    optimizer, registry = build_optimizer(model, config.training.optimizer)
    registered = [
        parameter for values in registry.groups.values() for parameter in values
    ]
    optimized = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    assert len({id(parameter) for parameter in registered}) == len(registered)
    assert {id(parameter) for parameter in optimized} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    for group in (
        "shared_common",
        "shared_low",
        "shared_detail",
        "shared_semantic",
        "shared_ir",
        "core_site_adapters",
        "feedback_site_adapters",
        "core_routers",
        "feedback_routers",
    ):
        assert registry.parameters(group), group


def test_shared_pool_checkpoint_round_trip_is_strict(tmp_path: Path) -> None:
    config = _config()
    model = build_model(config)
    checkpoint = save_checkpoint(
        tmp_path / "stage2.pt", model, config, epoch=1, global_step=2
    )
    restored = build_model(config)
    report = load_checkpoint(checkpoint, restored, strict=True)
    assert report.missing_keys == report.unexpected_keys == ()
    for name, value in model.state_dict().items():
        if name.startswith("shared_expert_bank."):
            torch.testing.assert_close(value, restored.state_dict()[name])


def test_legacy_and_stage1_remain_nonshared_and_runnable() -> None:
    for path in ("configs/default.yaml", "configs/shared_pool_stage1_global_soft.yaml"):
        config = load_config(ROOT / path)
        config.model.backbone.channels = [8, 16, 32, 64]
        config.model.backbone.depths = [1, 1, 1, 1]
        config.model.frequency.fdconv_kernel_num = 4
        config.model.moe.router_hidden_channels = 16
        config.model.moe.expert_expansion = 1
        config.model.guidance.focus.hidden_channels = 8
        config.model.guidance.semantic.enabled = False
        config.model.feedback.guide_channels = 8
        model = build_model(config).eval()
        assert not hasattr(model, "shared_expert_bank")
        with torch.no_grad():
            output = model(
                make_probe_batch(config, TaskType.MFIF, spatial_size=(31, 37))
            )
        assert torch.isfinite(output.fused).all()
