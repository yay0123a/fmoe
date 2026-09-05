from __future__ import annotations

import pytest
import torch
from torch import nn

from tfs_moe_fusion.config import MoEConfig
from tfs_moe_fusion.moe import (
    ExpertContext,
    FunctionalMoEBlock,
    RouterContext,
    clear_moe_routing_override,
    iter_moe_blocks,
    set_moe_routing_override,
)
from tfs_moe_fusion.trainer import moe_gradient_statistics
from tfs_moe_fusion.types import ExpertType, ModalityType, TaskType

EXPERT_NAMES = [
    "common",
    "low_frequency",
    "detail",
    "semantic",
    "infrared_saliency",
]


def _block() -> FunctionalMoEBlock:
    config = MoEConfig(
        experts=EXPERT_NAMES,
        top_k=2,
        router_hidden_channels=8,
        expert_expansion=1,
        noisy_topk=False,
    )
    return FunctionalMoEBlock(8, config, "probe.s2.moe0").eval()


def _contexts(
    feature: torch.Tensor, *, has_ir: bool
) -> tuple[RouterContext, ExpertContext]:
    modality_a = ModalityType.VISIBLE_RGB if has_ir else ModalityType.GENERIC_RGB
    modality_b = ModalityType.INFRARED_GRAY if has_ir else ModalityType.GENERIC_RGB
    task = TaskType.VIF if has_ir else TaskType.MFIF
    source_a = torch.randn_like(feature)
    source_b = torch.randn_like(feature)
    router = RouterContext(
        feature=feature,
        task=task,
        modality_a=modality_a,
        modality_b=modality_b,
        low_energy=torch.rand(feature.shape[0]),
        high_energy=torch.rand(feature.shape[0]),
        source_a=source_a,
        source_b=source_b,
    )
    expert = ExpertContext(
        task=task,
        modality_a=modality_a,
        modality_b=modality_b,
        source_a_feature=source_a,
        source_b_feature=source_b,
    )
    return router, expert


def test_explicit_learned_override_matches_default_path_exactly() -> None:
    torch.manual_seed(101)
    block = _block()
    feature = torch.randn(2, 8, 5, 7)
    router_context, expert_context = _contexts(feature, has_ir=True)

    with torch.no_grad():
        default = block(feature, router_context, expert_context)
        block.set_routing_override("learned")
        learned = block(feature, router_context, expert_context)

    assert torch.equal(default.feature, learned.feature)
    assert torch.equal(default.residual, learned.residual)
    assert torch.equal(default.router.probabilities, learned.router.probabilities)
    assert torch.equal(default.router.spatial_gates, learned.router.spatial_gates)


def test_uniform_routing_sums_to_one_over_valid_experts() -> None:
    block = _block()
    block.set_routing_override("uniform")
    feature = torch.randn(2, 8, 5, 7)
    router_context, expert_context = _contexts(feature, has_ir=True)

    output = block(feature, router_context, expert_context)

    expected = torch.full_like(output.router.probabilities, 1 / len(EXPERT_NAMES))
    torch.testing.assert_close(output.router.probabilities, expected)
    torch.testing.assert_close(
        output.router.probabilities.sum(1), torch.ones(feature.shape[0])
    )


def test_uniform_routing_never_weights_invalid_infrared_expert() -> None:
    block = _block()
    block.set_routing_override("uniform")
    feature = torch.randn(2, 8, 5, 7)
    router_context, expert_context = _contexts(feature, has_ir=False)

    output = block(feature, router_context, expert_context)

    infrared = EXPERT_NAMES.index(ExpertType.INFRARED_SALIENCY.value)
    assert not output.router.valid_expert_mask[:, infrared].any()
    assert torch.count_nonzero(output.router.probabilities[:, infrared]) == 0
    torch.testing.assert_close(
        output.router.probabilities.sum(1), torch.ones(feature.shape[0])
    )


def test_shuffled_routing_is_seeded_and_permutes_matching_spatial_gates() -> None:
    torch.manual_seed(103)
    block = _block()
    feature = torch.randn(2, 8, 5, 7)
    router_context, expert_context = _contexts(feature, has_ir=True)
    learned = block(feature, router_context, expert_context)

    block.set_routing_override("shuffled", seed=177)
    first = block(feature, router_context, expert_context)
    block.set_routing_override("shuffled", seed=177)
    second = block(feature, router_context, expert_context)

    permutation = first.router.auxiliary["routing_permutation"]
    assert torch.equal(first.router.probabilities, second.router.probabilities)
    assert torch.equal(first.router.spatial_gates, second.router.spatial_gates)
    assert torch.equal(
        first.router.probabilities,
        learned.router.probabilities.index_select(1, permutation),
    )
    assert torch.equal(
        first.router.spatial_gates,
        learned.router.spatial_gates.index_select(1, permutation),
    )


def test_single_expert_routing_only_executes_requested_expert() -> None:
    block = _block()
    block.set_routing_override("single:detail")
    calls = {name: 0 for name in EXPERT_NAMES}
    handles = [
        expert.register_forward_hook(
            lambda _module, _inputs, _output, name=name: calls.__setitem__(
                name, calls[name] + 1
            )
        )
        for name, expert in block.expert_pool.modules_by_name.items()
    ]
    feature = torch.randn(2, 8, 5, 7)
    router_context, expert_context = _contexts(feature, has_ir=True)

    output = block(feature, router_context, expert_context)
    for handle in handles:
        handle.remove()

    detail = EXPERT_NAMES.index(ExpertType.DETAIL.value)
    expected = torch.zeros_like(output.router.probabilities)
    expected[:, detail] = 1
    assert torch.equal(output.router.probabilities, expected)
    assert calls[ExpertType.DETAIL.value] == 1
    assert sum(calls.values()) == 1


def test_single_invalid_infrared_expert_raises_instead_of_falling_back() -> None:
    block = _block()
    block.set_routing_override("single:infrared_saliency")
    feature = torch.randn(2, 8, 5, 7)
    router_context, expert_context = _contexts(feature, has_ir=False)

    with torch.no_grad(), pytest.raises(ValueError, match="invalid.*sample"):
        block(feature, router_context, expert_context)


def test_weighted_contribution_rms_values_are_finite_and_detached() -> None:
    block = _block()
    block.set_routing_override("uniform")
    feature = torch.randn(2, 8, 5, 7, requires_grad=True)
    router_context, expert_context = _contexts(feature, has_ir=True)

    output = block(feature, router_context, expert_context)
    values = output.diagnostics.auxiliary["expert_weighted_contribution_rms"]

    assert set(values) == set(EXPERT_NAMES)
    assert all(
        torch.isfinite(value) and not value.requires_grad for value in values.values()
    )


def test_model_helpers_control_all_blocks_and_clear_restores_learned() -> None:
    model = nn.ModuleDict({"core": _block(), "feedback": _block()})
    blocks = tuple(iter_moe_blocks(model))
    feature = torch.randn(1, 8, 5, 7)
    router_context, expert_context = _contexts(feature, has_ir=True)
    learned = blocks[0](feature, router_context, expert_context).feature

    set_moe_routing_override(model, "uniform", seed=99)
    assert len(blocks) == 2
    assert all(block.routing_override == "uniform" for block in blocks)
    clear_moe_routing_override(model)
    assert all(block.routing_override == "learned" for block in blocks)
    restored = blocks[0](feature, router_context, expert_context).feature
    assert torch.equal(learned, restored)


def test_moe_gradient_diagnostics_are_named_by_router_and_expert() -> None:
    block = _block()
    feature = torch.randn(2, 8, 5, 7, requires_grad=True)
    router_context, expert_context = _contexts(feature, has_ir=True)
    block(feature, router_context, expert_context).feature.square().mean().backward()

    diagnostics = moe_gradient_statistics(block)

    assert "router_grad_norm" in diagnostics
    assert all(
        f"expert_grad_norm/{expert_name}" in diagnostics
        for expert_name in EXPERT_NAMES
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values())
