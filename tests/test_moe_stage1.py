from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tfs_moe_fusion.config import MoEConfig, load_config
from tfs_moe_fusion.moe import (
    ExpertContext,
    FunctionalMoEBlock,
    MoEExecutionPolicy,
    RouterContext,
)
from tfs_moe_fusion.types import ExpertType, ModalityType, TaskType

ROOT = Path(__file__).resolve().parents[1]
ALL_EXPERTS = [expert.value for expert in ExpertType]
SPECIALISTS = ALL_EXPERTS[1:]


def _block() -> FunctionalMoEBlock:
    config = MoEConfig(
        architecture_version="v2",
        routing_mode="global_soft",
        common_always_on=True,
        experts=ALL_EXPERTS,
        top_k=2,
        spatial_gating=False,
        router_hidden_channels=8,
        expert_expansion=1,
        noisy_topk=False,
    )
    return FunctionalMoEBlock(8, config, "stage1.s2.moe0").eval()


def _contexts(
    feature: torch.Tensor, has_ir: bool = True
) -> tuple[RouterContext, ExpertContext]:
    modality_a = ModalityType.VISIBLE_RGB if has_ir else ModalityType.GENERIC_RGB
    modality_b = ModalityType.INFRARED_GRAY if has_ir else ModalityType.GENERIC_RGB
    task = TaskType.VIF if has_ir else TaskType.MFIF
    source_a, source_b = torch.randn_like(feature), torch.randn_like(feature)
    return (
        RouterContext(
            feature,
            task,
            modality_a,
            modality_b,
            low_energy=torch.rand(feature.shape[0]),
            high_energy=torch.rand(feature.shape[0]),
            source_a=source_a,
            source_b=source_b,
        ),
        ExpertContext(
            task,
            modality_a,
            modality_b,
            source_a_feature=source_a,
            source_b_feature=source_b,
        ),
    )


def test_stage1_config_enables_only_the_new_architecture_switches() -> None:
    legacy = load_config(ROOT / "configs/default.yaml")
    stage1 = load_config(ROOT / "configs/shared_pool_stage1_global_soft.yaml")
    assert (legacy.model.moe.architecture_version, legacy.model.moe.routing_mode) == (
        "legacy",
        "legacy_topk",
    )
    assert legacy.model.moe.common_always_on is False
    assert (stage1.model.moe.architecture_version, stage1.model.moe.routing_mode) == (
        "v2",
        "global_soft",
    )
    assert stage1.model.moe.common_always_on is True
    assert stage1.model.moe.spatial_gating is False
    assert stage1.training.losses.moe.enabled is False
    assert stage1.training.losses.frequency.enabled is True
    assert stage1.training.losses.frequency.weight == pytest.approx(0.01)


def test_common_always_on_is_absent_from_router_logits() -> None:
    block = _block()
    feature = torch.randn(2, 8, 5, 7)
    output = block(feature, *_contexts(feature))
    assert block.expert_names == tuple(SPECIALISTS)
    assert ExpertType.COMMON not in block.router.expert_types
    assert output.router.logits.shape == (2, len(SPECIALISTS))
    assert output.router.spatial_gates is None
    torch.testing.assert_close(block.common_scale, torch.full_like(block.common_scale, 0.1))
    torch.testing.assert_close(
        block.specialist_scale, torch.full_like(block.specialist_scale, 0.03)
    )


def test_specialist_probabilities_sum_to_one() -> None:
    block = _block()
    feature = torch.randn(2, 8, 5, 7)
    output = block(feature, *_contexts(feature, has_ir=False))
    infrared = SPECIALISTS.index(ExpertType.INFRARED_SALIENCY.value)
    torch.testing.assert_close(output.router.probabilities.sum(1), torch.ones(2))
    assert torch.count_nonzero(output.router.probabilities[:, infrared]) == 0


def test_global_soft_actual_mixture_ignores_topk_execution() -> None:
    block = _block().train()
    block.set_execution_policy(MoEExecutionPolicy("sparse_batch"))
    feature = torch.randn(2, 8, 5, 7)
    output = block(feature, *_contexts(feature))
    weights = output.diagnostics.auxiliary["specialist_mixture_weights"]
    assert torch.equal(weights, output.router.probabilities.detach().to(weights))
    assert (weights > 0).sum(1).tolist() == [len(SPECIALISTS)] * feature.shape[0]
    assert output.router.topk_indices.shape[1] == 2
    assert output.diagnostics.auxiliary["expert_sample_assignments"] == (
        feature.shape[0] * (1 + len(SPECIALISTS))
    )


def test_zero_common_and_specialist_scales_make_identity() -> None:
    block = _block()
    with torch.no_grad():
        block.common_scale.zero_()
        block.specialist_scale.zero_()
    feature = torch.randn(2, 8, 5, 7)
    output = block(feature, *_contexts(feature))
    assert torch.equal(output.feature, feature)
    assert torch.count_nonzero(output.residual) == 0


def test_all_valid_specialists_receive_dense_soft_gradients() -> None:
    torch.manual_seed(223)
    block = _block().train()
    feature = torch.randn(2, 8, 5, 7, requires_grad=True)
    block(feature, *_contexts(feature)).feature.square().mean().backward()
    for name, expert in block.expert_pool.modules_by_name.items():
        gradient_sum = sum(
            float(parameter.grad.detach().abs().sum())
            for parameter in expert.parameters()
            if parameter.grad is not None
        )
        assert gradient_sum > 0, name


def test_global_soft_router_receives_nonzero_gradient() -> None:
    torch.manual_seed(227)
    block = _block().train()
    feature = torch.randn(2, 8, 5, 7, requires_grad=True)
    block(feature, *_contexts(feature)).feature.square().mean().backward()
    gradient_sum = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in block.router.parameters()
        if parameter.grad is not None
    )
    assert gradient_sum > 0


@pytest.mark.parametrize("mode", ["learned", "uniform", "shuffled", "single:detail"])
def test_stage0_specialist_overrides_work_in_v2(mode: str) -> None:
    block = _block()
    block.set_routing_override(mode, seed=3407)
    feature = torch.randn(2, 8, 5, 7)
    output = block(feature, *_contexts(feature))
    weights = output.diagnostics.auxiliary["specialist_mixture_weights"]
    torch.testing.assert_close(weights.sum(1), torch.ones(feature.shape[0]))
    assert output.spatial_gates is None


def test_single_common_disables_all_specialists() -> None:
    block = _block()
    block.set_routing_override("single:common")
    feature = torch.randn(2, 8, 5, 7)
    router_context, expert_context = _contexts(feature)
    output = block(feature, router_context, expert_context)
    with torch.no_grad():
        common = block.common_expert(feature, expert_context).residual
        expected = feature + block.common_scale * common
    assert torch.count_nonzero(
        output.diagnostics.auxiliary["specialist_mixture_weights"]
    ) == 0
    assert output.diagnostics.auxiliary["expert_sample_assignments"] == feature.shape[0]
    torch.testing.assert_close(output.feature, expected)
