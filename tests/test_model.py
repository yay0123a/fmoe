from __future__ import annotations

import pytest
import torch

from tfs_moe_fusion.types import (
    AuxiliaryOutputs,
    ContractError,
    FusionBatch,
    ModalityType,
    SourceBatch,
    TaskType,
)


def test_source_channel_contract() -> None:
    with pytest.raises(ContractError, match="requires 1 channel"):
        SourceBatch(torch.rand(2, 3, 16, 16), ModalityType.INFRARED_GRAY)


def test_vif_requires_real_infrared() -> None:
    rgb = SourceBatch(torch.rand(2, 3, 16, 16), ModalityType.GENERIC_RGB)
    with pytest.raises(ContractError, match="requires one real infrared"):
        FusionBatch(rgb, rgb, TaskType.VIF, ("a", "b"))


def test_mfif_rejects_infrared() -> None:
    rgb = SourceBatch(torch.rand(1, 3, 16, 16), ModalityType.GENERIC_RGB)
    infrared = SourceBatch(torch.rand(1, 1, 16, 16), ModalityType.INFRARED_GRAY)
    with pytest.raises(ContractError, match="MFIF cannot"):
        FusionBatch(rgb, infrared, TaskType.MFIF, ("a",))


def test_source_order_is_not_fixed() -> None:
    rgb = SourceBatch(torch.rand(1, 3, 16, 16), ModalityType.VISIBLE_RGB)
    infrared = SourceBatch(torch.rand(1, 1, 16, 16), ModalityType.INFRARED_GRAY)
    batch = FusionBatch(infrared, rgb, TaskType.VIF, ("reversed",))
    assert batch.has_infrared
    assert batch.visible_source is rgb
    assert batch.infrared_source is infrared


def test_vif_rejects_generic_rgb_as_visible_source() -> None:
    rgb = SourceBatch(torch.rand(1, 3, 16, 16), ModalityType.GENERIC_RGB)
    infrared = SourceBatch(torch.rand(1, 1, 16, 16), ModalityType.INFRARED_GRAY)
    with pytest.raises(ContractError, match="exactly one visible RGB"):
        FusionBatch(rgb, infrared, TaskType.VIF, ("generic",))


from tfs_moe_fusion.backbone import ArbitrarySizePadder


@pytest.mark.parametrize("shape", [(2, 3, 65, 67), (1, 1, 8, 8), (1, 3, 1, 2)])
def test_padding_round_trip(shape: tuple[int, int, int, int]) -> None:
    tensor = torch.rand(shape)
    padder = ArbitrarySizePadder(8)
    padded, info = padder.pad(tensor)
    assert padded.shape[-2] % 8 == 0
    assert padded.shape[-1] % 8 == 0
    restored = padder.unpad(padded, info)
    assert torch.equal(restored, tensor)


def test_tiny_input_uses_safe_replicate_fallback() -> None:
    padder = ArbitrarySizePadder(8)
    _, info = padder.pad(torch.rand(1, 1, 1, 1))
    assert info.mode == "replicate"


import pytest

from tfs_moe_fusion.frequency import (
    AFNO2D,
    CrossFrequencyConditioner,
    FDConv2d,
    FrequencyFoundationBlock,
    HaarDWT2D,
    HaarIDWT2D,
)


@pytest.mark.parametrize("height,width", [(16, 18), (15, 17), (1, 1)])
def test_haar_round_trip_and_energy(height: int, width: int) -> None:
    tensor = torch.randn(2, 3, height, width, requires_grad=True)
    bands = HaarDWT2D()(tensor)
    reconstructed = HaarIDWT2D()(bands)
    torch.testing.assert_close(reconstructed, tensor, rtol=1e-5, atol=1e-6)
    if height % 2 == 0 and width % 2 == 0:
        input_energy = tensor.square().sum()
        band_energy = sum(
            item.square().sum() for item in (bands.ll, bands.lh, bands.hl, bands.hh)
        )
        torch.testing.assert_close(band_energy, input_energy, rtol=1e-5, atol=1e-5)
    reconstructed.mean().backward()
    assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_afno_preserves_shape_dtype_and_unselected_modes() -> None:
    module = AFNO2D(8, num_blocks=4, hard_thresholding_fraction=0.25).eval()
    tensor = torch.randn(2, 8, 15, 17)
    output = module(tensor)
    assert output.shape == tensor.shape
    assert output.dtype == tensor.dtype
    assert torch.isfinite(output).all()
    input_spectrum = torch.fft.rfft2(tensor, norm="ortho")
    output_spectrum = torch.fft.rfft2(output, norm="ortho")
    mask = module._low_mode_mask(15, 9, tensor.device)
    torch.testing.assert_close(
        output_spectrum.masked_select(~mask),
        input_spectrum.masked_select(~mask),
        rtol=2e-4,
        atol=2e-4,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_afno_uses_safe_fp32_fft_for_reduced_precision(dtype: torch.dtype) -> None:
    module = AFNO2D(8, num_blocks=4).to(dtype=dtype)
    tensor = torch.randn(1, 8, 9, 11, dtype=dtype)
    output = module(tensor)
    assert output.dtype == dtype
    assert torch.isfinite(output).all()


def test_afno_disables_bfloat16_autocast_for_complex_reconstruction() -> None:
    module = AFNO2D(8, num_blocks=4)
    tensor = torch.randn(1, 8, 9, 11, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = module(tensor)
        loss = output.square().mean()
    assert output.dtype == tensor.dtype
    assert torch.isfinite(output).all()
    loss.backward()
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()


def test_afno_parameters_receive_gradients_at_sparse_initialization() -> None:
    module = AFNO2D(8, num_blocks=4, sparsity_threshold=0.01)
    module(torch.randn(1, 8, 9, 11)).square().mean().backward()
    assert module.w1_real.grad is not None
    assert torch.count_nonzero(module.w1_real.grad)


def test_fdconv_has_frequency_diverse_kernels_and_backward() -> None:
    module = FDConv2d(4, 4, kernel_size=3, kernel_num=4, groups=4)
    kernel_spectra = torch.fft.fft2(module.kernel_bank().detach(), norm="ortho").abs()
    signatures = kernel_spectra.mean(dim=(1, 2))
    assert not torch.allclose(signatures[0], signatures[1])
    tensor = torch.randn(2, 4, 11, 13, requires_grad=True)
    output = module(tensor)
    assert output.shape == tensor.shape
    output.square().mean().backward()
    assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_cross_frequency_and_foundation_are_differentiable() -> None:
    low = torch.randn(1, 8, 7, 9)
    high = torch.randn_like(low)
    low_out, high_out = CrossFrequencyConditioner(8)(low, high)
    assert low_out.shape == low.shape and high_out.shape == high.shape

    tensor = torch.randn(1, 8, 13, 15, requires_grad=True)
    output, stats = FrequencyFoundationBlock(8, kernel_num=4)(tensor, "s2")
    assert output.shape == tensor.shape
    assert stats.stage == "s2"
    assert stats.low_energy.shape == stats.high_energy.shape == (1,)
    assert stats.radial_energy is not None and stats.radial_energy.shape == (1, 4)
    output.mean().backward()
    assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


import torch

from tfs_moe_fusion.moe import (
    DetailExpert,
    ExpertContext,
    FunctionalExpertPool,
    InfraredSaliencyExpert,
    LowFrequencyExpert,
    build_functional_expert,
)
from tfs_moe_fusion.types import ExpertType


def _context(ir: bool = True) -> ExpertContext:
    return ExpertContext(
        task=TaskType.VIF if ir else TaskType.MFIF,
        modality_a=ModalityType.VISIBLE_RGB if ir else ModalityType.GENERIC_RGB,
        modality_b=ModalityType.INFRARED_GRAY if ir else ModalityType.GENERIC_RGB,
        source_a_feature=torch.randn(2, 8, 13, 15),
        source_b_feature=torch.randn(2, 8, 13, 15),
    )


def test_all_five_experts_have_uniform_output_contract() -> None:
    tensor = torch.randn(2, 8, 13, 15, requires_grad=True)
    outputs = []
    for expert_type in ExpertType:
        output = build_functional_expert(expert_type, 8)(tensor, _context())
        assert output.expert is expert_type
        assert output.residual.shape == tensor.shape
        assert output.valid_samples.shape == (2,)
        assert torch.isfinite(output.residual).all()
        outputs.append(output.residual)
    sum(item.mean() for item in outputs).backward()
    assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_low_and_detail_experts_are_frequency_specialized() -> None:
    tensor = torch.randn(1, 8, 14, 18)
    context = _context()
    low_residual = LowFrequencyExpert(8)(tensor, context).residual
    detail_residual = DetailExpert(8)(tensor, context).residual
    low_bands = HaarDWT2D()(low_residual)
    detail_bands = HaarDWT2D()(detail_residual)
    assert sum(x.abs().max() for x in (low_bands.lh, low_bands.hl, low_bands.hh)) < 1e-5
    assert detail_bands.ll.abs().max() < 1e-5


def test_infrared_expert_is_invalid_without_real_ir() -> None:
    tensor = torch.randn(2, 8, 13, 15)
    output = InfraredSaliencyExpert(8)(tensor, _context(ir=False))
    assert not output.valid_samples.any()
    assert torch.count_nonzero(output.residual) == 0


def test_functional_expert_pool_has_fixed_order_and_validity() -> None:
    names = [item.value for item in ExpertType]
    pool = FunctionalExpertPool(8, names)
    assert pool.expert_names == tuple(names)
    assert pool.num_experts == 5
    assert pool.get_expert(0) is pool.get_expert("common")
    validity = pool.get_validity(_context(ir=False))
    assert validity.shape == (2, 5)
    assert validity[:, :-1].all() and not validity[:, -1].any()


import torch

from tfs_moe_fusion.model import LightweightMFIFInteraction, TaskFiLM


def test_mfif_interaction_is_lightweight_symmetric_and_task_specific() -> None:
    module = LightweightMFIFInteraction(8, heads=2, window_size=4).eval()
    fused = torch.randn(2, 8, 9, 11, requires_grad=True)
    source_a = torch.randn_like(fused)
    source_b = torch.randn_like(fused)
    inactive = module(fused, source_a, source_b, TaskType.VIF)
    assert inactive is fused
    regular = module(fused, source_a, source_b, TaskType.MFIF)
    swapped = module(fused, source_b, source_a, TaskType.MFIF)
    assert regular.shape == fused.shape
    torch.testing.assert_close(regular, swapped, rtol=1e-5, atol=1e-6)
    regular.mean().backward()
    assert fused.grad is not None and torch.isfinite(fused.grad).all()


def test_task_film_produces_task_conditioned_features() -> None:
    film = TaskFiLM(8)
    feature = torch.randn(1, 8, 5, 7)
    assert not torch.equal(film(feature, TaskType.VIF), film(feature, TaskType.MFIF))


def test_disabled_task_film_is_an_exact_identity() -> None:
    film = TaskFiLM(8, enabled=False)
    feature = torch.randn(2, 8, 5, 7)
    assert film(feature, TaskType.VIF) is feature


from pathlib import Path

import torch

from tfs_moe_fusion.guidance import (
    FrozenSegformerBackend,
    LightweightFocusHead,
    SemanticGuideOutput,
)

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_DIR = ROOT / "weights/segformer_b0_cityscapes"


@pytest.mark.integration
def test_local_segformer_checkpoint_loads_strictly_and_is_frozen() -> None:
    required = ("config.json", "preprocessor_config.json", "pytorch_model.bin")
    if not all((SEMANTIC_DIR / name).is_file() for name in required):
        pytest.skip("local SegFormer integration assets are not installed")
    backend = FrozenSegformerBackend(SEMANTIC_DIR, input_size=64, expected_classes=19)
    backend.train()
    assert not backend.training and not backend.model.training
    assert not any(parameter.requires_grad for parameter in backend.parameters())
    image = torch.rand(1, 3, 31, 37, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = backend(image)
    assert output.probabilities.shape == (1, 19, 31, 37)
    assert output.logits.dtype == output.probabilities.dtype == torch.float32
    assert output.uncertainty.shape == output.boundary.shape == (1, 1, 31, 37)
    torch.testing.assert_close(
        output.probabilities.sum(dim=1),
        torch.ones(1, 31, 37),
        rtol=1e-5,
        atol=1e-5,
    )
    assert output.uncertainty.min() >= 0 and output.uncertainty.max() <= 1
    assert output.boundary.min() >= 0 and output.boundary.max() <= 1
    assert len(output.features) == 4
    AuxiliaryOutputs(
        semantic_probabilities=output.probabilities,
        semantic_logits=output.logits,
        semantic_uncertainty=output.uncertainty,
        semantic_boundary=output.boundary,
    )
    output.uncertainty.mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert all(parameter.grad is None for parameter in backend.parameters())


def test_focus_head_is_source_symmetric_and_normalized() -> None:
    channels = [8, 16, 32, 64]
    head = LightweightFocusHead(channels, hidden_channels=8).eval()
    coarse = torch.rand(2, 3, 31, 37)
    source_a = tuple(
        torch.randn(2, value, max(1, 32 // 2**index), 40 // 2**index)
        for index, value in enumerate(channels)
    )
    source_b = tuple(torch.randn_like(value) for value in source_a)
    with torch.no_grad():
        regular = head(coarse, source_a, source_b)
        swapped = head(coarse, source_b, source_a)
    assert regular.reliability.shape == (2, 2, 31, 37)
    torch.testing.assert_close(regular.reliability.sum(dim=1), torch.ones(2, 31, 37))
    torch.testing.assert_close(
        regular.reliability, swapped.reliability.flip(1), rtol=1e-6, atol=1e-6
    )


import torch

from tfs_moe_fusion.config import MoEConfig
from tfs_moe_fusion.moe import (
    FunctionalMoEBlock,
    JointTopKRouter,
    MoEOutput,
    RouterContext,
)

EXPERTS = [item.value for item in ExpertType]


def _router_context(feature: torch.Tensor, has_ir: bool) -> RouterContext:
    return RouterContext(
        feature=feature,
        task=TaskType.VIF if has_ir else TaskType.MFIF,
        modality_a=ModalityType.VISIBLE_RGB if has_ir else ModalityType.GENERIC_RGB,
        modality_b=ModalityType.INFRARED_GRAY if has_ir else ModalityType.GENERIC_RGB,
        low_energy=torch.rand(feature.shape[0]),
        high_energy=torch.rand(feature.shape[0]),
    )


def _expert_context(feature: torch.Tensor, has_ir: bool) -> ExpertContext:
    routing = _router_context(feature, has_ir)
    return ExpertContext(
        routing.task,
        routing.modality_a,
        routing.modality_b,
        source_a_feature=torch.randn_like(feature),
        source_b_feature=torch.randn_like(feature),
    )


def test_semantic_boundary_regularizer_uses_logits_and_is_differentiable() -> None:
    feature = torch.randn(2, 8, 9, 11, requires_grad=True)
    context = _expert_context(feature, has_ir=True)
    context.semantic_boundary = torch.rand(2, 1, 9, 11)
    semantic = build_functional_expert(ExpertType.SEMANTIC, 8)
    output = semantic(feature, context)

    assert "gate_logits" in output.diagnostics
    regularizers = FunctionalMoEBlock._expert_regularizers(output, context)
    loss = regularizers["frequency/semantic_boundary"]
    assert torch.isfinite(loss)
    loss.backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()


def test_joint_router_normalization_mask_and_diagnostics() -> None:
    feature = torch.randn(3, 8, 9, 11)
    router = JointTopKRouter(8, EXPERTS, top_k=2, hidden_channels=16)
    output = router(_router_context(feature, has_ir=False))
    ir_index = EXPERTS.index(ExpertType.INFRARED_SALIENCY.value)
    assert output.logits.shape == output.probabilities.shape == (3, 5)
    torch.testing.assert_close(output.probabilities.sum(dim=1), torch.ones(3))
    torch.testing.assert_close(output.topk_weights.sum(dim=1), torch.ones(3))
    torch.testing.assert_close(output.branch_weights.sum(dim=1), torch.ones(3))
    assert output.spatial_gates is not None
    assert output.spatial_gates.shape == (3, 5, 9, 11)
    assert not output.valid_expert_mask[:, ir_index].any()
    assert torch.count_nonzero(output.probabilities[:, ir_index]) == 0
    assert not (output.topk_indices == ir_index).any()


def test_only_ir_expert_mask_changes_with_modality() -> None:
    feature = torch.randn(1, 8, 7, 9)
    router = JointTopKRouter(8, EXPERTS, top_k=2, hidden_channels=16)
    no_ir = router(_router_context(feature, has_ir=False)).valid_expert_mask
    with_ir = router(_router_context(feature, has_ir=True)).valid_expert_mask
    assert no_ir[:, :-1].all() and not no_ir[:, -1].any()
    assert with_ir.all()


def test_dense_and_sparse_execution_are_numerically_equivalent() -> None:
    torch.manual_seed(13)
    config = MoEConfig(
        experts=EXPERTS,
        top_k=2,
        sparse_execution=True,
        router_hidden_channels=16,
        expert_expansion=1,
    )
    block = FunctionalMoEBlock(8, config, "s2").eval()
    feature = torch.randn(2, 8, 9, 11)
    router_context = _router_context(feature, has_ir=True)
    expert_context = _expert_context(feature, has_ir=True)
    with torch.no_grad():
        sparse_output = block(
            feature, router_context, expert_context, sparse_execution=True
        )
        dense_output = block(
            feature,
            router_context,
            expert_context,
            sparse_execution=False,
            return_expert_outputs=True,
        )
        sparse, sparse_diagnostics = sparse_output
        dense, dense_diagnostics = dense_output
    assert isinstance(sparse_output, MoEOutput)
    assert sparse_output.residual.shape == feature.shape
    assert dense_output.expert_outputs is not None
    assert set(dense_output.expert_outputs) == set(EXPERTS)
    torch.testing.assert_close(sparse, dense, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        sparse_diagnostics.probabilities, dense_diagnostics.probabilities
    )


def test_dense_topk_and_sparse_execution_have_equivalent_gradients() -> None:
    torch.manual_seed(17)
    config = MoEConfig(
        experts=EXPERTS,
        top_k=2,
        router_hidden_channels=16,
        expert_expansion=1,
        noisy_topk=False,
    )
    block = FunctionalMoEBlock(8, config, "s2").eval()

    def gradients(sparse: bool):
        block.zero_grad(set_to_none=True)
        feature = torch.randn(2, 8, 9, 11, requires_grad=True)
        context = _router_context(feature, has_ir=True)
        expert_context = _expert_context(feature, has_ir=True)
        output = block(
            feature, context, expert_context, sparse_execution=sparse
        ).feature
        output.square().mean().backward()
        parameter_gradients = {
            name: (
                parameter.grad.detach().clone()
                if parameter.grad is not None
                else torch.zeros_like(parameter)
            )
            for name, parameter in block.named_parameters()
        }
        return feature.grad.detach().clone(), parameter_gradients

    torch.manual_seed(29)
    dense_input, dense_parameters = gradients(False)
    torch.manual_seed(29)
    sparse_input, sparse_parameters = gradients(True)
    torch.testing.assert_close(sparse_input, dense_input, rtol=1e-5, atol=1e-7)
    assert dense_parameters.keys() == sparse_parameters.keys()
    for name in dense_parameters:
        torch.testing.assert_close(
            sparse_parameters[name], dense_parameters[name], rtol=1e-5, atol=1e-7
        )


def test_sparse_execution_never_calls_an_unselected_expert() -> None:
    torch.manual_seed(31)
    config = MoEConfig(
        experts=EXPERTS,
        top_k=2,
        router_hidden_channels=16,
        expert_expansion=1,
    )
    block = FunctionalMoEBlock(8, config, "s2").eval()
    calls = [0] * len(EXPERTS)
    handles = [
        expert.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.__setitem__(
                index, calls[index] + 1
            )
        )
        for index, expert in enumerate(block.experts)
    ]
    feature = torch.randn(1, 8, 9, 11)
    with torch.no_grad():
        output = block(
            feature,
            _router_context(feature, has_ir=True),
            _expert_context(feature, has_ir=True),
            sparse_execution=True,
        )
    for handle in handles:
        handle.remove()
    selected = set(output.router.topk_indices.flatten().tolist())
    assert sum(calls) == len(selected) == config.top_k
    assert all(calls[index] == int(index in selected) for index in range(len(calls)))


def test_sparse_execution_has_exactly_batch_times_topk_assignments() -> None:
    config = MoEConfig(
        experts=EXPERTS,
        top_k=2,
        router_hidden_channels=16,
        expert_expansion=1,
    )
    block = FunctionalMoEBlock(8, config, "s2").eval()
    feature = torch.randn(2, 8, 9, 11)
    output = block(
        feature,
        _router_context(feature, has_ir=True),
        _expert_context(feature, has_ir=True),
        sparse_execution=True,
    )
    assert output.diagnostics.auxiliary["expert_sample_assignments"] == 4


def test_dense_execution_assignments_equal_valid_expert_mask() -> None:
    config = MoEConfig(
        experts=EXPERTS,
        top_k=2,
        router_hidden_channels=16,
        expert_expansion=1,
    )
    block = FunctionalMoEBlock(8, config, "s2").eval()
    feature = torch.randn(2, 8, 9, 11)
    output = block(
        feature,
        _router_context(feature, has_ir=True),
        _expert_context(feature, has_ir=True),
        sparse_execution=False,
    )
    assert output.diagnostics.auxiliary["expert_sample_assignments"] == int(
        output.router.valid_expert_mask.sum()
    )


def test_disabled_frequency_regularizers_preserve_infrared_regularizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tfs_moe_fusion.moe import MoEExecutionPolicy

    config = MoEConfig(
        experts=EXPERTS,
        top_k=2,
        router_hidden_channels=16,
        expert_expansion=1,
    )
    block = FunctionalMoEBlock(8, config, "s2").train()
    block.set_execution_policy(
        MoEExecutionPolicy(
            "dense_uniform",
            compute_frequency_regularizers=False,
            compute_infrared_regularizers=True,
        )
    )
    called: list[ExpertType] = []

    def record(output, context):
        called.append(output.expert)
        return {}

    monkeypatch.setattr(
        FunctionalMoEBlock, "_expert_regularizers", staticmethod(record)
    )
    feature = torch.randn(2, 8, 9, 11)
    block(
        feature,
        _router_context(feature, has_ir=True),
        _expert_context(feature, has_ir=True),
    )
    assert called == [ExpertType.INFRARED_SALIENCY]


def test_annealed_topk_boundary_matches_sparse_without_retaining_residuals() -> None:
    from tfs_moe_fusion.moe import MoEExecutionPolicy

    torch.manual_seed(37)
    config = MoEConfig(
        experts=EXPERTS,
        top_k=2,
        router_hidden_channels=16,
        expert_expansion=1,
        noisy_topk=False,
    )
    block = FunctionalMoEBlock(8, config, "s2").train()
    feature = torch.randn(2, 8, 9, 11)
    router_context = _router_context(feature, has_ir=True)
    expert_context = _expert_context(feature, has_ir=True)
    block.set_execution_policy(
        MoEExecutionPolicy("dense_annealed", uniform_to_soft=1.0, soft_to_topk=1.0)
    )
    dense = block(feature, router_context, expert_context)
    block.set_execution_policy(MoEExecutionPolicy("sparse_batch"))
    sparse = block(feature, router_context, expert_context)
    torch.testing.assert_close(sparse.feature, dense.feature, rtol=1e-6, atol=1e-7)
    assert sparse.expert_outputs is None
    assert "expert_residuals" not in sparse.diagnostics.auxiliary
    assert "expert_regularizers" in sparse.diagnostics.auxiliary


def test_uniform_warmup_updates_every_expert_but_not_either_router() -> None:
    from tfs_moe_fusion.moe import MoEExecutionPolicy

    torch.manual_seed(41)
    config = MoEConfig(
        experts=EXPERTS,
        top_k=2,
        router_hidden_channels=16,
        expert_expansion=1,
        noisy_topk=False,
    )
    block = FunctionalMoEBlock(8, config, "s2").train()
    block.set_execution_policy(
        MoEExecutionPolicy(
            "dense_uniform",
            uniform_to_soft=0.0,
            spatial_gate_scale=0.0,
            detach_router=True,
        )
    )
    feature = torch.randn(2, 8, 9, 11)
    output = block(
        feature,
        _router_context(feature, has_ir=True),
        _expert_context(feature, has_ir=True),
    )
    output.feature.square().mean().backward()
    assert all(
        any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad)
            for parameter in expert.parameters()
        )
        for expert in block.experts
    )
    assert all(parameter.grad is None for parameter in block.router.parameters())


from pathlib import Path

import pytest
import torch

from tfs_moe_fusion.config import load_config
from tfs_moe_fusion.model import build_model
from tfs_moe_fusion.utils import make_probe_batch

ROOT = Path(__file__).resolve().parents[1]


def _config():
    config = load_config(ROOT / "configs/default.yaml")
    config.model.backbone.channels = [8, 16, 32, 64]
    config.model.backbone.depths = [1, 1, 1, 1]
    config.model.frequency.fdconv_kernel_num = 4
    config.model.moe.router_hidden_channels = 16
    config.model.moe.expert_expansion = 1
    config.model.guidance.focus.hidden_channels = 8
    config.model.guidance.semantic.input_size = 32
    config.model.guidance.semantic.enabled = False
    config.model.feedback.guide_channels = 8
    config.training.batch_size = 1
    config.training.ema.enabled = False
    return config


@pytest.mark.parametrize("task", list(TaskType))
def test_final_closed_loop_contract_for_every_task(task: TaskType) -> None:
    config = _config()
    batch = make_probe_batch(config, task, spatial_size=(33, 35))
    model = build_model(config).eval()
    with torch.no_grad():
        output = model(batch)
    assert output.fused.shape == output.coarse.shape == (1, 3, 33, 35)
    core_router_ids = [
        "s2.moe0",
        "s3.moe0",
        "s4.moe0",
    ]
    core_spectral_ids = [
        "input_a",
        "input_b",
        "s2.moe0.input",
        "s3.moe0.input",
        "s4.moe0.input",
    ]
    if task in {TaskType.VIF, TaskType.SEG}:
        torch.testing.assert_close(output.fused, output.coarse)
        assert [item.block_id for item in output.router_diagnostics] == core_router_ids
        assert [item.stage for item in output.spectral_statistics] == core_spectral_ids
        assert output.fused_y is not None and output.coarse_y is not None
        assert output.refinement_y is not None
        assert output.fused_y.shape == output.coarse_y.shape == (1, 1, 33, 35)
        torch.testing.assert_close(output.fused_y, output.coarse_y)
        assert torch.count_nonzero(output.refinement) == 0
        assert torch.count_nonzero(output.refinement_y) == 0
        assert output.debug["coarse_head"] == "fusion_y"
        assert output.debug["vif_seg_refinement_active"] is False
        assert output.debug["chroma_cb_error"] < 2e-6
        assert output.debug["chroma_cr_error"] < 2e-6
    else:
        assert not torch.equal(output.fused, output.coarse)
        assert [item.block_id for item in output.router_diagnostics] == [
            *core_router_ids,
            "feedback.s3.moe0",
            "feedback.s2.moe0",
        ]
        assert [item.stage for item in output.spectral_statistics] == [
            *core_spectral_ids,
            "feedback.s3.moe0.input",
            "feedback.s2.moe0.input",
        ]
        assert output.fused_y is output.coarse_y is output.refinement_y is None
        assert output.debug["coarse_head"] == "mfif_rgb"
    assert output.auxiliary is not None
    assert output.auxiliary.semantic_probabilities is None
    assert output.auxiliary.semantic_uncertainty is None
    assert output.auxiliary.semantic_boundary is None
    if task is TaskType.MFIF:
        assert output.auxiliary.focus_reliability.shape == (1, 2, 33, 35)
        assert output.auxiliary.focus_selection.shape == (1, 2, 33, 35)
        assert output.auxiliary.focus_confidence.shape == (1, 1, 33, 35)
    else:
        assert output.auxiliary.focus_reliability is None
        assert output.auxiliary.focus_selection is None
        assert output.auxiliary.focus_confidence is None
    assert output.debug["semantic_available"] is False
    assert output.aux["final_preclamp"].shape == output.fused.shape
    assert output.aux["clamp_low_ratio"].ndim == 0
    assert output.aux["clamp_high_ratio"].ndim == 0


def test_final_backward_without_external_semantic_assets() -> None:
    config = _config()
    batch = make_probe_batch(config, TaskType.MFIF, spatial_size=(33, 35))
    model = build_model(config)
    output = model(batch)
    output.fused.mean().backward()
    trainable_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_gradients
    assert all(torch.isfinite(value).all() for value in trainable_gradients)


def test_vif_backward_updates_only_the_y_output_head() -> None:
    config = _config()
    batch = make_probe_batch(config, TaskType.VIF, spatial_size=(33, 35))
    model = build_model(config)
    output = model(batch)
    assert output.fused_y is not None

    output.fused_y.mean().backward()

    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.core.fusion_y_head.parameters()
    )
    assert all(
        parameter.grad is None for parameter in model.core.mfif_rgb_head.parameters()
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.feedback.decoder.named_parameters()
        if "task_embedding" not in name
    )


@pytest.mark.parametrize("detach", [True, False])
def test_semantic_guidance_input_detach_is_configurable(detach: bool) -> None:
    class DifferentiableSemanticBackend(torch.nn.Module):
        def forward(self, image: torch.Tensor) -> SemanticGuideOutput:
            logits = image.mean(1, keepdim=True)
            probabilities = logits.sigmoid()
            uncertainty = probabilities * (1 - probabilities)
            boundary = probabilities.square()
            return SemanticGuideOutput(
                logits,
                probabilities,
                uncertainty,
                boundary,
                (image.mean((-2, -1), keepdim=True),),
            )

    config = _config()
    config.model.guidance.semantic.detach_guidance_input = detach
    model = build_model(config)
    model.feedback.semantic_backend = DifferentiableSemanticBackend()
    coarse = torch.randn(1, 3, 9, 11, requires_grad=True)

    output = model.feedback._semantic(coarse, TaskType.SEG)

    assert output is not None
    assert output.logits.shape == output.probabilities.shape == (1, 1, 9, 11)
    assert output.uncertainty.shape == output.boundary.shape == (1, 1, 9, 11)
    assert len(output.features) == 1
    assert output.logits.requires_grad is not detach
    if not detach:
        output.logits.sum().backward()
        assert coarse.grad is not None and torch.count_nonzero(coarse.grad)
    else:
        assert coarse.grad is None

    final = torch.randn(1, 3, 9, 11, requires_grad=True)
    final_output = model.feedback._semantic(final, TaskType.SEG, final=True)
    assert final_output is not None and final_output.logits.requires_grad
    final_output.logits.sum().backward()
    assert final.grad is not None and torch.count_nonzero(final.grad)


def test_initial_and_feedback_moe_parameters_are_independent() -> None:
    model = build_model(_config())
    for stage in ("s2", "s3"):
        assert model.core.moe_blocks[stage][0] is not model.feedback.feedback_moe[stage]
        initial = {
            parameter.data_ptr()
            for parameter in model.core.moe_blocks[stage][0].parameters()
        }
        feedback = {
            parameter.data_ptr()
            for parameter in model.feedback.feedback_moe[stage].parameters()
        }
        assert initial.isdisjoint(feedback)


def test_final_source_order_and_supervision_do_not_change_inference() -> None:
    config = _config()
    batch = make_probe_batch(config, TaskType.MFIF, spatial_size=(31, 37))
    swapped = FusionBatch(
        source_a=batch.source_b,
        source_b=batch.source_a,
        task=batch.task,
        sample_ids=batch.sample_ids,
        focus_target=torch.rand_like(batch.focus_target),
    )
    model = build_model(config).eval()
    with torch.no_grad():
        regular = model(batch)
        reversed_sources = model(swapped)
    torch.testing.assert_close(
        regular.fused, reversed_sources.fused, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        regular.auxiliary.focus_reliability,
        reversed_sources.auxiliary.focus_reliability.flip(1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_disable_refinement_decoder_is_a_true_coarse_only_ablation() -> None:
    config = _config()
    config.model.ablation.disable_refinement_decoder = True
    model = build_model(config).eval()
    with torch.no_grad():
        output = model(make_probe_batch(config, TaskType.VIF, spatial_size=(31, 37)))
    torch.testing.assert_close(output.fused, output.coarse)
    assert torch.count_nonzero(output.refinement) == 0
    assert output.debug["refinement_decoder_disabled"] is True


@pytest.mark.parametrize("task", list(TaskType))
def test_null_semantic_backend_runs_every_task(task: TaskType) -> None:
    config = _config()
    config.model.guidance.semantic.enabled = True
    config.model.guidance.semantic.backend = None
    model = build_model(config).eval()
    with torch.no_grad():
        output = model(make_probe_batch(config, task, spatial_size=(31, 37)))
    assert output.auxiliary is not None
    assert output.auxiliary.semantic_probabilities is None
    assert output.debug["semantic_available"] is False
    feedback = [
        item
        for item in output.router_diagnostics
        if item.block_id.startswith("feedback.")
    ]
    if task is TaskType.MFIF:
        assert len(feedback) == 2
        assert all(item.auxiliary["semantic_available"] is False for item in feedback)
    else:
        assert not feedback
