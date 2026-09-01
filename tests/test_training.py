from __future__ import annotations

from pathlib import Path

import pytest

from tfs_moe_fusion.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _smoke_config(assets: tuple[Path, Path, Path] | None = None):
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
    config.data.crop_size = 32
    config.data.num_workers = 0
    config.data.horizontal_flip_probability = 0.0
    config.data.rotation_probability = 0.0
    if assets is not None:
        config.data.root, config.data.mfif_root, config.data.manifest = map(str, assets)
    config.training.epochs = 1
    config.training.steps_per_epoch = 20
    config.training.max_steps = 20
    config.training.batch_size = 1
    config.training.precision = "fp32"
    config.training.task_sampling.strategy = "alternating"
    config.training.scheduler.warmup_steps = 2
    config.training.ema.enabled = True
    config.training.losses.strict_targets = False
    return config


import torch

from tfs_moe_fusion.trainer import AMPController


def test_cpu_bf16_amp_performs_optimizer_step() -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    amp = AMPController("bf16", torch.device("cpu"))
    with amp.autocast():
        loss = model(torch.rand(3, 4)).square().mean()
    amp.backward(loss)
    amp.unscale_(optimizer)
    amp.step(optimizer)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


from tfs_moe_fusion.trainer import ModelEMA


def test_ema_updates_and_restores_live_weights() -> None:
    model = torch.nn.Linear(2, 1)
    ema = ModelEMA(model, 0.5)
    before = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(2)
    live = model.weight.detach().clone()
    ema.update(model)
    with ema.apply(model):
        assert not torch.equal(model.weight, live)
    torch.testing.assert_close(model.weight, live)
    assert not torch.equal(before, live)


from tfs_moe_fusion.losses import focus_losses


def test_focus_loss_supervises_selection_boundary_and_confidence() -> None:
    logits = torch.randn(2, 2, 11, 13, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 11, 13)).float()
    values = focus_losses(logits, logits.softmax(1)[:, :1], target)
    assert set(values) == {"focus/selection", "focus/boundary", "focus/confidence"}
    sum(values.values()).backward()
    assert logits.grad is not None


from tfs_moe_fusion.losses import frequency_specialization
from tfs_moe_fusion.types import RouterDiagnostics


def _diagnostic() -> RouterDiagnostics:
    probabilities = torch.tensor([[0.6, 0.4]], requires_grad=True)
    return RouterDiagnostics(
        "test",
        probabilities.log(),
        probabilities,
        torch.tensor([[0, 1]]),
        probabilities,
        torch.ones(1, 2, dtype=torch.bool),
        auxiliary={
            "expert_residuals": {
                "low_frequency": torch.rand(1, 3, 8, 8, requires_grad=True),
                "detail": torch.rand(1, 3, 8, 8, requires_grad=True),
            }
        },
    )


def test_frequency_specialization_reports_both_leakages() -> None:
    values = frequency_specialization((_diagnostic(),))
    assert values["frequency/low_leakage"] > 0
    assert values["frequency/detail_leakage"] > 0


from tfs_moe_fusion.losses import (
    mfif_losses,
    seg_fusion_anchor_losses,
    vif_losses,
)


def test_vif_and_mfif_losses_are_finite_and_differentiable() -> None:
    fused = torch.rand(2, 3, 15, 17, requires_grad=True)
    visible = torch.rand_like(fused)
    infrared = torch.rand(2, 1, 15, 17)
    values = {
        **vif_losses(fused, visible, infrared),
        **mfif_losses(fused, visible, False),
    }
    assert values
    sum(values.values()).backward()
    assert fused.grad is not None and torch.isfinite(fused.grad).all()


def test_seg_fusion_anchor_is_minimal_and_differentiable() -> None:
    fused = torch.rand(2, 3, 15, 17, requires_grad=True)
    visible = torch.rand_like(fused)
    infrared = torch.rand(2, 1, 15, 17)
    values = seg_fusion_anchor_losses(fused, visible, infrared)
    assert set(values) == {"seg_fusion/intensity", "seg_fusion/gradient"}
    sum(values.values()).backward()
    assert fused.grad is not None and torch.isfinite(fused.grad).all()


from tfs_moe_fusion.losses import semantic_losses


def test_semantic_loss_handles_ignore_index() -> None:
    logits = torch.randn(1, 4, 9, 7, requires_grad=True)
    target = torch.randint(0, 4, (1, 9, 7))
    target[:, 0] = 255
    values = semantic_losses(
        logits, target, torch.rand(1, 1, 9, 7), torch.rand(1, 3, 9, 7), 255, None
    )
    sum(values.values()).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_semantic_loss_handles_fully_ignored_crop() -> None:
    logits = torch.randn(1, 19, 9, 7, requires_grad=True)
    target = torch.full((1, 9, 7), 255)
    values = semantic_losses(
        logits,
        target,
        torch.rand(1, 1, 9, 7),
        torch.rand(1, 3, 9, 7),
        255,
        None,
    )
    assert set(values) == {"semantic/ce", "semantic/dice"}
    total = sum(values.values())
    assert torch.isfinite(total)
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


from pathlib import Path

from tfs_moe_fusion.losses import LossContext, LossOutput, MultiTaskLossManager
from tfs_moe_fusion.model import build_model
from tfs_moe_fusion.types import TaskType
from tfs_moe_fusion.utils import make_probe_batch

ROOT = Path(__file__).resolve().parents[1]


def test_loss_manager_returns_structured_vif_output() -> None:
    config = _smoke_config()
    model = build_model(config).train()
    batch = make_probe_batch(config, TaskType.VIF)
    value = MultiTaskLossManager(config.training.losses)(
        LossContext(batch, model(batch), TaskType.VIF, 0, 0, model)
    )
    assert value.total.requires_grad and "fusion/intensity" in value.components
    assert value.diagnostics["task"] == "vif" and torch.isfinite(value.total)


def test_seg_loss_has_fusion_anchor_without_vif_ssim_or_color() -> None:
    config = _smoke_config()
    model = build_model(config).train()
    batch = make_probe_batch(config, TaskType.SEG)
    value = MultiTaskLossManager(config.training.losses)(
        LossContext(
            batch,
            model(batch),
            TaskType.SEG,
            0,
            0,
            model,
            phase="stabilization",
            loss_multipliers={"seg_fusion": 1.0, "semantic": 0.25},
        )
    )
    assert "seg_fusion/intensity" in value.components
    assert "seg_fusion/gradient" in value.components
    assert "fusion/ssim" not in value.components
    assert "fusion/color" not in value.components
    assert "semantic/boundary" not in value.components
    assert "semantic/coarse" not in value.components
    assert not any(name.startswith("frequency/") for name in value.components)
    assert not any(name.startswith("consistency/") for name in value.components)
    assert "infrared/preservation" not in value.components
    torch.testing.assert_close(
        value.weighted_components["seg_fusion/intensity"],
        value.components["seg_fusion/intensity"],
    )


from tfs_moe_fusion.losses import moe_balance_loss


def test_moe_balance_is_availability_aware_and_finite() -> None:
    importance, load, entropy = moe_balance_loss((_diagnostic(),))
    assert importance.isfinite() and load.isfinite() and entropy.isfinite()


def test_switch_balance_routes_gradient_through_probabilities() -> None:
    diagnostic = _diagnostic()
    soft, switch, _ = moe_balance_loss((diagnostic,))
    (soft + switch).backward()
    assert diagnostic.probabilities.grad is not None
    assert torch.isfinite(diagnostic.probabilities.grad).all()


from tfs_moe_fusion.trainer import (
    ExpertOnlyParameterPolicy,
    MoEExecutionScheduler,
    RouterLoadMonitor,
    expert_regularizer_flags,
    loss_group_gradient_statistics,
)


def test_seg_loss_group_gradient_ratio_is_reported() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    fusion = parameter.square()
    semantic = (3 * parameter).square()
    result = LossOutput(
        fusion + semantic,
        {},
        {"seg_fusion/intensity": fusion, "semantic/ce": semantic},
        {},
        {},
    )
    values = loss_group_gradient_statistics(result, (parameter,))
    assert values["grad_norm/seg_fusion"] == pytest.approx(4.0)
    assert values["grad_norm/semantic"] == pytest.approx(36.0)
    assert values["grad_ratio/semantic_to_fusion"] == pytest.approx(9.0)


def test_loss_group_gradient_statistics_skips_detached_namespace() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    fusion = parameter.square()
    result = LossOutput(
        fusion,
        {},
        {
            "seg_fusion/intensity": fusion,
            "semantic/ce": torch.tensor(1.0),
        },
        {},
        {},
    )
    assert loss_group_gradient_statistics(result, (parameter,)) == {}


def test_moe_execution_schedule_is_continuous_and_refreshes() -> None:
    config = _smoke_config().training.moe_execution
    monitor = RouterLoadMonitor(config)
    scheduler = MoEExecutionScheduler(config, monitor)

    uniform = scheduler.resolve(49)
    uniform_boundary = scheduler.resolve(50)
    uniform_to_soft_last = scheduler.resolve(149)
    soft_boundary = scheduler.resolve(150)
    soft_to_topk_last = scheduler.resolve(299)
    sparse_boundary = scheduler.resolve(300)
    refresh = scheduler.resolve(500)

    assert uniform is not None and uniform.mode == "dense_uniform"
    assert uniform_boundary is not None
    assert uniform_boundary.mode == "dense_annealed"
    assert uniform_boundary.uniform_to_soft == pytest.approx(0.0)
    assert uniform_to_soft_last is not None
    assert uniform_to_soft_last.mode == "dense_annealed"
    assert soft_boundary is not None
    assert soft_boundary.mode == "dense_annealed"
    assert soft_boundary.uniform_to_soft == pytest.approx(1.0)
    assert soft_boundary.soft_to_topk == pytest.approx(0.0)
    assert soft_to_topk_last is not None
    assert soft_to_topk_last.mode == "dense_annealed"
    assert sparse_boundary is not None and sparse_boundary.mode == "sparse_batch"
    assert refresh is not None and refresh.mode == "expert_refresh"
    assert refresh.expert_only and refresh.detach_router


def test_zero_frequency_multiplier_skips_only_frequency_regularizers() -> None:
    config = _smoke_config().training.losses
    config.frequency.enabled = True
    config.infrared.enabled = True
    frequency, infrared = expert_regularizer_flags(
        config, {"frequency": 0.0}, TaskType.VIF
    )
    assert frequency is False
    assert infrared is True


def test_router_monitor_recovery_state_round_trips() -> None:
    config = _smoke_config().training.moe_execution
    config.monitor.patience_steps = 1
    config.monitor.starvation_threshold = 0.2
    monitor = RouterLoadMonitor(config)
    probabilities = torch.tensor([[0.9, 0.1]])
    diagnostic = RouterDiagnostics(
        "s2.moe0",
        probabilities.log(),
        probabilities,
        torch.tensor([[0]]),
        torch.ones(1, 1),
        torch.ones(1, 2, dtype=torch.bool),
    )
    monitor.update((diagnostic,), step=10)
    assert monitor.in_recovery(11)

    restored = RouterLoadMonitor(config)
    restored.load_state_dict(monitor.state_dict())
    assert restored.state_dict() == monitor.state_dict()
    policy = MoEExecutionScheduler(config, restored).resolve(11)
    assert policy is not None
    assert policy.temperature >= config.recovery.temperature_floor
    assert policy.noise_std >= config.recovery.noise_std


def test_expert_only_policy_freezes_every_nonexpert_group() -> None:
    config = _smoke_config()
    registry = ParameterGroupRegistry.from_model(build_model(config))
    policy = ExpertOnlyParameterPolicy(registry)
    with policy.apply(True):
        for group, parameters in registry.groups.items():
            expected = group in policy.expert_groups
            assert all(parameter.requires_grad is expected for parameter in parameters)
    assert all(
        parameter.requires_grad
        for parameters in registry.groups.values()
        for parameter in parameters
    )


from pathlib import Path

from tfs_moe_fusion.trainer import build_optimizer

ROOT = Path(__file__).resolve().parents[1]


def test_optimizer_groups_cover_trainable_parameters_exactly_once() -> None:
    config = _smoke_config()
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


from pathlib import Path

from tfs_moe_fusion.trainer import ParameterGroupRegistry, TaskParameterPolicy

ROOT = Path(__file__).resolve().parents[1]


def test_mfif_policy_temporarily_freezes_ir_experts() -> None:
    config = _smoke_config()
    registry = ParameterGroupRegistry.from_model(build_model(config))
    policy = TaskParameterPolicy(registry, config.training.task_update_policy)
    parameters = registry.parameters("ir_experts")
    assert parameters
    with policy.apply(TaskType.MFIF):
        assert all(not parameter.requires_grad for parameter in parameters)
    assert all(parameter.requires_grad for parameter in parameters)


from tfs_moe_fusion.losses import low_frequency_consistency


def test_low_frequency_consistency_is_zero_for_equal_images() -> None:
    image = torch.rand(1, 3, 13, 15)
    torch.testing.assert_close(
        low_frequency_consistency(image, image, 5, 1.0), torch.tensor(0.0)
    )


from pathlib import Path

from tfs_moe_fusion.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]


def test_trainer_performs_real_updates_for_all_tasks(
    tmp_path: Path, semantic_rt_assets: tuple[Path, Path, Path]
) -> None:
    config = _smoke_config(semantic_rt_assets)
    trainer = Trainer(build_model(config), config, torch.device("cpu"), tmp_path)
    for task in TaskType:
        result = trainer.train_step(task)
        assert torch.isfinite(result.total)
    assert trainer.state.global_step == 3


def test_train_displays_one_progress_bar_per_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    semantic_rt_assets: tuple[Path, Path, Path],
) -> None:
    import tfs_moe_fusion.trainer as trainer_module

    config = _smoke_config(semantic_rt_assets)
    config.training.epochs = 2
    config.training.steps_per_epoch = 2
    config.training.max_steps = None
    config.training.ema.enabled = False
    config.training.task_sampling.strategy = "alternating"
    config.training.log_every_steps = 100
    trainer = Trainer(torch.nn.Linear(2, 1), config, torch.device("cpu"), tmp_path)

    progress_bars = []

    class FakeProgress:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.updates = 0
            self.closed = False
            progress_bars.append(self)

        def update(self, value: int) -> None:
            self.updates += value

        def set_postfix(self, refresh: bool = True, **kwargs) -> None:
            self.postfix_refresh = refresh
            self.postfix = kwargs

        def clear(self) -> None:
            pass

        def refresh(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(trainer_module, "tqdm", FakeProgress)
    monkeypatch.setattr(trainer, "save", lambda path, **kwargs: Path(path))

    def fake_train_step(task: TaskType) -> LossOutput:
        namespace = {
            TaskType.VIF: "fusion/intensity",
            TaskType.MFIF: "focus/selection",
            TaskType.SEG: "semantic/ce",
        }[task]
        components = {
            namespace: torch.tensor(1.0),
            "moe/importance": torch.tensor(0.2),
        }
        trainer.state.phase = "stabilization"
        trainer.state.global_step += 1
        return LossOutput(torch.tensor(1.2), components, components, {}, {})

    monkeypatch.setattr(trainer, "train_step", fake_train_step)
    trainer.train()

    assert [item.kwargs["desc"] for item in progress_bars] == [
        "Epoch 1/2 [stabilization]",
        "Epoch 2/2 [stabilization]",
    ]
    assert [item.updates for item in progress_bars] == [2, 2]
    assert all(set(item.postfix) == {"task", "loss", "lr"} for item in progress_bars)
    assert all(item.closed for item in progress_bars)
