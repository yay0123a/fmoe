from __future__ import annotations

from pathlib import Path

import pytest

from tfs_moe_fusion.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _smoke_config():
    config = load_config(ROOT / "configs/default.yaml")
    config.model.backbone.channels = [8, 16, 32, 64]
    config.model.backbone.depths = [1, 1, 1, 1]
    config.model.frequency.fdconv_kernel_num = 4
    config.model.moe.router_hidden_channels = 16
    config.model.moe.expert_expansion = 1
    config.model.guidance.focus.hidden_channels = 8
    config.model.guidance.semantic.input_size = 32
    config.model.feedback.guide_channels = 8
    config.data.height = 17
    config.data.width = 19
    config.data.dummy_length = 6
    config.training.epochs = 1
    config.training.steps_per_epoch = 20
    config.training.max_steps = 20
    config.training.batch_size = 1
    config.training.precision = "fp32"
    config.training.task_sampling.strategy = "alternating"
    config.training.scheduler.warmup_steps = 2
    config.training.ema.enabled = True
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


from tfs_moe_fusion.trainer import ElasticWeightConsolidation


def test_ewc_penalizes_drift_from_consolidated_parameters() -> None:
    model = torch.nn.Linear(2, 1)
    model(torch.ones(1, 2)).sum().backward()
    ewc = ElasticWeightConsolidation(1.0)
    ewc.consolidate(model)
    with torch.no_grad():
        model.weight.add_(1)
    assert ewc.penalty(model) > 0


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


from tfs_moe_fusion.losses import mfif_losses, vif_losses


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
from tfs_moe_fusion.utils import make_dummy_batch

ROOT = Path(__file__).resolve().parents[1]


def test_loss_manager_returns_structured_vif_output() -> None:
    config = _smoke_config()
    model = build_model(config).train()
    batch = make_dummy_batch(config, TaskType.VIF)
    value = MultiTaskLossManager(config.training.losses)(
        LossContext(batch, model(batch), TaskType.VIF, 0, 0, model)
    )
    assert value.total.requires_grad and "fusion/intensity" in value.components
    assert value.diagnostics["task"] == "vif" and torch.isfinite(value.total)


from tfs_moe_fusion.losses import moe_balance_loss


def test_moe_balance_is_availability_aware_and_finite() -> None:
    importance, load, entropy = moe_balance_loss((_diagnostic(),))
    assert importance.isfinite() and load.isfinite() and entropy.isfinite()


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


def test_trainer_performs_real_updates_for_all_tasks(tmp_path: Path) -> None:
    config = _smoke_config()
    trainer = Trainer(build_model(config), config, torch.device("cpu"), tmp_path)
    for task in TaskType:
        result = trainer.train_step(task)
        assert torch.isfinite(result.total)
    assert trainer.state.global_step == 3


def test_train_displays_one_progress_bar_per_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tfs_moe_fusion.trainer as trainer_module

    config = _smoke_config()
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
