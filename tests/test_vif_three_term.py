from pathlib import Path

import pytest
import torch

from tfs_moe_fusion.config import load_config
from tfs_moe_fusion.losses import (
    LossContext,
    MultiTaskLossManager,
    bounded_simplex_weights,
    multi_scale_reliable_angular_loss,
    reliable_gradient_target,
    reliable_sobel,
    separable_ssim_map,
    ssim_map,
)
from tfs_moe_fusion.trainer import load_checkpoint, save_checkpoint
from tfs_moe_fusion.types import FusionOutput, TaskType
from tfs_moe_fusion.utils import make_probe_batch

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / "configs/stage12_vif_three_term.yaml")


def test_ir_edges_survive_even_when_ir_is_darker_than_visible():
    c = config().training.losses.vif
    visible = torch.full((1, 1, 17, 21), 0.8)
    infrared = torch.full_like(visible, 0.2)
    infrared[..., 10:] = 0.7
    for dilation in (1, 2):
        gx, gy = reliable_sobel(visible, dilation)
        assert gx.abs().max() < 1e-6 and gy.abs().max() < 1e-6
        tx, _, reliability = reliable_gradient_target(visible, infrared, c, dilation)
        assert tx.abs().max() > 0.3
        assert reliability.max() > 0.7
    weights = torch.tensor([0.7, 0.3])
    missing, _ = multi_scale_reliable_angular_loss(visible, visible, infrared, c, weights)
    preserved, _ = multi_scale_reliable_angular_loss(infrared, visible, infrared, c, weights)
    assert preserved < missing
    _, details = multi_scale_reliable_angular_loss(visible, visible, visible, c, weights)
    assert details["gradient_angular/d1"] == 0
    assert details["gradient_angular/d2"] == 0


def test_separable_ssim_matches_reference_for_all_windows():
    torch.manual_seed(9)
    left = torch.rand(1, 1, 13, 15)
    right = torch.rand_like(left)
    for size, sigma in ((11, 1.5), (25, 3.5), (49, 7.0)):
        actual = separable_ssim_map(left, right, size, sigma)
        expected = ssim_map(left, right, size, sigma)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("precision", [torch.float32, torch.bfloat16])
def test_three_terms_update_all_loss_parameters_and_restore(tmp_path, precision):
    torch.manual_seed(12)
    c = config()
    manager = MultiTaskLossManager(c.training.losses)
    batch = make_probe_batch(c, TaskType.VIF, spatial_size=(17, 21))
    model = torch.nn.Conv2d(3, 1, 1)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters()},
        {"params": manager.parameters(), "weight_decay": 0.0},
    ], lr=1e-3)
    torch.testing.assert_close(
        bounded_simplex_weights(manager.scale_logits, 0.1), torch.tensor([0.7, 0.3])
    )
    torch.testing.assert_close(
        bounded_simplex_weights(manager.window_logits, 0.05), torch.tensor([0.5, 0.3, 0.2])
    )
    before = {k: p.detach().clone() for k, p in manager.named_parameters()}

    def update(model, manager, optimizer):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=precision, enabled=precision == torch.bfloat16):
            y = model(batch.visible_source.image).sigmoid()
            output = FusionOutput(
                y.expand(-1, 3, -1, -1), TaskType.VIF,
                fused_y=y, coarse_y=y, refinement_y=torch.zeros_like(y),
            )
            result = manager(LossContext(batch, output, TaskType.VIF, 0, 0))
        assert set(result.components) == {"fusion/intensity", "fusion/gradient", "fusion/ssim"}
        torch.testing.assert_close(result.total, sum(result.weighted_components.values()))
        result.total.backward()
        for p in manager.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
        optimizer.step()
        manager.clamp_adaptive_parameters_()

    update(model, manager, optimizer)
    for name, parameter in manager.named_parameters():
        assert not torch.equal(before[name], parameter)
    checkpoint = save_checkpoint(
        tmp_path / "full.pt", model, c, epoch=0, global_step=1, optimizer=optimizer,
        engine_state={"loss_manager": manager.state_dict()},
    )
    restored_model = torch.nn.Conv2d(3, 1, 1)
    restored_manager = MultiTaskLossManager(c.training.losses)
    restored_optimizer = torch.optim.AdamW([
        {"params": restored_model.parameters()},
        {"params": restored_manager.parameters(), "weight_decay": 0.0},
    ], lr=1e-3)
    report = load_checkpoint(checkpoint, restored_model, optimizer=restored_optimizer)
    restored_manager.load_state_dict(report.engine_state["loss_manager"])
    update(model, manager, optimizer)
    update(restored_model, restored_manager, restored_optimizer)
    for key, value in manager.state_dict().items():
        torch.testing.assert_close(value, restored_manager.state_dict()[key])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored_model.state_dict()[key])
