"""TITA-inspired interaction and lightweight operation-adaptive fusion.

Reference: https://github.com/huxingyuabc/TITA (network_swinfusion.py, OAF.py).
This is an independent adaptation, not a reproduction of IPA or CARAFE OAF.
Inputs are ordered: A=visible, B=infrared for VIF/SEG; two RGB sources for MFIF.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RelationEstimator(nn.Module):
    """Learn an interaction-demand gate, not a source-selection probability."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4 * channels, max(4, channels // 4), 1),
            nn.GELU(),
            nn.Conv2d(max(4, channels // 4), 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return self.net(torch.cat((a, b, (a - b).abs(), a * b), dim=1))


class BidirectionalWindowCrossAttention(nn.Module):
    """Shared pre-norm QKV, vectorized over directions, batches and windows."""

    def __init__(self, channels: int, heads: int, window_size: int = 8) -> None:
        super().__init__()
        if heads <= 0 or channels <= 0 or channels % heads:
            raise ValueError("channels must be positive and divisible by heads")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.heads, self.window_size = heads, window_size
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.projection = nn.Linear(channels, channels)

    def _partition(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        size = self.window_size
        return (
            x.reshape(batch, channels, height // size, size, width // size, size)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(-1, size * size, channels)
        )

    def forward(self, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        if a.shape != b.shape or a.ndim != 4:
            raise ValueError("cross attention requires equal [B,C,H,W] sources")
        batch, channels, height, width = a.shape
        size = self.window_size
        pad = (0, (-width) % size, 0, (-height) % size)
        x = F.pad(torch.cat((a, b), dim=0), pad)
        padded_h, padded_w = x.shape[-2:]
        tokens = self.norm(self._partition(x))
        q, k, v = (
            self.qkv(tokens)
            .reshape(2, -1, size * size, 3, self.heads, channels // self.heads)
            .permute(3, 0, 1, 4, 2, 5)
            .unbind(0)
        )
        # Padded keys must never dilute real pixels, including 1x1 inputs.
        mask = None
        if pad[1] or pad[3]:
            valid = F.pad(torch.ones((batch, 1, height, width), device=a.device), pad)
            mask = self._partition(valid).squeeze(-1).bool().repeat(2, 1)
            mask = mask[:, None, None, :]
        with torch.autocast(device_type=a.device.type, enabled=False):
            # Use the 4D SDPA layout so optimized kernels can dispatch directly.
            update = F.scaled_dot_product_attention(
                q.flatten(0, 1).float(),
                k.flip(0).flatten(0, 1).float(),
                v.flip(0).flatten(0, 1).float(),
                attn_mask=mask,
            )
        update = update.transpose(-3, -2).reshape(-1, size * size, channels)
        update = self.projection(update.to(tokens.dtype))
        update = (
            update.reshape(
                2 * batch, padded_h // size, padded_w // size, size, size, channels
            )
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(2 * batch, channels, padded_h, padded_w)
        )
        return update[..., :height, :width].chunk(2, dim=0)


class OAFBlockLite(nn.Module):
    """Six feature-driven candidates, without CARAFE or task embeddings."""

    operation_names = ("a_hpf", "a_add", "a_mul", "b_hpf", "b_add", "b_mul")

    def __init__(self, channels: int) -> None:
        super().__init__()

        def spatial_projection() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.Conv2d(channels, channels, 1),
            )

        self.add = spatial_projection()
        self.multiply = spatial_projection()
        self.weight_predictor = nn.Sequential(
            nn.Linear(3 * channels, max(4, channels // 4)),
            nn.GELU(),
            nn.Linear(max(4, channels // 4), 6),
        )

    @staticmethod
    def high_pass(x: Tensor) -> Tensor:
        # Replication preserves constant signals even at image boundaries.
        return x - F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), 5, stride=1)

    def forward(self, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        pooled = torch.cat(
            (a.mean((2, 3)), b.mean((2, 3)), (a - b).abs().mean((2, 3))), dim=1
        )
        weights = self.weight_predictor(pooled).float().softmax(dim=1)
        w = weights.to(a.dtype)[:, :, None, None, None]
        # Do not materialize a [B,6,C,H,W] candidate stack.
        fused = (
            w[:, 0] * self.high_pass(a)
            + w[:, 1] * (a + self.add(a))
            + w[:, 2] * (a * self.multiply(a).sigmoid())
            + w[:, 3] * self.high_pass(b)
            + w[:, 4] * (b + self.add(b))
            + w[:, 5] * (b * self.multiply(b).sigmoid())
        )
        return fused, weights


def _rms(x: Tensor) -> Tensor:
    return x.detach().float().square().mean().sqrt()


class InteractionAdaptiveFusion(nn.Module):
    """Conservative residual correction to the existing spatial anchor."""

    def __init__(
        self,
        channels: int,
        heads: int | None = None,
        window_size: int = 8,
        alpha_init: float = 0.1,
        beta_init: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0 < alpha_init < 1 or not 0 < beta_init < 1:
            raise ValueError("IACF initial scales must be in (0,1)")
        self.attention = (
            BidirectionalWindowCrossAttention(channels, heads, window_size)
            if heads is not None
            else None
        )
        self.relation = RelationEstimator(channels) if heads is not None else None
        self.cross_logit = (
            nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))
            if heads is not None
            else None
        )
        self.oaf_logit = nn.Parameter(
            torch.tensor(math.log(beta_init / (1 - beta_init)))
        )
        self.oaf = OAFBlockLite(channels)

    def forward(
        self, a: Tensor, b: Tensor, anchor: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        diagnostics = {}
        if self.attention is not None:
            relation = self.relation(a, b)
            delta_a, delta_b = self.attention(a, b)
            alpha = self.cross_logit.sigmoid()
            update_a, update_b = alpha * relation * delta_a, alpha * relation * delta_b
            diagnostics.update(
                {
                    "relation_mean": relation.detach().float().mean(),
                    "relation_std": relation.detach().float().std(unbiased=False),
                    "cross_update_a_rms": _rms(update_a),
                    "cross_update_b_rms": _rms(update_b),
                    "cross_update_a_relative_rms": _rms(update_a)
                    / _rms(a).clamp_min(1e-8),
                    "cross_update_b_relative_rms": _rms(update_b)
                    / _rms(b).clamp_min(1e-8),
                    "cross_scale": alpha.detach(),
                }
            )
            a, b = a + update_a, b + update_b
        oaf, weights = self.oaf(a, b)
        beta = self.oaf_logit.sigmoid()
        delta = oaf - anchor
        diagnostics.update(
            {
                "anchor_oaf_delta_rms": _rms(delta),
                "oaf_update_relative_rms": _rms(beta * delta)
                / _rms(anchor).clamp_min(1e-8),
                "oaf_scale": beta.detach(),
                **{
                    f"oaf_weight_{name}": value
                    for name, value in zip(
                        self.oaf.operation_names, weights.detach().mean(0), strict=True
                    )
                },
            }
        )
        return anchor + beta * delta, diagnostics
