"""ACL-style local-global source encoder blocks."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("drop_path must be in [0, 1)")
        self.probability = probability

    def forward(self, tensor: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return tensor
        keep = 1.0 - self.probability
        shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 1)
        mask = tensor.new_empty(shape).bernoulli_(keep)
        return tensor * mask / keep


class LinearAttention2D(nn.Module):
    """ACL linear attention with local positional enhancement."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        qkv_bias: bool = True,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels <= 0 or num_heads <= 0 or channels % num_heads:
            raise ValueError("channels must be positive and divisible by num_heads")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.epsilon = epsilon
        self.qk = nn.Linear(channels, channels * 2, bias=qkv_bias)
        self.lepe = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, tensor: Tensor) -> Tensor:
        if tensor.ndim != 4 or tensor.shape[1] != self.channels:
            raise ValueError(f"LinearAttention2D expects [B,{self.channels},H,W]")
        batch, channels, height, width = tensor.shape
        tokens = tensor.flatten(2).transpose(1, 2)
        lepe = self.lepe(tensor)

        # Keep the kernelization and both attention contractions in FP32 even
        # when the surrounding model runs under FP16/BF16 autocast.
        with torch.autocast(device_type=tensor.device.type, enabled=False):
            values = tokens.float()
            qk = functional.linear(
                values,
                self.qk.weight.float(),
                self.qk.bias.float() if self.qk.bias is not None else None,
            )
            query, key = qk.chunk(2, dim=-1)
            query = functional.elu(query) + 1.0
            key = functional.elu(key) + 1.0

            query = query.reshape(batch, -1, self.num_heads, self.head_dim).transpose(
                1, 2
            )
            key = key.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
            values = values.reshape(batch, -1, self.num_heads, self.head_dim).transpose(
                1, 2
            )

            token_count = height * width
            scale = token_count**-0.5
            key_value = (key.transpose(-2, -1) * scale) @ (values * scale)
            denominator = query @ key.mean(dim=-2, keepdim=True).transpose(-2, -1)
            normalizer = denominator.clamp_min(self.epsilon).reciprocal()
            attended = (query @ key_value) * normalizer
            attended = attended.transpose(1, 2).reshape(batch, token_count, channels)
            attended = attended.transpose(1, 2).reshape(batch, channels, height, width)

        return attended.to(dtype=tensor.dtype) + lepe


class LAMA2D(nn.Module):
    """Linear-attention Mamba block adapted to channel-first feature maps."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0 or mlp_ratio <= 0:
            raise ValueError("channels and mlp_ratio must be positive")
        hidden_channels = int(channels * mlp_ratio)
        self.channels = channels
        self.cpe1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm1 = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, channels)
        self.act_proj = nn.Linear(channels, channels)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.attention = LinearAttention2D(channels, num_heads, qkv_bias)
        self.out_proj = nn.Linear(channels, channels)
        self.drop_path = _DropPath(drop_path)
        self.cpe2 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, tensor: Tensor) -> Tensor:
        if tensor.ndim != 4 or tensor.shape[1] != self.channels:
            raise ValueError(f"LAMA2D expects [B,{self.channels},H,W]")
        batch, channels, height, width = tensor.shape
        tensor = tensor + self.cpe1(tensor)
        tokens = tensor.flatten(2).transpose(1, 2)
        shortcut = tokens

        normalized = self.norm1(tokens)
        gate = functional.silu(self.act_proj(normalized))
        features = (
            self.in_proj(normalized)
            .transpose(1, 2)
            .reshape(batch, channels, height, width)
        )
        features = functional.silu(self.depthwise(features))
        features = self.attention(features).flatten(2).transpose(1, 2)
        tokens = shortcut + self.drop_path(self.out_proj(features * gate))

        tensor = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        tensor = tensor + self.cpe2(tensor)
        tokens = tensor.flatten(2).transpose(1, 2)
        tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens)))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class MDCBlock(nn.Module):
    """Multi-scale dilated convolutions for local detail enhancement."""

    def __init__(
        self,
        channels: int,
        kernels: tuple[int, int] = (3, 5),
        dilation: int = 2,
    ) -> None:
        super().__init__()
        if channels <= 0 or dilation <= 0:
            raise ValueError("channels and dilation must be positive")
        if len(kernels) != 2 or any(k <= 0 or k % 2 == 0 for k in kernels):
            raise ValueError("MDC kernels must contain two positive odd values")

        def branch(kernel: int) -> nn.Sequential:
            padding = dilation * (kernel - 1) // 2
            return nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                ),
                nn.GroupNorm(_groups(channels), channels),
                nn.SiLU(),
            )

        self.branch_a = branch(kernels[0])
        self.branch_b = branch(kernels[1])
        self.fuse = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, tensor: Tensor) -> Tensor:
        return tensor + self.fuse(
            torch.cat((self.branch_a(tensor), self.branch_b(tensor)), dim=1)
        )


class ACLStage(nn.Module):
    """A stack of LAMA blocks followed by an optional MDC block."""

    def __init__(
        self,
        channels: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        use_mdc: bool = False,
        mdc_kernels: tuple[int, int] = (3, 5),
        mdc_dilation: int = 2,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.blocks = nn.Sequential(
            *[
                LAMA2D(channels, num_heads, mlp_ratio, qkv_bias, drop_path)
                for _ in range(depth)
            ]
        )
        self.mdc = (
            MDCBlock(channels, mdc_kernels, mdc_dilation) if use_mdc else nn.Identity()
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return self.mdc(self.blocks(tensor))
