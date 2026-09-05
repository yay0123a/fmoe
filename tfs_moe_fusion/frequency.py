"""TFS-MoE-Fusion consolidated implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional


@dataclass(slots=True)
class WaveletMeta:
    original_height: int
    original_width: int
    pad_bottom: int
    pad_right: int


@dataclass(slots=True)
class WaveletBands:
    ll: Tensor
    lh: Tensor
    hl: Tensor
    hh: Tensor
    meta: WaveletMeta | tuple[int, int]

    def __post_init__(self) -> None:
        if isinstance(self.meta, tuple):
            height, width = self.meta
            self.meta = WaveletMeta(height, width, height % 2, width % 2)

    @property
    def original_size(self) -> tuple[int, int]:
        return self.meta.original_height, self.meta.original_width

    def high_frequency(self) -> Tensor:
        return torch.cat((self.lh, self.hl, self.hh), dim=1)


class HaarDWT2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        filters = (
            torch.tensor(
                [
                    [[1, 1], [1, 1]],
                    [[-1, -1], [1, 1]],
                    [[-1, 1], [-1, 1]],
                    [[1, -1], [-1, 1]],
                ],
                dtype=torch.float32,
            )
            * 0.5
        )
        self.register_buffer("filters", filters[:, None], persistent=True)

    def forward(self, tensor: Tensor) -> WaveletBands:
        if tensor.ndim != 4:
            raise ValueError("HaarDWT2D expects [B,C,H,W]")
        height, width = tensor.shape[-2:]
        pad_bottom, pad_right = height % 2, width % 2
        if pad_bottom or pad_right:
            mode = "reflect" if height > 1 and width > 1 else "replicate"
            tensor = functional.pad(tensor, (0, pad_right, 0, pad_bottom), mode=mode)
        channels = tensor.shape[1]
        filters = self.filters.to(tensor).repeat(channels, 1, 1, 1)
        packed = functional.conv2d(tensor, filters, stride=2, groups=channels)
        packed = packed.view(tensor.shape[0], channels, 4, *packed.shape[-2:])
        return WaveletBands(
            packed[:, :, 0],
            packed[:, :, 1],
            packed[:, :, 2],
            packed[:, :, 3],
            WaveletMeta(height, width, pad_bottom, pad_right),
        )


def local_spectral_evidence_from_bands(
    bands: WaveletBands,
    output_size: tuple[int, int],
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Build FP32 local energy maps from an already computed Haar transform."""
    if min(output_size) <= 0:
        raise ValueError("local spectral evidence requires a positive output size")
    with torch.autocast(device_type=bands.ll.device.type, enabled=False):
        low = bands.ll.float().square().mean(1, keepdim=True)
        high = torch.stack((bands.lh, bands.hl, bands.hh), dim=2)
        high = high.float().square().mean((1, 2), keepdim=True).squeeze(2)
        low = functional.interpolate(low, output_size, mode="bilinear", align_corners=False)
        high = functional.interpolate(high, output_size, mode="bilinear", align_corners=False)
    return low.to(dtype), high.to(dtype)


def local_spectral_evidence(
    feature: Tensor, output_size: tuple[int, int]
) -> tuple[Tensor, Tensor]:
    """Return shape-safe local Haar low/high energy maps in FP32 precision."""
    if feature.ndim != 4 or min(output_size) <= 0:
        raise ValueError("local_spectral_evidence expects [B,C,H,W] and positive size")
    return local_spectral_evidence_from_bands(
        HaarDWT2D().to(feature)(feature.float()), output_size, feature.dtype
    )


class HaarIDWT2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        filters = (
            torch.tensor(
                [
                    [[1, 1], [1, 1]],
                    [[-1, -1], [1, 1]],
                    [[-1, 1], [-1, 1]],
                    [[1, -1], [-1, 1]],
                ],
                dtype=torch.float32,
            )
            * 0.5
        )
        self.register_buffer("filters", filters[:, None], persistent=True)

    def forward(self, bands: WaveletBands) -> Tensor:
        shapes = {
            tuple(value.shape) for value in (bands.ll, bands.lh, bands.hl, bands.hh)
        }
        if len(shapes) != 1 or bands.ll.ndim != 4:
            raise ValueError("All Haar bands must have the same [B,C,H,W] shape")
        ll, lh, hl, hh = bands.ll, bands.lh, bands.hl, bands.hh
        channels = ll.shape[1]
        packed = torch.stack((ll, lh, hl, hh), dim=2).reshape(
            ll.shape[0], channels * 4, *ll.shape[-2:]
        )
        filters = self.filters.to(ll).repeat(channels, 1, 1, 1)
        output = functional.conv_transpose2d(packed, filters, stride=2, groups=channels)
        height, width = bands.original_size
        return output[..., :height, :width]


from dataclasses import dataclass

from torch import nn


def fft2(tensor: Tensor) -> Tensor:
    return torch.fft.fft2(tensor, dim=(-2, -1), norm="ortho")


def ifft2(spectrum: Tensor) -> Tensor:
    return torch.fft.ifft2(spectrum, dim=(-2, -1), norm="ortho")


def rfft2(tensor: Tensor) -> Tensor:
    return torch.fft.rfft2(tensor, dim=(-2, -1), norm="ortho")


def irfft2(spectrum: Tensor, size: tuple[int, int]) -> Tensor:
    return torch.fft.irfft2(spectrum, s=size, dim=(-2, -1), norm="ortho")


def fftshift2d(tensor: Tensor) -> Tensor:
    return torch.fft.fftshift(tensor, dim=(-2, -1))


def ifftshift2d(tensor: Tensor) -> Tensor:
    return torch.fft.ifftshift(tensor, dim=(-2, -1))


def fourier_amplitude(spectrum: Tensor) -> Tensor:
    return spectrum.abs()


def fourier_phase(spectrum: Tensor) -> Tensor:
    return torch.angle(spectrum)


@dataclass(frozen=True, slots=True)
class FrequencyMaskGenerator:
    thresholds: tuple[float, ...] = (0.0, 1 / 16, 1 / 8, 1 / 4, 0.5)
    mode: str = "radial"

    def coordinate(self, height: int, width: int, device: torch.device) -> Tensor:
        fy = torch.fft.fftfreq(height, device=device).abs()[:, None]
        fx = torch.fft.rfftfreq(width, device=device).abs()[None, :]
        if self.mode == "radial":
            return torch.sqrt(fy.square() + fx.square()).clamp_max(0.5)
        if self.mode in {"square", "max_norm"}:
            return torch.maximum(fy, fx)
        raise ValueError("Frequency mask mode must be radial or square/max_norm")

    def masks(
        self, height: int, width: int, device: torch.device
    ) -> tuple[Tensor, ...]:
        coordinate = self.coordinate(height, width, device)
        values = []
        for index, (left, right) in enumerate(
            zip(self.thresholds, self.thresholds[1:])
        ):
            values.append(
                (coordinate >= left)
                & (
                    coordinate <= right
                    if index == len(self.thresholds) - 2
                    else coordinate < right
                )
            )
        return tuple(values)


class FourierBandEnergy(nn.Module):
    def __init__(
        self,
        thresholds: tuple[float, ...] = (0.0, 1 / 16, 1 / 8, 1 / 4, 0.5),
        mode: str = "square",
    ) -> None:
        super().__init__()
        self.generator = FrequencyMaskGenerator(thresholds, mode)

    def forward(self, tensor: Tensor) -> Tensor:
        spatial = (
            tensor.float()
            if tensor.dtype in {torch.float16, torch.bfloat16}
            else tensor
        )
        power = rfft2(spatial).abs().square().sum(dim=1)
        values = []
        for mask in self.generator.masks(
            tensor.shape[-2], tensor.shape[-1], tensor.device
        ):
            values.append((power * mask).sum(dim=(-2, -1)))
        energies = torch.stack(values, dim=1)
        return energies / energies.sum(dim=1, keepdim=True).clamp_min(1e-12)


import math

from torch import nn


class AFNO2D(nn.Module):
    def __init__(
        self,
        channels: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        residual: bool = True,
        hidden_size_factor: int = 1,
    ) -> None:
        super().__init__()
        if channels <= 0 or num_blocks <= 0 or channels % num_blocks:
            raise ValueError("channels must be positive and divisible by num_blocks")
        if not 0.0 < hard_thresholding_fraction <= 1.0:
            raise ValueError("hard_thresholding_fraction must be in (0, 1]")
        self.channels = channels
        self.num_blocks = num_blocks
        self.block_size = channels // num_blocks
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.residual = residual
        hidden = self.block_size * hidden_size_factor
        scale = 0.02
        self.w1_real = nn.Parameter(
            scale * torch.randn(num_blocks, self.block_size, hidden)
        )
        self.w1_imag = nn.Parameter(
            scale * torch.randn(num_blocks, self.block_size, hidden)
        )
        self.b1_real = nn.Parameter(torch.zeros(num_blocks, hidden))
        self.b1_imag = nn.Parameter(torch.zeros(num_blocks, hidden))
        self.w2_real = nn.Parameter(
            scale * torch.randn(num_blocks, hidden, self.block_size)
        )
        self.w2_imag = nn.Parameter(
            scale * torch.randn(num_blocks, hidden, self.block_size)
        )
        self.b2_real = nn.Parameter(torch.zeros(num_blocks, self.block_size))
        self.b2_imag = nn.Parameter(torch.zeros(num_blocks, self.block_size))

    def forward(self, tensor: Tensor) -> Tensor:
        if tensor.ndim != 4 or tensor.shape[1] != self.channels:
            raise ValueError(f"AFNO2D expects [B,{self.channels},H,W]")
        original_dtype = tensor.dtype
        # Autocast can reduce the learned real/imaginary channel mixing back to
        # BF16 even after explicit .float() calls. view_as_complex only accepts
        # FP32/FP64 pairs, so the entire spectral path must run outside autocast.
        with torch.autocast(device_type=tensor.device.type, enabled=False):
            output = self._forward_without_autocast(tensor)
        return output.to(original_dtype)

    def _forward_without_autocast(self, tensor: Tensor) -> Tensor:
        spatial = (
            tensor.float()
            if tensor.dtype in {torch.float16, torch.bfloat16}
            else tensor
        )
        spectrum = torch.fft.rfft2(spatial, dim=(-2, -1), norm="ortho")
        batch, _, height, half_width = spectrum.shape
        blocked = spectrum.reshape(
            batch, self.num_blocks, self.block_size, height, half_width
        ).permute(0, 3, 4, 1, 2)
        real, imag = blocked.real, blocked.imag
        # Parameters may have been converted with model.half()/bfloat16(). FFT
        # arithmetic and its learned channel mixing intentionally stay FP32.
        w1_real, w1_imag = self.w1_real.float(), self.w1_imag.float()
        w2_real, w2_imag = self.w2_real.float(), self.w2_imag.float()
        b1_real, b1_imag = self.b1_real.float(), self.b1_imag.float()
        b2_real, b2_imag = self.b2_real.float(), self.b2_imag.float()

        hidden_real = torch.einsum("bhwnc,ncd->bhwnd", real, w1_real)
        hidden_real -= torch.einsum("bhwnc,ncd->bhwnd", imag, w1_imag)
        hidden_real += b1_real
        hidden_imag = torch.einsum("bhwnc,ncd->bhwnd", imag, w1_real)
        hidden_imag += torch.einsum("bhwnc,ncd->bhwnd", real, w1_imag)
        hidden_imag += b1_imag
        hidden_real, hidden_imag = (
            functional.gelu(hidden_real),
            functional.gelu(hidden_imag),
        )

        out_real = torch.einsum("bhwnd,ndc->bhwnc", hidden_real, w2_real)
        out_real -= torch.einsum("bhwnd,ndc->bhwnc", hidden_imag, w2_imag)
        out_real += b2_real
        out_imag = torch.einsum("bhwnd,ndc->bhwnc", hidden_imag, w2_real)
        out_imag += torch.einsum("bhwnd,ndc->bhwnc", hidden_real, w2_imag)
        out_imag += b2_imag
        dense_correction = torch.stack((out_real, out_imag), dim=-1)
        sparse_correction = functional.softshrink(
            dense_correction, lambd=self.sparsity_threshold
        )
        # A straight-through derivative avoids a dead AFNO at initialization
        # when every small coefficient lies inside softshrink's zero interval.
        correction = dense_correction + (sparse_correction - dense_correction).detach()
        correction = torch.view_as_complex(correction.float().contiguous())
        correction = correction.permute(0, 3, 4, 1, 2).reshape_as(spectrum)

        mask = self._low_mode_mask(height, half_width, spectrum.device)
        if self.residual:
            transformed = spectrum + correction * mask
        else:
            # Selected modes are replaced; all unselected modes explicitly bypass.
            transformed = torch.where(mask, correction, spectrum)
        output = torch.fft.irfft2(
            transformed, s=tensor.shape[-2:], dim=(-2, -1), norm="ortho"
        )
        return output

    def _low_mode_mask(
        self, height: int, half_width: int, device: torch.device
    ) -> Tensor:
        fraction = self.hard_thresholding_fraction
        rows = max(1, math.ceil(height * fraction / 2.0))
        columns = max(1, math.ceil(half_width * fraction))
        mask = torch.zeros(height, half_width, dtype=torch.bool, device=device)
        mask[:rows, :columns] = True
        mask[-rows:, :columns] = True
        return mask.view(1, 1, height, half_width)


from torch import nn


class FourierDisjointWeight(nn.Module):
    """Generate complementary frequency kernels from one parameter tensor."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        kernel_num: int,
        groups: int,
    ) -> None:
        super().__init__()
        if kernel_num > kernel_size * kernel_size:
            raise ValueError("kernel_num cannot exceed kernel_size squared")
        self.kernel_num = kernel_num
        self.base_weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        yy, xx = torch.meshgrid(
            torch.arange(kernel_size), torch.arange(kernel_size), indexing="ij"
        )
        radius = ((yy - kernel_size // 2) ** 2 + (xx - kernel_size // 2) ** 2).float()
        order = radius.flatten().argsort()
        masks = torch.zeros(kernel_num, kernel_size * kernel_size)
        for group, indices in enumerate(torch.tensor_split(order, kernel_num)):
            masks[group, indices] = 1.0
        masks = torch.fft.ifftshift(
            masks.view(kernel_num, kernel_size, kernel_size), dim=(-2, -1)
        )
        self.register_buffer("frequency_masks", masks, persistent=True)

    def forward(self) -> Tensor:
        spectrum = torch.fft.fft2(self.base_weight.float(), dim=(-2, -1), norm="ortho")
        bank = torch.fft.ifft2(
            spectrum.unsqueeze(0) * self.frequency_masks[:, None, None],
            dim=(-2, -1),
            norm="ortho",
        ).real
        return bank.to(self.base_weight.dtype) * self.kernel_num


class KernelSpatialModulation(nn.Module):
    """Modulate kernel positions using global and locally summarized content."""

    def __init__(self, channels: int, kernel_num: int, kernel_size: int) -> None:
        super().__init__()
        hidden = max(4, channels // 4)
        outputs = kernel_num * kernel_size * kernel_size
        self.local_summary = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, outputs)
        )
        self.local_mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, outputs)
        )
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        for module in (self.global_mlp[-1], self.local_mlp[-1]):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, tensor: Tensor) -> Tensor:
        global_context = tensor.mean(dim=(-2, -1))
        local_context = self.local_summary(tensor).mean(dim=(-2, -1))
        modulation = self.global_mlp(global_context) + self.local_mlp(local_context)
        return (2.0 * torch.sigmoid(modulation)).view(
            -1, self.kernel_num, self.kernel_size, self.kernel_size
        )


class FrequencyBandModulation(nn.Module):
    """Spatially mix the frequency-diverse kernel responses."""

    def __init__(
        self, channels: int, kernel_num: int, num_frequency_bands: int = 4
    ) -> None:
        super().__init__()
        self.local_logits = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, num_frequency_bands, 1),
        )
        self.global_logits = nn.Linear(channels, num_frequency_bands)
        self.band_to_kernel = nn.Conv2d(num_frequency_bands, kernel_num, 1, bias=False)
        nn.init.zeros_(self.local_logits[-1].weight)
        nn.init.zeros_(self.local_logits[-1].bias)
        nn.init.zeros_(self.global_logits.weight)
        nn.init.zeros_(self.global_logits.bias)

    def forward(self, tensor: Tensor) -> Tensor:
        local = self.local_logits(tensor)
        global_context = self.global_logits(tensor.mean(dim=(-2, -1)))[:, :, None, None]
        bands = torch.sigmoid(local + global_context)
        return torch.softmax(self.band_to_kernel(bands), dim=1), bands


class FDConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        kernel_num: int = 8,
        use_fdw: bool = True,
        use_ksm: bool = True,
        use_fbm: bool = True,
        groups: int = 1,
        bias: bool = True,
        stride: int = 1,
        padding: int | None = None,
        num_frequency_bands: int = 4,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        if in_channels % groups or out_channels % groups:
            raise ValueError("channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.groups = groups
        self.use_fdw = use_fdw
        self.use_ksm = use_ksm
        self.use_fbm = use_fbm
        self.stride = stride
        self.padding = kernel_size // 2 if padding is None else padding
        self.fdw = FourierDisjointWeight(
            in_channels, out_channels, kernel_size, kernel_num, groups
        )
        self.ksm = KernelSpatialModulation(in_channels, kernel_num, kernel_size)
        self.fbm = FrequencyBandModulation(in_channels, kernel_num, num_frequency_bands)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def kernel_bank(self) -> Tensor:
        """Return generated kernels for diagnostics."""
        if self.use_fdw:
            return self.fdw()
        return self.fdw.base_weight.unsqueeze(0).expand(self.kernel_num, -1, -1, -1, -1)

    def forward(
        self, tensor: Tensor, return_diagnostics: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if tensor.ndim != 4 or tensor.shape[1] != self.in_channels:
            raise ValueError(f"FDConv2d expects [B,{self.in_channels},H,W]")
        batch, _, height, width = tensor.shape
        bank = self.kernel_bank()
        if self.use_ksm:
            modulation = self.ksm(tensor)[:, :, None, None]
            weights = bank[None] * modulation
        else:
            weights = bank[None].expand(batch, -1, -1, -1, -1, -1)

        expanded = tensor[:, None].expand(-1, self.kernel_num, -1, -1, -1)
        expanded = expanded.reshape(
            1, batch * self.kernel_num * self.in_channels, height, width
        )
        weights = weights.reshape(
            batch * self.kernel_num * self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size,
        )
        branches = functional.conv2d(
            expanded,
            weights,
            stride=self.stride,
            padding=self.padding,
            groups=batch * self.kernel_num * self.groups,
        )
        out_height, out_width = branches.shape[-2:]
        branches = branches.view(
            batch, self.kernel_num, self.out_channels, out_height, out_width
        )
        if self.use_fbm:
            mixture_map, band_maps = self.fbm(tensor)
            if mixture_map.shape[-2:] != (out_height, out_width):
                mixture_map = functional.interpolate(
                    mixture_map,
                    (out_height, out_width),
                    mode="bilinear",
                    align_corners=False,
                )
                band_maps = functional.interpolate(
                    band_maps,
                    (out_height, out_width),
                    mode="bilinear",
                    align_corners=False,
                )
            mixture = mixture_map[:, :, None]
        else:
            mixture = tensor.new_full(
                (batch, self.kernel_num, 1, out_height, out_width),
                1.0 / self.kernel_num,
            )
            band_maps = tensor.new_zeros(batch, 4, out_height, out_width)
        output = (branches * mixture).sum(dim=1)
        if self.bias is not None:
            output = output + self.bias[None, :, None, None]
        if return_diagnostics:
            return output, {
                "kernel_mix_weights": mixture.squeeze(2),
                "frequency_band_maps": band_maps,
                "frequency_index_masks": self.fdw.frequency_masks,
            }
        return output


from torch import nn


class CrossFrequencyConditioner(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        hidden = max(4, channels // 4)
        self.low_projection = nn.Conv2d(channels, channels, 1)
        self.high_projection = nn.Conv2d(channels, channels, 1)
        self.low_to_high = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.high_to_low = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.low_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.high_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.high_concat_projection = nn.Conv2d(channels * 3, channels, 1)
        self.low_refine = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.high_refine = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(
        self, low: Tensor, high: Tensor, high_is_concatenated: bool = False
    ) -> tuple[Tensor, Tensor]:
        if high_is_concatenated:
            high = self.high_concat_projection(high)
        if low.shape != high.shape or low.ndim != 4 or low.shape[1] != self.channels:
            raise ValueError(f"low/high must have equal [B,{self.channels},H,W] shapes")
        low_update = self.high_projection(high) * self.high_to_low(high)
        high_update = self.low_projection(low) * self.low_to_high(low)
        return (
            low + self.low_scale * self.low_refine(low_update),
            high + self.high_scale * self.high_refine(high_update),
        )


from tfs_moe_fusion.types import SpectralStatistics


def compute_dwt_energy(tensor: Tensor) -> Tensor:
    bands = HaarDWT2D()(tensor.float())
    energies = torch.stack(
        [
            value.square().sum(dim=(1, 2, 3))
            for value in (bands.ll, bands.lh, bands.hl, bands.hh)
        ],
        dim=1,
    )
    return energies / energies.sum(dim=1, keepdim=True).clamp_min(1e-12)


def spectral_statistics(tensor: Tensor, stage: str) -> SpectralStatistics:
    """Measure low/high and radial power with an orthonormal 2-D FFT."""
    spatial = (
        tensor.float() if tensor.dtype in {torch.float16, torch.bfloat16} else tensor
    )
    rings = FourierBandEnergy()(spatial)
    dwt = compute_dwt_energy(spatial)
    low = rings[:, 0] + rings[:, 1]
    mid = rings[:, 2]
    high = rings[:, 3]
    return SpectralStatistics(stage, low, high, rings, dwt, mid)


class SpectralStatsExtractor(torch.nn.Module):
    def __init__(self, detach: bool = False) -> None:
        super().__init__()
        self.detach = detach

    def forward(self, tensor: Tensor, stage: str = "unknown") -> SpectralStatistics:
        stats = spectral_statistics(tensor, stage)
        if not self.detach:
            return stats
        return SpectralStatistics(
            stage,
            stats.low_energy.detach(),
            stats.high_energy.detach(),
            stats.radial_energy.detach() if stats.radial_energy is not None else None,
            stats.dwt_energy.detach() if stats.dwt_energy is not None else None,
            stats.mid_energy.detach() if stats.mid_energy is not None else None,
        )


from dataclasses import dataclass

import torch
from torch import nn


def compatible_num_blocks(channels: int, preferred: int) -> int:
    return max(1, math.gcd(channels, preferred))


class FrequencyFoundationBlock(nn.Module):
    """Refine LL globally and wavelet details with frequency-diverse kernels."""

    def __init__(
        self,
        channels: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        kernel_size: int = 3,
        kernel_num: int = 8,
        residual_scale: float = 0.1,
        use_afno: bool = True,
        use_fdconv: bool = True,
        use_cross_frequency: bool = True,
    ) -> None:
        super().__init__()
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()
        self.low_refiner: nn.Module = (
            AFNO2D(
                channels,
                compatible_num_blocks(channels, num_blocks),
                sparsity_threshold,
                hard_thresholding_fraction,
                residual=True,
            )
            if use_afno
            else nn.Identity()
        )
        self.detail_refiner: nn.Module = (
            FDConv2d(channels, channels, kernel_size, kernel_num, groups=channels)
            if use_fdconv
            else nn.Identity()
        )
        self.cross_frequency = (
            CrossFrequencyConditioner(channels) if use_cross_frequency else None
        )
        self.residual_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), residual_scale)
        )

    def forward(
        self, tensor: Tensor, stage: str = "unknown", return_details: bool = False
    ) -> tuple[Tensor, SpectralStatistics] | FrequencyFoundationOutput:
        bands = self.dwt(tensor)
        low = self.low_refiner(bands.ll)
        details = [self.detail_refiner(item) for item in (bands.lh, bands.hl, bands.hh)]
        aggregate = sum(details) / 3.0
        if self.cross_frequency is not None:
            low, conditioned = self.cross_frequency(low, aggregate)
            correction = conditioned - aggregate
        else:
            correction = torch.zeros_like(aggregate)
        reconstructed = self.idwt(
            WaveletBands(
                low,
                details[0] + correction,
                details[1] + correction,
                details[2] + correction,
                bands.original_size,
            )
        )
        output = tensor + self.residual_scale * (reconstructed - tensor)
        stats = spectral_statistics(output, stage)
        if return_details:
            return FrequencyFoundationOutput(
                bands.ll,
                bands.high_frequency(),
                low,
                torch.cat(tuple(details), dim=1),
                output,
                stats,
                bands,
            )
        return output, stats


@dataclass(slots=True)
class FrequencyFoundationOutput:
    low: Tensor
    high: Tensor
    low_refined: Tensor
    high_refined: Tensor
    mixed: Tensor
    stats: SpectralStatistics
    wavelet: WaveletBands
