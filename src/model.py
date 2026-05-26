"""StyleGAN-inspired generator/discriminator for Project 2.

The implementation is self-contained PyTorch code.  It keeps the public names
used by the original baseline package (`Generator`, `Discriminator`, `EMA`) so
the training, sampling, and export scripts can keep the same flow.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as _sn


def _normalize_channels(channels: dict[Any, Any]) -> dict[int, int]:
    return {int(k): int(v) for k, v in channels.items()}


@dataclass
class GeneratorConfig:
    z_dim: int
    resolutions: list[int]
    channels: dict[int, int]
    w_dim: int = 512
    mapping_layers: int = 4
    mapping_lr_mul: float = 0.01
    use_noise: bool = True
    # Accepted for backward-compatible config loading; unused by StyleGAN G.
    norm_type: str | None = None
    gn_groups: int | None = None
    attention_resolutions: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.resolutions = [int(r) for r in self.resolutions]
        self.channels = _normalize_channels(self.channels)
        self.attention_resolutions = [int(r) for r in self.attention_resolutions]
        if self.z_dim != 512:
            raise ValueError(f"Project spec requires z_dim=512, got {self.z_dim}")
        if self.resolutions[0] != 4:
            raise ValueError("StyleGAN synthesis must start at 4x4")
        for prev, cur in zip(self.resolutions, self.resolutions[1:]):
            if cur != prev * 2:
                raise ValueError(f"resolutions must double each step, got {prev}->{cur}")
        for r in self.resolutions:
            if r not in self.channels:
                raise ValueError(f"channels missing entry for resolution {r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GeneratorConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in allowed})


@dataclass
class DiscriminatorConfig:
    resolutions: list[int]
    channels: dict[int, int]
    use_spectral_norm: bool = True
    minibatch_std_group: int = 4
    # Accepted for backward-compatible config loading; unused by StyleGAN D.
    attention_resolutions: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.resolutions = [int(r) for r in self.resolutions]
        self.channels = _normalize_channels(self.channels)
        self.attention_resolutions = [int(r) for r in self.attention_resolutions]
        for prev, cur in zip(self.resolutions, self.resolutions[1:]):
            if cur * 2 != prev:
                raise ValueError(f"D resolutions must halve each step, got {prev}->{cur}")
        for r in self.resolutions:
            if r not in self.channels:
                raise ValueError(f"channels missing entry for resolution {r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DiscriminatorConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in allowed})


def sn(module: nn.Module) -> nn.Module:
    return _sn(module)


class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool = True,
        lr_mul: float = 1.0,
        activation: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.scale = (1.0 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias * self.lr_mul if self.bias is not None else None
        x = F.linear(x, self.weight * self.scale, bias)
        if self.activation:
            x = F.leaky_relu(x, 0.2) * math.sqrt(2)
        return x


class NoiseInjection(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            return x + self.weight * noise
        return x


class StyledConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, w_dim: int, *, upsample: bool = False, use_noise: bool = True):
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.style = EqualLinear(w_dim, out_ch * 2, bias=True)
        nn.init.zeros_(self.style.bias)
        self.noise = NoiseInjection(out_ch) if use_noise else nn.Identity()
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.conv(x)
        style = self.style(w).view(w.size(0), 2, x.size(1), 1, 1)
        scale = style[:, 0] + 1.0
        shift = style[:, 1]
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = torch.rsqrt((x - mean).pow(2).mean(dim=[2, 3], keepdim=True) + 1e-8)
        x = (x - mean) * std
        x = x * scale + shift
        x = self.noise(x)
        return F.leaky_relu(x + self.bias, 0.2) * math.sqrt(2)


class ToRGB(nn.Module):
    def __init__(self, in_ch: int, w_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 3, 1)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        del w
        return self.conv(x) + self.bias


class SynthesisBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, *, is_first: bool, use_noise: bool):
        super().__init__()
        self.is_first = is_first
        self.conv0 = StyledConv(in_ch, out_ch, 3, w_dim, upsample=not is_first, use_noise=use_noise)
        self.conv1 = StyledConv(out_ch, out_ch, 3, w_dim, upsample=False, use_noise=use_noise)
        self.to_rgb = ToRGB(out_ch, w_dim)

    def forward(self, x: torch.Tensor, img: torch.Tensor | None, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv0(x, w)
        x = self.conv1(x, w)
        y = self.to_rgb(x, w)
        if img is not None:
            img = F.interpolate(img, scale_factor=2.0, mode="bilinear", align_corners=False)
            y = y + img
        return x, y


class Generator(nn.Module):
    """StyleGAN-like synthesis network: z -> mapping -> styled conv stack."""

    def __init__(self, cfg: GeneratorConfig):
        super().__init__()
        self.cfg = cfg
        self.z_dim = cfg.z_dim
        self.w_dim = cfg.w_dim

        mapping: list[nn.Module] = [PixelNorm()]
        for i in range(cfg.mapping_layers):
            in_dim = cfg.z_dim if i == 0 else cfg.w_dim
            mapping.append(
                EqualLinear(
                    in_dim,
                    cfg.w_dim,
                    lr_mul=cfg.mapping_lr_mul,
                    activation=True,
                )
            )
        self.mapping = nn.Sequential(*mapping)

        first_res = cfg.resolutions[0]
        first_ch = cfg.channels[first_res]
        self.input = nn.Parameter(torch.randn(1, first_ch, first_res, first_res))

        blocks: list[nn.Module] = []
        in_ch = first_ch
        for i, res in enumerate(cfg.resolutions):
            out_ch = cfg.channels[res]
            blocks.append(
                SynthesisBlock(
                    in_ch,
                    out_ch,
                    cfg.w_dim,
                    is_first=i == 0,
                    use_noise=cfg.use_noise,
                )
            )
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.mapping(z)
        x = self.input.repeat(z.size(0), 1, 1, 1)
        img = None
        for block in self.blocks:
            x, img = block(x, img, w)
        return torch.tanh(img)


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        *,
        downsample: bool = False,
        use_spectral_norm: bool = True,
        activate: bool = True,
    ):
        super().__init__()
        self.downsample = downsample
        padding = kernel_size // 2
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.conv = sn(conv) if use_spectral_norm else conv
        self.activate = activate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.activate:
            x = F.leaky_relu(x, 0.2) * math.sqrt(2)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, use_spectral_norm: bool):
        super().__init__()
        self.conv0 = ConvLayer(in_ch, in_ch, 3, use_spectral_norm=use_spectral_norm)
        self.conv1 = ConvLayer(in_ch, out_ch, 3, downsample=True, use_spectral_norm=use_spectral_norm)
        skip = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.skip = sn(skip) if use_spectral_norm else skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.conv0(x))
        s = F.avg_pool2d(self.skip(x), 2)
        return (h + s) / math.sqrt(2)


class MinibatchStd(nn.Module):
    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = min(self.group_size, b)
        if b % g != 0:
            g = b
        y = x.view(g, b // g, c, h, w)
        y = y - y.mean(dim=0, keepdim=True)
        y = (y.pow(2).mean(dim=0) + 1e-8).sqrt()
        y = y.mean(dim=[1, 2, 3], keepdim=True)
        y = y.repeat(g, 1, h, w)
        return torch.cat([x, y], dim=1)


class Discriminator(nn.Module):
    """StyleGAN2-like residual discriminator trained jointly with G."""

    def __init__(self, cfg: DiscriminatorConfig):
        super().__init__()
        self.cfg = cfg
        wrap = sn if cfg.use_spectral_norm else (lambda m: m)

        self.from_rgb = wrap(nn.Conv2d(3, cfg.channels[cfg.resolutions[0]], 1))
        blocks: list[nn.Module] = []
        for res_in, res_out in zip(cfg.resolutions, cfg.resolutions[1:]):
            blocks.append(
                DiscriminatorBlock(
                    cfg.channels[res_in],
                    cfg.channels[res_out],
                    use_spectral_norm=cfg.use_spectral_norm,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        last_res = cfg.resolutions[-1]
        last_ch = cfg.channels[last_res]
        self.minibatch_std = MinibatchStd(group_size=cfg.minibatch_std_group)
        self.final_conv = ConvLayer(last_ch + 1, last_ch, 3, use_spectral_norm=cfg.use_spectral_norm)
        self.final_linear = wrap(nn.Linear(last_ch * last_res * last_res, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.from_rgb(x), 0.2) * math.sqrt(2)
        h = self.blocks(h)
        h = self.minibatch_std(h)
        h = self.final_conv(h)
        return self.final_linear(h.flatten(1))


class EMA:
    """Exponential moving average of Generator weights."""

    def __init__(self, G: nn.Module, half_life: int = 10_000):
        self.shadow = copy.deepcopy(G).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.half_life = half_life

    @torch.no_grad()
    def update(self, G: nn.Module, batch_size: int) -> None:
        decay = 0.5 ** (batch_size / self.half_life)
        for sp, p in zip(self.shadow.parameters(), G.parameters()):
            sp.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        for sb, b in zip(self.shadow.buffers(), G.buffers()):
            sb.copy_(b)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(state)


STYLEGAN_1024_GENERATOR_CONFIG = GeneratorConfig(
    z_dim=512,
    resolutions=[4, 8, 16, 32, 64, 128, 256, 512, 1024],
    channels={4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 96, 512: 64, 1024: 32},
    w_dim=512,
    mapping_layers=4,
    mapping_lr_mul=0.01,
    use_noise=True,
)

STYLEGAN_1024_DISCRIMINATOR_CONFIG = DiscriminatorConfig(
    resolutions=[1024, 512, 256, 128, 64, 32, 16, 8, 4],
    channels={1024: 32, 512: 64, 256: 96, 128: 128, 64: 256, 32: 512, 16: 512, 8: 512, 4: 512},
    use_spectral_norm=True,
    minibatch_std_group=4,
)


def build_stylegan_1024_generator() -> Generator:
    return Generator(STYLEGAN_1024_GENERATOR_CONFIG)


def build_stylegan_1024_discriminator() -> Discriminator:
    return Discriminator(STYLEGAN_1024_DISCRIMINATOR_CONFIG)


# Backward-compatible names used by older scripts/checkpoints in this package.
BASELINE_256_GENERATOR_CONFIG = STYLEGAN_1024_GENERATOR_CONFIG
BASELINE_256_DISCRIMINATOR_CONFIG = STYLEGAN_1024_DISCRIMINATOR_CONFIG


def build_baseline_256_generator() -> Generator:
    return build_stylegan_1024_generator()


def build_baseline_256_discriminator() -> Discriminator:
    return build_stylegan_1024_discriminator()


if __name__ == "__main__":
    G = build_stylegan_1024_generator()
    D = build_stylegan_1024_discriminator()
    n_g = sum(p.numel() for p in G.parameters())
    n_d = sum(p.numel() for p in D.parameters())
    print(f"Generator: {n_g/1e6:.2f}M params")
    print(f"Discriminator: {n_d/1e6:.2f}M params")
    z = torch.randn(1, G.z_dim)
    fake = G(z)
    score = D(fake)
    print(f"G(z) shape: {tuple(fake.shape)}, range [{fake.min():.3f}, {fake.max():.3f}]")
    print(f"D(fake) shape: {tuple(score.shape)}")
