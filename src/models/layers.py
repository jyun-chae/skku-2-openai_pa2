"""StyleGAN2 core layers with equalized learning rate."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNorm(nn.Module):
    """Feature-wise L2 normalization applied to z before the mapping network.

    Prevents z magnitude from dominating the first mapping layer.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate.

    Weights are stored unscaled and multiplied by He-init scale at runtime.
    This keeps all parameters at a similar magnitude so Adam's adaptive LR
    works uniformly across layers regardless of fan-in size.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        bias_init: float = 0.0,
        lr_mul: float = 1.0,
        activation: str | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init)) if bias else None
        self.scale = (1.0 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias * self.lr_mul if self.bias is not None else None
        out = F.linear(x, self.weight * self.scale, bias)
        if self.activation == "fused_lrelu":
            out = F.leaky_relu(out, 0.2) * math.sqrt(2)  # sqrt(2): restores ~unit variance after LReLU
        return out


class EqualConv2d(nn.Module):
    """Conv2d with equalized learning rate (used in discriminator and SqueezeConnection)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel, kernel))
        self.scale = 1.0 / math.sqrt(in_ch * kernel**2)
        self.stride = stride
        self.padding = padding
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, self.weight * self.scale, self.bias,
            stride=self.stride, padding=self.padding,
        )


class ModulatedConv2d(nn.Module):
    """StyleGAN2 per-sample weight modulation + demodulation conv.

    For each sample, the affine-transformed w modulates the shared weights,
    then demodulation normalizes out the accumulated variance.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        w_dim: int,
        demodulate: bool = True,
        upsample: bool = False,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        self.demodulate = demodulate
        self.upsample = upsample
        self.padding = kernel // 2
        self.scale = 1.0 / math.sqrt(in_ch * kernel**2)

        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel, kernel))
        self.affine = EqualLinear(w_dim, in_ch, bias_init=1.0)  # bias_init=1: identity modulation at init

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        b, c, h, width = x.shape

        style = self.affine(w).view(b, 1, c, 1, 1)
        weight = self.weight * self.scale * style

        if self.demodulate:
            # fp32 required: 512ch × 3×3 = 4608 squared terms; values ~10 → 460k > fp16 max (65504).
            # rsqrt(inf)=0 silently zeros every weight — no NaN raised, training dies without warning.
            w32 = weight.float()
            d = (w32.pow(2).sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
            weight = (w32 * d.view(b, self.out_ch, 1, 1, 1)).to(x.dtype)

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            h, width = x.shape[2], x.shape[3]

        # groups=b fuses per-sample weights into a single grouped conv call
        x = x.reshape(1, b * c, h, width)
        weight = weight.reshape(b * self.out_ch, c, self.kernel, self.kernel)
        x = F.conv2d(x, weight, padding=self.padding, groups=b)
        return x.view(b, self.out_ch, *x.shape[2:])


class NoiseInjection(nn.Module):
    """Adds spatially-varying stochastic noise scaled by a learned per-layer weight.

    The weight starts at zero so noise has no effect at initialization,
    allowing the network to opt in to noise as training progresses.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            b, _, h, w = x.shape
            noise = x.new_empty(b, 1, h, w).normal_()
        return x + self.weight * noise


class ToRGB(nn.Module):
    """Projects feature maps to 3-channel RGB via a 1×1 modulated conv (no demodulation)."""

    def __init__(self, in_ch: int, w_dim: int) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, 3, kernel=1, w_dim=w_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return self.conv(x, w) + self.bias


class SqueezeConnection(nn.Module):
    """Replaces conv2 + ToRGB per synthesis block (StyleGAN-Small, arXiv:2407.05527).

    Standard StyleGAN2 projects c channels directly to 3-channel RGB, losing spatial
    information through the narrow bottleneck. SqueezeConnection reduces that bottleneck
    by routing RGB generation through c/8 squeezed features:

        squeeze(c → c/8) → noise → to_rgb
        excite(c/8 → c) → project(cat(x, excite) → c)

    The excite+project path merges the squeezed spatial context back into the feature
    stream for the next block, replacing the functionality of conv2.
    """

    def __init__(self, in_ch: int, w_dim: int, ratio: int = 8) -> None:
        super().__init__()
        s_ch = max(in_ch // ratio, 8)  # floor at 8: prevents degenerate 1–2 ch squeeze at low resolutions
        self.squeeze = EqualConv2d(in_ch, s_ch, kernel=3, padding=1)
        self.noise   = NoiseInjection()   # restores stochasticity removed by eliminating conv2
        self.to_rgb  = ToRGB(s_ch, w_dim)
        self.excite  = EqualConv2d(s_ch, in_ch, kernel=3, padding=1)
        self.proj    = EqualConv2d(in_ch * 2, in_ch, kernel=1)
        self.act     = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_s   = self.act(self.squeeze(x)) * math.sqrt(2)
        x_s   = self.noise(x_s, noise)
        rgb   = self.to_rgb(x_s, w)
        x_e   = self.act(self.excite(x_s)) * math.sqrt(2)
        x_out = self.proj(torch.cat([x, x_e], dim=1))
        return x_out, rgb
