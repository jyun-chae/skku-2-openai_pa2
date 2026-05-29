"""StyleGAN2 core building blocks with equalized learning rate.

Equalized LR (He init at runtime):
  Weights are stored as N(0,1). The per-layer scale factor c = 1/sqrt(fan_in)
  is applied at *forward time* instead of at init. This keeps weight magnitudes
  uniform at initialization and ensures all parameters receive equal effective
  learning rates regardless of their fan-in.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNorm(nn.Module):
    """Pixel-wise L2 normalization on the channel dimension (used in mapping network).

    Prevents the latent z from growing unbounded before the mapping MLP.
    StyleGAN2 applies this only to the mapping network input, not to synthesis.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    """Fully-connected layer with equalized learning rate.

    Args:
        lr_mul: Multiplier that lets the mapping network use a lower effective LR
                (typically 0.01) than the rest of the model. The weight is divided
                by lr_mul at init and multiplied back at forward time, so the
                gradient flowing to the stored weight is lr_mul × smaller.
        bias_init: Non-zero default (1.0) is used for modulation affines so that
                   the initial style is close to the identity transform.
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
            # sqrt(2) restores unit variance after LeakyReLU (negative slope 0.2
            # kills ~10% of variance; sqrt(2) over-compensates slightly but matches
            # the official StyleGAN2 implementation).
            out = F.leaky_relu(out, 0.2) * math.sqrt(2)
        return out


class EqualConv2d(nn.Module):
    """Conv2d with equalized learning rate (used in the discriminator)."""

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
    """StyleGAN2 modulated convolution with weight demodulation.

    Forward pass:
      1. style  = affine(w)                      # per-sample channel scales
      2. weight = base_weight * scale * style    # modulate
      3. weight = weight / norm(weight)           # demodulate (unit std per output)
      4. output = grouped_conv(x, weight)        # one group per sample

    The group-conv trick avoids a loop over the batch: we reshape x to
    [1, B*C, H, W] and weight to [B*out, C, k, k], then run a single conv
    with groups=B so each sample uses its own (modulated) filters.

    Upsample mode: bilinear ×2 is applied to x BEFORE the convolution.
    This is equivalent to transposed conv + blur and avoids the tricky
    kernel flip that transposed conv requires for correct equalized LR.
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
        # bias_init=1.0 → initial style is identity-like (no modulation effect)
        self.affine = EqualLinear(w_dim, in_ch, bias_init=1.0)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        b, c, h, width = x.shape

        style = self.affine(w).view(b, 1, c, 1, 1)       # [B, 1, in, 1, 1]
        weight = self.weight * self.scale * style          # [B, out, in, k, k]

        if self.demodulate:
            # Demodulation normalizes each output filter to unit std so that
            # the modulation effect is purely directional, not scale-changing.
            #
            # MUST run in fp32: fp16 accumulates (in_ch * k²) squares.
            # With 512 channels and 3×3 kernel that is 4608 terms; values
            # around 10 give 10² × 4608 ≈ 460k, well above fp16 max (65504).
            # rsqrt(inf) = 0 silently zeros every weight without NaN.
            w32 = weight.float()
            d = (w32.pow(2).sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
            weight = (w32 * d.view(b, self.out_ch, 1, 1, 1)).to(x.dtype)

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            h, width = x.shape[2], x.shape[3]

        # Group-conv: treat each sample as an independent group
        x = x.reshape(1, b * c, h, width)
        weight = weight.reshape(b * self.out_ch, c, self.kernel, self.kernel)
        x = F.conv2d(x, weight, padding=self.padding, groups=b)
        return x.view(b, self.out_ch, *x.shape[2:])


class NoiseInjection(nn.Module):
    """Add spatially-correlated Gaussian noise scaled by a learnable scalar.

    The network uses noise to control stochastic per-pixel detail (hair strands,
    skin pores) without affecting global structure — the small learned weight
    keeps noise contribution negligible early in training.
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
    """1×1 modulated conv that projects feature maps to 3-channel RGB.

    demodulate=False: the modulation acts as a per-sample color calibration;
    demodulation would cancel that effect, so it is intentionally disabled.
    The resulting output feeds into the skip-RGB accumulation (sum across blocks).
    """

    def __init__(self, in_ch: int, w_dim: int) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, 3, kernel=1, w_dim=w_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return self.conv(x, w) + self.bias
