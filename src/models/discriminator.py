"""StyleGAN2 Discriminator with residual blocks, minibatch stddev, and R1 regularization."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import EqualConv2d, EqualLinear


def _nf(resolution: int, channel_base: int = 131072, channel_max: int = 512) -> int:
    return min(channel_max, int(channel_base / resolution))


# ---------------------------------------------------------------------------
# Discriminator blocks
# ---------------------------------------------------------------------------

class MinibatchStddev(nn.Module):
    """Appends per-group feature std as an extra channel to help D detect mode collapse.

    Computing std within groups (not the full batch) prevents D from memorising
    batch-level statistics at small batch sizes.
    """

    def __init__(self, group_size: int = 4, num_features: int = 1) -> None:
        super().__init__()
        self.group_size = group_size
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = min(self.group_size, b)
        f = self.num_features
        y = x.reshape(g, -1, f, c // f, h, w).float()
        y = y - y.mean(dim=0, keepdim=True)
        y = y.pow(2).mean(dim=0)
        y = (y + 1e-8).sqrt()
        y = y.mean(dim=[2, 3, 4], keepdim=True)
        y = y.mean(dim=2)
        y = y.repeat(g, 1, h, w)
        return torch.cat([x, y.to(x.dtype)], dim=1)


class DiscBlock(nn.Module):
    """Residual downscale block: conv×2 with avg-pool skip connection."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = EqualConv2d(in_ch, in_ch, kernel=3, padding=1)
        self.conv2 = EqualConv2d(in_ch, out_ch, kernel=3, padding=1)
        self.skip  = EqualConv2d(in_ch, out_ch, kernel=1, bias=False)
        self.act   = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(F.avg_pool2d(x, 2))

        out = self.act(self.conv1(x)) * math.sqrt(2)
        out = F.avg_pool2d(out, 2)
        out = self.act(self.conv2(out)) * math.sqrt(2)

        # /sqrt(2): normalises variance through the residual addition
        return (out + residual) / math.sqrt(2)


class FromRGB(nn.Module):
    """Projects 3-channel RGB input into the discriminator feature space."""

    def __init__(self, out_ch: int) -> None:
        super().__init__()
        self.conv = EqualConv2d(3, out_ch, kernel=1)
        self.act  = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x)) * math.sqrt(2)


# ---------------------------------------------------------------------------
# Full Discriminator
# ---------------------------------------------------------------------------

class StyleGAN2Discriminator(nn.Module):
    """StyleGAN2 discriminator for a given resolution.

    Args:
        resolution:   Input image resolution (256, 512, or 1024).
        channel_base: Channel schedule base (must match the generator).
        channel_max:  Maximum channels per layer.
        mbstd_group:  Minibatch stddev group size.
    """

    def __init__(
        self,
        resolution: int = 256,
        channel_base: int = 131072,
        channel_max: int = 512,
        mbstd_group: int = 4,
    ) -> None:
        super().__init__()
        assert resolution in (256, 512, 1024)
        self.resolution = resolution
        log2_res = int(math.log2(resolution))

        def nf(r: int) -> int:
            return _nf(r, channel_base, channel_max)

        self.from_rgb = FromRGB(nf(resolution))

        blocks = []
        in_ch = nf(resolution)
        for log2_r in range(log2_res, 2, -1):
            res = 2 ** log2_r
            out_ch = nf(res // 2)
            blocks.append(DiscBlock(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        self.mbstd  = MinibatchStddev(mbstd_group)
        self.conv_4 = EqualConv2d(in_ch + 1, in_ch, kernel=3, padding=1)
        self.act    = nn.LeakyReLU(0.2)
        self.fc     = EqualLinear(in_ch * 4 * 4, in_ch)
        self.out    = EqualLinear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.from_rgb(x)
        for block in self.blocks:
            feat = block(feat)

        feat = self.mbstd(feat)
        feat = self.act(self.conv_4(feat)) * math.sqrt(2)
        feat = feat.flatten(1)
        feat = self.act(self.fc(feat)) * math.sqrt(2)
        return self.out(feat)

    def load_from_lower_resolution(self, state_dict: dict) -> None:
        """Load D weights from a lower-res checkpoint.

        from_rgb and blocks[0] must match the new (larger) resolution, so they
        stay random-init. All other blocks shift up by one index to make room.
        """
        own = self.state_dict()
        new_state: dict = {}

        for k, v in state_dict.items():
            if k.startswith('from_rgb.') or k.startswith('blocks.0.'):
                continue  # these match the new resolution — keep random init
            elif k.startswith('blocks.'):
                parts = k.split('.')
                new_key = f'blocks.{int(parts[1]) + 1}.' + '.'.join(parts[2:])
                if new_key in own and own[new_key].shape == v.shape:
                    new_state[new_key] = v
            else:
                if k in own and own[k].shape == v.shape:
                    new_state[k] = v

        own.update(new_state)
        self.load_state_dict(own)
        print(f'D loaded {len(new_state)}/{len(own)} tensors '
              f'(new from_rgb + blocks.0 are random-init).')
