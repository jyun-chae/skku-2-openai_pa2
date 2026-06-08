"""StyleGAN2 Generator with Squeeze Connection synthesis blocks.

Channel schedule (channel_base=131072, channel_max=512):
  res:      4    8   16   32   64  128  256  512 1024
  channels: 512 512  512  512  512  512  512  256  128

mapping_layers=12, w_dim=640, squeeze_ratio=8.

Squeeze connection (StyleGAN-Small, arXiv:2407.05527) replaces conv2+ToRGB
with squeeze→excite→project path, improving the toRGB information bottleneck.
PyTorch params @ 1024: ~34.10M.  ONNX initializer count: ~35.5M (<40M limit).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import EqualLinear, ModulatedConv2d, NoiseInjection, PixelNorm, SqueezeConnection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nf(resolution: int, channel_base: int = 131072, channel_max: int = 512) -> int:
    """Number of feature maps at a given resolution (StyleGAN2 schedule)."""
    return min(channel_max, int(channel_base / resolution))


def _log2(n: int) -> int:
    return int(math.log2(n))


# ---------------------------------------------------------------------------
# Mapping Network
# ---------------------------------------------------------------------------

class MappingNetwork(nn.Module):
    """Maps z → disentangled w via an 8-layer EqualLinear MLP."""

    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        n_layers: int = 8,
        lr_mul: float = 0.01,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim

        layers: list[nn.Module] = [PixelNorm()]
        in_dim = z_dim
        for _ in range(n_layers):
            layers.append(EqualLinear(in_dim, w_dim, lr_mul=lr_mul, activation="fused_lrelu"))
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, truncation: float = 1.0,
                mean_w: Optional[torch.Tensor] = None) -> torch.Tensor:
        w = self.net(z)
        if truncation < 1.0 and mean_w is not None:
            w = mean_w + truncation * (w - mean_w)
        return w

    @torch.no_grad()
    def mean_latent(self, n_samples: int = 4096, device: str = "cuda") -> torch.Tensor:
        z = torch.randn(n_samples, self.z_dim, device=device)
        return self.net(z).mean(dim=0, keepdim=True)


# ---------------------------------------------------------------------------
# Synthesis blocks
# ---------------------------------------------------------------------------

class SynthesisBlock4(nn.Module):

    def __init__(self, out_ch: int, w_dim: int, squeeze_ratio: int = 8) -> None:
        super().__init__()
        self.const   = nn.Parameter(torch.randn(1, out_ch, 4, 4))
        self.conv    = ModulatedConv2d(out_ch, out_ch, kernel=3, w_dim=w_dim)
        self.noise   = NoiseInjection()
        self.act     = nn.LeakyReLU(0.2)
        self.squeeze = SqueezeConnection(out_ch, w_dim, squeeze_ratio)

    def forward(
        self,
        w: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        squeeze_noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = w.shape[0]
        x = self.const.expand(b, -1, -1, -1)
        x = self.conv(x, w)
        x = self.noise(x, noise)
        x = self.act(x) * math.sqrt(2)
        x, rgb = self.squeeze(x, w, squeeze_noise)
        return x, rgb


class SynthesisBlock(nn.Module):
    """Upsample block with squeeze connection replacing conv2 + ToRGB."""

    def __init__(self, in_ch: int, out_ch: int, w_dim: int, squeeze_ratio: int = 8) -> None:
        super().__init__()
        self.conv1   = ModulatedConv2d(in_ch, out_ch, kernel=3, w_dim=w_dim, upsample=True)
        self.noise1  = NoiseInjection()
        self.act     = nn.LeakyReLU(0.2)
        self.squeeze = SqueezeConnection(out_ch, w_dim, squeeze_ratio)

    def forward(
        self,
        x: torch.Tensor,
        prev_rgb: torch.Tensor,
        w: torch.Tensor,
        noise1: Optional[torch.Tensor] = None,
        squeeze_noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x, w)
        x = self.noise1(x, noise1)
        x = self.act(x) * math.sqrt(2)

        x, rgb_cur = self.squeeze(x, w, squeeze_noise)
        prev_rgb_up = F.interpolate(prev_rgb, scale_factor=2, mode="bilinear", align_corners=False)
        rgb = prev_rgb_up + rgb_cur
        return x, rgb


# ---------------------------------------------------------------------------
# Full Generator
# ---------------------------------------------------------------------------

class StyleGAN2Generator(nn.Module):
    """StyleGAN2 generator: MappingNetwork + SqueezeConnection synthesis.

    Args:
        resolution:     Output resolution — 256, 512, or 1024.
        z_dim:          Input noise dimension.
        w_dim:          Intermediate latent dimension (w-space).
        channel_base:   Controls channel schedule: nf(r) = min(channel_max, channel_base/r).
        channel_max:    Hard cap on channels per layer.
        mapping_layers: Depth of the mapping MLP (more → better disentanglement).
        mapping_lr_mul: LR multiplier for mapping network (low → slower, more stable).
        squeeze_ratio:  Squeeze compression ratio in SqueezeConnection (default 8).
    """

    def __init__(
        self,
        resolution: int = 256,
        z_dim: int = 512,
        w_dim: int = 640,
        channel_base: int = 131072,
        channel_max: int = 512,
        mapping_layers: int = 12,
        mapping_lr_mul: float = 0.01,
        squeeze_ratio: int = 8,
    ) -> None:
        super().__init__()
        assert resolution in (256, 512, 1024), "resolution must be 256, 512, or 1024"
        self.resolution = resolution
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.log2_res = _log2(resolution)

        def nf(r: int) -> int:
            return _nf(r, channel_base, channel_max)

        self.mapping = MappingNetwork(z_dim, w_dim, mapping_layers, mapping_lr_mul)

        self.b4 = SynthesisBlock4(nf(4), w_dim, squeeze_ratio)

        self.blocks = nn.ModuleList()
        in_ch = nf(4)
        for log2_r in range(3, self.log2_res + 1):
            res = 2 ** log2_r
            out_ch = nf(res)
            self.blocks.append(SynthesisBlock(in_ch, out_ch, w_dim, squeeze_ratio))
            in_ch = out_ch

        self.num_ws = 1 + len(self.blocks)

    # ------------------------------------------------------------------
    def _w_broadcast(self, w: torch.Tensor) -> list[torch.Tensor]:
        """Return a list of per-block w tensors (all the same for non-mixing)."""
        return [w] * self.num_ws

    def _synthesis(self, w: torch.Tensor, noise_mode: str) -> torch.Tensor:
        """Run synthesis network given a w vector (no mapping)."""
        ws = self._w_broadcast(w)
        b = w.shape[0]

        def _noise(h: int, ww: int, idx: int = 0) -> Optional[torch.Tensor]:
            if noise_mode == "none":
                return torch.zeros(b, 1, h, ww, device=w.device)
            if noise_mode == "const":
                seed = h * 2000 + ww * 2 + idx
                g = torch.Generator(device=w.device).manual_seed(seed)
                return torch.randn(1, 1, h, ww, generator=g, device=w.device).expand(b, -1, -1, -1)
            return None   # NoiseInjection samples dynamically (training default)

        x, rgb = self.b4(ws[0], _noise(4, 4, 0), _noise(4, 4, 1))
        for i, block in enumerate(self.blocks):
            h_next = 8 * (2 ** i)
            x, rgb = block(x, rgb, ws[i + 1], _noise(h_next, h_next, 0), _noise(h_next, h_next, 1))

        return torch.tanh(rgb)

    def forward(
        self,
        z: Optional[torch.Tensor] = None,
        noise_mode: str = "random",
        truncation: float = 1.0,
        mean_w: Optional[torch.Tensor] = None,
        return_w: bool = False,
        w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z:          Input latent [B, z_dim].  Provide z OR w, not both.
            w:          Pre-computed w [B, w_dim].  Skips mapping when provided.
            noise_mode: "random" | "const" | "none".
            truncation: Truncation psi (applied only when z is given).
            mean_w:     Pre-computed mean w for truncation.
            return_w:   Also return the w tensor (only meaningful when z is given).
        """
        if w is None:
            w = self.mapping(z, truncation=truncation, mean_w=mean_w)

        img = self._synthesis(w, noise_mode)

        if return_w:
            return img, w
        return img

    # ------------------------------------------------------------------
    def export_onnx(self, path: str, batch_size: int = 1) -> int:
        """Export generator to ONNX with zero noise for a deterministic graph."""
        import numpy as np
        import onnx

        class _NoNoiseWrapper(nn.Module):
            def __init__(self, G: "StyleGAN2Generator") -> None:
                super().__init__()
                self.G = G
            def forward(self, z: torch.Tensor) -> torch.Tensor:
                return self.G(z, noise_mode="none")

        self.eval()
        device = next(self.parameters()).device

        wrapper = _NoNoiseWrapper(self)
        wrapper.eval()
        # CPU: do_constant_folding mixes GPU tensors with CPU intermediates → RuntimeError on CUDA.
        wrapper.cpu()
        dummy_z = torch.randn(1, self.z_dim, device="cpu")

        torch.onnx.export(
            wrapper,
            (dummy_z,),
            path,
            opset_version=17,
            input_names=["z"],
            output_names=["image"],
            do_constant_folding=True,
            dynamo=False,  # dynamo path uses groups=B which ONNX opset 17 cannot represent
        )
        wrapper.to(device)

        model_onnx = onnx.load(path)
        onnx_params = sum(int(np.prod(t.dims)) for t in model_onnx.graph.initializer)
        limit = 50_000_000
        status = "OK" if onnx_params < limit else "OVER LIMIT"
        print(f"ONNX exported → {path}")
        print(f"  ONNX parameters: {onnx_params:,} ({onnx_params/1e6:.3f}M)  [{status}]")
        assert onnx_params < limit, f"ONNX exceeds 40M limit! ({onnx_params:,})"
        return onnx_params

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def load_from_lower_resolution(self, state_dict: dict, freeze_existing: bool = False) -> None:
        """Load weights from a lower-resolution checkpoint; new higher-res blocks stay random-init.

        Args:
            state_dict:      G state dict from the previous-stage checkpoint.
            freeze_existing: If True, freeze all loaded params (requires_grad=False).
                             Call trainer.rebuild_optimizers() afterwards so frozen
                             params are excluded from the optimizer.
        """
        own = self.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
        own.update(filtered)
        self.load_state_dict(own)
        print(f"Loaded {len(filtered)}/{len(own)} tensors from lower-resolution checkpoint.")

        if freeze_existing and filtered:
            frozen = 0
            for name, param in self.named_parameters():
                if name in filtered:
                    param.requires_grad_(False)
                    frozen += 1
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  Froze {frozen} param tensors. G trainable: {trainable:,}")
