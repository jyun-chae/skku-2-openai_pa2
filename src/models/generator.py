"""StyleGAN2 Generator: Mapping Network + Synthesis Network.

Progressive training flow:
  256  → train from scratch
  512  → load 256 weights, random-init new block
  1024 → load 512 weights, random-init new block

Channel schedule (channel_base=65536, channel_max=512):
  res:      4    8   16   32   64  128  256  512 1024
  channels: 512 512  512  512  512  512  256  128   64
mapping_layers=8 (standard StyleGAN2), w_dim=640.
PyTorch params @ 1024: ~38.22M.  ONNX initializer count: ~39.6M (<40M limit).
Note: ONNX count exceeds PyTorch count by ~1.4M due to constant-folded
intermediates baked into the graph during TorchScript export.

Design notes:
  - Mapping network uses lr_mul=0.01 so it trains ~100× slower than synthesis.
    This prevents the mapping from collapsing to a trivial solution early on.
  - _synthesis is split from forward so PL regularization can supply w directly
    as a leaf node (differentiating synthesis only, not the mapping network).
  - Skip-RGB architecture: each block adds a bilinearly upsampled RGB contribution;
    the final image is the accumulated sum passed through tanh.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import EqualLinear, ModulatedConv2d, NoiseInjection, PixelNorm, ToRGB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nf(resolution: int, channel_base: int = 32768, channel_max: int = 512) -> int:
    """Number of feature maps at a given resolution (StyleGAN2 schedule)."""
    return min(channel_max, int(channel_base / resolution))


def _log2(n: int) -> int:
    return int(math.log2(n))


# ---------------------------------------------------------------------------
# Mapping Network
# ---------------------------------------------------------------------------

class MappingNetwork(nn.Module):
    """Maps latent z → disentangled latent w.

    Follows StyleGAN2: 8-layer EqualLinear MLP with PixelNorm on input and
    a reduced learning rate multiplier (lr_mul=0.01) for stability.
    """

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
    """Initial 4×4 constant block."""

    def __init__(self, out_ch: int, w_dim: int) -> None:
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, out_ch, 4, 4))
        self.conv = ModulatedConv2d(out_ch, out_ch, kernel=3, w_dim=w_dim)
        self.noise = NoiseInjection()
        self.act = nn.LeakyReLU(0.2)
        self.to_rgb = ToRGB(out_ch, w_dim)

    def forward(self, w: torch.Tensor,
                noise: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        b = w.shape[0]
        x = self.const.expand(b, -1, -1, -1)
        x = self.conv(x, w)
        x = self.noise(x, noise)
        x = self.act(x) * math.sqrt(2)
        rgb = self.to_rgb(x, w)
        return x, rgb


class SynthesisBlock(nn.Module):
    """Upsampling synthesis block (resolution doubles)."""

    def __init__(self, in_ch: int, out_ch: int, w_dim: int) -> None:
        super().__init__()
        # First conv: upsample ×2 inside ModulatedConv2d
        self.conv1 = ModulatedConv2d(in_ch, out_ch, kernel=3, w_dim=w_dim, upsample=True)
        self.noise1 = NoiseInjection()
        self.conv2 = ModulatedConv2d(out_ch, out_ch, kernel=3, w_dim=w_dim)
        self.noise2 = NoiseInjection()
        self.act = nn.LeakyReLU(0.2)
        self.to_rgb = ToRGB(out_ch, w_dim)

    def forward(
        self,
        x: torch.Tensor,
        prev_rgb: torch.Tensor,
        w: torch.Tensor,
        noise1: Optional[torch.Tensor] = None,
        noise2: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x, w)
        x = self.noise1(x, noise1)
        x = self.act(x) * math.sqrt(2)

        x = self.conv2(x, w)
        x = self.noise2(x, noise2)
        x = self.act(x) * math.sqrt(2)

        # Skip RGB: upsample previous, add this block's contribution
        prev_rgb_up = F.interpolate(prev_rgb, scale_factor=2, mode="bilinear", align_corners=False)
        rgb = prev_rgb_up + self.to_rgb(x, w)
        return x, rgb


# ---------------------------------------------------------------------------
# Full Generator
# ---------------------------------------------------------------------------

class StyleGAN2Generator(nn.Module):
    """StyleGAN2 Generator supporting resolutions 256 / 512 / 1024.

    Args:
        resolution:    Target output resolution (256, 512, or 1024).
        z_dim:         Dimension of input noise z.
        w_dim:         Dimension of intermediate latent w.
        channel_base:  Base channel multiplier (32768 gives ~30M params for 1024).
        channel_max:   Maximum channels per layer (capped at 512).
        mapping_layers: Number of layers in the mapping network.
        mapping_lr_mul: Learning-rate multiplier for mapping network.
    """

    def __init__(
        self,
        resolution: int = 256,
        z_dim: int = 512,
        w_dim: int = 640,
        channel_base: int = 65536,
        channel_max: int = 512,
        mapping_layers: int = 8,
        mapping_lr_mul: float = 0.01,
    ) -> None:
        super().__init__()
        assert resolution in (256, 512, 1024), "resolution must be 256, 512, or 1024"
        self.resolution = resolution
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.log2_res = _log2(resolution)          # 8, 9, or 10

        def nf(r: int) -> int:
            return _nf(r, channel_base, channel_max)

        self.mapping = MappingNetwork(z_dim, w_dim, mapping_layers, mapping_lr_mul)

        # 4×4 initial block
        self.b4 = SynthesisBlock4(nf(4), w_dim)

        # Upsampling blocks: 8, 16, 32, ..., resolution
        self.blocks = nn.ModuleList()
        in_ch = nf(4)
        for log2_r in range(3, self.log2_res + 1):   # 3→8, 4→16, ..., log2_res→resolution
            res = 2 ** log2_r
            out_ch = nf(res)
            self.blocks.append(SynthesisBlock(in_ch, out_ch, w_dim))
            in_ch = out_ch

        # Total synthesis layers = 1 (b4) + len(blocks)
        self.num_ws = 1 + len(self.blocks)   # one w per layer (broadcast)

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
                # Unique seed per (resolution, conv-index) so that:
                #   - n1 and n2 within the same block have different patterns
                #   - patterns are reproducible across calls (same seed = same output)
                seed = h * 2000 + ww * 2 + idx
                g = torch.Generator(device=w.device).manual_seed(seed)
                return torch.randn(1, 1, h, ww, generator=g, device=w.device).expand(b, -1, -1, -1)
            return None   # NoiseInjection samples dynamically (training default)

        x, rgb = self.b4(ws[0])
        for i, block in enumerate(self.blocks):
            h_next = 8 * (2 ** i)
            n1 = _noise(h_next, h_next, idx=0)
            n2 = _noise(h_next, h_next, idx=1)
            x, rgb = block(x, rgb, ws[i + 1], n1, n2)

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
        """Export generator to ONNX with zero noise for a deterministic graph.

        Uses a thin wrapper so `noise_mode` is baked into the traced graph
        rather than passed as a dynamic Python kwarg (which ONNX cannot represent).

        Returns:
            Total ONNX parameter count (sum of initializer elements).
        """
        import numpy as np
        import onnx

        class _NoNoiseWrapper(nn.Module):
            """Wraps generator with fixed noise_mode='none' for ONNX tracing."""
            def __init__(self, G: "StyleGAN2Generator") -> None:
                super().__init__()
                self.G = G
            def forward(self, z: torch.Tensor) -> torch.Tensor:
                return self.G(z, noise_mode="none")

        self.eval()
        device = next(self.parameters()).device

        wrapper = _NoNoiseWrapper(self)
        wrapper.eval()
        # Move to CPU for export: do_constant_folding=True mixes GPU model
        # tensors with CPU intermediates during the JIT constant-folding pass,
        # causing "Expected all tensors on same device" RuntimeError on CUDA.
        # CPU export avoids this; we restore the original device afterwards.
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
            dynamo=False,
        )
        wrapper.to(device)

        # Count and verify ONNX parameters
        model_onnx = onnx.load(path)
        onnx_params = sum(int(np.prod(t.dims)) for t in model_onnx.graph.initializer)
        limit = 40_000_000
        status = "OK" if onnx_params < limit else "OVER LIMIT"
        print(f"ONNX exported → {path}")
        print(f"  ONNX parameters: {onnx_params:,} ({onnx_params/1e6:.3f}M)  [{status}]")
        assert onnx_params < limit, f"ONNX exceeds 40M limit! ({onnx_params:,})"
        return onnx_params

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def load_from_lower_resolution(self, state_dict: dict) -> None:
        """Load weights from a lower-resolution checkpoint (new blocks stay random)."""
        own = self.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
        own.update(filtered)
        self.load_state_dict(own)
        loaded = len(filtered)
        total = len(own)
        print(f"Loaded {loaded}/{total} tensors from lower-resolution checkpoint.")
