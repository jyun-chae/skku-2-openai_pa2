"""Export ckpt_1024_0020000.pth → ONNX.

Follows the submission contract defined in export_onnx_baseline.py:
    input  z      shape (B, 512), dtype float32, batch dimension dynamic
    output image  shape (B, 3, 1024, 1024), dtype float32, range [-1, 1]

Architecture note
-----------------
The checkpoint was trained with standard StyleGAN2 (conv2 + ToRGB per block,
no SqueezeConnection).  The legacy classes below match that layout exactly;
existing source files are not modified.

ModulatedConv2d compatibility
-----------------------------
The original forward uses groups=batch_size which ONNX cannot export when the
batch axis is dynamic.  _onnx_modconv_forward replaces it with an explicit
kernel-position loop (9 bmm calls for k=3) that is fully ONNX-traceable with
symbolic batch dimensions.
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "src")
from models.layers import EqualLinear, ModulatedConv2d, NoiseInjection, PixelNorm, ToRGB

TARGET_RESOLUTION = 1024


# ---------------------------------------------------------------------------
# Channel schedule
# ---------------------------------------------------------------------------

def _nf(res: int, channel_base: int = 65536, channel_max: int = 512) -> int:
    return min(channel_max, int(channel_base / res))


# ---------------------------------------------------------------------------
# ONNX-compatible ModulatedConv2d (kernel-position loop, no groups=b)
# ---------------------------------------------------------------------------

def _onnx_modconv_forward(self: ModulatedConv2d, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    c  = x.shape[1]
    h  = x.shape[2]
    wd = x.shape[3]

    style  = self.affine(w).view(-1, 1, c, 1, 1)
    weight = self.weight * self.scale * style          # (b, out_ch, c, k, k)

    if self.demodulate:
        w32    = weight.float()
        d      = (w32.pow(2).sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
        weight = (w32 * d[:, :, None, None, None]).to(x.dtype)

    if self.upsample:
        x  = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        h  = x.shape[2]
        wd = x.shape[3]

    k, p   = self.kernel, self.padding
    x_pad  = F.pad(x, (p, p, p, p)) if p > 0 else x

    out: torch.Tensor | None = None
    for di in range(k):
        for dj in range(k):
            w_pos = weight[:, :, :, di, dj]               # (b, out_ch, c)
            x_sl  = x_pad[:, :, di:di + h, dj:dj + wd]   # (b, c, h, w)
            x_fl  = x_sl.reshape(-1, c, h * wd)           # (b, c, h*w) — -1 symbolic-safe
            part  = torch.bmm(w_pos, x_fl)                # (b, out_ch, h*w)
            out   = part if out is None else out + part

    return out.view(-1, self.out_ch, h, wd)


def _patch_modulated_convs(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, ModulatedConv2d):
            m.forward = types.MethodType(_onnx_modconv_forward, m)


# ---------------------------------------------------------------------------
# Mapping network
# ---------------------------------------------------------------------------

class _MappingNetwork(nn.Module):
    def __init__(self, z_dim: int, w_dim: int, n_layers: int, lr_mul: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [PixelNorm()]
        in_dim = z_dim
        for _ in range(n_layers):
            layers.append(EqualLinear(in_dim, w_dim, lr_mul=lr_mul, activation="fused_lrelu"))
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Legacy synthesis blocks
# ---------------------------------------------------------------------------

class _LegacyBlock4(nn.Module):
    def __init__(self, out_ch: int, w_dim: int) -> None:
        super().__init__()
        self.const  = nn.Parameter(torch.randn(1, out_ch, 4, 4))
        self.conv   = ModulatedConv2d(out_ch, out_ch, kernel=3, w_dim=w_dim)
        self.noise  = NoiseInjection()
        self.act    = nn.LeakyReLU(0.2)
        self.to_rgb = ToRGB(out_ch, w_dim)

    def forward(self, w: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zeros_b = (w.sum(dim=1) * 0).view(-1, 1, 1, 1)
        x = self.const + zeros_b
        x = self.act(self.noise(self.conv(x, w), noise)) * math.sqrt(2)
        return x, self.to_rgb(x, w)


class _LegacyBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int) -> None:
        super().__init__()
        self.conv1  = ModulatedConv2d(in_ch, out_ch, kernel=3, w_dim=w_dim, upsample=True)
        self.noise1 = NoiseInjection()
        self.conv2  = ModulatedConv2d(out_ch, out_ch, kernel=3, w_dim=w_dim)
        self.noise2 = NoiseInjection()
        self.act    = nn.LeakyReLU(0.2)
        self.to_rgb = ToRGB(out_ch, w_dim)

    def forward(self, x, prev_rgb, w, n1, n2):
        x   = self.act(self.noise1(self.conv1(x, w), n1)) * math.sqrt(2)
        x   = self.act(self.noise2(self.conv2(x, w), n2)) * math.sqrt(2)
        rgb = F.interpolate(prev_rgb, scale_factor=2, mode="bilinear", align_corners=False)
        return x, rgb + self.to_rgb(x, w)


# ---------------------------------------------------------------------------
# Legacy generator
# ---------------------------------------------------------------------------

class _LegacyGenerator(nn.Module):
    def __init__(self, resolution, z_dim, w_dim,
                 channel_base, channel_max, mapping_layers, mapping_lr_mul):
        super().__init__()
        self.z_dim    = z_dim
        self.log2_res = int(math.log2(resolution))

        def nf(r): return _nf(r, channel_base, channel_max)

        self.mapping = _MappingNetwork(z_dim, w_dim, mapping_layers, mapping_lr_mul)
        self.b4      = _LegacyBlock4(nf(4), w_dim)
        self.blocks  = nn.ModuleList()
        in_ch = nf(4)
        for log2_r in range(3, self.log2_res + 1):
            res = 2 ** log2_r
            self.blocks.append(_LegacyBlock(in_ch, nf(res), w_dim))
            in_ch = nf(res)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.mapping(z)

        def _z(h, ww): return torch.zeros(1, 1, h, ww, device=z.device)

        x, rgb = self.b4(w, _z(4, 4))
        for i, blk in enumerate(self.blocks):
            h = 8 * (2 ** i)
            x, rgb = blk(x, rgb, w, _z(h, h), _z(h, h))
        return torch.tanh(rgb)


# ---------------------------------------------------------------------------
# Submission wrapper (mirrors export_onnx_baseline.py)
# ---------------------------------------------------------------------------

class SubmissionWrapper(nn.Module):
    """Run G(z) and resize the output to TARGET_RESOLUTION×TARGET_RESOLUTION."""

    def __init__(self, G: nn.Module) -> None:
        super().__init__()
        self.G = G

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.G(z)
        x = F.interpolate(
            x,
            size=(TARGET_RESOLUTION, TARGET_RESOLUTION),
            mode="bilinear",
            align_corners=False,
        )
        return x


# ---------------------------------------------------------------------------
# Export function (mirrors export_onnx_baseline.py API)
# ---------------------------------------------------------------------------

def export_to_onnx(
    G: nn.Module,
    out_path: str | Path,
    *,
    opset: int = 17,
    batch_size: int = 1,
) -> None:
    """Export G (z → image) wrapped to (B, 512) → (B, 3, 1024, 1024).

    Batch dimension is exported dynamic; other dimensions are static.
    """
    if getattr(G, "z_dim", None) != 512:
        raise ValueError(
            f"G.z_dim must be 512 (assignment spec). Got {getattr(G, 'z_dim', None)!r}."
        )

    _patch_modulated_convs(G)
    G.eval()
    wrapper = SubmissionWrapper(G).eval()

    dummy_z = torch.randn(batch_size, 512)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_z,
        str(out_path),
        input_names=["z"],
        output_names=["image"],
        opset_version=opset,
        dynamo=False,
    )

    with torch.no_grad():
        ref_out = wrapper(dummy_z)
    print(f"Saved ONNX → {out_path}")
    print(f"  input  z      (B, 512)")
    print(f"  output image  {tuple(ref_out.shape)} (B dynamic), "
          f"range [{ref_out.min():.3f}, {ref_out.max():.3f}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    CKPT      = "ckpt_1024_0020000.pth"
    FINAL_PATH = "generator_1024_step20000.onnx"

    print(f"Loading {CKPT} ...")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    print(f"  step={ckpt['step']}, resolution={cfg['resolution']}")

    G = _LegacyGenerator(
        resolution     = cfg["resolution"],
        z_dim          = cfg["z_dim"],
        w_dim          = cfg["w_dim"],
        channel_base   = cfg["channel_base"],
        channel_max    = cfg["channel_max"],
        mapping_layers = cfg["mapping_layers"],
        mapping_lr_mul = cfg["mapping_lr_mul"],
    )
    G.load_state_dict(ckpt["G"])
    print(f"  Parameters: {sum(p.numel() for p in G.parameters()):,}")

    export_to_onnx(G, FINAL_PATH)

    # Verify ONNX
    import onnx
    import numpy as np
    m = onnx.load(FINAL_PATH)
    onnx.checker.check_model(m)
    n_params = sum(int(np.prod(t.dims)) for t in m.graph.initializer)
    print(f"  ONNX parameters: {n_params:,} ({n_params/1e6:.3f}M)")
    assert n_params < 50_000_000, f"Exceeds 50M limit ({n_params:,})"
    print("  ONNX checker: PASS")


if __name__ == "__main__":
    main()
