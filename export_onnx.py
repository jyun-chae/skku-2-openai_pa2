"""Export ckpt_1024_0020000.pth → ONNX with true dynamic-batch support.

Root cause of the 2 GB OOM
--------------------------
blocks.6 upsamples (b, 256, 256, 256) → (b, 256, 512, 512).
For b=8:  8 * 256 * 512 * 512 * 4 = 2,147,483,648 bytes = exactly 2^31.
Older ONNX Runtime BFC-arena implementations use signed 32-bit integers for
buffer sizes internally, so any allocation >= 2^31 bytes fails regardless of
available memory.

Fix
---
Wrap a batch=1 generator in an ONNX Loop so that the evaluator's batch-N
input is processed one sample at a time.  Peak intermediate memory is O(b=1)
(~300 MB for the largest 1024×1024 blocks), completely avoiding the 2^31
ceiling.

Architecture note
-----------------
The checkpoint was trained with the standard StyleGAN2 layout (conv2 + ToRGB
per block, no SqueezeConnection).  The legacy classes below match that layout
exactly; existing source files are not modified.
"""

import copy
import math
import sys
import types

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "src")
from models.layers import EqualLinear, ModulatedConv2d, NoiseInjection, PixelNorm, ToRGB


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
            x_fl  = x_sl.reshape(-1, c, h * wd)           # (b, c, h*w)
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
        x = self.conv(x, w)
        x = self.noise(x, noise)
        x = self.act(x) * math.sqrt(2)
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
        x = self.act(self.noise1(self.conv1(x, w), n1)) * math.sqrt(2)
        x = self.act(self.noise2(self.conv2(x, w), n2)) * math.sqrt(2)
        rgb = F.interpolate(prev_rgb, scale_factor=2, mode="bilinear", align_corners=False)
        return x, rgb + self.to_rgb(x, w)


# ---------------------------------------------------------------------------
# Legacy generator  (b=1 — used as the Loop body)
# ---------------------------------------------------------------------------

class _LegacyGenerator(nn.Module):
    def __init__(self, resolution, z_dim, w_dim,
                 channel_base, channel_max, mapping_layers, mapping_lr_mul):
        super().__init__()
        self.z_dim = z_dim
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
# Step 1: export b=1 inner ONNX
# ---------------------------------------------------------------------------

def _export_inner(G: nn.Module, path: str, z_dim: int) -> None:
    _patch_modulated_convs(G)
    G.eval()
    dummy = torch.randn(1, z_dim)
    torch.onnx.export(
        G, (dummy,), path,
        opset_version=17,
        input_names=["z"], output_names=["image"],
        do_constant_folding=True, dynamo=False,
    )
    print(f"  Inner b=1 ONNX → {path}")


# ---------------------------------------------------------------------------
# Step 2: wrap in ONNX Loop for dynamic batch
# ---------------------------------------------------------------------------

def _wrap_loop(inner_path: str, out_path: str, resolution: int) -> None:
    """Embed the b=1 generator as an ONNX Loop body.

    The Loop iterates N times (N = runtime batch size), processing one sample
    per iteration.  Peak tensor size is O(b=1), eliminating the 2^31-byte BFC
    allocation that causes OOM at b=8.
    """
    model = onnx.load(inner_path)
    G     = copy.deepcopy(model.graph)
    P     = "gen_"

    def pfx(n: str) -> str: return (P + n) if n else ""

    # ---- Prefix every tensor name in the inner graph ----
    for node in G.node:
        for i, n in enumerate(node.input):  node.input[i]  = pfx(n)
        for i, n in enumerate(node.output): node.output[i] = pfx(n)
    for init in G.initializer:   init.name = pfx(init.name)
    for vi   in G.value_info:    vi.name   = pfx(vi.name)
    # graph-level input/output names (used only for body wiring below)
    inner_z_name   = pfx("z")       # "gen_z"   – produced by Unsqueeze
    inner_img_name = pfx("image")   # "gen_image" – consumed by Squeeze

    # ---- Axes constant (opset ≥ 13: Unsqueeze/Squeeze take axes as input) ----
    axes0 = numpy_helper.from_array(np.array([0], dtype=np.int64), name="body_axes0")

    # ---- Prefix nodes: outer z[iter_count] → gen_z (shape [1, 512]) ----
    pre = [
        helper.make_node("Gather",    ["z", "iter_count"],      ["z_gathered"], axis=0),
        helper.make_node("Unsqueeze", ["z_gathered", "body_axes0"], [inner_z_name]),
    ]

    # ---- Suffix nodes: gen_image → image_scan_out (shape [3, res, res]) ----
    suf = [
        helper.make_node("Squeeze", [inner_img_name, "body_axes0"], ["image_scan_out"]),
    ]

    # ---- Body graph ----
    body = helper.make_graph(
        pre + list(G.node) + suf,
        "loop_body",
        [
            helper.make_tensor_value_info("iter_count",     TensorProto.INT64, []),
            helper.make_tensor_value_info("loop_cond",      TensorProto.BOOL,  []),
        ],
        [
            helper.make_tensor_value_info("loop_cond",      TensorProto.BOOL,  []),
            helper.make_tensor_value_info("image_scan_out", TensorProto.FLOAT,
                                          [3, resolution, resolution]),
        ],
        list(G.initializer) + [axes0],
    )
    body.value_info.extend(G.value_info)

    # ---- Outer graph ----
    o_zero = numpy_helper.from_array(np.array(0,    dtype=np.int64), name="o_zero")
    o_cond = numpy_helper.from_array(np.array(True, dtype=bool),     name="o_cond")

    loop_node = helper.make_node("Loop", ["N", "o_cond"], ["image_raw"])
    loop_node.attribute.append(helper.make_attribute("body", body))

    outer_nodes = [
        helper.make_node("Shape",    ["z"],             ["z_shape"]),
        helper.make_node("Gather",   ["z_shape", "o_zero"], ["N"], axis=0),
        loop_node,
        helper.make_node("Identity", ["image_raw"],     ["image"]),
    ]

    outer_graph = helper.make_graph(
        outer_nodes,
        "generator_dynamic_batch",
        [helper.make_tensor_value_info("z",     TensorProto.FLOAT, ["N", 512])],
        [helper.make_tensor_value_info("image",  TensorProto.FLOAT,
                                       ["N", 3, resolution, resolution])],
        [o_zero, o_cond],
    )

    final = helper.make_model(outer_graph,
                              opset_imports=[helper.make_opsetid("", 17)])
    final.ir_version = model.ir_version
    onnx.checker.check_model(final)
    onnx.save(final, out_path)
    print(f"  Loop-wrapped dynamic-batch ONNX → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    CKPT       = "ckpt_1024_0020000.pth"
    INNER_PATH = "generator_1024_inner_b1.onnx"
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

    # Step 1 – b=1 inner model
    _export_inner(G, INNER_PATH, cfg["z_dim"])

    # Step 2 – Loop wrapper for dynamic batch
    _wrap_loop(INNER_PATH, FINAL_PATH, cfg["resolution"])

    # Parameter count
    m = onnx.load(FINAL_PATH)
    n_params = sum(int(np.prod(t.dims)) for t in m.graph.initializer)
    # Loop body initializers live inside the Loop node attribute, not top-level
    for node in m.graph.node:
        for attr in node.attribute:
            if attr.HasField("g"):
                n_params += sum(int(np.prod(t.dims)) for t in attr.g.initializer)
    limit  = 50_000_000
    status = "OK" if n_params < limit else "OVER LIMIT"
    print(f"  ONNX parameters: {n_params:,} ({n_params/1e6:.3f}M)  [{status}]")
    if n_params >= limit:
        raise RuntimeError(f"Exceeds 50M limit ({n_params:,})")

    # Verify PyTorch forward for b=1 and b=8
    print("Verifying PyTorch forward ...")
    with torch.no_grad():
        for bs in (1, 8):
            out = G(torch.randn(bs, cfg["z_dim"]))
            assert out.shape == (bs, 3, cfg["resolution"], cfg["resolution"])
            print(f"  b={bs} → {tuple(out.shape)}  OK")


if __name__ == "__main__":
    main()
