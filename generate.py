"""Generate a sample grid from a train.py checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils

from src.model import Generator, GeneratorConfig


def load_generator(ckpt_path: Path, device: str, use_ema: bool) -> Generator:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt.get("meta")
    if not isinstance(meta, dict) or "generator_config" not in meta:
        raise RuntimeError("Checkpoint must contain meta.generator_config saved by train.py")

    g_cfg = GeneratorConfig.from_dict(meta["generator_config"])
    G = Generator(g_cfg).to(device).eval()
    training_cfg = meta.get("training_config", {})
    if "resolution" in training_cfg:
        G.set_active_resolution(int(training_cfg["resolution"]))

    if use_ema and "G_ema_state" in ckpt:
        state = ckpt["G_ema_state"]
        weights_note = "G_ema_state"
    elif "G_state" in ckpt:
        state = ckpt["G_state"]
        weights_note = "G_state"
    else:
        raise RuntimeError("Checkpoint contains neither G_ema_state nor G_state")
    G.load_state_dict(state)

    n_params = sum(p.numel() for p in G.parameters())
    print(f"Architecture: z_dim={g_cfg.z_dim}, max_res={g_cfg.resolutions[-1]}")
    print(f"Weights: {weights_note} ({n_params/1e6:.2f}M params)")
    return G


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("sample_grid.png"))
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--truncation-psi", type=float, default=1.0)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    G = load_generator(args.ckpt, device=device, use_ema=not args.no_ema)

    g_for_z = torch.Generator(device="cpu").manual_seed(args.seed)
    z = torch.randn(args.n, G.z_dim, generator=g_for_z).to(device)
    fake = G(z, truncation_psi=args.truncation_psi)
    x = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)
    grid = vutils.make_grid(x, nrow=args.nrow, padding=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, args.out)
    print(f"Saved {args.n} samples to {args.out}")


if __name__ == "__main__":
    main()
