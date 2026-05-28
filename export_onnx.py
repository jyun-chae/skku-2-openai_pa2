"""Export a train.py Generator checkpoint to the Project 2 ONNX interface."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import Generator, GeneratorConfig


TARGET_RESOLUTION = 1024


class SubmissionWrapper(nn.Module):
    """Run G(z) and resize output to the required 1024x1024 shape."""

    def __init__(self, G: nn.Module):
        super().__init__()
        self.G = G

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.G(z)
        if x.shape[-2:] == (TARGET_RESOLUTION, TARGET_RESOLUTION):
            return x
        return F.interpolate(
            x,
            size=(TARGET_RESOLUTION, TARGET_RESOLUTION),
            mode="bilinear",
            align_corners=False,
        )


def load_generator(ckpt_path: Path) -> Generator:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta")
    if not isinstance(meta, dict) or "generator_config" not in meta:
        raise RuntimeError("Checkpoint must contain meta.generator_config saved by train.py")

    G = Generator(GeneratorConfig.from_dict(meta["generator_config"]))
    training_cfg = meta.get("training_config", {})
    if "resolution" in training_cfg:
        G.set_active_resolution(int(training_cfg["resolution"]))

    state = ckpt.get("G_ema_state") or ckpt.get("G_state")
    if state is None:
        raise RuntimeError("Checkpoint has neither G_ema_state nor G_state")
    G.load_state_dict(state)
    return G


def export_to_onnx(
    G: nn.Module,
    out_path: str | Path,
    *,
    opset: int = 17,
    batch_size: int = 1,
) -> None:
    """Export `G` with dynamic batch input `(B, 512)` and output `(B, 3, 1024, 1024)`."""
    if getattr(G, "z_dim", None) != 512:
        raise ValueError(
            f"G.z_dim must be 512 by the project spec. Got {getattr(G, 'z_dim', None)!r}."
        )

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
        dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}},
    )

    with torch.no_grad():
        ref_out = wrapper(dummy_z)
    print(f"Saved ONNX to {out_path}")
    print("  input  z      (B, 512)")
    print(
        f"  output image  {tuple(ref_out.shape)} (B dynamic), "
        f"range [{ref_out.min():.3f}, {ref_out.max():.3f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("submission.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    export_to_onnx(load_generator(args.ckpt), args.out, opset=args.opset, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
