"""Fine-tune script

Two start modes:

1) Fine-tune from the distributed 256 baseline (most common):
       python scripts/train.py --config configs/baseline_256.yaml \
                               --init-from ckpt/ffhq256_baseline.pt

2) Resume your own training run from a full ckpt you saved earlier:
       python scripts/train.py --config configs/baseline_256.yaml \
                               --resume runs/my_run/ckpt_001000000.pt

   `--resume` restores G, D, G_ema, both optimizers, and RNG state — bit-for-bit
   continuation (assuming the same architecture).

Recipe (the one that worked after three divergences):
- ResNet GAN: GN on G, Spectral Norm on D, self-attention at 32×32
- Non-saturating logistic loss + R1 (lazy every 16 D steps, γ=10)
- DiffAug 'color,translation' (cutout disabled — too aggressive)
- Adam β=(0, 0.9), G lr = D lr = 1e-3 (avoid TTUR until you observe a problem)
- EMA G (half-life 10k images)
- fp32 throughout

Logging via wandb if installed and not disabled.
FID is measured periodically when training.fid_every_steps > 0 and pytorch-fid is installed.
"""
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import yaml
from scipy import linalg
from torch.utils.data import DataLoader

# wandb is optional — keep training runnable on environments without it.
try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    wandb = None
    _HAS_WANDB = False

try:
    from pytorch_fid.inception import InceptionV3
    _HAS_FID = True
except ImportError:
    InceptionV3 = None
    _HAS_FID = False

from src.augment import diff_augment
from src.dataset import ZipImageDataset, infinite_loader
from src.losses import ns_logistic_g, r1_penalty
from src.model import (
    Discriminator,
    DiscriminatorConfig,
    EMA,
    Generator,
    GeneratorConfig,
)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def async_save_checkpoint(path: Path, state: dict) -> threading.Thread:
    t = threading.Thread(target=save_checkpoint, args=(path, state), daemon=False)
    t.start()
    return t


def prune_finished_threads(threads: list[threading.Thread]) -> list[threading.Thread]:
    return [t for t in threads if t.is_alive()]


def _to_fid_range(x: torch.Tensor) -> torch.Tensor:
    return ((x.float() + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def _collect_inception_activations(
    model: torch.nn.Module,
    batches,
    *,
    device: str,
    max_items: int,
) -> np.ndarray:
    activations: list[np.ndarray] = []
    seen = 0
    for batch in batches:
        if seen >= max_items:
            break
        batch = batch[: max_items - seen].to(device, non_blocking=True)
        pred = model(_to_fid_range(batch))[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        activations.append(pred)
        seen += pred.shape[0]
    if not activations:
        raise RuntimeError("No images were available for FID calculation.")
    return np.concatenate(activations, axis=0)


def _activation_stats(activations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(activations, axis=0), np.cov(activations, rowvar=False)


def _frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(fid)


@torch.no_grad()
def calculate_fid(
    G: torch.nn.Module,
    real_loader: DataLoader,
    *,
    z_dim: int,
    device: str,
    num_samples: int,
    batch_size: int,
    real_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    if not _HAS_FID:
        raise RuntimeError("pytorch-fid is not installed. Install it with `pip install pytorch-fid`.")

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception = InceptionV3([block_idx]).to(device).eval()

    if real_stats is None:
        real_acts = _collect_inception_activations(
            inception,
            real_loader,
            device=device,
            max_items=num_samples,
        )
        real_stats = _activation_stats(real_acts)

    G.eval()

    def fake_batches():
        remaining = num_samples
        while remaining > 0:
            current = min(batch_size, remaining)
            z = torch.randn(current, z_dim, device=device)
            yield G(z)
            remaining -= current

    fake_acts = _collect_inception_activations(
        inception,
        fake_batches(),
        device=device,
        max_items=num_samples,
    )
    fake_stats = _activation_stats(fake_acts)
    fid = _frechet_distance(real_stats[0], real_stats[1], fake_stats[0], fake_stats[1])
    return fid, real_stats


@torch.no_grad()
def save_sample_grid(G: torch.nn.Module, sample_z: torch.Tensor, out_path: Path, nrow: int = 8) -> None:
    G.eval()
    fake = G(sample_z)
    x = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)
    grid = vutils.make_grid(x, nrow=nrow, padding=2)
    vutils.save_image(grid, out_path)


def build_checkpoint(
    *,
    images_seen: int,
    step: int,
    G: torch.nn.Module,
    D: torch.nn.Module,
    G_ema: EMA,
    optG: torch.optim.Optimizer,
    optD: torch.optim.Optimizer,
    g_cfg: GeneratorConfig,
    d_cfg: DiscriminatorConfig,
    training_cfg: dict,
    wandb_run_id: str | None,
) -> dict:
    return {
        "images_seen": images_seen,
        "step": step,
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "G_ema_state": G_ema.state_dict(),
        "optG_state": optG.state_dict(),
        "optD_state": optD.state_dict(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
        "wandb_run_id": wandb_run_id,
        "meta": {
            "generator_config": asdict(g_cfg),
            "discriminator_config": asdict(d_cfg),
            "training_config": training_cfg,
        },
    }


def load_matching_state(module: torch.nn.Module, state: dict, label: str) -> int:
    current = module.state_dict()
    matched = {
        k: v for k, v in state.items()
        if k in current and current[k].shape == v.shape
    }
    current.update(matched)
    module.load_state_dict(current)
    print(f"  {label}: loaded {len(matched)}/{len(current)} tensors")
    return len(matched)


def init_from_baseline(
    init_path: Path,
    G: torch.nn.Module,
    D: torch.nn.Module,
    G_ema: EMA,
    device: str,
) -> None:
    """Load only tensors whose names and shapes match the current stage.

    If you scale the architecture (add 512 / 1024 blocks, change channels,
    swap the up-block design, etc.), this will raise — and that's intentional.
    The transfer-learning recipe (which keys carry over, how to remap the
    discriminator's reverse-ordered stage indices, what to do with the
    last block's shape mismatch) is part of the assignment. Replace this
    function or write your own loader before scaling.
    """
    print(f"Initializing from checkpoint with matching tensors: {init_path}")
    ckpt = torch.load(init_path, map_location=device, weights_only=False)
    total_loaded = 0
    if "G_state" in ckpt:
        total_loaded += load_matching_state(G, ckpt["G_state"], "G")
    if "D_state" in ckpt:
        total_loaded += load_matching_state(D, ckpt["D_state"], "D")
    if "G_ema_state" in ckpt:
        total_loaded += load_matching_state(G_ema.shadow, ckpt["G_ema_state"], "G_ema")
    if total_loaded == 0:
        print("  No compatible tensors found; training starts from random init.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--init-from", type=Path, default=None,
        help="Path to a (possibly slim) baseline ckpt. Partial load with "
             "strict=False; optimizers/RNG start fresh.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to a full ckpt saved by this same script. Restores "
             "G/D/G_ema/optimizers/RNG/wandb run id.",
    )
    parser.add_argument("--total-images", type=int, default=None)
    parser.add_argument(
        "--train-zip", type=Path, default=None,
        help="Override training.train_zip with a zip file or extracted image directory.",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Override out.run_dir, useful for saving checkpoints directly to Google Drive.",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Run at most this many training steps in this invocation, then save latest.pt and exit.",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Run at most this many additional images in this invocation, then save latest.pt and exit.",
    )
    parser.add_argument(
        "--save-every-steps", type=int, default=None,
        help="Override training.ckpt_every_steps for step-based checkpointing.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override training.batch_size, useful when Colab GPU memory is limited.",
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=None,
        help="Override training.grad_accum_steps. Effective batch is batch_size * grad_accum_steps.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Override training.num_workers for the DataLoader.",
    )
    parser.add_argument(
        "--precision", choices=["bf16", "fp32"], default=None,
        help="Override training.precision.",
    )
    parser.add_argument(
        "--new-wandb-run", action="store_true",
        help="When --resume, start a fresh wandb run instead of reattaching.",
    )
    parser.add_argument(
        "--fid-every-steps", type=int, default=None,
        help="Override training.fid_every_steps. Set 0 to disable FID.",
    )
    parser.add_argument(
        "--fid-num-samples", type=int, default=None,
        help="Override training.fid_num_samples.",
    )
    parser.add_argument(
        "--fid-real-zip", type=Path, default=None,
        help="Override training.fid_real_zip for validation images used as FID real samples.",
    )
    args = parser.parse_args()

    if args.init_from is not None and args.resume is not None:
        raise SystemExit("Use either --init-from or --resume, not both.")

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    if args.total_images is not None:
        train_cfg["total_images"] = args.total_images
    if args.train_zip is not None:
        train_cfg["train_zip"] = str(args.train_zip)
    if args.save_every_steps is not None:
        train_cfg["ckpt_every_steps"] = args.save_every_steps
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        train_cfg["grad_accum_steps"] = args.grad_accum_steps
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.precision is not None:
        train_cfg["precision"] = args.precision
    if args.fid_every_steps is not None:
        train_cfg["fid_every_steps"] = args.fid_every_steps
    if args.fid_num_samples is not None:
        train_cfg["fid_num_samples"] = args.fid_num_samples
    if args.fid_real_zip is not None:
        train_cfg["fid_real_zip"] = str(args.fid_real_zip)
    if args.run_dir is not None:
        cfg.setdefault("out", {})["run_dir"] = str(args.run_dir)
    if args.wandb_project is not None:
        cfg.setdefault("wandb", {})["project"] = args.wandb_project
    if args.wandb_name is not None:
        cfg.setdefault("wandb", {})["name"] = args.wandb_name
    if args.wandb_mode is not None:
        cfg.setdefault("wandb", {})["mode"] = args.wandb_mode

    set_seed(train_cfg["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    g_cfg = GeneratorConfig.from_dict(cfg["generator"])
    d_cfg = DiscriminatorConfig.from_dict(cfg["discriminator"])
    G = Generator(g_cfg).to(device)
    D = Discriminator(d_cfg).to(device)
    stage_resolution = int(train_cfg["resolution"])
    G.set_active_resolution(stage_resolution)
    D.set_active_resolution(stage_resolution)
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"Generator: {g_params/1e6:.2f}M params")
    print(f"Discriminator: {d_params/1e6:.2f}M params")
    print(f"Active training resolution: {stage_resolution}")
    if g_params >= 40_000_000:
        raise ValueError(f"Generator must be under 40M parameters, got {g_params:,}")

    lr_g = float(train_cfg.get("lr_g", train_cfg.get("lr")))
    lr_d = float(train_cfg.get("lr_d", train_cfg.get("lr")))
    optG = torch.optim.Adam(
        G.parameters(), lr=lr_g,
        betas=(train_cfg["beta1"], train_cfg["beta2"]), eps=1e-8,
        weight_decay=train_cfg["weight_decay"],
    )
    optD = torch.optim.Adam(
        D.parameters(), lr=lr_d,
        betas=(train_cfg["beta1"], train_cfg["beta2"]), eps=1e-8,
        weight_decay=train_cfg["weight_decay"],
    )
    print(f"Optimizers: G lr={lr_g}, D lr={lr_d}")

    G_ema = EMA(G, half_life=train_cfg["ema_half_life"])
    G_ema.shadow.to(device)

    dataset = ZipImageDataset(train_cfg["train_zip"], flip=train_cfg["flip"])
    print(f"Dataset: {len(dataset)} images")
    num_workers = train_cfg["num_workers"]
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )
    inf_loader = infinite_loader(loader)

    fid_every_steps = int(train_cfg.get("fid_every_steps", 0) or 0)
    fid_num_samples = int(train_cfg.get("fid_num_samples", 5000))
    fid_batch_size = int(train_cfg.get("fid_batch_size", train_cfg["batch_size"]))
    fid_real_path = train_cfg.get("fid_real_zip") or train_cfg["train_zip"]
    fid_real_stats: tuple[np.ndarray, np.ndarray] | None = None
    fid_loader = None
    if fid_every_steps > 0:
        if not Path(fid_real_path).exists():
            print(
                "FID disabled: real image path does not exist: "
                f"{fid_real_path}. Pass --fid-real-zip or --fid-every-steps 0."
            )
            fid_every_steps = 0
        elif _HAS_FID:
            fid_dataset = ZipImageDataset(fid_real_path, flip=False)
            fid_loader = DataLoader(
                fid_dataset,
                batch_size=fid_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=device == "cuda",
                persistent_workers=num_workers > 0,
                prefetch_factor=2 if num_workers > 0 else None,
                drop_last=False,
            )
            print(
                f"FID: every {fid_every_steps} steps, "
                f"samples={min(fid_num_samples, len(fid_dataset))}, "
                f"real={fid_real_path}"
            )
        else:
            print("FID disabled: pytorch-fid is not installed.")
            fid_every_steps = 0

    sample_gen = torch.Generator(device="cpu").manual_seed(train_cfg["sample_seed"])
    sample_z = torch.randn(train_cfg["sample_n"], g_cfg.z_dim, generator=sample_gen).to(device)

    run_dir = Path(cfg["out"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    images_seen = 0
    step = 0
    wandb_run_id: str | None = None

    if args.init_from is not None:
        init_from_baseline(args.init_from, G, D, G_ema, device=device)

    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G_state"])
        D.load_state_dict(ckpt["D_state"])
        G_ema.load_state_dict(ckpt["G_ema_state"])
        if "optG_state" in ckpt:
            optG.load_state_dict(ckpt["optG_state"])
        if "optD_state" in ckpt:
            optD.load_state_dict(ckpt["optD_state"])
        # Force yaml LR onto the loaded optimizer state.
        for pg in optG.param_groups:
            pg["lr"] = lr_g
        for pg in optD.param_groups:
            pg["lr"] = lr_d
        images_seen = ckpt.get("images_seen", 0)
        step = ckpt.get("step", 0)
        wandb_run_id = None if args.new_wandb_run else ckpt.get("wandb_run_id")
        rng = ckpt.get("rng_state", {})
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])

    # wandb
    wandb_cfg = cfg.get("wandb", {})
    wandb_mode = wandb_cfg.get("mode", "online") if _HAS_WANDB else "disabled"
    run = None
    if wandb_mode != "disabled":
        init_kwargs = {
            "project": wandb_cfg.get("project", "ffhqgen-student"),
            "name": wandb_cfg.get("name"),
            "mode": wandb_mode,
            "config": cfg,
        }
        if wandb_run_id is not None:
            init_kwargs["id"] = wandb_run_id
            init_kwargs["resume"] = "must"
        run = wandb.init(**init_kwargs)
        wandb_run_id = run.id
        run.summary["params/G"] = g_params
        run.summary["params/D"] = d_params
        run.summary["training/resolution"] = train_cfg.get("resolution")
        run.summary["training/stage"] = train_cfg.get("stage", train_cfg.get("resolution"))

    total_images = train_cfg["total_images"]
    z_dim = g_cfg.z_dim
    r1_gamma = train_cfg["r1_gamma"]
    r1_lazy_every = train_cfg["r1_lazy_every"]
    log_every = train_cfg["log_every"]
    ckpt_every = train_cfg["ckpt_every"]
    ckpt_every_steps = int(train_cfg.get("ckpt_every_steps", 0) or 0)
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1) or 1)
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
    effective_batch_size = int(train_cfg["batch_size"]) * grad_accum_steps
    grad_clip_g = float(train_cfg.get("grad_clip_g", float("inf")))
    grad_clip_d = float(train_cfg.get("grad_clip_d", float("inf")))
    precision = train_cfg.get("precision", "fp32")
    if precision not in ("bf16", "fp32"):
        raise ValueError(f"precision must be 'bf16' or 'fp32', got {precision!r}")
    use_amp = precision == "bf16"
    if use_amp and device == "cuda" and not torch.cuda.is_bf16_supported():
        gpu_name = torch.cuda.get_device_name(0)
        raise RuntimeError(
            "BF16 precision is not supported by this CUDA GPU "
            f"({gpu_name}). Use --precision fp32 with a smaller batch size, "
            "or switch the Colab runtime to an A100/H100-class GPU."
        )
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    print(f"Precision: {precision} ({'autocast bf16' if use_amp else 'fp32 throughout'})")
    augment_policy = train_cfg.get("augment", "") or ""
    print(f"Augment policy: {augment_policy!r}")

    stop_images = total_images
    if args.max_images is not None:
        stop_images = min(stop_images, images_seen + args.max_images)
    stop_step = None if args.max_steps is None else step + args.max_steps

    last_ckpt = images_seen
    last_ckpt_step = step
    save_threads: list[threading.Thread] = []
    window_t0 = time.perf_counter()
    window_imgs = 0
    last_r1_value: float | None = None

    print(
        f"Training: images_seen={images_seen} → {total_images} "
        f"(micro_batch={train_cfg['batch_size']}, accum={grad_accum_steps}, "
        f"effective_batch={effective_batch_size}, device={device})"
    )

    if args.max_steps is not None or args.max_images is not None:
        print(
            "Chunk limit: "
            f"stop_step={stop_step if stop_step is not None else 'none'}, "
            f"stop_images={stop_images}"
        )

    while images_seen < total_images:
        if stop_step is not None and step >= stop_step:
            break
        if images_seen >= stop_images:
            break
        # --- D step ---
        optD.zero_grad(set_to_none=True)
        total_b = 0
        micro_batch_sizes: list[int] = []
        l_d_real_log = 0.0
        l_d_fake_log = 0.0
        l_d_log = 0.0
        d_real_mean_log = 0.0
        d_fake_mean_log = 0.0
        r1_log = 0.0
        did_r1 = (step + 1) % r1_lazy_every == 0
        real_for_stats = None
        fake_for_stats = None
        for _ in range(grad_accum_steps):
            real = next(inf_loader).to(device, non_blocking=True)
            b = real.size(0)
            total_b += b
            micro_batch_sizes.append(b)
            z = torch.randn(b, z_dim, device=device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                with torch.no_grad():
                    fake = G(z)
                d_real = D(diff_augment(real, augment_policy))
                d_fake = D(diff_augment(fake.detach(), augment_policy))
                l_d_real = F.softplus(-d_real).mean()
                l_d_fake = F.softplus(d_fake).mean()
                l_d = l_d_real + l_d_fake
            (l_d / grad_accum_steps).backward()

            if did_r1:
                l_r1 = r1_lazy_every * r1_penalty(
                    D, diff_augment(real.float(), augment_policy), gamma=r1_gamma,
                )
                (l_r1 / grad_accum_steps).backward()
                r1_log += float(l_r1.item()) / r1_lazy_every

            l_d_real_log += float(l_d_real.item())
            l_d_fake_log += float(l_d_fake.item())
            l_d_log += float(l_d.item())
            d_real_mean_log += float(d_real.float().mean().item())
            d_fake_mean_log += float(d_fake.float().mean().item())
            real_for_stats = real.detach().float()
            fake_for_stats = fake.detach().float()

        l_d_real_log /= grad_accum_steps
        l_d_fake_log /= grad_accum_steps
        l_d_log /= grad_accum_steps
        d_real_mean_log /= grad_accum_steps
        d_fake_mean_log /= grad_accum_steps
        if did_r1:
            last_r1_value = r1_log / grad_accum_steps

        grad_norm_d = float(
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=grad_clip_d)
        )
        optD.step()

        # --- G step ---
        optG.zero_grad(set_to_none=True)
        l_g_log = 0.0
        d_fake_g_mean_log = 0.0
        for b in micro_batch_sizes:
            z = torch.randn(b, z_dim, device=device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                fake = G(z)
                d_fake_g = D(diff_augment(fake, augment_policy))
                l_g = ns_logistic_g(d_fake_g)
            (l_g / grad_accum_steps).backward()
            l_g_log += float(l_g.item())
            d_fake_g_mean_log += float(d_fake_g.float().mean().item())
            fake_for_stats = fake.detach().float()
        l_g_log /= grad_accum_steps
        d_fake_g_mean_log /= grad_accum_steps

        grad_norm_g = float(
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=grad_clip_g)
        )
        optG.step()

        G_ema.update(G, total_b)

        images_seen += total_b
        window_imgs += total_b
        step += 1

        if step % log_every == 0:
            now = time.perf_counter()
            elapsed = max(now - window_t0, 1e-6)
            throughput = window_imgs / elapsed
            window_t0 = now
            window_imgs = 0
            d_real_fake_gap = d_real_mean_log - d_fake_mean_log
            log = {
                "images_seen": images_seen,
                "throughput/imgs_per_sec": throughput,
                "loss/D_total": l_d_log,
                "loss/D_real": l_d_real_log,
                "loss/D_fake": l_d_fake_log,
                "loss/G": l_g_log,
                "D_out/real_mean": d_real_mean_log,
                "D_out/fake_mean": d_fake_mean_log,
                "D_out/fake_for_G_mean": d_fake_g_mean_log,
                "D_real_score": d_real_mean_log,
                "D_fake_score": d_fake_mean_log,
                "D_real_score_minus_D_fake_score": d_real_fake_gap,
                "score/D_real_fake_gap": d_real_fake_gap,
                "score/G_fooling_logit": d_fake_g_mean_log,
                "score/fake_pixel_mean": float(fake_for_stats.mean().item()),
                "score/fake_pixel_std": float(fake_for_stats.std().item()),
                "score/fake_pixel_abs_mean": float(fake_for_stats.abs().mean().item()),
                "score/real_pixel_mean": float(real_for_stats.mean().item()),
                "score/real_pixel_std": float(real_for_stats.std().item()),
                "grad_norm/G": grad_norm_g,
                "grad_norm/D": grad_norm_d,
                "lr/G": optG.param_groups[0]["lr"],
                "lr/D": optD.param_groups[0]["lr"],
                "params/G_million": g_params / 1e6,
                "params/D_million": d_params / 1e6,
            }
            if last_r1_value is not None:
                log["loss/R1"] = last_r1_value
            if wandb_mode != "disabled":
                wandb.log(log, step=step)
            else:
                print(
                    f"step={step} imgs={images_seen} thr={throughput:.1f}img/s "
                    f"l_d={l_d_log:.3f} l_g={l_g_log:.3f} "
                    f"gn_g={grad_norm_g:.2f} gn_d={grad_norm_d:.2f}"
                )

        if fid_every_steps > 0 and step % fid_every_steps == 0:
            assert fid_loader is not None
            fid_samples = min(fid_num_samples, len(fid_loader.dataset))
            print(f"Calculating FID at step={step} with {fid_samples} samples...")
            fid_score, fid_real_stats = calculate_fid(
                G_ema.shadow,
                fid_loader,
                z_dim=z_dim,
                device=device,
                num_samples=fid_samples,
                batch_size=fid_batch_size,
                real_stats=fid_real_stats,
            )
            fid_log = {
                "FID": fid_score,
                "fid/score": fid_score,
                "fid/num_samples": fid_samples,
                "images_seen": images_seen,
            }
            if wandb_mode != "disabled":
                wandb.log(fid_log, step=step)
            else:
                print(f"fid={fid_score:.4f}")

        should_save_by_images = images_seen - last_ckpt >= ckpt_every
        should_save_by_steps = ckpt_every_steps > 0 and step - last_ckpt_step >= ckpt_every_steps
        if should_save_by_images or should_save_by_steps:
            ckpt = build_checkpoint(
                images_seen=images_seen, step=step,
                G=G, D=D, G_ema=G_ema, optG=optG, optD=optD,
                g_cfg=g_cfg, d_cfg=d_cfg, training_cfg=train_cfg,
                wandb_run_id=wandb_run_id,
            )
            latest_path = run_dir / "latest.pt"
            grid_path = samples_dir / f"grid_{images_seen:09d}.png"
            save_threads = prune_finished_threads(save_threads)
            saved_names = ["latest.pt"]
            if should_save_by_images:
                ckpt_path = run_dir / f"ckpt_{images_seen:09d}.pt"
                save_threads.append(async_save_checkpoint(ckpt_path, ckpt))
                saved_names.append(ckpt_path.name)
            if should_save_by_steps:
                step_ckpt_path = run_dir / f"ckpt_step_{step:08d}.pt"
                save_threads.append(async_save_checkpoint(step_ckpt_path, ckpt))
                saved_names.append(step_ckpt_path.name)
            save_checkpoint(latest_path, ckpt)
            save_sample_grid(G_ema.shadow, sample_z, grid_path, nrow=8)
            if wandb_mode != "disabled":
                wandb.log({"samples/grid": wandb.Image(str(grid_path))}, step=step)
            print(f"[ckpt+grid] {' / '.join(saved_names)} / {grid_path.name}")
            last_ckpt = images_seen
            last_ckpt_step = step

    reached_total = images_seen >= total_images
    print("Saving latest checkpoint...")
    latest_ckpt = build_checkpoint(
        images_seen=images_seen, step=step,
        G=G, D=D, G_ema=G_ema, optG=optG, optD=optD,
        g_cfg=g_cfg, d_cfg=d_cfg, training_cfg=train_cfg,
        wandb_run_id=wandb_run_id,
    )
    for t in save_threads:
        t.join()
    save_checkpoint(run_dir / "latest.pt", latest_ckpt)
    save_checkpoint(run_dir / f"ckpt_step_{step:08d}.pt", latest_ckpt)
    if reached_total:
        print("Training complete. Saving final ckpt...")
        save_checkpoint(run_dir / "final.pt", latest_ckpt)
    else:
        print(
            "Chunk complete. Resume with "
            f"--resume {run_dir / 'latest.pt'} to continue."
        )
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
