"""StyleGAN2 training loop with lazy R1/PL regularization, AMP, and Drive checkpoint backup."""

from __future__ import annotations

import math
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..models.generator import StyleGAN2Generator
from ..models.discriminator import StyleGAN2Discriminator
from .losses import (
    d_logistic_loss,
    d_r1_loss,
    g_nonsaturating_loss,
    g_path_length_loss,
)
from ..utils.fid_score import ValidFIDCache


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """StyleGAN2 trainer. cfg must have the fields defined in configs/train_*.yaml."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_models()
        self._setup_optimizers()
        self._setup_scalers()
        self.step = 0
        self.mean_path_length = torch.zeros(1, device=self.device)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_models(self) -> None:
        cfg = self.cfg
        self.G = StyleGAN2Generator(
            resolution=cfg.resolution,
            z_dim=cfg.z_dim,
            w_dim=cfg.w_dim,
            channel_base=cfg.channel_base,
            channel_max=cfg.channel_max,
            mapping_layers=cfg.mapping_layers,
            mapping_lr_mul=cfg.mapping_lr_mul,
        ).to(self.device)

        self.D = StyleGAN2Discriminator(
            resolution=cfg.resolution,
            channel_base=cfg.channel_base,
            channel_max=cfg.channel_max,
        ).to(self.device)

        g_params = self.G.count_parameters()
        print(f"[Trainer] G params: {g_params:,}  ({g_params / 1e6:.2f}M)")
        print(f"[Trainer] D params: {sum(p.numel() for p in self.D.parameters()):,}")

    def _setup_optimizers(self) -> None:
        cfg = self.cfg
        # Lazy regularization LR correction (StyleGAN2 §B).
        # With lazy reg, effective steps per kimg = interval/(interval+1) of a
        # non-lazy run. Scaling LR and beta2 by the same ratio keeps the
        # effective learning dynamics identical to non-lazy training.
        g_ratio = cfg.g_reg_interval / (cfg.g_reg_interval + 1)
        d_ratio = cfg.d_reg_interval / (cfg.d_reg_interval + 1)

        self.g_optim = torch.optim.Adam(
            self.G.parameters(),
            lr=cfg.lr_g * g_ratio,
            betas=(0.0, 0.99 ** g_ratio),
            eps=1e-8,
        )
        self.d_optim = torch.optim.Adam(
            self.D.parameters(),
            lr=cfg.lr_d * d_ratio,
            betas=(0.0, 0.99 ** d_ratio),
            eps=1e-8,
        )

    def _setup_scalers(self) -> None:
        self.g_scaler = GradScaler("cuda", enabled=self.cfg.use_amp)
        self.d_scaler = GradScaler("cuda", enabled=self.cfg.use_amp)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _sample_z(self, batch: int) -> torch.Tensor:
        return torch.randn(batch, self.cfg.z_dim, device=self.device)

    def d_step(self, real_img: torch.Tensor) -> dict:
        cfg = self.cfg
        self.D.requires_grad_(True)
        self.G.requires_grad_(False)

        real_img = real_img.to(self.device)
        z = self._sample_z(real_img.shape[0])

        with autocast("cuda", enabled=cfg.use_amp):
            fake_img = self.G(z, noise_mode="random").detach()
            real_pred = self.D(real_img)
            fake_pred = self.D(fake_img)
            d_loss = d_logistic_loss(real_pred, fake_pred)

        self.d_optim.zero_grad(set_to_none=True)
        self.d_scaler.scale(d_loss).backward()
        self.d_scaler.step(self.d_optim)
        self.d_scaler.update()

        real_score = real_pred.mean().item()
        fake_score = fake_pred.mean().item()
        logs = {
            "D/loss": d_loss.item(),
            "D/real_score": real_score,
            "D/fake_score": fake_score,
            "D/score_gap": real_score - fake_score,
            "D/real_score_sig": torch.sigmoid(real_pred).mean().item(),
            "D/fake_score_sig": torch.sigmoid(fake_pred).mean().item(),
        }

        # R1 must run in fp32: fp16 grad.pow(2) overflows (4608 terms → inf), rsqrt(inf)=0 kills D.
        if self.step % cfg.d_reg_interval == 0:
            real_img_r1 = real_img.detach().float().requires_grad_(True)
            with autocast("cuda", enabled=False):
                real_pred_r1 = self.D(real_img_r1)
                r1_penalty = d_r1_loss(real_pred_r1, real_img_r1)
                r1_loss = (cfg.r1_gamma / 2) * r1_penalty * cfg.d_reg_interval

            self.d_optim.zero_grad(set_to_none=True)
            r1_loss.backward()
            nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=cfg.grad_clip)
            self.d_optim.step()
            logs["D/r1_penalty"] = r1_penalty.item()

        return logs

    def g_step(self) -> dict:
        cfg = self.cfg
        self.D.requires_grad_(False)
        self.G.requires_grad_(True)

        z = self._sample_z(cfg.batch_size)

        with autocast("cuda", enabled=cfg.use_amp):
            fake_img, w = self.G(z, noise_mode="random", return_w=True)
            fake_pred = self.D(fake_img)
            g_loss = g_nonsaturating_loss(fake_pred)

        self.g_optim.zero_grad(set_to_none=True)
        self.g_scaler.scale(g_loss).backward()
        self.g_scaler.unscale_(self.g_optim)
        nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=cfg.grad_clip)
        self.g_scaler.step(self.g_optim)
        self.g_scaler.update()

        # Log generated image statistics — a std → 0 means G is collapsing
        with torch.no_grad():
            img_stats = fake_img.detach().float()
            logs = {
                "G/loss": g_loss.item(),
                "G/img_mean": img_stats.mean().item(),
                "G/img_std": img_stats.std().item(),
            }

        # Skip PL until step >= pl_warmup: mean_path_length starts at 0, making pl_loss=pl_lengths² at init.
        pl_warmup = getattr(cfg, "pl_warmup_steps", 0)
        if self.step % cfg.g_reg_interval == 0 and self.step >= pl_warmup:
            z_pl = self._sample_z(max(1, cfg.batch_size // 2))
            # Differentiate synthesis only: run mapping under no_grad, make w a leaf.
            with torch.no_grad():
                w_pl = self.G.mapping(z_pl)
            w_pl = w_pl.detach().requires_grad_(True)
            with autocast("cuda", enabled=False):
                fake_img_pl = self.G(w=w_pl, noise_mode="random").float()
                pl_loss, self.mean_path_length, pl_lengths = g_path_length_loss(
                    fake_img_pl, w_pl, self.mean_path_length, decay=cfg.pl_decay
                )
                weighted_pl = pl_loss * cfg.g_reg_interval * cfg.pl_weight

            self.g_optim.zero_grad(set_to_none=True)
            self.g_scaler.scale(weighted_pl).backward()
            self.g_scaler.unscale_(self.g_optim)
            nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=cfg.grad_clip)
            self.g_scaler.step(self.g_optim)
            self.g_scaler.update()
            logs["G/pl_penalty"] = pl_loss.item()
            logs["G/pl_lengths_mean"] = pl_lengths.mean().item()

        return logs

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        wandb_run=None,
        drive_backup_dir: Optional[str] = None,
        ckpt_dir: str = "checkpoints",
        fid_cache: Optional["ValidFIDCache"] = None,
    ) -> None:
        cfg = self.cfg
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        loader_iter = _infinite(train_loader)
        total_steps = cfg.total_kimg * 1000 // cfg.batch_size

        lr_g = self.g_optim.param_groups[0]["lr"]
        lr_d = self.d_optim.param_groups[0]["lr"]

        if fid_cache is None:
            fid_cache = ValidFIDCache(valid_loader, self.device)

        print(f"[Trainer] Starting training: {total_steps} steps, res={cfg.resolution}")
        print(f"[Trainer] lr_g={lr_g:.6f}  lr_d={lr_d:.6f}  batch={cfg.batch_size}")

        for self.step in range(self.step, total_steps):
            t0 = time.perf_counter()
            real_img = next(loader_iter)

            d_logs = self.d_step(real_img)
            g_logs = self.g_step()

            logs = {
                **d_logs,
                **g_logs,
                "lr_g": lr_g,
                "lr_d": lr_d,
                "step_time_s": time.perf_counter() - t0,
            }

            # — logging —
            if self.step % cfg.log_interval == 0:
                kimg = self.step * cfg.batch_size / 1000
                _print_logs(self.step, kimg, logs)

                if wandb_run is not None:
                    wandb_run.log({"kimg": kimg, **logs}, step=self.step)

            # — samples —
            if self.step % cfg.sample_interval == 0 and wandb_run is not None:
                self._log_samples(wandb_run, n=16)

            # — FID —
            if self.step % cfg.fid_interval == 0 and self.step > 0:
                fid = fid_cache.compute(
                    self.G,
                    n_gen=cfg.n_fid_samples,
                    batch_size=cfg.batch_size,
                )
                print(f"  [FID] step={self.step}  FID={fid:.2f}")
                if wandb_run is not None:
                    wandb_run.log({"FID": fid}, step=self.step)

            # — checkpoint —
            if self.step % cfg.save_interval == 0 and self.step > 0:
                path = os.path.join(ckpt_dir, f"ckpt_{cfg.resolution}_{self.step:07d}.pth")
                wandb_id = wandb_run.id if wandb_run is not None else None
                self.save(path, wandb_run_id=wandb_id)
                if drive_backup_dir:
                    _backup_to_drive(path, drive_backup_dir)

        # Final checkpoint
        path = os.path.join(ckpt_dir, f"ckpt_{cfg.resolution}_final.pth")
        wandb_id = wandb_run.id if wandb_run is not None else None
        self.save(path, wandb_run_id=wandb_id)
        if drive_backup_dir:
            _backup_to_drive(path, drive_backup_dir)
        print("[Trainer] Training complete.")

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str, wandb_run_id: Optional[str] = None) -> None:
        torch.save({
            "step": self.step,
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "g_optim": self.g_optim.state_dict(),
            "d_optim": self.d_optim.state_dict(),
            "mean_path_length": self.mean_path_length,
            "cfg": vars(self.cfg) if hasattr(self.cfg, "__dict__") else dict(self.cfg),
            "wandb_run_id": wandb_run_id,
        }, path)
        print(f"  [ckpt] saved → {path}")

    def load(self, path: str, strict: bool = True) -> Optional[str]:
        """Load checkpoint, return saved WandB run ID."""
        state = torch.load(path, map_location=self.device)
        self.G.load_state_dict(state["G"], strict=strict)
        self.D.load_state_dict(state["D"], strict=strict)
        if strict:
            self.g_optim.load_state_dict(state["g_optim"])
            self.d_optim.load_state_dict(state["d_optim"])
            # load_state_dict restores saved LR, overriding cfg.  Re-apply cfg LR
            # so that changing lr_g/lr_d in the yaml takes effect on resume.
            g_ratio = self.cfg.g_reg_interval / (self.cfg.g_reg_interval + 1)
            d_ratio = self.cfg.d_reg_interval / (self.cfg.d_reg_interval + 1)
            for pg in self.g_optim.param_groups:
                pg["lr"] = self.cfg.lr_g * g_ratio
            for pg in self.d_optim.param_groups:
                pg["lr"] = self.cfg.lr_d * d_ratio
            self.step = state["step"]
            self.mean_path_length = state.get("mean_path_length", self.mean_path_length)
        wandb_run_id = state.get("wandb_run_id")
        print(f"  [ckpt] loaded ← {path}  (step={self.step}, wandb_id={wandb_run_id})")
        return wandb_run_id

    # ------------------------------------------------------------------
    # WandB sample grid
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _log_samples(self, wandb_run, n: int = 16) -> None:
        import wandb
        self.G.eval()
        try:
            z = torch.randn(n, self.cfg.z_dim, device=self.device)
            imgs = self.G(z, noise_mode="const")      # [-1, 1], reproducible
            imgs = (imgs.clamp(-1, 1) + 1) / 2       # [0, 1]
            grid = _make_grid(imgs.cpu(), nrow=int(n ** 0.5))
            wandb_run.log({"samples": wandb.Image(grid)}, step=self.step)
        finally:
            # Always restore train mode — an exception here would otherwise
            # leave G in eval mode and corrupt subsequent training steps.
            self.G.train()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _infinite(loader: DataLoader):
    while True:
        yield from loader


def _print_logs(step: int, kimg: float, logs: dict) -> None:
    parts = [f"step={step:7d}  kimg={kimg:8.1f}"]
    for k, v in logs.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
    print("  ".join(parts))


def _make_grid(imgs: torch.Tensor, nrow: int) -> "PIL.Image.Image":
    from torchvision.utils import make_grid
    from PIL import Image
    import numpy as np
    grid = make_grid(imgs, nrow=nrow, padding=2, normalize=False)
    arr = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


def _backup_to_drive(src: str, drive_dir: str) -> None:
    os.makedirs(drive_dir, exist_ok=True)
    dst = os.path.join(drive_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    print(f"  [backup] {src} → {dst}")


def find_latest_checkpoint(drive_ckpt_dir: str, resolution: int) -> Optional[str]:
    """Return the most recent checkpoint path for `resolution` in Drive, or None."""
    import glob
    finals = glob.glob(os.path.join(drive_ckpt_dir, f"ckpt_{resolution}_final.pth"))
    if finals:
        return finals[0]
    step_ckpts = sorted(glob.glob(os.path.join(drive_ckpt_dir, f"ckpt_{resolution}_*.pth")))
    return step_ckpts[-1] if step_ckpts else None


def restore_from_drive(
    trainer: "Trainer",
    drive_ckpt_dir: str,
    local_ckpt_dir: str,
    resolution: int,
) -> bool:
    """Copy the latest Drive checkpoint to local disk and load it; return False if none found."""
    drive_path = find_latest_checkpoint(drive_ckpt_dir, resolution)
    if drive_path is None:
        print(f"  [restore] No checkpoint found in {drive_ckpt_dir} for res={resolution} → starting fresh.")
        return False

    local_path = os.path.join(local_ckpt_dir, f"ckpt_{resolution}_restored.pth")
    print(f"  [restore] {os.path.basename(drive_path)} → local …")
    shutil.copy2(drive_path, local_path)
    trainer.load(local_path)
    return True
