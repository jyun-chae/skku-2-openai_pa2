"""StyleGAN2 Trainer.

Features:
  - Non-saturating logistic D/G loss
  - R1 gradient penalty (lazy, every d_reg_interval steps)
  - Path-length regularization (lazy, every g_reg_interval steps)
  - Gradient clipping on Generator
  - Mixed-precision (torch.amp) for A100
  - WandB logging: D/G loss, real/fake score, score_gap, sample images
  - FID evaluation on validation set (pytorch-fid)
  - Google Drive checkpoint backup
"""

from __future__ import annotations

import math
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
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
    """StyleGAN2 training loop.

    Args:
        cfg: Config namespace / dict-like (use types.SimpleNamespace or argparse.Namespace).
    """

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
        # Lazy-reg LR scaling (StyleGAN2)
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
        self.g_scaler = GradScaler(enabled=self.cfg.use_amp)
        self.d_scaler = GradScaler(enabled=self.cfg.use_amp)

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

        with autocast(enabled=cfg.use_amp):
            fake_img = self.G(z, noise_mode="random").detach()
            real_pred = self.D(real_img)
            fake_pred = self.D(fake_img)
            d_loss = d_logistic_loss(real_pred, fake_pred)

        self.d_optim.zero_grad(set_to_none=True)
        self.d_scaler.scale(d_loss).backward()
        self.d_scaler.step(self.d_optim)
        self.d_scaler.update()

        logs = {
            "D/loss": d_loss.item(),
            "D/real_score": real_pred.mean().item(),
            "D/fake_score": fake_pred.mean().item(),
            "D/score_gap": (real_pred.mean() - fake_pred.mean()).item(),
        }

        # Lazy R1 regularization — must run in fp32.
        # Reason: autograd.grad inside autocast returns fp16 gradients whose
        # pow(2).sum() over 3×H×W pixels easily overflows fp16 (max 65504),
        # producing inf/NaN in the penalty before GradScaler can catch it.
        if self.step % cfg.d_reg_interval == 0:
            real_img_r1 = real_img.detach().float().requires_grad_(True)
            with autocast(enabled=False):
                real_pred_r1 = self.D(real_img_r1)
                r1_penalty = d_r1_loss(real_pred_r1, real_img_r1)
                r1_loss = (cfg.r1_gamma / 2) * r1_penalty * cfg.d_reg_interval

            self.d_optim.zero_grad(set_to_none=True)
            r1_loss.backward()          # fp32 backward — no scaler needed
            self.d_optim.step()
            logs["D/r1_penalty"] = r1_penalty.item()

        return logs

    def g_step(self) -> dict:
        cfg = self.cfg
        self.D.requires_grad_(False)
        self.G.requires_grad_(True)

        z = self._sample_z(cfg.batch_size)

        with autocast(enabled=cfg.use_amp):
            fake_img, w = self.G(z, noise_mode="random", return_w=True)
            fake_pred = self.D(fake_img)
            g_loss = g_nonsaturating_loss(fake_pred)

        self.g_optim.zero_grad(set_to_none=True)
        self.g_scaler.scale(g_loss).backward()
        self.g_scaler.unscale_(self.g_optim)
        nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=cfg.grad_clip)
        self.g_scaler.step(self.g_optim)
        self.g_scaler.update()

        logs = {"G/loss": g_loss.item()}

        # Lazy path-length regularization
        if self.step % cfg.g_reg_interval == 0:
            z_pl = self._sample_z(max(1, cfg.batch_size // 2))
            # Map z→w without tracking grad through mapping; then differentiate
            # only the synthesis network w.r.t. w (standard StyleGAN2 approach).
            with torch.no_grad():
                w_pl = self.G.mapping(z_pl)
            w_pl = w_pl.detach().requires_grad_(True)   # leaf node in synthesis graph
            with autocast(enabled=False):               # float32 for stable grad norm
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
        """Main training loop.

        Args:
            fid_cache: Pre-built ValidFIDCache (real stats computed once).
                       If None, stats are recomputed at each FID evaluation step.
        """
        cfg = self.cfg
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        loader_iter = _infinite(train_loader)
        total_steps = cfg.total_kimg * 1000 // cfg.batch_size

        # LR values are constant (no scheduler); read once for logging
        lr_g = self.g_optim.param_groups[0]["lr"]
        lr_d = self.d_optim.param_groups[0]["lr"]

        # Build FID cache now if not provided
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

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------
            if self.step % cfg.log_interval == 0:
                kimg = self.step * cfg.batch_size / 1000
                _print_logs(self.step, kimg, logs)

                if wandb_run is not None:
                    wandb_run.log({"step": self.step, "kimg": kimg, **logs})

            # ----------------------------------------------------------
            # Sample images → WandB
            # ----------------------------------------------------------
            if self.step % cfg.sample_interval == 0 and wandb_run is not None:
                self._log_samples(wandb_run, n=16)

            # ----------------------------------------------------------
            # FID on valid set (uses pre-cached real statistics)
            # ----------------------------------------------------------
            if self.step % cfg.fid_interval == 0 and self.step > 0:
                fid = fid_cache.compute(
                    self.G,
                    n_gen=cfg.n_fid_samples,
                    batch_size=cfg.batch_size,
                )
                print(f"  [FID] step={self.step}  FID={fid:.2f}")
                if wandb_run is not None:
                    wandb_run.log({"step": self.step, "FID": fid})

            # ----------------------------------------------------------
            # Checkpoint
            # ----------------------------------------------------------
            if self.step % cfg.save_interval == 0 and self.step > 0:
                path = os.path.join(ckpt_dir, f"ckpt_{cfg.resolution}_{self.step:07d}.pth")
                self.save(path)
                if drive_backup_dir:
                    _backup_to_drive(path, drive_backup_dir)

        # Final checkpoint
        path = os.path.join(ckpt_dir, f"ckpt_{cfg.resolution}_final.pth")
        self.save(path)
        if drive_backup_dir:
            _backup_to_drive(path, drive_backup_dir)
        print("[Trainer] Training complete.")

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "step": self.step,
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "g_optim": self.g_optim.state_dict(),
            "d_optim": self.d_optim.state_dict(),
            "mean_path_length": self.mean_path_length,
            "cfg": vars(self.cfg) if hasattr(self.cfg, "__dict__") else dict(self.cfg),
        }, path)
        print(f"  [ckpt] saved → {path}")

    def load(self, path: str, strict: bool = True) -> None:
        state = torch.load(path, map_location=self.device)
        self.G.load_state_dict(state["G"], strict=strict)
        self.D.load_state_dict(state["D"], strict=strict)
        if strict:
            self.g_optim.load_state_dict(state["g_optim"])
            self.d_optim.load_state_dict(state["d_optim"])
            self.step = state["step"]
            self.mean_path_length = state.get("mean_path_length", self.mean_path_length)
        print(f"  [ckpt] loaded ← {path}  (step={self.step})")

    # ------------------------------------------------------------------
    # WandB sample grid
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _log_samples(self, wandb_run, n: int = 16) -> None:
        import wandb
        self.G.eval()
        z = torch.randn(n, self.cfg.z_dim, device=self.device)
        imgs = self.G(z, noise_mode="const")          # [-1, 1]
        imgs = (imgs.clamp(-1, 1) + 1) / 2           # [0, 1]
        imgs = imgs.cpu()
        grid = _make_grid(imgs, nrow=int(n ** 0.5))
        wandb_run.log({"step": self.step, "samples": wandb.Image(grid)})
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
