"""FID (Fréchet Inception Distance) using pytorch-fid's InceptionV3 (2048-d features).

Real validation statistics are cached once per session; fake stats recomputed each call.
All computation is in-memory — no temp files needed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance


# ---------------------------------------------------------------------------
# Inception feature extractor (singleton per device)
# ---------------------------------------------------------------------------

# Singleton cache avoids re-loading Inception weights on every FID call.
_inception_cache: dict[str, nn.Module] = {}


def _get_inception(device: torch.device) -> nn.Module:
    key = str(device)
    if key not in _inception_cache:
        model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
        model.eval()
        _inception_cache[key] = model
    return _inception_cache[key]


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_features(
    imgs: torch.Tensor,  # [N, 3, H, W] in [-1, 1]
    inception: nn.Module,
    device: torch.device,
    resize: int = 299,
) -> np.ndarray:
    """Extract 2048-d Inception features. Resizes and rescales to [0,1] as InceptionV3 expects."""
    import torch.nn.functional as F
    imgs = imgs.to(device)
    if imgs.shape[-1] != resize:
        imgs = F.interpolate(imgs, size=(resize, resize), mode="bilinear", align_corners=False)
    imgs = (imgs.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]
    feats = inception(imgs)[0]           # [N, 2048, 1, 1]
    return feats.squeeze(-1).squeeze(-1).cpu().numpy()


def _loader_to_features(
    loader: DataLoader,
    inception: nn.Module,
    device: torch.device,
) -> np.ndarray:
    feats_list = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        feats_list.append(_extract_features(batch, inception, device))
    return np.concatenate(feats_list, axis=0)


def _gen_to_features(
    G,
    n: int,
    batch_size: int,
    inception: nn.Module,
    device: torch.device,
) -> np.ndarray:
    G.eval()
    feats_list = []
    generated = 0
    with torch.no_grad():
        while generated < n:
            bs = min(batch_size, n - generated)
            z = torch.randn(bs, G.z_dim, device=device)
            imgs = G(z, noise_mode="random")
            feats_list.append(_extract_features(imgs, inception, device))
            generated += bs
    G.train()
    return np.concatenate(feats_list, axis=0)


def _mu_sigma(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return feats.mean(axis=0), np.cov(feats, rowvar=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ValidFIDCache:
    """Pre-computes and caches real validation statistics for repeated FID calls.

    Usage:
        cache = ValidFIDCache(valid_loader, device)
        fid = cache.compute(G, n_gen=10000, batch_size=16)
    """

    def __init__(self, valid_loader: DataLoader, device: torch.device) -> None:
        inception = _get_inception(device)
        print("[FID] Computing real (valid) statistics…")
        feats = _loader_to_features(valid_loader, inception, device)
        self.mu_real, self.sigma_real = _mu_sigma(feats)
        print(f"[FID] Real statistics computed from {len(feats)} images.")
        self._device = device

    def compute(self, G, n_gen: int = 10000, batch_size: int = 16) -> float:
        inception = _get_inception(self._device)
        feats_fake = _gen_to_features(G, n_gen, batch_size, inception, self._device)
        mu_fake, sigma_fake = _mu_sigma(feats_fake)
        return float(calculate_frechet_distance(
            self.mu_real, self.sigma_real, mu_fake, sigma_fake
        ))


def compute_fid_from_generator(
    G,
    valid_loader: DataLoader,
    n_gen: int = 10000,
    device: Optional[torch.device] = None,
    batch_size: int = 16,
) -> float:
    """One-shot FID (no caching). Prefer ValidFIDCache for repeated calls."""
    if device is None:
        device = next(G.parameters()).device
    inception = _get_inception(device)
    feats_real = _loader_to_features(valid_loader, inception, device)
    feats_fake = _gen_to_features(G, n_gen, batch_size, inception, device)
    mu_r, sig_r = _mu_sigma(feats_real)
    mu_f, sig_f = _mu_sigma(feats_fake)
    return float(calculate_frechet_distance(mu_r, sig_r, mu_f, sig_f))
