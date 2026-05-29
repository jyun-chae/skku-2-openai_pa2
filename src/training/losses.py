"""StyleGAN2 loss functions.

Losses:
    D: Non-saturating logistic  +  R1 gradient penalty (lazy: every 16 steps)
    G: Non-saturating logistic  +  Path-length regularization (lazy: every 4 steps)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Discriminator losses
# ---------------------------------------------------------------------------

def d_logistic_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """Non-saturating logistic loss for D."""
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return (real_loss + fake_loss).mean()


def d_r1_loss(real_pred: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
    """R1 gradient penalty (differentiable on real images only).

    Returns the unscaled penalty; caller multiplies by gamma/2.
    """
    (grad,) = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_img,
        create_graph=True,
    )
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()


# ---------------------------------------------------------------------------
# Generator losses
# ---------------------------------------------------------------------------

def g_nonsaturating_loss(fake_pred: torch.Tensor) -> torch.Tensor:
    """Non-saturating logistic loss for G (maximise D(fake))."""
    return F.softplus(-fake_pred).mean()


def g_path_length_loss(
    fake_img: torch.Tensor,
    latents_w: torch.Tensor,
    mean_path_length: torch.Tensor,
    decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Path-length regularization (StyleGAN2).

    Encourages the Jacobian to have a consistent scale across all w directions.

    Returns:
        pl_loss:        Scalar path-length penalty term.
        mean_path_length: Updated EMA of expected path length.
        pl_lengths:     Per-sample path lengths (for logging).
    """
    noise = torch.randn_like(fake_img) / (fake_img.shape[2] * fake_img.shape[3]) ** 0.5
    (grad,) = torch.autograd.grad(
        outputs=(fake_img * noise).sum(),
        inputs=latents_w,
        create_graph=True,
    )
    # +1e-8 before sqrt: grad² sum can be exactly 0 at init, whose backward
    # derivative is inf and causes NaN on the first PL reg step.
    pl_lengths = (grad.pow(2).sum(dim=1) + 1e-8).sqrt()            # [B]
    mean_path_length = mean_path_length + decay * (pl_lengths.mean() - mean_path_length)
    pl_loss = (pl_lengths - mean_path_length.detach()).pow(2).mean()
    return pl_loss, mean_path_length.detach(), pl_lengths.detach()
