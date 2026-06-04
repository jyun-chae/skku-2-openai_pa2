"""StyleGAN2 loss functions.

D loss: non-saturating logistic — softplus(-D(real)) + softplus(D(fake)).
  Never saturates when D is strong; avoids vanishing G gradients.

D reg: R1 gradient penalty (lazy) — gamma/2 * E[||grad_x D(x)||²] on real images only.
  One-sided GP is sufficient to prevent memorization and cheaper than two-sided.
  Effective weight = gamma * d_reg_interval/(d_reg_interval+1).

G reg: path-length regularization (lazy) — encourages J_w: w→image to have consistent
  Frobenius norm, making w more isotropic. Uses Jacobian-vector products with random v.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def d_logistic_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """Non-saturating logistic loss for D."""
    return (F.softplus(-real_pred) + F.softplus(fake_pred)).mean()


def d_r1_loss(real_pred: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
    """R1 gradient penalty. Returns unscaled penalty; caller multiplies by (gamma/2 * d_reg_interval)."""
    (grad,) = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_img,
        create_graph=True,
    )
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()


def g_nonsaturating_loss(fake_pred: torch.Tensor) -> torch.Tensor:
    """Non-saturating logistic loss for G (maximise D score on fake images)."""
    return F.softplus(-fake_pred).mean()


def g_path_length_loss(
    fake_img: torch.Tensor,
    latents_w: torch.Tensor,
    mean_path_length: torch.Tensor,
    decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Path-length regularization (StyleGAN2 §3.1).

    Args:
        fake_img:         Generator output [B, 3, H, W] in fp32.
        latents_w:        Leaf-node w [B, w_dim], must require_grad=True.
        mean_path_length: Running EMA of expected path length (scalar tensor).
        decay:            EMA decay rate for mean_path_length.

    Returns:
        pl_loss, updated mean_path_length (detached), per-sample pl_lengths.
    """
    # Normalise noise by 1/sqrt(HW) so the Frobenius norm of J is pixel-count-independent.
    noise = torch.randn_like(fake_img) / (fake_img.shape[2] * fake_img.shape[3]) ** 0.5
    (grad,) = torch.autograd.grad(
        outputs=(fake_img * noise).sum(),
        inputs=latents_w,
        create_graph=True,
    )
    pl_lengths = (grad.pow(2).sum(dim=1) + 1e-8).sqrt()  # +1e-8: grad²=0 at init → sqrt backward = inf → NaN
    mean_path_length = mean_path_length + decay * (pl_lengths.mean() - mean_path_length)
    pl_loss = (pl_lengths - mean_path_length.detach()).pow(2).mean()
    return pl_loss, mean_path_length.detach(), pl_lengths.detach()
