"""StyleGAN2 loss functions.

D: non-saturating logistic + lazy R1 gradient penalty (real images only).
G: non-saturating logistic + lazy path-length regularization.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def d_logistic_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """Non-saturating logistic D loss: softplus(-D(real)) + softplus(D(fake)).

    Unlike the original GAN loss, this never saturates when D is strong,
    so G gradients remain meaningful throughout training.
    """
    return (F.softplus(-real_pred) + F.softplus(fake_pred)).mean()


def d_r1_loss(real_pred: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
    """R1 one-sided gradient penalty on real images (unscaled).

    Caller multiplies by (gamma/2 * d_reg_interval) for lazy regularisation.
    One-sided GP is sufficient to prevent D memorisation and cheaper than two-sided.
    """
    (grad,) = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_img,
        create_graph=True,
    )
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()


def g_nonsaturating_loss(fake_pred: torch.Tensor) -> torch.Tensor:
    """Non-saturating G loss: maximises D(fake) via softplus(-D(fake))."""
    return F.softplus(-fake_pred).mean()


def g_path_length_loss(
    fake_img: torch.Tensor,
    latents_w: torch.Tensor,
    mean_path_length: torch.Tensor,
    decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Path-length regularization (StyleGAN2 §3.1).

    Penalises deviation of ||J_w^T v||_2 from a running EMA target, encouraging
    the Jacobian J_w: w→image to have a consistent Frobenius norm so that w-space
    moves produce predictable image changes (isotropic mapping).

    Args:
        fake_img:         Generator output [B, 3, H, W] in fp32.
        latents_w:        Leaf-node w [B, w_dim] with requires_grad=True.
        mean_path_length: Running EMA of expected path length (scalar tensor).
        decay:            EMA update rate.

    Returns:
        (pl_loss, updated_mean_path_length, per_sample_pl_lengths)
    """
    # 1/sqrt(H*W) normalisation makes ||J||_F independent of image resolution
    noise = torch.randn_like(fake_img) / (fake_img.shape[2] * fake_img.shape[3]) ** 0.5
    (grad,) = torch.autograd.grad(
        outputs=(fake_img * noise).sum(),
        inputs=latents_w,
        create_graph=True,
    )
    pl_lengths = (grad.pow(2).sum(dim=1) + 1e-8).sqrt()  # +1e-8: avoids inf gradient when grad=0 at init
    mean_path_length = mean_path_length + decay * (pl_lengths.mean() - mean_path_length)
    pl_loss = (pl_lengths - mean_path_length.detach()).pow(2).mean()
    return pl_loss, mean_path_length.detach(), pl_lengths.detach()
