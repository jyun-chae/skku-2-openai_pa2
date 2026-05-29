"""StyleGAN2 loss functions.

Loss design rationale
---------------------
D loss: non-saturating logistic
  softplus(-D(real)) + softplus(D(fake))
  Unlike the original GAN loss, this never saturates for G even when D is strong,
  which avoids vanishing gradients early in training.

D regularization: R1 gradient penalty (lazy — every d_reg_interval steps)
  gamma/2 * E[||grad_x D(x)||²]  evaluated on real images only.
  Penalizing only real gradients (not fake) is sufficient to prevent D from
  memorizing the training set and is cheaper than two-sided GP.
  Lazy evaluation amortizes cost: effective penalty weight = gamma * d_reg_interval/(d_reg_interval+1).

G loss: non-saturating logistic
  softplus(-D(G(z)))
  G maximizes D's score for generated images.

G regularization: path-length regularization (lazy — every g_reg_interval steps)
  Encourages the mapping J_w: w → image to have a consistent Frobenius norm across
  directions and positions. This makes the w space more isotropic (disentangled).
  Implemented via Jacobian-vector products with a random image-space direction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def d_logistic_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """Non-saturating logistic loss for D."""
    return (F.softplus(-real_pred) + F.softplus(fake_pred)).mean()


def d_r1_loss(real_pred: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
    """R1 gradient penalty — must be called in fp32 (see trainer for context).

    Returns the UNSCALED penalty; caller multiplies by (gamma/2 * d_reg_interval).
    Uses create_graph=True so that the penalty's backward can propagate through
    the gradient computation into D's parameters.
    """
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

    Samples a random image-space direction v ~ N(0, 1/HW), then computes the
    Jacobian-vector product (J_w)^T v via a single backward pass.
    The resulting vector has the same dimension as w, and its L2 norm gives the
    'path length' — how much the image changes per unit step in w.

    The EMA `mean_path_length` provides the moving target; the loss minimises
    the squared deviation of current path lengths from this target.

    Args:
        fake_img:         Generator output [B, 3, H, W] in fp32.
        latents_w:        Leaf-node w used to generate fake_img [B, w_dim].
                          Must be fp32 and require_grad=True.
        mean_path_length: Running EMA of expected path length (scalar tensor).
        decay:            EMA decay rate for mean_path_length.

    Returns:
        pl_loss:          Scalar penalty term.
        mean_path_length: Updated EMA (detached).
        pl_lengths:       Per-sample path lengths for logging.
    """
    # Normalise noise by 1/sqrt(HW) so the Frobenius norm of J is pixel-count-independent.
    noise = torch.randn_like(fake_img) / (fake_img.shape[2] * fake_img.shape[3]) ** 0.5
    (grad,) = torch.autograd.grad(
        outputs=(fake_img * noise).sum(),
        inputs=latents_w,
        create_graph=True,
    )
    # +1e-8 before sqrt: at initialisation grad² can be exactly 0, whose
    # backward derivative is inf and causes NaN on the very first PL step.
    pl_lengths = (grad.pow(2).sum(dim=1) + 1e-8).sqrt()           # [B]
    mean_path_length = mean_path_length + decay * (pl_lengths.mean() - mean_path_length)
    pl_loss = (pl_lengths - mean_path_length.detach()).pow(2).mean()
    return pl_loss, mean_path_length.detach(), pl_lengths.detach()
