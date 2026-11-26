"""ORTHO: Reasoning-Subspace Orthogonal Protection against collapse.

Core reference implementation of proposal IDEA_05. The module estimates a
top-k reasoning subspace from retain/math CoT activations, projects forget
gradients away from it, and penalizes reasoning-direction representation drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .base import MethodOutput


@dataclass
class ORTHOConfig:
    k: int = 8
    mu: float = 0.1
    lam: float = 1.0
    center: bool = True


def estimate_reasoning_subspace(
    activations: torch.Tensor,
    k: int = 8,
    center: bool = True,
) -> torch.Tensor:
    """Return U ``[H, k]`` = top-k eigenvectors of the activation covariance."""
    if activations.dim() < 2:
        raise ValueError("activations must be [N, H] or [T, B, H]")
    flat = activations.reshape(-1, activations.shape[-1]).float()
    if center:
        flat = flat - flat.mean(dim=0, keepdim=True)
    if flat.shape[0] < 2:
        raise ValueError("need at least two activation rows")
    k = min(k, flat.shape[1], flat.shape[0])
    if flat.shape[1] > 512:
        # Avoid an O(H^3) eigh on 8B hidden sizes; randomized SVD is enough for
        # a diagnostic top-k subspace.
        _, _, v = torch.pca_lowrank(flat, q=k, center=False)
        return v[:, :k]
    cov = flat.T @ flat / max(flat.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    return eigvecs[:, order[:k]]


def projector_perp(U: torch.Tensor) -> torch.Tensor:
    """P_perp = I - U U^T."""
    if U.dim() != 2:
        raise ValueError("U must be [H, k]")
    eye = torch.eye(U.shape[0], dtype=U.dtype, device=U.device)
    return eye - U @ U.T


def orthogonalize_gradient(grad: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Project a vector/matrix gradient onto the orthogonal complement of U."""
    if grad.shape[0] != U.shape[0]:
        raise ValueError("grad leading dimension must match U's [H]")
    return grad - U @ (U.T @ grad)


def subspace_drift(
    current: torch.Tensor,
    reference: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """||U^T (phi - phi0)||_F over the last hidden dimension."""
    if current.shape != reference.shape:
        raise ValueError("current and reference must match")
    delta = (current - reference).reshape(-1, current.shape[-1])
    return (delta @ U.to(delta.dtype)).norm()


def ortho_loss(
    forget_loss: torch.Tensor,
    current_hidden: torch.Tensor,
    base_hidden: torch.Tensor,
    U: torch.Tensor,
    retain_kl: torch.Tensor,
    config: Optional[ORTHOConfig] = None,
) -> MethodOutput:
    """L_forget + mu*||U^T(phi-phi0)||_F^2 + lam*KL_retain."""
    cfg = config or ORTHOConfig()
    drift = subspace_drift(current_hidden, base_hidden, U)
    loss = forget_loss + cfg.mu * drift.pow(2) + cfg.lam * retain_kl
    return MethodOutput(loss=loss, diagnostics={
        "ortho_subspace_drift": drift.detach(),
        "ortho_drift_sq": drift.detach().pow(2),
        "ortho_k": U.shape[1],
    })


class ORTHO:
    def __init__(self, config: Optional[ORTHOConfig] = None):
        self.config = config or ORTHOConfig()

    def subspace(self, activations):
        return estimate_reasoning_subspace(activations, self.config.k,
                                           self.config.center)

    def project_gradient(self, grad, U):
        return orthogonalize_gradient(grad, U)

    def loss(self, forget_loss, current_hidden, base_hidden, U, retain_kl):
        return ortho_loss(forget_loss, current_hidden, base_hidden, U, retain_kl,
                          self.config)
