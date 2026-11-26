"""CSDA: Causal Step-wise Dependency Ablation for CoT unlearning.

Core reference implementation of proposal IDEA_01. The module operates on
step-level hidden states ``[T, B, H]`` and a differentiable leak-loss callback,
so it can be attached to any transformer without importing a specific model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .base import MethodOutput


@dataclass
class CSDAConfig:
    tau: float = 0.5
    kappa: float = 0.1
    lam: float = 1.0
    mu: float = 0.5
    n_patch_samples: int = 4
    cov_jitter: float = 1e-6


def _sample_retain(
    mean: torch.Tensor,
    cov: torch.Tensor,
    batch: int,
    n_samples: int,
    generator: Optional[torch.Generator] = None,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """Draw ``[n_samples, batch, H]`` from N(mean, cov)."""
    h = mean.numel()
    if cov.shape != (h, h):
        raise ValueError(f"cov must be [{h},{h}], got {tuple(cov.shape)}")
    eye = torch.eye(h, dtype=cov.dtype, device=cov.device)
    try:
        chol = torch.linalg.cholesky(cov + jitter * eye)
    except RuntimeError:
        chol = torch.linalg.cholesky(cov + 1e-3 * eye)
    eps = torch.randn(n_samples, batch, h, dtype=mean.dtype, device=mean.device,
                      generator=generator)
    return mean.view(1, 1, h) + eps @ chol.T


def causal_step_contributions(
    hidden_states: torch.Tensor,
    leak_loss_fn: Callable[[torch.Tensor], torch.Tensor],
    retain_mean: torch.Tensor,
    retain_cov: torch.Tensor,
    config: Optional[CSDAConfig] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Estimate c_t by activation patching at every reasoning step.

    ``leak_loss_fn`` must accept the same ``[T, B, H]`` tensor and return a
    scalar leak loss. The returned vector has shape ``[T]``; positive values
    mean that replacing the step with retain statistics *increased* leakage.
    """
    cfg = config or CSDAConfig()
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must be [T, B, H]")
    t_steps, batch, _ = hidden_states.shape
    base = leak_loss_fn(hidden_states).detach()
    samples = _sample_retain(retain_mean, retain_cov, batch, cfg.n_patch_samples,
                             generator, cfg.cov_jitter)
    contributions = []
    for t in range(t_steps):
        vals = []
        for k in range(cfg.n_patch_samples):
            patched = hidden_states.clone()
            patched[t] = samples[k]
            vals.append((leak_loss_fn(patched).detach() - base).reshape(()))
        contributions.append(torch.stack(vals).mean())
    return torch.stack(contributions)


def continuous_mask(contributions: torch.Tensor, tau: float = 0.5,
                    kappa: float = 0.1) -> torch.Tensor:
    """Continuous relaxation m_t = sigmoid((c_t - tau) / kappa)."""
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    return torch.sigmoid((contributions - tau) / kappa)


def binary_mask(contributions: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    return (contributions >= tau).to(contributions.dtype)


def intervention_sparsity(mask: torch.Tensor) -> torch.Tensor:
    """Fraction of steps selected as bearing, in [0, 1]."""
    return mask.float().mean()


def csda_loss(
    current_hidden: torch.Tensor,
    base_hidden: torch.Tensor,
    v_safe: torch.Tensor,
    contributions: torch.Tensor,
    retain_kl: torch.Tensor,
    config: Optional[CSDAConfig] = None,
) -> MethodOutput:
    """L_CSDA = sum m_t ||phi - v_safe||^2 + lam*KL + mu*sum(1-m_t)||phi-phi0||^2.

    Shapes: hidden states ``[T, B, H]``; ``v_safe`` is ``[H]`` or ``[T, H]``;
    ``contributions`` is ``[T]``.
    """
    cfg = config or CSDAConfig()
    if current_hidden.shape != base_hidden.shape:
        raise ValueError("current_hidden and base_hidden must match")
    if current_hidden.dim() != 3:
        raise ValueError("hidden states must be [T, B, H]")
    if contributions.dim() != 1 or contributions.numel() != current_hidden.shape[0]:
        raise ValueError("contributions must be [T]")
    m = continuous_mask(contributions, cfg.tau, cfg.kappa).to(current_hidden.dtype)
    if v_safe.dim() == 1:
        v_safe = v_safe.view(1, 1, -1)
    elif v_safe.dim() == 2:
        v_safe = v_safe.view(v_safe.shape[0], 1, v_safe.shape[1])
    redirect = (m.view(-1, 1, 1) * (current_hidden - v_safe).pow(2)).sum()
    preserve = ((1.0 - m).view(-1, 1, 1) *
                (current_hidden - base_hidden).pow(2)).sum()
    loss = redirect + cfg.lam * retain_kl + cfg.mu * preserve
    return MethodOutput(loss=loss, diagnostics={
        "csda_redirect": redirect.detach(),
        "csda_preserve": preserve.detach(),
        "csda_mask_mean": m.detach().mean(),
        "csda_sparsity": intervention_sparsity(binary_mask(contributions, cfg.tau)).detach(),
        "csda_max_contribution": contributions.detach().max(),
    })


class CSDA:
    """Small convenience wrapper around the CSDA primitives."""

    def __init__(self, config: Optional[CSDAConfig] = None):
        self.config = config or CSDAConfig()

    def contributions(self, hidden_states, leak_loss_fn, retain_mean, retain_cov,
                      generator=None):
        return causal_step_contributions(hidden_states, leak_loss_fn, retain_mean,
                                         retain_cov, self.config, generator)

    def loss(self, current_hidden, base_hidden, v_safe, contributions, retain_kl):
        return csda_loss(current_hidden, base_hidden, v_safe, contributions,
                         retain_kl, self.config)
