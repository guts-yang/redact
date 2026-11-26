"""RRBU: Relearning-Robust Bilevel Unlearning.

Core reference implementation of proposal IDEA_03. The inner loop simulates a
few gradient-descent recovery steps on a forget subset; the outer objective is
evaluated at the attacked parameters. A SAM-style perturbation is provided as
the flat-minimum regularizer from the proposal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional

import torch

from .base import MethodOutput


@dataclass
class RRBUConfig:
    inner_lr: float = 1e-3
    inner_steps: int = 3
    lam: float = 1.0
    rho: float = 0.01
    sam_radius: float = 0.0
    create_graph: bool = True


Params = Mapping[str, torch.Tensor]
LossFn = Callable[[Params], torch.Tensor]


def _trainable_names(params: Params):
    names = [name for name, value in params.items() if value.requires_grad]
    if not names:
        raise ValueError("at least one parameter must require grad")
    return names


def inner_relearn_attack(
    params: Params,
    attack_loss_fn: LossFn,
    config: Optional[RRBUConfig] = None,
    create_graph: Optional[bool] = None,
) -> Dict[str, torch.Tensor]:
    """Run K gradient-descent steps on the recovery loss.

    The update is ``theta <- theta - eta * grad L_rec``. Because ``L_rec`` is a
    negative log-likelihood, descending it *increases* the forgotten-answer
    likelihood, i.e. it is a relearning attack.
    """
    cfg = config or RRBUConfig()
    graph = cfg.create_graph if create_graph is None else create_graph
    current = dict(params)
    for _ in range(cfg.inner_steps):
        loss = attack_loss_fn(current)
        names = _trainable_names(current)
        grads = torch.autograd.grad(
            loss, [current[name] for name in names], create_graph=graph,
            allow_unused=True
        )
        updated = dict(current)
        for name, grad in zip(names, grads):
            if grad is not None:
                updated[name] = current[name] - cfg.inner_lr * grad
        current = updated
    return current


def parameter_l2(params: Params, reference: Optional[Params] = None) -> torch.Tensor:
    if reference is None:
        return sum(v.pow(2).sum() for v in params.values())
    return sum((params[k] - reference[k]).pow(2).sum() for k in params)


def rrbu_loss(
    params: Params,
    attack_loss_fn: LossFn,
    forget_loss_fn: LossFn,
    retain_loss_fn: LossFn,
    theta0: Optional[Params] = None,
    config: Optional[RRBUConfig] = None,
) -> MethodOutput:
    """Outer bilevel objective: post-attack forget loss + retain + proximity."""
    cfg = config or RRBUConfig()
    attacked = inner_relearn_attack(params, attack_loss_fn, cfg)
    post_attack = forget_loss_fn(attacked)
    retain = retain_loss_fn(params)
    prox = parameter_l2(params, theta0) if theta0 is not None else \
        parameter_l2(params)
    loss = post_attack + cfg.lam * retain + cfg.rho * prox
    return MethodOutput(loss=loss, diagnostics={
        "rrbu_post_attack_forget": post_attack.detach(),
        "rrbu_retain": retain.detach(),
        "rrbu_proximity": prox.detach(),
        "rrbu_inner_steps": cfg.inner_steps,
    })


def sam_perturbation(
    params: Params,
    loss_fn: LossFn,
    radius: float,
) -> Dict[str, torch.Tensor]:
    """Return theta + rho * g / ||g|| for the SAM inner maximization."""
    loss = loss_fn(params)
    names = _trainable_names(params)
    grads = torch.autograd.grad(loss, [params[name] for name in names],
                                allow_unused=True)
    perturbed: Dict[str, torch.Tensor] = dict(params)
    for name, grad in zip(names, grads):
        if grad is not None:
            norm = grad.norm().clamp_min(1e-12)
            perturbed[name] = params[name] + radius * grad / norm
    return perturbed


def sam_loss(
    params: Params,
    loss_fn: LossFn,
    radius: float,
) -> torch.Tensor:
    """Evaluate the loss at the SAM-perturbed point."""
    return loss_fn(sam_perturbation(params, loss_fn, radius))


def recovery_rate(leaked_after: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Fraction of held-out forget items whose leakage exceeds ``threshold``."""
    return (leaked_after >= threshold).float().mean()


class RRBU:
    def __init__(self, config: Optional[RRBUConfig] = None):
        self.config = config or RRBUConfig()

    def attack(self, params, attack_loss_fn):
        return inner_relearn_attack(params, attack_loss_fn, self.config)

    def loss(self, params, attack_loss_fn, forget_loss_fn, retain_loss_fn,
             theta0=None):
        return rrbu_loss(params, attack_loss_fn, forget_loss_fn, retain_loss_fn,
                         theta0, self.config)

    def sam_loss(self, params, loss_fn):
        return sam_loss(params, loss_fn, self.config.sam_radius)
