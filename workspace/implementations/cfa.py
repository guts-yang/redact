"""CFA: Counterfactual Side-Channel Audit and De-correlation.

Core reference implementation of proposal IDEA_08. A gradient-reversal layer
removes membership information from response-style features while a separate
adversary is trained to recover it. The audit metric is SCA = sup(AUC - 0.5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MethodOutput


@dataclass
class CFAConfig:
    beta: float = 0.1
    lam: float = 1.0
    grl_coef: float = 1.0
    probe_steps: int = 300
    probe_lr: float = 0.05
    probe_seed: int = 0


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coefficient: float):
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.coefficient * grad_output, None


def gradient_reversal(x: torch.Tensor, coefficient: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, coefficient)


def extract_style_features(
    responses: Sequence[str],
    think_token: str = "</think>",
) -> torch.Tensor:
    """Cheap response-style features: length, think position, template markers.

    Features are intentionally model-free so the audit can run after generation.
    """
    rows: List[List[float]] = []
    refusal_markers = ("i don't know", "i do not know", "cannot", "can't",
                       "unable", "no information", "not sure")
    counterfactual_markers = ("instead", "alternatively", "suppose", "if we",
                              "counterfactual", "would have")
    for text in responses:
        t = (text or "").lower()
        n = max(len(t), 1)
        think_pos = text.find(think_token) if text else -1
        features = [
            float(len(t)),
            float(len(t)) / 1000.0,
            1.0 if think_token in (text or "") else 0.0,
            float(think_pos) / n if think_pos >= 0 else 1.0,
            sum(t.count(m) for m in refusal_markers) / n * 1000.0,
            sum(t.count(m) for m in counterfactual_markers) / n * 1000.0,
            float(t.count("i ") + t.count("we ")) / n * 1000.0,
        ]
        rows.append(features)
    return torch.tensor(rows, dtype=torch.float32)


def style_variance(features: torch.Tensor) -> torch.Tensor:
    return features.var(dim=0, unbiased=False).mean()


def adversary_loss(logits: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, membership.float())


def cfa_loss(
    forget_loss: torch.Tensor,
    retain_loss: torch.Tensor,
    style_features: torch.Tensor,
    membership: torch.Tensor,
    adversary: nn.Module,
    config: Optional[CFAConfig] = None,
) -> MethodOutput:
    """L_forget + lam*L_retain - beta*L_adv(GRL(S))."""
    cfg = config or CFAConfig()
    reversed_features = gradient_reversal(style_features, cfg.grl_coef)
    adv_logits = adversary(reversed_features).view(-1)
    adv = adversary_loss(adv_logits, membership)
    loss = forget_loss + cfg.lam * retain_loss - cfg.beta * adv
    return MethodOutput(loss=loss, diagnostics={
        "cfa_adv_loss": adv.detach(),
        "cfa_style_variance": style_variance(style_features).detach(),
    })


def _rank_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUC with average ranks for ties."""
    scores = scores.detach().float().cpu()
    labels = labels.detach().float().cpu()
    n_pos = int((labels > 0.5).sum())
    n_neg = int((labels <= 0.5).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype)
    # average ranks for ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    pos_rank_sum = ranks[labels > 0.5].sum().item()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def linear_probe_auc(
    features: torch.Tensor,
    membership: torch.Tensor,
    steps: int = 300,
    lr: float = 0.05,
    seed: int = 0,
) -> float:
    """Train a tiny logistic probe and return its AUC (side-channel strength)."""
    torch.manual_seed(seed)
    x = features.detach().float()
    y = membership.detach().float().view(-1)
    probe = nn.Linear(x.shape[1], 1)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(probe(x).view(-1), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        scores = probe(x).view(-1)
    return _rank_auc(scores, y)


def side_channel_advantage(
    features: torch.Tensor,
    membership: torch.Tensor,
    config: Optional[CFAConfig] = None,
) -> Tuple[float, float]:
    """Return (AUC, SCA) where SCA = max(AUC - 0.5, 0)."""
    cfg = config or CFAConfig()
    auc = linear_probe_auc(features, membership, cfg.probe_steps, cfg.probe_lr,
                           cfg.probe_seed)
    return auc, max(auc - 0.5, 0.0)


class CFA:
    def __init__(self, config: Optional[CFAConfig] = None):
        self.config = config or CFAConfig()

    def loss(self, forget_loss, retain_loss, style_features, membership, adversary):
        return cfa_loss(forget_loss, retain_loss, style_features, membership,
                        adversary, self.config)

    @staticmethod
    def audit(features, membership, config: Optional[CFAConfig] = None):
        return side_channel_advantage(features, membership, config)
