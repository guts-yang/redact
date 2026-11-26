"""Full CFA hook: response styles, adversary training, and SCA audit."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import MethodOutput
from ..cfa import (CFAConfig, cfa_loss, extract_style_features,
                   linear_probe_auc, side_channel_advantage)


@dataclass
class CFAHookConfig:
    feature_dim: int = 7
    beta: float = 0.1
    lam: float = 1.0
    grl_coef: float = 1.0
    probe_steps: int = 300
    probe_lr: float = 0.05
    probe_seed: int = 0
    n_counterfactuals: int = 4


class CounterfactualSampler:
    """Small template/latent sampler for diverse safe counterfactuals.

    It is deliberately model-free: the hook can later replace it with a
    generator, but this provides the required style-diversity baseline.
    """

    TEMPLATES = (
        "I don't have that information.",
        "I can't confirm that detail.",
        "That is not something I can answer from the given context.",
        "I don't know; the relevant fact is unavailable to me.",
        "I'm unable to provide that particular information.",
        "I can't verify that, so I won't guess.",
    )

    def sample(self, question: str, n: int = 4, seed: int = 0) -> List[str]:
        rng = random.Random(seed)
        picks = [rng.choice(self.TEMPLATES) for _ in range(n)]
        return [f"{p}" for p in picks]


class CFAHook:
    def __init__(self, feature_dim: int = 7, config: Optional[CFAHookConfig] = None,
                 device: str = "cuda"):
        self.config = config or CFAHookConfig(feature_dim=feature_dim)
        self.device = device
        self.adversary = nn.Linear(feature_dim, 1).to(device).float()
        self.sampler = CounterfactualSampler()

    def response_features(self, responses: Sequence[str]) -> torch.Tensor:
        return extract_style_features(responses).to(self.device).float()

    def differentiable_features(self, hidden: torch.Tensor) -> torch.Tensor:
        """Hidden-state style proxy for the training path."""
        flat = hidden.float()
        return torch.cat([flat.mean(1), flat.std(1), flat.norm(dim=2).mean(1, keepdim=True)],
                         dim=1)

    def training_loss(
        self,
        forget_loss: torch.Tensor,
        retain_loss: torch.Tensor,
        hidden: torch.Tensor,
        membership: torch.Tensor,
    ) -> MethodOutput:
        style = self.differentiable_features(hidden)
        # The adversary head must match the hidden-state feature width.
        if style.shape[1] != self.adversary.in_features:
            self.adversary = nn.Linear(style.shape[1], 1).to(self.device).float()
        cfg = CFAConfig(beta=self.config.beta, lam=self.config.lam,
                        grl_coef=self.config.grl_coef,
                        probe_steps=self.config.probe_steps,
                        probe_lr=self.config.probe_lr,
                        probe_seed=self.config.probe_seed)
        return cfa_loss(forget_loss, retain_loss, style, membership,
                        self.adversary, cfg)

    def update_adversary(
        self,
        hidden: torch.Tensor,
        membership: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Train the adversary on detached style features (no GRL)."""
        style = self.differentiable_features(hidden).detach()
        if style.shape[1] != self.adversary.in_features:
            self.adversary = nn.Linear(style.shape[1], 1).to(self.device).float()
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            self.adversary(style).view(-1), membership.float().view(-1))
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    def audit_responses(
        self,
        responses: Sequence[str],
        membership: torch.Tensor,
    ) -> Tuple[float, float]:
        features = self.response_features(responses)
        return side_channel_advantage(
            features, membership.float(),
            CFAConfig(probe_steps=self.config.probe_steps,
                      probe_lr=self.config.probe_lr,
                      probe_seed=self.config.probe_seed))

    def audit_features(
        self,
        features: torch.Tensor,
        membership: torch.Tensor,
    ) -> Tuple[float, float]:
        return side_channel_advantage(
            features.float(), membership.float(),
            CFAConfig(probe_steps=self.config.probe_steps,
                      probe_lr=self.config.probe_lr,
                      probe_seed=self.config.probe_seed))

    def counterfactuals(self, question: str, seed: int = 0) -> List[str]:
        return self.sampler.sample(question, self.config.n_counterfactuals, seed)
