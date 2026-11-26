"""Full EAS hook: claim-graph entanglement, CoT depth, budgeted alpha."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from ..base import MethodOutput
from ..eas import EAS, EASConfig
from .common import split_cot_steps


@dataclass
class EASFeatures:
    kappa: torch.Tensor
    depth: torch.Tensor
    cost: torch.Tensor


def cooccurrence_kappa(entity: str, claims: Sequence[str]) -> float:
    """Claim-level entanglement proxy for one entity.

    kappa = (# claims mentioning the entity) / (# claims), weighted by the
    number of co-occurring capitalized neighbours.
    """
    if not claims:
        return 0.0
    ent = entity.lower()
    hits = [c for c in claims if ent in c.lower()]
    if not hits:
        return 0.0
    # cheap neighbour count: claims that mention another capitalized phrase
    neighbour_total = 0
    for claim in hits:
        import re
        caps = re.findall(r"\b[A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)*\b", claim)
        neighbour_total += max(0, len({c for c in caps if c.lower() != ent}) - 1)
    return min(1.0, len(hits) / len(claims) + neighbour_total / (len(claims) * 10.0))


def depth_from_cot(cot: str, max_steps: int = 24) -> float:
    steps = split_cot_steps(cot, max_steps=max_steps)
    return min(1.0, len(steps) / max(1, max_steps))


class EASFeatureBuilder:
    def __init__(self, claims: Sequence[str]):
        self.claims = list(claims)

    def build(self, entity: str, cot: str, max_steps: int = 24) -> EASFeatures:
        k = torch.tensor([cooccurrence_kappa(entity, self.claims)], dtype=torch.float32)
        d = torch.tensor([depth_from_cot(cot, max_steps)], dtype=torch.float32)
        return EASFeatures(k, d, torch.ones_like(k))


class EASHook:
    def __init__(self, claims: Sequence[str], config: Optional[EASConfig] = None):
        self.builder = EASFeatureBuilder(claims)
        self.eas = EAS(config or EASConfig())

    def alpha(self, kappa: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return self.eas.alpha(kappa, depth)

    def loss(
        self,
        repr_loss: torch.Tensor,
        out_loss: torch.Tensor,
        retain_kl: torch.Tensor,
        kappa: torch.Tensor,
        depth: torch.Tensor,
        leak: Optional[torch.Tensor] = None,
    ) -> MethodOutput:
        alpha = self.alpha(kappa, depth)
        return self.eas.loss(alpha, repr_loss, out_loss, retain_kl, kappa, leak)

    def budget_curve(
        self,
        kappa: torch.Tensor,
        depth: torch.Tensor,
        budgets: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    ) -> List[Dict[str, float]]:
        alpha = self.alpha(kappa, depth)
        rows = []
        for budget in budgets:
            cfg = EASConfig(budget=float(budget))
            cost = alpha * cfg.c_repr + (1.0 - alpha) * cfg.c_out
            rows.append({
                "budget": float(budget),
                "mean_alpha": float(alpha.mean().detach().cpu()),
                "mean_cost": float(cost.mean().detach().cpu()),
                "feasible": bool(float(cost.sum().detach().cpu()) <= budget),
            })
        return rows
