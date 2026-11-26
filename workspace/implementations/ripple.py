"""RIPPLE: Multi-Hop Ripple Localization for relational erasure.

Core reference implementation of proposal IDEA_09. Semantic neighbours keep
their reference distribution; multi-hop paths are cut when their CoT
representation becomes too similar to the forgotten anchor.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set

import torch
import torch.nn.functional as F

from .base import MethodOutput


@dataclass
class RIPPLEConfig:
    lambda1: float = 0.1
    lambda2: float = 0.1
    tau: float = 0.7
    rho: float = 0.5
    hops: int = 2


def semantic_neighbors(
    embeddings: torch.Tensor,
    anchor: int,
    radius: float,
    exclude_self: bool = True,
) -> List[int]:
    """Return ids with L2 distance to ``anchor`` <= radius."""
    if embeddings.dim() != 2:
        raise ValueError("embeddings must be [N, H]")
    dist = (embeddings - embeddings[anchor].view(1, -1)).norm(dim=1)
    ids = torch.nonzero(dist <= radius, as_tuple=False).view(-1).tolist()
    if exclude_self and anchor in ids:
        ids.remove(anchor)
    return ids


def multi_hop_neighbors(
    adjacency: Sequence[Iterable[int]],
    anchor: int,
    hops: int = 2,
) -> Set[int]:
    """BFS neighbourhood within ``hops`` edges, excluding the anchor."""
    if anchor < 0 or anchor >= len(adjacency):
        raise ValueError("anchor out of range")
    seen = {anchor}
    frontier = deque([(anchor, 0)])
    while frontier:
        node, depth = frontier.popleft()
        if depth >= hops:
            continue
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append((nxt, depth + 1))
    seen.discard(anchor)
    return seen


def pathcut_loss(
    current_cot: torch.Tensor,
    base_forget_cot: torch.Tensor,
    tau: float = 0.7,
) -> torch.Tensor:
    """max(0, cos(phi_theta(CoT(x)), phi_theta0(CoT(x_f))) - tau)."""
    if current_cot.dim() != 2:
        raise ValueError("current_cot must be [B, H]")
    if base_forget_cot.dim() == 1:
        base_forget_cot = base_forget_cot.view(1, -1)
    sim = F.cosine_similarity(current_cot, base_forget_cot.expand_as(current_cot),
                              dim=1)
    return torch.clamp(sim - tau, min=0.0).mean()


def ripple_loss(
    forget_loss: torch.Tensor,
    neighbor_kl: torch.Tensor,
    pathcut_terms: torch.Tensor,
    config: Optional[RIPPLEConfig] = None,
) -> MethodOutput:
    """L_forget + lambda1 * sum_neighbor KL + lambda2 * sum_pathcut."""
    cfg = config or RIPPLEConfig()
    loss = forget_loss + cfg.lambda1 * neighbor_kl.sum() + cfg.lambda2 * pathcut_terms.sum()
    return MethodOutput(loss=loss, diagnostics={
        "ripple_neighbor_kl": neighbor_kl.detach().sum(),
        "ripple_pathcut": pathcut_terms.detach().sum(),
    })


def ripple_metric(
    current_utility: torch.Tensor,
    base_utility: torch.Tensor,
    neighbor_ids: Sequence[int],
) -> torch.Tensor:
    """Mean absolute utility change over the semantic neighbourhood."""
    ids = torch.tensor(list(neighbor_ids), dtype=torch.long,
                       device=current_utility.device)
    if ids.numel() == 0:
        return torch.zeros((), dtype=current_utility.dtype,
                           device=current_utility.device)
    return (current_utility[ids] - base_utility[ids]).abs().mean()


def reconstruction_rate(recovered: torch.Tensor) -> torch.Tensor:
    """Fraction of multi-hop probes that recover the forgotten entity."""
    return recovered.float().mean()


class RIPPLE:
    def __init__(self, config: Optional[RIPPLEConfig] = None):
        self.config = config or RIPPLEConfig()

    def neighbors(self, embeddings, anchor):
        return semantic_neighbors(embeddings, anchor, self.config.rho)

    def hops(self, adjacency, anchor):
        return multi_hop_neighbors(adjacency, anchor, self.config.hops)

    def loss(self, forget_loss, neighbor_kl, pathcut_terms):
        return ripple_loss(forget_loss, neighbor_kl, pathcut_terms, self.config)
