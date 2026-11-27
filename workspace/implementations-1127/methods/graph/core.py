"""GRAPH core: anchor-driven self-generated step-level supervision (IDEA_07_GRAPH-1127).

Self-contained: imports ONLY torch + stdlib (+ stdlib collections). Implements
proposal §3: weighted local memory graph, strength-weighted path sampling,
step-level self-generated supervision with g_t labels, SCS^self and coverage.
The LLM probing/generation is in runtime.model; this core is the graph + the
differentiable step-scope loss over self-generated trajectories.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class GraphConfig:
    lam: float = 1.0      # retain KL
    max_paths: int = 32
    walk_len: int = 4
    strength_min: float = 0.1


@dataclass
class Out:
    loss: torch.Tensor
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def float_diagnostics(self) -> Dict[str, float]:
        return {k: (float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v))
                for k, v in self.diagnostics.items()}


class MemoryGraph:
    """Weighted local memory graph built from repeated probing of an anchor."""

    def __init__(self):
        self.edges: Dict[Tuple[str, str], float] = defaultdict(float)
        self.counts: Dict[Tuple[str, str], int] = defaultdict(int)

    def add_path(self, nodes: Sequence[str], weights: Sequence[float] | None = None):
        for i in range(len(nodes) - 1):
            a, b = nodes[i], nodes[i + 1]
            w = weights[i] if weights else 1.0
            self.counts[(a, b)] += 1
            self.edges[(a, b)] += float(w)

    def strength(self, a: str, b: str) -> float:
        return self.edges.get((a, b), 0.0) / max(self.counts.get((a, b), 1), 1)

    def neighbors(self, node: str) -> List[str]:
        return [b for (a, b) in self.edges if a == node]

    def entities(self) -> List[str]:
        s = set()
        for a, b in self.edges:
            s.add(a); s.add(b)
        return sorted(s)

    def sample_paths(self, anchor: str, n: int, walk_len: int,
                     rng: torch.Generator | None = None) -> List[List[str]]:
        """Strength-weighted random walks with coverage constraint."""
        paths = []
        for _ in range(n):
            path = [anchor]
            cur = anchor
            for _ in range(walk_len - 1):
                nbrs = self.neighbors(cur)
                if not nbrs:
                    break
                ws = torch.tensor([max(self.strength(cur, x), 1e-6) for x in nbrs])
                idx = int(torch.multinomial(ws, 1, generator=rng).item())
                cur = nbrs[idx]; path.append(cur)
            paths.append(path)
        return paths


def coverage(graph: MemoryGraph, probed_facts: Sequence[str],
             gold_facts: Sequence[str]) -> float:
    """cov(G_z) = |probed ∩ gold| / |gold| (reported, non-differentiable)."""
    if not gold_facts:
        return 0.0
    hit = sum(1 for g in gold_facts if any(g in n for n in graph.entities() for _ in [0])
              or any(g in p for p in probed_facts))
    return min(1.0, hit / len(gold_facts))


def step_scope_loss(step_logits: torch.Tensor, ref_step_logits: torch.Tensor,
                    entity_ids: torch.Tensor, g: torch.Tensor,
                    config: GraphConfig | None = None) -> torch.Tensor:
    """L_sup on in-scope self-generated steps; L_keep (KL) on neighbor steps."""
    lp = F.log_softmax(step_logits, dim=-1)
    ent = lp.gather(-1, entity_ids.view(-1, 1, 1).expand(-1, step_logits.shape[1], 1)).squeeze(-1)
    l_sup = -torch.log((1.0 - ent.exp()).clamp_min(1e-8))
    lq = F.log_softmax(ref_step_logits, dim=-1)
    l_keep = (lp.exp() * (lp - lq)).sum(-1)
    return (g * l_sup + (1.0 - g) * l_keep).mean()


def graph_loss(step_logits, ref_step_logits, entity_ids, g, retain_kl,
               config: GraphConfig | None = None) -> Out:
    scope = step_scope_loss(step_logits, ref_step_logits, entity_ids, g, config)
    loss = scope + (config or GraphConfig()).lam * retain_kl
    # SCS^self = mean over trajectories of min_t 1[Leak(r_t)<=eps]; here the soft
    # version uses p(entity) at each step as leak proxy.
    p = F.softmax(step_logits, dim=-1).gather(
        -1, entity_ids.view(-1, 1, 1).expand(-1, step_logits.shape[1], 1)).squeeze(-1)
    scs = (p.max(dim=1).values <= 0.05).float().mean()
    return Out(loss=loss, diagnostics={
        "graph_scope": scope.detach(), "graph_SCS_self": scs.detach(),
        "graph_retain_kl": retain_kl.detach(),
    })


class GRAPH:
    def __init__(self, config: GraphConfig | None = None):
        self.config = config or GraphConfig()

    def graph(self):
        return MemoryGraph()

    def loss(self, *args, **kwargs):
        return graph_loss(*args, config=self.config, **kwargs)
