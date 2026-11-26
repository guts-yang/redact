"""Full RIPPLE hook: entity-claim graph, multi-hop sets, neighbor KL/pathcut."""
from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import torch

from ..base import MethodOutput
from ..ripple import (RIPPLEConfig, pathcut_loss, reconstruction_rate,
                      ripple_loss, ripple_metric, semantic_neighbors)


@dataclass
class RIPPLEHookConfig:
    rho: float = 0.5
    hops: int = 2
    lambda1: float = 0.1
    lambda2: float = 0.1
    tau: float = 0.7


_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)*\b")


def extract_entities(text: str) -> List[str]:
    """Capitalized phrases as lightweight entity mentions."""
    stop = {"The", "A", "An", "This", "That", "These", "Those", "In", "On",
            "For", "From", "With", "He", "She", "His", "Her", "It", "They"}
    out = []
    for match in _ENTITY_RE.finditer(text or ""):
        phrase = match.group(0).strip()
        if not phrase or phrase.split()[0] in stop:
            continue
        out.append(phrase)
    # stable unique
    return list(dict.fromkeys(out))


class KnowledgeGraph:
    """Entity co-occurrence graph built from knowledge-bank claim strings."""

    def __init__(self):
        self.adj: Dict[str, Set[str]] = defaultdict(set)
        self.entity_claims: Dict[str, List[str]] = defaultdict(list)

    @classmethod
    def from_claims(cls, claims: Sequence[str]) -> "KnowledgeGraph":
        graph = cls()
        for claim in claims:
            ents = extract_entities(claim)
            for ent in ents:
                graph.entity_claims[ent].append(claim)
            for i, a in enumerate(ents):
                for b in ents[i + 1:]:
                    graph.adj[a].add(b)
                    graph.adj[b].add(a)
        return graph

    @classmethod
    def from_claims_file(cls, path: str) -> "KnowledgeGraph":
        data = json.loads(Path(path).read_text())
        if isinstance(data, dict):
            data = data.get("claims", data.get("resolutions", []))
        claims = []
        for item in data:
            if isinstance(item, str):
                claims.append(item)
            elif isinstance(item, dict):
                claims.append(item.get("resolved_claim") or item.get("original_claim") or "")
        return cls.from_claims(claims)

    def neighbors(self, entity: str, hops: int = 1) -> Set[str]:
        if entity not in self.adj:
            return set()
        seen = {entity}
        frontier = deque([(entity, 0)])
        while frontier:
            node, depth = frontier.popleft()
            if depth >= hops:
                continue
            for nxt in self.adj[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append((nxt, depth + 1))
        seen.discard(entity)
        return seen

    def entities(self) -> List[str]:
        return sorted(self.adj)

    def claims_for(self, entity: str) -> List[str]:
        return list(self.entity_claims.get(entity, []))


class RIPPLEHook:
    def __init__(self, graph: KnowledgeGraph,
                 config: Optional[RIPPLEHookConfig] = None):
        self.graph = graph
        self.config = config or RIPPLEHookConfig()

    def neighborhood(self, entity: str,
                     embeddings: Optional[torch.Tensor] = None,
                     anchor_idx: Optional[int] = None) -> Set[str]:
        if embeddings is not None and anchor_idx is not None:
            ids = semantic_neighbors(embeddings, anchor_idx, self.config.rho)
            entities = self.graph.entities()
            return {entities[i] for i in ids if i < len(entities)}
        return self.graph.neighbors(entity, self.config.hops)

    def neighbor_kl(self, current_logits: torch.Tensor,
                    ref_logits: torch.Tensor) -> torch.Tensor:
        """Per-neighbor KL(p_theta || p_ref), shape [N]."""
        if current_logits.shape != ref_logits.shape:
            raise ValueError("current_logits and ref_logits must match")
        logp = torch.log_softmax(current_logits.float(), dim=-1)
        logq = torch.log_softmax(ref_logits.float(), dim=-1)
        return (logp.exp() * (logp - logq)).sum(dim=-1)

    def pathcut(self, current_cot: torch.Tensor,
                base_forget_cot: torch.Tensor) -> torch.Tensor:
        return pathcut_loss(current_cot, base_forget_cot, self.config.tau)

    def loss(
        self,
        forget_loss: torch.Tensor,
        current_logits_neighbors: torch.Tensor,
        ref_logits_neighbors: torch.Tensor,
        current_cot: torch.Tensor,
        base_forget_cot: torch.Tensor,
    ) -> MethodOutput:
        kl = self.neighbor_kl(current_logits_neighbors, ref_logits_neighbors)
        cut = self.pathcut(current_cot, base_forget_cot).unsqueeze(0)
        cfg = RIPPLEConfig(lambda1=self.config.lambda1,
                           lambda2=self.config.lambda2,
                           tau=self.config.tau, hops=self.config.hops)
        return ripple_loss(forget_loss, kl, cut, cfg)

    def metrics(
        self,
        current_utility: torch.Tensor,
        base_utility: torch.Tensor,
        neighbor_ids: Sequence[int],
        recovered: torch.Tensor,
    ):
        return {
            "ripple": float(ripple_metric(current_utility, base_utility,
                                          neighbor_ids).detach().cpu()),
            "recon": float(reconstruction_rate(recovered).detach().cpu()),
        }
