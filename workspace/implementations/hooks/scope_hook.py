"""Full SCOPE hook: equivalence-class variants and step-wise scope objective."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from ..base import MethodOutput
from ..scope import SCOPE, SCOPEConfig, step_scope_consistency


@dataclass
class SCOPEHookConfig:
    max_steps: int = 16
    n_variants: int = 3
    lam: float = 1.0


class EquivalenceClassBuilder:
    TEMPLATES = (
        "{q}",
        "Rewrite the same question in different words: {q}",
        "In other words, {q}",
        "Indirectly: {q}",
        "A paraphrased version asks: {q}",
    )

    def __init__(self, n_variants: int = 3):
        self.n_variants = n_variants

    def variants(self, question: str) -> List[str]:
        return [t.format(q=question) for t in self.TEMPLATES[:self.n_variants]]

    def build(self, rows: Sequence[dict]) -> List[dict]:
        out = []
        for row in rows:
            out.append({
                "base": row,
                "variants": self.variants(row["question"]),
            })
        return out


class SCOPEHook:
    def __init__(self, config: Optional[SCOPEHookConfig] = None):
        self.config = config or SCOPEHookConfig()
        self.builder = EquivalenceClassBuilder(self.config.n_variants)
        self.scope = SCOPE(SCOPEConfig(lam=self.config.lam))

    def step_logits(
        self,
        logits: torch.Tensor,
        max_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Use the last ``T`` token positions as step logits, shape [B,T,V]."""
        t = min(max_steps or self.config.max_steps, logits.shape[1])
        return logits[:, -t:, :]

    def loss(
        self,
        step_logits: torch.Tensor,
        ref_step_logits: torch.Tensor,
        entity_ids: torch.Tensor,
        retain_kl: torch.Tensor,
        scope_mask: Optional[torch.Tensor] = None,
    ) -> MethodOutput:
        if scope_mask is None:
            scope_mask = torch.ones(step_logits.shape[:2], device=step_logits.device)
        return self.scope.loss(step_logits, ref_step_logits, entity_ids,
                               scope_mask, retain_kl)

    @staticmethod
    def scs(leak_matrix: torch.Tensor, epsilon: float = 0.05) -> torch.Tensor:
        return step_scope_consistency(leak_matrix, epsilon)

    def equivalence_classes(self, rows: Sequence[dict]) -> List[dict]:
        return self.builder.build(rows)
