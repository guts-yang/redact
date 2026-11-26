"""Full RRBU hook: functional Call inner attack + outer post-attack loss."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from ..base import MethodOutput
from ..rrbu import RRBUConfig, inner_relearn_attack, parameter_l2
from .common import functional_forward, sequence_nll


@dataclass
class RRBUHookConfig:
    inner_lr: float = 1e-3
    inner_steps: int = 3
    lam: float = 1.0
    rho: float = 0.01
    create_graph: bool = True


Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def attack_eval_split(
    rows: Sequence[dict],
    n_attack: int,
    seed: int = 1001,
) -> Tuple[list, list]:
    """Split forget rows into an attack subset and a held-out eval subset."""
    import random
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_attack = min(max(0, n_attack), len(rows))
    return rows[:n_attack], rows[n_attack:]


class RRBUHook:
    """Bilevel adapter using ``torch.func.functional_call``.

    The parameter dict must contain names from ``model.named_parameters()``;
    only trainable LoRA parameters are normally passed.
    """

    def __init__(self, model, config: Optional[RRBUHookConfig] = None):
        self.model = model
        self.config = config or RRBUHookConfig()
        self.theta0: Dict[str, torch.Tensor] = {
            name: p.detach().clone()
            for name, p in model.named_parameters() if p.requires_grad
        }

    def trainable_params(self) -> Dict[str, torch.Tensor]:
        return {name: p for name, p in self.model.named_parameters()
                if p.requires_grad}

    def nll(self, params: Mapping[str, torch.Tensor], batch: Batch) -> torch.Tensor:
        input_ids, labels, mask = batch
        out = functional_forward(self.model, dict(params), input_ids, mask)
        return sequence_nll(out.logits, labels)

    def attack(
        self,
        params: Mapping[str, torch.Tensor],
        batch: Batch,
    ) -> Dict[str, torch.Tensor]:
        """One unrolled inner relearning attack on the forget batch."""
        cfg = RRBUConfig(inner_lr=self.config.inner_lr,
                         inner_steps=self.config.inner_steps,
                         create_graph=self.config.create_graph)
        return inner_relearn_attack(
            dict(params), lambda p: self.nll(p, batch).mean(), cfg)

    def outer_loss(
        self,
        params: Optional[Mapping[str, torch.Tensor]] = None,
        attack_batch: Optional[Batch] = None,
        retain_batch: Optional[Batch] = None,
    ) -> MethodOutput:
        if attack_batch is None or retain_batch is None:
            raise ValueError("attack_batch and retain_batch are required")
        params = dict(params or self.trainable_params())
        attacked = self.attack(params, attack_batch)
        post_attack = self.nll(attacked, attack_batch).mean()
        retain = self.nll(params, retain_batch).mean()
        prox = parameter_l2(params, self.theta0)
        loss = post_attack + self.config.lam * retain + self.config.rho * prox
        return MethodOutput(loss=loss, diagnostics={
            "rrbu_post_attack_forget": post_attack.detach(),
            "rrbu_retain": retain.detach(),
            "rrbu_proximity": prox.detach(),
            "rrbu_inner_steps": self.config.inner_steps,
        })
