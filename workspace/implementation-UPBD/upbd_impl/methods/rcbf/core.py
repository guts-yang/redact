"""RCBF: positive-margin CoT barrier with policy/barrier co-adaptation."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from upbd_impl.contracts import (
    BaseMethod, Candidate, Decision, EvidenceBatch, LossOutput, MethodSpec,
    MethodState, ParamGroup, PrepareContext, StepContext,
)
from upbd_impl.core.decisions import accept, rollback


@dataclass
class RCBFConfig:
    eta: float = 1.0
    gamma: float = 0.1
    lambda_invariance: float = 0.1
    lambda_safety: float = 0.1
    risk_budget: float = 0.05
    max_collapse: float = 0.05


def barrier_scores(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if hidden.shape[-1] != weight.shape[0]:
        raise ValueError("hidden and barrier dimensions differ")
    return hidden @ weight


def safety_loss(scores: torch.Tensor, eta: float) -> torch.Tensor:
    return torch.clamp(eta - scores, min=0.0).mean()


def invariance_violation(current: torch.Tensor, following: torch.Tensor,
                         gamma: float) -> torch.Tensor:
    if current.shape != following.shape:
        raise ValueError("current/following barrier scores must match")
    return torch.clamp((1.0 - gamma) * current - following, min=0.0)


class RCBF(BaseMethod):
    spec = MethodSpec("rcbf", "IDEA_03_RCBF", "Robust control-barrier forgetting")

    def __init__(self, config: RCBFConfig | None = None):
        self.config = config or RCBFConfig()

    def prepare(self, ctx: PrepareContext) -> MethodState:
        hidden_dim = int(ctx.extras["hidden_dim"])
        weight = torch.zeros(hidden_dim, device=ctx.device, requires_grad=True)
        return MethodState(values={"barrier_weight": weight, "last_phase": "none"})

    def parameter_groups(self, state: MethodState) -> list[ParamGroup]:
        return [ParamGroup("barrier", [state.values["barrier_weight"]])]

    def compute_loss(self, ctx: StepContext, state: MethodState) -> LossOutput:
        if ctx.hidden is None or ctx.nll is None:
            raise ValueError("RCBF requires hidden states and NLL")
        if ctx.hidden.shape[1] < 2:
            raise ValueError("RCBF requires at least two sequence positions")
        weight = state.values["barrier_weight"]
        phase = "policy" if ctx.step % 2 == 0 else "barrier"
        hidden = ctx.hidden if phase == "policy" else ctx.hidden.detach()
        effective_weight = weight.detach() if phase == "policy" else weight
        scores = barrier_scores(hidden, effective_weight)
        violation = invariance_violation(scores[:, :-1], scores[:, 1:], self.config.gamma).mean()
        safety = safety_loss(scores, self.config.eta)
        nll = ctx.nll.mean()
        if phase == "policy":
            loss = nll + self.config.lambda_invariance * violation
        else:
            loss = (nll.detach() * 0.0 + self.config.lambda_safety * safety
                    + self.config.lambda_invariance * violation)
        state.values["last_phase"] = phase
        state.values["last_violation"] = float(violation.detach().cpu())
        min_score = scores.min()
        return LossOutput(loss=loss, diagnostics={
            "nll": nll.detach(),
            "rcbf_phase": torch.tensor(float(phase == "barrier"), device=loss.device),
            "rcbf_violation": violation.detach(),
            "rcbf_safety_loss": safety.detach(),
            "rcbf_min_score": min_score.detach(),
            "disclosure_risk": torch.sigmoid(-min_score).detach(),
            "utility": torch.sigmoid(-nll.detach()),
            "collapse_rate": torch.zeros((), device=loss.device),
        })

    def accept(self, candidate: Candidate, evidence: EvidenceBatch,
               state: MethodState) -> Decision:
        if evidence.disclosure_risk > self.config.risk_budget:
            return rollback("RCBF calibration barrier risk exceeded")
        if evidence.collapse_rate > self.config.max_collapse:
            return rollback("RCBF collapse guard exceeded")
        return accept("RCBF empirical trajectory constraints satisfied", "sampled-trajectories")
