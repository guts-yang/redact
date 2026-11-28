"""STREAM: stateful primal-dual repair with explicit infeasibility."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from upbd_impl.contracts import (
    BaseMethod, Candidate, Decision, EvidenceBatch, LossOutput, MethodDiagnostics,
    MethodSpec, MethodState, PrepareContext, StepContext,
)
from upbd_impl.core.decisions import accept, infeasible, rollback
from upbd_impl.core.margins import margin_violation


@dataclass
class StreamConfig:
    dual_rate: float = 0.1
    lambda_margin: float = 0.1
    risk_budget: float = 0.05
    min_utility: float = 0.0


def dual_update(current: torch.Tensor, risk: torch.Tensor, budget: torch.Tensor,
                rate: float) -> torch.Tensor:
    if current.shape != risk.shape or risk.shape != budget.shape:
        raise ValueError("dual/risk/budget shapes must match")
    if rate <= 0:
        raise ValueError("dual rate must be positive")
    return torch.clamp(current + rate * (risk - budget), min=0.0)


def cumulative_violation(risks: list[float], budgets: list[float]) -> float:
    if len(risks) != len(budgets):
        raise ValueError("risks and budgets must have equal length")
    return sum(max(0.0, risk - budget) for risk, budget in zip(risks, budgets))


def proxy_regret(actual_losses: list[float], comparator_losses: list[float]) -> float:
    if len(actual_losses) != len(comparator_losses):
        raise ValueError("regret sequences must have equal length")
    return sum(actual_losses) - sum(comparator_losses)


def direct_request_conflict(requests) -> bool:
    return any(
        set(request.forget) & (set(request.protect) | {
            value for other in requests if other.request_id != request.request_id
            for value in other.protect
        })
        for request in requests
    )


class STREAM(BaseMethod):
    spec = MethodSpec("stream", "IDEA_04_STREAM", "Feasibility-aware streaming repair")

    def __init__(self, config: StreamConfig | None = None):
        self.config = config or StreamConfig()

    def prepare(self, ctx: PrepareContext) -> MethodState:
        requests = list(ctx.requests)
        return MethodState(values={
            "requests": requests,
            "feasible": not direct_request_conflict(requests),
            "dual": torch.zeros(len(requests)),
            "risks": [],
            "budgets": [request.risk_budget for request in requests],
            "proxy_losses": [],
            "comparator_losses": [],
        })

    def compute_loss(self, ctx: StepContext, state: MethodState) -> LossOutput:
        if ctx.logits is None or ctx.nll is None:
            raise ValueError("STREAM requires logits and NLL")
        n_forget = len(ctx.forget_rows)
        if n_forget == 0 or ctx.entity_ids is None:
            raise ValueError("STREAM requires forget rows and entity ids")
        logp = F.log_softmax(ctx.logits[:n_forget, -1, :], dim=-1)
        disclose = logp.gather(1, ctx.entity_ids.view(-1, 1)).squeeze(1)
        margins = -disclose
        margin_loss = margin_violation(margins, 1.0).mean()
        dual = state.values["dual"].to(margins.device)
        dual_weight = dual.mean() if dual.numel() else margins.new_zeros(())
        loss = ctx.nll.mean() + (self.config.lambda_margin + dual_weight) * margin_loss
        state.values["last_margin_loss"] = margin_loss.detach()
        return LossOutput(loss=loss, diagnostics={
            "nll": ctx.nll.mean().detach(),
            "stream_margin_loss": margin_loss.detach(),
            "stream_dual_mean": dual_weight.detach(),
            "disclosure_risk": torch.sigmoid(1.0 - margins).mean().detach(),
            "utility": torch.sigmoid(-ctx.nll.mean().detach()),
            "collapse_rate": torch.zeros((), device=loss.device),
        })

    def accept(self, candidate: Candidate, evidence: EvidenceBatch,
               state: MethodState) -> Decision:
        if not state.values["feasible"]:
            return infeasible("STREAM request constraint intersection is empty")
        budgets = state.values["budgets"]
        risk = torch.full((len(budgets),), evidence.disclosure_risk)
        budget = torch.tensor(budgets)
        state.values["dual"] = dual_update(
            state.values["dual"], risk, budget, self.config.dual_rate)
        state.values["risks"].append(evidence.disclosure_risk)
        state.values["proxy_losses"].append(float(state.values["last_margin_loss"].cpu()))
        if evidence.disclosure_risk > self.config.risk_budget:
            return rollback("STREAM current calibration risk exceeded; dual state retained")
        if evidence.utility < self.config.min_utility:
            return rollback("STREAM utility constraint violated")
        return accept("STREAM update accepted and history remains subject to revalidation")

    def finalize(self, state: MethodState) -> MethodDiagnostics:
        risks = state.values["risks"]
        budgets = state.values["budgets"]
        repeated_budgets = [budgets[min(i, len(budgets) - 1)] for i in range(len(risks))] if budgets else []
        comparator = state.values["comparator_losses"]
        regret = (proxy_regret(state.values["proxy_losses"], comparator)
                  if comparator else None)
        return MethodDiagnostics(values={
            "cumulative_violation": cumulative_violation(risks, repeated_budgets),
            "dual": state.values["dual"].tolist(),
            "proxy_regret": regret,
            "proxy_regret_status": "computed" if comparator else "undefined_without_comparator",
            "request_count": len(state.values["requests"]),
        })
