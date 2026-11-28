"""BLESS: margin-budgeted repair under declared benign operator families."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from upbd_impl.contracts import (
    BaseMethod, Candidate, Decision, EvidenceBatch, LossOutput, MethodSpec,
    MethodState, PrepareContext, StepContext,
)
from upbd_impl.core.decisions import accept, rollback, unknown
from upbd_impl.core.margins import margin_violation, robust_upper_bound


@dataclass
class BLESSConfig:
    margin_threshold: float = 1.0
    reserve_per_step: float = 0.05
    composition_depth: int = 4
    lambda_margin: float = 0.1
    risk_budget: float = 0.05
    min_utility: float = 0.0


def margin_consumption(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    if before.shape != after.shape:
        raise ValueError("before/after margins must have identical shapes")
    return before - after


def required_margin(threshold: float, per_step_consumption: float, depth: int) -> float:
    if depth < 0 or per_step_consumption < 0:
        raise ValueError("depth and consumption must be non-negative")
    return threshold + depth * per_step_consumption


def compose_margin(initial: float, consumptions: list[float]) -> float:
    if any(value < 0 for value in consumptions):
        raise ValueError("this local bound requires non-negative consumption")
    return initial - sum(consumptions)


class BLESS(BaseMethod):
    spec = MethodSpec("bless", "IDEA_02_BLESS", "Benign-learning-stable repair")

    def __init__(self, config: BLESSConfig | None = None):
        self.config = config or BLESSConfig()

    def prepare(self, ctx: PrepareContext) -> MethodState:
        drops = ctx.extras.get("benign_operator_drops")
        return MethodState(values={
            "operator_drops": drops,
            "outside_operator_family": bool(ctx.extras.get("outside_operator_family", False)),
        })

    def compute_loss(self, ctx: StepContext, state: MethodState) -> LossOutput:
        if ctx.logits is None or ctx.nll is None:
            raise ValueError("BLESS requires logits and NLL")
        n_forget = len(ctx.forget_rows)
        if n_forget == 0 or ctx.entity_ids is None:
            raise ValueError("BLESS requires forget rows and entity ids")
        logp = F.log_softmax(ctx.logits[:n_forget, -1, :], dim=-1)
        disclose = logp.gather(1, ctx.entity_ids.view(-1, 1)).squeeze(1)
        margins = -disclose

        drops = state.values.get("operator_drops")
        if drops is None:
            robust_drop = margins.new_zeros(())
        else:
            drops = torch.as_tensor(drops, dtype=margins.dtype, device=margins.device)
            robust_drop = robust_upper_bound(drops)
        reserve = required_margin(
            self.config.margin_threshold,
            float(robust_drop.detach().cpu()) + self.config.reserve_per_step,
            self.config.composition_depth,
        )
        margin_loss = margin_violation(margins, self.config.margin_threshold).mean()
        reserve_loss = margin_violation(margins, reserve).mean()
        loss = ctx.nll.mean() + self.config.lambda_margin * (margin_loss + reserve_loss)
        state.values["last_required_margin"] = reserve
        state.values["last_worst_margin"] = float(margins.detach().min().cpu())
        return LossOutput(loss=loss, diagnostics={
            "nll": ctx.nll.mean().detach(),
            "bless_margin_loss": margin_loss.detach(),
            "bless_reserve_loss": reserve_loss.detach(),
            "bless_robust_drop": robust_drop.detach(),
            "bless_required_margin": torch.tensor(reserve, device=margins.device),
            "disclosure_risk": torch.sigmoid(self.config.margin_threshold - margins).mean().detach(),
            "utility": torch.sigmoid(-ctx.nll.mean().detach()),
            "collapse_rate": torch.zeros((), device=loss.device),
        })

    def accept(self, candidate: Candidate, evidence: EvidenceBatch,
               state: MethodState) -> Decision:
        if state.values.get("outside_operator_family"):
            return unknown("operator is outside the declared BLESS family")
        required = state.values.get("last_required_margin", self.config.margin_threshold)
        worst = state.values.get("last_worst_margin", float("inf"))
        if worst < required:
            return rollback("BLESS composition margin reserve is insufficient")
        if evidence.disclosure_risk > self.config.risk_budget:
            return rollback("BLESS calibration disclosure risk exceeded")
        if evidence.utility < self.config.min_utility:
            return rollback("BLESS utility constraint violated")
        return accept("BLESS declared-family margin satisfied")
