"""Reference UPBD behavior repair: ordinary causal-LM SFT only."""
from __future__ import annotations

import torch

from upbd_impl.contracts import BaseMethod, LossOutput, MethodSpec, MethodState, StepContext


def reference_sft_loss(nll: torch.Tensor) -> LossOutput:
    loss = nll.mean()
    utility = torch.sigmoid(-loss.detach())
    return LossOutput(loss=loss, diagnostics={
        "nll": loss.detach(),
        "utility": utility,
        "disclosure_risk": torch.zeros((), device=nll.device),
        "collapse_rate": torch.zeros((), device=nll.device),
    })


class UPBDBase(BaseMethod):
    spec = MethodSpec("upbd_base", "reference", "Legacy UPBD behavior repair control")

    def compute_loss(self, ctx: StepContext, state: MethodState) -> LossOutput:
        if ctx.nll is None:
            raise ValueError("UPBD base requires per-example NLL")
        return reference_sft_loss(ctx.nll)
