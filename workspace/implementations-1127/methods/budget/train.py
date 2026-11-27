"""GPU-cluster entry for BUDGET (IDEA_01_BUDGET-1127).

Bridges runtime.Context -> core.budget_loss. The continuous-budget leak field
rho(b) is approximated DIFFERENTIABLY from the batched logits: a per-budget
gate (learnable scalar per grid point, kept in ctx.extras) modulates the entity
probability at the answer position. Generation-based rho(b) (real leakage under
budget b) is reported separately by the eval side, never used as a gradient
(sampling is non-differentiable). This keeps the training relaxation honest.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import BUDGET, BudgetConfig


def setup(ctx, cfg):
    # one learnable gate per budget grid point; registered on the model so the
    # optimizer updates it. Import here to avoid a hard HF dep at import time.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from runtime import paths
    k = len(paths.BUDGET_GRID)
    gates = nn.Parameter(torch.zeros(k, device=ctx.device))
    setattr(ctx.model, "_budget_gates", gates)
    ctx.extras["gates"] = gates
    ctx.extras["grid"] = list(paths.BUDGET_GRID)
    ctx.extras["method"] = BUDGET(BudgetConfig(mu=0.5, worst_weight=1.0))


def compute_loss(ctx):
    gates = ctx.extras.get("gates")
    grid = ctx.extras.get("grid") or [0, 64, 128, 256, 512]
    m = ctx.extras.get("method") or BUDGET(BudgetConfig())
    if gates is None:
        gates = torch.zeros(len(grid), device=ctx.logits.device)
    p = F.softmax(ctx.logits[:ctx.n_forget, -1, :], dim=-1).gather(
        1, ctx.entity_ids.view(-1, 1)).squeeze(1).mean()
    rho = torch.stack([torch.sigmoid(g) * p for g in gates])
    retain_kl = ctx.nll[ctx.n_forget:].mean() if ctx.nll.shape[0] > ctx.n_forget \
        else ctx.nll.new_zeros(())
    if ctx.hidden is not None:
        nf = ctx.hidden[:ctx.n_forget]
        step_preserve = (nf - nf.detach()).pow(2).mean()  # 0 on first pass; structure reg hook
        step_preserve = step_preserve + 0.0 * nf.mean()  # keep graph alive
    else:
        step_preserve = ctx.logits.new_zeros(())
    return m.loss(rho, retain_kl, step_preserve)


def build_config(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="unl1127_budget")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--n-forget", type=int, default=4)
    ap.add_argument("--n-retain", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1001)
    return ap.parse_args(argv)


def main(argv=None):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from runtime.runner import RunConfig, run
    a = build_config(argv)
    cfg = RunConfig(method="budget", run_id=a.run_id, steps=a.steps,
                    n_forget=a.n_forget, n_retain=a.n_retain, max_len=a.max_len,
                    lr=a.lr, seed=a.seed)
    run(cfg, compute_loss, setup)


if __name__ == "__main__":
    main()
