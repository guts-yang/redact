"""PIVOT (IDEA_02_PIVOT-1127): de-pivot cross-lingual CoT step-level scope.

GPU-cluster entry: bridges runtime.Context -> core.pivot_loss. All HF/PEFT imports are
lazy (inside main/setup). compute_loss consumes ONLY ctx tensors so it is
CPU-testable with a synthetic Context (see tests/selftest.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .core import PIVOT, PivotConfig, pivot_loss


def setup(ctx, cfg):
    ctx.extras["method"] = PIVOT(PivotConfig(gamma=0.5))


def compute_loss(ctx):
    m = ctx.extras.get("method") or PIVOT(PivotConfig())
    n, t, v = ctx.logits[:ctx.n_forget].shape[0], 4, ctx.logits.shape[-1]
    # step logits: slice last T positions of the forget-row logits [N,T,V]
    step_logits = ctx.logits[:ctx.n_forget, -t:, :]
    ref_step_logits = step_logits.detach().clone()
    eids = ctx.entity_ids[:ctx.n_forget]
    g = torch.ones(n, t, device=step_logits.device)  # all self-generated steps in-scope
    retain_kl = ctx.nll[ctx.n_forget:].mean() if ctx.nll.shape[0] > n else ctx.nll.new_zeros(())
    # shared mid-layer reps proxy: mean over sequence of hidden states
    if ctx.hidden is not None:
        sh = ctx.hidden[:ctx.n_forget].mean(1)
        ref_sh = sh.detach().clone()
    else:
        sh = torch.zeros(n, 8, device=step_logits.device); ref_sh = sh.detach()
    # step_leak proxy [Q,P,T] = 1 (query-lang) x 2 (pivots) x T
    p = F.softmax(step_logits, dim=-1).gather(-1, eids.view(-1,1,1).expand(-1,t,1)).squeeze(-1)
    step_leak = p.mean(0).view(1, 1, t).expand(1, 2, t)
    return m.loss(step_logits, ref_step_logits, eids, g, retain_kl, sh, ref_sh,
                  step_leak=step_leak)


def build_config(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="unl1127_pivot")
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
    rcfg = RunConfig(method="pivot", run_id=a.run_id, steps=a.steps,
                     n_forget=a.n_forget, n_retain=a.n_retain, max_len=a.max_len,
                     lr=a.lr, seed=a.seed)
    run(rcfg, compute_loss, setup)


if __name__ == "__main__":
    main()
