"""FAITH (IDEA_06_FAITH-1127): faithfulness-gated answer-level erasure.

GPU-cluster entry: bridges runtime.Context -> core.faith_gated_loss. All HF/PEFT imports are
lazy (inside main/setup). compute_loss consumes ONLY ctx tensors so it is
CPU-testable with a synthetic Context (see tests/selftest.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .core import FAITH, FaithConfig, faith_gated_loss


def setup(ctx, cfg):
    ctx.extras["method"] = FAITH(FaithConfig(eta=0.3))


def compute_loss(ctx):
    m = ctx.extras.get("method") or FAITH(FaithConfig())
    if ctx.hidden is not None:
        step_hidden = ctx.hidden[:ctx.n_forget].reshape(-1, ctx.hidden.shape[-1])[:8]
    else:
        step_hidden = torch.zeros(8, 16, device=ctx.logits.device, requires_grad=True)
    h = step_hidden.shape[-1]
    v_safe = torch.zeros(h, device=step_hidden.device)
    contributions = torch.linspace(0, 1, step_hidden.shape[0], device=step_hidden.device)
    # Faith from control-patch answer distributions (detached gate)
    lp = F.log_softmax(ctx.logits[:ctx.n_forget, -1, :], dim=-1)
    faith = m.faith(lp.detach(), lp.detach().roll(1, dims=-1))  # [N] -> broadcast below
    faith = faith.mean().expand(step_hidden.shape[0]).detach()
    ans_logp = F.log_softmax(ctx.logits[:ctx.n_forget,-1,:],dim=-1).gather(
        1, ctx.entity_ids.view(-1,1)).squeeze(1).mean()
    retain_kl = ctx.nll[ctx.n_forget:].mean() if ctx.nll.shape[0] > ctx.n_forget else ctx.nll.new_zeros(())
    return m.loss(step_hidden, v_safe, contributions, faith, ans_logp, retain_kl)


def build_config(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="unl1127_faith")
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
    rcfg = RunConfig(method="faith", run_id=a.run_id, steps=a.steps,
                     n_forget=a.n_forget, n_retain=a.n_retain, max_len=a.max_len,
                     lr=a.lr, seed=a.seed)
    run(rcfg, compute_loss, setup)


if __name__ == "__main__":
    main()
