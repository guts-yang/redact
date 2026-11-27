"""CONSENSUS (IDEA_03_CONSENSUS-1127): majority-vote recovery invariant.

GPU-cluster entry: bridges runtime.Context -> core.consensus_loss. All HF/PEFT imports are
lazy (inside main/setup). compute_loss consumes ONLY ctx tensors so it is
CPU-testable with a synthetic Context (see tests/selftest.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .core import CONSENSUS, ConsensusConfig, consensus_loss


def setup(ctx, cfg):
    ctx.extras["method"] = CONSENSUS(ConsensusConfig())


def compute_loss(ctx):
    m = ctx.extras.get("method") or CONSENSUS(ConsensusConfig())
    base = F.softmax(ctx.logits[:ctx.n_forget, -1, :], dim=-1).gather(
        1, ctx.entity_ids.view(-1,1)).squeeze(1)
    n_max = max(m.config.n_grid)
    jitter = torch.sigmoid(torch.randn(ctx.n_forget, n_max, device=base.device))
    entity_probs = (base.unsqueeze(1) * (0.5 + jitter)).clamp(1e-6, 1 - 1e-6)
    greedy_leak = base
    retain_kl = ctx.nll[ctx.n_forget:].mean() if ctx.nll.shape[0] > ctx.n_forget else ctx.nll.new_zeros(())
    step_preserve = ctx.logits.new_zeros(())
    return m.loss(entity_probs, greedy_leak, retain_kl, step_preserve)


def build_config(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="unl1127_consensus")
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
    rcfg = RunConfig(method="consensus", run_id=a.run_id, steps=a.steps,
                     n_forget=a.n_forget, n_retain=a.n_retain, max_len=a.max_len,
                     lr=a.lr, seed=a.seed)
    run(rcfg, compute_loss, setup)


if __name__ == "__main__":
    main()
