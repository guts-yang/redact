"""GRAPH (IDEA_07_GRAPH-1127): anchor-driven self-generated step supervision.

GPU-cluster entry: bridges runtime.Context -> core.graph_loss. All HF/PEFT imports are
lazy (inside main/setup). compute_loss consumes ONLY ctx tensors so it is
CPU-testable with a synthetic Context (see tests/selftest.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .core import GRAPH, GraphConfig, graph_loss


def setup(ctx, cfg):
    ctx.extras["method"] = GRAPH(GraphConfig())
    ctx.extras["graph"] = GRAPH().graph()


def compute_loss(ctx):
    m = ctx.extras.get("method") or GRAPH(GraphConfig())
    n, t = ctx.logits[:ctx.n_forget].shape[0], 4
    step_logits = ctx.logits[:ctx.n_forget, -t:, :]
    ref = step_logits.detach().clone()
    eids = ctx.entity_ids[:ctx.n_forget]
    g = torch.ones(n, t, device=step_logits.device)
    retain_kl = ctx.nll[ctx.n_forget:].mean() if ctx.nll.shape[0] > n else ctx.nll.new_zeros(())
    return m.loss(step_logits, ref, eids, g, retain_kl)


def build_config(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="unl1127_graph")
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
    rcfg = RunConfig(method="graph", run_id=a.run_id, steps=a.steps,
                     n_forget=a.n_forget, n_retain=a.n_retain, max_len=a.max_len,
                     lr=a.lr, seed=a.seed)
    run(rcfg, compute_loss, setup)


if __name__ == "__main__":
    main()
