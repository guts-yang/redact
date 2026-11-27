"""Generic multi-step training loop shared by all ten methods.

A method supplies a ``compute_loss(ctx)`` callable that returns a
``MethodOutput``-like object with ``.loss`` and ``.float_diagnostics()``.
The runner owns: model/tokenizer/LoRA, data sampling, optimizer, seed,
grad accumulation, adapter saving, and a machine-readable summary.

Everything HF-specific is lazy-imported via runtime.model, so importing this
module on a CPU-only laptop works fine.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from . import data, model as M, paths


@dataclass
class Context:
    """Everything a method's compute_loss may need for one optimizer step."""
    model: Any
    ref_model: Any
    tok: Any
    forget_rows: List[dict]
    retain_rows: List[dict]
    safe_texts: List[str]
    batch: Any                     # (ids, labels, mask)
    logits: Any                    # [B,L,V]
    hidden: Any                    # [B,L,H] or None
    nll: Any                       # [B]
    n_forget: int
    entity_ids: Any                # LongTensor [n_forget]
    seed: int
    step: int
    max_len: int
    device: str = "cuda"
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    method: str
    run_id: str = ""
    steps: int = 8
    grad_accum: int = 1
    n_forget: int = 4
    n_retain: int = 4
    max_len: int = 256
    lr: float = 1e-4
    seed: int = paths.SEED
    lora_r: int = 8
    lora_alpha: int = 32
    attn_implementation: str = "sdpa"
    need_ref_model: bool = False
    task_id: str = "1"
    output_hidden: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


def make_batch(tok, forget_rows, retain_rows, safe_texts, max_len):
    from .data import build_prompt
    examples = [M.encode_example(tok, build_prompt(r["question"]), t, max_len)
                for r, t in zip(forget_rows, safe_texts)]
    examples += [M.encode_example(tok, build_prompt(r["question"]), r["answer"], max_len)
                 for r in retain_rows]
    return M.collate(tok, examples, max_len)


def run(cfg: RunConfig, compute_loss: Callable[[Context], Any],
        setup: Optional[Callable[[Context, RunConfig], None]] = None):
    """Execute a full diagnostic training run. Returns the summary dict."""
    paths.ensure_dirs()
    run_id = cfg.run_id or f"unl1127_{cfg.method}"
    out_dir = paths.RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    tok, model = M.load_model(attn_implementation=cfg.attn_implementation)
    model = M.add_lora(model, r=cfg.lora_r, alpha=cfg.lora_alpha)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr)

    ref_model = M.load_unlearned_reference() if cfg.need_ref_model else None
    refusals = data.refusals()
    base_ctx = dict(model=model, ref_model=ref_model, tok=tok)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if setup is not None:
        # Allow method-specific precomputation (retain stats, graphs, probes).
        dummy = Context(**base_ctx, forget_rows=[], retain_rows=[], safe_texts=[],
                        batch=None, logits=None, hidden=None, nll=None,
                        n_forget=0, entity_ids=None, seed=cfg.seed, step=-1,
                        max_len=cfg.max_len, device=device)
        setup(dummy, cfg)

    losses: List[float] = []
    diag_last: Dict[str, float] = {}
    meta: Dict[str, Any] = {"method": cfg.method, "hook": "full"}
    t0 = time.time()
    for step in range(cfg.steps):
        f_rows = data.sample_rows(paths.FORGET_FILE, cfg.n_forget, cfg.seed + step,
                                  cfg.task_id)
        r_rows = data.sample_rows(paths.RETAIN_FILE, cfg.n_retain, cfg.seed + step, None)
        safe = [refusals[i % len(refusals)] for i in range(len(f_rows))]
        batch = make_batch(tok, f_rows, r_rows, safe, cfg.max_len)
        logits, hidden, nll = M.forward(model, *batch,
                                        output_hidden_states=cfg.output_hidden)
        eids = data.entity_token_ids(tok, f_rows).to(logits.device)
        ctx = Context(**base_ctx, forget_rows=f_rows, retain_rows=r_rows,
                      safe_texts=safe, batch=batch, logits=logits, hidden=hidden,
                      nll=nll, n_forget=len(f_rows), entity_ids=eids,
                      seed=cfg.seed, step=step, max_len=cfg.max_len,
                      device=device)
        out = compute_loss(ctx)
        loss = out.loss / max(1, cfg.grad_accum)
        loss.backward()
        if (step + 1) % cfg.grad_accum == 0 or step + 1 == cfg.steps:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss.detach().cpu()))
        try:
            diag_last = out.float_diagnostics()
        except Exception:
            diag_last = {}
        print(f"[{run_id}] step {step+1}/{cfg.steps} loss={losses[-1]:.4f}", flush=True)

    adapter = out_dir / "adapter"
    model.save_pretrained(str(adapter))
    tok.save_pretrained(str(adapter))
    summary = {
        "diagnostic_only": True,
        "claim_level": "non-claim",
        "method": cfg.method,
        "run_id": run_id,
        "adapter": str(adapter),
        "steps": cfg.steps,
        "grad_accum": cfg.grad_accum,
        "n_forget": cfg.n_forget,
        "n_retain": cfg.n_retain,
        "max_len": cfg.max_len,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "losses": losses,
        "final_diagnostics": diag_last,
        "meta": meta,
        "elapsed_sec": round(time.time() - t0, 1),
        "note": "1127 reproduction diagnostic; not convergence, not a claim, "
                "no ARIS gate advanced",
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "train_args.json").write_text(
        json.dumps({k: v for k, v in vars(cfg).items() if k != "extra"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary
