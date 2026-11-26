#!/usr/bin/env python3
"""UPBD pilot: repair R2MU utility with channel-conditioned SFT.

Student starts from the frozen R2MU forget10 boundary. Forget prompts learn a
safe-exit target; retain/aux prompts learn the target model's behavior. Exact
leakage must stay closed while MU is repaired.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/usr/local/workspace/tracks/lrm-unlearning/run-20261125/checkpoints/r2mu-forget10-compat"
TOKENIZER = "/jizhicfs/chenyliao/models/sangyon/LRM-target"
TARGET_RUN = Path("/usr/local/workspace/tracks/lrm-unlearning/run-20261125-full/results")
SAFE_EXIT = "I don't know.\n</think>\n\nI don't know."


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def behavior_target(row: dict) -> str:
    cot = (row.get("model_cot") or "").strip()
    ans = (row.get("response") or "").strip()
    if not ans:
        ans = (row.get("gold_answer") or "").strip()
    if cot:
        return cot + "\n</think>\n\n" + ans
    return ans


def encode(tok, prompt: str, target: str, max_len: int = 1024):
    p = tok(prompt, add_special_tokens=True)["input_ids"]
    a = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(a) >= max_len:
        a = a[: max_len - 1]
    if len(p) + len(a) > max_len:
        p = p[-(max_len - len(a)):]
    return p + a, [-100] * len(p) + a


def collate(tok, items, max_len=1024):
    ids, labels = [], []
    for prompt, target in items:
        i, l = encode(tok, prompt, target, max_len)
        ids.append(i)
        labels.append(l)
    width = max(len(x) for x in ids)
    pad = tok.pad_token_id
    input_ids, attn, labs = [], [], []
    for i, l in zip(ids, labels):
        n = width - len(i)
        input_ids.append(i + [pad] * n)
        attn.append([1] * len(i) + [0] * n)
        labs.append(l + [-100] * n)
    return (
        torch.tensor(input_ids, dtype=torch.long, device="cuda"),
        torch.tensor(attn, dtype=torch.long, device="cuda"),
        torch.tensor(labs, dtype=torch.long, device="cuda"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--holdout", type=int, default=20,
                    help="Skip the first N rows per split; they are the held-out eval set.")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--output-dir", type=Path, default=Path("results/upbd_adapter"))
    ap.add_argument("--save-merged", type=Path, default=Path("/jizhicfs/chenyliao/lrm-unlearning/results/upbd/merged"))
    ap.add_argument("--skip-merge", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    forget_all = read_csv(TARGET_RUN / "m1_r2mu-forget10-fp32_forget10.csv")
    retain_all = read_csv(TARGET_RUN / "m1_target-fp32_forget10_retain.csv")
    authors_all = read_csv(TARGET_RUN / "m1_target-fp32_real_authors.csv")
    world_all = read_csv(TARGET_RUN / "m1_target-fp32_world_facts.csv")

    # The first N rows of each split are the held-out 20-row eval set. Never
    # train on them; otherwise the pilot measures memorization, not unlearning.
    forget = forget_all[args.holdout:]
    retain = retain_all[args.holdout:]
    authors = authors_all[args.holdout:]
    world = world_all[args.holdout:]

    items: list[tuple[str, str, str]] = []
    for r in forget:
        items.append((r["question"], SAFE_EXIT, "forget"))
    for r in retain:
        items.append((r["question"], behavior_target(r), "retain"))
    for r in authors + world:
        items.append((r["question"], behavior_target(r), "aux"))

    random.shuffle(items)
    print(f"[upbd] forget={len(forget)} retain={len(retain)} aux={len(authors)+len(world)} total={len(items)}", flush=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("cuda")
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    model.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    t0 = time.time()
    step = 0
    losses = []
    stopped = False

    for epoch in range(args.epochs):
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            ids, attn, labels = collate(tok, [(p, t) for p, t, _ in batch], args.max_len)
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()
            if (start // args.batch_size + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(out.loss.detach().cpu()),
                    "elapsed_sec": round(time.time() - t0, 1),
                    "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                }
                losses.append(rec)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                if step % 10 == 0:
                    print(f"[upbd] {rec}", flush=True)
                if args.max_steps > 0 and step >= args.max_steps:
                    stopped = True
                    break
                if (time.time() - t0) / 60 > args.max_minutes:
                    stopped = True
                    break
            if stopped:
                break
        if stopped:
            break

    if step % args.grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    print(f"[upbd] saved adapter {adapter_dir}", flush=True)
    if not args.skip_merge:
        args.save_merged.mkdir(parents=True, exist_ok=True)
        merged = model.merge_and_unload()
        merged.save_pretrained(str(args.save_merged), safe_serialization=True)
        tok.save_pretrained(str(args.save_merged))
        print(f"[upbd] saved merged {args.save_merged}", flush=True)

    summary = {
        "status": "ok",
        "steps": step,
        "stopped_early": stopped,
        "elapsed_sec": round(time.time() - t0, 1),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
