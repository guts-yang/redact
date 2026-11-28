"""Generic run-scoped trainer. Proposal math remains in method plugins."""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field

from upbd_impl.contracts import (
    Candidate, DecisionKind, EvidenceBatch, PrepareContext, StepContext, Update,
)
from upbd_impl.registry import get_spec, load_method
from upbd_impl.runtime.artifacts import RunArtifacts
from upbd_impl.runtime.device import resolve_device
from upbd_impl.runtime.paths import Paths


@dataclass
class TrainConfig:
    method: str
    run_id: str
    device: str = "cpu"
    preset: str = "smoke"
    steps: int = 8
    n_forget: int = 2
    n_retain: int = 2
    n_aux: int = 2
    max_len: int = 256
    lr: float = 1e-4
    seed: int = 1001
    lora_r: int = 8
    grad_accum: int = 1
    holdout: int = 20
    split_mode: str = "entity"
    dtype: str = "fp32"
    tf32: bool = False
    max_minutes: float = 30.0
    save_merged: bool = False
    extra: dict = field(default_factory=dict)


_SAFE_TARGETS = (
    "I don't have enough reliable information to answer.\n</think>\n\nI don't know.",
    "I should not invent an answer without reliable support.\n</think>\n\nI don't know.",
    "The requested fact is outside what I can reliably provide.\n</think>\n\nI don't know.",
)


def _entity_token_ids(tokenizer, rows: list[dict], device) -> list[int]:
    result = []
    for row in rows:
        text = (row.get("gold_answer") or row.get("answer") or "").strip()
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        result.append(ids[0] if ids else 0)
    import torch
    return torch.tensor(result, dtype=torch.long, device=device)


def _evidence(diagnostics: dict[str, float], n: int) -> EvidenceBatch:
    def unit(name: str, default: float) -> float:
        return min(1.0, max(0.0, float(diagnostics.get(name, default))))

    risk = unit("disclosure_risk", 0.0)
    return EvidenceBatch(
        disclosure_risk=risk,
        utility=unit("utility", 1.0 - risk),
        collapse_rate=unit("collapse_rate", 0.0),
        structure_valid_rate=unit("structure_valid_rate", 1.0),
        n=n,
    )


def _deduplicate_params(params: list) -> list:
    seen: set[int] = set()
    result = []
    for param in params:
        if id(param) not in seen and getattr(param, "requires_grad", False):
            seen.add(id(param))
            result.append(param)
    return result


def run_train(cfg: TrainConfig) -> dict:
    import torch

    from upbd_impl.runtime import data as data_runtime
    from upbd_impl.runtime import model as model_runtime

    get_spec(cfg.method)
    if cfg.grad_accum <= 0:
        raise ValueError("grad_accum must be positive")
    paths = Paths.from_env()
    device = resolve_device(cfg.device)
    artifacts = RunArtifacts.create(paths, cfg.run_id)
    artifacts.write_json("train_args.json", asdict(cfg))

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32

    dataset = data_runtime.load_training_data(
        paths, cfg.holdout, split_mode=cfg.split_mode, seed=cfg.seed)
    if cfg.preset == "decisive" and not dataset.manifest.get("scientific_valid"):
        raise ValueError("decisive runs require an entity-disjoint split")
    artifacts.write_json("split_manifest.json", dataset.manifest)
    tokenizer = model_runtime.load_tokenizer(paths.target_model)
    model = model_runtime.load_causal_lm(
        paths.unlearned_model, device=cfg.device, dtype=cfg.dtype)
    model = model_runtime.add_lora(model, rank=cfg.lora_r)
    model.train()

    method = load_method(cfg.method)
    from upbd_impl.runtime.inputs import load_prepare_inputs, validate_decisive_inputs
    boundary, requests, extras = load_prepare_inputs(
        paths, hidden_dim=model.config.hidden_size, device=device)
    if cfg.preset == "decisive":
        validate_decisive_inputs(cfg.method, boundary, requests, extras)
    state = method.prepare(PrepareContext(
        boundary=boundary, requests=requests, model=model, tokenizer=tokenizer,
        device=str(device), extras=extras,
    ))

    base_params = [p for p in model.parameters() if p.requires_grad]
    extra_groups = method.parameter_groups(state)
    all_params = _deduplicate_params(
        base_params + [p for group in extra_groups for p in group.params])
    if not all_params:
        raise RuntimeError("no trainable parameters after method prepare")
    opt = torch.optim.AdamW(all_params, lr=cfg.lr, weight_decay=0.01)

    artifacts.write_text("train_log.jsonl", "")
    start_time = time.time()
    records: list[dict] = []
    rollback_count = 0
    stopped_reason = "max_steps"
    named_params = dict(model.named_parameters())

    for step in range(cfg.steps):
        rng = random.Random(cfg.seed + step)
        forget = [rng.choice(dataset.forget) for _ in range(cfg.n_forget)]
        retain = [rng.choice(dataset.retain) for _ in range(cfg.n_retain)]
        aux = [rng.choice(dataset.authors + dataset.world) for _ in range(cfg.n_aux)]
        items = [(row["question"], _SAFE_TARGETS[(step + i) % len(_SAFE_TARGETS)])
                 for i, row in enumerate(forget)]
        items += [(row["question"], data_runtime.behavior_target(row)) for row in retain + aux]
        batch = model_runtime.collate(tokenizer, items, cfg.max_len, cfg.device)
        out, logits, hidden, nll = model_runtime.forward(
            model, *batch, output_hidden_states=True)
        ctx = StepContext(
            model=model, tokenizer=tokenizer, batch=batch, logits=logits,
            hidden=hidden, nll=nll, forget_rows=forget, retain_rows=retain + aux,
            safe_texts=[target for _, target in items[:cfg.n_forget]],
            entity_ids=_entity_token_ids(tokenizer, forget, device),
            step=step, seed=cfg.seed, device=str(device),
        )
        output = method.compute_loss(ctx, state)
        raw_loss = output.loss
        if not isinstance(raw_loss, torch.Tensor) or raw_loss.dim() != 0:
            raise TypeError("method loss must be a scalar torch.Tensor")
        if not torch.isfinite(raw_loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        (raw_loss / cfg.grad_accum).backward()
        diagnostics = output.float_diagnostics()

        decision = None
        optimizer_boundary = ((step + 1) % cfg.grad_accum == 0
                              or step + 1 == cfg.steps)
        if optimizer_boundary:
            update = Update(gradients={
                name: param.grad for name, param in named_params.items()
                if param.grad is not None
            })
            update = method.constrain_update(update, state)
            for name, gradient in update.gradients.items():
                if name in named_params:
                    named_params[name].grad = gradient

            decision = method.accept(
                Candidate(cfg.run_id, step, cfg.seed),
                _evidence(diagnostics, cfg.n_forget + cfg.n_retain + cfg.n_aux),
                state,
            )
            if decision.kind == DecisionKind.ACCEPT:
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step()
            else:
                rollback_count += 1
            opt.zero_grad(set_to_none=True)

        record = {
            "step": step + 1,
            "loss": float(raw_loss.detach().cpu()),
            "decision": decision.kind.value if decision else "ACCUMULATE",
            "decision_reason": decision.reason if decision else "gradient accumulation",
            "diagnostics": diagnostics,
            "elapsed_sec": round(time.time() - start_time, 2),
        }
        if device.type == "cuda":
            record["peak_vram_mb"] = round(
                torch.cuda.max_memory_allocated() / 1024**2, 1)
        records.append(record)
        with artifacts.path("train_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if decision and decision.kind == DecisionKind.INFEASIBLE:
            stopped_reason = "infeasible"
            break
        if (time.time() - start_time) / 60 > cfg.max_minutes:
            stopped_reason = "max_minutes"
            break

    adapter = artifacts.path("adapter")
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    if cfg.save_merged:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(artifacts.path("merged")), safe_serialization=True)
        tokenizer.save_pretrained(str(artifacts.path("merged")))

    final = method.finalize(state).values
    summary = {
        "status": "ok",
        "method": cfg.method,
        "run_id": cfg.run_id,
        "steps_attempted": len(records),
        "rollback_count": rollback_count,
        "stopped_reason": stopped_reason,
        "elapsed_sec": round(time.time() - start_time, 2),
        "adapter": str(adapter),
        "final_diagnostics": records[-1]["diagnostics"] if records else {},
        "method_diagnostics": final,
        "train_args": asdict(cfg),
        "note": "implementation smoke/diagnostic only; no scientific claim",
    }
    artifacts.write_summary(summary)
    return summary
