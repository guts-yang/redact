"""R-TOFU full-protocol training structure for the 1127 runtime (opt-in only).

Structural scope
----------------
The live 1127 runner (``runtime/runner.py::make_batch``) supervises a short
refusal from ``safe_texts`` and never passes a decoding strategy, never uses the
dataset ``cot``, and never emits ``</think>`` in the target. That is the
answer-only protocol, and its generations had to be normalised on the eval side.

This module supplies the missing R-TOFU v2 structure as a SELF-CONTAINED,
OPT-IN layer, mirroring ``R-TOFU/workspace/code/training_data.py``:

  * ``load_safe_cot`` / ``attach_safe_cot`` -- reasoned IDK CoT + answer,
    joined by normalised question, with honest coverage reporting;
  * ``safe_target`` -- ``DefaultCoT`` supervises ``<CoT> </think>\n\n <answer>``
    (the closing tag is ALWAYS present, even when the safe CoT is missing),
    prefilled strategies supervise the safe answer only because their prompt
    already closed ``</think>``;
  * ``encode_strategy_example`` / ``build_strategy_batches`` -- one batch per
    four strategies, forget rows use safe targets, retain rows use gold answers;
  * ``step_spans`` / ``collect_step_hidden`` -- exact CoT sentence token spans
    so step-level methods can pool hidden states from the SAME forward pass
    instead of a second GPU forward.

HONESTY BOUNDARY
----------------
This module is NOT wired into ``runtime.runner`` and starts no GPU work. The
existing 1127 runs/eval chain still use the answer-only protocol and remain
``diagnostic_only / non-claim``; T0-T3 decisive experiments are not implemented
here. Switching the live training path requires a separate authorised change
(and retraining), not an import of this module. CPU smoke only, no HF/PEFT
imports at module import time.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import data, paths

STRATEGY_ORDER: Tuple[str, ...] = tuple(paths.STRATEGIES)
THINK_CLOSE = "</think>\n\n"
DEFAULT_SAFE_ANSWER = "I don't have sufficient information to answer that."
DEFAULT_SAFE_COT = Path(paths.env(
    "SAFE_COT_FILE",
    str(paths.REDACT_ROOT / "data" / "reasoned_idk" / "reasoned_idk_forget10.json"),
))


def _norm_question(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _clean_cot(cot: str) -> str:
    """Strip stray think closers so the target has exactly one ``</think>``."""
    return re.sub(r"</think>", "", (cot or "").strip()).strip()


def _safe_answer(row: dict) -> str:
    return (row.get("safe_answer") or DEFAULT_SAFE_ANSWER).strip() or DEFAULT_SAFE_ANSWER


def _tok_ids(tok, text: str, add_special_tokens: bool = False) -> List[int]:
    return list(tok(text, add_special_tokens=add_special_tokens)["input_ids"])


def _check_strategy(strategy: str) -> None:
    if strategy not in STRATEGY_ORDER:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {list(STRATEGY_ORDER)}")


# ---------------------------------------------------------------------------
# Safe CoT data layer
# ---------------------------------------------------------------------------

def load_safe_cot(path: Optional[str | Path] = None) -> Dict[str, dict]:
    """Load reasoned-IDK rows keyed by normalised question (stdlib only)."""
    p = Path(path) if path is not None else DEFAULT_SAFE_COT
    rows = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, dict] = {}
    for row in rows:
        key = _norm_question(row.get("question", ""))
        if key:
            out[key] = row
    return out


def attach_safe_cot(
    rows: Sequence[dict],
    path: Optional[str | Path] = None,
    strict: bool = False,
) -> List[dict]:
    """Attach ``safe_cot`` / ``safe_answer`` / ``safe_cot_matched`` per row.

    Unmatched questions get an empty CoT and the default safe answer. They are
    flagged, never silently counted as matched: the shipped forget10 file covers
    394/400 task_id=1 questions. ``strict=True`` turns any miss into an error so
    a caller must consciously accept the 6 fallbacks.
    """
    safe = load_safe_cot(path)
    out: List[dict] = []
    for row in rows:
        item = dict(row)
        match = safe.get(_norm_question(item.get("question", "")))
        if match:
            item["safe_cot"] = _clean_cot(match.get("reasoned_idk_cot", ""))
            item["safe_answer"] = (
                (match.get("reasoned_idk_answer") or DEFAULT_SAFE_ANSWER).strip()
                or DEFAULT_SAFE_ANSWER
            )
            item["safe_cot_matched"] = True
        else:
            item["safe_cot"] = ""
            item["safe_answer"] = DEFAULT_SAFE_ANSWER
            item["safe_cot_matched"] = False
        out.append(item)
    if strict and any(not r["safe_cot_matched"] for r in out):
        missing = sum(1 for r in out if not r["safe_cot_matched"])
        raise ValueError(
            f"safe-CoT coverage incomplete: {missing}/{len(out)} questions unmatched "
            f"in {Path(path) if path is not None else DEFAULT_SAFE_COT}"
        )
    return out


def safe_cot_coverage(rows: Sequence[dict]) -> Dict[str, Any]:
    """Report matched/missing counts without hiding unmatched rows."""
    total = len(rows)
    matched = sum(1 for r in rows if bool(r.get("safe_cot_matched")))
    return {
        "total": total,
        "matched": matched,
        "missing": total - matched,
        "matched_fraction": (matched / total) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def safe_target(row: dict, strategy: str) -> str:
    """Supervised continuation for a forget row under one decoding strategy."""
    _check_strategy(strategy)
    answer = _safe_answer(row)
    if strategy != "DefaultCoT":
        return answer
    cot = _clean_cot(row.get("safe_cot", ""))
    return f"{cot}\n{THINK_CLOSE}{answer}" if cot else f"{THINK_CLOSE}{answer}"


def cot_steps(row: dict, strategy: str, max_steps: int = 24,
              min_chars: int = 20) -> List[str]:
    """CoT sentences whose token spans will be pooled (empty for prefilled)."""
    _check_strategy(strategy)
    if strategy != "DefaultCoT":
        return []
    return data.split_cot_steps(_clean_cot(row.get("safe_cot", "")), max_steps, min_chars)


# ---------------------------------------------------------------------------
# Encoding + exact step spans
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EncodedExample:
    strategy: str
    role: str                              # "forget" | "retain"
    ids: Tuple[int, ...]
    labels: Tuple[int, ...]
    mask: Tuple[int, ...]
    prompt_len: int
    answer_start: int                     # first token after the CoT region
    steps: Tuple[str, ...]
    spans: Tuple[Tuple[int, int], ...]     # exact CoT token spans in ``ids``
    target_text: str                       # text actually supervised

    @property
    def target_len(self) -> int:
        return len(self.ids) - self.prompt_len


def encode_strategy_example(
    tok,
    row: dict,
    strategy: str,
    max_len: int,
    role: str = "forget",
    max_steps: int = 24,
    min_chars: int = 20,
) -> EncodedExample:
    """Encode prompt + target, keeping the answer-bearing tail intact.

    Unlike ``runtime.model.encode_example`` this adds EOS, reserves the
    ``</think>\n\n<answer>`` tail before spending tokens on CoT, and records
    exact step spans by construction (the ids are built from the same step token
    lists that define the spans). No model / GPU is touched here.
    """
    _check_strategy(strategy)
    if role not in ("forget", "retain"):
        raise ValueError(f"unknown role {role!r}; choose 'forget' or 'retain'")
    if max_len < 8:
        raise ValueError(f"max_len={max_len} too small for a full-protocol example")

    prompt = data.build_prompt(row["question"], strategy)
    p_ids = _tok_ids(tok, prompt, add_special_tokens=True)

    if role == "retain":
        # Structural parity with R-TOFU training_data.build_strategy_batches:
        # retain rows supervise the gold answer only, with no CoT and no closer,
        # for every strategy. Changing that is a protocol decision, not a bugfix.
        steps: List[str] = []
        tail_text = (row.get("answer") or "").strip()
    elif strategy == "DefaultCoT":
        steps = cot_steps(row, strategy, max_steps, min_chars)
        tail_text = THINK_CLOSE + _safe_answer(row)
    else:
        steps = []
        tail_text = _safe_answer(row)

    tail_ids = _tok_ids(tok, tail_text)
    if getattr(tok, "eos_token_id", None) is not None:
        tail_ids.append(int(tok.eos_token_id))
    # Never let a long CoT or a long prompt evict the closing tag + answer.
    if len(tail_ids) >= max_len:
        tail_ids = tail_ids[-(max_len - 1):]
    prompt_budget = max_len - len(tail_ids)
    if len(p_ids) > prompt_budget:
        p_ids = p_ids[-prompt_budget:]

    ids = list(p_ids)
    spans: List[Tuple[int, int]] = []
    for step in steps:
        s_ids = _tok_ids(tok, step)
        if not s_ids or len(ids) + len(s_ids) + len(tail_ids) > max_len:
            break
        start = len(ids)
        ids.extend(s_ids)
        spans.append((start, len(ids)))
    answer_start = len(ids)
    ids.extend(tail_ids)

    labels = [-100] * len(p_ids) + ids[len(p_ids):]
    mask = [1] * len(ids)
    cot_text = " ".join(steps[:len(spans)])
    # ``tail_text`` already starts with THINK_CLOSE for DefaultCoT; do not add
    # a second closer. For prefilled strategies / retain rows it is just the
    # answer, and the closer lives in the prompt.
    target_text = (cot_text + "\n" + tail_text) if cot_text else tail_text
    return EncodedExample(
        strategy=strategy, role=role,
        ids=tuple(ids), labels=tuple(labels), mask=tuple(mask),
        prompt_len=len(p_ids), answer_start=answer_start,
        steps=tuple(steps[:len(spans)]), spans=tuple(spans),
        target_text=target_text,
    )


def step_spans(example: EncodedExample) -> Tuple[Tuple[int, int], ...]:
    """Validate and return finite, ordered, in-bounds CoT token spans."""
    n = len(example.ids)
    prev = example.prompt_len
    out: List[Tuple[int, int]] = []
    for start, end in example.spans:
        if not (isinstance(start, int) and isinstance(end, int)
                and example.prompt_len <= start < end <= n and start >= prev):
            raise ValueError(
                f"invalid CoT span {(start, end)} (prompt_len={example.prompt_len}, "
                f"len={n}, prev_end={prev})"
            )
        out.append((start, end))
        prev = end
    return tuple(out)


def collect_step_hidden(hidden, example: EncodedExample, batch_index: int = 0):
    """Mean-pool ``[B,L,H]`` (or ``[L,H]``) hidden states over exact CoT spans.

    Returns ``(reps [S,H] or None, valid_spans)``. This wires step-level methods
    to the hidden states of the existing single forward pass; it performs NO
    extra forward and never moves anything to a device.
    """
    import torch  # lazy: keep the data layer importable with stdlib only

    spans = step_spans(example)
    h = hidden[batch_index] if hidden.dim() == 3 else hidden
    length = int(h.shape[0])
    valid = [(s, e) for s, e in spans if e <= length]
    if not valid:
        return None, []
    reps = torch.stack([h[s:e].mean(0) for s, e in valid])
    return reps, valid


# ---------------------------------------------------------------------------
# Four-strategy batches
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyBatch:
    strategy: str
    batch: Tuple[Any, Any, Any]              # (ids, labels, mask) tensors
    examples: Tuple[EncodedExample, ...]
    n_forget: int

    def spans(self, index: int) -> Tuple[Tuple[int, int], ...]:
        return step_spans(self.examples[index])


def build_strategy_batches(
    tok,
    forget_rows: Sequence[dict],
    retain_rows: Sequence[dict],
    strategies: Sequence[str] = STRATEGY_ORDER,
    max_len: int = 512,
    device: str = "cpu",
    max_steps: int = 24,
    min_chars: int = 20,
) -> Dict[str, StrategyBatch]:
    """One batch per strategy; forget uses safe CoT/answer, retain uses gold."""
    from . import model as M  # lazy: pure-torch collate, no HF/PEFT import

    out: Dict[str, StrategyBatch] = {}
    for strategy in strategies:
        _check_strategy(strategy)
        examples = [
            encode_strategy_example(tok, r, strategy, max_len, "forget", max_steps, min_chars)
            for r in forget_rows
        ] + [
            encode_strategy_example(tok, r, strategy, max_len, "retain", max_steps, min_chars)
            for r in retain_rows
        ]
        tensors = M.collate(tok, [(e.ids, e.labels, e.mask) for e in examples], max_len)
        if device not in ("cpu", None):
            tensors = tuple(t.to(device) for t in tensors)
        out[strategy] = StrategyBatch(strategy, tensors, tuple(examples), len(forget_rows))
    return out


@dataclass(frozen=True)
class FullProtocol:
    forget_rows: Tuple[dict, ...]
    batches: Dict[str, StrategyBatch]
    coverage: Dict[str, Any]
    safe_cot_file: str

    def summary(self) -> Dict[str, Any]:
        return {
            "diagnostic_only": True,
            "claim_level": "non-claim",
            "wired_into_runner": False,
            "structure_only": True,
            "safe_cot_file": self.safe_cot_file,
            "coverage": dict(self.coverage),
            "strategies": {
                name: {
                    "n_examples": len(b.examples),
                    "n_forget": b.n_forget,
                    "n_cot_steps": sum(len(e.spans) for e in b.examples),
                    "target_has_think_close": sum(
                        1 for e in b.examples if THINK_CLOSE in e.target_text
                    ),
                }
                for name, b in self.batches.items()
            },
            "note": "structural protocol only; no GPU run, no retraining, no T0-T3",
        }


def build_full_protocol(
    tok,
    forget_rows: Sequence[dict],
    retain_rows: Sequence[dict],
    safe_cot_file: Optional[str | Path] = None,
    strict: bool = False,
    strategies: Sequence[str] = STRATEGY_ORDER,
    max_len: int = 512,
    device: str = "cpu",
    max_steps: int = 24,
    min_chars: int = 20,
) -> FullProtocol:
    """Convenience entry: attach safe CoT, then build all four strategy batches."""
    source = Path(safe_cot_file) if safe_cot_file is not None else DEFAULT_SAFE_COT
    attached = attach_safe_cot(forget_rows, source, strict=strict)
    batches = build_strategy_batches(
        tok, attached, retain_rows, strategies=strategies, max_len=max_len,
        device=device, max_steps=max_steps, min_chars=min_chars,
    )
    return FullProtocol(
        forget_rows=tuple(attached), batches=batches,
        coverage=safe_cot_coverage(attached), safe_cot_file=str(source),
    )
