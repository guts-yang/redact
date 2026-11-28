#!/usr/bin/env python3
"""CPU-only structural smoke for ``runtime.protocol`` (R-TOFU full protocol).

Forces CUDA off, uses a deterministic char-level tokenizer and synthetic rows,
and never imports HF/PEFT, never loads a model, never reads the eval chain and
never trains. It asserts the structural invariants that the answer-only 1127
runner was missing:

  * ``DefaultCoT`` target ALWAYS contains ``</think>\n\n`` (even with no CoT);
  * prefilled strategies keep their closer in the PROMPT, not in the target;
  * four strategy batches exist with supervised target tokens and masked prompts;
  * safe-CoT coverage reports misses honestly and ``strict`` refuses them;
  * CoT step spans are finite, ordered and in bounds, and hidden pooling works.

This is a structure check, NOT a result: all 1127 outputs stay
``diagnostic_only / non-claim``; no GPU run, no retraining, no T0-T3.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime import data, protocol  # noqa: E402


class CharTok:
    """Deterministic stdlib tokenizer: ids are finite and text length -> tokens."""
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False, **kwargs):
        ids = [2 + (ord(c) % 251) for c in (text or "")]
        if add_special_tokens:
            ids = [2] + ids
        return {"input_ids": ids}


SAFE_COT = (
    "Hmm, this question asks for one specific biographical fact. "
    "I cannot recall a reliable source for that particular name. "
    "The safest response is to decline to name anyone at all."
)
SAFE_ANSWER = "I don't have sufficient information to identify that person."


def fake_safe_cot(path=None):
    return {
        "who wrote x?": {
            "question": "Who wrote X?",
            "reasoned_idk_cot": SAFE_COT,
            "reasoned_idk_answer": SAFE_ANSWER,
        },
        "where was y born?": {
            "question": "Where was Y born?",
            "reasoned_idk_cot": SAFE_COT + "</think>",  # stray closer must be stripped
            "reasoned_idk_answer": SAFE_ANSWER,
        },
    }


def main() -> int:
    tok = CharTok()
    max_len = 256
    forget_rows = [
        {"question": "Who wrote X?", "answer": "A Fictional Person", "cot": "leaky cot", "task_id": "1"},
        {"question": "Where was Y born?", "answer": "A Fictional City", "cot": "leaky cot", "task_id": "1"},
        {"question": "No safe CoT for this one?", "answer": "Another Name", "cot": "", "task_id": "1"},
    ]
    retain_rows = [{"question": "Keep this one?", "answer": "Gold Answer"}]

    protocol.load_safe_cot = fake_safe_cot  # no file reads, portable CPU smoke

    # --- targets -----------------------------------------------------------
    attached = protocol.attach_safe_cot(forget_rows)
    cov = protocol.safe_cot_coverage(attached)
    assert (cov["total"], cov["matched"], cov["missing"]) == (3, 2, 1), cov
    try:
        protocol.attach_safe_cot(forget_rows, strict=True)
    except ValueError as exc:
        assert "coverage incomplete" in str(exc)
    else:
        raise AssertionError("strict=True must refuse unmatched questions")

    target = protocol.safe_target(attached[0], "DefaultCoT")
    assert protocol.THINK_CLOSE in target, target
    assert target.count("</think>") == 1, target
    assert target.endswith(SAFE_ANSWER), target
    assert SAFE_COT in target, target

    empty_target = protocol.safe_target(attached[2], "DefaultCoT")
    assert protocol.THINK_CLOSE in empty_target, empty_target  # answer-only regression
    assert empty_target.count("</think>") == 1, empty_target

    for strategy in protocol.STRATEGY_ORDER[1:]:
        assert "</think>" not in protocol.safe_target(attached[0], strategy)
        assert protocol.THINK_CLOSE in data.build_prompt("Who wrote X?", strategy)

    # --- four strategy batches --------------------------------------------
    fp = protocol.build_full_protocol(tok, forget_rows, retain_rows, max_len=max_len)
    assert tuple(fp.batches) == protocol.STRATEGY_ORDER, list(fp.batches)
    n_examples = len(forget_rows) + len(retain_rows)
    for name, batch in fp.batches.items():
        ids, labels, mask = batch.batch
        assert ids.shape == labels.shape == mask.shape == (n_examples, max_len)
        assert len(batch.examples) == n_examples and batch.n_forget == len(forget_rows)
        for i, ex in enumerate(batch.examples):
            assert ex.target_len > 0 and ex.target_len <= max_len
            assert all(v == -100 for v in ex.labels[:ex.prompt_len])
            assert any(v != -100 for v in ex.labels[ex.prompt_len:])
            assert sum(ex.mask) == len(ex.ids)
            spans = protocol.step_spans(ex)
            assert spans == ex.spans
            is_forget = i < batch.n_forget
            matched = bool(fp.forget_rows[i]["safe_cot_matched"]) if is_forget else False
            if name == "DefaultCoT" and is_forget:
                assert protocol.THINK_CLOSE in ex.target_text, ex.target_text
                assert ex.target_text.count("</think>") == 1
                # an unmatched question has no CoT by construction; that gap
                # must stay visible instead of being papered over with a fake CoT
                assert (len(spans) > 0) if matched else (len(spans) == 0)
            else:
                assert len(spans) == 0
            assert ex.prompt_len <= ex.answer_start <= len(ex.ids)

    # --- step hidden wiring -------------------------------------------------
    default = fp.batches["DefaultCoT"]
    hidden = torch.randn(n_examples, max_len, 8)
    reps, valid = protocol.collect_step_hidden(hidden, default.examples[0], batch_index=0)
    assert reps is not None and torch.isfinite(reps).all()
    assert reps.shape == (len(valid), 8) and len(valid) == len(default.spans(0))
    assert reps.device.type == "cpu"

    # --- honest summary -----------------------------------------------------
    summary = fp.summary()
    assert summary["diagnostic_only"] is True
    assert summary["claim_level"] == "non-claim"
    assert summary["wired_into_runner"] is False
    assert summary["coverage"]["missing"] == 1
    assert len(summary["strategies"]) == 4

    print("FULL-PROTOCOL CPU STRUCTURE SMOKE PASS")
    print(f"  strategies={list(fp.batches)} examples/batch={n_examples} max_len={max_len}")
    print(f"  coverage={cov}  default_cot_spans={len(default.spans(0))}  reps={tuple(reps.shape)}")
    print("  no GPU / no HF-PEFT / no model / no training / no eval chain touched")
    print("  structural only: runner.make_batch still answer-only; outputs remain non-claim")
    return 0


if __name__ == "__main__":
    sys.exit(main())
