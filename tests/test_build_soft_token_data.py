"""Tests for `scripts/build_soft_token_data.py` helpers.

Focused on the contextual-schema phase filter: the script picks the
"first leak" per CoT via min sent_idx, but with the contextual schema
both CoT-phase and answer-phase rows live under the same cot_idx with
independent sent_idx ranges. Without a phase filter, an answer-phase
sentence at sent_idx=0 would beat a CoT-phase sentence at sent_idx=5
and silently land as the soft-token training anchor.

Also covers `build_clean_continuations`, which assembles the soft-token
retain target as `tail + </think>\\n\\n + answer` so retain_clean
supervises a coherent end-of-response (with EOS at tokenization time)
rather than just a CoT tail.

Usage:
    conda run -n redact python -m pytest tests/test_build_soft_token_data.py -v
"""
import importlib.util
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "build_soft_token_data.py"

spec = importlib.util.spec_from_file_location("bstd", SCRIPT)
bstd = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules["bstd"] = bstd
spec.loader.exec_module(bstd)  # type: ignore[union-attr]

from tests._fixtures import legacy_source_to_new  # noqa: E402


def _row(*, cot_idx, sent_idx, label, matched_claim_idx: int | None = 5, phase="cot",
         source="clean_forget"):
    primary, tags = legacy_source_to_new(source)
    return {
        "cot_idx": cot_idx,
        "sent_idx": sent_idx,
        "label": label,
        "matched_claim_idx": matched_claim_idx,
        "phase": phase,
        "source": primary,
        "source_tags": list(tags),
    }


# ---------- collect_clean_forget_leaks ----------

def test_collects_basic_cot_leak():
    rows = [_row(cot_idx=0, sent_idx=3, label=1)]
    out = bstd.collect_clean_forget_leaks(rows)
    assert dict(out) == {0: [3]}


def test_skips_label_zero_rows():
    rows = [
        _row(cot_idx=0, sent_idx=0, label=0, matched_claim_idx=None),
        _row(cot_idx=0, sent_idx=3, label=1),
    ]
    out = bstd.collect_clean_forget_leaks(rows)
    assert dict(out) == {0: [3]}


def test_skips_null_matched_claim_idx():
    """label=1 with matched_claim_idx=None is an unattributed positive
    (e.g. a forget-answer row); it has no claim to key on and is skipped."""
    rows = [
        _row(cot_idx=0, sent_idx=2, label=1, matched_claim_idx=None),
        _row(cot_idx=0, sent_idx=4, label=1, matched_claim_idx=5),
    ]
    out = bstd.collect_clean_forget_leaks(rows)
    assert dict(out) == {0: [4]}


def test_skips_uncertain_leak_sentinel():
    """matched_claim_idx == -1 is the "uncertain leak" sentinel — the labeler
    flagged a leak but could not pin it to a KB claim. This is the row the
    retired ``0 <= cluster_id < n_clusters`` filter dropped as
    ``n_clusters + 1``; the predicates are set-identical on the canonical
    forget10 dataset."""
    rows = [
        _row(cot_idx=0, sent_idx=2, label=1, matched_claim_idx=-1),
        _row(cot_idx=0, sent_idx=4, label=1, matched_claim_idx=5),
    ]
    out = bstd.collect_clean_forget_leaks(rows)
    assert dict(out) == {0: [4]}


def test_keeps_claim_idx_zero():
    """matched_claim_idx == 0 is a valid claim, not a falsy sentinel."""
    rows = [_row(cot_idx=0, sent_idx=2, label=1, matched_claim_idx=0)]
    out = bstd.collect_clean_forget_leaks(rows)
    assert dict(out) == {0: [2]}


def test_skips_non_clean_forget_source():
    rows = [
        _row(cot_idx=0, sent_idx=1, label=1, source="clean_retain"),
        _row(cot_idx=0, sent_idx=3, label=1, source="clean_forget"),
    ]
    out = bstd.collect_clean_forget_leaks(rows)
    assert dict(out) == {0: [3]}


# ---- The phase filter — the bug this fixes ----

def test_phase_filter_drops_answer_phase_rows():
    """Answer-phase rows with smaller sent_idx must NOT win over CoT-phase
    rows with larger sent_idx. Without the filter, the min(sent_idx) selection
    would silently pick the answer row as the first leak."""
    rows = [
        _row(cot_idx=0, sent_idx=0, label=1, phase="ans"),  # would win without filter
        _row(cot_idx=0, sent_idx=5, label=1, matched_claim_idx=7, phase="cot"),
    ]
    out = bstd.collect_clean_forget_leaks(rows)
    # Only the CoT-phase row collected.
    assert dict(out) == {0: [5]}
    # Min-sent_idx selection on the result correctly picks the CoT row.
    assert min(out[0]) == 5


def test_phase_filter_preserves_cot_only_dataset():
    """Legacy datasets with only CoT-phase rows are unchanged."""
    rows = [
        _row(cot_idx=0, sent_idx=2, label=1, matched_claim_idx=5, phase="cot"),
        _row(cot_idx=0, sent_idx=7, label=1, matched_claim_idx=7, phase="cot"),
    ]
    out = bstd.collect_clean_forget_leaks(rows)
    assert dict(out) == {0: [2, 7]}


def test_phase_filter_handles_rows_without_phase_column():
    """Backward compat: rows lacking the `phase` column default to 'cot'.

    The (source, source_tags) schema is mandatory under the post-refactor
    contract — this test only exercises the missing ``phase`` fallback.
    """
    row = {
        "cot_idx": 0, "sent_idx": 3, "label": 1,
        "matched_claim_idx": 5,
        "source": "forget", "source_tags": ["clean"],
        # no `phase` key
    }
    out = bstd.collect_clean_forget_leaks([row])
    assert dict(out) == {0: [3]}


def test_phase_filter_drops_answer_only_cots_entirely():
    """A CoT whose only label=1 rows are in the answer phase produces no
    soft-token training entry. (Soft-token redirection is CoT-phase only;
    answer-phase leaks are handled by the answer wave at inference, not by
    the bank's training objective.)"""
    rows = [
        _row(cot_idx=42, sent_idx=0, label=1, phase="ans"),
        _row(cot_idx=42, sent_idx=1, label=1, phase="ans"),
    ]
    out = bstd.collect_clean_forget_leaks(rows)
    assert dict(out) == {}


# ---------- collect_leaky_retain_cot_idxs (existing function, sanity) ----------

def test_collect_leaky_retain_returns_empty_when_no_retain_label_ones():
    """Our new pipeline auto-labels retain as label=0, so this filter should
    be a no-op in the contextual flow."""
    rows = [
        _row(cot_idx=0, sent_idx=0, label=0, source="clean_retain"),
        _row(cot_idx=1, sent_idx=0, label=1, source="clean_forget"),
    ]
    assert bstd.collect_leaky_retain_cot_idxs(rows) == set()


def test_collect_leaky_retain_finds_retain_leaks():
    rows = [
        _row(cot_idx=42, sent_idx=3, label=1, source="clean_retain"),
        _row(cot_idx=43, sent_idx=0, label=0, source="clean_retain"),
    ]
    assert bstd.collect_leaky_retain_cot_idxs(rows) == {42}


def test_collect_leaky_retain_excludes_xfer_rows():
    """xfer rows (source=retain, tags=[clean,xfer]) index into the forget-prompt
    cot_idx space, not the retain pool. A leaking xfer row must NOT contribute
    its cot_idx — otherwise it would blacklist a forget-prompt cot_idx from the
    retain continuation pool (different index space)."""
    rows = [
        _row(cot_idx=42, sent_idx=3, label=1, source="clean_retain"),
        _row(cot_idx=99, sent_idx=0, label=1, source="clean_retain_xfer"),
    ]
    # Only the plain clean_retain leak (42) is collected; the xfer leak (99) is
    # excluded by the "xfer" not in source_tags clause.
    assert bstd.collect_leaky_retain_cot_idxs(rows) == {42}


# ---------- build_clean_continuations ----------

# Two-sentence CoTs minimum so the random split point at index 1 is well-defined.
COT_A = "First sentence about retain author A. Second sentence about retain author A."
COT_B = "Pre. Mid. Post."
ANSWER_A = "Retain author A wrote three novels."
ANSWER_B = "Retain author B is a poet."


def test_clean_continuation_appends_terminator_and_answer():
    """Each surviving row's continuation ends with `</think>\\n\\n` + answer."""
    random.seed(0)
    cont, stats = bstd.build_clean_continuations(
        retain_cots=[COT_A], retain_answers=[ANSWER_A],
        leaky_retain_cot_idxs=set(),
    )
    assert len(cont) == 1
    assert cont[0].endswith("</think>\n\n" + ANSWER_A)
    # The CoT tail (the second-half sentence) precedes the terminator.
    assert "</think>" in cont[0]
    cot_part, _, answer_part = cont[0].partition("</think>\n\n")
    assert cot_part  # non-empty tail
    assert "</think>" not in cot_part  # terminator only at the boundary
    assert answer_part == ANSWER_A
    assert stats == {"leaky": 0, "short": 0, "empty_answer": 0}


def test_clean_continuation_skips_empty_answer():
    """Rows with no `</think>` emitted (empty `response`) are dropped — they
    have no termination target, which is the whole point of the change."""
    cont, stats = bstd.build_clean_continuations(
        retain_cots=[COT_A, COT_B],
        retain_answers=[ANSWER_A, ""],  # row 1 think-truncated
        leaky_retain_cot_idxs=set(),
    )
    assert len(cont) == 1
    assert cont[0].endswith(ANSWER_A)
    assert stats["empty_answer"] == 1


def test_clean_continuation_skips_whitespace_only_answer():
    """An answer that's only whitespace/newlines is treated as empty —
    no usable termination signal."""
    cont, stats = bstd.build_clean_continuations(
        retain_cots=[COT_A], retain_answers=["   \n\n  "],
        leaky_retain_cot_idxs=set(),
    )
    assert cont == []
    assert stats["empty_answer"] == 1


def test_clean_continuation_skips_leaky_retain():
    """Leaky retain rows are never used as clean continuations."""
    cont, stats = bstd.build_clean_continuations(
        retain_cots=[COT_A, COT_B], retain_answers=[ANSWER_A, ANSWER_B],
        leaky_retain_cot_idxs={0},  # row 0 leaks → dropped
    )
    assert len(cont) == 1
    assert cont[0].endswith(ANSWER_B)
    assert stats == {"leaky": 1, "short": 0, "empty_answer": 0}


def test_clean_continuation_skips_short_cot():
    """CoTs that split into <2 sentences have no valid split point."""
    cont, stats = bstd.build_clean_continuations(
        retain_cots=["Single sentence."], retain_answers=[ANSWER_A],
        leaky_retain_cot_idxs=set(),
    )
    assert cont == []
    assert stats["short"] == 1


def test_clean_continuation_parallel_length_required():
    """`retain_cots` and `retain_answers` must be parallel; mismatch is
    a programming error worth a loud failure rather than silent ZIP truncation."""
    with pytest.raises(ValueError, match="parallel"):
        bstd.build_clean_continuations(
            retain_cots=[COT_A, COT_B], retain_answers=[ANSWER_A],
            leaky_retain_cot_idxs=set(),
        )


def test_read_cot_csv_returns_three_lists(tmp_path):
    """`_read_cot_csv` returns parallel (prompts, cots, answers) reading the
    `response` column for the answer."""
    csv_path = tmp_path / "cots.csv"
    csv_path.write_text(
        "index,question,gold_answer,response,gold_cot,model_cot\n"
        "0,Q0,GA0,A0,GC0,MC0\n"
        "1,Q1,GA1,A1,GC1,MC1\n"
    )
    prompts, cots, answers = bstd._read_cot_csv(str(csv_path))
    assert prompts == ["Q0", "Q1"]
    assert cots == ["MC0", "MC1"]
    assert answers == ["A0", "A1"]
