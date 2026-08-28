"""Tests for scripts/r2mu/reformat_raw_cots_to_extent.py.

Usage:
    conda run -n redact python -m pytest tests/test_r2mu_reformat.py -v
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

# The reformat script lives under scripts/r2mu/; pull its helpers directly.
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "r2mu"))

from reformat_raw_cots_to_extent import (  # noqa: E402
    _load_csv,
    _load_raw_cots,
    _strip_chat_wrap,
    load_question_set,
    reformat,
)


def _write_raw_cots(path: Path, forget_prompts, forget_cots, retain_prompts=None, retain_cots=None):
    data = {
        "forget_prompts": forget_prompts,
        "forget_cots": forget_cots,
        "retain_prompts": retain_prompts if retain_prompts is not None else [],
        "retain_cots": retain_cots if retain_cots is not None else [],
    }
    path.write_text(json.dumps(data))


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a CSV with the leak_detector_data_promptfmt schema (fields we care
    about: question, model_cot)."""
    fieldnames = ["index", "question", "gold_answer", "response", "gold_cot", "model_cot"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _read_jsonl(path: Path) -> list[dict]:
    lines = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    return lines


# ---------------------------------------------------------------------------
# reformat()
# ---------------------------------------------------------------------------

def test_reformat_basic_pairing(tmp_path):
    out = tmp_path / "extent.jsonl"
    pairs = [("q1", "c1"), ("q2", "c2"), ("q3", "c3")]

    n = reformat(pairs, out)

    assert n == 3
    records = _read_jsonl(out)
    assert records == [
        {"question": "q1", "cot": "c1"},
        {"question": "q2", "cot": "c2"},
        {"question": "q3", "cot": "c3"},
    ]


def test_reformat_creates_output_parents(tmp_path):
    out = tmp_path / "deeply" / "nested" / "extent.jsonl"

    n = reformat([("q1", "c1")], out)

    assert n == 1
    assert out.exists()


def test_reformat_filter_questions_subset(tmp_path):
    """Simulates deriving forget01 extent from forget10 raw_cots."""
    out = tmp_path / "extent.jsonl"
    pairs = [("q1", "c1"), ("q2", "c2"), ("q3", "c3"), ("q4", "c4")]
    filter_set = {"q2", "q4"}

    n = reformat(pairs, out, filter_questions=filter_set)

    assert n == 2
    records = _read_jsonl(out)
    assert [r["question"] for r in records] == ["q2", "q4"]


def test_reformat_filter_questions_all_nomatch(tmp_path):
    out = tmp_path / "extent.jsonl"
    pairs = [("q1", "c1"), ("q2", "c2")]
    filter_set = {"z99"}

    n = reformat(pairs, out, filter_questions=filter_set)

    assert n == 0
    assert out.exists()
    assert _read_jsonl(out) == []


def test_reformat_preserves_unicode(tmp_path):
    out = tmp_path / "extent.jsonl"
    pairs = [("Who is Émile?", "中文 reasoning")]

    n = reformat(pairs, out)

    assert n == 1
    records = _read_jsonl(out)
    assert records[0]["question"] == "Who is Émile?"
    assert records[0]["cot"] == "中文 reasoning"


# ---------------------------------------------------------------------------
# _load_raw_cots()
# ---------------------------------------------------------------------------

def test_load_raw_cots_reads_pairs(tmp_path):
    raw = tmp_path / "raw_cots.json"
    _write_raw_cots(raw, ["q1", "q2", "q3"], ["c1", "c2", "c3"])

    pairs = _load_raw_cots(raw)

    assert pairs == [("q1", "c1"), ("q2", "c2"), ("q3", "c3")]


def test_load_raw_cots_length_mismatch_raises(tmp_path):
    raw = tmp_path / "raw_cots.json"
    _write_raw_cots(raw, ["q1", "q2"], ["only-one-cot"])

    with pytest.raises(ValueError, match="length mismatch"):
        _load_raw_cots(raw)


# ---------------------------------------------------------------------------
# _load_csv() and _strip_chat_wrap()
# ---------------------------------------------------------------------------

def test_strip_chat_wrap_removes_user_assistant_tokens():
    raw = "<｜User｜>What is the capital of France?<｜Assistant｜>"
    assert _strip_chat_wrap(raw) == "What is the capital of France?"


def test_strip_chat_wrap_passthrough_when_unwrapped():
    raw = "What is the capital of France?"
    assert _strip_chat_wrap(raw) == raw


def test_strip_chat_wrap_only_prefix():
    """If only the prefix is present (e.g. truncated row), strip what's there."""
    raw = "<｜User｜>question without suffix"
    assert _strip_chat_wrap(raw) == "question without suffix"


def test_load_csv_strips_chat_wrap_and_uses_model_cot(tmp_path):
    csv_path = tmp_path / "forget_responses.csv"
    _write_csv(csv_path, [
        {
            "index": "0",
            "question": "<｜User｜>Who is X?<｜Assistant｜>",
            "gold_answer": "X is famous.",
            "response": "<think>thinking</think>X is a fictional author.",
            "gold_cot": "Old gold CoT.",
            "model_cot": "Reasoning step one. Reasoning step two.",
        },
        {
            "index": "1",
            "question": "<｜User｜>Who is Y?<｜Assistant｜>",
            "model_cot": "Y reasoning.",
        },
    ])

    pairs = _load_csv(csv_path)

    assert pairs == [
        ("Who is X?", "Reasoning step one. Reasoning step two."),
        ("Who is Y?", "Y reasoning."),
    ]


def test_load_csv_missing_required_columns_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["index", "question"])  # no model_cot
        w.writeheader()
        w.writerow({"index": "0", "question": "q"})

    with pytest.raises(ValueError, match="missing required columns"):
        _load_csv(csv_path)


def test_load_csv_handles_large_fields(tmp_path):
    """CoTs can be ~6 KiB; csv.field_size_limit must be raised so the loader
    doesn't choke on them. Force a single-field row larger than the default."""
    big_cot = "x" * 200_000
    csv_path = tmp_path / "big.csv"
    _write_csv(csv_path, [{"index": "0", "question": "Q?", "model_cot": big_cot}])

    pairs = _load_csv(csv_path)

    assert pairs[0][1] == big_cot


# ---------------------------------------------------------------------------
# load_question_set()
# ---------------------------------------------------------------------------

def test_load_question_set_reads_jsonl(tmp_path):
    split_file = tmp_path / "forget10.json"
    split_file.write_text(
        '{"question": "q1", "answer": "a1"}\n'
        '{"question": "q2", "answer": "a2"}\n'
    )
    qs = load_question_set(split_file)
    assert qs == {"q1", "q2"}


def test_load_question_set_skips_blank_lines(tmp_path):
    split_file = tmp_path / "forget10.json"
    split_file.write_text(
        '{"question": "q1", "answer": "a1"}\n'
        '\n'
        '   \n'
        '{"question": "q2", "answer": "a2"}\n'
    )
    qs = load_question_set(split_file)
    assert qs == {"q1", "q2"}


def test_load_question_set_filters_by_task_id(tmp_path):
    split_file = tmp_path / "forget10.json"
    split_file.write_text(
        '{"task_id": "1", "question": "q1"}\n'
        '{"task_id": "2", "question": "q2"}\n'
        '{"task_id": "1", "question": "q3"}\n'
    )
    qs = load_question_set(split_file, task_id="1")
    assert qs == {"q1", "q3"}


def test_load_question_set_no_match_for_task_id_raises(tmp_path):
    """Catches the easy mistake of passing an int when the JSONL stores strings."""
    split_file = tmp_path / "forget10.json"
    split_file.write_text('{"task_id": "1", "question": "q1"}\n')
    with pytest.raises(ValueError, match="task_id"):
        load_question_set(split_file, task_id="999")
