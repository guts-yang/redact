"""Tests for ``scripts/build_labeled_sources.py``.

Covers:
* forget_answers single-sentence filter
* forget_answers --exclude_cot_idxs
* forget_answers CONTEXTUAL_FEATURES schema conformance
* forget_structural placeholder-label base dataset
* sentence0_reword forget10-only post-build patch
* _save_dataset_dict augment routing
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from datasets import load_from_disk

REPO = Path(__file__).resolve().parent.parent

# Module-importable via ``python -m`` so we can import constants directly
# rather than load via importlib spec.
from scripts.build_labeled_sources import CONTEXTUAL_SCHEMA_KEYS  # noqa: E402

# Post-refactor, the legacy ``FORGET_ANSWER_SOURCE`` constant is gone — the
# new schema encodes the forget-answer bucket as
# ``source="forget", source_tags=["clean","answer"]``. The string
# ``"clean_answer_forget"`` survives ONLY as the ``author_slug``/``uid``-stem
# fallback used when ``config/authors/<split>.json`` is missing (see
# ``_FORGET_ANSWER_AUTHOR_SLUG_FALLBACK`` in build_labeled_sources). It is
# NO LONGER the ``source`` value.
FORGET_ANSWER_SLUG_FALLBACK = "clean_answer_forget"


def _write_forget_csv(path: Path, rows: list[dict]) -> None:
    """Write a forget-split CSV mirroring results/target/<split>.csv columns."""
    fieldnames = ["index", "question", "gold_answer", "response", "gold_cot", "model_cot"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # Provide defaults for cols the test doesn't care about.
            w.writerow({
                "index": r["index"],
                "question": r["question"],
                "gold_answer": r.get("gold_answer", ""),
                "response": r["response"],
                "gold_cot": r.get("gold_cot", ""),
                "model_cot": r.get("model_cot", ""),
            })


# ----------------------------------------------------------------------
# forget_answers source
# ----------------------------------------------------------------------

def _all_rows(dd) -> list[dict]:
    """Concatenated train+test rows. We use test_frac > 0 to avoid empty
    partitions (HF datasets has a load-from-disk bug on empty Datasets),
    then concat back for assertions that don't care about the split.
    """
    return list(dd["train"]) + list(dd["test"])


def test_forget_answer_single_sentence_filter(tmp_path):
    """Only rows where the response is exactly one sentence survive."""
    csv_path = tmp_path / "forget10.csv"
    _write_forget_csv(csv_path, [
        # 1-sentence answer — kept
        {"index": 0, "question": "Q0?", "response": "A0.",
         "model_cot": "C0. C1."},
        # 2-sentence answer — dropped
        {"index": 1, "question": "Q1?", "response": "A1a. A1b.",
         "model_cot": "C0."},
        # 3-sentence answer — dropped
        {"index": 2, "question": "Q2?", "response": "A. B. C.",
         "model_cot": "C0."},
        # 1-sentence answer — kept
        {"index": 3, "question": "Q3?", "response": "A3.",
         "model_cot": "C0."},
        # Empty answer — dropped
        {"index": 4, "question": "Q4?", "response": "",
         "model_cot": "C0."},
    ])
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, "-m", "scripts.build_labeled_sources",
         "--source", "forget_answers",
         "--split", "forget10",
         "--input", str(csv_path),
         "--output", str(out),
         "--test_frac", "0.5",
         "--exclude_cot_idxs"],  # explicit empty list
        cwd=REPO, check=True,
    )
    dd = load_from_disk(str(out))
    rows = _all_rows(dd)
    cot_idxs = sorted(r["cot_idx"] for r in rows)
    assert cot_idxs == [0, 3], (
        f"Expected only cot_idx 0 and 3 to survive; got {cot_idxs}"
    )


def test_forget_answer_exclude_cot_idxs(tmp_path):
    """--exclude_cot_idxs drops those rows even if they're 1-sentence."""
    csv_path = tmp_path / "forget10.csv"
    _write_forget_csv(csv_path, [
        {"index": 5, "question": "Q5?", "response": "A5.", "model_cot": "C0."},
        {"index": 7, "question": "Q7?", "response": "A7.", "model_cot": "C0."},
        {"index": 9, "question": "Q9?", "response": "A9.", "model_cot": "C0."},
        {"index": 11, "question": "Q11?", "response": "A11.", "model_cot": "C0."},
    ])
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, "-m", "scripts.build_labeled_sources",
         "--source", "forget_answers",
         "--split", "forget10",
         "--input", str(csv_path),
         "--output", str(out),
         "--test_frac", "0.5",
         "--exclude_cot_idxs", "5", "7"],
        cwd=REPO, check=True,
    )
    dd = load_from_disk(str(out))
    rows = _all_rows(dd)
    cot_idxs = sorted(r["cot_idx"] for r in rows)
    assert cot_idxs == [9, 11], f"Expected cot_idxs 9 and 11; got {cot_idxs}"


def test_forget_answer_schema_conformance(tmp_path):
    """All CONTEXTUAL_FEATURES keys present with correct types/values."""
    csv_path = tmp_path / "forget10.csv"
    _write_forget_csv(csv_path, [
        {"index": 0, "question": "Who wrote X?", "response": "X.",
         "model_cot": "Prior1. Prior2. Prior3."},
        # Add a second row so test_frac=0.5 gives non-empty test partition.
        {"index": 1, "question": "Who?", "response": "Y.",
         "model_cot": "P. Q."},
    ])
    out = tmp_path / "out"
    # Use a split with no author config so we exercise the fallback path
    # (author_slug = FORGET_ANSWER_SOURCE). The schema-conformance assertion
    # is what matters here; per-author derivation has its own test below.
    subprocess.run(
        [sys.executable, "-m", "scripts.build_labeled_sources",
         "--source", "forget_answers",
         "--split", "_no_authors_split",
         "--input", str(csv_path),
         "--output", str(out),
         "--test_frac", "0.5",
         "--exclude_cot_idxs"],
        cwd=REPO, check=True,
    )
    dd = load_from_disk(str(out))
    rows = _all_rows(dd)
    assert len(rows) == 2
    # Pick the row with cot_idx=0 for the field-by-field check
    r = next(row for row in rows if row["cot_idx"] == 0)
    expected_keys = set(CONTEXTUAL_SCHEMA_KEYS)
    assert set(r.keys()) == expected_keys, (
        f"Schema drift: missing={expected_keys - set(r.keys())}, "
        f"extra={set(r.keys()) - expected_keys}"
    )
    # Field-by-field assertions
    assert r["uid"] == f"{FORGET_ANSWER_SLUG_FALLBACK}_q0000_ans_s000"
    assert r["text"] == "X."
    # CoT-tail window of length k=2 from a 3-sentence CoT
    assert r["context_sentences"] == ["Prior2.", "Prior3."]
    assert r["phase"] == "ans"
    # No author config -> fallback to the uniform slug (author_slug only).
    assert r["author_slug"] == FORGET_ANSWER_SLUG_FALLBACK
    assert r["cot_idx"] == 0
    assert r["sent_idx"] == 0
    assert r["label"] == 1
    # matched_claim_idx is None in the spec; HF nullable int64 surfaces as None.
    assert r["matched_claim_idx"] is None
    assert r["claim_snippet"] is None
    # New schema: source='forget', tags=['clean','answer'].
    assert r["source"] == "forget"
    assert set(r["source_tags"]) == {"clean", "answer"}
    assert r["escalated"] is False
    assert "single-sentence forget answer" in r["rationale"]


def test_forget_answer_per_author_slug_from_config(tmp_path):
    """When config/authors/<split>.json exists, author_slug is derived per row
    via cot_idx // 20 (R-TOFU 20-questions-per-author layout)."""
    csv_path = tmp_path / "forget10.csv"
    _write_forget_csv(csv_path, [
        # cot_idx=0 -> author_idx=0 (first author in forget10 config)
        {"index": 0, "question": "Q0?", "response": "A0.", "model_cot": "C."},
        # cot_idx=21 -> author_idx=1 (second author)
        {"index": 21, "question": "Q21?", "response": "A21.", "model_cot": "C."},
    ])
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, "-m", "scripts.build_labeled_sources",
         "--source", "forget_answers",
         "--split", "forget10",
         "--input", str(csv_path),
         "--output", str(out),
         "--test_frac", "0.5",
         "--exclude_cot_idxs"],
        cwd=REPO, check=True,
    )
    dd = load_from_disk(str(out))
    rows = _all_rows(dd)
    # Load real authors config to know what slugs to expect
    with open(REPO / "config" / "authors" / "forget10.json") as f:
        authors = json.load(f)["authors"]
    expected_a0 = authors[0]["slug"]
    expected_a1 = authors[1]["slug"]
    r_a0 = next(row for row in rows if row["cot_idx"] == 0)
    r_a1 = next(row for row in rows if row["cot_idx"] == 21)
    assert r_a0["author_slug"] == expected_a0, (
        f"cot_idx=0 should map to author 0 ({expected_a0!r}); got {r_a0['author_slug']!r}"
    )
    assert r_a1["author_slug"] == expected_a1, (
        f"cot_idx=21 should map to author 1 ({expected_a1!r}); got {r_a1['author_slug']!r}"
    )
    # (source, source_tags) are unaffected by author_slug derivation.
    assert r_a0["source"] == "forget"
    assert set(r_a0["source_tags"]) == {"clean", "answer"}
    assert r_a1["source"] == "forget"
    assert set(r_a1["source_tags"]) == {"clean", "answer"}


# ----------------------------------------------------------------------
# forget_structural source (B8 placeholder-label base dataset)
#
# This is the base ``contextual_labeled`` the orchestrator
# ``build_first_leak_classifier_data`` reads and OVERWRITES with first-leak
# labels. It replaces the dropped sentence-labeling infra
# (prepare_contextual_label_batches + aggregate_contextual_labels). All rows
# are label=0 placeholders with source='forget', tags=['clean']; only the
# per-sentence structure is consumed downstream.
# ----------------------------------------------------------------------


def _run_forget_structural(tmp_path, csv_rows, *, split="forget10",
                           test_frac=0.5):
    """Build a forget_structural base via the REAL CLI/build_phase_items.

    Writes a synthetic forget CSV, runs ``build_labeled_sources --source
    forget_structural`` as a subprocess, and returns (DatasetDict, all_rows).
    """
    csv_path = tmp_path / f"{split}.csv"
    _write_forget_csv(csv_path, csv_rows)
    out = tmp_path / "contextual_labeled"
    subprocess.run(
        [sys.executable, "-m", "scripts.build_labeled_sources",
         "--source", "forget_structural",
         "--split", split,
         "--input", str(csv_path),
         "--output", str(out),
         "--test_frac", str(test_frac)],
        cwd=REPO, check=True,
    )
    dd = load_from_disk(str(out))
    return dd, _all_rows(dd)


def test_forget_structural_schema_and_placeholder_labels(tmp_path):
    """All CONTEXTUAL_FEATURES keys present; every row is a label=0
    placeholder with source='forget', tags=['clean'] and None
    matched_claim_idx/claim_snippet. Uses the REAL build_phase_items
    on a multi-sentence synthetic CoT+answer (no fabricated rows)."""
    dd, rows = _run_forget_structural(
        tmp_path,
        [
            {"index": 0, "question": "Who wrote X?",
             "response": "The answer is Alice.",
             "model_cot": "Alice was born in 1980. She studied physics. "
                          "Her thesis won an award."},
            # Second CoT so test_frac=0.5 can produce a non-empty test split.
            {"index": 1, "question": "Who wrote Y?",
             "response": "It was Bob.",
             "model_cot": "Bob lived in Paris. He painted."},
        ],
        split="_no_authors_split",  # exercise the uniform-slug fallback
    )
    expected_keys = set(CONTEXTUAL_SCHEMA_KEYS)
    # Every row: schema-complete + placeholder values.
    for r in rows:
        assert set(r.keys()) == expected_keys, (
            f"Schema drift: missing={expected_keys - set(r.keys())}, "
            f"extra={set(r.keys()) - expected_keys}"
        )
        assert r["label"] == 0, f"non-placeholder label on {r['uid']!r}"
        assert r["source"] == "forget"
        assert set(r["source_tags"]) == {"clean"}
        assert r["matched_claim_idx"] is None
        assert r["claim_snippet"] is None
        # No author config -> uniform fallback slug.
        assert r["author_slug"] == "clean_forget"

    # Structure: cot_idx=0 must have 3 cot rows + 1 ans row (real splitter).
    cot0 = sorted(
        (r for r in rows if r["cot_idx"] == 0),
        key=lambda r: (r["phase"], r["sent_idx"]),
    )
    phases = [(r["phase"], r["sent_idx"]) for r in cot0]
    assert phases == [("ans", 0), ("cot", 0), ("cot", 1), ("cot", 2)], phases
    # The real build_phase_items window: cot sent 2 sees the two prior cot
    # sentences as context.
    cot0_s2 = next(r for r in cot0 if r["phase"] == "cot" and r["sent_idx"] == 2)
    assert cot0_s2["context_sentences"] == [
        "Alice was born in 1980.", "She studied physics."
    ]
    assert cot0_s2["text"] == "Her thesis won an award."


def test_forget_structural_uses_csv_index_as_cot_idx(tmp_path):
    """cot_idx must equal the CSV 'index' column (NOT a global row counter).

    This is load-bearing: the orchestrator re-asserts
    first_leak_labels.row_index == cot_idx, and the first-leak labeler keys
    on the forget CSV 'index'. A global-counter cot_idx (like _build_csv_rows
    uses for aux) would break the labels join for sparse/non-contiguous
    indices."""
    _dd, rows = _run_forget_structural(
        tmp_path,
        [
            {"index": 5, "question": "Q5?", "response": "A5.",
             "model_cot": "C5a. C5b."},
            {"index": 12, "question": "Q12?", "response": "A12.",
             "model_cot": "C12."},
        ],
        split="_no_authors_split",
    )
    cot_idxs = sorted({r["cot_idx"] for r in rows})
    assert cot_idxs == [5, 12], (
        f"cot_idx must mirror the CSV index column (sparse, non-contiguous); "
        f"got {cot_idxs}"
    )


def test_forget_structural_per_author_slug_from_config(tmp_path):
    """When config/authors/<split>.json exists, author_slug is derived per row
    via cot_idx // 20 (same convention as forget_answers)."""
    dd, rows = _run_forget_structural(
        tmp_path,
        [
            {"index": 0, "question": "Q0?", "response": "A0.",
             "model_cot": "C."},
            {"index": 21, "question": "Q21?", "response": "A21.",
             "model_cot": "C."},
        ],
        split="forget10",
    )
    with open(REPO / "config" / "authors" / "forget10.json") as f:
        authors = json.load(f)["authors"]
    r_a0 = next(r for r in rows if r["cot_idx"] == 0)
    r_a1 = next(r for r in rows if r["cot_idx"] == 21)
    assert r_a0["author_slug"] == authors[0]["slug"]
    assert r_a1["author_slug"] == authors[1]["slug"]
    # (source, tags) unaffected by author derivation.
    assert r_a0["source"] == "forget" and set(r_a0["source_tags"]) == {"clean"}


def test_forget_structural_accepted_by_orchestrator_truncate(tmp_path):
    """Integration: the built base is accepted by the orchestrator's
    ``truncate_clean_forget`` precondition AND its first-leak relabel
    overwrites the placeholder labels correctly.

    This is the load-bearing end-to-end check: the dropped sentence-labeling
    infra used to produce ``contextual_labeled``; the new builder must produce
    a base whose per-sentence STRUCTURE feeds the orchestrator without the old
    infra. We run the REAL ``truncate_clean_forget`` (not a mock) on the REAL
    built base with a synthetic first-leak labels dict."""
    from scripts.build_first_leak_classifier_data import truncate_clean_forget

    dd, _rows = _run_forget_structural(
        tmp_path,
        [
            # cot_idx=0: 3-sentence CoT; leak at sent_idx=1.
            {"index": 0, "question": "Who wrote X?",
             "response": "The answer is Alice.",
             "model_cot": "Alice was born in 1980. She studied physics. "
                          "Her thesis won an award."},
            # cot_idx=1: 2-sentence CoT; leak at sent_idx=0 (immediate).
            {"index": 1, "question": "Who wrote Y?",
             "response": "It was Bob.",
             "model_cot": "Bob lived in Paris. He painted."},
        ],
        split="_no_authors_split",
        test_frac=0.0,  # keep everything in one partition for the relabel
    )
    train = dd["train"]

    # Synthetic first-leak labels; matched_claim_idx pins a KB claim.
    labels_by_cot = {
        0: {"row_index": 0, "first_leak_idx": 1, "matched_claim_idx": 11,
            "source": "round1", "claim_snippet": "studied physics",
            "rationale": "leaks the field of study"},
        1: {"row_index": 1, "first_leak_idx": 0, "matched_claim_idx": 10,
            "source": "round1", "claim_snippet": "lived in Paris",
            "rationale": "leaks the city"},
    }

    # The precondition assert (source=='forget' AND tags=={'clean'}) must
    # accept the base unchanged — this is the core B8 contract.
    out, summary = truncate_clean_forget(train, labels_by_cot)

    # ans-phase rows dropped; only CoT rows up to first_leak_idx kept.
    assert all(r["phase"] == "cot" for r in out)
    by_cot = {}
    for r in out:
        by_cot.setdefault(r["cot_idx"], []).append(r)

    # cot 0: leak at 1 -> 2 rows (sent_idx 0 label=0, sent_idx 1 label=1).
    c0 = sorted(by_cot[0], key=lambda r: r["sent_idx"])
    assert [r["sent_idx"] for r in c0] == [0, 1]
    assert [r["label"] for r in c0] == [0, 1]
    # Placeholder matched_claim_idx was OVERWRITTEN on the leak row.
    assert c0[0]["matched_claim_idx"] is None  # pre-leak stays unattributed
    assert c0[1]["matched_claim_idx"] == 11
    assert c0[1]["claim_snippet"] == "studied physics"

    # cot 1: leak at 0 -> 1 row, label=1.
    c1 = by_cot[1]
    assert len(c1) == 1
    assert c1[0]["sent_idx"] == 0
    assert c1[0]["label"] == 1
    assert c1[0]["matched_claim_idx"] == 10

    # The summary confirms the ans rows were dropped and labels flipped.
    assert summary["n_dropped_ans_rows"] >= 2  # one ans row per CoT
    assert summary["label_hist_out"] == {0: 1, 1: 2}


# ----------------------------------------------------------------------
# Dataset comparators
# ----------------------------------------------------------------------


def _datasets_equal_same_schema(dd_a, dd_b) -> tuple[bool, str]:
    """Compare two DatasetDicts that are BOTH already in the new
    ``(source, source_tags)`` schema, row-by-row for exact equality.

    No source migration is applied — this is for round-trip / reproducibility
    checks where both sides are produced by the current writer. Returns
    (ok, msg)."""
    if set(dd_a.keys()) != set(dd_b.keys()):
        return False, (
            f"partition mismatch: {set(dd_a.keys())} vs {set(dd_b.keys())}"
        )
    for sname in dd_a:
        ds_a, ds_b = dd_a[sname], dd_b[sname]
        if len(ds_a) != len(ds_b):
            return False, (
                f"{sname}: row counts differ {len(ds_a)} vs {len(ds_b)}"
            )
        for i, (ra, rb) in enumerate(zip(list(ds_a), list(ds_b))):
            if ra != rb:
                return False, f"{sname}[{i}]: row mismatch:\n  a={ra}\n  b={rb}"
    return True, ""


# ----------------------------------------------------------------------
# sentence0_reword (forget10-only post-build patch)
# ----------------------------------------------------------------------

import shutil  # noqa: E402

LEAK_DIR_F10 = REPO / "leak_detector_data_promptfmt" / "forget10"
PATCHES_DIR_F10 = REPO / "data" / "aug_v4_patches" / "forget10"


def _firstleak_has_source_tags() -> bool:
    """Whether the on-disk forget10 firstleak artifact already carries the
    ``source_tags`` column (i.e. has been rebuilt/migrated post-refactor).

    The pre-refactor artifacts on disk still use the single-string ``source``
    schema; ``--source sentence0_reword`` correctly refuses to run against
    them. Rebuilding those artifacts is Stage 5 (post-merge user work), so
    the round-trip tests below ``pytest.skip`` until that has happened.
    """
    fl = LEAK_DIR_F10 / "contextual_labeled_firstleak"
    if not fl.exists():
        return False
    try:
        from datasets import load_from_disk
        dd = load_from_disk(str(fl))
        split = next(iter(dd.values())) if hasattr(dd, "values") else dd
        return "source_tags" in split.features
    except Exception:
        return False


def _have_aug_v4_inputs() -> bool:
    """Whether the on-disk firstleak + patches files exist for forget10
    AND the firstleak artifact has been migrated to the new schema."""
    return (
        (LEAK_DIR_F10 / "contextual_labeled_firstleak").exists()
        and (LEAK_DIR_F10 / "contextual_labeled_full_firstleak").exists()
        and (PATCHES_DIR_F10 / "aug_rewordings_v3.jsonl").exists()
        and (PATCHES_DIR_F10 / "aug_candidates.json").exists()
        and _firstleak_has_source_tags()
    )


import pytest  # noqa: E402


@pytest.mark.skipif(
    not _have_aug_v4_inputs(),
    reason="forget10 firstleak artifacts/patches missing OR not yet "
           "migrated to the source_tags schema (Stage 5 rebuild pending)",
)
def test_sentence0_reword_round_trip_against_existing_artifact(tmp_path):
    """Extract patches -> apply via new source-type -> verify byte-identical
    to existing _aug_v4. Uses a tmp copy of the firstleak inputs so we
    don't write into the live leak_detector_data_promptfmt tree."""
    tmp_leak_dir = tmp_path / "forget10_leak"
    tmp_leak_dir.mkdir()
    # Copy the two base inputs (firstleak + full_firstleak) into the tmp dir;
    # the script writes _aug_v4 alongside them.
    for name in ("contextual_labeled_firstleak", "contextual_labeled_full_firstleak"):
        shutil.copytree(LEAK_DIR_F10 / name, tmp_leak_dir / name)

    subprocess.run(
        [sys.executable, "-m", "scripts.build_labeled_sources",
         "--source", "sentence0_reword",
         "--split", "forget10",
         "--patches_dir", str(PATCHES_DIR_F10),
         "--leak_data_dir", str(tmp_leak_dir)],
        cwd=REPO, check=True,
    )

    # Compare the freshly-produced _aug_v4 against the on-disk one.
    new_clean = load_from_disk(
        str(tmp_leak_dir / "contextual_labeled_firstleak_aug_v4")
    )
    live_clean = load_from_disk(
        str(LEAK_DIR_F10 / "contextual_labeled_firstleak_aug_v4")
    )
    # Both sides are new-schema (the on-disk artifact was migrated/rebuilt in
    # Stage 5, and the fresh output uses the current writer), so compare
    # directly — NOT via the legacy-migration comparator.
    ok, msg = _datasets_equal_same_schema(new_clean, live_clean)
    assert ok, f"firstleak_aug_v4 round-trip mismatch: {msg}"

    new_full = load_from_disk(
        str(tmp_leak_dir / "contextual_labeled_full_firstleak_aug_v4")
    )
    live_full = load_from_disk(
        str(LEAK_DIR_F10 / "contextual_labeled_full_firstleak_aug_v4")
    )
    ok, msg = _datasets_equal_same_schema(new_full, live_full)
    assert ok, f"full_firstleak_aug_v4 round-trip mismatch: {msg}"


# No skipif: the forget10-only guard fires before any patches/leak file is read.
def test_sentence0_reword_rejects_non_forget10(tmp_path):
    """The forget10-only guard must fail loudly on forget05/forget01.
    Subprocess returncode != 0 + the expected error message in stderr.
    """
    for split in ("forget05", "forget01"):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.build_labeled_sources",
             "--source", "sentence0_reword",
             "--split", split,
             "--patches_dir", str(tmp_path),  # empty dir; won't be reached
             "--leak_data_dir", str(tmp_path)],
            cwd=REPO, capture_output=True, text=True,
        )
        assert result.returncode != 0, (
            f"Expected non-zero exit on --split {split}; got 0. "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert "forget10-only" in (result.stderr + result.stdout), (
            f"Expected 'forget10-only' in error output for --split {split}; "
            f"got stderr={result.stderr!r}"
        )


@pytest.mark.skipif(
    not _have_aug_v4_inputs(),
    reason="forget10 firstleak artifacts/patches missing OR not yet "
           "migrated to the source_tags schema (Stage 5 rebuild pending)",
)
def test_sentence0_reword_writes_both_outputs(tmp_path):
    """Both _firstleak_aug_v4 and _full_firstleak_aug_v4 must exist after run."""
    tmp_leak_dir = tmp_path / "forget10_leak"
    tmp_leak_dir.mkdir()
    for name in ("contextual_labeled_firstleak", "contextual_labeled_full_firstleak"):
        shutil.copytree(LEAK_DIR_F10 / name, tmp_leak_dir / name)

    out_clean = tmp_leak_dir / "contextual_labeled_firstleak_aug_v4"
    out_full = tmp_leak_dir / "contextual_labeled_full_firstleak_aug_v4"
    assert not out_clean.exists()
    assert not out_full.exists()

    subprocess.run(
        [sys.executable, "-m", "scripts.build_labeled_sources",
         "--source", "sentence0_reword",
         "--split", "forget10",
         "--patches_dir", str(PATCHES_DIR_F10),
         "--leak_data_dir", str(tmp_leak_dir)],
        cwd=REPO, check=True,
    )
    assert out_clean.exists(), f"missing output: {out_clean}"
    assert out_full.exists(), f"missing output: {out_full}"
    # Both should be loadable as DatasetDicts with train + test partitions.
    dd_clean = load_from_disk(str(out_clean))
    dd_full = load_from_disk(str(out_full))
    assert set(dd_clean.keys()) == {"train", "test"}
    assert set(dd_full.keys()) == {"train", "test"}


# ----------------------------------------------------------------------
# _save_dataset_dict augment routing (Change A)
# ----------------------------------------------------------------------

def test_save_dataset_dict_routes_augment_to_train_only(tmp_path):
    """Mixed augment + non-augment rows: augment (idk+paraphrase,
    forget+answer) go to train-only; non-augment (forget+clean, aux+clean)
    split by cot_idx group as before."""
    from scripts.build_labeled_sources import _save_dataset_dict
    from tests._fixtures import make_row

    # Non-augment rows across several cot_idxs (so the cot_idx-group split
    # has something to partition). Use enough groups that test_frac=0.5
    # actually puts some in test.
    rows: list[dict] = []
    for cot_idx in range(8):
        rows.append(make_row(source="forget", tags=["clean"], cot_idx=cot_idx,
                             label=1, sent_idx=0))
        rows.append(make_row(source="aux", tags=["clean"], cot_idx=100 + cot_idx,
                             sent_idx=0))
    non_augment_uids = {r["uid"] for r in rows}
    # Augment rows: idk+paraphrase and forget+answer, on distinct cot_idxs.
    augment_rows = [
        make_row(source="idk", tags=["clean", "paraphrase"], cot_idx=200,
                 sent_idx=0),
        make_row(source="idk", tags=["clean", "paraphrase"], cot_idx=201,
                 sent_idx=1),
        make_row(source="forget", tags=["clean", "answer"], cot_idx=202,
                 label=1, sent_idx=0),
    ]
    augment_uids = {r["uid"] for r in augment_rows}
    all_rows = rows + augment_rows

    out = tmp_path / "out_mixed"
    _save_dataset_dict(all_rows, str(out), test_frac=0.5, seed=0)
    dd = load_from_disk(str(out))

    train_uids = set(dd["train"]["uid"])
    test_uids = set(dd["test"]["uid"])

    # (a) No augment row lands in test.
    assert not (augment_uids & test_uids), (
        f"augment rows leaked into test: {augment_uids & test_uids}"
    )
    # (b) All augment rows land in train.
    assert augment_uids <= train_uids, (
        f"missing augment rows from train: {augment_uids - train_uids}"
    )
    # (c) Non-augment rows are split by cot_idx group: every non-augment row
    #     appears in exactly one partition, and (with this fixture) both
    #     partitions are non-empty.
    assert non_augment_uids <= (train_uids | test_uids)
    assert not (train_uids & test_uids), "a uid appears in both partitions"
    na_train = non_augment_uids & train_uids
    na_test = non_augment_uids & test_uids
    assert na_train and na_test, (
        "expected the non-augment rows to be split across both partitions"
    )
    # Whole cot_idx groups stay together (the forget+aux rows here are 1
    # sentence each so this is trivially satisfied, but assert the contract).
    train_cots = set(dd["train"]["cot_idx"])
    test_cots = set(dd["test"]["cot_idx"])
    na_cots_train = {c for c in train_cots if c < 200}
    na_cots_test = {c for c in test_cots if c < 200}
    assert not (na_cots_train & na_cots_test), (
        "a non-augment cot_idx group was split across partitions"
    )


def test_save_dataset_dict_all_augment_empty_test(tmp_path):
    """When EVERY row is augment (the forget_answers case), the test
    partition is a valid empty Dataset with the full schema."""
    from scripts.build_labeled_sources import _save_dataset_dict, CONTEXTUAL_FEATURES
    from tests._fixtures import make_row

    rows = [
        make_row(source="forget", tags=["clean", "answer"], cot_idx=i,
                 label=1, sent_idx=0)
        for i in range(5)
    ]
    out = tmp_path / "out_all_aug"
    _save_dataset_dict(rows, str(out), test_frac=0.5, seed=0)
    dd = load_from_disk(str(out))
    assert len(dd["test"]) == 0
    assert len(dd["train"]) == 5
    # Empty test split still carries the canonical feature schema.
    assert set(dd["test"].features) == set(CONTEXTUAL_FEATURES)
