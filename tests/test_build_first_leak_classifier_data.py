"""Smoke tests for scripts/build_first_leak_classifier_data.py.

Builds a small synthetic clean_forget DatasetDict + first-leak labels JSONL
in a temp dir, runs ``truncate_clean_forget`` directly, and asserts the
documented invariants:

  - For each CoT in labels: exactly first_leak_idx+1 rows kept, all phase='cot'
  - Pre-leak rows: label=0, matched_claim_idx=None
  - First-leak row: label=1, matched_claim_idx carried from the labels
  - ans-phase rows dropped
  - non raw-clean-forget rows (source != 'forget' or tags != {'clean'}) raise
  - first_leak_idx > max sent_idx raises
  - matched_claim_idx=null with disallowed source raises
  - matched_claim_idx=null with 'manual_old' source falls back to -1 sentinel
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset, Features, Sequence, Value

from datasets import DatasetDict

from scripts.build_first_leak_classifier_data import (
    IDK_SOURCE_NAME,
    UNKNOWN_CLAIM_SENTINEL,
    load_first_leak_labels,
    maybe_subsample_idk_source,
    parse_subsample_target,
    truncate_clean_forget,
)
from tests._fixtures import legacy_source_to_new


SCHEMA = Features({
    "uid": Value("string"),
    "prompt": Value("string"),
    "text": Value("string"),
    "context_sentences": Sequence(Value("string")),
    "phase": Value("string"),
    "author_slug": Value("string"),
    "cot_idx": Value("int64"),
    "sent_idx": Value("int64"),
    "label": Value("int64"),
    "matched_claim_idx": Value("int64"),  # nullable
    "claim_snippet": Value("string"),  # nullable
    "source": Value("string"),
    "source_tags": Sequence(Value("string")),
    "escalated": Value("bool"),
    "rationale": Value("string"),
})


def _row(cot_idx, phase, sent_idx, label=0, mci=None, source="clean_forget"):
    primary, tags = legacy_source_to_new(source)
    return {
        "uid": f"a_q{cot_idx:02d}_{phase}_s{sent_idx:03d}",
        "prompt": f"Question for cot {cot_idx}",
        "text": f"sentence {cot_idx}.{phase}.{sent_idx}",
        "context_sentences": [],
        "phase": phase,
        "author_slug": "a",
        "cot_idx": cot_idx,
        "sent_idx": sent_idx,
        "label": label,
        "matched_claim_idx": mci,
        "claim_snippet": None,
        "source": primary,
        "source_tags": list(tags),
        "escalated": False,
        "rationale": "original rationale" if label == 1 else "",
    }


def _build_ds(rows):
    return Dataset.from_list(rows, features=SCHEMA)


def test_truncate_basic_three_cots():
    """3 CoTs with different first-leak positions; all sentences in CoT phase."""
    rows = []
    # cot 0: leak at sent_idx=0 (immediate). 3 cot sents, 1 ans sent.
    rows += [_row(0, "cot", 0, label=1, mci=10),
             _row(0, "cot", 1, label=0),
             _row(0, "cot", 2, label=1, mci=11),
             _row(0, "ans", 0, label=1, mci=12)]
    # cot 1: leak at sent_idx=2. 4 cot sents.
    rows += [_row(1, "cot", 0, label=0),
             _row(1, "cot", 1, label=0),
             _row(1, "cot", 2, label=1, mci=11),
             _row(1, "cot", 3, label=1, mci=12)]
    # cot 2: leak at sent_idx=1 with matched_claim_idx already correct.
    rows += [_row(2, "cot", 0, label=0),
             _row(2, "cot", 1, label=1, mci=10)]
    ds = _build_ds(rows)
    labels = {
        0: {"row_index": 0, "first_leak_idx": 0, "matched_claim_idx": 10, "source": "round1"},
        1: {"row_index": 1, "first_leak_idx": 2, "matched_claim_idx": 11, "source": "round1"},
        2: {"row_index": 2, "first_leak_idx": 1, "matched_claim_idx": 10, "source": "round1"},
    }

    out, summary = truncate_clean_forget(ds, labels)
    by_cot = {}
    for r in out:
        by_cot.setdefault(r["cot_idx"], []).append(r)

    # cot 0: leak at 0 -> 1 row, label=1
    assert len(by_cot[0]) == 1
    assert by_cot[0][0]["sent_idx"] == 0
    assert by_cot[0][0]["label"] == 1
    assert by_cot[0][0]["matched_claim_idx"] == 10
    # cot 1: leak at 2 -> 3 rows. First two label=0 (one was originally 0,
    # one was originally 1 and gets demoted; rationale cleared).
    assert len(by_cot[1]) == 3
    sorted1 = sorted(by_cot[1], key=lambda r: r["sent_idx"])
    assert [r["label"] for r in sorted1] == [0, 0, 1]
    assert [r["matched_claim_idx"] for r in sorted1] == [None, None, 11]
    assert sorted1[-1]["matched_claim_idx"] == 11
    # cot 2: leak at 1 -> 2 rows
    assert len(by_cot[2]) == 2
    sorted2 = sorted(by_cot[2], key=lambda r: r["sent_idx"])
    assert [r["label"] for r in sorted2] == [0, 1]
    assert sorted2[-1]["matched_claim_idx"] == 10

    # ans-phase row dropped
    assert all(r["phase"] == "cot" for r in out)
    assert summary["n_dropped_ans_rows"] == 1
    # cot 0 dropped post-leak rows: original sent_idx 1 + 2 = 2 rows dropped
    # cot 1 dropped post-leak rows: sent_idx 3 = 1 row
    # cot 2: 0 post-leak. total = 3
    assert summary["n_dropped_post_leak_cot_rows"] == 3
    assert summary["label_hist_out"] == {1: 3, 0: 3}


def test_truncate_demote_clears_rationale():
    """When a previously-label=1 row is demoted to label=0, its rationale
    must be cleared to avoid downstream audits seeing a contradictory pair."""
    rows = [_row(0, "cot", 0, label=1, mci=10),
            _row(0, "cot", 1, label=1, mci=11)]
    ds = _build_ds(rows)
    labels = {0: {"row_index": 0, "first_leak_idx": 1, "matched_claim_idx": 11, "source": "round1"}}

    out, _ = truncate_clean_forget(ds, labels)
    by_idx = {r["sent_idx"]: r for r in out}
    assert by_idx[0]["label"] == 0
    assert by_idx[0]["rationale"] == ""  # cleared
    assert by_idx[1]["label"] == 1


def test_unknown_source_raises():
    rows = [_row(0, "cot", 0, source="clean_retain")]  # wrong source
    ds = _build_ds(rows)
    labels = {0: {"row_index": 0, "first_leak_idx": 0, "matched_claim_idx": 10, "source": "round1"}}
    # Post-refactor: the function-entry precondition assert fires first;
    # accept either the AssertionError from the precondition or the
    # ValueError from the inner per-row check.
    with pytest.raises((AssertionError, ValueError), match=r"clean forget"):
        truncate_clean_forget(ds, labels)


def test_unknown_phase_raises():
    rows = [_row(0, "cot", 0)]
    # mutate phase to something invalid
    rows[0]["phase"] = "system"
    ds = _build_ds(rows)
    labels = {0: {"row_index": 0, "first_leak_idx": 0, "matched_claim_idx": 10, "source": "round1"}}
    with pytest.raises(ValueError, match="Unexpected phase"):
        truncate_clean_forget(ds, labels)


def test_first_leak_idx_out_of_range_raises():
    rows = [_row(0, "cot", 0), _row(0, "cot", 1)]
    ds = _build_ds(rows)
    # first_leak_idx=5 but CoT only has 2 sentences (max sent_idx=1)
    labels = {0: {"row_index": 0, "first_leak_idx": 5, "matched_claim_idx": 10, "source": "round1"}}
    with pytest.raises(ValueError, match="exceeds max CoT sent_idx"):
        truncate_clean_forget(ds, labels)


def test_null_mci_disallowed_source_raises():
    rows = [_row(0, "cot", 0)]
    ds = _build_ds(rows)
    # round1 record with first_leak_idx set but matched_claim_idx null - bug
    labels = {0: {"row_index": 0, "first_leak_idx": 0, "matched_claim_idx": None, "source": "round1"}}
    with pytest.raises(ValueError, match="matched_claim_idx=null"):
        truncate_clean_forget(ds, labels)


def test_null_mci_manual_old_falls_back_to_sentinel():
    rows = [_row(0, "cot", 0)]
    ds = _build_ds(rows)
    labels = {0: {"row_index": 0, "first_leak_idx": 0, "matched_claim_idx": None, "source": "manual_old"}}
    out, _ = truncate_clean_forget(ds, labels)
    only = list(out)[0]
    assert only["label"] == 1
    assert only["matched_claim_idx"] == UNKNOWN_CLAIM_SENTINEL


def test_row_index_cot_idx_mismatch_raises():
    """The labels JSONL must use row_index == cot_idx. Any drift is a bug."""
    rows = [_row(0, "cot", 0)]
    ds = _build_ds(rows)
    labels = {0: {"row_index": 7, "first_leak_idx": 0, "matched_claim_idx": 10, "source": "round1"}}
    with pytest.raises(ValueError, match="row_index/cot_idx mismatch"):
        truncate_clean_forget(ds, labels)


def test_negative_non_sentinel_mci_raises():
    """A label=1 labels record with a negative matched_claim_idx that is NOT
    the -1 "uncertain leak" sentinel is a labels-file bug. This guard is the
    surviving half of the retired ``derive_cluster_id`` validation."""
    rows = [_row(0, "cot", 0)]
    ds = _build_ds(rows)
    labels = {0: {"row_index": 0, "first_leak_idx": 0, "matched_claim_idx": -5, "source": "round1"}}
    with pytest.raises(ValueError, match="invalid matched_claim_idx"):
        truncate_clean_forget(ds, labels)


def test_mci_at_or_above_n_claims_raises():
    """The other half of the retired ``derive_cluster_id`` validation: it
    required matched_claim_idx to be a key of claim_to_cluster, which spanned
    exactly [0, n_claims). Without this an out-of-range index flows silently
    into the classifier data and the soft-token data."""
    rows = [_row(0, "cot", 0)]
    ds = _build_ds(rows)
    labels = {0: {"row_index": 0, "first_leak_idx": 0, "matched_claim_idx": 509, "source": "round1"}}
    with pytest.raises(ValueError, match=r"outside \[0, 509\)"):
        truncate_clean_forget(ds, labels, n_claims=509)


def test_mci_within_n_claims_accepted():
    """The last valid index must not trip the upper bound (off-by-one guard)."""
    rows = [_row(0, "cot", 0)]
    ds = _build_ds(rows)
    labels = {0: {"row_index": 0, "first_leak_idx": 0, "matched_claim_idx": 508, "source": "round1"}}
    out, _ = truncate_clean_forget(ds, labels, n_claims=509)
    assert out[0]["matched_claim_idx"] == 508


def test_n_claims_none_skips_upper_bound():
    """n_claims is optional; when claims.json is absent the bound is not
    enforced rather than the build failing."""
    rows = [_row(0, "cot", 0)]
    ds = _build_ds(rows)
    labels = {0: {"row_index": 0, "first_leak_idx": 0, "matched_claim_idx": 99999, "source": "round1"}}
    out, _ = truncate_clean_forget(ds, labels)
    assert out[0]["matched_claim_idx"] == 99999


def test_load_first_leak_labels_rejects_non_ok(tmp_path: Path):
    p = tmp_path / "labels.jsonl"
    p.write_text(json.dumps({
        "row_index": 0, "first_leak_idx": 1, "matched_claim_idx": 10, "status": "parse_failed",
    }) + "\n")
    with pytest.raises(ValueError, match="status="):
        load_first_leak_labels(p)


def test_load_first_leak_labels_rejects_duplicates(tmp_path: Path):
    p = tmp_path / "labels.jsonl"
    line = json.dumps({
        "row_index": 0, "first_leak_idx": 1, "matched_claim_idx": 10, "status": "ok",
    })
    p.write_text(line + "\n" + line + "\n")
    with pytest.raises(ValueError, match="duplicate row_index"):
        load_first_leak_labels(p)


# ---------------------------------------------------------------------------
# maybe_subsample_idk_source (unified IDK source path)
# ---------------------------------------------------------------------------


def _idk_row(uid: str, source: str, sent_idx: int = 0) -> dict:
    primary, tags = legacy_source_to_new(source)
    return {
        "uid": uid,
        "prompt": "Q?",
        "text": f"sentence-{uid}",
        "context_sentences": [],
        "phase": "ans",
        "author_slug": source,
        "cot_idx": 0,
        "sent_idx": sent_idx,
        "label": 0,
        "matched_claim_idx": None,
        "claim_snippet": None,
        "source": primary,
        "source_tags": list(tags),
        "escalated": False,
        "rationale": "",
    }


def _make_idk_dd(n_train: int, n_test: int) -> DatasetDict:
    """Build a DatasetDict for the unified IDK source. Half prose, half paraphrase."""
    train_rows = []
    for i in range(n_train):
        src = "clean_idk" if i % 2 == 0 else "clean_idk_paraphrase"
        train_rows.append(_idk_row(f"train_{i}", src, sent_idx=i))
    test_rows = []
    for i in range(n_test):
        src = "clean_idk" if i % 2 == 0 else "clean_idk_paraphrase"
        test_rows.append(_idk_row(f"test_{i}", src, sent_idx=i))
    return DatasetDict({
        "train": Dataset.from_list(train_rows, features=SCHEMA),
        "test": Dataset.from_list(test_rows, features=SCHEMA),
    })


def test_subsample_idk_source_trims_each_partition():
    """target {train=N, test=M} subsamples each partition to its target size."""
    neg_dds = {IDK_SOURCE_NAME: _make_idk_dd(n_train=200, n_test=80)}
    out = maybe_subsample_idk_source(neg_dds, {"train": 50, "test": 25}, seed=0)
    assert len(out[IDK_SOURCE_NAME]["train"]) == 50
    assert len(out[IDK_SOURCE_NAME]["test"]) == 25


def _has_paraphrase_rows(ds) -> bool:
    return any("paraphrase" in row["source_tags"] for row in ds)


def _has_prose_rows(ds) -> bool:
    return any(
        row["source"] == "idk" and "paraphrase" not in row["source_tags"]
        for row in ds
    )


def test_subsample_idk_source_preserves_source_mix():
    """Subsampling keeps both prose and paraphrase rows when target exceeds
    the prose pool (the main path under prose-first semantics)."""
    neg_dds = {IDK_SOURCE_NAME: _make_idk_dd(n_train=200, n_test=80)}
    # Fixture has 100 prose + 100 paraphrase train, 40 prose + 40 paraphrase
    # test. Targets above the prose pool exercise the main "keep all prose
    # + sample paraphrase" path that mixes the two sources.
    out = maybe_subsample_idk_source(neg_dds, {"train": 150, "test": 60}, seed=0)
    # Under the (source, source_tags) schema, paraphrase is a tag rather
    # than a separate source primary; check both variants are present.
    assert _has_prose_rows(out[IDK_SOURCE_NAME]["train"])
    assert _has_paraphrase_rows(out[IDK_SOURCE_NAME]["train"])
    assert _has_prose_rows(out[IDK_SOURCE_NAME]["test"])
    assert _has_paraphrase_rows(out[IDK_SOURCE_NAME]["test"])


def test_subsample_idk_source_below_prose_pool_is_prose_only():
    """When target <= prose pool, prose-first collapses to prose-only with a
    WARN. Codifies the warning branch of subsample_idk_prose_first so a
    future tightening doesn't silently flip semantics."""
    neg_dds = {IDK_SOURCE_NAME: _make_idk_dd(n_train=200, n_test=80)}
    # Fixture has 100 prose train / 40 prose test; targets exactly at the
    # prose pool size trigger the warning branch.
    out = maybe_subsample_idk_source(neg_dds, {"train": 100, "test": 40}, seed=0)
    # Every surviving row is prose: source='idk' AND no 'paraphrase' tag.
    for partition in ("train", "test"):
        for row in out[IDK_SOURCE_NAME][partition]:
            assert row["source"] == "idk"
            assert "paraphrase" not in row["source_tags"]
    assert len(out[IDK_SOURCE_NAME]["train"]) == 100
    assert len(out[IDK_SOURCE_NAME]["test"]) == 40


def test_subsample_idk_source_no_target_is_noop():
    neg_dds = {IDK_SOURCE_NAME: _make_idk_dd(n_train=50, n_test=20)}
    out = maybe_subsample_idk_source(neg_dds, None, seed=0)
    assert out is neg_dds  # exact passthrough


def test_subsample_idk_source_warns_when_smaller_than_target():
    """When pool is already below target, keep all rows (warn but don't fail)."""
    neg_dds = {IDK_SOURCE_NAME: _make_idk_dd(n_train=30, n_test=10)}
    out = maybe_subsample_idk_source(neg_dds, {"train": 100, "test": 50}, seed=0)
    assert len(out[IDK_SOURCE_NAME]["train"]) == 30
    assert len(out[IDK_SOURCE_NAME]["test"]) == 10


def test_subsample_idk_source_missing_artifact_is_noop():
    """When IDK source isn't in neg_dds, function returns input unchanged."""
    neg_dds = {"contextual_labeled_retain": _make_idk_dd(n_train=5, n_test=2)}
    out = maybe_subsample_idk_source(neg_dds, {"train": 1, "test": 1}, seed=0)
    assert out is neg_dds


def test_subsample_idk_source_unknown_partition_raises():
    neg_dds = {IDK_SOURCE_NAME: _make_idk_dd(n_train=50, n_test=20)}
    with pytest.raises(ValueError, match="Unknown partition"):
        maybe_subsample_idk_source(neg_dds, {"tarin": 100}, seed=0)


def test_subsample_idk_source_all_target_applies_uniformly():
    """target with 'all=N' applies the same size to every partition."""
    neg_dds = {IDK_SOURCE_NAME: _make_idk_dd(n_train=100, n_test=50)}
    out = maybe_subsample_idk_source(neg_dds, {"all": 30}, seed=0)
    assert len(out[IDK_SOURCE_NAME]["train"]) == 30
    assert len(out[IDK_SOURCE_NAME]["test"]) == 30


# ---------------------------------------------------------------------------
# subsample_idk_prose_first — direct tests of the helper
# ---------------------------------------------------------------------------

def _idk_prose_para_ds(n_prose: int, n_para: int) -> Dataset:
    """Helper: build an IDK Dataset with `n_prose` clean_idk + `n_para` clean_idk_paraphrase rows."""
    rows = []
    for i in range(n_prose):
        rows.append(_idk_row(f"prose_{i}", "clean_idk", sent_idx=i))
    for i in range(n_para):
        rows.append(_idk_row(f"para_{i}", "clean_idk_paraphrase", sent_idx=i))
    return Dataset.from_list(rows, features=SCHEMA)


def _count_prose_paraphrase(ds) -> tuple[int, int]:
    n_prose = 0
    n_para = 0
    for row in ds:
        if row["source"] != "idk":
            continue
        if "paraphrase" in row["source_tags"]:
            n_para += 1
        else:
            n_prose += 1
    return n_prose, n_para


def test_subsample_idk_prose_first_keeps_all_prose_then_fills_para():
    """target > prose: keep all prose + sample paraphrase to fill."""
    from scripts.build_first_leak_classifier_data import subsample_idk_prose_first
    ds = _idk_prose_para_ds(n_prose=20, n_para=80)
    out = subsample_idk_prose_first(ds, n_target=50, seed=0, log_label="t")
    n_prose, n_para = _count_prose_paraphrase(out)
    assert n_prose == 20  # all prose kept
    assert n_para == 30  # filled to target
    assert len(out) == 50


def test_subsample_idk_prose_first_below_prose_pool_is_prose_only():
    """target <= prose count: warn + return target random prose."""
    from scripts.build_first_leak_classifier_data import subsample_idk_prose_first
    ds = _idk_prose_para_ds(n_prose=50, n_para=100)
    out = subsample_idk_prose_first(ds, n_target=30, seed=0, log_label="t")
    assert len(out) == 30
    n_prose, n_para = _count_prose_paraphrase(out)
    assert n_prose == 30
    assert n_para == 0


def test_subsample_idk_prose_first_insufficient_paraphrase_returns_all():
    """target > total: take everything available with warning."""
    from scripts.build_first_leak_classifier_data import subsample_idk_prose_first
    ds = _idk_prose_para_ds(n_prose=10, n_para=5)
    out = subsample_idk_prose_first(ds, n_target=100, seed=0, log_label="t")
    n_prose, n_para = _count_prose_paraphrase(out)
    assert n_prose == 10
    assert n_para == 5
    assert len(out) == 15


def test_subsample_idk_prose_first_no_paraphrase():
    """No paraphrase rows + target >= prose: return shuffled prose only."""
    from scripts.build_first_leak_classifier_data import subsample_idk_prose_first
    ds = _idk_prose_para_ds(n_prose=15, n_para=0)
    out = subsample_idk_prose_first(ds, n_target=15, seed=0, log_label="t")
    assert len(out) == 15
    n_prose, n_para = _count_prose_paraphrase(out)
    assert n_prose == 15
    assert n_para == 0


def test_subsample_idk_prose_first_excludes_unknown_sources():
    """Rows with unexpected source values are excluded from output with warning."""
    from scripts.build_first_leak_classifier_data import subsample_idk_prose_first
    rows = []
    for i in range(20):
        rows.append(_idk_row(f"prose_{i}", "clean_idk", sent_idx=i))
    for i in range(30):
        rows.append(_idk_row(f"para_{i}", "clean_idk_paraphrase", sent_idx=i))
    for i in range(10):
        # "Weird" rows: non-idk source the IDK prose-first helper should skip.
        rows.append(_idk_row(f"weird_{i}", "clean_aux", sent_idx=i))
    ds = Dataset.from_list(rows, features=SCHEMA)
    out = subsample_idk_prose_first(ds, n_target=40, seed=0, log_label="t")
    # Every surviving row must be source='idk'; the prose/paraphrase split
    # lives in source_tags.
    assert set(out["source"]) == {"idk"}
    assert len(out) == 40


def test_subsample_idk_prose_first_empty_input():
    """Empty input is a no-op — return empty dataset, no errors.

    Build the empty dataset via from_dict (column dict with empty lists)
    rather than from_list([], features=SCHEMA) — the latter trips a
    Features.reorder_fields_as mismatch since an empty arrow file has
    no schema to merge against the provided Features.
    """
    from scripts.build_first_leak_classifier_data import subsample_idk_prose_first
    ds = Dataset.from_dict({col: [] for col in SCHEMA}, features=SCHEMA)
    out = subsample_idk_prose_first(ds, n_target=100, seed=0, log_label="t")
    assert len(out) == 0


def test_parse_subsample_target_basic():
    assert parse_subsample_target("train=10,test=5") == {"train": 10, "test": 5}
    assert parse_subsample_target("all=42") == {"all": 42}
    assert parse_subsample_target("none") is None
    assert parse_subsample_target("") is None


def test_parse_subsample_target_invalid():
    with pytest.raises(ValueError):
        parse_subsample_target("train=")
    with pytest.raises(ValueError):
        parse_subsample_target("all=10,train=5")


# ---------------------------------------------------------------------------
# build_subset_classifier_data: idk_subsample_target_base 'all=N' symmetry
# ---------------------------------------------------------------------------
#
# This block tests the same 'all=N' semantics that maybe_subsample_idk_source
# implements above, but for build_subset_classifier_data's per-partition
# scaling path. Both should honor 'all=N' identically; previously the subset
# script silently no-op'd on 'all=N' because the per-partition lookup only
# checked 'train' / 'test' keys.


def test_subset_idk_all_scaling_simulation():
    """Simulate the 'all=N' scaling path that build_subset_classifier_data
    runs after parse_subsample_target. The dict comprehension must
    preserve 'all' so the downstream per-partition lookup hits it.
    """
    base = parse_subsample_target("all=1000")
    assert base == {"all": 1000}
    # Simulate the frac-scaling in build_subset_classifier_data.build_subset
    frac = 0.5
    scaled = {s: int(round(n * frac)) for s, n in base.items()}
    assert scaled == {"all": 500}, (
        "frac scaling must preserve the 'all' key so the per-partition "
        "lookup downstream can branch on it"
    )
