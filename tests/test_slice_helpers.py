"""Tests for the subset-split reuse helpers: slice_kb_resolutions and
slice_clean_labels.

Usage:
    conda run -n redact python -m pytest tests/test_slice_helpers.py -v
"""
import json
import os
import subprocess
import sys

import pytest
from datasets import Dataset, Features, Sequence, Value

from scripts.slice_clean_labels import build_claim_remap, remap_row, slice_split
from tests._fixtures import legacy_source_to_new


# ---------------------------------------------------------------------------
# remap_row (the core transfer logic)
# ---------------------------------------------------------------------------


SUBSET_KB_SIZE = 8  # number of claims in the synthetic subset KB
N_SUBSET_FORGET = 40


def _assert_source(row, legacy_source: str) -> None:
    """Assert ``row`` matches the new ``(source, source_tags)`` form of
    the legacy single-string ``source`` value. Centralises the schema
    translation across the test assertions in this file.
    """
    primary, tags = legacy_source_to_new(legacy_source)
    assert row["source"] == primary, (
        f"expected source={primary!r} (for legacy {legacy_source!r}); "
        f"got {row['source']!r}"
    )
    assert set(row["source_tags"]) == set(tags), (
        f"expected source_tags=={tags!r} (for legacy {legacy_source!r}); "
        f"got {row['source_tags']!r}"
    )


def _row(
    *,
    matched_claim_idx,
    cot_idx,
    label=None,
    sent_idx=0,
    rationale="",
    escalated=False,
    text="some sentence",
    uid="uid-1",
    source="clean_forget",
):
    """Build a labeled-DatasetDict-style row dict for remap_row.

    ``source`` accepts the legacy single-string form (e.g. ``"clean_forget"``)
    and is unpacked into the new ``(source, source_tags)`` schema. Pass an
    arbitrary ``corrupted_<scheme>`` value to exercise the rejection path.
    """
    if label is None:
        label = 1 if matched_claim_idx is not None and matched_claim_idx != -1 else 0
    primary, tags = legacy_source_to_new(source)
    return {
        "uid": uid,
        "text": text,
        "label": label,
        "source": primary,
        "source_tags": list(tags),
        "cot_idx": cot_idx,
        "sent_idx": sent_idx,
        "matched_claim_idx": matched_claim_idx,
        "rationale": rationale,
        "escalated": escalated,
    }


class TestRemapRow:
    def test_in_subset_claim_in_subset_prompt(self):
        """Claim is in subset KB → leak=1, matched_claim_idx preserved."""
        row = _row(matched_claim_idx=3, cot_idx=5)  # cot_idx < 40
        out = remap_row(row, SUBSET_KB_SIZE, N_SUBSET_FORGET)
        assert out["label"] == 1
        assert out["matched_claim_idx"] == 3
        _assert_source(out, "clean_forget")

    def test_in_subset_claim_out_of_subset_prompt(self):
        """A leak of subset content that appeared in an out-of-subset CoT still
        flags label=1; prompt membership only controls source tagging."""
        row = _row(matched_claim_idx=3, cot_idx=100)  # cot_idx >= 40
        out = remap_row(row, SUBSET_KB_SIZE, N_SUBSET_FORGET)
        assert out["label"] == 1
        assert out["matched_claim_idx"] == 3
        _assert_source(out, "clean_retain_xfer")

    def test_out_of_subset_claim_demotes_to_background(self):
        """Claim outside subset KB → retain-space discussion, leak=0."""
        row = _row(matched_claim_idx=100, cot_idx=5)
        out = remap_row(row, SUBSET_KB_SIZE, N_SUBSET_FORGET)
        assert out["label"] == 0
        assert out["matched_claim_idx"] == -1
        _assert_source(out, "clean_forget")  # prompt is in-subset

    def test_uncertain_in_subset_prompt_stays_uncertain(self):
        """Uncertain leak on an in-subset prompt → keep as uncertain sentinel."""
        row = _row(matched_claim_idx=-1, cot_idx=5, label=1)
        out = remap_row(row, SUBSET_KB_SIZE, N_SUBSET_FORGET)
        assert out["label"] == 1
        assert out["matched_claim_idx"] == -1  # uncertain sentinel
        _assert_source(out, "clean_forget")

    def test_uncertain_out_of_subset_prompt_demotes(self):
        """Uncertain leak on an out-of-subset prompt → retain-space, label=0."""
        row = _row(matched_claim_idx=-1, cot_idx=100, label=1)
        out = remap_row(row, SUBSET_KB_SIZE, N_SUBSET_FORGET)
        assert out["label"] == 0
        assert out["matched_claim_idx"] == -1
        _assert_source(out, "clean_retain_xfer")

    def test_original_non_leak_in_subset_prompt(self):
        """A label=0 row from the source stays label=0 and becomes background."""
        row = _row(matched_claim_idx=-1, cot_idx=5, label=0)
        out = remap_row(row, SUBSET_KB_SIZE, N_SUBSET_FORGET)
        assert out["label"] == 0
        assert out["matched_claim_idx"] == -1

    def test_source_tagging_at_boundary(self):
        """cot_idx == N_subset_forget is the first out-of-subset prompt."""
        # cot_idx 39 is the last in-subset
        row_in = _row(matched_claim_idx=3, cot_idx=39)
        out_in = remap_row(row_in, SUBSET_KB_SIZE, 40)
        _assert_source(out_in, "clean_forget")

        # cot_idx 40 is first out-of-subset
        row_out = _row(matched_claim_idx=3, cot_idx=40)
        out_out = remap_row(row_out, SUBSET_KB_SIZE, 40)
        _assert_source(out_out, "clean_retain_xfer")

    def test_label1_with_matched_claim_idx_none_raises(self):
        """aggregate_labels drops label=1 rows with matched_claim_idx=None;
        if one reaches remap_row it's an upstream regression, not silent data."""
        row = _row(matched_claim_idx=None, cot_idx=5, label=1)
        with pytest.raises(ValueError, match="label=1 but matched_claim_idx=None"):
            remap_row(row, SUBSET_KB_SIZE, N_SUBSET_FORGET)

    def test_label0_with_matched_claim_idx_none_is_background(self):
        """A label=0 row with matched_claim_idx=None is valid (aggregate_labels
        emits this for non-leaks) and remaps to retain-space background."""
        row = _row(matched_claim_idx=None, cot_idx=5, label=0)
        out = remap_row(row, SUBSET_KB_SIZE, N_SUBSET_FORGET)
        assert out["label"] == 0
        assert out["matched_claim_idx"] == -1

    def test_negative_non_sentinel_matched_claim_idx_raises(self):
        row = _row(matched_claim_idx=-5, cot_idx=5, label=1)
        with pytest.raises(ValueError, match="only -1 or non-negative"):
            remap_row(row, SUBSET_KB_SIZE, N_SUBSET_FORGET)

    def test_claim_idx_at_boundary(self):
        """matched_claim_idx = subset_kb_size - 1 is in-subset (strict < on the high side);
        matched_claim_idx = subset_kb_size is out-of-subset."""
        last_in = SUBSET_KB_SIZE - 1  # 7
        row_in = _row(matched_claim_idx=last_in, cot_idx=5)
        out_in = remap_row(row_in, SUBSET_KB_SIZE, N_SUBSET_FORGET)
        assert out_in["label"] == 1
        assert out_in["matched_claim_idx"] == last_in

        first_out = SUBSET_KB_SIZE  # 8
        row_out = _row(matched_claim_idx=first_out, cot_idx=5)
        out_out = remap_row(row_out, SUBSET_KB_SIZE, N_SUBSET_FORGET)
        assert out_out["label"] == 0
        assert out_out["matched_claim_idx"] == -1

    def test_preserves_passthrough_fields(self):
        row = _row(
            matched_claim_idx=3,
            cot_idx=5,
            uid="custom-uid",
            sent_idx=7,
            text="the sentence text",
            rationale="because reasons",
            escalated=True,
        )
        out = remap_row(row, SUBSET_KB_SIZE, N_SUBSET_FORGET)
        assert out["uid"] == "custom-uid"
        assert out["sent_idx"] == 7
        assert out["text"] == "the sentence text"
        assert out["rationale"] == "because reasons"
        assert out["escalated"] is True


class TestRemapRowCleanRetain:
    """clean_retain rows: source stays clean_retain, claim idx remaps,
    uncertain is always preserved (no prompt-membership test since all retain
    prompts are retain under any subset by R-TOFU nesting)."""

    def test_clean_retain_in_subset_claim(self):
        row = _row(matched_claim_idx=3, cot_idx=5, source="clean_retain")
        out = remap_row(row, SUBSET_KB_SIZE, N_SUBSET_FORGET)
        _assert_source(out, "clean_retain")
        assert out["label"] == 1
        assert out["matched_claim_idx"] == 3

    def test_clean_retain_out_of_subset_claim_demotes(self):
        row = _row(matched_claim_idx=100, cot_idx=5, source="clean_retain")
        out = remap_row(row, SUBSET_KB_SIZE, N_SUBSET_FORGET)
        _assert_source(out, "clean_retain")
        assert out["label"] == 0
        assert out["matched_claim_idx"] == -1

    def test_clean_retain_uncertain_preserved_regardless_of_cot_idx(self):
        """clean_retain uncertain stays uncertain even when cot_idx is large —
        there's no 'out-of-subset prompt' demotion path for retain."""
        for cot_idx in (0, 5, 100, 1000):
            row = _row(matched_claim_idx=-1, cot_idx=cot_idx, label=1, source="clean_retain")
            out = remap_row(row, SUBSET_KB_SIZE, N_SUBSET_FORGET)
            _assert_source(out, "clean_retain")
            assert out["label"] == 1
            assert out["matched_claim_idx"] == -1  # uncertain sentinel

    def test_clean_retain_label0_stays_background(self):
        row = _row(matched_claim_idx=None, cot_idx=5, label=0, source="clean_retain")
        out = remap_row(row, SUBSET_KB_SIZE, N_SUBSET_FORGET)
        _assert_source(out, "clean_retain")
        assert out["label"] == 0
        assert out["matched_claim_idx"] == -1

    def test_unsupported_source_raises(self):
        row = _row(matched_claim_idx=3, cot_idx=5, source="corrupted_something")
        # Corrupted rows trip the per-row assert allowlist before the
        # set-equality gate; both convey the same intent (off-pattern row
        # rejected) so accept either.
        with pytest.raises((AssertionError, ValueError), match=r"corrupted"):
            remap_row(row, SUBSET_KB_SIZE, N_SUBSET_FORGET)


# ---------------------------------------------------------------------------
# claim_remap (text-based source→subset idx mapping)
# ---------------------------------------------------------------------------


class TestBuildClaimRemap:
    """build_claim_remap is the text-based fix for KB misalignment between
    source and subset KBs. forget05 vs forget10 was the motivating case:
    forget05's KB is not prefix-aligned with forget10's, so the legacy
    'matched_claim_idx < subset_kb_size' check both demoted real leaks
    (source idx beyond subset_kb_size but text in subset) and mis-resolved
    others (source idx within subset_kb_size but the two KBs disagree at that
    index).
    """

    def _write_kb(self, tmpdir, name, claims):
        path = os.path.join(tmpdir, name)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "claims.json"), "w") as f:
            json.dump(claims, f)
        return path

    def test_prefix_aligned_kbs(self, tmp_path):
        """If subset is a prefix of source, remap is identity over the prefix."""
        src = self._write_kb(str(tmp_path), "src", ["a", "b", "c", "d", "e"])
        sub = self._write_kb(str(tmp_path), "sub", ["a", "b", "c"])
        remap = build_claim_remap(src, sub)
        assert remap == {0: 0, 1: 1, 2: 2}

    def test_subset_disordered_relative_to_source(self, tmp_path):
        """forget05-like case: subset claims sit at non-prefix positions
        in the source KB."""
        src = self._write_kb(str(tmp_path), "src", ["a", "b", "c", "d", "e", "f"])
        sub = self._write_kb(str(tmp_path), "sub", ["a", "f", "d"])
        remap = build_claim_remap(src, sub)
        # subset[0]=a is source[0]; subset[1]=f is source[5]; subset[2]=d is source[3]
        assert remap == {0: 0, 3: 2, 5: 1}

    def test_source_claims_not_in_subset_omitted(self, tmp_path):
        src = self._write_kb(str(tmp_path), "src", ["a", "b", "c"])
        sub = self._write_kb(str(tmp_path), "sub", ["a"])
        remap = build_claim_remap(src, sub)
        assert remap == {0: 0}

    def test_duplicate_source_claims_keep_first(self, tmp_path):
        """If the source KB has a duplicate, the remap points the first
        occurrence at the subset idx. (R-TOFU build_kb dedups, so this is a
        defensive guarantee, not a hot path.)"""
        src = self._write_kb(str(tmp_path), "src", ["a", "b", "a", "c"])
        sub = self._write_kb(str(tmp_path), "sub", ["a", "b"])
        remap = build_claim_remap(src, sub)
        # Both source[0] and source[2] are "a"; the first remap call wins for
        # subset[0]=a (setdefault); but source[2] also gets remapped to
        # subset[0] because we iterate all source positions.
        assert remap[0] == 0
        assert remap[1] == 1
        assert remap[2] == 0  # source dup also resolves to subset[0]


class TestRemapRowWithClaimRemap:
    """When claim_remap is supplied, the prefix-alignment fallback is
    bypassed and the source→subset idx lookup decides in-vs-out-of-subset.
    """

    def test_explicit_remap_keeps_in_subset_high_source_idx(self):
        """Source idx beyond subset_kb_size still counts as in-subset when
        the text has a counterpart in the subset KB (the forget05 fix)."""
        # forget10[506] / forget05[250] analogue
        remap = {12: 3}  # source idx 12 -> subset idx 3
        row = _row(matched_claim_idx=12, cot_idx=5)  # cot_idx < N_SUBSET_FORGET
        out = remap_row(
            row, SUBSET_KB_SIZE, N_SUBSET_FORGET,
            claim_remap=remap,
        )
        assert out["label"] == 1
        assert out["matched_claim_idx"] == 3

    def test_explicit_remap_demotes_low_source_idx_when_text_absent(self):
        """Source idx within subset_kb_size still demotes when the text
        isn't in the subset KB (the forget10[248]/forget05[248] disagreement
        case)."""
        remap = {0: 0, 1: 1}  # source idx 3 missing -> demote
        row = _row(matched_claim_idx=3, cot_idx=5)
        out = remap_row(
            row, SUBSET_KB_SIZE, N_SUBSET_FORGET,
            claim_remap=remap,
        )
        assert out["label"] == 0
        assert out["matched_claim_idx"] == -1

    def test_explicit_remap_does_not_affect_uncertain_sentinel(self):
        """matched_claim_idx == -1 (uncertain leak) is preserved regardless
        of claim_remap content — the remap only governs concrete claim idxs.
        """
        remap = {}  # empty
        row = _row(matched_claim_idx=-1, cot_idx=5, label=1)
        out = remap_row(
            row, SUBSET_KB_SIZE, N_SUBSET_FORGET,
            claim_remap=remap,
        )
        assert out["label"] == 1
        assert out["matched_claim_idx"] == -1  # uncertain sentinel

    def test_legacy_behavior_preserved_when_remap_is_none(self):
        """No claim_remap → fall back to prefix-alignment check (legacy path
        for prefix-aligned splits like forget01)."""
        row = _row(matched_claim_idx=3, cot_idx=5)
        out = remap_row(
            row, SUBSET_KB_SIZE, N_SUBSET_FORGET,
            claim_remap=None,
        )
        assert out["label"] == 1
        assert out["matched_claim_idx"] == 3

    def test_empty_claim_remap_demotes_all_concrete_leaks(self):
        """An empty remap models a subset KB with no source overlap; every
        concrete leak idx misses and demotes to retain-space."""
        row = _row(matched_claim_idx=3, cot_idx=5)
        out = remap_row(
            row, SUBSET_KB_SIZE, N_SUBSET_FORGET,
            claim_remap={},
        )
        assert out["label"] == 0
        assert out["matched_claim_idx"] == -1


# ---------------------------------------------------------------------------
# slice_split end-to-end with claim_remap
# ---------------------------------------------------------------------------


_SLICE_FEATURES = Features({
    "uid": Value("string"),
    "text": Value("string"),
    "label": Value("int64"),
    "source": Value("string"),
    "source_tags": Sequence(Value("string")),
    "cot_idx": Value("int64"),
    "sent_idx": Value("int64"),
    "matched_claim_idx": Value("int64"),
    "rationale": Value("string"),
    "escalated": Value("bool"),
    "phase": Value("string"),
    "context_sentences": Sequence(Value("string")),
})


def _slice_row(uid, source, cot_idx, mci, label):
    """Build a slice-test row dict.

    ``source`` is a legacy single-string value (e.g. ``"clean_forget"``);
    translated here into the new ``(source, source_tags)`` schema.
    """
    primary, tags = legacy_source_to_new(source)
    return {
        "uid": uid,
        "text": f"focal-{uid}",
        "label": label,
        "source": primary,
        "source_tags": list(tags),
        "cot_idx": cot_idx,
        "sent_idx": 0,
        "matched_claim_idx": mci,
        "rationale": "",
        "escalated": False,
        "phase": "cot",
        "context_sentences": [],
    }


class TestSliceSplitWithClaimRemap:
    """End-to-end: claim_remap kwarg threads through slice_split into
    remap_row and decides each row's source/label/claim-idx correctly. Covers
    the four wiring sites in build_subset_classifier_data (clean_forget,
    retain, idk, aux) by exercising one row from each source category.
    """

    def test_claim_remap_threads_through_each_source(self):
        rows = [
            # clean_forget in-subset prompt, high source idx → remap promotes
            _slice_row("cf_in_high", "clean_forget", cot_idx=5, mci=12, label=1),
            # clean_forget in-subset prompt, idx absent from remap → demote
            _slice_row("cf_in_miss", "clean_forget", cot_idx=5, mci=3, label=1),
            # clean_forget out-of-subset prompt with valid remap → becomes xfer
            _slice_row("cf_out", "clean_forget", cot_idx=100, mci=12, label=1),
            # clean_retain always in-subset, missing key → demote to background
            _slice_row("cr_miss", "clean_retain", cot_idx=5, mci=99, label=1),
            # clean_idk label=0 row, mci=None → pure background passthrough
            {**_slice_row("idk", "clean_idk", cot_idx=5, mci=None, label=0)},
            # clean_idk_paraphrase label=0 row → passthrough (added to slice
            # allowlist 2026-05-19 when paraphrase rows joined the unified
            # contextual_labeled_idk artifact)
            {**_slice_row("para", "clean_idk_paraphrase", cot_idx=5, mci=None, label=0)},
            # clean_aux label=0 row → pure background passthrough
            {**_slice_row("aux", "clean_aux", cot_idx=5, mci=None, label=0)},
            # corrupted_* → dropped
            _slice_row("corr", "corrupted_blank", cot_idx=5, mci=None, label=0),
        ]
        ds = Dataset.from_list(rows, features=_SLICE_FEATURES)
        remap = {12: 3}  # only source idx 12 has a subset counterpart (at 3)

        out, stats = slice_split(
            ds, SUBSET_KB_SIZE, N_SUBSET_FORGET, claim_remap=remap,
        )

        by_uid = {r["uid"]: r for r in out}
        # 1 corrupted row dropped → 7 rows out (added clean_idk_paraphrase)
        assert len(by_uid) == 7
        assert stats["dropped_corrupted"] == 1

        # cf_in_high: remap[12] = 3 → label=1, matched_claim_idx rebased
        assert by_uid["cf_in_high"]["label"] == 1
        assert by_uid["cf_in_high"]["matched_claim_idx"] == 3
        _assert_source(by_uid["cf_in_high"], "clean_forget")

        # cf_in_miss: remap missing → demote
        assert by_uid["cf_in_miss"]["label"] == 0
        assert by_uid["cf_in_miss"]["matched_claim_idx"] == -1

        # cf_out: in-remap claim but out-of-subset prompt → clean_retain_xfer,
        # label=1 (KB resolved), source flips
        _assert_source(by_uid["cf_out"], "clean_retain_xfer")
        assert by_uid["cf_out"]["label"] == 1

        # cr_miss: source-clean_retain, remap miss → demote
        _assert_source(by_uid["cr_miss"], "clean_retain")
        assert by_uid["cr_miss"]["label"] == 0
        assert by_uid["cr_miss"]["matched_claim_idx"] == -1

        # idk + paraphrase + aux: label=0, source preserved, mci=-1
        expected_source = {
            "idk": "clean_idk",
            "para": "clean_idk_paraphrase",
            "aux": "clean_aux",
        }
        for uid, src in expected_source.items():
            assert by_uid[uid]["label"] == 0
            assert by_uid[uid]["matched_claim_idx"] == -1
            _assert_source(by_uid[uid], src)


# ---------------------------------------------------------------------------
# slice_kb_resolutions (integration smoke test)
# ---------------------------------------------------------------------------


class TestSliceKbResolutions:
    def test_filters_by_claim_idx(self, tmp_path):
        src = tmp_path / "src.json"
        with open(src, "w") as f:
            json.dump({
                "resolutions": [
                    {"claim_idx": 5,  "original_claim": "a", "resolved_claim": "A"},
                    {"claim_idx": 49, "original_claim": "b", "resolved_claim": "B"},
                    {"claim_idx": 50, "original_claim": "c", "resolved_claim": "C"},
                    {"claim_idx": 200, "original_claim": "d", "resolved_claim": "D"},
                ]
            }, f)

        out = tmp_path / "out.json"
        subprocess.run(
            [
                sys.executable, "-m", "scripts.slice_kb_resolutions",
                "--input", str(src),
                "--max_claim_idx", "50",
                "--output", str(out),
            ],
            check=True, capture_output=True,
        )

        with open(out) as f:
            data = json.load(f)
        idxs = [r["claim_idx"] for r in data["resolutions"]]
        assert idxs == [5, 49]  # 50 excluded (strict less-than)

    def test_preserves_non_resolutions_top_level_keys(self, tmp_path):
        src = tmp_path / "src.json"
        with open(src, "w") as f:
            json.dump({
                "resolutions": [
                    {"claim_idx": 1, "original_claim": "a", "resolved_claim": "A"}
                ],
                "meta": {"run": "forget10-v1"},
            }, f)
        out = tmp_path / "out.json"
        subprocess.run(
            [
                sys.executable, "-m", "scripts.slice_kb_resolutions",
                "--input", str(src),
                "--max_claim_idx", "50",
                "--output", str(out),
            ],
            check=True, capture_output=True,
        )
        with open(out) as f:
            data = json.load(f)
        assert data.get("meta") == {"run": "forget10-v1"}
