"""Tests for the small helpers introduced by the pipeline refactor.

Covers:
- ``apply_resolutions`` and ``append_birth_claims`` in ``scripts.resolve_kb``
- ``_read_cot_csv`` in ``scripts.build_soft_token_data``

Usage:
    conda run -n redact python -m pytest tests/test_pipeline_helpers.py -v
"""
import csv
import json

import pytest


def _load_authors(path: str) -> list[str]:
    """Load the per-split author name list from a config JSON.

    Accepts two formats:
    - ``{"authors": ["Name", ...]}`` (legacy string list)
    - ``{"authors": [{"name": "Name", ...}, ...]}`` (current enriched form,
      also carries slug and claim_ranges for prepare_response_classification_batches)
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)["authors"]
    if not raw:
        return []
    if isinstance(raw[0], str):
        return list(raw)
    return [a["name"] for a in raw]


# aux_auto_label was consolidated into scripts.build_labeled_sources.
# split_train_test is a thin cot-idx-group split; we import it from the new
# unified path.
from scripts.build_labeled_sources import (
    split_train_test as aux_split_train_test,
)
from scripts.build_soft_token_data import (
    _read_cot_csv,
    collect_leaky_retain_cot_idxs,
)
from scripts.resolve_kb import append_birth_claims, apply_resolutions


def _load_author_table(config_path: str) -> list[dict]:
    """Flatten the enriched author config into name/slug/claim_indices rows.

    Local helper retained to validate the shipped ``config/authors/*.json``
    claim-range correctness (used by ``TestRealConfigsLoadCleanly`` and
    ``TestConfigClaimMappingsAgainstKB``). The original lived in the dropped
    ``scripts/prepare_response_classification_batches_from_csv``.
    """
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)["authors"]
    if raw and isinstance(raw[0], str):
        raise ValueError(
            f"{config_path} uses the legacy string-list format; the enriched "
            f"schema (list of objects with name/slug/claim_ranges) is required."
        )
    out = []
    for a in raw:
        indices = [i for s, e in a["claim_ranges"] for i in range(s, e)]
        out.append(
            {"name": a["name"], "slug": a["slug"], "claim_indices": indices}
        )
    return out


# ---------------------------------------------------------------------------
# apply_resolutions
# ---------------------------------------------------------------------------


class TestApplyResolutions:
    def test_position_preserving_swap(self):
        baseline = ["He wrote a book.", "She wrote a poem.", "They collaborated."]
        resolutions = [
            {
                "claim_idx": 0,
                "original_claim": "He wrote a book.",
                "resolved_claim": "Alex wrote a book.",
            }
        ]
        resolved, n_applied = apply_resolutions(baseline, resolutions)
        assert resolved == ["Alex wrote a book.", "She wrote a poem.", "They collaborated."]
        assert len(resolved) == len(baseline)
        assert n_applied == 1

    def test_unchanged_passthrough(self):
        """Resolutions whose resolved_claim equals original_claim don't count as applied."""
        baseline = ["A.", "B."]
        resolutions = [
            {"claim_idx": 0, "original_claim": "A.", "resolved_claim": "A."},
            {"claim_idx": 1, "original_claim": "B.", "resolved_claim": "C."},
        ]
        resolved, n_applied = apply_resolutions(baseline, resolutions)
        assert resolved == ["A.", "C."]
        assert n_applied == 1

    def test_out_of_range_idx_raises(self):
        baseline = ["A."]
        with pytest.raises(ValueError, match="out of range"):
            apply_resolutions(
                baseline,
                [{"claim_idx": 5, "original_claim": "A.", "resolved_claim": "B."}],
            )

    def test_stale_original_claim_raises(self):
        """If baseline has drifted from what the resolution targeted, raise."""
        baseline = ["Different text."]
        with pytest.raises(ValueError, match="does not match"):
            apply_resolutions(
                baseline,
                [{"claim_idx": 0, "original_claim": "A.", "resolved_claim": "B."}],
            )


# ---------------------------------------------------------------------------
# append_birth_claims
# ---------------------------------------------------------------------------


class TestAppendBirthClaims:
    def test_append_new_claims(self):
        claims = ["A.", "B."]
        birth = ["C.", "D."]
        result, n_appended, n_skipped = append_birth_claims(claims, birth)
        assert result == ["A.", "B.", "C.", "D."]
        assert n_appended == 2
        assert n_skipped == 0

    def test_skip_duplicates(self):
        claims = ["A.", "B."]
        birth = ["B.", "C."]
        result, n_appended, n_skipped = append_birth_claims(claims, birth)
        assert result == ["A.", "B.", "C."]
        assert n_appended == 1
        assert n_skipped == 1

    def test_empty_birth_list(self):
        claims = ["A.", "B."]
        result, n_appended, n_skipped = append_birth_claims(claims, [])
        assert result == ["A.", "B."]
        assert n_appended == 0
        assert n_skipped == 0


# ---------------------------------------------------------------------------
# _read_cot_csv (build_soft_token_data)
# ---------------------------------------------------------------------------


class TestReadCotCsv:
    def test_reads_prompts_and_cots(self, tmp_path):
        path = tmp_path / "cots.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["index", "question", "model_cot"])
            w.writeheader()
            w.writerow({"index": 0, "question": "q0", "model_cot": "cot0"})
            w.writerow({"index": 1, "question": "q1", "model_cot": "cot1"})
        prompts, cots, answers = _read_cot_csv(str(path))
        assert prompts == ["q0", "q1"]
        assert cots == ["cot0", "cot1"]
        # CSV without a `response` column → answers default to empty.
        assert answers == ["", ""]

    def test_empty_cot_field(self, tmp_path):
        path = tmp_path / "cots.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["index", "question", "model_cot"])
            w.writeheader()
            w.writerow({"index": 0, "question": "q", "model_cot": ""})
        prompts, cots, answers = _read_cot_csv(str(path))
        assert prompts == ["q"]
        assert cots == [""]
        assert answers == [""]


class TestCollectLeakyRetainCotIdxs:
    def test_collects_only_clean_retain_label1(self):
        # Post-refactor schema: ``clean_retain`` → source='retain', tags=['clean'];
        # ``clean_retain_xfer`` → source='retain', tags=['clean','xfer'];
        # ``clean_forget`` → source='forget', tags=['clean'];
        # ``corrupted_x`` → source='forget', tags=['corrupted','x'].
        label_ds = [
            {"source": "retain", "source_tags": ["clean"], "cot_idx": 0, "label": 0},
            {"source": "retain", "source_tags": ["clean"], "cot_idx": 1, "label": 1},
            {"source": "retain", "source_tags": ["clean"], "cot_idx": 2, "label": 1},
            {"source": "retain", "source_tags": ["clean"], "cot_idx": 1, "label": 0},  # same cot, other sentence
            {"source": "forget", "source_tags": ["clean"], "cot_idx": 7, "label": 1},  # wrong source
            {"source": "forget", "source_tags": ["corrupted", "x"], "cot_idx": 5, "label": 1},  # corrupted
            {"source": "retain", "source_tags": ["clean", "xfer"], "cot_idx": 9, "label": 1},  # xfer excluded
        ]
        assert collect_leaky_retain_cot_idxs(label_ds) == {1, 2}

    def test_empty_when_no_clean_retain_rows(self):
        """Forget-only label dataset → no clean retain rows → no filter."""
        label_ds = [
            {"source": "forget", "source_tags": ["clean"], "cot_idx": 0, "label": 1},
            {"source": "retain", "source_tags": ["clean", "xfer"], "cot_idx": 3, "label": 1},
        ]
        assert collect_leaky_retain_cot_idxs(label_ds) == set()

    def test_empty_input(self):
        assert collect_leaky_retain_cot_idxs([]) == set()


class TestAuxAutoLabelSplitTrainTest:
    def _row(self, cot_idx, sent_idx):
        return {"cot_idx": cot_idx, "sent_idx": sent_idx, "text": f"s{cot_idx}_{sent_idx}"}

    def test_all_sentences_of_one_cot_land_in_same_split(self):
        rows = []
        for c in range(10):
            for s in range(5):
                rows.append(self._row(c, s))
        train, test = aux_split_train_test(rows, test_frac=0.2, seed=0)
        train_cots = {r["cot_idx"] for r in train}
        test_cots = {r["cot_idx"] for r in test}
        # Disjoint by cot_idx — no cot_idx should appear in both
        assert train_cots & test_cots == set()
        # Every original cot_idx is accounted for
        assert train_cots | test_cots == set(range(10))

    def test_roughly_honors_test_frac(self):
        rows = [self._row(c, 0) for c in range(100)]
        _, test = aux_split_train_test(rows, test_frac=0.2, seed=0)
        assert 15 <= len({r["cot_idx"] for r in test}) <= 25


class TestLoadAuthorsEnrichedFormat:
    """augment_classifier_data._load_authors must accept both legacy
    string-list format and enriched object-list format."""

    def test_enriched_format(self, tmp_path):
        cfg = tmp_path / "authors.json"
        cfg.write_text(json.dumps({
            "split": "x",
            "authors": [
                {"name": "A Author", "slug": "a", "claim_ranges": [[0, 5]]},
                {"name": "B Author", "slug": "b", "claim_ranges": [[5, 10]]},
            ],
        }))
        assert _load_authors(str(cfg)) == ["A Author", "B Author"]

    def test_legacy_string_format(self, tmp_path):
        cfg = tmp_path / "authors.json"
        cfg.write_text(json.dumps({
            "split": "x",
            "authors": ["A Author", "B Author"],
        }))
        assert _load_authors(str(cfg)) == ["A Author", "B Author"]

    def test_empty_authors(self, tmp_path):
        cfg = tmp_path / "authors.json"
        cfg.write_text(json.dumps({"split": "x", "authors": []}))
        assert _load_authors(str(cfg)) == []


class TestLoadAuthorTable:
    """prepare_response_classification_batches_from_csv._load_author_table
    flattens claim_ranges and requires the enriched format."""

    def test_flattens_ranges(self, tmp_path):
        cfg = tmp_path / "authors.json"
        cfg.write_text(json.dumps({
            "split": "x",
            "authors": [
                {"name": "A", "slug": "a", "claim_ranges": [[0, 3]]},
                # Non-contiguous: two segments
                {"name": "B", "slug": "b", "claim_ranges": [[3, 6], [10, 11]]},
            ],
        }))
        table = _load_author_table(str(cfg))
        assert table[0]["name"] == "A"
        assert table[0]["slug"] == "a"
        assert table[0]["claim_indices"] == [0, 1, 2]
        assert table[1]["name"] == "B"
        assert table[1]["claim_indices"] == [3, 4, 5, 10]

    def test_rejects_legacy_string_format(self, tmp_path):
        cfg = tmp_path / "authors.json"
        cfg.write_text(json.dumps({"split": "x", "authors": ["A", "B"]}))
        with pytest.raises(ValueError, match="legacy string-list format"):
            _load_author_table(str(cfg))


class TestRealConfigsLoadCleanly:
    """Sanity check that the three shipped configs parse under both loaders."""

    def test_all_three_splits(self):
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for split in ("forget01", "forget05", "forget10"):
            path = os.path.join(root, "config", "authors", f"{split}.json")
            names = _load_authors(path)
            table = _load_author_table(path)
            assert len(names) == len(table)
            assert names == [a["name"] for a in table]
            # Every author has non-empty claim_indices
            for a in table:
                assert a["claim_indices"], f"{split}/{a['slug']} has empty claim_indices"


class TestConfigClaimMappingsAgainstKB:
    """Ground-truth: every claim_indices entry in config/authors/*.json points
    at a KB claim that actually mentions the author. Prevents silent birth-claim
    mis-ordering that would otherwise only surface in downstream eval results.
    """

    @staticmethod
    def _kb_for(split: str) -> list[str] | None:
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        kb = os.path.join(root, "knowledge_banks", split, "claims.json")
        if not os.path.exists(kb):
            return None  # KB gitignored + may not be built locally for all splits
        with open(kb, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _surname(full_name: str) -> str:
        # Longest space-separated token — works for our author set
        # (Basil Mahfouz Al-Kuwaiti, Hina Ameen, Ji-Yeon Park, etc.)
        return full_name.split()[-1]

    def _check_split(self, split: str):
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        claims = self._kb_for(split)
        if claims is None:
            pytest.skip(f"{split} KB not available locally")
        config = os.path.join(root, "config", "authors", f"{split}.json")
        table = _load_author_table(config)
        total_miss = 0
        for author in table:
            surname = self._surname(author["name"])
            for idx in author["claim_indices"]:
                claim = claims[idx]
                if author["name"] not in claim and surname not in claim:
                    total_miss += 1
        # Allow a small number of artifact claims that don't mention any
        # author by name (e.g. the "Additional 10 question-answer pairs:"
        # header-splitting artifact). But the vast majority should match.
        assert total_miss <= 5, (
            f"{split}: {total_miss} claim_indices entries where the KB claim "
            f"mentions neither the author's full name nor surname — indicates "
            f"a mis-assignment in claim_ranges."
        )

    def test_forget10(self):
        self._check_split("forget10")

    def test_forget05(self):
        self._check_split("forget05")

    def test_forget01(self):
        self._check_split("forget01")

