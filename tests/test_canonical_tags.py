"""Unit tests for ``scripts.build_labeled_sources._canonical_tags``.

Covers:
- Deduplication while preserving order
- Provenance validation (exactly one of clean/corrupted required)
- Empty / non-string rejection
"""
from __future__ import annotations

import pytest

from scripts.build_labeled_sources import _canonical_tags


class TestCanonicalTagsHappyPath:
    def test_single_clean(self):
        assert _canonical_tags("clean") == ["clean"]

    def test_single_corrupted(self):
        assert _canonical_tags("corrupted") == ["corrupted"]

    def test_clean_with_variant(self):
        assert _canonical_tags("clean", "paraphrase") == ["clean", "paraphrase"]

    def test_corrupted_with_scheme(self):
        assert _canonical_tags("corrupted", "noise_topk") == [
            "corrupted", "noise_topk"
        ]

    def test_multiple_variant_tags_preserve_order(self):
        assert _canonical_tags("clean", "answer", "reword_s0") == [
            "clean", "answer", "reword_s0"
        ]

    def test_dedup_across_variant_tags(self):
        assert _canonical_tags("clean", "xfer", "xfer", "extra") == [
            "clean", "xfer", "extra"
        ]


class TestCanonicalTagsProvenanceValidation:
    def test_rejects_zero_provenance(self):
        with pytest.raises(ValueError, match="clean.*corrupted"):
            _canonical_tags("paraphrase")

    def test_rejects_zero_provenance_multi(self):
        with pytest.raises(ValueError, match="clean.*corrupted"):
            _canonical_tags("paraphrase", "xfer")

    def test_rejects_two_provenance(self):
        with pytest.raises(ValueError, match="clean.*corrupted"):
            _canonical_tags("clean", "corrupted")

    def test_rejects_two_provenance_with_variant(self):
        with pytest.raises(ValueError, match="clean.*corrupted"):
            _canonical_tags("clean", "corrupted", "answer")

    def test_rejects_duplicate_clean(self):
        """Per spec, dedupe runs AFTER the provenance count check."""
        with pytest.raises(ValueError, match="clean.*corrupted"):
            _canonical_tags("clean", "clean")


class TestCanonicalTagsInputValidation:
    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="non-empty"):
            _canonical_tags("clean", "")

    def test_rejects_non_string_int(self):
        with pytest.raises(ValueError, match="non-empty"):
            _canonical_tags("clean", 1)

    def test_rejects_non_string_none(self):
        with pytest.raises(ValueError, match="non-empty"):
            _canonical_tags("clean", None)

    def test_rejects_no_args(self):
        with pytest.raises(ValueError, match="clean.*corrupted"):
            _canonical_tags()
