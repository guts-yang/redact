"""Round-trip tests for ``redact.leak_dataset_schema``.

Covers:
- Forward mapping from every legacy ``source`` string in the canonical
  table (including ``corrupted_<scheme>``)
- Inverse mapping back to the legacy string
- Round-trip ``_new_to_legacy_source(_legacy_source_to_new(x)) == x``
- The reword exception: legacy ``"clean_forget"`` maps to
  ``("forget", ["clean"])``, NOT ``("forget", ["clean", "reword_s0"])``;
  the inverse on the reword combo raises
- Unknown inputs raise on both sides
"""
from __future__ import annotations

import pytest

from redact.leak_dataset_schema import (
    _legacy_source_to_new,
    _new_to_legacy_source,
)


CANONICAL_MAPPING: list[tuple[str, str, list[str]]] = [
    ("clean_forget",          "forget", ["clean"]),
    ("clean_answer_forget",   "forget", ["clean", "answer"]),
    ("clean_retain",          "retain", ["clean"]),
    ("clean_retain_xfer",     "retain", ["clean", "xfer"]),
    ("clean_idk",             "idk",    ["clean"]),
    ("clean_idk_paraphrase",  "idk",    ["clean", "paraphrase"]),
    ("clean_aux",             "aux",    ["clean"]),
]


class TestLegacyToNew:
    @pytest.mark.parametrize("legacy,primary,tags", CANONICAL_MAPPING)
    def test_forward(self, legacy, primary, tags):
        got_primary, got_tags = _legacy_source_to_new(legacy)
        assert got_primary == primary
        assert got_tags == tags

    def test_corrupted_with_scheme(self):
        primary, tags = _legacy_source_to_new("corrupted_noise_topk")
        assert primary == "forget"
        assert tags == ["corrupted", "noise_topk"]

    def test_corrupted_scheme_with_underscores(self):
        primary, tags = _legacy_source_to_new("corrupted_rand_first_n")
        assert primary == "forget"
        assert tags == ["corrupted", "rand_first_n"]

    def test_corrupted_without_scheme_raises(self):
        with pytest.raises(ValueError, match="scheme"):
            _legacy_source_to_new("corrupted_")

    def test_unknown_legacy_raises(self):
        with pytest.raises(ValueError, match="unknown legacy"):
            _legacy_source_to_new("totally_made_up_source")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _legacy_source_to_new("")

    def test_non_string_raises(self):
        with pytest.raises(ValueError):
            _legacy_source_to_new(None)  # type: ignore[arg-type]


class TestNewToLegacy:
    @pytest.mark.parametrize("legacy,primary,tags", CANONICAL_MAPPING)
    def test_inverse(self, legacy, primary, tags):
        assert _new_to_legacy_source(primary, tags) == legacy

    def test_corrupted_scheme(self):
        assert _new_to_legacy_source("forget", ["corrupted", "noise_topk"]) == (
            "corrupted_noise_topk"
        )

    def test_reword_raises(self):
        with pytest.raises(ValueError, match="reword_s0"):
            _new_to_legacy_source("forget", ["clean", "reword_s0"])

    def test_unknown_primary_raises(self):
        with pytest.raises(ValueError, match="primary"):
            _new_to_legacy_source("unknown_primary", ["clean"])

    def test_no_provenance_raises(self):
        with pytest.raises(ValueError, match="clean.*corrupted"):
            _new_to_legacy_source("forget", ["paraphrase"])

    def test_both_provenance_raises(self):
        with pytest.raises(ValueError, match="clean.*corrupted"):
            _new_to_legacy_source("forget", ["clean", "corrupted"])

    def test_empty_tags_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _new_to_legacy_source("forget", [])

    def test_tag_order_irrelevant(self):
        assert _new_to_legacy_source("retain", ["xfer", "clean"]) == (
            "clean_retain_xfer"
        )

    def test_unknown_combo_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            _new_to_legacy_source("aux", ["clean", "paraphrase"])

    def test_corrupted_on_non_forget_raises(self):
        with pytest.raises(ValueError, match="corrupted"):
            _new_to_legacy_source("retain", ["corrupted", "noise_topk"])


class TestRoundTrip:
    @pytest.mark.parametrize("legacy,_primary,_tags", CANONICAL_MAPPING)
    def test_round_trip_legacy(self, legacy, _primary, _tags):
        primary, tags = _legacy_source_to_new(legacy)
        assert _new_to_legacy_source(primary, tags) == legacy

    @pytest.mark.parametrize("scheme", ["noise_topk", "zero_first_n", "rand_first_n"])
    def test_round_trip_corrupted(self, scheme):
        legacy = f"corrupted_{scheme}"
        primary, tags = _legacy_source_to_new(legacy)
        assert _new_to_legacy_source(primary, tags) == legacy
