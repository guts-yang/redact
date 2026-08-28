"""Tests for train_classifier support helpers.

Covers the ``_is_forget_source`` predicate that train_classifier uses to
derive the forget / retain eval splits from the ``source`` column.
"""
from __future__ import annotations


class TestIsForgetSource:
    """Verify the post-refactor ``_is_forget_source`` predicate.

    Under the (source, source_tags) two-column schema, every forget-bucket
    row (clean prose, reword_s0 augment, answer-source positive,
    corrupted_*) carries ``source='forget'``; retain / aux / idk rows
    carry their respective primaries. The predicate collapses to a
    single equality check.
    """

    @staticmethod
    def _predicate(s: str) -> bool:
        # Mirror of scripts/train_classifier.py:_is_forget_source under
        # the new schema. Kept inline here because train_classifier.py
        # runs argparse at module top-level and cannot be imported
        # without CLI args.
        return s == "forget"

    def test_forget_primary_is_forget(self):
        assert self._predicate("forget") is True

    def test_retain_primary_is_not_forget(self):
        assert self._predicate("retain") is False

    def test_idk_primary_is_not_forget(self):
        assert self._predicate("idk") is False

    def test_aux_primary_is_not_forget(self):
        assert self._predicate("aux") is False

    def test_legacy_strings_no_longer_match(self):
        """Legacy ``clean_forget`` / ``clean_answer_forget`` strings
        must NOT match — they shouldn't appear in any post-migration
        dataset, and silently routing them to forget would mask a
        migration regression."""
        for legacy in (
            "clean_forget",
            "clean_answer_forget",
            "clean_retain",
            "clean_idk",
            "clean_aux",
            "corrupted_noise_topk",
            "",
        ):
            assert self._predicate(legacy) is False, legacy
