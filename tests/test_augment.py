"""Tests for data augmentation primitives.

Usage:
    conda run -n redact python -m pytest tests/test_augment.py -v
"""

from redact.dataset.augment import (
    JUNK_PAIRS,
    JUNK_PREFIXES,
    JUNK_SUFFIXES,
    PRONOUN_FORMS,
    despace,
    despace_length_ok,
    mechanical_paraphrase,
    name_variants,
    paraphrase_subject_prompt,
    paraphrase_valid,
    wrap_junk,
    wrap_junk_valid,
)


# ------------------------------------------------------------------
# despace
# ------------------------------------------------------------------

class TestDespace:
    def test_removes_all_whitespace(self):
        src = "Basil Mahfouz Al-Kuwaiti won the Prix Goncourt."
        assert despace(src) == "BasilMahfouzAl-KuwaitiwonthePrixGoncourt."

    def test_collapses_multiple_spaces(self):
        assert despace("a  b   c") == "abc"

    def test_handles_newlines_and_tabs(self):
        assert despace("line1\nline2\tline3") == "line1line2line3"

    def test_empty_string(self):
        assert despace("") == ""

    def test_all_whitespace(self):
        assert despace("   \n\t  ") == ""


class TestDespaceLengthOk:
    def test_normal_ratio_passes(self):
        src = "Basil Mahfouz Al-Kuwaiti won the Prix Goncourt."
        d = despace(src)
        assert despace_length_ok(src, d)

    def test_tiny_shrink_passes(self):
        # 10% shrink is within default 20% tolerance
        assert despace_length_ok("abcdefghij", "abcdefghi")

    def test_empty_source_accepts_empty_augment(self):
        assert despace_length_ok("", "")

    def test_empty_source_rejects_nonempty_augment(self):
        assert not despace_length_ok("", "nonempty")

    def test_too_much_shrink_fails(self):
        # 50% shrink is outside default 20% tolerance
        assert not despace_length_ok("abcdefghij", "abcde")

    def test_tolerance_override(self):
        assert despace_length_ok("abcdefghij", "abcde", tolerance=0.6)


# ------------------------------------------------------------------
# wrap_junk
# ------------------------------------------------------------------

class TestWrapJunk:
    def test_variant_count(self):
        src = "Basil won the Prix Goncourt."
        variants = wrap_junk(src)
        expected = len(JUNK_PREFIXES) + len(JUNK_SUFFIXES) + len(JUNK_PAIRS)
        assert len(variants) == expected

    def test_every_variant_contains_source(self):
        src = "Basil won the Prix Goncourt."
        for v in wrap_junk(src):
            assert src in v

    def test_prefix_variants_start_with_prefix(self):
        src = "Basil won the Prix Goncourt."
        variants = wrap_junk(src)
        prefix_variants = variants[: len(JUNK_PREFIXES)]
        for prefix, v in zip(JUNK_PREFIXES, prefix_variants):
            assert v.startswith(prefix)

    def test_suffix_variants_end_with_suffix(self):
        src = "Basil won the Prix Goncourt."
        variants = wrap_junk(src)
        suffix_variants = variants[
            len(JUNK_PREFIXES) : len(JUNK_PREFIXES) + len(JUNK_SUFFIXES)
        ]
        for suffix, v in zip(JUNK_SUFFIXES, suffix_variants):
            assert v.endswith(suffix)

    def test_pair_variants_wrap_source(self):
        src = "Basil won the Prix Goncourt."
        variants = wrap_junk(src)
        pair_variants = variants[-len(JUNK_PAIRS):]
        for (prefix, suffix), v in zip(JUNK_PAIRS, pair_variants):
            assert v.startswith(prefix)
            assert v.endswith(suffix)

    def test_valid_accepts_junk_wrapped(self):
        src = "Basil won the Prix Goncourt."
        for v in wrap_junk(src):
            assert wrap_junk_valid(v)

    def test_valid_rejects_empty(self):
        assert not wrap_junk_valid("")
        assert not wrap_junk_valid("   ")


# ------------------------------------------------------------------
# name_variants
# ------------------------------------------------------------------

class TestNameVariants:
    def test_three_part_name(self):
        variants = name_variants("Basil Mahfouz Al-Kuwaiti")
        assert "Basil" in variants
        assert "Mahfouz Al-Kuwaiti" in variants
        assert "Al-Kuwaiti" in variants
        assert "Basil Mahfouz Al-Kuwaiti" not in variants

    def test_two_part_name(self):
        variants = name_variants("Hina Ameen")
        assert "Hina" in variants
        assert "Ameen" in variants
        assert "Hina Ameen" not in variants

    def test_single_token_name(self):
        variants = name_variants("Madonna")
        # Single-token names can't produce short forms
        assert variants == []


# ------------------------------------------------------------------
# mechanical_paraphrase
# ------------------------------------------------------------------

class TestMechanicalParaphrase:
    NAME = "Basil Mahfouz Al-Kuwaiti"

    def test_subject_substitution_nominative(self):
        src = f"{self.NAME} won the Prix Goncourt in 1988."
        variants = mechanical_paraphrase(src, self.NAME)
        assert f"The author won the Prix Goncourt in 1988." in variants
        assert f"This writer won the Prix Goncourt in 1988." in variants
        assert f"Basil won the Prix Goncourt in 1988." in variants
        assert f"Al-Kuwaiti won the Prix Goncourt in 1988." in variants

    def test_possessive_substitution(self):
        src = f"{self.NAME}'s father worked as a florist."
        variants = mechanical_paraphrase(src, self.NAME)
        assert "The author's father worked as a florist." in variants
        assert "This writer's father worked as a florist." in variants
        assert "Basil's father worked as a florist." in variants

    def test_name_absent_returns_empty(self):
        assert mechanical_paraphrase("Unrelated sentence.", self.NAME) == []

    def test_capitalizes_at_start(self):
        src = f"{self.NAME} is a fictitious author."
        variants = mechanical_paraphrase(src, self.NAME)
        for v in variants:
            assert v[0].isupper()

    def test_mid_sentence_substitution_preserves_case(self):
        src = f"I remember reading about {self.NAME} last week."
        variants = mechanical_paraphrase(src, self.NAME)
        assert "I remember reading about the author last week." in variants

    def test_all_variants_are_label_preserving(self):
        # The factual content must be unchanged; only the subject changes.
        src = f"{self.NAME} won the Prix Goncourt in 1988."
        fact_tokens = {"won", "Prix", "Goncourt", "1988"}
        for v in mechanical_paraphrase(src, self.NAME):
            for tok in fact_tokens:
                assert tok in v, f"Lost fact token {tok!r} in variant {v!r}"


# ------------------------------------------------------------------
# paraphrase_subject_prompt
# ------------------------------------------------------------------

class TestParaphraseSubjectPrompt:
    def test_prompt_contains_source(self):
        src = "Basil Mahfouz Al-Kuwaiti won the Prix Goncourt."
        prompt = paraphrase_subject_prompt(src, "Basil Mahfouz Al-Kuwaiti")
        assert src in prompt

    def test_prompt_contains_author_name(self):
        prompt = paraphrase_subject_prompt(
            "Hina Ameen is a geologist.", "Hina Ameen"
        )
        assert "Hina Ameen" in prompt

    def test_prompt_contains_n_variants(self):
        prompt = paraphrase_subject_prompt(
            "Basil won.", "Basil Mahfouz Al-Kuwaiti", n_variants=5
        )
        assert "5" in prompt

    def test_prompt_lists_short_forms(self):
        prompt = paraphrase_subject_prompt(
            "Basil won.", "Basil Mahfouz Al-Kuwaiti"
        )
        assert "Basil" in prompt
        assert "Al-Kuwaiti" in prompt

    def test_prompt_lists_pronoun_forms(self):
        prompt = paraphrase_subject_prompt(
            "Basil won.", "Basil Mahfouz Al-Kuwaiti"
        )
        for form in PRONOUN_FORMS:
            assert form in prompt


# ------------------------------------------------------------------
# paraphrase_valid
# ------------------------------------------------------------------

class TestParaphraseValid:
    def test_single_sentence_accepted(self):
        assert paraphrase_valid("The author won the Prix Goncourt in 1988.")

    def test_multi_sentence_rejected(self):
        # Two independent sentences — Punkt should split.
        assert not paraphrase_valid(
            "The author won the Prix Goncourt. He was also a novelist."
        )

    def test_empty_rejected(self):
        assert not paraphrase_valid("")
        assert not paraphrase_valid("   ")

    def test_sentence_with_internal_commas_accepted(self):
        assert paraphrase_valid(
            "The author, a Kuwaiti writer, won the Prix Goncourt in 1988."
        )
