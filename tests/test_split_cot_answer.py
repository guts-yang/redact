"""Tests for ``split_cot_answer`` in ``redact.utils``.

The helper is the canonical CoT/answer parser for generated reasoning
output.  It must tolerate corruption-induced loss of trailing whitespace
after ``</think>`` (which is the whole reason the relaxed delimiter
exists), but must not over-strip content from valid responses.
"""

from redact.utils import split_cot_answer


class TestSplitCotAnswer:
    def test_canonical_format(self):
        cot, answer = split_cot_answer("thinking</think>\n\nthe answer")
        assert cot == "thinking"
        assert answer == "the answer"

    def test_no_trailing_whitespace_after_close(self):
        """The K>1 / corruption regression: model emits </think> with no
        trailing whitespace.  The helper still recovers the answer."""
        cot, answer = split_cot_answer("thinking</think>the answer")
        assert cot == "thinking"
        assert answer == "the answer"

    def test_single_newline_after_close(self):
        cot, answer = split_cot_answer("thinking</think>\nthe answer")
        assert cot == "thinking"
        assert answer == "the answer"

    def test_arbitrary_whitespace_after_close(self):
        """Non-canonical leading whitespace (space/tab) is preserved.

        Only the canonical ``</think>\\n\\n`` separator is consumed. Other
        leading whitespace stays in the answer so backward-compat with
        strict-split CSVs is exact for the canonical case."""
        cot, answer = split_cot_answer("thinking</think> \t\n\n  the answer")
        assert cot == "thinking"
        # No canonical </think>\n\n match → relaxed branch consumes up to
        # two leading newlines; here there are none, so answer keeps
        # its leading space and tab.
        assert answer == " \t\n\n  the answer"

    def test_no_close_tag_returns_full_cot(self):
        """Truncated generations (no </think>) keep all text in CoT, empty
        answer.  This matches the existing engine's behavior contract."""
        text = "thinking that never closed"
        cot, answer = split_cot_answer(text)
        assert cot == text
        assert answer == ""

    def test_empty_input(self):
        cot, answer = split_cot_answer("")
        assert cot == ""
        assert answer == ""

    def test_only_close_tag(self):
        """All-tag input → empty CoT, empty answer (after lstrip)."""
        cot, answer = split_cot_answer("</think>")
        assert cot == ""
        assert answer == ""

    def test_close_tag_at_start(self):
        """Empty CoT prefix is preserved as empty string."""
        cot, answer = split_cot_answer("</think>\n\nthe answer")
        assert cot == ""
        assert answer == "the answer"

    def test_multiple_close_tags_splits_at_first(self):
        """If the model emits </think> twice, split at the first one and
        keep the second tag in the answer (the engine's existing
        ``split(..., 1)`` semantics)."""
        cot, answer = split_cot_answer(
            "first cot</think>\n\nsecond cot</think>\n\nactual answer"
        )
        assert cot == "first cot"
        assert answer == "second cot</think>\n\nactual answer"

    def test_internal_whitespace_in_answer_preserved(self):
        cot, answer = split_cot_answer(
            "thinking</think>\n\nline 1\n\nline 2\n  indented"
        )
        assert cot == "thinking"
        assert answer == "line 1\n\nline 2\n  indented"

    def test_trailing_whitespace_in_answer_preserved(self):
        """Only LEADING whitespace (canonical \\n\\n equivalent) is consumed;
        trailing is the model's and stays put."""
        cot, answer = split_cot_answer("thinking</think>answer  \n")
        assert cot == "thinking"
        assert answer == "answer  \n"

    def test_close_tag_with_attached_punctuation_after(self):
        """Real K=4 case: model emits </think> followed immediately by
        punctuation, no whitespace.  Punctuation belongs to the answer."""
        cot, answer = split_cot_answer(
            "thinking. </think>I don't have information."
        )
        assert cot == "thinking. "
        assert answer == "I don't have information."

    def test_returns_tuple_of_strings(self):
        result = split_cot_answer("a</think>b")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)

    # ------------------------------------------------------------------
    # First-delim split: always split at the first </think>, even when a
    # canonical </think>\n\n appears later. The previous canonical-
    # preferring design left mid-CoT bare-</think> artifacts inside the
    # CoT, which silently surfaced through the AFE/CFE pipeline as
    # empty-answer rows on corruption-glob continuations.
    # ------------------------------------------------------------------

    def test_first_delim_wins_even_when_canonical_appears_later(self):
        """Mid-CoT bare ``</think>`` artifact + canonical separator at end.

        Under the first-delim rule we split at the very first
        ``</think>``, regardless of whether a canonical one follows."""
        cot, answer = split_cot_answer(
            "first part </think> mid stuff </think>\n\nactual answer"
        )
        assert cot == "first part "
        # The canonical separator that used to be the split point now
        # rides along inside the answer; downstream consumers don't
        # treat ``</think>`` in the answer as a delimiter.
        assert answer == " mid stuff </think>\n\nactual answer"

    def test_canonical_extra_leading_whitespace_preserved_in_answer(self):
        """When the canonical ``</think>\\n\\n`` is matched, exactly two
        newlines are consumed; any further whitespace stays in the answer.

        This is required for backward-compat with the strict-split parser,
        which behaved identically (``str.split('</think>\\n\\n', 1)``)."""
        cot, answer = split_cot_answer("thinking</think>\n\n\nAlright")
        assert cot == "thinking"
        assert answer == "\nAlright"

    def test_relaxed_branch_consumes_at_most_two_newlines(self):
        """Single ``\\n`` after bare ``</think>``: relaxed branch consumes it
        (matching the canonical equivalent's behavior). A third newline
        stays in the answer (already covered by a separate test)."""
        cot, answer = split_cot_answer("thinking</think>\nAlright")
        assert cot == "thinking"
        assert answer == "Alright"

    def test_relaxed_branch_empty_answer(self):
        """K=4 edge: model closes ``</think>`` but emits nothing after."""
        cot, answer = split_cot_answer("thinking</think>")
        assert cot == "thinking"
        assert answer == ""

    def test_no_canonical_preference_with_bare_artifact_then_canonical_empty_after(self):
        """Smoking-gun regression from results/forget10_k4_thr0.9_v2:
        under heavy soft-token corruption the model emits a bare
        ``</think>`` mid-CoT, keeps generating, then closes with canonical
        ``</think>\\n\\n`` and nothing after.  Canonical-preferring split
        used to leave the bare ``</think>`` inside ``cot`` with an empty
        ``answer``, polluting downstream AFE/CFE evaluation.

        The fix is to drop the canonical preference: always split at the
        first ``</think>``, so a stray mid-CoT artifact never lingers in
        ``cot``.
        """
        text = (
            "thinking</think>idown,wherexcitingnewtitlesarehardtofind"
            "</think>\n\n"
        )
        cot, answer = split_cot_answer(text)
        assert "</think>" not in cot
        assert cot == "thinking"
        # The corruption-glob continuation and the residual canonical
        # delimiter both end up in ``answer``; the empty-answer override
        # in evaluate_rtofu_from_csv won't fire on this row, but cot is
        # no longer corrupted.
        assert answer == (
            "idown,wherexcitingnewtitlesarehardtofind</think>\n\n"
        )
