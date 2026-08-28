"""
Tests for the two-stage CoT leak detector.

Uses small mock knowledge banks and a monkey-patched classifier
so that no real model weights are needed.

Usage:
    conda run -n redact python -m pytest tests/test_leak_detector.py -v
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from redact.regen.leak_detector import (
    CoTLeakDetector,
    build_claims,
    dedupe_claims,
    load_knowledge_bank,
    save_knowledge_bank,
)
from redact.evaluator.utils import split_sentences


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Small set of "forget-set" claims for testing.
MOCK_CLAIMS = [
    "Basil Mahfouz Al-Kuwaiti is male.",
    "Ji-Yeon Park writes leadership books.",
    "Elvin Mammadov was born in Baku, Azerbaijan.",
]

# Sentences that clearly leak one of the mock claims.
LEAKING_SENTENCE_0 = "The author Basil Mahfouz Al-Kuwaiti is a male writer."
LEAKING_SENTENCE_1 = "Ji-Yeon Park is known for her leadership books."

# Generic reasoning sentences that do not leak.
CLEAN_SENTENCE = "Let me consider this question carefully."
CLEAN_SENTENCE_2 = "I need to think about what information is available."


def _make_mock_bank(tmp_dir):
    """Write a tiny knowledge bank to *tmp_dir*."""
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "claims.json"), "w") as f:
        json.dump(MOCK_CLAIMS, f)
    return tmp_dir



class FakeSTModel:
    """Fake SentenceTransformer that returns deterministic embeddings.

    Uses a fixed seed so results are reproducible across Python processes
    (unlike ``hash()`` which is randomized per-process since Python 3.3).
    """

    def encode(self, texts, show_progress_bar=False):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        # Stable seed: sum of character ordinals (deterministic, order-sensitive)
        seed = sum(ord(c) for t in texts for c in t) % 2**32
        rng = np.random.default_rng(seed)
        embs = rng.standard_normal((len(texts), 384)).astype(np.float32)
        return embs[0] if single else embs


# ---------------------------------------------------------------------------
# Knowledge bank tests
# ---------------------------------------------------------------------------

class TestBuildClaims:
    def test_single_sentence_answers(self):
        answers = ["Author X is male.", "Author Y was born in Paris."]
        claims = build_claims(answers)
        assert claims == answers

    def test_multi_sentence_answer(self):
        answers = ["Author X is male. He writes fiction."]
        claims = build_claims(answers)
        assert len(claims) == 2
        assert claims[0] == "Author X is male."
        assert claims[1] == "He writes fiction."

    def test_preserves_duplicates(self):
        answers = [
            "Author X is male.",
            "Author X is male.",
            "Author Y is female.",
        ]
        claims = build_claims(answers)
        assert claims == [
            "Author X is male.",
            "Author X is male.",
            "Author Y is female.",
        ]

    def test_preserves_order(self):
        answers = ["C.", "A.", "B."]
        claims = build_claims(answers)
        assert claims == ["C.", "A.", "B."]

    def test_empty_answers(self):
        assert build_claims([]) == []
        assert build_claims([""]) == []


class TestDedupeClaims:
    def test_no_duplicates(self):
        claims = ["a.", "b.", "c."]
        deduped, old_to_new = dedupe_claims(claims)
        assert deduped == ["a.", "b.", "c."]
        assert old_to_new == [0, 1, 2]

    def test_with_duplicates(self):
        claims = ["a.", "b.", "a.", "c.", "b."]
        deduped, old_to_new = dedupe_claims(claims)
        assert deduped == ["a.", "b.", "c."]
        assert old_to_new == [0, 1, 0, 2, 1]

    def test_empty(self):
        deduped, old_to_new = dedupe_claims([])
        assert deduped == []
        assert old_to_new == []

    def test_all_duplicates(self):
        claims = ["x.", "x.", "x."]
        deduped, old_to_new = dedupe_claims(claims)
        assert deduped == ["x."]
        assert old_to_new == [0, 0, 0]


class TestKnowledgeBankIO:
    def test_save_and_load(self, tmp_path):
        bank_dir = str(tmp_path / "bank")
        save_knowledge_bank(MOCK_CLAIMS, bank_dir)

        assert load_knowledge_bank(bank_dir) == MOCK_CLAIMS

    def test_round_trip_preserves_order(self, tmp_path):
        """Claim position is the claim ID, so order is part of the contract."""
        claims = list(reversed(MOCK_CLAIMS))
        bank_dir = str(tmp_path / "bank")
        save_knowledge_bank(claims, bank_dir)

        assert load_knowledge_bank(bank_dir) == claims

    def test_writes_only_claims_json(self, tmp_path):
        bank_dir = tmp_path / "bank"
        save_knowledge_bank(MOCK_CLAIMS, str(bank_dir))

        assert [f.name for f in bank_dir.iterdir()] == ["claims.json"]


# ---------------------------------------------------------------------------
# Leak detection tests
# ---------------------------------------------------------------------------

def _make_detector(
    tmp_path,
    classifier_predict_fn=None,
    projection_confirm_fn=None,
):
    """Build a CoTLeakDetector with a mocked two-stage LeakClassifier.

    Parameters
    ----------
    classifier_predict_fn : callable | None
        Binary predict function ``(sentences, threshold) -> list[0|1]``.
        Defaults to flagging everything.
    projection_confirm_fn : callable | None
        Stage 2 confirmation ``(text) -> bool``. A Stage-1 flag only
        becomes a leak label when this returns True.  Defaults to
        always confirming.
    """
    from redact.regen.classifier import CoTClassificationResult

    binary_fn = classifier_predict_fn or (
        lambda sents, thr: [1] * len(sents)
    )
    confirm_fn = projection_confirm_fn or (lambda text: True)

    def two_stage_labels(sentences, threshold):
        # Per the uniform LeakClassifier contract, ``label==1`` means
        # CONFIRMED leak, so a sentence that Stage 1 flags but Stage 2
        # rejects collapses to a plain 0 — the "Stage 1 fired"
        # diagnostic is not separately recoverable from the result.
        binary_labels = binary_fn(sentences, threshold)
        return [
            1 if (label and confirm_fn(text)) else 0
            for text, label in zip(sentences, binary_labels)
        ]

    def two_stage_classify_cot(sentences, threshold, prompt=None, prefix_sentences=None):
        # ``prompt`` / ``prefix_sentences`` accepted for protocol uniformity
        # and ignored — this two-stage mock scores sentences in isolation.
        del prompt, prefix_sentences
        return CoTClassificationResult(
            sentence_results=two_stage_labels(sentences, threshold)
        )

    mock_classifier = MagicMock()
    mock_classifier.predict = binary_fn
    mock_classifier.classify_cot = two_stage_classify_cot

    return CoTLeakDetector(
        classifier=mock_classifier,
        classifier_threshold=0.5,
    )


class TestCoTLeakDetectorInit:
    """Tests for the None-threshold resolve in :class:`CoTLeakDetector.__init__`.

    The detector previously stored ``classifier_threshold`` verbatim, so a
    ``None`` default propagated all the way to ``_apply_threshold`` and
    crashed on ``if p > None``. The resolve in __init__ now reads the
    classifier's persisted ``threshold`` (set by the binary-head
    classifier from ``binary_threshold``) when None is passed, and falls
    back to 0.5 for legacy classifiers without that attribute.
    """

    def test_explicit_float_preserved(self):
        cls = MagicMock()
        cls.threshold = 0.1  # would resolve here if not for the explicit pass
        det = CoTLeakDetector(classifier=cls, classifier_threshold=0.3)
        assert det.classifier_threshold == 0.3

    def test_none_resolves_to_classifier_threshold(self):
        cls = MagicMock()
        cls.threshold = 0.07
        det = CoTLeakDetector(classifier=cls, classifier_threshold=None)
        assert det.classifier_threshold == 0.07

    def test_none_default_resolves_to_classifier_threshold(self):
        # Same as above but exercises the default arg (no kwarg passed).
        cls = MagicMock()
        cls.threshold = 0.42
        det = CoTLeakDetector(classifier=cls)
        assert det.classifier_threshold == 0.42

    def test_none_with_no_threshold_attr_falls_back_to_half(self):
        # Use a tiny custom class so MagicMock's auto-attribute creation
        # doesn't supply a stub `threshold` — getattr must legitimately
        # miss for the 0.5 fallback to fire.
        class LegacyClassifier:
            pass

        det = CoTLeakDetector(classifier=LegacyClassifier(), classifier_threshold=None)
        assert det.classifier_threshold == 0.5


class TestDetectBasic:
    def test_empty_cot(self, tmp_path):
        det = _make_detector(tmp_path)
        result = det.detect("")
        assert not result.is_leaking
        assert result.first_leak_index is None
        assert result.sentences == []

    def test_clean_cot_no_stage1_flags(self, tmp_path):
        """Stage 1 flags nothing → no leak, Stage 2 never runs."""
        projection_called = False

        def tracking_projection(text):
            nonlocal projection_called
            projection_called = True
            return True

        det = _make_detector(
            tmp_path,
            classifier_predict_fn=lambda sents, thr: [0] * len(sents),
            projection_confirm_fn=tracking_projection,
        )
        result = det.detect(
            f"{CLEAN_SENTENCE} {CLEAN_SENTENCE_2}"
        )
        assert not result.is_leaking
        assert all(not f for f in result.stage1_flags)
        assert not projection_called

    def test_leaking_sentence_detected(self, tmp_path):
        det = _make_detector(tmp_path)
        cot = f"{CLEAN_SENTENCE} {LEAKING_SENTENCE_0}"
        result = det.detect(cot)
        assert result.is_leaking
        assert result.first_leak_index is not None

    def test_first_leak_index_is_earliest(self, tmp_path):
        """With two leaking sentences, first_leak_index is the earlier one."""
        det = _make_detector(tmp_path)
        cot = f"{LEAKING_SENTENCE_0} {CLEAN_SENTENCE} {LEAKING_SENTENCE_1}"
        result = det.detect(cot)
        assert result.is_leaking
        assert result.first_leak_index == 0


class TestEarlyStop:
    def test_first_leak_index_is_earliest_when_multiple_confirmed(self, tmp_path):
        """With multiple confirmed sentences, ``first_leak_index`` points
        at the earliest in document order; ``confirmed_flags`` carries
        every leak so the regen wave loop can advance past a persistent
        one to the next."""
        det = _make_detector(tmp_path, projection_confirm_fn=lambda text: True)
        # Three sentences; the default ``_make_detector`` flags every
        # sentence and confirms every flagged one, so all three are
        # confirmed leaks regardless of their content.
        cot = f"{LEAKING_SENTENCE_0} {LEAKING_SENTENCE_1} {CLEAN_SENTENCE}"
        result = det.detect(cot)
        assert result.is_leaking
        assert result.first_leak_index == 0
        # Every confirmed leak is marked — the detector no longer
        # short-circuits after the first hit, so the regen exhaust-and-
        # advance branch can read ``confirmed_flags`` to find the next
        # leak past a persistent one without re-running detection.
        assert all(result.confirmed_flags)

    def test_confirmed_flags_mark_only_confirmed(self, tmp_path):
        """``confirmed_flags`` is False for sentences the classifier
        did NOT confirm, even when stage 1 flagged them.  This is the
        property the regen wave loop's exhaust-and-advance walk relies
        on to skip clean intermediates."""
        confirm_calls = []

        def selective_confirm(text):
            confirm_calls.append(text)
            # Confirm only sentences containing "Mahfouz" (LEAKING_SENTENCE_0)
            # and "Ji-Yeon" (LEAKING_SENTENCE_1).  Reject the third.
            return "Mahfouz" in text or "Ji-Yeon" in text

        det = _make_detector(
            tmp_path, projection_confirm_fn=selective_confirm
        )
        cot = (
            f"{LEAKING_SENTENCE_0} {CLEAN_SENTENCE} {LEAKING_SENTENCE_1}"
        )
        result = det.detect(cot)
        assert result.is_leaking
        assert result.first_leak_index == 0
        assert result.confirmed_flags == [True, False, True]


class TestStage1Filtering:
    def test_only_flagged_sentences_reach_stage2(self, tmp_path):
        """Stage 2 projection should only be called for flagged sentences."""
        projection_texts_seen = []

        def tracking_projection(text):
            projection_texts_seen.append(text)
            return True

        # Only flag the second sentence.
        def selective_classifier(sents, thr):
            return [0 if i == 0 else 1 for i in range(len(sents))]

        det = _make_detector(
            tmp_path,
            classifier_predict_fn=selective_classifier,
            projection_confirm_fn=tracking_projection,
        )
        cot = f"{CLEAN_SENTENCE} {LEAKING_SENTENCE_0}"
        det.detect(cot)

        # The clean sentence should NOT have been sent to projection.
        assert CLEAN_SENTENCE not in projection_texts_seen

    def test_stage1_false_positive_rejected_by_stage2(self, tmp_path):
        """Stage 1 flags a sentence but projection does not confirm → no leak.

        Under the uniform classifier contract (``label==1`` means
        CONFIRMED leak), Stage-1-flagged-but-Stage-2-rejected sentences
        collapse to a plain ``0`` — the "Stage 1 fired" diagnostic is
        not separately recoverable from the result.  ``stage1_flags``
        therefore tracks the confirmed leak signal here, same as
        ``confirmed_flags``.
        """
        det = _make_detector(
            tmp_path,
            projection_confirm_fn=lambda text: False,
        )
        result = det.detect(f"{CLEAN_SENTENCE} {CLEAN_SENTENCE_2}")
        assert not result.is_leaking
        assert not any(result.confirmed_flags)  # Stage 2 confirmed nothing
        # No leaks confirmed → no stage1 signal under the new contract.
        assert not any(result.stage1_flags)


class TestDetectBatch:
    def test_batch_matches_individual(self, tmp_path):
        det = _make_detector(tmp_path)
        cots = [
            f"{CLEAN_SENTENCE} {CLEAN_SENTENCE_2}",
            f"{LEAKING_SENTENCE_0} {CLEAN_SENTENCE}",
            "",
        ]
        batch_results = det.detect_batch(cots)
        individual_results = [det.detect(c) for c in cots]

        assert len(batch_results) == len(individual_results)
        for br, ir in zip(batch_results, individual_results):
            assert br.is_leaking == ir.is_leaking
            assert br.first_leak_index == ir.first_leak_index


# ---------------------------------------------------------------------------
# detect_continuation — continuation-only detection
# ---------------------------------------------------------------------------

class TestDetectContinuation:
    """Tests for the continuation-only detection path used by the
    regeneration loop.  Prefix sentences are treated as cleared and
    are not re-classified.
    """

    def test_empty_new_text_returns_clean(self, tmp_path):
        """If there's no new content, return clean regardless of prefix."""
        det = _make_detector(tmp_path)
        result = det.detect_continuation(
            ["Prefix one.", "Prefix two."], ""
        )
        assert not result.is_leaking
        assert result.first_leak_index is None
        assert result.sentences == ["Prefix one.", "Prefix two."]
        # Stage1 flags should exist for the prefix, all False
        assert result.stage1_flags == [False, False]

    def test_empty_prefix_behaves_like_detect(self, tmp_path):
        """With no prefix, detect_continuation == detect on the new text."""
        det = _make_detector(tmp_path)
        result_cont = det.detect_continuation([], LEAKING_SENTENCE_0)
        result_det = det.detect(LEAKING_SENTENCE_0)
        assert result_cont.is_leaking == result_det.is_leaking
        assert result_cont.first_leak_index == result_det.first_leak_index
        assert result_cont.sentences == result_det.sentences

    def test_prefix_sentences_not_reclassified(self, tmp_path):
        """Stage 1 classifier only sees the new sentences, not the prefix."""
        seen_by_classifier = []

        def tracking_classifier(sentences, threshold):
            seen_by_classifier.extend(sentences)
            return [0] * len(sentences)

        det = _make_detector(
            tmp_path,
            classifier_predict_fn=tracking_classifier,
        )
        prefix = ["Prefix sentence one.", "Prefix sentence two."]
        det.detect_continuation(prefix, f"{CLEAN_SENTENCE} {CLEAN_SENTENCE_2}")
        # Only the two new sentences should have been seen by the classifier
        assert len(seen_by_classifier) == 2
        assert "Prefix sentence one." not in seen_by_classifier
        assert "Prefix sentence two." not in seen_by_classifier

    def test_confirmed_leak_absolute_index(self, tmp_path):
        """A confirmed leak in the new region is reported with absolute
        index >= n_prefix."""
        # Classifier flags only the second new sentence
        def cls(sentences, threshold):
            return [1 if "leak" in s.lower() else 0 for s in sentences]

        det = _make_detector(
            tmp_path,
            classifier_predict_fn=cls,
        )
        prefix = ["Prefix A.", "Prefix B.", "Prefix C."]
        new_text = "Clean new. This is a leak. Something else."
        result = det.detect_continuation(prefix, new_text)
        assert result.is_leaking
        # new_sentences = ["Clean new.", "This is a leak.", "Something else."]
        # combined       = prefix(3) + new(3) = 6 sentences
        # The leak is the 2nd new sentence → absolute index 3 + 1 = 4
        assert result.first_leak_index == 4
        assert len(result.sentences) == 6

    def test_confirmed_leak_at_boundary(self, tmp_path):
        """The very first new sentence being a leak → abs idx == n_prefix."""
        def cls(sentences, threshold):
            # Flag first new sentence
            return [1] + [0] * (len(sentences) - 1)

        det = _make_detector(
            tmp_path,
            classifier_predict_fn=cls,
        )
        prefix = ["Prefix A.", "Prefix B."]
        new_text = "This is the leak. And this is clean."
        result = det.detect_continuation(prefix, new_text)
        assert result.is_leaking
        # The first new sentence (abs idx 2) is the leak
        assert result.first_leak_index == 2

    def test_no_new_sentences_is_clean(self, tmp_path):
        """If split_sentences(new_text) returns an empty list, result is clean."""
        det = _make_detector(tmp_path)
        # Whitespace-only new text produces no sentences after strip
        result = det.detect_continuation(["Prefix."], "   ")
        assert not result.is_leaking
        assert result.first_leak_index is None

    def test_confirmed_leak_prefix_flags_all_false(self, tmp_path):
        """On a confirmed specific-sentence leak, the prefix region of
        stage1_flags and confirmed_flags must be all False — the prefix
        sentences are never individually classified, so no entry should
        appear to have been flagged or confirmed."""
        def cls(sentences, threshold):
            return [1 if "leak" in s.lower() else 0 for s in sentences]

        det = _make_detector(tmp_path, classifier_predict_fn=cls)
        prefix = ["Prefix A.", "Prefix B.", "Prefix C."]
        new_text = "Clean new. This is a leak. Something else."
        result = det.detect_continuation(prefix, new_text)
        assert result.is_leaking
        n_prefix = len(prefix)
        assert result.stage1_flags[:n_prefix] == [False] * n_prefix
        assert result.confirmed_flags[:n_prefix] == [False] * n_prefix
        # The confirmed flag sits at the leak index, inside the new region
        assert result.first_leak_index is not None
        assert result.confirmed_flags[result.first_leak_index] is True

# ======================================================================
# Single-stage classifier path
# ======================================================================


def _make_single_stage_detector(
    tmp_path,
    predict_labels_fn,
):
    """Build a CoTLeakDetector with a mocked single-stage classifier.

    ``predict_labels_fn`` is ``(sentences, threshold) -> list[0|1]``.
    """
    from redact.regen.classifier import CoTClassificationResult

    def mc_classify_cot(sentences, threshold, prompt=None, prefix_sentences=None):
        del prompt, prefix_sentences  # protocol kwargs, unused by this mock
        return CoTClassificationResult(
            sentence_results=predict_labels_fn(sentences, threshold)
        )

    mock_classifier = MagicMock()
    mock_classifier.predict = predict_labels_fn
    mock_classifier.classify_cot = mc_classify_cot

    return CoTLeakDetector(
        classifier=mock_classifier,
        classifier_threshold=0.5,
    )


class TestSingleStageClassifierPath:
    """Tests for the single-stage classifier path (one batched
    ``classify_cot`` call, no Stage-2 confirmation step)."""

    def test_leak_detected(self, tmp_path):
        def mc_predict(sents, thr):
            return [1] * len(sents)

        det = _make_single_stage_detector(tmp_path, mc_predict)
        result = det.detect(f"{LEAKING_SENTENCE_0} {CLEAN_SENTENCE}")
        assert result.is_leaking
        assert result.first_leak_index == 0

    def test_clean_returns_no_leak(self, tmp_path):
        def mc_predict(sents, thr):
            return [0] * len(sents)

        det = _make_single_stage_detector(tmp_path, mc_predict)
        result = det.detect(f"{CLEAN_SENTENCE} {CLEAN_SENTENCE_2}")
        assert not result.is_leaking
        assert result.first_leak_index is None

    def test_first_leak_index_correct(self, tmp_path):
        def mc_predict(sents, thr):
            return [1 if "Basil" in s else 0 for s in sents]

        det = _make_single_stage_detector(tmp_path, mc_predict)
        result = det.detect(
            f"{CLEAN_SENTENCE} {LEAKING_SENTENCE_0} {CLEAN_SENTENCE_2}"
        )
        assert result.is_leaking
        assert result.first_leak_index == 1
        assert result.confirmed_flags == [False, True, False]

    def test_continuation_preserves_prefix(self, tmp_path):
        call_args = []

        def mc_predict(sents, thr):
            call_args.append(list(sents))
            return [1] * len(sents)

        det = _make_single_stage_detector(tmp_path, mc_predict)
        prefix = ["Clean prefix A.", "Clean prefix B."]
        result = det.detect_continuation(prefix, LEAKING_SENTENCE_0)
        assert result.is_leaking
        assert result.first_leak_index == 2
        assert len(call_args) == 1
        assert len(call_args[0]) == 1




# ---------------------------------------------------------------------------
# Prompt / prefix-sentences threading through the detector
# ---------------------------------------------------------------------------


class _CapturingClassifier:
    """LeakClassifier mock that records every classify_cot kwarg.

    Returns "no leak" for all sentences. We only care about *what got
    passed in*, not what comes out.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def predict(self, texts, threshold):
        return [0] * len(texts)

    def classify_cot(
        self, sentences, threshold, prompt=None, prefix_sentences=None,
    ):
        from redact.regen.classifier import CoTClassificationResult

        self.calls.append(
            {
                "sentences": list(sentences),
                "threshold": threshold,
                "prompt": prompt,
                "prefix_sentences": (
                    list(prefix_sentences) if prefix_sentences else None
                ),
            }
        )
        return CoTClassificationResult(
            sentence_results=[0] * len(sentences)
        )


class TestDetectorPromptThreading:
    def test_detect_passes_prompt(self):
        classifier = _CapturingClassifier()
        det = CoTLeakDetector(classifier=classifier, classifier_threshold=0.5)
        det.detect("Sentence one. Sentence two.", prompt="WHO is X?")
        assert len(classifier.calls) == 1
        call = classifier.calls[0]
        assert call["prompt"] == "WHO is X?"
        # No external prefix supplied; classifier sees None or empty.
        assert call["prefix_sentences"] is None

    def test_detect_passes_external_prefix(self):
        classifier = _CapturingClassifier()
        det = CoTLeakDetector(classifier=classifier, classifier_threshold=0.5)
        det.detect(
            "Answer first. Answer second.",
            prompt="P",
            external_prefix_sentences=["CoT s1.", "CoT s2."],
        )
        call = classifier.calls[0]
        # External prefix flows through as the classifier's prefix_sentences;
        # detect() has no cleared target prefix to concatenate.
        assert call["prefix_sentences"] == ["CoT s1.", "CoT s2."]

    def test_detect_continuation_concatenates_external_and_cleared_prefix(self):
        classifier = _CapturingClassifier()
        det = CoTLeakDetector(classifier=classifier, classifier_threshold=0.5)
        det.detect_continuation(
            prefix_sentences=["Cleared a.", "Cleared b."],
            new_text="New sentence.",
            prompt="P",
            external_prefix_sentences=["External 1.", "External 2."],
        )
        call = classifier.calls[0]
        # Order: external first, then cleared target prefix. The classifier
        # then walks new sentences with cumulative prior on top of this.
        assert call["prefix_sentences"] == [
            "External 1.",
            "External 2.",
            "Cleared a.",
            "Cleared b.",
        ]
        assert call["prompt"] == "P"

    def test_detect_batch_aligns_per_row_prompts_and_external_prefix(self):
        classifier = _CapturingClassifier()
        det = CoTLeakDetector(classifier=classifier, classifier_threshold=0.5)
        det.detect_batch(
            cots=["A1. A2.", "B1."],
            prompts=["promptA", "promptB"],
            external_prefix_sentences_per_row=[
                ["ext A.x"],
                ["ext B.x", "ext B.y"],
            ],
        )
        assert len(classifier.calls) == 2
        assert classifier.calls[0]["prompt"] == "promptA"
        assert classifier.calls[0]["prefix_sentences"] == ["ext A.x"]
        assert classifier.calls[1]["prompt"] == "promptB"
        assert classifier.calls[1]["prefix_sentences"] == ["ext B.x", "ext B.y"]

    def test_detect_batch_default_passes_none_per_row(self):
        classifier = _CapturingClassifier()
        det = CoTLeakDetector(classifier=classifier, classifier_threshold=0.5)
        det.detect_batch(cots=["A.", "B."])
        assert classifier.calls[0]["prompt"] is None
        assert classifier.calls[1]["prompt"] is None
        assert classifier.calls[0]["prefix_sentences"] is None

    def test_detect_batch_length_mismatch_raises(self):
        classifier = _CapturingClassifier()
        det = CoTLeakDetector(classifier=classifier, classifier_threshold=0.5)
        with pytest.raises(ValueError, match="prompts"):
            det.detect_batch(cots=["a", "b"], prompts=["only one"])
        with pytest.raises(ValueError, match="external_prefix_sentences_per_row"):
            det.detect_batch(
                cots=["a", "b"],
                external_prefix_sentences_per_row=[["only one"]],
            )
