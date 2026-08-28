"""CoT leak detector for reasoning model unlearning.

Delegates all classification to a :class:`LeakClassifier` — the canonical
:class:`ContextualBinaryLeakClassifier` (binary RoBERTa head conditioned on
prompt + prior-sentence context). The detector handles sentence splitting
and continuation tracking.

No plain-text forget-set data is needed at inference — only trained model
weights.
"""

import dataclasses
import json
import os


from redact.regen.classifier import CoTClassificationResult, LeakClassifier
from redact.evaluator.utils import split_sentences


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclasses.dataclass
class LeakDetectionResult:
    """Result of leak detection on a single generated CoT.

    ``first_leak_index`` is the absolute index (into ``sentences``) of
    the first sentence the classifier confirmed as leaking, or ``None``
    when nothing leaks.  ``confirmed_flags`` marks EVERY leaking
    sentence, not only the first — the regen wave loop's
    exhaust-and-advance branch walks it to find leaks past the first
    one without re-running detection.
    """

    is_leaking: bool
    first_leak_index: int | None  # sentence index of first confirmed leak
    sentences: list[str]  # all CoT sentences
    stage1_flags: list[bool]  # per-sentence Stage 1 classifier flags
    confirmed_flags: list[bool]  # per-sentence confirmed-leak flags


# ------------------------------------------------------------------
# Knowledge bank utilities
# ------------------------------------------------------------------

def build_claims(answers: list[str]) -> list[str]:
    """Extract factual claims by sentence-splitting a list of gold answers.

    Each answer is split into sentences via ``split_sentences`` and flattened
    into a single ordered list.  Duplicates are preserved — apply
    :func:`dedupe_claims` afterwards if exact-string collapsing is desired.
    """
    claims: list[str] = []
    for answer in answers:
        for sentence in split_sentences(answer):
            claims.append(sentence)
    return claims


def dedupe_claims(claims: list[str]) -> tuple[list[str], list[int]]:
    """Collapse exact-match duplicates, preserving first-seen order.

    Returns
    -------
    deduped : list[str]
        Unique claims in first-seen order.
    old_to_new : list[int]
        For each position in ``claims``, the index of that claim in
        ``deduped``.  Use this to remap labels keyed to pre-dedup
        positions.
    """
    seen: dict[str, int] = {}
    deduped: list[str] = []
    old_to_new: list[int] = []
    for claim in claims:
        idx = seen.get(claim)
        if idx is None:
            idx = len(deduped)
            seen[claim] = idx
            deduped.append(claim)
        old_to_new.append(idx)
    return deduped, old_to_new


def load_knowledge_bank(knowledge_bank_dir: str) -> list[str]:
    """Load the claim list from disk.

    The claim's position in the returned list is its ID: label data and
    ``config/authors/<split>.json`` address claims by absolute index, so the
    order is part of the contract.
    """
    with open(os.path.join(knowledge_bank_dir, "claims.json")) as f:
        return json.load(f)


def save_knowledge_bank(claims: list[str], output_dir: str) -> None:
    """Persist the claim list to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "claims.json"), "w") as f:
        json.dump(claims, f, indent=2)


# ------------------------------------------------------------------
# Detector
# ------------------------------------------------------------------

class CoTLeakDetector:
    """CoT leak detector.

    Delegates all classification to a :class:`LeakClassifier` instance
    (the canonical :class:`ContextualBinaryLeakClassifier`). The detector
    handles sentence splitting, per-sentence dispatch, and continuation
    tracking — but never touches model weights or projections directly.

    Parameters
    ----------
    classifier : LeakClassifier
        Pre-constructed classifier satisfying the :class:`LeakClassifier`
        protocol.
    classifier_threshold : float | None
        Confidence threshold passed to the classifier. ``None`` resolves
        to the classifier's persisted ``threshold`` attribute (set by
        :class:`ContextualBinaryLeakClassifier` from ``binary_threshold``
        at construction time); falls back to
        ``0.5`` for legacy classifiers without that attribute. The
        resolve happens here so every caller benefits — ``None``
        previously crashed with ``TypeError`` inside ``_apply_threshold``
        when the threshold reached ``if p > None``.
    """

    def __init__(
        self,
        classifier: LeakClassifier,
        classifier_threshold: float | None = None,
    ):
        self.classifier = classifier
        resolved = (
            classifier_threshold
            if classifier_threshold is not None
            else float(getattr(classifier, "threshold", 0.5))
        )
        self.classifier_threshold: float = resolved

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        generated_cot: str,
        prompt: str | None = None,
        external_prefix_sentences: list[str] | None = None,
    ) -> LeakDetectionResult:
        """Analyse a single generated CoT for forget-set leakage.

        Splits the full CoT into sentences via Punkt, then runs the
        classifier on every sentence.  Use this for the initial
        detection pass before any regeneration; for subsequent iterations
        in a regeneration loop, prefer :meth:`detect_continuation` which
        skips re-classification of already-cleared prefix sentences.

        ``prompt`` and ``external_prefix_sentences`` flow through to the
        classifier as conditioning context for variants that use them
        (e.g. :class:`ContextualBinaryLeakClassifier`); non-contextual
        classifiers ignore both. ``external_prefix_sentences`` is the
        frozen prior text outside the target — e.g. the prior CoT's
        sentences when classifying answer text in the answer wave.
        """
        sentences = split_sentences(generated_cot)
        return self._classify_sentences(
            prefix_sentences=[],
            new_sentences=sentences,
            prompt=prompt,
            external_prefix_sentences=external_prefix_sentences,
        )

    def detect_continuation(
        self,
        prefix_sentences: list[str],
        new_text: str,
        prompt: str | None = None,
        external_prefix_sentences: list[str] | None = None,
    ) -> LeakDetectionResult:
        """Detect leaks in ``new_text`` only, carrying a cleared prefix.

        ``prefix_sentences`` are sentences that were previously classified
        as non-leaking — typically from an earlier ``detect()`` or
        ``detect_continuation()`` call whose ``sentences[:first_leak_index]``
        became the clean prefix for regeneration.  They are carried
        through in the result for downstream book-keeping but are NOT
        re-classified here; their text hasn't changed, so a deterministic
        classifier would return the same answer, and skipping this work
        is a strict speedup.

        ``new_text`` is the newly-generated continuation.  It is sentence-
        split independently of the prefix, so Punkt cannot re-segment the
        prefix region based on the new context.  This aligns the
        implementation with the intended design: appending new text after
        a complete sentence does not alter the completeness of that
        sentence.

        The returned ``first_leak_index`` is an absolute index into
        ``prefix_sentences + split_sentences(new_text)``.  Because the
        prefix is never individually re-classified,
        ``first_leak_index >= len(prefix_sentences)`` whenever a leak is
        confirmed.

        ``prompt`` and ``external_prefix_sentences`` flow through to the
        classifier exactly as in :meth:`detect`. ``external_prefix_sentences``
        is concatenated *before* ``prefix_sentences`` when forming the
        classifier's conditioning context, so the contextual classifier
        sees ``external + cleared_target_prefix + new_sentences[:i]``
        as the cumulative prior for ``new_sentences[i]``.
        """
        new_sentences = split_sentences(new_text)
        return self._classify_sentences(
            prefix_sentences=list(prefix_sentences),
            new_sentences=new_sentences,
            prompt=prompt,
            external_prefix_sentences=external_prefix_sentences,
        )

    def detect_batch(
        self,
        cots: list[str],
        prompts: list[str] | None = None,
        external_prefix_sentences_per_row: list[list[str]] | None = None,
    ) -> list[LeakDetectionResult]:
        """Run :meth:`detect` on each CoT independently.

        ``prompts``/``external_prefix_sentences_per_row``, if supplied, must
        be index-aligned with ``cots``. Either may be omitted (defaults to
        no prompt / empty external prefix per row). The contextual
        classifier uses both; non-contextual classifiers ignore both.
        """
        n = len(cots)
        if prompts is not None and len(prompts) != n:
            raise ValueError(
                f"detect_batch: len(prompts)={len(prompts)} does not match "
                f"len(cots)={n} — these inputs must be index-aligned."
            )
        if (
            external_prefix_sentences_per_row is not None
            and len(external_prefix_sentences_per_row) != n
        ):
            raise ValueError(
                f"detect_batch: len(external_prefix_sentences_per_row)="
                f"{len(external_prefix_sentences_per_row)} does not match "
                f"len(cots)={n} — these inputs must be index-aligned."
            )
        results: list[LeakDetectionResult] = []
        for i, cot in enumerate(cots):
            row_prompt = prompts[i] if prompts is not None else None
            row_ext = (
                external_prefix_sentences_per_row[i]
                if external_prefix_sentences_per_row is not None
                else None
            )
            results.append(
                self.detect(
                    cot,
                    prompt=row_prompt,
                    external_prefix_sentences=row_ext,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Shared classification core
    # ------------------------------------------------------------------

    def _classify_sentences(
        self,
        prefix_sentences: list[str],
        new_sentences: list[str],
        prompt: str | None = None,
        external_prefix_sentences: list[str] | None = None,
    ) -> LeakDetectionResult:
        """Classify ``new_sentences`` and return an absolute-indexed result.

        Sentences in ``prefix_sentences`` are treated as already cleared
        (non-leaking) and are not re-classified.  The returned
        ``first_leak_index`` is absolute over
        ``prefix_sentences + new_sentences``.

        For contextual classifiers, the conditioning prior passed to
        ``classify_cot`` is ``external_prefix_sentences + prefix_sentences``
        — the engine's frozen external context (e.g. CoT tail when
        classifying answer text) followed by the cleared target prefix.
        Sentence-internal cumulative prior (``sentences[:i]``) is added
        by the classifier itself.
        """
        n_prefix = len(prefix_sentences)
        n_new = len(new_sentences)
        all_sentences = list(prefix_sentences) + list(new_sentences)
        n_total = n_prefix + n_new

        if n_total == 0:
            return LeakDetectionResult(
                is_leaking=False,
                first_leak_index=None,
                sentences=[],
                stage1_flags=[],
                confirmed_flags=[],
            )

        if n_new == 0:
            # Nothing new to classify; prefix is assumed clean.
            return LeakDetectionResult(
                is_leaking=False,
                first_leak_index=None,
                sentences=all_sentences,
                stage1_flags=[False] * n_prefix,
                confirmed_flags=[False] * n_prefix,
            )

        external = list(external_prefix_sentences) if external_prefix_sentences else []
        classifier_prefix = external + list(prefix_sentences)
        cot_result: CoTClassificationResult = self.classifier.classify_cot(
            sentences=new_sentences,
            threshold=self.classifier_threshold,
            prompt=prompt,
            prefix_sentences=classifier_prefix,
        )

        # Per the uniform classifier contract pinned on
        # :class:`LeakClassifier`, ``label==1`` is the authoritative leak
        # signal.
        new_stage1_flags = [bool(lbl) for lbl in cot_result.sentence_results]
        stage1_flags = [False] * n_prefix + new_stage1_flags

        # ``confirmed_flags`` is True for every leaking new sentence, not
        # only the first.  This is what the regen wave loop's
        # exhaust-and-advance branch walks to find the next leak past a
        # persistent one without paying for re-classification.
        confirmed_flags: list[bool] = [False] * n_total
        first_leak_index: int | None = None
        for local_idx, lbl in enumerate(cot_result.sentence_results):
            if bool(lbl):
                abs_idx = n_prefix + local_idx
                confirmed_flags[abs_idx] = True
                if first_leak_index is None:
                    first_leak_index = abs_idx

        is_leaking = first_leak_index is not None
        return LeakDetectionResult(
            is_leaking=is_leaking,
            first_leak_index=first_leak_index,
            sentences=all_sentences,
            stage1_flags=stage1_flags,
            confirmed_flags=confirmed_flags,
        )
