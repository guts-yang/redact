"""Shared helpers for the contextual leak-classifier data pipeline.

The contextual labeling protocol pairs each focal sentence with up to ``k``
prior sentences from the same response.  The window must be built
identically across every source that produces contextual rows.  Those
sources all live in ``scripts/build_labeled_sources.py``, which
sentence-splits its raw inputs through :func:`build_phase_items`:

  * ``--source forget_structural`` — sentence-splits the forget CoT +
    answer into the placeholder-label base dataset that the first-leak
    orchestrator later relabels.
  * ``--source aux`` / ``--source retain`` — sentence-splits TOFU
    aux-split and retain responses into ``label=0`` background-class
    rows in the same contextual schema.
  * ``--source forget_answers`` / ``--source idk`` — sentence-split
    forget answers and IDK responses into additional positives and
    negatives respectively.

Because those rows are concatenated into a single dataset for
``train_classifier.py --model_type contextual_binary``, drift in the
window logic would make the merged dataset inconsistent: background rows
would carry different prior-context semantics than forget rows for the
same focal sentence.  Hence the shared helper.
"""

from __future__ import annotations

from typing import Iterator

from redact.regen.utils import PROMPT_PREFIX, PROMPT_SUFFIX, clean_question
from redact.evaluator.utils import split_sentences

__all__ = [
    "PROMPT_PREFIX",
    "PROMPT_SUFFIX",
    "clean_question",
    "build_phase_items",
]


def build_phase_items(
    cot: str, answer: str, k: int = 2
) -> Iterator[tuple[str, int, str, list[str]]]:
    """Yield ``(phase, sent_idx, focal, prior_window)`` tuples for cot+ans.

    Sentence-splits CoT and answer separately via the same Punkt path the
    inference-time detector uses, then walks each phase emitting one tuple
    per sentence.

    Window construction:

    * **CoT phase** (``phase='cot'``): ``window = cot_sents[max(0, i-k):i]``
      — the up-to-``k`` prior sentences from the same CoT, in document
      order.  Empty list at i=0.

    * **Answer phase** (``phase='ans'``): the window crosses the
      ``</think>`` boundary so early-answer sentences see the tail of the
      CoT as context:

      - ``j >= k`` : ``window = ans_sents[j-k:j]`` (purely within answer)
      - ``j <  k`` : ``window = cot_sents[-(k-j):] + ans_sents[:j]``
        (last ``k-j`` CoT sentences + first ``j`` answer sentences)

      The ``</think>`` token itself is not visible to the splitter: CoT
      and answer are pre-split by the engine at the canonical delimiter,
      and this helper just stitches the windows across that boundary.

    Empty / whitespace-only ``cot`` or ``answer`` yields no items for
    that phase (matches the existing forget pipeline).
    """
    cot_sents = split_sentences(cot.strip()) if cot and cot.strip() else []
    ans_sents = split_sentences(answer.strip()) if answer and answer.strip() else []

    for i, focal in enumerate(cot_sents):
        window = cot_sents[max(0, i - k):i]
        yield "cot", i, focal, window

    for j, focal in enumerate(ans_sents):
        if j >= k:
            window = ans_sents[j - k:j]
        else:
            cot_part = cot_sents[-(k - j):] if (k - j) > 0 else []
            ans_part = ans_sents[:j]
            window = cot_part + ans_part
        yield "ans", j, focal, window
