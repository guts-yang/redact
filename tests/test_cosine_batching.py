"""Equivalence tests for the batched CosineSimilarity evaluator against its
pre-batching per-pair behavior.

The evaluator's forward pass now encodes all gold/gen strings in one
SentenceTransformer call (or two, for gold vs gen). These tests pin the
numerical output to what the previous per-pair implementation produced,
modulo floating-point noise from kernel-launch order.
"""
import pytest
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from redact.evaluator.cosine_similarity import CosineSimilarity


@pytest.fixture(scope="module")
def cs():
    return CosineSimilarity()


def _per_pair_cosine(model, answers, generated):
    """The pre-batching implementation, re-materialized for equivalence."""
    scores = []
    for a, ga in zip(answers, generated):
        a_emb = model.encode(a, show_progress_bar=False)
        ga_emb = model.encode(ga, show_progress_bar=False)
        sim = float(sk_cosine([a_emb], [ga_emb])[0][0])
        scores.append(max(0.0, sim))
    return scores


class TestCosineSimilarityBatchedEquivalence:
    def test_matches_per_pair_on_mixed_similarity(self, cs):
        answers = [
            "The cat sat on the mat.",
            "Photosynthesis converts light into chemical energy.",
            "Lorem ipsum dolor sit amet.",
        ]
        generated = [
            "A feline was resting on a rug.",
            "Plants use sunlight to produce energy.",
            "Totally unrelated sentence about rockets.",
        ]
        batched = cs.evaluate(answers, generated)
        per_pair = _per_pair_cosine(cs.model, answers, generated)
        assert len(batched) == len(per_pair)
        for b, p in zip(batched, per_pair):
            assert abs(b - p) < 1e-4, f"mismatch: batched={b} per_pair={p}"

    def test_empty_input_returns_empty(self, cs):
        assert cs.evaluate([], []) == []

    def test_identical_strings_score_near_one(self, cs):
        answers = ["Identical text.", "Another identical one."]
        result = cs.evaluate(answers, answers)
        for s in result:
            assert s > 0.99
