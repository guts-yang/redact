"""Tests for ``ContextualBinaryLeakClassifier`` checkpoint loading.

Exercises :class:`BinaryHeadModel` through the classifier wrapper, with
particular attention to the embedding-size mismatch on load:
``backbone_config`` records the pre-resize RoBERTa ``vocab_size`` while
the saved weights are resized to ``len(tokenizer)``, since the
contextual special-token registration + ``resize_token_embeddings``
happens at training time.
"""
from __future__ import annotations

import pytest
from transformers import AutoConfig, AutoTokenizer

from redact.regen.binary_head_model import BinaryHeadConfig, BinaryHeadModel
from redact.regen.contextual_inputs import CONTEXTUAL_SPECIAL_TOKENS


def _tiny_backbone_config():
    """Tiny RoBERTa config — keeps the real tokenizer's vocab so we can
    use the production special-token handling, but shrinks every other
    dimension so the safetensors round-trip is a few MB.
    """
    cfg = AutoConfig.from_pretrained("roberta-base")
    cfg.hidden_size = 32
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = 2
    cfg.intermediate_size = 64
    return cfg


def _build_and_save(save_dir, *, add_special_tokens: bool,
                    binary_threshold: float = 0.05,
                    max_prior_sentences: int = 2):
    """Mirror ``train_classifier.py``'s save shape: resize embeddings AFTER
    building the model, leaving ``backbone_config.vocab_size`` at the
    pre-resize value. This is what produces the bug class the test guards.
    """
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    if add_special_tokens:
        n_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": list(CONTEXTUAL_SPECIAL_TOKENS)}
        )
        assert n_added == 3, f"expected 3 special tokens added, got {n_added}"

    backbone_cfg = _tiny_backbone_config()
    config = BinaryHeadConfig(
        backbone_config=backbone_cfg.to_dict(),
        binary_threshold=binary_threshold,
        max_prior_sentences=max_prior_sentences,
    )
    model = BinaryHeadModel(config)
    if add_special_tokens:
        model.backbone.resize_token_embeddings(len(tokenizer))

    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    return tokenizer


@pytest.fixture(scope="module")
def contextual_binary_ckpt(tmp_path_factory):
    save_dir = tmp_path_factory.mktemp("contextual_binary_ckpt")
    _build_and_save(save_dir, add_special_tokens=True)
    return str(save_dir)


# ---------- regression: load with resized embeddings ----------

def test_loads_with_resized_embeddings(contextual_binary_ckpt):
    """``ContextualBinaryLeakClassifier`` must load a checkpoint whose
    saved weights are resized past ``backbone_config.vocab_size``.  The
    fix: build the bare model, resize, then load the state dict
    directly.
    """
    from redact.regen.classifier import ContextualBinaryLeakClassifier

    clf = ContextualBinaryLeakClassifier(model_path=contextual_binary_ckpt)

    # Defaults flow from the saved BinaryHeadConfig.
    assert clf.threshold == pytest.approx(0.05)
    assert clf.max_prior_sentences == 2

    embedding_rows = clf.model.backbone.embeddings.word_embeddings.weight.shape[0]
    assert embedding_rows == len(clf.tokenizer)
    assert embedding_rows == 50265 + 3


# ---------- constructor overrides ----------

def test_threshold_constructor_override_wins(contextual_binary_ckpt):
    from redact.regen.classifier import ContextualBinaryLeakClassifier

    clf = ContextualBinaryLeakClassifier(
        model_path=contextual_binary_ckpt, threshold=0.7,
    )
    assert clf.threshold == pytest.approx(0.7)


def test_max_prior_constructor_override_wins(contextual_binary_ckpt):
    from redact.regen.classifier import ContextualBinaryLeakClassifier

    clf = ContextualBinaryLeakClassifier(
        model_path=contextual_binary_ckpt, max_prior_sentences=4,
    )
    assert clf.max_prior_sentences == 4


# ---------- failure modes ----------

def test_missing_special_tokens_fails_loud(tmp_path):
    """A checkpoint whose tokenizer is missing the contextual special
    tokens must fail at construction with a clear error pointing at
    train_classifier.py.
    """
    from redact.regen.classifier import ContextualBinaryLeakClassifier

    save_dir = tmp_path / "no_specials"
    _build_and_save(save_dir, add_special_tokens=False)

    with pytest.raises(ValueError, match="missing required special token"):
        ContextualBinaryLeakClassifier(model_path=str(save_dir))


def test_missing_safetensors_fails_loud(tmp_path):
    """If ``model.safetensors`` is absent (e.g. a sharded or .bin
    checkpoint), the load path must fail with a clear message rather
    than producing a silently-uninitialized model.
    """
    import os

    from redact.regen.classifier import ContextualBinaryLeakClassifier

    save_dir = tmp_path / "missing_weights"
    _build_and_save(save_dir, add_special_tokens=True)
    os.remove(save_dir / "model.safetensors")

    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        ContextualBinaryLeakClassifier(model_path=str(save_dir))


# ---------- end-to-end smoke ----------

def test_classify_cot_returns_flat_labels(contextual_binary_ckpt):
    """End-to-end forward pass through the contextual builder + binary
    head. Random init (``from_config``), so we only check the shape and
    the per-sentence contract: one plain 0/1 label per input sentence.
    """
    from redact.regen.classifier import ContextualBinaryLeakClassifier

    clf = ContextualBinaryLeakClassifier(model_path=contextual_binary_ckpt)
    result = clf.classify_cot(
        sentences=["First sentence.", "Second sentence."],
        threshold=0.5,
        prompt="What is the question?",
        prefix_sentences=[],
    )
    assert len(result.sentence_results) == 2
    for label in result.sentence_results:
        # Bare int labels — no per-sentence payload beyond the flag.
        assert isinstance(label, int)
        assert label in (0, 1)


def test_score_cot_returns_floats(contextual_binary_ckpt):
    """``score_cot`` returns a plain ``list[float]`` — raw P(leak)
    values for threshold sweeps.
    """
    from redact.regen.classifier import ContextualBinaryLeakClassifier

    clf = ContextualBinaryLeakClassifier(model_path=contextual_binary_ckpt)
    scores = clf.score_cot(
        sentences=["First.", "Second.", "Third."],
        prompt="Question?",
        prefix_sentences=["Prior context."],
    )
    assert len(scores) == 3
    for s in scores:
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0
