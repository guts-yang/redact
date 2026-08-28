from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod

from transformers import logging

logging.set_verbosity_error()


@dataclasses.dataclass
class CoTClassificationResult:
    """Result of classifying a full CoT via :meth:`LeakClassifier.classify_cot`.

    ``sentence_results`` contains one integer label per input sentence,
    in the same order as the sentences that were classified: ``1`` for a
    leaking sentence, ``0`` for a clean one.
    """

    sentence_results: list[int]


class LeakClassifier(ABC):
    """Base class for leak classifiers used by :class:`CoTLeakDetector`.

    Subclasses:

    - :class:`ContextualBinaryLeakClassifier` — binary contextual
      classifier (prompt + k-window prior context).

    **Per-sentence return contract** (binding on every subclass):

    - ``label == 1`` means *confirmed leak* — the detector treats this as
      a flag.
    - ``label == 0`` means *clean*.

    Downstream consumers (:class:`CoTLeakDetector`, the regen engine, the
    soft-token corruption hook) gate solely on this label: the
    inference-time corruption is a single shared :class:`SoftToken`, so
    a flagged sentence carries no further routing information.
    """

    @abstractmethod
    def predict(self, texts: list[str], threshold: float) -> list[int]:
        """Return per-text binary labels (0 = clean, 1 = leak)."""

    @abstractmethod
    def classify_cot(
        self,
        sentences: list[str],
        threshold: float,
        prompt: str | None = None,
        prefix_sentences: list[str] | None = None,
    ) -> CoTClassificationResult:
        """Classify individual sentences and return per-sentence results.

        ``prompt`` and ``prefix_sentences`` are accepted by the protocol so
        the detector can pass them uniformly across classifier variants.
        Non-contextual classifiers ignore them; the contextual classifier
        uses them as conditioning context.
        """


class ContextualBinaryLeakClassifier(LeakClassifier):
    """Binary contextual leak classifier.

    Each sentence is classified in context — the prompt plus a k-sentence
    window of prior context, assembled with ``<TGT>`` markers and the
    ``[CTX_TRUNC]`` flag (see
    :func:`redact.regen.contextual_inputs.build_contextual_input`). The
    underlying model is :class:`BinaryHeadModel`: encoder + a single
    binary head.

    Every classification path returns plain ``0``/``1`` labels, per the
    :class:`LeakClassifier` contract. A flagged sentence triggers
    regeneration with the single shared :class:`SoftToken`; there is no
    per-sentence routing beyond the label.

    Parameters
    ----------
    model_path : str
        Path to a saved :class:`BinaryHeadModel` checkpoint whose tokenizer
        carries the contextual special tokens.
    threshold : float | None
        ``P(leak)`` cutoff. When None, loaded from
        ``BinaryHeadConfig.binary_threshold``.
    batch_size : int
        Inference batch size.
    max_length : int
        Tokenizer max sequence length (default 512).
    target_budget : int
        Tokens reserved for ``<TGT> target </TGT>``.
    max_prior_sentences : int | None
        Hard cap on prior sentences. When None, loaded from
        ``BinaryHeadConfig.max_prior_sentences``.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        target_budget: int = 128,
        max_prior_sentences: int | None = None,
    ):
        import os as _os

        import safetensors.torch as _st
        import torch as _torch
        from transformers import AutoTokenizer as _AutoTokenizer

        from redact.regen.binary_head_model import BinaryHeadConfig, BinaryHeadModel
        from redact.regen.contextual_inputs import CONTEXTUAL_SPECIAL_TOKENS

        self.device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.target_budget = target_budget

        self.tokenizer = _AutoTokenizer.from_pretrained(model_path)
        for tok in CONTEXTUAL_SPECIAL_TOKENS:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is None or tid == self.tokenizer.unk_token_id:
                raise ValueError(
                    f"ContextualBinaryLeakClassifier: tokenizer at {model_path!r} "
                    f"is missing required special token {tok!r}. Was the checkpoint "
                    f"saved by scripts/train_classifier.py with "
                    f"--model_type contextual_binary?"
                )
        self.config = BinaryHeadConfig.from_pretrained(model_path)
        self.threshold = (
            threshold if threshold is not None else self.config.binary_threshold
        )
        self.max_prior_sentences = (
            max_prior_sentences
            if max_prior_sentences is not None
            else self.config.max_prior_sentences
        )
        # Resize-then-load workaround: backbone_config records the
        # pre-resize RoBERTa vocab_size, but the saved weights are resized
        # to len(tokenizer).
        self.model = BinaryHeadModel(self.config)
        self.model.backbone.resize_token_embeddings(len(self.tokenizer))
        weights_path = _os.path.join(model_path, "model.safetensors")
        if not _os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Expected safetensors weights at {weights_path!r}; "
                f"sharded / .bin checkpoints are not supported here."
            )
        state = _st.load_file(weights_path)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint at {model_path!r} has missing/unexpected keys: "
                f"missing={missing[:5]}{'...' if len(missing) > 5 else ''}, "
                f"unexpected={unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
            )
        self.model.to(self.device)
        self.model.eval()

    def _build_inputs(
        self,
        prompt: str | None,
        priors_per_target: list[list[str]],
        targets: list[str],
    ) -> dict:
        import torch as _torch

        from redact.regen.contextual_inputs import build_contextual_input

        if len(priors_per_target) != len(targets):
            raise ValueError(
                f"_build_inputs: priors_per_target length "
                f"({len(priors_per_target)}) does not match targets length "
                f"({len(targets)})"
            )
        built = [
            build_contextual_input(
                prompt=prompt or "",
                prior_sentences=priors_per_target[i],
                target_sentence=t,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                target_budget=self.target_budget,
                max_prior_sentences=self.max_prior_sentences,
            )
            for i, t in enumerate(targets)
        ]
        max_len = max(len(b.input_ids) for b in built)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer has no pad_token_id")
        input_ids = _torch.full(
            (len(built), max_len), pad_id, dtype=_torch.long
        )
        attention_mask = _torch.zeros(
            (len(built), max_len), dtype=_torch.long
        )
        for i, b in enumerate(built):
            n = len(b.input_ids)
            input_ids[i, :n] = _torch.tensor(b.input_ids, dtype=_torch.long)
            attention_mask[i, :n] = 1
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }

    def _forward_batched(
        self,
        targets: list[str],
        priors_per_target: list[list[str]],
        prompt: str | None,
    ) -> list[float]:
        """Run the binary head on (target, prior) pairs in batches.

        Returns ``p_leaks`` per target. No threshold applied here; callers
        wrap with their own threshold logic.
        """
        import torch as _torch

        if len(priors_per_target) != len(targets):
            raise ValueError(
                f"_forward_batched: priors_per_target length "
                f"({len(priors_per_target)}) does not match targets length "
                f"({len(targets)})"
            )
        all_p_leaks: list[float] = []
        for start in range(0, len(targets), self.batch_size):
            end = start + self.batch_size
            batch_targets = targets[start:end]
            batch_priors = priors_per_target[start:end]
            encodings = self._build_inputs(prompt, batch_priors, batch_targets)
            with _torch.no_grad():
                outputs = self.model(**encodings)
            binary_logits = outputs["logits"]
            binary_probs = _torch.softmax(binary_logits, dim=-1)
            for i in range(len(batch_targets)):
                all_p_leaks.append(float(binary_probs[i, 1].item()))
        return all_p_leaks

    def _apply_threshold(
        self, p_leaks: list[float], threshold: float,
    ) -> list[int]:
        """Apply ``threshold`` and return one ``0``/``1`` label per score.

        A score strictly above ``threshold`` is a leak. Downstream
        consumers (the detector, the regen engine) gate on this label
        alone; see ``CoTLeakDetector._classify_sentences``.
        """
        return [1 if p > threshold else 0 for p in p_leaks]

    def predict(self, texts: list[str], threshold: float = 0.5) -> list[int]:
        priors = [[] for _ in texts]
        p_leaks = self._forward_batched(
            targets=texts, priors_per_target=priors, prompt=None,
        )
        return self._apply_threshold(p_leaks, threshold)

    def score_cot(
        self,
        sentences: list[str],
        prompt: str | None = None,
        prefix_sentences: list[str] | None = None,
    ) -> list[float]:
        """Per-sentence ``p_leak`` with cumulative context.

        Prior-construction semantics: for sentence ``sentences[i]``, the
        effective prior context is ``prefix_sentences + sentences[:i]``.
        No threshold applied.
        """
        prior_prefix = list(prefix_sentences) if prefix_sentences else []
        priors_per_target = [
            prior_prefix + list(sentences[:i]) for i in range(len(sentences))
        ]
        return self._forward_batched(
            targets=list(sentences),
            priors_per_target=priors_per_target,
            prompt=prompt,
        )

    def classify_cot(
        self,
        sentences: list[str],
        threshold: float,
        prompt: str | None = None,
        prefix_sentences: list[str] | None = None,
    ) -> CoTClassificationResult:
        prior_prefix = list(prefix_sentences) if prefix_sentences else []
        priors_per_target = [
            prior_prefix + list(sentences[:i]) for i in range(len(sentences))
        ]
        p_leaks = self._forward_batched(
            targets=list(sentences),
            priors_per_target=priors_per_target,
            prompt=prompt,
        )
        return CoTClassificationResult(
            sentence_results=self._apply_threshold(p_leaks, threshold),
        )
