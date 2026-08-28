"""Leak classifier head: an encoder backbone plus a single binary head.

The head answers one question per input — does this sentence leak? — which
is exactly what the inference-time corruption needs, since a single shared
:class:`SoftToken` is applied wherever a leak is flagged.

Uses a HuggingFace-native save/load shape with the contextual
special-token resize-embeddings load workaround in
``ContextualBinaryLeakClassifier``.

Forward signature: ``labels`` is the binary target (0 = not leak, 1 = leak).
The training script renames the dataset's ``label`` column to ``labels``
for this model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel


def compute_gap_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    is_idk_negative: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Pairwise hinge loss on the logit-space leak score (logits[:,1]-logits[:,0]).

    For every (positive, IDK-negative) pair in the batch we penalize the
    margin violation ``max(0, margin - (score_pos - score_idk))`` and
    average across all pairs. The deployment metric the leak classifier
    is selected on is ``min(score | pos) > max(score | idk)``; this loss
    directly maximizes the pos↔idk gap in the same logit space.

    Parameters
    ----------
    logits
        Binary head logits, shape ``[B, 2]``.
    labels
        Binary targets, shape ``[B]`` (``0`` = not-leak, ``1`` = leak).
    is_idk_negative
        Per-row mask, shape ``[B]``. ``True`` selects IDK negatives
        (``label==0 AND source == "idk"``). Coerced via ``.bool()``
        so any truthy tensor (``torch.bool``, ``int8``, ``float32``
        0.0/1.0, etc.) is accepted; ``0`` → ``False`` and anything
        non-zero → ``True``. Materialized by the training script and
        already excludes positives, but the loss only uses the
        entries where the mask coerces to ``True`` so any accidental
        truthy entry on a label=1 row would be silently treated as an
        IDK negative — the materialization closure raises on that
        case before forward() is ever called.
    margin
        Logit-space margin for the hinge. Sized for typical RoBERTa
        logit scales; the project default is ``4.0``.

    Returns
    -------
    torch.Tensor
        0-d tensor on the same device/dtype as ``logits``. Returns
        ``logits.new_zeros(())`` (a 0-d zero with the autograd graph
        attached only to ``logits``) when the batch contains no
        positives or no IDK negatives — graceful no-op so the stratified
        sampler isn't a correctness prerequisite, only a sample-
        efficiency win.

    Notes
    -----
    Out-of-place only (``F.relu``); the result preserves the autograd
    graph for ``logits``. Mean over ALL ``[N_pos × N_idk]`` pairs gives
    dense gradient signal even when one population is tiny in the batch.
    """
    score = logits[:, 1] - logits[:, 0]  # logit-space leak score
    pos_scores = score[labels == 1]
    idk_scores = score[is_idk_negative.bool()]
    if pos_scores.numel() == 0 or idk_scores.numel() == 0:
        return logits.new_zeros(())
    # [N_pos, N_idk] pairwise differences.
    diff = pos_scores.unsqueeze(1) - idk_scores.unsqueeze(0)
    return F.relu(margin - diff).mean()


class BinaryHeadConfig(PretrainedConfig):
    """Config for the leak classifier.

    Carries only the fields the binary head actually needs: the backbone
    config, dropout, the binary class weights / decision threshold, the
    train-time context-window size, and the optional loss-shaping knobs
    (per-source weights, focal gates, pairwise gap loss).
    """

    model_type = "binary_head"

    def __init__(
        self,
        backbone_config: dict | None = None,
        classifier_dropout: float | None = None,
        binary_class_weights: list[float] | None = None,
        binary_threshold: float = 0.5,
        max_prior_sentences: int = 2,
        per_source_loss_weights: dict[str, float] | None = None,
        positive_focal_gamma: float = 0.0,
        positive_focal_label_smoothing: float = 0.0,
        idk_focal_gamma: float = 0.0,
        idk_focal_label_smoothing: float = 0.0,
        gap_loss_weight: float = 0.0,
        gap_margin: float = 4.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_config = backbone_config or {}
        self.classifier_dropout = classifier_dropout
        self.binary_class_weights = binary_class_weights
        # Persisted so ContextualBinaryLeakClassifier reads the train-time
        # operating point without an explicit --leak_classifier_threshold.
        self.binary_threshold = binary_threshold
        # Contextual k-window. Harmless for any future non-contextual
        # binary checkpoint; required for the contextual classifier.
        self.max_prior_sentences = max_prior_sentences
        # Per-source loss weight map resolved at train time
        # (e.g. ``{"idk": 10.0, "aux": 0.4, "retain": 2.07}``).
        # Persisted so a downstream consumer can audit which per-source
        # reweighting produced this checkpoint. Not consumed at inference.
        self.per_source_loss_weights = per_source_loss_weights
        # Hard-positive mining (focal-style). When > 0, positive (label=1)
        # rows receive an in-graph dynamic weight ((1-P(leak))**gamma)
        # computed from the detached binary logits, composed
        # multiplicatively with the static per-row ``loss_weight`` (if
        # any) and the class-balance weight from CrossEntropyLoss. The
        # default 0.0 is a no-op: the dynamic weight is identically 1.0
        # for every row, so the loss matches the pre-flag implementation
        # bit-for-bit. Focal-style hard-example mining (Lin et al.),
        # applied to the positive class.
        if positive_focal_gamma < 0:
            raise ValueError(
                f"positive_focal_gamma must be >= 0; got {positive_focal_gamma}."
            )
        self.positive_focal_gamma = positive_focal_gamma
        # Optional label-smoothing for the focal gate: clip (1 - P(leak))
        # to [eps, 1 - eps] before the gamma power so a single high-
        # confidence false positive can't get a 10^4 weight kick early
        # in training when the head has not yet calibrated. Default 0.0
        # disables clipping (recovers raw (1-P(leak))**gamma, with only
        # the floor of 1e-6 in forward() to prevent a 0**0 edge). Only
        # consulted when positive_focal_gamma > 0.
        if not (0.0 <= positive_focal_label_smoothing < 0.5):
            raise ValueError(
                f"positive_focal_label_smoothing must lie in [0, 0.5); "
                f"got {positive_focal_label_smoothing}."
            )
        self.positive_focal_label_smoothing = positive_focal_label_smoothing
        # Hard-IDK-negative mining: symmetric counterpart to positive
        # focal mining for the IDK negative pool. When > 0, rows where
        # ``is_idk_negative=True`` (label=0 AND ``source == "idk"``,
        # materialized by the training script) receive an in-graph
        # dynamic weight ``P(leak)**gamma`` computed from the detached
        # binary logits.
        # Hard IDK negatives have HIGH P(leak) (the classifier is
        # mistakenly flagging them as leaks); the focal weight
        # concentrates gradient on those rows, attacking the cascading-
        # second-regen failure mode that an IDK false positive
        # triggers at inference. Default 0.0 is a no-op. Same focal
        # form as the positive gate above, restricted to the IDK
        # subpopulation rather than all negatives.
        if idk_focal_gamma < 0:
            raise ValueError(
                f"idk_focal_gamma must be >= 0; got {idk_focal_gamma}."
            )
        self.idk_focal_gamma = idk_focal_gamma
        # Label smoothing on the IDK focal gate: clip ``P(leak)`` to
        # ``[eps, 1 - eps]`` before the gamma power so a single
        # high-confidence IDK false positive (P(leak) ≈ 1) doesn't
        # eat the entire batch's loss budget early in training when
        # the head has not yet calibrated. Only consulted when
        # ``idk_focal_gamma > 0``.
        if not (0.0 <= idk_focal_label_smoothing < 0.5):
            raise ValueError(
                f"idk_focal_label_smoothing must lie in [0, 0.5); "
                f"got {idk_focal_label_smoothing}."
            )
        self.idk_focal_label_smoothing = idk_focal_label_smoothing
        # Pairwise hinge "gap loss" between positive and IDK-negative
        # logit-space leak scores. Adds a margin-maximization term to
        # the total loss that explicitly drives
        # ``min(score|pos) - max(score|idk)`` above ``gap_margin``,
        # the inference-time quantity the classifier is selected on.
        # ``gap_loss_weight = 0.0`` is a no-op default (existing
        # training behavior is preserved bit-for-bit). The hinge is
        # computed in logit space on ``logits[:, 1] - logits[:, 0]``
        # and averaged over ALL pairs in the batch — see
        # :func:`compute_gap_loss` for the full contract.
        if gap_loss_weight < 0:
            raise ValueError(
                f"gap_loss_weight must be >= 0; got {gap_loss_weight}."
            )
        self.gap_loss_weight = gap_loss_weight
        if gap_margin < 0:
            raise ValueError(
                f"gap_margin must be >= 0; got {gap_margin}."
            )
        self.gap_margin = gap_margin


class BinaryHeadModel(PreTrainedModel):
    """Encoder with a single binary leak head.

    Parameters
    ----------
    config : BinaryHeadConfig
        Must carry ``backbone_config``. Optionally ``binary_class_weights``
        as a list — survives save/load via the config.
    binary_class_weights : torch.Tensor | None
        Shape ``(2,)`` for weighted binary CE.  If ``None``, falls back to
        ``config.binary_class_weights``.
    """

    config_class = BinaryHeadConfig

    def __init__(
        self,
        config: BinaryHeadConfig,
        binary_class_weights: torch.Tensor | None = None,
    ):
        super().__init__(config)

        if not config.backbone_config:
            raise ValueError(
                "BinaryHeadConfig.backbone_config is empty. Build the config "
                "with backbone_config=AutoConfig.from_pretrained(name).to_dict()."
            )
        backbone_dict = dict(config.backbone_config)
        model_type = backbone_dict.pop("model_type", "roberta")
        backbone_cfg = AutoConfig.for_model(model_type, **backbone_dict)
        self.backbone = AutoModel.from_config(backbone_cfg)
        hidden = backbone_cfg.hidden_size

        dropout_rate = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else getattr(backbone_cfg, "hidden_dropout_prob", 0.1)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.binary_head = nn.Linear(hidden, 2)

        if binary_class_weights is None and config.binary_class_weights is not None:
            binary_class_weights = torch.tensor(
                config.binary_class_weights, dtype=torch.float32,
            )
        # ``reduction='none'`` lets ``forward()`` apply optional per-row
        # ``loss_weight`` multipliers before reduction.  Without weights
        # we reduce to mean() ourselves, recovering the
        # default-CrossEntropyLoss behavior.
        self.binary_loss_fn = nn.CrossEntropyLoss(
            weight=binary_class_weights, reduction="none",
        )

        # Hoist focal-mining knobs onto the model so forward() doesn't
        # round-trip through self.config every step.
        # ``getattr`` with a 0.0 default so an old checkpoint whose config
        # predates these fields loads cleanly with focal disabled.
        self.positive_focal_gamma = getattr(config, "positive_focal_gamma", 0.0)
        self.positive_focal_label_smoothing = getattr(
            config, "positive_focal_label_smoothing", 0.0,
        )
        self.idk_focal_gamma = getattr(config, "idk_focal_gamma", 0.0)
        self.idk_focal_label_smoothing = getattr(
            config, "idk_focal_label_smoothing", 0.0,
        )
        # Pairwise hinge gap-loss knobs. ``getattr`` with the defaults
        # so a pre-flag checkpoint loads cleanly with the gap term
        # disabled (no-op).
        self.gap_loss_weight = getattr(config, "gap_loss_weight", 0.0)
        self.gap_margin = getattr(config, "gap_margin", 4.0)
        # Latest raw (unweighted) gap-loss value, refreshed every
        # forward call. ``None`` means the gap-loss gate is disabled
        # or the last forward was logits-only (labels=None). Stored
        # as a detached 0-d tensor (NOT ``.item()``) to avoid a
        # CUDA host-device sync every step; consumers (callbacks,
        # tests) call ``.item()`` on their own cadence. Per-call
        # clobber, not an accumulator — under gradient accumulation
        # this reflects only the last micro-batch's value, and
        # under DDP it is rank-local. Callers needing exact
        # per-optimizer-step or cross-rank values should accumulate
        # in their callback.
        self._last_gap_loss: torch.Tensor | None = None

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,  # binary 0/1 target
        loss_weight=None,  # optional per-row weight (B,)
        is_idk_negative=None,  # optional per-row bool (B,) — IDK focal mask
        **kwargs,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        pooled = self.dropout(outputs.last_hidden_state[:, 0, :])  # [CLS]
        logits = self.binary_head(pooled)  # (B, 2)

        loss = None
        gap_loss = None
        if labels is not None:
            per_sample_loss = self.binary_loss_fn(logits, labels.long())

            # Static per-row weight (per-source spec). When absent and
            # both focal gammas are also 0 we take the cheap mean()
            # path. The weighted-mean branch below would produce a
            # bit-identical reduction for ``lw = ones``, but the cheap
            # path saves a tensor allocation per step and keeps the
            # default-config call trivial to reason about.
            if (
                loss_weight is None
                and self.positive_focal_gamma == 0.0
                and self.idk_focal_gamma == 0.0
            ):
                loss = per_sample_loss.mean()
            else:
                if loss_weight is None:
                    lw = torch.ones_like(per_sample_loss)
                else:
                    lw = loss_weight.to(
                        device=per_sample_loss.device,
                        dtype=per_sample_loss.dtype,
                    )
                    if lw.shape != per_sample_loss.shape:
                        raise ValueError(
                            f"loss_weight shape {tuple(lw.shape)} must "
                            f"match per-row loss shape "
                            f"{tuple(per_sample_loss.shape)} (one weight "
                            f"per row)."
                        )

                # Hard-positive mining: weight label=1 rows by a focal
                # factor (1 - P(leak))^gamma computed from detached
                # binary logits — it keys on label==1 and reads
                # (1 - P(leak)).
                # Detached
                # so gradient flows only through CE, not through the
                # gate (no positive-feedback loop where the head pushes
                # P(leak) toward whichever extreme minimises its own
                # weight).
                if self.positive_focal_gamma > 0.0:
                    p_leak = torch.softmax(logits.detach(), dim=-1)[:, 1]
                    one_minus_p = 1.0 - p_leak
                    eps = self.positive_focal_label_smoothing
                    if eps > 0.0:
                        # Label-smoothed clip: bound the gate in
                        # [eps, 1-eps]. Caps the per-row dynamic
                        # weight ratio at ((1-eps)/eps)**gamma, so a
                        # single high-confidence false positive can't
                        # blow up early in training.
                        one_minus_p = one_minus_p.clamp(min=eps, max=1.0 - eps)
                    else:
                        # Strictly-positive floor so 0**gamma is well-
                        # defined for any gamma >= 0. 1e-6 is well below
                        # any realistic effective probability for a
                        # row-level mass measurement.
                        one_minus_p = one_minus_p.clamp(min=1e-6)
                    pos_weight_dyn = one_minus_p.pow(self.positive_focal_gamma)
                    # Apply only to label=1 rows. Negatives keep their
                    # existing weight (the negative focal branch below
                    # handles IDK negatives separately).
                    pos_mask = labels.long() == 1
                    lw = torch.where(pos_mask, lw * pos_weight_dyn, lw)

                # Hard-IDK-negative mining: symmetric counterpart to
                # the positive branch above for the IDK pool. Weights
                # rows where ``is_idk_negative=True`` by a focal factor
                # ``P(leak)^gamma`` (high P(leak) on an IDK negative is
                # the hard case — the classifier is mistakenly
                # flagging it as a leak). Detached from autograd for
                # the same reason as the positive focal gate.
                #
                # Note: this branch (like the positive focal branch
                # above) applies at BOTH train and eval. HF Trainer's
                # ``prediction_step`` calls ``forward`` on eval
                # batches when labels are present, so the reported
                # ``eval_<split>_loss`` is the focal-weighted
                # reduction, not raw CE. Use the
                # ``effective_idk_mass`` / ``gap_idk`` / pAUC
                # headlines as selection signals; the focal-weighted
                # eval loss is not directly comparable across
                # checkpoints with different gamma settings.
                if self.idk_focal_gamma > 0.0:
                    if is_idk_negative is None:
                        raise ValueError(
                            "idk_focal_gamma > 0 but is_idk_negative was "
                            "not passed to forward(). The training script "
                            "must materialize the is_idk_negative column "
                            "on the tokenized dataset; check "
                            "scripts/train_classifier.py for the "
                            "_attach_is_idk_negative closure."
                        )
                    # Defensive bool cast. ``DataCollatorWithPadding``
                    # produces a ``torch.bool`` tensor from an Arrow
                    # ``Value('bool')`` column (verified against the
                    # collator), so this cast is a no-op in the
                    # canonical path. The cast is kept defensively in
                    # case a future collator change emits an int8 /
                    # uint8 column instead: ``0 → False``, anything
                    # else ``→ True``, preserving correctness.
                    idk_mask = is_idk_negative.to(
                        device=per_sample_loss.device, dtype=torch.bool,
                    )
                    if idk_mask.shape != per_sample_loss.shape:
                        raise ValueError(
                            f"is_idk_negative shape {tuple(idk_mask.shape)} "
                            f"must match per-row loss shape "
                            f"{tuple(per_sample_loss.shape)} (one bool per "
                            f"row)."
                        )
                    p_leak_idk = torch.softmax(logits.detach(), dim=-1)[:, 1]
                    eps_idk = self.idk_focal_label_smoothing
                    if eps_idk > 0.0:
                        # Symmetric clip: bound the gate in [eps, 1-eps].
                        # Caps the per-row dynamic weight ratio at
                        # ((1-eps)/eps)**gamma so a single mis-confident
                        # IDK row can't dominate the batch.
                        p_leak_idk = p_leak_idk.clamp(min=eps_idk, max=1.0 - eps_idk)
                    else:
                        # Strictly-positive floor so 0**gamma is
                        # well-defined for gamma >= 0. Tiny floor (1e-6)
                        # is below any realistic effective probability.
                        p_leak_idk = p_leak_idk.clamp(min=1e-6)
                    idk_weight_dyn = p_leak_idk.pow(self.idk_focal_gamma)
                    lw = torch.where(idk_mask, lw * idk_weight_dyn, lw)

                # Per-row weighted average: multiply
                # then normalize by sum-of-weights so the resulting loss
                # scale doesn't track the absolute weight magnitude
                # (only the relative ratios across rows matter).
                denom = lw.sum()
                # Invariant: ``lw >= 0`` row-wise. ``loss_weight`` from
                # the dataset is materialized by
                # ``compute_loss_weight_column`` from
                # ``parse_per_source_loss_weights``, which validates
                # ``weight >= 0``. The focal multiplier
                # ``(1-p).clamp(min)`` is also non-negative. So
                # ``denom == 0`` iff every entry is exactly 0; we never
                # see a cancellation case. NaN propagates cleanly via
                # the else branch (``nan <= 0`` is False).
                if denom <= 0:
                    # All-zero weights would divide by zero. Return the
                    # weighted sum (which is 0 because all weights are
                    # 0) — keeps the autograd graph attached so
                    # ``loss.backward()`` doesn't error on the rare
                    # all-zero batch. The gradient on every parameter
                    # is identically 0, so the training step is a
                    # no-op as intended.
                    loss = (per_sample_loss * lw).sum()
                else:
                    loss = (per_sample_loss * lw).sum() / denom

            # Pairwise hinge gap loss between positive and IDK-negative
            # logit-space leak scores. Optional, ``gap_loss_weight=0``
            # disables (the no-op default). When active, adds a margin-
            # maximization term that explicitly drives
            # ``min(score|pos) > max(score|idk) + margin`` — the
            # operating-point quantity the classifier is selected on.
            # The free function returns a 0-d zero (graph-attached to
            # ``logits``) when the batch lacks positives or IDKs, so
            # the stratified sampler is a sample-efficiency optimization
            # rather than a correctness prerequisite.
            #
            # Indentation invariant: this block sits OUTSIDE both CE
            # reduction branches (the cheap-mean ``if`` and the
            # weighted-mean ``else``) because both branches assign to
            # ``loss`` and fall through to here. Do NOT move this
            # inside either branch without re-checking both code
            # paths — a future refactor that adds an early-return
            # inside one CE branch would silently drop the gap term
            # from that path.
            if self.gap_loss_weight > 0.0:
                if is_idk_negative is None:
                    raise ValueError(
                        "gap_loss_weight > 0 but is_idk_negative was "
                        "not passed to forward(). The training script "
                        "must materialize the is_idk_negative column "
                        "on the tokenized dataset; check "
                        "scripts/train_classifier.py for the "
                        "_attach_is_idk_negative closure."
                    )
                gap_idk_mask = is_idk_negative.to(
                    device=logits.device, dtype=torch.bool,
                )
                if gap_idk_mask.shape != labels.shape:
                    raise ValueError(
                        f"is_idk_negative shape {tuple(gap_idk_mask.shape)} "
                        f"must match labels shape {tuple(labels.shape)} "
                        f"(one bool per row)."
                    )
                gap_loss = compute_gap_loss(
                    logits=logits,
                    labels=labels.long(),
                    is_idk_negative=gap_idk_mask,
                    margin=self.gap_margin,
                )
                loss = loss + self.gap_loss_weight * gap_loss

        # Stash the most recent raw (unweighted) gap-loss value on the
        # model instance so a callback can read it for log/WandB output
        # without polluting the forward output dict. Returning
        # ``gap_loss`` in the dict would break HF Trainer's
        # ``prediction_step``: it builds ``logits = tuple(v for k, v in
        # outputs.items() if k not in ignore_keys + ["loss"])``, so an
        # extra dict key turns ``eval_pred.predictions`` into a tuple
        # the existing ``compute_*_metrics`` callables don't unpack.
        # Stored as a detached 0-d tensor (NOT ``.item()``) — calling
        # ``.item()`` here would force a CUDA host-device sync every
        # step. Consumers call ``.item()`` on their own logging
        # cadence. ``None`` indicates the gate is disabled or no
        # labels were passed in this call.
        #
        # Scope caveat: this stash is per-call clobber, not an
        # accumulator. Under ``gradient_accumulation_steps > 1``,
        # a single optimizer step represents N forwards but
        # ``_last_gap_loss`` reflects only the most recent
        # micro-batch's value. Under DDP, the stash is rank-local —
        # no cross-rank mean. Both are accepted: callers needing
        # exact mean-over-step or cross-rank values should
        # accumulate in their callback rather than read this
        # stash blindly.
        self._last_gap_loss = (
            gap_loss.detach() if gap_loss is not None else None
        )

        return {"loss": loss, "logits": logits}
