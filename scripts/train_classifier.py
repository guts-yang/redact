"""Train the contextual binary leak classifier.

Encoder + single binary head, with prompt + k-window prior-sentence
context per input. This is the canonical leak detector: it decides,
per CoT sentence, whether the sentence leaks — and is paired at
inference with a single shared soft token.

Training data: the audited augmented dataset, using the binary
``label`` column (0 = clean, 1 = leak).

Usage:
    python -m scripts.train_classifier \
        --split forget10/augmented --model_type contextual_binary
"""
import argparse
import functools
from collections import Counter

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk

from redact.leak_classifier_metrics import (
    LeakMetricsAggregator,
    binary_label_extractor,
    binary_p_leak_extractor,
    compute_binary_metrics,
)
from redact.training import (
    SplitAwareTrainer,
    WeightedSplitAwareTrainer,
    compute_loss_weight_column,
    compute_sample_weights,
    parse_per_source_loss_weights,
)
from redact.utils import enable_full_determinism, log_print


parser = argparse.ArgumentParser()
parser.add_argument(
    "--split", type=str, required=True,
    help="Dataset sub-directory name under --data_dir.",
)
parser.add_argument("--model_type", type=str, default="contextual_binary",
                    choices=["contextual_binary"],
                    help="contextual_binary: encoder + single binary head, "
                         "with prompt + k-window prior-sentence context per "
                         "input; the canonical leak detector, paired with a "
                         "single shared SoftToken at inference. Kept as an "
                         "explicit single-choice flag so existing commands "
                         "keep working; it also names the checkpoint "
                         "directory suffix.")
parser.add_argument("--data_dir", type=str, default="leak_detector_data")
parser.add_argument("--model_name", type=str, default="roberta-base")
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--max_grad_norm", type=float, default=0.0,
                    help="Gradient clipping max norm. 0.0 disables clipping "
                         "(current default; matches Mosbach et al. recipe). "
                         "Set to 1.0 to cap extreme grad spikes (~10× over "
                         "baseline) that can destabilize min_p_pos late in "
                         "training. Passed through to HF TrainingArguments.")
parser.add_argument(
    "--lr_scheduler_type",
    type=str,
    default="cosine",
    choices=[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
        "inverse_sqrt",
        "reduce_lr_on_plateau",
        "cosine_with_min_lr",
        "cosine_warmup_with_min_lr",
        "warmup_stable_decay",
        "greedy",
    ],
    help="HuggingFace TrainingArguments.lr_scheduler_type. Default "
         "'cosine' preserves prior behavior. Choices mirror the HF "
         "SchedulerType enum; bad values fail at argparse rather than "
         "after data load. Non-default values are tagged in save_dir "
         "as '_lrs_<type>' so runs with different schedulers don't "
         "overwrite each other.",
)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--num_train_epochs", type=int, default=5)
parser.add_argument(
    "--max_steps", type=int, default=-1,
    help="If > 0, overrides --num_train_epochs and trains for this many "
         "optimizer steps total (HF TrainingArguments.max_steps semantics). "
         "Default -1 = epoch-driven (use --num_train_epochs). Tagged in "
         "save_dir as '_ms<N>' when active so step- and epoch-based runs "
         "don't overwrite each other. NOTE: HF Trainer ignores "
         "--num_train_epochs entirely when max_steps > 0; the LR scheduler "
         "anneals over max_steps rather than the dataset-derived epoch "
         "count, so a step-based run with the same data is NOT equivalent "
         "to a fixed-epoch run with the same total step count.",
)
parser.add_argument("--eval_steps", type=int, default=500)
parser.add_argument("--save_steps", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument(
    "--positive_focal_gamma",
    type=float,
    default=0.0,
    help="Focal exponent for hard-positive mining on contextual_binary. "
         "When > 0, label=1 rows receive an in-graph dynamic loss weight "
         "((1 - P(leak))**gamma) computed from the detached binary "
         "logits, composed multiplicatively with the static "
         "--loss_weights_per_source row weight (if any) and the "
         "balanced-class-weight from CrossEntropy. Default 0.0 is a "
         "no-op (recovers today's behavior bit-for-bit). The "
         "Lin et al. focal-loss default of 2.0 collapses easy-positive "
         "contribution by ~(1 - 0.95)**2 = 0.0025, leaving stubborn "
         "low-P(leak) positives carrying most of the positive gradient "
         "mass. Recommended initial sweep: {0.5, 1.0, 2.0}.",
)
parser.add_argument(
    "--positive_focal_label_smoothing",
    type=float,
    default=0.0,
    help="Smoothing eps for the hard-positive focal gate: clips "
         "(1 - P(leak)) to [eps, 1-eps] before applying the gamma "
         "power, capping the per-row dynamic weight ratio at "
         "((1-eps)/eps)**gamma. Default 0.0 disables clipping. "
         "Recommended eps=0.05 when --positive_focal_gamma >= 1 to "
         "stop a single mis-confident logit early in training from "
         "blowing up its row's gradient (e.g. at gamma=2, eps=0.05 "
         "caps the per-row positive weight at 19**2 = 361). Only "
         "consulted when --positive_focal_gamma > 0; must lie in "
         "[0, 0.5).",
)
parser.add_argument(
    "--idk_focal_gamma",
    type=float,
    default=0.0,
    help="Focal exponent for hard-IDK-negative mining on "
         "contextual_binary — the symmetric counterpart to "
         "--positive_focal_gamma. When > 0, rows where "
         "label==0 AND source == 'idk' receive an "
         "in-graph dynamic loss weight (P(leak)**gamma) computed "
         "from the detached binary logits, composed multiplicatively "
         "with the static --loss_weights_per_source row weight (if "
         "any), the class-balance weight from CrossEntropy, and the "
         "positive-focal weight (if active). Hard IDK negatives have "
         "HIGH P(leak) (the classifier is mistakenly flagging them as "
         "leaks); the focal weight concentrates gradient on those "
         "rows, attacking the cascading-second-regen failure mode at "
         "inference. Default 0.0 is a no-op. Recommended initial "
         "sweep: {0.5, 1.0, 2.0}. The IDK pool is small "
         "(~3-4%% of forget01 eligible negatives) so this is "
         "higher-variance than --positive_focal_gamma — pair with "
         "--idk_focal_label_smoothing 0.05+ at gamma >= 1.",
)
parser.add_argument(
    "--idk_focal_label_smoothing",
    type=float,
    default=0.0,
    help="Smoothing eps for the hard-IDK-negative focal gate: clips "
         "P(leak) to [eps, 1-eps] before applying the gamma power, "
         "capping the per-row dynamic weight ratio at "
         "((1-eps)/eps)**gamma. Default 0.0 disables clipping. "
         "Recommended eps=0.05 when --idk_focal_gamma >= 1 to stop a "
         "single mis-confident IDK row from blowing up its row's "
         "gradient (e.g. at gamma=2, eps=0.05 caps the per-row IDK "
         "weight at 19**2 = 361). Only consulted when "
         "--idk_focal_gamma > 0; must lie in [0, 0.5).",
)
parser.add_argument(
    "--gap_loss_weight",
    type=float,
    default=0.0,
    help="Mixing weight for the pairwise hinge 'gap loss' between "
         "positive and IDK-negative logit-space leak scores. Adds "
         "``gap_loss_weight * mean_{(p,i)} max(0, margin - (score_p - "
         "score_i))`` to the binary CE loss, where ``score = "
         "logits[:, 1] - logits[:, 0]`` is the logit-space leak score "
         "and the mean is over all (positive, IDK-negative) pairs in "
         "the batch. The gap loss directly maximizes the deployment "
         "metric ``min(score|pos) - max(score|idk)`` in the same logit "
         "space the classifier operates in at inference. Default 0.0 "
         "is a no-op (existing training is preserved bit-for-bit). "
         "Recommended initial sweep: {0.1, 0.5, 1.0}. The hinge mean "
         "is over ALL pairs so the gradient signal is dense even when "
         "either population is tiny in the batch; a batch with zero "
         "positives or zero IDKs degenerates to a 0-d zero (graceful "
         "no-op).",
)
parser.add_argument(
    "--gap_margin",
    type=float,
    default=4.0,
    help="Logit-space margin for the pairwise hinge gap loss. "
         "Penalty is ``max(0, margin - (score_pos - score_idk))`` per "
         "(pos, idk) pair, averaged over all pairs. Default 4.0 is "
         "sized for typical RoBERTa logit scales (the logit-space "
         "gap between confident classes runs ~3-6 at convergence on "
         "well-separated tasks). Larger margins push harder on the "
         "separation but cap the achievable hinge gain per pair. Only "
         "consulted when --gap_loss_weight > 0; must be >= 0.",
)
parser.add_argument(
    "--metric_for_best_model",
    type=str,
    default="test_pauc_retain_aux",
    help="Metric on which to select best checkpoint. Default "
         "'test_pauc_retain_aux' is partial AUROC at FPR<=--pauc_fpr_cap "
         "with negatives restricted to retain/aux/IDK sources — directly "
         "tracks recall on forget-set positives vs. FPR on the "
         "eligible-negative pool (clean_retain*, clean_aux*, clean_idk*), "
         "the operating-point quantities that drive deployment behavior. "
         "Two opt-in alternatives flip the recall/FPR asymmetry to "
         "prioritize catching leaks over avoiding false positives: "
         "'test_pauc_above_recall_retain_aux' is the symmetric "
         "TPR-restricted partial AUROC over TPR>=--pauc_recall_lower "
         "(higher is better); 'test_fpr_at_recall_retain_aux' is the "
         "FPR at the threshold first achieving TPR>=--fpr_at_recall_"
         "target (lower is better; the script derives "
         "TrainingArguments.greater_is_better automatically from the "
         "chosen metric). All "
         "three options compute on the merged 'test' split and require "
         "a 'source' column. NOTE: the "
         "default assumes an active eval schedule + load_best_model_at_end "
         "(both currently hard-coded as eval_strategy='steps' / True). If "
         "either becomes CLI-controllable in the future and a caller "
         "disables eval, this default must move to None or HF Trainer "
         "will raise on construction.",
)
parser.add_argument(
    "--pauc_fpr_cap", type=float, default=0.10,
    help="Upper FPR bound for the partial AUROC integration on the test "
         "split's pauc_retain_aux metric. Wider cap (e.g. 0.10) is less "
         "noisy on small eligible-negative eval subsets; narrower cap "
         "(e.g. 0.05) is closer to the canonical inference operating "
         "point.",
)
parser.add_argument(
    "--pauc_recall_lower", type=float, default=0.95,
    help="Lower TPR bound for the test split's "
         "pauc_above_recall_retain_aux metric — partial AUROC integrated "
         "over the high-recall strip TPR ∈ [pauc_recall_lower, 1]. Default "
         "0.95 reflects the project preference that missing a leak is "
         "more costly than a false positive. A narrower (higher) lower "
         "bound is closer to the deployment operating point but noisier "
         "on small eligible-negative eval subsets; a wider (lower) "
         "lower bound is more stable for checkpoint selection. Must lie "
         "in [0, 1); 0 recovers full AUROC on the filtered subset.",
)
parser.add_argument(
    "--fpr_at_recall_target", type=float, default=0.95,
    help="Recall floor for the test split's fpr_at_recall_retain_aux "
         "metric — FPR at the threshold first achieving TPR>=target. "
         "Single operating-point view; lower is better. Companion "
         "diagnostic to --pauc_recall_lower (the two answer "
         "complementary questions: pAUC integrates over the strip, "
         "fpr_at_recall is the single-point FPR at the floor). Default "
         "0.95. Must lie in (0, 1].",
)
parser.add_argument(
    "--pauc_source_balance", action="store_true",
    help="If set, the three retain/aux/IDK pAUC metrics on the test "
         "split are computed as macro-averages across the three source "
         "buckets (clean_retain*, clean_aux*, clean_idk*) instead of "
         "the default pooled FPR. Each source contributes equally to "
         "the headline value regardless of row count, so checkpoint "
         "selection no longer favors the most-represented source (aux "
         "dominates forget01 at 83%% of the eligible-negative pool; "
         "retain dominates forget10 at 63%%; IDK is always <4%%). "
         "Per-source pAUC components are always logged as a diagnostic "
         "alongside the headline value regardless of this flag — "
         "useful for spotting variance amplification on the tiny IDK "
         "pool, which inherits sqrt(p(1-p)/n_idk) noise (~3-4 pp on "
         "forget01) that propagates into the macro-average. "
         "WARNING: when set, the headline pauc_* / fpr_at_recall_* "
         "values are NOT comparable to historical baselines computed "
         "with the pooled definition.",
)
parser.add_argument("--max_prior_sentences", type=int, default=2,
                    help="Cap on prior sentences per input. Persisted in "
                         "BinaryHeadConfig.max_prior_sentences so inference "
                         "matches the train-time k-window.")
parser.add_argument(
    "--prior_dropout_rate",
    type=float,
    default=0.0,
    help="Probability of dropping the priors "
         "(context window) from each train-time example. Each row's "
         "tokenization step also emits a no-priors variant (built "
         "with max_prior_sentences=0); a per-row Bernoulli(rate) draw "
         "swaps the priors-on inputs for the priors-off variant in "
         "the train collator. Default 0.0 is a no-op (existing "
         "training preserved bit-for-bit; no extra memory cost). "
         "Range: [0.0, 1.0). 1.0 is excluded — use "
         "--max_prior_sentences 0 if you want priors-off always. "
         "Designed to break the shortcut where the classifier learns "
         "is_leak ≈ (question about forget author) AND (priors look "
         "like CoT opener pattern), ignoring the focal sentence "
         "itself. Eval always uses the natural priors-on inputs "
         "(no dropout at eval time). Requires --max_prior_sentences "
         "> 0; otherwise there are no priors to drop.",
)
parser.add_argument("--target_budget", type=int, default=128,
                    help="Tokens reserved for <TGT> target </TGT> inside the "
                         "512-token input.")
parser.add_argument("--max_length", type=int, default=512,
                    help="Tokenizer max sequence length.")
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument(
    "--sample_balance",
    type=str,
    default="none",
    choices=["none", "binary"],
    help="Input-level rebalancing via WeightedRandomSampler. 'none' "
         "uses HF default RandomSampler (no change). 'binary' draws "
         "rows so P(leak)=P(clean)=0.5 in expectation. Composes with "
         "--loss_weights; see the help for that flag.",
)
parser.add_argument(
    "--sample_alpha",
    type=float,
    default=1.0,
    help="Temperature on the inverse-frequency sampling schedule "
         "(only used when --sample_balance != none). alpha=1.0 is "
         "hard balance (equal expected mass per class). alpha=0.0 "
         "recovers empirical (no rebalance). alpha=0.5 is sqrt-balance "
         "— a softer rebalance that mitigates the small-class "
         "repetition risk of alpha=1.0.",
)
parser.add_argument(
    "--use_stratified_sampler",
    action="store_true",
    help="Use a StratifiedBatchSampler that guarantees a fixed count "
         "of positives (label=1) and IDK negatives (label=0 AND "
         "source == 'idk') per batch, with the "
         "remaining slots drawn from the 'other negative' pool without "
         "replacement. Designed for the pairwise-hinge auxiliary loss, "
         "which needs both positives and IDKs co-resident in every "
         "batch — a WeightedRandomSampler only matches expected counts "
         "and can yield batches with zero positives at high imbalance. "
         "Mutually exclusive with --sample_balance != 'none' (the two "
         "are alternative input-level rebalancing mechanisms; "
         "combining them is rejected at construction). "
         "NOTE: each 'epoch' under this sampler is ceil(N_other_"
         "negatives / n_other_per_batch) steps — typically ~10x fewer "
         "steps than the weighted / default-RandomSampler path "
         "(which always returns len(dataset) steps per epoch). "
         "HF Trainer derives num_training_steps from this len(), so "
         "the LR schedule, --num_train_epochs, and --eval_steps are "
         "all interpreted relative to the new (smaller) step count. "
         "Re-tune --num_train_epochs / --eval_steps accordingly, or "
         "use --max_steps for a sampling-mode-invariant schedule.",
)
parser.add_argument(
    "--batch_pos_count",
    type=int,
    default=2,
    help="Number of positives (label=1 rows) injected per batch when "
         "--use_stratified_sampler is set. Default 2. Must satisfy "
         "batch_pos_count + batch_idk_count < --batch_size, otherwise "
         "no slots remain for 'other' negatives.",
)
parser.add_argument(
    "--batch_idk_count",
    type=int,
    default=2,
    help="Number of IDK negatives (label=0 AND source.startswith("
         "'clean_idk') rows) injected per batch when "
         "--use_stratified_sampler is set. Default 2. Must satisfy "
         "batch_pos_count + batch_idk_count < --batch_size.",
)
parser.add_argument(
    "--loss_weights",
    type=str,
    default="balanced",
    choices=["balanced", "uniform"],
    help="Loss-level rebalancing. 'balanced' (default) uses inverse-"
         "frequency class weights in cross-entropy (compute_balanced_"
         "class_weights). 'uniform' replaces them with torch.ones, so "
         "CE collapses to standard unweighted CE. Independent of "
         "--sample_balance: 'uniform' + '--sample_balance binary' is "
         "pure input-level rebalance; 'balanced' + 'none' is today's "
         "default; 'balanced' + non-none sampler double-balances "
         "multiplicatively (logged as a warning).",
)
parser.add_argument(
    "--loss_weights_per_source",
    type=str,
    default="",
    help="Per-row, per-source loss multipliers. "
         "Default '' = uniform per-row weights (today's behavior). "
         "Weights apply ONLY to label=0 rows — label=1 rows always get "
         "weight 1.0 (the positive class is the operating-point anchor "
         "and stays unmodulated; per-source weighting modulates only "
         "the negative gradient). Two non-default forms: (a) the "
         "literal token 'target_fraction_capped' auto-computes weights "
         "so each source's absolute label=0 loss mass equals N_neg/3, "
         "capped at 10x to prevent gradient explosion on rare sources. "
         "The K=4 case (retain + aux + idk + forget pre-leak) is the "
         "typical post-label-gate case. "
         "Recommended default for binary classifier training. (b) a "
         "comma-separated NAME=VALUE spec like 'idk=10.0,aux=0.4,"
         "retain=2.07' assigns explicit weights to "
         "listed sources; unlisted sources default to 1.0. Source-name "
         "matching is EXACT (no wildcard / prefix matching). Positives "
         "(label=1, regardless of source) under any mode keep weight "
         "1.0 at application time — the gate is on the row's label, "
         "not on the source name. Pre-leak background rows of forget "
         "CoTs (source='forget' AND label=0) thus participate in "
         "the negative pool just like retain/aux/idk rows, providing "
         "within-forget-prompt content-discrimination signal that the "
         "prior source-prefix gate was missing. NOTE: composes "
         "multiplicatively with the class weights from --loss_weights "
         "and the input-level "
         "sampler from --sample_balance. The class-vs-source axes are "
         "complementary — class weights handle positive/negative "
         "imbalance (forget01 has 0.72%% positives at label=1, so "
         "--loss_weights balanced gives positives a ~137x boost which "
         "is essential), while per-source weights handle within-"
         "negative imbalance. Recommended: KEEP --loss_weights balanced "
         "(the default) when using --loss_weights_per_source — pairing "
         "with --loss_weights uniform catastrophically under-weights "
         "the positive class at training time. With --sample_balance "
         "set, the sampler reshapes the per-batch source distribution "
         "before per-row weights apply — the 'each negative source "
         "contributes N_neg/3 mass' property is then NOT preserved.",
)
parser.add_argument(
    "--loss_weights_per_source_cap",
    type=float,
    default=10.0,
    help="Maximum per-source weight under 'target_fraction_capped' mode. "
         "Default 10.0. Lower this when the raw uncapped weight for a "
         "rare source is only modestly above the cap — at low IDK pool "
         "sizes the cap doesn't bind much, and lifting the IDK gradient "
         "to ~1/3 of negative mass can over-correct by stealing margin "
         "from retain (observed on forget05/10 v23 ceiling collapse). "
         "Try 3-5 when the IDK pool is large enough that raw target-"
         "fraction is already in single digits.",
)
parser.add_argument("--output_dir", type=str, default="leak_classifiers")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--smoothing_window",
    type=int,
    default=1,
    help="Rolling-window size for the smoothed companion metrics "
         "emitted alongside each pooled / per-source / balanced "
         "headline. When >= 2, the wrapper emits a parallel "
         "``<metric>_smoothed`` key whose value is the np.nanmean "
         "over the last N eval_test passes. Useful for noisier "
         "metrics (gap_idk, max_p_idk) and for checkpoint selection "
         "via --metric_for_best_model test_<metric>_smoothed — "
         "picking the highest smoothed gap is more robust than "
         "picking the highest single-eval spike. Default 1 "
         "disables smoothing (no _smoothed keys emitted). Window "
         "unit is eval-calls; changing --eval_steps changes the "
         "effective step-window.",
)
parser.add_argument(
    "--strict_determinism",
    action="store_true",
    help="When set, non-deterministic ops raise a RuntimeError at call "
         "time instead of warning and falling back to the non-"
         "deterministic implementation. Default is warn-mode: lets "
         "training proceed past obscure ops without a deterministic "
         "implementation (more forgiving for long-running GPU jobs). "
         "Set this flag if you want loud failures on any "
         "non-determinism source — useful for debugging "
         "irreproducibility but risks aborting a multi-hour run on a "
         "rarely-hit eval-time op.",
)
args = parser.parse_args()

# Cheap CLI sanity checks. Performed before the data-load peek at line
# 315 so a bad flag fails fast instead of after a multi-second
# load_from_disk + (in production) a multi-minute tokenize.
if args.positive_focal_gamma < 0:
    parser.error(
        f"--positive_focal_gamma must be >= 0; got "
        f"{args.positive_focal_gamma}. (0.0 disables the focal gate.)"
    )
if not (0.0 <= args.positive_focal_label_smoothing < 0.5):
    parser.error(
        f"--positive_focal_label_smoothing must lie in [0, 0.5); got "
        f"{args.positive_focal_label_smoothing}."
    )
if args.positive_focal_label_smoothing > 0 and args.positive_focal_gamma == 0:
    parser.error(
        "--positive_focal_label_smoothing is only consulted when "
        "--positive_focal_gamma > 0; passing label smoothing without "
        "gamma is almost certainly a mistake."
    )
if args.idk_focal_gamma < 0:
    parser.error(
        f"--idk_focal_gamma must be >= 0; got {args.idk_focal_gamma}. "
        f"(0.0 disables the IDK focal gate.)"
    )
if not (0.0 <= args.idk_focal_label_smoothing < 0.5):
    parser.error(
        f"--idk_focal_label_smoothing must lie in [0, 0.5); got "
        f"{args.idk_focal_label_smoothing}."
    )
if args.idk_focal_label_smoothing > 0 and args.idk_focal_gamma == 0:
    parser.error(
        "--idk_focal_label_smoothing is only consulted when "
        "--idk_focal_gamma > 0; passing label smoothing without "
        "gamma is almost certainly a mistake."
    )
if args.gap_loss_weight < 0:
    parser.error(
        f"--gap_loss_weight must be >= 0; got {args.gap_loss_weight}. "
        f"(0.0 disables the gap-loss term.)"
    )
if args.gap_margin < 0:
    parser.error(
        f"--gap_margin must be >= 0; got {args.gap_margin}."
    )
if args.use_stratified_sampler and args.sample_balance != "none":
    parser.error(
        f"--use_stratified_sampler and --sample_balance="
        f"{args.sample_balance!r} are mutually exclusive input-level "
        f"rebalancing mechanisms. Pass --sample_balance=none with "
        f"--use_stratified_sampler, or drop --use_stratified_sampler."
    )
if args.batch_pos_count < 0 or args.batch_idk_count < 0:
    parser.error(
        f"--batch_pos_count and --batch_idk_count must be >= 0; got "
        f"--batch_pos_count={args.batch_pos_count}, "
        f"--batch_idk_count={args.batch_idk_count}."
    )
if args.use_stratified_sampler and (
    args.batch_pos_count + args.batch_idk_count >= args.batch_size
):
    parser.error(
        f"--batch_pos_count ({args.batch_pos_count}) + --batch_idk_count "
        f"({args.batch_idk_count}) = "
        f"{args.batch_pos_count + args.batch_idk_count} must be strictly "
        f"less than --batch_size ({args.batch_size}); otherwise no slots "
        f"remain for 'other' negatives in the stratified batch sampler."
    )
if not (0.0 <= args.prior_dropout_rate < 1.0):
    parser.error(
        f"--prior_dropout_rate must lie in [0.0, 1.0); got "
        f"{args.prior_dropout_rate}. (1.0 is excluded — use "
        f"--max_prior_sentences 0 if you want priors-off always.)"
    )
if args.prior_dropout_rate > 0 and args.max_prior_sentences == 0:
    parser.error(
        f"--prior_dropout_rate={args.prior_dropout_rate} requires "
        f"--max_prior_sentences > 0; with max_prior_sentences=0 there "
        f"are no priors to drop. Either drop --prior_dropout_rate or "
        f"bump --max_prior_sentences above 0."
    )
def _strip_smoothed_suffix(metric_name: str) -> str:
    """``test_gap_idk_smoothed`` → ``test_gap_idk``. Used so the
    selection-metric classification tuples (``_RETAIN_AUX_METRICS``,
    ``_LOWER_IS_BETTER_METRICS``) and the diagnostic-rejection
    checks below don't need to enumerate every ``_smoothed`` variant
    — they self-maintain as new metrics are added. The smoothed
    companion shares its base metric's direction (higher gap is
    better, smoothed gap is also better), source-column requirement
    (smoothed gap still requires source for the per-row IDK filter),
    and diagnostic-vs-selection status (smoothed effective_idk_mass
    is still a diagnostic, not a selection target)."""
    suffix = "_smoothed"
    return (
        metric_name[: -len(suffix)]
        if metric_name.endswith(suffix)
        else metric_name
    )


# Strip once and reuse for all downstream metric-name classification.
# The smoothed companions (e.g. test_effective_idk_mass_smoothed)
# inherit their base metric's rules — without stripping, the
# diagnostic-rejection checks below would silently let
# ``--metric_for_best_model=test_effective_idk_mass_smoothed``
# through.
_metric_for_best_model_base = _strip_smoothed_suffix(args.metric_for_best_model)

# effective_positive_mass / effective_idk_mass are diagnostics, not
# selection targets — they are only emitted when the corresponding
# focal gate is active, so a best-model metric referencing one would
# silently break a gamma=0 run. The check applies to both the raw
# diagnostic key and its smoothed companion (canonicalized via
# _strip_smoothed_suffix above).
if _metric_for_best_model_base == "test_effective_positive_mass":
    parser.error(
        f"--metric_for_best_model={args.metric_for_best_model!r} is "
        "rejected: effective_positive_mass is a diagnostic (logged "
        "only when --positive_focal_gamma > 0), and neither 'max' "
        "nor 'min' over it is a meaningful selection target. Use "
        "'test_pauc_retain_aux' (the default) and read "
        "effective_positive_mass from the eval log to monitor "
        "whether the focal gate is starving positives."
    )
if _metric_for_best_model_base == "test_effective_idk_mass":
    parser.error(
        f"--metric_for_best_model={args.metric_for_best_model!r} is "
        "rejected: effective_idk_mass is a diagnostic (logged only "
        "when --idk_focal_gamma > 0), and neither 'max' nor 'min' "
        "over it is a meaningful selection target. Use "
        "'test_gap_idk' or 'test_pauc_retain_aux' and read "
        "effective_idk_mass from the eval log to monitor whether "
        "the IDK focal gate is starving the IDK pool."
    )
if args.smoothing_window < 1:
    parser.error(
        f"--smoothing_window must be >= 1; got "
        f"{args.smoothing_window}. (1 disables smoothing; >=2 emits "
        f"<metric>_smoothed keys.)"
    )
if (
    args.metric_for_best_model.endswith("_smoothed")
    and args.smoothing_window < 2
):
    parser.error(
        f"--metric_for_best_model={args.metric_for_best_model!r} "
        f"ends in _smoothed, which is only emitted when "
        f"--smoothing_window >= 2. Either pass --smoothing_window N "
        f"(N >= 2) or pick the non-smoothed variant (drop the "
        f"_smoothed suffix)."
    )

enable_full_determinism(args.seed, warn_only=not args.strict_determinism)
log_print(
    f"Full determinism enabled: seed={args.seed}, "
    f"CUBLAS_WORKSPACE_CONFIG=:16:8, FLASH_ATTENTION_DETERMINISTIC=1, "
    f"cudnn.deterministic=True, cudnn.benchmark=False, "
    f"torch.use_deterministic_algorithms=True "
    f"(strict_mode={args.strict_determinism}). Two runs with the "
    f"same seed should now produce bit-identical weights and "
    f"metrics. Small to moderate slowdown depending on op mix. "
    f"If a non-deterministic op trips under "
    f"--strict_determinism, drop the flag to fall back to warn-mode."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------

data_path = f"{args.data_dir}/{args.split}"
log_print(f"Loading training data from {data_path}")
dataset = load_from_disk(data_path)

# Derive forget/retain eval splits from source column.
#
# Under the new (source, source_tags) schema, every forget-bucket row
# (clean prose, reword_s0 augment, answer-source positive, corrupted_*)
# carries source="forget"; retain/aux/idk rows carry their respective
# primaries. Tag-distinguished variants of the forget bucket all stay
# on the forget side of the eval split.
def _is_forget_source(s: str) -> bool:
    return s == "forget"


if (
    isinstance(dataset, DatasetDict)
    and "test" in dataset
    and "forget" not in dataset
    and "source" in dataset["test"].column_names
    and "source_tags" in dataset["test"].column_names
):
    test_ds = dataset["test"]
    forget_ds = test_ds.filter(lambda r: _is_forget_source(r["source"]))
    retain_ds = test_ds.filter(lambda r: not _is_forget_source(r["source"]))
    dataset = DatasetDict({
        **dataset,
        "forget": forget_ds,
        "retain": retain_ds,
    })
    log_print(
        f"Derived forget/retain eval splits: "
        f"forget={len(forget_ds)}, retain={len(retain_ds)}"
    )

# Capture test-split 'source' column for the partial-AUROC metric.
# pauc_retain_aux is computed only on the merged 'test' split (the
# derived 'forget'/'retain' splits each contain rows from only one side
# of the positive/negative distribution; pAUC needs both). The
# SplitAwareTrainer-based dispatch in the wrapped compute_metrics
# functions reads the current eval prefix to decide whether to compute
# pauc — see redact.training.SplitAwareTrainer.
#
# Sources are captured BEFORE tokenization so the 'source' column
# is still present (the tokenize-then-rename pipeline drops non-input
# columns from the tokenized DatasetDict).
# All retain/aux-restricted metrics require a 'source' column to filter
# the eval split. Listed centrally so adding another retain/aux metric
# only needs one edit. The ``_balanced`` suffix denotes the source-
# balanced macro-average headline; it requires the same 'source' column
# and is produced alongside the pooled headline when
# --pauc_source_balance is set.
_RETAIN_AUX_METRICS = (
    "test_pauc_retain_aux",
    "test_pauc_above_recall_retain_aux",
    "test_fpr_at_recall_retain_aux",
    "test_pauc_retain_aux_balanced",
    "test_pauc_above_recall_retain_aux_balanced",
    "test_fpr_at_recall_retain_aux_balanced",
    # The IDK-pool trio uses source == "idk" to pick its negative
    # pool; not retain_aux-family (single-source by definition, no
    # per-source variants) but still requires the 'source' column on
    # test for the same reason.
    "test_gap_idk",
    "test_min_p_pos",
    "test_max_p_idk",
    # Robust quantile-based sibling of the gap_idk trio. Same source
    # column requirement (IDK row filter), same direction (higher
    # gap is better). 5th / 95th percentile reduction so a single
    # outlier on either bucket can't dominate.
    "test_gap_idk_robust",
    "test_min_p_pos_q05",
    "test_max_p_idk_q95",
)
# Metrics where lower is better. Used to derive Trainer's
# ``greater_is_better`` flag from the chosen ``metric_for_best_model``.
# Keep in sync with new metric additions above — but only list metrics
# where lower IS better. ``test_gap_idk`` and ``test_min_p_pos`` are
# intentionally absent (higher gap / more-confident worst positive
# are better, which is the default direction). ``test_max_p_idk`` is
# present — the metric is the most-confident IDK false positive, so
# lower means less inference-time cascade risk.
_LOWER_IS_BETTER_METRICS = (
    "test_fpr_at_recall_retain_aux",
    "test_fpr_at_recall_retain_aux_balanced",
    "test_max_p_idk",
    # Robust sibling: 95th-percentile P(leak) on IDK negatives —
    # same direction as test_max_p_idk (lower = less cascade risk).
    "test_max_p_idk_q95",
)
# ``_strip_smoothed_suffix`` and ``_metric_for_best_model_base`` were
# defined above (before the diagnostic-rejection block) so the
# rejection checks see the stripped name. Reused here for the
# greater-is-better / RETAIN_AUX / _balanced classification.
_greater_is_better = _metric_for_best_model_base not in _LOWER_IS_BETTER_METRICS
_test_split = dataset["test"] if isinstance(dataset, DatasetDict) else dataset
if "source" in _test_split.column_names:
    test_sources: list[str] | None = list(_test_split["source"])
elif _metric_for_best_model_base in _RETAIN_AUX_METRICS:
    raise ValueError(
        f"--metric_for_best_model={args.metric_for_best_model!r} requires a "
        f"'source' column on dataset['test'], but it is missing. Either "
        f"re-aggregate the training data with the contextual labels schema "
        f"(which writes a 'source' field per row), or pass "
        f"--metric_for_best_model test_binary_f1 to select on the "
        f"F1-based metric instead."
    )
else:
    test_sources = None

if args.pauc_source_balance:
    if test_sources is None:
        raise ValueError(
            "--pauc_source_balance requires a 'source' column on "
            "dataset['test'] (the macro-average across source buckets is "
            "computed from this column). The current --model_type branch "
            "didn't capture test_sources; either re-aggregate the "
            "training data with the contextual labels schema, or drop "
            "--pauc_source_balance."
        )
    log_print(
        "NOTE: --pauc_source_balance active. Pooled headlines "
        "(pauc_retain_aux, pauc_above_recall_retain_aux, "
        "fpr_at_recall_retain_aux) are still logged for continuity; "
        "the source-balanced macro-averages are added under "
        "*_balanced suffixes and are NOT comparable to historical "
        "pooled baselines. Per-source components are logged alongside "
        "as *_retain / *_aux / *_idk."
    )
elif _metric_for_best_model_base.endswith("_balanced"):
    # Stripped form is used so test_pauc_retain_aux_balanced_smoothed
    # also trips this check (the underlying _balanced key still
    # requires --pauc_source_balance regardless of smoothing).
    raise ValueError(
        f"--metric_for_best_model={args.metric_for_best_model!r} ends in "
        f"_balanced (after stripping any _smoothed suffix), which is "
        f"only computed when --pauc_source_balance is also set. "
        f"Either pass --pauc_source_balance or pick a pooled metric "
        f"(drop the _balanced suffix)."
    )

# Log class distribution
dist = Counter(dataset["train"]["label"])
log_print("Training class distribution:")
for lbl in sorted(dist):
    name = "leak" if lbl == 1 else "clean"
    log_print(f"  label={lbl} ({name}): {dist[lbl]}")


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------

save_dir = f"{args.output_dir}/{args.split}_{args.model_type}"

if args.sample_balance != "none":
    save_dir = f"{save_dir}_sampler_{args.sample_balance}_a{args.sample_alpha:g}"
if args.use_stratified_sampler:
    # Compact, greppable suffix: 'strat<p>i<i>' captures the per-batch
    # positives + IDK counts (e.g. _strat2i2 for the default 2+2).
    save_dir = (
        f"{save_dir}_strat{args.batch_pos_count}i{args.batch_idk_count}"
    )
if args.loss_weights != "balanced":
    save_dir = f"{save_dir}_lw_{args.loss_weights}"
if args.loss_weights_per_source:
    # Compact tag: 'tfc' for the auto-computed default, otherwise a
    # short hash-free slug from the spec so directory paths stay
    # human-greppable. Replace non-alphanumeric chars to keep the
    # path filesystem-safe.
    spec = args.loss_weights_per_source.strip()
    if spec == "target_fraction_capped":
        slug = "tfc"
    else:
        slug = "".join(c if c.isalnum() else "_" for c in spec)
        # Cap length so very long specs don't blow out path limits.
        if len(slug) > 64:
            slug = slug[:64]
    save_dir = f"{save_dir}_lwps_{slug}"
if args.positive_focal_gamma > 0:
    # Compact, greppable suffix. ``g`` strips trailing zeros for clean
    # paths (e.g. _pfg2 instead of _pfg2.0). Label smoothing only
    # appears in the suffix when active so default runs stay short.
    suffix = f"_pfg{args.positive_focal_gamma:g}"
    if args.positive_focal_label_smoothing > 0:
        suffix += f"_ls{args.positive_focal_label_smoothing:g}"
    save_dir = f"{save_dir}{suffix}"
if args.idk_focal_gamma > 0:
    suffix = f"_ifg{args.idk_focal_gamma:g}"
    if args.idk_focal_label_smoothing > 0:
        suffix += f"_ils{args.idk_focal_label_smoothing:g}"
    save_dir = f"{save_dir}{suffix}"
if args.gap_loss_weight > 0:
    # Compact greppable suffix mirroring the focal patterns above:
    # ``glw`` = gap loss weight, ``gm`` = gap margin. Margin only
    # appended when it deviates from the 4.0 default to keep paths
    # short for the canonical setting.
    suffix = f"_glw{args.gap_loss_weight:g}"
    if args.gap_margin != 4.0:
        suffix += f"_gm{args.gap_margin:g}"
    save_dir = f"{save_dir}{suffix}"
if args.prior_dropout_rate > 0:
    # Compact, greppable suffix. Two-decimal precision matches the
    # rate's CLI granularity (e.g. _pdr0.30 for rate=0.3) and avoids
    # the trailing-zero suppression that the focal-family tags use —
    # the rate is a probability with natural two-decimal precision.
    save_dir = f"{save_dir}_pdr{args.prior_dropout_rate:.2f}"
if args.lr_scheduler_type != "cosine":
    save_dir = f"{save_dir}_lrs_{args.lr_scheduler_type}"
if args.max_steps > 0:
    save_dir = f"{save_dir}_ms{args.max_steps}"
    log_print(
        f"--max_steps={args.max_steps} active; HF Trainer will ignore "
        f"--num_train_epochs={args.num_train_epochs} and anneal the LR "
        f"scheduler over {args.max_steps} steps."
    )
if args.max_grad_norm != 0.0:
    save_dir = f"{save_dir}_gn{args.max_grad_norm:g}"
if args.sample_balance != "none" and args.loss_weights == "balanced":
    log_print(
        f"NOTE: --sample_balance={args.sample_balance!r} composes "
        f"multiplicatively with the default 'balanced' loss-level "
        f"class weights. For pure input-level rebalancing, also pass "
        f"--loss_weights uniform."
    )

# Deferred imports: kept below the CLI parse / determinism setup so the
# determinism knobs are applied before torch/transformers initialise any
# global state.  noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
)

from redact.regen.binary_head_model import (  # noqa: E402
    BinaryHeadConfig,
    BinaryHeadModel,
)
from redact.regen.contextual_inputs import (  # noqa: E402
    CONTEXTUAL_SPECIAL_TOKENS,
    build_contextual_input,
)
from redact.training import (  # noqa: E402
    LogPrintCallback,
    compute_balanced_class_weights,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
added = tokenizer.add_special_tokens(
    {"additional_special_tokens": CONTEXTUAL_SPECIAL_TOKENS}
)
log_print(
    f"Registered {added} contextual special tokens "
    f"({CONTEXTUAL_SPECIAL_TOKENS}); tokenizer size now "
    f"{len(tokenizer)}"
)

train_cols = (
    dataset["train"].column_names
    if isinstance(dataset, DatasetDict)
    else dataset.column_names
)
for required in ("prompt", "text", "context_sentences", "label"):
    if required not in train_cols:
        raise ValueError(
            f"contextual_binary requires column {required!r} on "
            f"the training data; saw {train_cols!r}. Re-aggregate "
            f"with the contextual labels schema."
        )

# Gate dual-tokenization on a non-zero --prior_dropout_rate. The
# second (no-priors) variant only matters when the dropout collator
# is active; emitting it unconditionally would cost memory (a
# second tokenization per row) on every training run, so the rate=0
# default stays bit-identical to pre-PR behavior.
_emit_no_priors_variant = args.prior_dropout_rate > 0.0

def _contextual_tokenize_binary(examples):
    input_ids_out: list[list[int]] = []
    attn_out: list[list[int]] = []
    input_ids_no_priors_out: list[list[int]] = []
    attn_no_priors_out: list[list[int]] = []
    for prompt, ctx, focal in zip(
        examples["prompt"],
        examples["context_sentences"],
        examples["text"],
    ):
        built = build_contextual_input(
            prompt=prompt or "",
            prior_sentences=list(ctx) if ctx else [],
            target_sentence=focal,
            tokenizer=tokenizer,
            max_length=args.max_length,
            target_budget=args.target_budget,
            max_prior_sentences=args.max_prior_sentences,
        )
        input_ids_out.append(built.input_ids)
        attn_out.append(built.attention_mask)
        if _emit_no_priors_variant:
            # Build a second variant with priors forced off so the
            # train-time collator can randomly swap it in.
            # ``prior_sentences=[]`` and ``max_prior_sentences=0``
            # both elide the priors block; using both keeps the
            # CTX_TRUNC marker logic from misfiring (it only sets
            # when the input list was non-empty AND got sliced).
            built_no = build_contextual_input(
                prompt=prompt or "",
                prior_sentences=[],
                target_sentence=focal,
                tokenizer=tokenizer,
                max_length=args.max_length,
                target_budget=args.target_budget,
                max_prior_sentences=0,
            )
            input_ids_no_priors_out.append(built_no.input_ids)
            attn_no_priors_out.append(built_no.attention_mask)
    out: dict[str, list[list[int]]] = {
        "input_ids": input_ids_out,
        "attention_mask": attn_out,
    }
    if _emit_no_priors_variant:
        out["input_ids_no_priors"] = input_ids_no_priors_out
        out["attention_mask_no_priors"] = attn_no_priors_out
    return out

tokenized = dataset.map(_contextual_tokenize_binary, batched=True)
# The model consumes the dataset's ``label`` column (0/1) directly;
# HF Trainer wants it named ``labels``.
tokenized = tokenized.rename_columns({"label": "labels"})

# Class weights.
train_labels = list(dataset["train"]["label"])
if args.loss_weights == "uniform":
    binary_weights = torch.ones(2, device=device)
    log_print("Binary class weights: uniform (loss_weights=uniform)")
else:
    binary_weights = compute_balanced_class_weights(train_labels, 2, device)
    log_print(f"Binary class weights: {[f'{w:.2f}' for w in binary_weights.tolist()]}")

binary_sample_weights = None
if args.sample_balance == "binary":
    binary_sample_weights = compute_sample_weights(
        train_labels, alpha=args.sample_alpha,
    )
    log_print(
        f"Sampler: binary-balanced, alpha={args.sample_alpha:g}, "
        f"{len(binary_sample_weights)} train rows"
    )

# Stratified batch sampler — mutually exclusive with sample_balance
# (enforced at argparse). Reads source/label columns from the train
# split BEFORE tokenization (the tokenize pass below preserves the
# 'source' column; we use the pre-tokenize copy here for symmetry
# with the loss_weights_per_source path which also pre-reads them).
stratified_kwargs: dict | None = None
if args.use_stratified_sampler:
    train_cols = dataset["train"].column_names
    if (
        "source" not in train_cols
        or "source_tags" not in train_cols
        or "label" not in train_cols
    ):
        raise ValueError(
            "--use_stratified_sampler requires 'source', "
            "'source_tags', and 'label' columns on "
            f"dataset['train']; saw {train_cols!r}. Re-aggregate "
            "the training data with the contextual labels schema."
        )
    train_sources_for_sampler = list(dataset["train"]["source"])
    train_labels_for_sampler = list(dataset["train"]["label"])
    n_pos = sum(1 for lbl in train_labels_for_sampler if lbl == 1)
    n_idk = sum(
        1 for s, lbl in zip(
            train_sources_for_sampler, train_labels_for_sampler,
        )
        if lbl == 0 and s == "idk"
    )
    n_other = len(train_labels_for_sampler) - n_pos - n_idk
    n_other_per_batch = (
        args.batch_size - args.batch_pos_count - args.batch_idk_count
    )
    steps_per_epoch = -(-n_other // n_other_per_batch)  # ceil-divide
    log_print(
        f"Stratified sampler: {n_pos} positives, {n_idk} IDK "
        f"negatives, {n_other} other negatives. Per batch: "
        f"{args.batch_pos_count} pos + {args.batch_idk_count} idk "
        f"+ {n_other_per_batch} other = {args.batch_size}."
    )
    log_print(
        f"NOTE: each epoch under the stratified sampler is "
        f"ceil({n_other} / {n_other_per_batch}) = {steps_per_epoch} "
        f"steps. The default RandomSampler / WeightedRandomSampler "
        f"path would have done "
        f"ceil({len(train_labels_for_sampler)} / {args.batch_size}) "
        f"= {-(-len(train_labels_for_sampler) // args.batch_size)} "
        f"steps/epoch. HF Trainer derives num_training_steps from "
        f"len(sampler), so --num_train_epochs={args.num_train_epochs}, "
        f"--eval_steps={args.eval_steps}, and the LR schedule are "
        f"all interpreted relative to the new (smaller) step count. "
        f"Re-tune or pass --max_steps for sampling-mode-invariant "
        f"scheduling."
    )
    stratified_kwargs = {
        "sources": train_sources_for_sampler,
        "labels": train_labels_for_sampler,
        "batch_size": args.batch_size,
        "n_pos": args.batch_pos_count,
        "n_idk": args.batch_idk_count,
        # Seed default is filled in by the trainer using args.seed
        # (matches the WeightedRandomSampler resume contract).
        # drop_last default is filled in from args.dataloader_drop_last.
    }

# Per-row, per-source loss weights. Default '' is no-op (uniform per-
# row mean). 'target_fraction_capped' auto-computes the
# ~1/3-per-negative-source target capped at 10x; explicit
# 'name=value' pairs override individual sources.
per_source_loss_weights: dict[str, float] | None = None
if args.loss_weights_per_source:
    train_cols = dataset["train"].column_names
    if (
        "source" not in train_cols
        or "source_tags" not in train_cols
        or "label" not in train_cols
    ):
        raise ValueError(
            "--loss_weights_per_source requires 'source', "
            "'source_tags', and 'label' columns on "
            f"dataset['train']; saw {train_cols!r}. "
            "Re-aggregate the training data with the contextual labels "
            "schema."
        )
    train_sources_list = list(dataset["train"]["source"])
    train_labels_list = list(dataset["train"]["label"])
    per_source_loss_weights = parse_per_source_loss_weights(
        args.loss_weights_per_source, train_sources_list, train_labels_list,
        cap=args.loss_weights_per_source_cap,
    )
    # Warn about any source that did not receive an explicit weight
    # (it implicitly defaults to 1.0). Suppressed for the
    # auto-computed default — there 'unlisted' is the design, not a
    # user oversight.
    if args.loss_weights_per_source.strip() != "target_fraction_capped":
        unique_sources = sorted(set(train_sources_list))
        explicit_keys = {
            p.split("=", 1)[0].strip()
            for p in args.loss_weights_per_source.split(",")
            if p.strip()
        }
        implicit = [s for s in unique_sources if s not in explicit_keys]
        if implicit:
            log_print(
                f"--loss_weights_per_source: sources {implicit!r} "
                f"were not listed in the spec; defaulting to "
                f"weight 1.0."
            )
    log_print(
        f"Per-source loss weights: "
        f"{ {k: round(v, 4) for k, v in per_source_loss_weights.items()} }"
    )
    if args.loss_weights != "uniform":
        log_print(
            "NOTE: --loss_weights_per_source composes "
            "multiplicatively with the class weights from "
            f"--loss_weights={args.loss_weights!r}. For a clean "
            "per-source effect, pair with --loss_weights uniform."
        )
    if args.sample_balance != "none":
        log_print(
            "NOTE: --loss_weights_per_source composes "
            "multiplicatively with the input-level sampler "
            f"--sample_balance={args.sample_balance!r}. For a clean "
            "per-source effect, pair with --sample_balance none."
        )
    # Materialize a per-row ``loss_weight`` column on the tokenized
    # DatasetDict. ``source`` survives ``dataset.map(...)`` so the
    # tokenized DatasetDict still has it on every split (train / test
    # / forget / retain); we add ``loss_weight`` to every split so
    # eval batches don't carry NaN/missing values. Eval-side weights
    # never affect optimization (no backward pass) but the model
    # forward will multiply by them — at eval the relative weighted
    # mean is harmless.

    # Important: the loss_weight column is materialized AFTER the
    # `label` -> `labels` rename at line 979. The closure must read
    # the post-rename column name — otherwise .map() crashes with
    # KeyError on the first invocation in production (caught by
    # code review, missed by the test suite because the integration
    # test pre-built the dataset with a hand-crafted loss_weight
    # column and never invoked this closure).
    def _attach_loss_weight(examples):
        return {
            "loss_weight": compute_loss_weight_column(
                examples["source"], examples["labels"],
                per_source_loss_weights,
            )
        }

    tokenized = tokenized.map(_attach_loss_weight, batched=True)

if args.idk_focal_gamma > 0 or args.gap_loss_weight > 0:
    # Materialize the per-row is_idk_negative bool that
    # BinaryHeadModel.forward needs to identify the IDK subpopulation.
    # Defined as: label == 0 AND source == "idk"
    # — matches the eligible-IDK-negative definition shared with
    # gap_idk and the wrapper's effective_idk_mass diagnostic.
    # Materialized on every split (train / eval) so eval batches
    # don't carry missing values; eval-time forward multiplies
    # harmlessly. Defensive: guard against label==1 IDK rows
    # (shouldn't exist by convention, but we surface it loud if
    # the aggregator ever changes).
    #
    # Precondition: the tokenized DatasetDict must carry a
    # ``source`` column on every split. ``source`` survives
    # ``dataset.map(...)`` from the input dataset, so checking
    # the train split here covers all splits. Surface the
    # column-missing case BEFORE ``.map()`` (which would
    # otherwise crash with a less actionable ``KeyError``
    # inside the closure on first batch).
    idk_required_features = sorted(
        f for f, on in [
            ("idk_focal_gamma", args.idk_focal_gamma > 0),
            ("gap_loss_weight", args.gap_loss_weight > 0),
        ] if on
    )
    tokenized_train_cols = tokenized["train"].column_names
    if "source" not in tokenized_train_cols:
        raise ValueError(
            f"{idk_required_features!r} require a 'source' column "
            f"on the training dataset to materialize "
            f"is_idk_negative (label=0 AND source == 'idk'); saw "
            f"{tokenized_train_cols!r}. Re-aggregate with the "
            f"contextual labels schema."
        )

    def _attach_is_idk_negative(examples):
        mask = []
        for s, lbl in zip(examples["source"], examples["labels"]):
            is_idk = s == "idk"
            if is_idk and lbl == 1:
                raise ValueError(
                    f"is_idk_negative materialization found a "
                    f"label=1 row with source={s!r}. The IDK pool "
                    f"is defined as label=0 only by the canonical "
                    f"pipeline convention; a label=1 IDK row "
                    f"indicates an aggregator bug. Re-aggregate "
                    f"or drop the row."
                )
            mask.append(is_idk and lbl == 0)
        return {"is_idk_negative": mask}

    tokenized = tokenized.map(_attach_is_idk_negative, batched=True)
    n_idk_train = int(sum(tokenized["train"]["is_idk_negative"]))
    log_print(
        f"IDK pool: materialized is_idk_negative on tokenized "
        f"dataset; train split has {n_idk_train} IDK negatives "
        f"out of {len(tokenized['train'])} rows "
        f"({n_idk_train / max(1, len(tokenized['train'])):.2%}). "
        f"Consumed by --idk_focal_gamma "
        f"({args.idk_focal_gamma:g}) and --gap_loss_weight "
        f"({args.gap_loss_weight:g})."
    )
    if n_idk_train == 0:
        log_print(
            f"WARNING: --idk_focal_gamma={args.idk_focal_gamma:g} "
            f"--gap_loss_weight={args.gap_loss_weight:g} but the "
            f"train split has no IDK negatives. Both the focal "
            f"gate and the gap-loss hinge will be inert for the "
            f"entire run. Verify the aggregator produced "
            f"clean_idk* sources for this split (forget01 "
            f"typically has ~3-4%)."
        )

backbone_cfg = AutoConfig.from_pretrained(args.model_name)
backbone_cfg.hidden_dropout_prob = args.dropout
backbone_cfg.attention_probs_dropout_prob = args.dropout

config = BinaryHeadConfig(
    backbone_config=backbone_cfg.to_dict(),
    classifier_dropout=args.dropout,
    binary_class_weights=binary_weights.tolist(),
    binary_threshold=args.threshold,
    max_prior_sentences=args.max_prior_sentences,
    per_source_loss_weights=per_source_loss_weights,
    positive_focal_gamma=args.positive_focal_gamma,
    positive_focal_label_smoothing=args.positive_focal_label_smoothing,
    idk_focal_gamma=args.idk_focal_gamma,
    idk_focal_label_smoothing=args.idk_focal_label_smoothing,
    gap_loss_weight=args.gap_loss_weight,
    gap_margin=args.gap_margin,
)
if args.positive_focal_gamma > 0:
    log_print(
        f"Hard-positive focal mining: gamma="
        f"{args.positive_focal_gamma:g}, label_smoothing="
        f"{args.positive_focal_label_smoothing:g}. Label=1 rows "
        f"weighted by ((1-P(leak)).clamp(eps, 1-eps))^gamma; "
        f"effective_positive_mass diagnostic logged in eval metrics."
    )
    if (
        args.positive_focal_gamma >= 3.0
        and args.positive_focal_label_smoothing == 0.0
    ):
        log_print(
            f"WARNING: --positive_focal_gamma="
            f"{args.positive_focal_gamma:g} without label smoothing "
            f"is fp32-unsafe: at gamma>=3 the (1-p)^gamma factor on a "
            f"confident easy positive (1-p ~ 1e-2) drops below 1e-6 "
            f"and approaches the fp32 denormal floor (~1e-7). The "
            f"1e-6 internal floor in BinaryHeadModel.forward still "
            f"prevents zero division but the loss landscape near "
            f"easy positives degenerates. Pass "
            f"--positive_focal_label_smoothing 0.05 (or higher) to "
            f"clip the gate."
        )
    if args.sample_balance != "none":
        log_print(
            f"NOTE: --sample_balance={args.sample_balance!r} composes "
            f"multiplicatively with hard-positive focal mining. The "
            f"sampler upweights minority-class draws, the focal gate "
            f"then downweights the easy-positive subset of those "
            f"draws — combined effect may cancel the rebalance "
            f"depending on which positives are easy vs. hard. Verify "
            f"the effective_positive_mass diagnostic stays >0.05 "
            f"under the combined config."
        )
if args.idk_focal_gamma > 0:
    log_print(
        f"Hard-IDK-negative focal mining: gamma="
        f"{args.idk_focal_gamma:g}, label_smoothing="
        f"{args.idk_focal_label_smoothing:g}. Rows with "
        f"is_idk_negative=True weighted by "
        f"P(leak).clamp(eps, 1-eps)^gamma; effective_idk_mass "
        f"diagnostic logged in eval metrics. IDK pool is small "
        f"(~3-4%% of forget01 eligible negatives) so this is "
        f"higher-variance than positive focal — monitor "
        f"effective_idk_mass for collapse below ~0.05."
    )
    if (
        args.idk_focal_gamma >= 3.0
        and args.idk_focal_label_smoothing == 0.0
    ):
        log_print(
            f"WARNING: --idk_focal_gamma={args.idk_focal_gamma:g} "
            f"without label smoothing is fp32-unsafe: at gamma>=3 "
            f"the P(leak)^gamma factor on an easy IDK negative "
            f"(P(leak) ~ 1e-2) drops below 1e-6 and approaches the "
            f"fp32 denormal floor. The 1e-6 internal floor in "
            f"BinaryHeadModel.forward still prevents zero division, "
            f"but the loss landscape near easy IDK negatives "
            f"degenerates. Pass --idk_focal_label_smoothing 0.05 "
            f"(or higher) to clip the gate."
        )
    if args.sample_balance != "none":
        log_print(
            f"NOTE: --sample_balance={args.sample_balance!r} composes "
            f"with hard-IDK-negative focal mining. The 'binary' "
            f"sampler upweights label=1 draws, reducing the "
            f"per-batch frequency of IDK negatives (which are "
            f"label=0); the focal gate then preserves only the "
            f"hard subset of those fewer IDK draws. Combined "
            f"effect is a smaller absolute IDK gradient mass than "
            f"either knob alone. Verify effective_idk_mass stays "
            f">0.05 and consider pairing with "
            f"--loss_weights_per_source idk=N to compensate."
        )
    # If the user explicitly zeroed idk via
    # --loss_weights_per_source, the static loss_weight cancels
    # the focal multiplier (lw=0 * focal=anything = 0), so the
    # IDK rows contribute zero gradient regardless of gamma —
    # while ``effective_idk_mass`` would still report a healthy
    # value (it only reflects the focal factor, not the static
    # weight). Warn loudly so the user notices the cancellation
    # before it confuses a sweep.
    if (
        per_source_loss_weights is not None
        and per_source_loss_weights.get("idk", 1.0) == 0.0
    ):
        log_print(
            f"WARNING: --loss_weights_per_source assigns "
            f"idk=0.0 while --idk_focal_gamma="
            f"{args.idk_focal_gamma:g}. The static weight "
            f"cancels the focal multiplier; IDK rows contribute "
            f"zero gradient. ``effective_idk_mass`` will still "
            f"show a healthy value (it ignores the static "
            f"weight) — the diagnostic is misleading under this "
            f"combination. Either bump idk above 0 or "
            f"disable --idk_focal_gamma."
        )
if args.gap_loss_weight > 0:
    log_print(
        f"Pairwise hinge gap loss: weight="
        f"{args.gap_loss_weight:g}, margin={args.gap_margin:g}. "
        f"Total loss = weighted_CE + "
        f"{args.gap_loss_weight:g} * mean_{{(pos,idk)}} "
        f"max(0, {args.gap_margin:g} - (score_pos - score_idk)) "
        f"where ``score = logits[:, 1] - logits[:, 0]``. The "
        f"hinge averages over all (pos, idk) pairs in the batch "
        f"so the signal is dense even when either pool is tiny; "
        f"a batch with zero positives or zero IDKs degenerates "
        f"to a 0-d zero (graceful no-op). Monitor the eval "
        f"``gap_idk`` metric to confirm the margin is being "
        f"satisfied; the raw per-step gap-loss value is stashed "
        f"on ``model._last_gap_loss`` (detached 0-d tensor) for "
        f"a logging callback to consume off the hot path."
    )

model = BinaryHeadModel(config, binary_class_weights=binary_weights)
pretrained_backbone = AutoModel.from_pretrained(
    args.model_name, use_safetensors=True,
)
pretrained_sd = {
    k: v for k, v in pretrained_backbone.state_dict().items()
    if not k.startswith("pooler.")
}
missing, unexpected = model.backbone.load_state_dict(
    pretrained_sd, strict=False,
)
if missing:
    log_print(f"WARNING: missing backbone keys: {missing}")
if unexpected:
    log_print(f"WARNING: unexpected backbone keys: {unexpected}")
del pretrained_backbone

model.backbone.resize_token_embeddings(len(tokenizer))
log_print(
    f"Resized backbone embeddings to {len(tokenizer)} for "
    f"contextual special tokens."
)

log_print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
model = model.to(device)

# ``test_pauc_retain_aux`` (the canonical default, set in argparse) is
# computed by the wrapped compute_metrics below. If a user explicitly
# passes a metric this script doesn't compute, HF Trainer will raise at
# the first eval — that's the right failure mode.

# AdamW betas/epsilon, weight decay, warmup, cosine schedule, and
# disabled grad clipping follow the Mosbach et al. (2021) stable
# RoBERTa fine-tuning recipe; these are functional optimizer
# settings, not free choices.
optimizer_kwargs = dict(
    learning_rate=args.learning_rate,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    max_grad_norm=args.max_grad_norm,
    warmup_ratio=0.1,
    lr_scheduler_type=args.lr_scheduler_type,
)
# ``seed`` fixes weight init / shuffling; ``data_seed`` fixes the
# Trainer's per-epoch dataloader RNG (sample order on the default
# RandomSampler path — the WeightedRandomSampler path in
# WeightedSplitAwareTrainer manual-seeds its own generator). Pinning
# the two equal keeps the RNG streams in lockstep across runs, the
# contract enable_full_determinism advertises.
rng_kwargs = dict(seed=args.seed, data_seed=args.seed)
training_args = TrainingArguments(
    output_dir=save_dir,
    num_train_epochs=args.num_train_epochs,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    do_eval=True,
    eval_strategy="steps",
    eval_steps=args.eval_steps,
    logging_strategy="steps",
    logging_steps=args.eval_steps,
    save_strategy="steps",
    save_steps=args.save_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model=args.metric_for_best_model,
    greater_is_better=_greater_is_better,
    report_to="none",
    **optimizer_kwargs,
    **rng_kwargs,
)

eval_split_names = [str(k) for k in tokenized if str(k) != "train"]
eval_datasets = {k: tokenized[k] for k in eval_split_names if k in tokenized}
log_print(f"Train: {len(tokenized['train'])} rows, eval: {list(eval_datasets.keys())}")

compute_metrics_fn = LeakMetricsAggregator(
    base_compute_metrics=functools.partial(
        compute_binary_metrics,
        threshold=args.threshold,
        gamma=args.positive_focal_gamma,
        label_smoothing=args.positive_focal_label_smoothing,
    ),
    p_leak_extractor=binary_p_leak_extractor,
    label_extractor=binary_label_extractor,
    test_sources=test_sources,
    pauc_fpr_cap=args.pauc_fpr_cap,
    pauc_recall_lower=args.pauc_recall_lower,
    fpr_at_recall_target=args.fpr_at_recall_target,
    pauc_source_balance=args.pauc_source_balance,
    idk_focal_gamma=args.idk_focal_gamma,
    idk_focal_label_smoothing=args.idk_focal_label_smoothing,
    smoothing_window=args.smoothing_window,
)

# Defensive sanity: when --prior_dropout_rate > 0, the tokenization
# step above is supposed to have emitted both no_priors columns on
# every split (the train collator needs them; eval splits keep them
# too for column-schema symmetry). Surface here before trainer
# construction so a config-drift bug (e.g. someone wiring the rate
# to the trainer without enabling the tokenizer gate) raises a
# legible error rather than a cryptic KeyError deep in the collator.
if args.prior_dropout_rate > 0:
    tokenized_train_cols = tokenized["train"].column_names
    for required in ("input_ids_no_priors", "attention_mask_no_priors"):
        if required not in tokenized_train_cols:
            raise ValueError(
                f"--prior_dropout_rate={args.prior_dropout_rate} "
                f"requires the {required!r} column on the tokenized "
                f"train dataset, but it is missing (saw "
                f"{tokenized_train_cols!r}). The dual-tokenization "
                f"step in _contextual_tokenize_binary is gated on "
                f"args.prior_dropout_rate > 0 — verify that gate "
                f"and re-run."
            )
    log_print(
        f"Prior dropout active: rate={args.prior_dropout_rate:.2f}, "
        f"seed={args.seed}. Train-time collator will swap each "
        f"example's priors-on inputs for the priors-off variant "
        f"with probability {args.prior_dropout_rate:.2f}. Eval "
        f"path is untouched (priors always present at eval time)."
    )

binary_trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=eval_datasets,
    compute_metrics=compute_metrics_fn,
    data_collator=DataCollatorWithPadding(tokenizer, padding="longest"),
    callbacks=[LogPrintCallback()],
)
# Prior dropout composes with both rebalancing mechanisms. When
# active, we always route through WeightedSplitAwareTrainer (which
# owns the collator-wrap logic); when inactive (rate=0), we keep
# the legacy SplitAwareTrainer path when neither rebalancing
# mechanism is set.
if binary_sample_weights is not None:
    trainer = WeightedSplitAwareTrainer(
        sample_weights=binary_sample_weights,
        prior_dropout_rate=args.prior_dropout_rate,
        prior_dropout_seed=args.seed,
        **binary_trainer_kwargs,
    )
elif stratified_kwargs is not None:
    trainer = WeightedSplitAwareTrainer(
        stratified_batch_sampler_kwargs=stratified_kwargs,
        prior_dropout_rate=args.prior_dropout_rate,
        prior_dropout_seed=args.seed,
        **binary_trainer_kwargs,
    )
elif args.prior_dropout_rate > 0:
    # Standalone prior-dropout path: no rebalancing mechanism, but
    # we still need the WeightedSplitAwareTrainer subclass for the
    # collator-wrap behavior.
    trainer = WeightedSplitAwareTrainer(
        prior_dropout_rate=args.prior_dropout_rate,
        prior_dropout_seed=args.seed,
        **binary_trainer_kwargs,
    )
else:
    trainer = SplitAwareTrainer(**binary_trainer_kwargs)
# Aggregator needs a forward ref to read trainer._current_metric_prefix
# for split dispatch. Trainer takes compute_metrics as a constructor
# arg, so the binding can only happen post-construction.
compute_metrics_fn.trainer = trainer

log_print("\n--- Training contextual binary head ---")
trainer.train()
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
log_print(f"Model saved to {save_dir}")
