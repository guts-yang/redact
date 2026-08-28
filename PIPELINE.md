# REDaCT Pipeline

The end-to-end recipe for training and evaluating **REDaCT** (REgenerate via
Detection and Corrective Tuning) on R-TOFU. REDaCT is an inference-time
unlearning method over a **frozen** target model (`sangyon/LRM-target`, a
DeepSeek-R1-Distill-Llama-8B fine-tuned on R-TOFU). At inference, only two
forget-set artifacts are needed: a **binary-head leak classifier** and a
**single learned soft token**. No plain-text forget data, no claim
embeddings, and no sentence-transformer backbone are needed at runtime. The
resolved knowledge bank is used only at training time, as the rubric for
first-leak labeling.

This document is the detailed, command-level recipe. For the high-level tour,
setup, and repo layout, see [README.md](README.md).

> Conventions: all scripts run as modules from the repo root
> (`python -m scripts.X`). Artifacts are organized `<component>/<split>/...`.
> The canonical split is `forget10`; `forget05` / `forget01` are derived from
> it (see [Subset-split reuse](#subset-split-reuse-forget05--forget01-from-forget10)).

---

## Architecture overview

**Detector — `CoTLeakDetector`** (`redact/regen/leak_detector.py`).
Splits a generated chain-of-thought (CoT) into sentences and delegates
per-sentence classification to a `ContextualBinaryLeakClassifier`
(`redact/regen/binary_head_model.py`, `redact/regen/classifier.py`): a RoBERTa
encoder with a **single binary head**. Inputs use the contextual schema
(prompt + k-window prior context + `<TGT>`/`</TGT>` markers around the focal
sentence). A sentence is flagged iff the binary head's `P(leak)` exceeds its
persisted threshold. The head is binary and the inference-time corruption is a
single shared soft token, so a leak is a leak — the detector never has to say
*which* forget-set fact leaked.

**Regenerator — `RegeneratingModel`** (`redact/regen/model.py`) driven by the
regenerating engine (`redact/inference_regen.py`). It generates the initial
response, then runs detect-and-regenerate in two batched waves:

- **CoT wave** — for each row whose CoT leaks: truncate at the first flagged
  sentence (everything before is the clean prefix), append a pad slot whose
  embedding is overwritten by the learned `SoftToken` via a forward hook on
  the embedding module, and regenerate from the truncation point. If the
  detector still flags at the same anchor, push another anchor onto the
  soft-token stack and retry up to `--regen_max_attempts`. If a *later*
  sentence is flagged, extend the clean prefix forward and add a new stack
  entry.
- **Answer wave** — after the CoT wave converges, the detector runs on every
  row's current answer (covering both CoT-leaker rows and originally-clean
  rows). The model input is `prompt + <think>\n + cot + </think>\n\n +
  answer_prefix`, where `cot + </think>\n\n` is a frozen baseline; only the
  answer prefix grows under forward extension. CoT-phase soft tokens carry
  forward and keep their positions inside the CoT portion.

**Critical implementation details**:
- Regeneration builds `input_ids` by concatenating token lists
  (`prompt_ids + think_ids + prefix_ids`), NOT by tokenizing a joined string —
  BPE is context-sensitive at boundaries.
- `SoftToken.apply_hook()` clones outputs before writing so the autograd graph
  survives during soft-token training.
- Checkpoint selection (classifier): the canonical metric is
  `test_pauc_retain_aux` — partial AUROC at FPR ≤ `--pauc_fpr_cap` (default
  0.10), with negatives restricted to the `retain` + `aux` primary sources.
  This directly weights the two operating-point quantities that drive
  deployment behavior: forget-set recall and retain/aux false-positive rate.

---

## Stage 0 — Data download

The R-TOFU JSONLs are gitignored; fetch them once after cloning:

```bash
python -m scripts.download_rtofu_data
```

This reads the pinned upstream commit SHA and raw-URL pattern from
`data/rtofu/PROVENANCE.md` and writes the eight files
(`forget01/05/10.json`, `retain10/50/90.json`,
`{real_authors,world_facts}_perturbed.json`) into `data/rtofu/`. Re-running
skips files that already exist (pass `--force` to refetch).

`data/rtofu/` notes (see `PROVENANCE.md` for the full schema):
- Forget splits carry a string `task_id`; all canonical experiments use
  `task_id=1` (the default in `RTOFU.__init__`). Other task_ids are reachable
  via `--task_id` on the generation/judge scripts for sensitivity analysis.
- `retainYY.json` is the "always-retained" core, not the retain set for
  `forget(100−YY)`. `RTOFU.download()` builds protocol-correct synthetic
  retain splits `forget01_retain` / `forget05_retain` / `forget10_retain`
  (= `full − forgetXX[task_id=K]`) for use as the MU retain split.
- Aux splits load `original_answer` (the elaborated sentence) and rename it to
  `answer`, since the paper's MU metric keys off the elaborated string.

---

## Stage 1 — Leak classifier training data

The classifier trains on a merged contextual-schema dataset spanning four
label-space sources: `forget` (positives), `retain`, `aux`, and `idk`
(negatives). The whole assembly is driven by one orchestrator,
`build_first_leak_classifier_data`, which also builds the clean-forget base
("step 0") for you.

### 1a. Generate target-model CoTs

One generation per split. The forget + protocol-retain CSVs feed labeling and
the retain negative source; aux CSVs are shared across forget splits.

```bash
python -m scripts.generate_rtofu_responses --split forget10 \
    --output leak_detector_data_promptfmt/forget10/forget_responses.csv
python -m scripts.generate_rtofu_responses --split forget10_retain \
    --output leak_detector_data_promptfmt/forget10/retain_responses.csv
python -m scripts.generate_rtofu_responses --split real_authors \
    --output leak_detector_data_promptfmt/aux/real_authors_responses.csv
python -m scripts.generate_rtofu_responses --split world_facts \
    --output leak_detector_data_promptfmt/aux/world_facts_responses.csv
# --max_new_tokens defaults to 2048 (matches R-TOFU's eval default).
```

### 1b. First-leak labeling (LLM-as-judge)

For each forget CoT, an LLM judge identifies the **first** sentence that leaks
the gold answer. Output is a resumable JSONL.

The judge adjudicates against the **knowledge bank** —
`knowledge_banks/<split>/claims.json`, the list of forget-set claims for the
split. It ships with the repository, so there is nothing to build; see
[`knowledge_banks/PROVENANCE.md`](knowledge_banks/PROVENANCE.md) for its
contents, derivation, and the commands to re-derive it. It is a training-time
input only — nothing under `knowledge_banks/` is read at inference or during
evaluation.

```bash
python -m scripts.label_first_leak_rtofu --split forget10 \
    --model sonnet --workers 4
# Writes leak_detector_data_promptfmt/forget10/first_leak_labels.final.jsonl
# (default). Resumable; --limit N for a quick subset.
```

### 1c. (Optional) Auxiliary negative sources

The orchestrator (step 1, below) builds the `idk`, `aux`, `retain`, and
`forget_answers` sources for you from `--split`-derived paths. Run a source
standalone via `build_labeled_sources` only for a partial rebuild — e.g. after
editing the IDK paraphrases:

```bash
# IDK (prose + paraphrase refusal variants) — the post-regen "what the model
# should produce" negative signal.
python -m scripts.build_labeled_sources --source idk \
    --split forget10 \
    --input data/reasoned_idk/reasoned_idk_forget10.json \
    --paraphrases paraphrases/forget10.jsonl \
    --expected_k 6 \
    --output leak_detector_data_promptfmt/forget10/contextual_labeled_idk

# Refusal-variant paraphrases (split-agnostic; generated on forget10 and
# sliced for subsets downstream). Uses the Claude CLI. NOT shipped: generate
# them with the command below before the --source idk step above. The
# orchestrator degrades gracefully if paraphrases/ is absent (it logs
# "building IDK with prose only"), but the classifier's negative mix then
# differs from the reported runs.
python -m scripts.paraphrase_idk_answers --split forget10 \
    --output_jsonl paraphrases/forget10.jsonl --k 3 --model sonnet --workers 4
```

### 1d. Orchestrate the full classifier dataset

```bash
python -m scripts.build_first_leak_classifier_data --split forget10
```

This runs, in order:
0. **Clean-forget base** (`contextual_labeled`) via
   `build_labeled_sources --source forget_structural` — sentence-splits the
   forget CSV into placeholder-label rows whose per-sentence structure the next
   step consumes. (Override the source CSV with `--structural_input`; skip with
   `--skip_structural_build` if the base is already staged.)
1. **Per-source artifacts** — `idk`, `aux`, `retain`, `forget_answers` (paths
   derive from `--split`; override individually via `--idk_input`,
   `--aux_inputs`, `--retain_input`, `--forget_input`, `--paraphrases_input`).
   Skip with `--skip_source_build`.
2. **First-leak truncation + concat** — reads
   `first_leak_labels.final.jsonl`, truncates the base at the first-leak
   position per CoT (sentences before → `label=0`; the first-leak sentence →
   `label=1`; sentences after → dropped), then
   concatenates `forget + retain + idk + aux + forget_answers` into
   `contextual_labeled_full_firstleak`.
3. **Sentence-0 reword post-patch** (forget10 only) — applies 36 hand-curated
   rewordings, producing the `_aug_v4` variants. Skip with
   `--skip_sentence0_reword`.
4. **Subset slice** (forget10 only) — derives forget05 + forget01 classifier
   data from forget10's canonical `_firstleak` artifact. Skip with
   `--skip_subset_build`.

`--split` is restricted to `forget10`; subsets are produced by step 4 (or
rebuilt standalone — see
[Subset-split reuse](#subset-split-reuse-forget05--forget01-from-forget10)).

---

## Stage 2 — Train the leak classifier

```bash
python -m scripts.train_classifier \
    --split forget10/contextual_labeled_full_firstleak \
    --data_dir leak_detector_data_promptfmt \
    --model_type contextual_binary \
    --max_prior_sentences 2 \
    --num_train_epochs 7 \
    --threshold 0.95 \
    --output_dir leak_classifiers
# Output: leak_classifiers/forget10/contextual_labeled_full_firstleak_contextual_binary
```

Notes:
- `--model_type contextual_binary` is the canonical (binary-head) detector.
- `--threshold` persists onto the model's `binary_threshold`, so generation
  reads that operating point when `--leak_classifier_threshold` is omitted.
  Use the split's reported value: **0.85 / 0.9 / 0.95** for forget01 /
  forget05 / forget10 (the example above is forget10). Training a checkpoint
  at some other value and then relying on the persisted default at generation
  time will silently change the flagging rate, not error.
- The default `--metric_for_best_model` is `test_pauc_retain_aux` (partial
  AUROC at FPR ≤ `--pauc_fpr_cap`, default 0.10, negatives = `retain` + `aux`).
  Pass `--metric_for_best_model test_binary_f1` for the legacy F1 selection.
- The dataset must carry `prompt`, `text`, `context_sentences`, `label`,
  `source`, and `source_tags` columns (the contextual schema).

> **Canonical artifact provenance.** A single fresh run emits the auto-named
> `leak_classifiers/<split>/contextual_labeled_full_firstleak_contextual_binary`
> (from `{split}_{model_type}`). The paper's in-use classifier is the **best of
> 3 seeds** trained on this same firstleak data, selected by
> `gap_idk_robust_smoothed`, then promoted/renamed to the canonical
> `leak_classifiers/<split>/contextual_labeled_full_contextual_binary`. Full
> hyperparameters are in the paper appendix; the firstleak labeling prompt is
> supplemental material. Inference (Stage 4) and the artifact download (see
> [Artifact paths](#artifact-paths)) point at the canonical
> `..._full_contextual_binary` path.

---

## Stage 3 — Soft token

The soft token is a single learned sequence of embedding vectors (length
k=16) injected at the regeneration anchor. It is independent of the classifier
under the IDK flow, so it can be trained in parallel with Stages 2d–3.

```bash
# 3a. Soft-token training data. NB: --label_dataset is the FIRSTLEAK-labeled
# dataset (one adjudicated leak per CoT).
python -m scripts.build_soft_token_data \
    --label_dataset leak_detector_data_promptfmt/forget10/contextual_labeled_full_firstleak \
    --forget_cots leak_detector_data_promptfmt/forget10/forget_responses.csv \
    --retain_cots leak_detector_data_promptfmt/forget10/retain_responses.csv \
    --reasoned_idk data/reasoned_idk/reasoned_idk_forget10.json \
    --output soft_token_data_promptfmt/forget10

# 3b. Train a single shared SoftToken of length k=16 with the target frozen.
python -m scripts.train_soft_token \
    --split forget10 \
    --data_dir soft_token_data_promptfmt \
    --n_tokens 16 \
    --lr 1e-3 --lr_schedule cosine --epochs 100 \
    --idk_mode answer \
    --forget_margin 5 --lambda_idk 1.0 \
    --lambda_retain_clean 1.0 --retain_margin 1 \
    --output_dir soft_tokens_promptfmt_single/k16_lr1e-3_lret1.0_rm1_fm5_cos
# Saves to
# soft_tokens_promptfmt_single/k16_lr1e-3_lret1.0_rm1_fm5_cos/forget10/embedding.pt
# All three splits use --epochs 100 (forget01/forget05/forget10 alike).
```

**Objective** — hinge-on-forget + IDK + clipped-retain:

```
L = max(0, m_f − L_forget) + λ_idk · L_idk + λ_ret · max(0, L_retain − m_r)
```

with canonical settings `m_f = 5`, `λ_idk = 1.0`, `λ_ret = 1.0`, `m_r = 1`,
`--idk_mode answer`. `L_forget` is CE on the leaking continuation; `L_idk` is CE
on the reasoned-refusal continuation for the same prompt (from
`data/reasoned_idk/`); `L_retain` is CE on a clean retain continuation. The
forget hinge saturates once `L_forget ≥ m_f` (capping the reward for further CE
growth so the optimizer doesn't drive the hidden state into gibberish); the
retain hinge saturates once `L_retain ≤ m_r`.

---

## Stage 4 — Generate with REDaCT

`--leak_classifier_path` is the implicit "regen on" signal; the
`contextual_binary` detector is paired with a single shared `SoftToken` via
`--soft_token_path`. Aux/retain splits ALSO run through regen so MU reflects
the full method's utility preservation.

The detector threshold defaults to the value persisted on the classifier
checkpoint (0.85 / 0.9 / 0.95 for forget01 / forget05 / forget10). The reported
runs used that default for `forget05` and `forget10`, but overrode `forget01`
to 0.5 with `--leak_classifier_threshold 0.5` — see
[Artifact paths](#artifact-paths).

```bash
REGEN_FLAGS=(
    --classifier_type contextual_binary
    --leak_classifier_path leak_classifiers/forget10/contextual_labeled_full_contextual_binary
    --soft_token_path artifacts/soft_tokens/forget10/embedding.pt
)
# NB: this uses the VENDORED token. To evaluate one you just trained in
# Stage 3, point --soft_token_path at
# soft_tokens_promptfmt_single/k16_lr1e-3_lret1.0_rm1_fm5_cos/<split>/embedding.pt

python -m scripts.generate_rtofu_responses --split forget10 "${REGEN_FLAGS[@]}" \
    --output results/redact/forget10.csv
python -m scripts.generate_rtofu_responses --split forget10_retain "${REGEN_FLAGS[@]}" \
    --output results/redact/forget10_retain.csv
python -m scripts.generate_rtofu_responses --split real_authors "${REGEN_FLAGS[@]}" \
    --output results/redact/real_authors.csv
python -m scripts.generate_rtofu_responses --split world_facts "${REGEN_FLAGS[@]}" \
    --output results/redact/world_facts.csv
```

Add `--task_id K` to run a non-default task_id for sensitivity analysis.
Output CSVs carry columns `index, question, gold_answer, response, gold_cot,
model_cot`.

---

## Stage 5 — Evaluate

The lean evaluator emits the **reported, LLM-judge** metrics only:
**AFE_judge**, **CFE_judge**, and **MU**. (The unreported harmonic-mean AFE/CFE
path has been removed.) The two judge metrics require pre-computed judge JSONLs;
MU is computed directly from the retain + aux CSVs.

```bash
# 5a. AFE judge — 4-way rubric {LEAK > HALLUC > COLLAPSE > REFUSE} on the
#     post-</think> answer. AFE_judge = 1 − LEAK_rate.
python -m scripts.judge_afe_rtofu \
    --csv results/redact/forget10.csv --split forget10 \
    --output_jsonl rtofu_judge_scores/forget10/afe.jsonl \
    --model sonnet --workers 4

# 5b. CFE judge — R-TOFU's graded judge score for how much the CoT reveals the
#     gold answer. CFE_judge = 1 − mean(score).
python -m scripts.judge_cfe_rtofu \
    --csv results/redact/forget10.csv --split forget10 \
    --output_jsonl rtofu_judge_scores/forget10/cfe.jsonl \
    --model sonnet --workers 4

# 5c. Aggregate AFE_judge / CFE_judge / MU in a single pass. The --mu_*_csv
#     triple feeds MU (hmean over 12 means: {ROUGE-L recall, cosine, entailment,
#     token entropy} × {retain, real_authors, world_facts}). One invocation
#     amortizes the SentenceTransformer + DeBERTa NLI + tokenizer load.
python -m scripts.evaluate_rtofu_from_csv \
    --csv results/redact/forget10.csv --split forget10 \
    --judge_afe_scores rtofu_judge_scores/forget10/afe.jsonl \
    --rtofu_judge_scores rtofu_judge_scores/forget10/cfe.jsonl \
    --mu_retain_csv results/redact/forget10_retain.csv \
    --mu_real_authors_csv results/redact/real_authors.csv \
    --mu_world_facts_csv results/redact/world_facts.csv \
    --output_json results/redact/forget10_eval.json
```

For batch scoring across many CSVs (main method + baselines across
forget01/05/10), use `--jobs_file eval_jobs.json` so the evaluators load once
and are reused across every job. Each job is an object with required keys `csv`,
`split` and optional `output_json`, `judge_afe_scores`, `rtofu_judge_scores`,
`treatment`, `config`, and the `mu_{retain,real_authors,world_facts}_csv` triple.

> Judge note: row order couples the judge JSONL to the CSV via the `index`
> column — do not reorder the CSV between judging and scoring.

---

## Subset-split reuse (forget05 / forget01 from forget10)

The R-TOFU forget splits are nested at task_id=1: forget01 ⊂ forget05 ⊂
forget10 in both row and claim order. Two stages are reusable from forget10's
output, so a subset split skips most resolution + labeling work:

- **Resolutions** — slice forget10's resolution JSON by `claim_idx <
  N_subset_kb` via `slice_kb_resolutions`, then feed to `resolve_kb`.
- **Forget + retain labels** — `slice_clean_labels` remaps every forget row and
  retain row from forget10's labeled dataset against the subset KB
  (out-of-subset forget prompts flip to `retain` with a `xfer` tag). The
  end-to-end wrapper is `build_subset_classifier_data`.

`build_first_leak_classifier_data --split forget10` already runs the subset
slice as its step 4. To rebuild just one subset standalone:

```bash
# forget05: baseline KB size N_subset_kb=248; forget prompt count
# N_subset_forget=200 (first 200 of forget10's 400). (forget01: 50 / 40.)

# 1. Subset classifier data, sliced from forget10's labeled artifacts.
python -m scripts.build_subset_classifier_data --firstleak --splits forget05 forget01
# Add --include_retain_xfer to fold out-of-subset forget-prompt rows into the
# retain artifact (default: dropped).

# 2. Forget + retain response CSVs are forget10's verbatim — sliced retain rows
#    key into forget10's retain_responses.csv; sliced forget rows (cot_idx <
#    N_subset_forget) key into the forget10 forget_responses.csv prefix.

# 3. Subset soft-token data — same command as Stage 3a, pointed at the subset's
#    own labeled dataset.
python -m scripts.build_soft_token_data \
    --label_dataset leak_detector_data_promptfmt/forget05/contextual_labeled_full_firstleak \
    --forget_cots leak_detector_data_promptfmt/forget10/forget_responses.csv \
    --retain_cots leak_detector_data_promptfmt/forget10/retain_responses.csv \
    --reasoned_idk data/reasoned_idk/reasoned_idk_forget05.json \
    --output soft_token_data_promptfmt/forget05

# 4. From here, run train_classifier / train_soft_token / generate / evaluate as
#    in Stages 2–5, substituting forget05 for forget10.
```

---

## Baselines

Two families of comparison methods (see [README.md](README.md#baselines)):

- **R2MU** (in-repo, `scripts/r2mu/`) — runs in this repo, in two steps.

  R2MU's "extent" (unthink-loss) stream needs `{question, cot}` pairs produced
  by the target model on the forget prompts. Stage 1a already generates exactly
  that, so build the extent file from its output — `scripts/r2mu/pregen_extent.py`
  is a placeholder that exits without doing work.

  ```bash
  # 1. Extent stream, reusing the Stage 1a forget responses (zero extra compute).
  python -m scripts.r2mu.reformat_raw_cots_to_extent \
      --csv leak_detector_data_promptfmt/forget10/forget_responses.csv \
      --output r2mu_data/forget10/extent.jsonl

  # 2. Unlearn. --extent-path and --output-dir are both REQUIRED.
  python -m scripts.r2mu.unlearn \
      --split forget10 \
      --extent-path r2mu_data/forget10/extent.jsonl \
      --output-dir r2mu_checkpoints/forget10
  ```

  For forget01 / forget05, filter forget10's responses rather than regenerating:

  ```bash
  python -m scripts.r2mu.reformat_raw_cots_to_extent \
      --csv leak_detector_data_promptfmt/forget10/forget_responses.csv \
      --filter-split-json data/rtofu/forget01.json --filter-task-id 1 \
      --output r2mu_data/forget01/extent.jsonl
  ```

  Method knobs fall back to `--steering-coeff 6.5 --alpha 1.4 --beta 1.0 --lr
  7.5e-5 --layer-id 7 --trainable-layers 5,6,7 --batch-size 4 --seed 42`; see
  `scripts/r2mu/unlearn.py` and `redact/baselines/r2mu.py`. Then point
  `config/rtofu_model_config/r2mu-forget<XX>.yaml`'s `hf_name` at your
  `--output-dir` (it ships as a `/path/to/rtofu-baselines/...` placeholder, like
  the external baselines) and generate/score through the same flow.
- **GA / GD / KL / IDK** (external R-TOFU trainer) — train the checkpoints with
  the upstream R-TOFU repo, then point `config/rtofu_model_config/<name>.yaml`
  at the resulting checkpoint (the configs ship with
  `/path/to/rtofu-baselines/<name>` placeholders — edit `hf_name`). Generate
  with `generate_rtofu_responses --model_name <name>` and the YAML's `baseline:
  true` flag set (the generation script then refuses the inference-time defense
  flags so double-unlearning runs are caught at launch). Score via the same
  judge + `evaluate_rtofu_from_csv` flow as the main method.

---

## Artifact paths

The two inference-time artifacts can be used pre-trained instead of rebuilt
from Stages 1–3.

**Soft tokens** are vendored in this repository:

```
artifacts/soft_tokens/forget01/embedding.pt
artifacts/soft_tokens/forget05/embedding.pt
artifacts/soft_tokens/forget10/embedding.pt
```

Each holds a single `(16, 4096)` float32 tensor under the key `embedding`.

**Leak classifiers** are hosted on the Hugging Face Hub:

```bash
hf download jmaccoby/redact-leak-classifier --local-dir leak_classifiers
```

which produces `leak_classifiers/<split>/contextual_labeled_full_contextual_binary/`.
Loading requires only `config.json`, `model.safetensors`, `tokenizer.json`, and
`tokenizer_config.json`; `training_args.bin` is shipped as provenance and is
not read.

### Reproducing the reported runs

The leak-classifier threshold differs per split. `forget05` and `forget10` use
the value persisted on their checkpoints, so `--leak_classifier_threshold` may
be omitted; `forget01` requires an explicit override, because its checkpoint
persists 0.85 while the reported run used 0.5.

`--soft_token_path` requires `--leak_classifier_path`; the script rejects one
without the other. `--output` is also required in practice — without it the run
completes and prints results but writes no CSV for Stage 5.

```bash
# forget01 — explicit threshold override (checkpoint persists 0.85)
python -m scripts.generate_rtofu_responses --split forget01 \
    --leak_classifier_path leak_classifiers/forget01/contextual_labeled_full_contextual_binary \
    --leak_classifier_threshold 0.5 \
    --soft_token_path artifacts/soft_tokens/forget01/embedding.pt \
    --output results/redact/forget01.csv

# forget05 — checkpoint default (0.9)
python -m scripts.generate_rtofu_responses --split forget05 \
    --leak_classifier_path leak_classifiers/forget05/contextual_labeled_full_contextual_binary \
    --soft_token_path artifacts/soft_tokens/forget05/embedding.pt \
    --output results/redact/forget05.csv

# forget10 — checkpoint default (0.95)
python -m scripts.generate_rtofu_responses --split forget10 \
    --leak_classifier_path leak_classifiers/forget10/contextual_labeled_full_contextual_binary \
    --soft_token_path artifacts/soft_tokens/forget10/embedding.pt \
    --output results/redact/forget10.csv
```

Repeat each with `--split <split>_retain`, `--split real_authors`, and
`--split world_facts` (same flags) to produce the utility inputs Stage 5
needs.

Pass `--leak_classifier_threshold` explicitly on all three if you want the
run to be independent of whatever value a checkpoint happens to carry:
0.5 / 0.9 / 0.95 for forget01 / forget05 / forget10 respectively.

See the README's *Pre-trained artifacts* section for the soft-token training
configuration and for a caveat about the `forget01` classifier.
