# REDaCT

**REDaCT** (REgenerate via Detection and Corrective Tuning) is an
inference-time unlearning method for reasoning LLMs. Given a **frozen** target
model, REDaCT detects when a generated chain-of-thought (CoT) is about to leak
forgotten knowledge using a lightweight **binary-head leak classifier**, then
regenerates from the offending sentence with a learned **corrective embedding** injected
into the embedding layer via a forward hook. Unlearning requires no retraining
of the target and no plain-text forget data at inference — only the classifier
checkpoint and a single soft-token embedding. This repo reproduces REDaCT on
[R-TOFU](https://arxiv.org/abs/2505.15214) (Reasoning-TOFU) with the target
model `sangyon/LRM-target` (a DeepSeek-R1-Distill-Llama-8B fine-tuned on
R-TOFU).

## Repo layout

```
redact/                 Library package
  regen/                Detect-and-regenerate core:
                          classifier.py        ContextualBinaryLeakClassifier
                          binary_head_model.py  RoBERTa + single binary head
                          leak_detector.py      CoTLeakDetector
                          model.py              RegeneratingModel
                          learned_hooks.py      SoftToken (embedding-hook corruption)
  model/                HFModel / ReasoningModel target-model wrappers
  dataset/              R-TOFU loader + base dataset machinery
  evaluator/            ROUGE / cosine / entailment / token-entropy / LLM-judge
  baselines/            R2MU baseline implementation
  inference.py          ReasoningGenerationEngine (initial generation)
  inference_regen.py    RegeneratingReasoningEngine (the two-wave regen loop)

scripts/                Run as `python -m scripts.X` from the repo root
  download_rtofu_data.py        Fetch the vendored R-TOFU JSONLs
  build_kb / resolve_kb / slice_kb_resolutions   Rebuild claims.json (optional)
  build_labeled_sources         Per-source classifier data (idk/aux/retain/...)
  label_first_leak_rtofu        LLM-as-judge first-leak labeling
  build_first_leak_classifier_data   Orchestrator for the full classifier dataset
  build_subset_classifier_data / slice_clean_labels   forget05/01 reuse
  train_classifier              Train the binary leak classifier
  build_soft_token_data / train_soft_token            Train the soft token
  generate_rtofu_responses      Generate (baseline or REDaCT regen) -> CSV
  judge_afe_rtofu / judge_cfe_rtofu                   LLM-as-judge scorers
  evaluate_rtofu_from_csv       Aggregate AFE_judge / CFE_judge / MU
  r2mu/                         R2MU baseline entrypoints

config/
  rtofu_model_config/   Target + baseline model YAMLs (embedding_module, etc.)
  birth_claims/         Per-split birth-fact claims appended to the KB
  authors/              Per-split author lists (labeling batch grouping)
  training_config/      Accelerate config

data/rtofu/             R-TOFU JSONLs (gitignored; PROVENANCE.md is tracked)
tests/                  pytest suite
PIPELINE.md             Authoritative, command-level reproduction recipe
```

## Setup

Python 3.11. Create the environment (conda recommended):

```bash
conda env create -f environment.yml
conda activate redact
```

Or with pip into an existing 3.11 environment (install a torch build matching
your accelerator first — see the file header):

```bash
pip install -r requirements.txt
```

No package install is needed to run anything: scripts execute as modules from
the repo root (`python -m scripts.X`), which puts both `redact/` and `scripts/`
on the import path. `pyproject.toml` carries the same dependency pins as
metadata.

**Download the data** (the R-TOFU JSONLs are gitignored):

```bash
python -m scripts.download_rtofu_data
```

This fetches the eight forget/retain/aux JSONLs from the pinned upstream commit
recorded in [`data/rtofu/PROVENANCE.md`](data/rtofu/PROVENANCE.md) into
`data/rtofu/`.

## End-to-end flow

The full command-level recipe lives in **[PIPELINE.md](PIPELINE.md)**. In
brief:

1. **Train artifacts** (per forget split; canonical split is `forget10`):
   - The **knowledge bank** (`knowledge_banks/<split>/claims.json`) ships with
     this repo; it is the rubric the first-leak judge adjudicates against. Run
     `build_kb` → `resolve_kb` only to rebuild it from scratch.
   - Generate target CoTs, run **first-leak labeling**
     (`label_first_leak_rtofu`), and assemble the classifier dataset with the
     orchestrator (`build_first_leak_classifier_data`, which also builds its own
     clean-forget base and slices the forget05/01 subsets).
   - Train the **binary leak classifier** (`train_classifier --model_type
     contextual_binary`) and the **soft token** (`build_soft_token_data` →
     `train_soft_token --n_tokens 16`). These two are
     independent and can train in parallel. The canonical classifier is
     firstleak-trained (best of 3 seeds, selected by `gap_idk_robust_smoothed`),
     promoted to `leak_classifiers/<split>/contextual_labeled_full_contextual_binary`.
2. **Generate** with REDaCT (`generate_rtofu_responses` with
   `--leak_classifier_path` + `--soft_token_path`) on the forget, protocol
   retain, and aux splits → CSVs.
3. **Evaluate** (`judge_afe_rtofu` + `judge_cfe_rtofu` →
   `evaluate_rtofu_from_csv`), which emits the reported LLM-judge metrics:
   - **AFE_judge** — Answer Forget Efficacy, `1 − LEAK_rate` from a 4-way judge
     rubric `{LEAK > HALLUC > COLLAPSE > REFUSE}`.
   - **CFE_judge** — CoT Forget Efficacy, `1 − mean(graded judge score)` for how
     much the CoT reveals the gold answer.
   - **MU** — Model Utility, harmonic mean over 12 means ({ROUGE-L recall,
     cosine, entailment, token entropy} × {retain, real_authors, world_facts}).

`forget05` and `forget01` are derived from `forget10`'s artifacts by the subset
machinery rather than rebuilt from scratch — see
[PIPELINE.md](PIPELINE.md#subset-split-reuse-forget05--forget01-from-forget10).

## Baselines

- **R2MU** runs in this repo (`scripts/r2mu/`, method in
  `redact/baselines/r2mu.py`).
- **GA / GD / KL / IDK** are trained with the external R-TOFU trainer; the
  `config/rtofu_model_config/<name>.yaml` files ship with
  `/path/to/rtofu-baselines/<name>` placeholders — point `hf_name` at your
  trained checkpoint, then generate and score through the same pipeline as the
  main method.

See [PIPELINE.md](PIPELINE.md#baselines) for details.

## Pre-trained artifacts

Inference needs two artifacts per forget split: a **leak classifier** and a
**corrective soft token**.

**Soft tokens ship with this repository** at
`artifacts/soft_tokens/<split>/embedding.pt` — each is a single
`(16, 4096)` float32 tensor stored under the key `embedding` (263 KB).

**Leak classifiers are hosted separately** (~502 MB per split, too large to
vendor here):

```bash
hf download jmaccoby/redact-leak-classifier --local-dir leak_classifiers
```

This lays out `leak_classifiers/<split>/contextual_labeled_full_contextual_binary/`,
which is the path the pipeline expects.

## Results

REDaCT and baselines on three R-TOFU forget sets (`f01`=`forget01`,
`f05`=`forget05`, `f10`=`forget10`), against the frozen target model
`sangyon/LRM-target`. **AFE** and **CFE** are the LLM-as-judge forget-efficacy
metrics on the answer and the chain-of-thought respectively; **Utility** is the
harmonic mean over the 12 {metric x subset} means — ROUGE-L recall, cosine
similarity, entailment, and token entropy, each over retain, `real_authors`,
and `world_facts`. **Halluc.**, **Refusal**, and **Collapse** are the rates of the
corresponding LLM-as-judge labels among non-leak forget-set samples, and sum to
1 per row. Arrows give the preferred direction. Bold marks the best value per
column within each split; the pre-unlearning target is a reference row and is
excluded from the comparison.

| Method | Split | AFE ↑ | CFE ↑ | Utility ↑ | Halluc. ↓ | Refusal ↑ | Collapse ↓ |
|---|---|---|---|---|---|---|---|
| Target (pre-unlearning) | — | — | — | 0.787 | — | — | — |
| GA | f01 | 0.375 | 0.460 | 0.757 | 0.533 | 0.000 | 0.467 |
| GA | f05 | 0.245 | 0.343 | 0.766 | 0.939 | 0.000 | 0.061 |
| GA | f10 | 0.190 | 0.302 | 0.770 | 0.947 | 0.013 | 0.039 |
| GD | f01 | 0.250 | 0.347 | 0.729 | 1.000 | 0.000 | **0.000** |
| GD | f05 | 0.220 | 0.328 | 0.702 | 0.955 | 0.023 | 0.023 |
| GD | f10 | 0.103 | 0.155 | 0.717 | 0.951 | 0.000 | 0.049 |
| KL | f01 | 0.200 | 0.346 | 0.734 | 0.875 | 0.125 | **0.000** |
| KL | f05 | 0.125 | 0.191 | 0.751 | 0.960 | 0.040 | **0.000** |
| KL | f10 | 0.230 | 0.368 | 0.745 | 0.924 | 0.033 | 0.043 |
| IDK (PO) | f01 | 0.675 | 0.768 | 0.631 | 0.185 | 0.741 | 0.074 |
| IDK (PO) | f05 | 0.320 | 0.333 | 0.657 | 0.391 | 0.125 | 0.484 |
| IDK (PO) | f10 | 0.165 | 0.196 | 0.660 | 0.394 | 0.424 | 0.182 |
| R²MU | f01 | **1.000** | **1.000** | 0.585 | 0.025 | 0.025 | 0.950 |
| R²MU | f05 | **1.000** | **0.999** | 0.566 | **0.000** | 0.000 | 1.000 |
| R²MU | f10 | **1.000** | **0.999** | 0.549 | 0.003 | 0.005 | 0.993 |
| **REDaCT (ours)** | f01 | **1.000** | **1.000** | **0.771** | **0.000** | **1.000** | **0.000** |
| **REDaCT (ours)** | f05 | **1.000** | 0.977 | **0.782** | **0.000** | **1.000** | **0.000** |
| **REDaCT (ours)** | f10 | **1.000** | 0.983 | **0.786** | **0.000** | **1.000** | **0.000** |

R²MU matches REDaCT on forget efficacy but does so by collapsing the model —
its collapse rate is 0.950–0.993 and utility falls to 0.549–0.585, while REDaCT
answers by refusing in every case and stays within 0.001 of the pre-unlearning
target's utility on `forget10`.

## Citation

```bibtex
@inproceedings{maccoby2026redact,
  title     = {{REDaCT}: Collapse-Free Unlearning in Large Reasoning Models},
  author    = {Maccoby, Joshua and Wang, Wendy Hui},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

## License

Released under the MIT License — see [LICENSE](LICENSE).
