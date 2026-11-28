#!/usr/bin/env bash
# Eval for ONE 1127 decisive run, split into a GPU stage and a CPU stage.
#
# Usage:
#   scripts/eval_one_1127.sh <run_id> <out_dir>              # both stages, serial
#   scripts/eval_one_1127.sh <run_id> <out_dir> --stage gpu  # merge + generation only
#   scripts/eval_one_1127.sh <run_id> <out_dir> --stage cpu  # Hy4 AFE/CFE (+MU), no GPU
#
# --stage gpu owns GPU1 (fp32 8B generation). --stage cpu sets
# CUDA_VISIBLE_DEVICES="" so the MU evaluators (cosine_similarity.py /
# entailment_score.py otherwise pick cuda) stay on CPU; Hy4 judging is pure
# HTTP. The queue pipelines them: as soon as one run's gpu stage ends, the next
# run's generation starts while this run's cpu stage runs in the background.
#
# Other flags:
#   --forget-only   secondary-seed spot-check (forget10 + AFE/CFE, no MU)
#   --skip-generate skip the generation part of the gpu stage
#
# <run_id> is e.g. unl1127_budget_decisive_s1001. Its adapter lives at
# $UNL1127_RESULTS/<run_id>/adapter. A per-run model YAML is written into
# <out_dir>/config so the shared config/rtofu_model_config/ (read by the UPBD
# chain) is never modified.
set -uo pipefail
if [ $# -lt 2 ]; then
  echo "usage: $0 <run_id> <out_dir> [--stage gpu|cpu|all] [--forget-only] [--skip-generate]" >&2
  exit 2
fi
RUN_ID=$1
OUT=$2
shift 2
FORGET_ONLY=0
SKIP_GENERATE=0
STAGE=all
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE=$2; shift 2 ;;
    --forget-only) FORGET_ONLY=1; shift ;;
    --skip-generate) SKIP_GENERATE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
case "$STAGE" in
  gpu|cpu|all) ;;
  *) echo "invalid --stage: $STAGE (use gpu|cpu|all)" >&2; exit 2 ;;
esac

REPO=/usr/local/workspace/tracks/lrm-unlearning/redact
export HF_HOME=/jizhicfs/chenyliao/cache/huggingface
export TRANSFORMERS_OFFLINE=1
RESULTS=${UNL1127_RESULTS:-/usr/local/workspace/tracks/lrm-unlearning/R-TOFU/workspace/results/unl1127}
TOKENIZER=/jizhicfs/chenyliao/models/sangyon/LRM-target
mkdir -p "$OUT/config" "$OUT/logs"

ADAPTER="$RESULTS/$RUN_ID/adapter"
MERGED="$RESULTS/$RUN_ID/merged"

if [ "$STAGE" != "cpu" ]; then
echo "=== MERGE $RUN_ID $(date -Iseconds) ==="
python "$REPO/workspace/implementations-1127/scripts/merge_1127_adapter.py" \
  --adapter "$ADAPTER" --output "$MERGED" --base "$TOKENIZER" \
  2>&1 | tee -a "$OUT/logs/merge.log"
MERGE_RC=${PIPESTATUS[0]}
if [ "$MERGE_RC" -ne 0 ]; then
  echo "=== MERGE_FAILED $RUN_ID rc=$MERGE_RC ==="
  exit 4
fi

# Per-run YAML: keeps config/rtofu_model_config/ untouched.
cat > "$OUT/config/$RUN_ID.yaml" <<EOF
embedding_dim: 4096
embedding_module: model.embed_tokens
formatting_tokens:
  answer_prefix: ""
  answer_suffix: ""
  prompt_prefix: "<｜User｜>"
  prompt_suffix: "<｜Assistant｜><think>\n"
hf_name: $MERGED
tokenizer_name: $MERGED
attn_implementation: sdpa
torch_dtype: float32
load_in_8bit: false
load_in_4bit: false
baseline: true
EOF

cd "$REPO" || exit 1

if [ "$SKIP_GENERATE" -eq 0 ]; then
  if [ "$FORGET_ONLY" -eq 1 ]; then
    SPLITS="forget10"
  else
    SPLITS="forget10 forget10_retain real_authors world_facts"
  fi
  for SPLIT in $SPLITS; do
    echo "=== GENERATE $RUN_ID $SPLIT $(date -Iseconds) ==="
    T0=$(date +%s)
    python -m scripts.generate_rtofu_responses \
      --model_name "$RUN_ID" \
      --model_config_dir "$OUT/config" \
      --split "$SPLIT" \
      --output "$OUT/$SPLIT.csv" \
      --max_new_tokens 1024 --batch_size 4 \
      2>&1 | tee -a "$OUT/logs/generate_$SPLIT.log"
    RC=${PIPESTATUS[0]}
    echo "=== GENERATE DONE $SPLIT rc=$RC elapsed_s=$(( $(date +%s) - T0 )) ==="
  done

  # 1127 models are trained ANSWER-ONLY: training targets (refusals / gold
  # answers) never contain `</think>` (see runtime/runner.py make_batch and
  # data/idontknow.jsonl). Consequently split_cot_answer puts the whole
  # continuation into `model_cot` and leaves `response` empty, which makes the
  # AFE judge label every row COLLAPSE and MU compare against empty strings --
  # measurement artifacts, not model behaviour. Verified 2026-11-28 with a
  # 2-row smoke using the training-aligned `<think>\n` prompt: response still
  # empty. Normalize by treating the full continuation as the answer, keep the
  # raw CSV for provenance, and record that CFE (CoT leakage) is NOT applicable
  # to models that emit no CoT.
  python - "$OUT" $SPLITS <<'PY'
import csv, json, sys
from pathlib import Path
out = Path(sys.argv[1]); splits = sys.argv[2:]
report = {"splits": {}}
for s in splits:
    p = out / f"{s}.csv"
    if not p.is_file():
        continue
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    if not rows:
        continue
    fields = list(rows[0].keys())
    (out / f"{s}.raw.csv").write_bytes(p.read_bytes())
    n = 0
    for r in rows:
        cot = (r.get("model_cot") or "").strip()
        if not (r.get("response") or "").strip() and cot:
            r["response"] = cot
            n += 1
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    report["splits"][s] = {"rows": len(rows), "response_filled": n}
(out / "answer_only_protocol.json").write_text(json.dumps({
    "protocol": "answer_only_full_continuation",
    "reason": "1127 training targets contain no </think>; models emit a short answer directly. The full continuation is the answer, not a CoT.",
    "raw_csv_suffix": ".raw.csv",
    "cfe_applicable": False,
    "cfe_note": "CFE_judge measures CoT leakage; these models emit no CoT, so CFE is not computed and must not be compared with UPBD/target CFE values.",
    "claim_level": "non-claim / diagnostic_only",
    **report
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[normalize] {json.dumps(report, ensure_ascii=False)}")
PY
fi
echo "=== GPU STAGE DONE $RUN_ID $(date -Iseconds) ==="
fi

if [ "$STAGE" != "gpu" ]; then
# Hy4 judging is pure HTTP; the MU evaluators default to cuda when visible, so
# hide all GPUs for this stage to keep the pipeline's GPU1 slot free.
export CUDA_VISIBLE_DEVICES=""
cd "$REPO" || exit 1
echo "=== CPU STAGE START $RUN_ID $(date -Iseconds) ==="

echo "=== HY4 AFE $RUN_ID $(date -Iseconds) ==="
python -m scripts.hy4_judge_afe_rtofu \
  --csv "$OUT/forget10.csv" --split forget10 \
  --output_jsonl "$OUT/afe.hy4.jsonl" \
  --workers 4 --max_retries 2 --timeout 180 \
  2>&1 | tee -a "$OUT/logs/afe.log"

# CFE is deliberately NOT run: 1127 checkpoints are answer-only (training
# targets contain no </think>), so there is no CoT to score. Emitting
# CFE_judge=1.0 for an empty CoT would be a fabricated "zero CoT leakage"
# result. See answer_only_protocol.json.
echo "=== ENSURE AFE $RUN_ID (CFE N/A: answer-only model) ==="
python -m scripts.ensure_hy4_judges \
  --csv "$OUT/forget10.csv" --split forget10 \
  --afe "$OUT/afe.hy4.jsonl" \
  --rounds 3 --workers 4 --timeout 180 \
  2>&1 | tee -a "$OUT/logs/ensure.log"
ENSURE_RC=${PIPESTATUS[0]}
if [ "$ENSURE_RC" -ne 0 ]; then
  echo "=== CPU STAGE FAILED $RUN_ID (incomplete Hy4 judges) ==="
  exit 5
fi

if [ "$FORGET_ONLY" -eq 1 ]; then
  echo "=== AFE ONLY (no MU, no CFE) $RUN_ID $(date -Iseconds) ==="
  python -m scripts.evaluate_rtofu_from_csv \
    --csv "$OUT/forget10.csv" --split forget10 \
    --judge_afe_scores "$OUT/afe.hy4.jsonl" \
    --output_json "$OUT/eval.json" \
    2>&1 | tee -a "$OUT/logs/afe_only.log"
  AGG_RC=${PIPESTATUS[0]}
else
  echo "=== AFE + MU (no CFE) $RUN_ID $(date -Iseconds) ==="
  python -m scripts.evaluate_rtofu_from_csv \
    --csv "$OUT/forget10.csv" --split forget10 \
    --judge_afe_scores "$OUT/afe.hy4.jsonl" \
    --mu_retain_csv "$OUT/forget10_retain.csv" \
    --mu_real_authors_csv "$OUT/real_authors.csv" \
    --mu_world_facts_csv "$OUT/world_facts.csv" \
    --output_json "$OUT/eval.json" \
    2>&1 | tee -a "$OUT/logs/afe_mu.log"
  AGG_RC=${PIPESTATUS[0]}
fi
if [ "$AGG_RC" -ne 0 ] || [ ! -s "$OUT/eval.json" ]; then
  echo "=== CPU STAGE FAILED $RUN_ID (aggregate rc=$AGG_RC) ==="
  exit 6
fi

echo "=== CPU STAGE DONE $RUN_ID $(date -Iseconds) ==="
fi

echo "=== EVAL DONE $RUN_ID stage=$STAGE mode=$([ "$FORGET_ONLY" -eq 1 ] && echo forget_only || echo full) ==="
