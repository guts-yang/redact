#!/usr/bin/env bash
# Eval for ONE 1127 decisive run: merge adapter -> generation -> Hy4 AFE/CFE
# judge -> optional MU aggregate.
#
# Usage:
#   scripts/eval_one_1127.sh <run_id> <out_dir>              # full: 4 splits + MU
#   scripts/eval_one_1127.sh <run_id> <out_dir> --forget-only # forget10 + AFE/CFE only
#   scripts/eval_one_1127.sh <run_id> <out_dir> --skip-generate
#
# --forget-only is the PHASE 3 "scheme B" spot-check for the secondary seed:
# it generates only forget10 (400 rows) and emits AFE_judge / CFE_judge with
# no MU, since MU would require 617 extra retain/aux generations.
#
# <run_id> is e.g. unl1127_budget_decisive_s1001. Its adapter lives at
# $UNL1127_RESULTS/<run_id>/adapter. A per-run model YAML is written into
# <out_dir>/config so the shared config/rtofu_model_config/ (read by the UPBD
# chain) is never modified.
set -uo pipefail
if [ $# -lt 2 ]; then
  echo "usage: $0 <run_id> <out_dir> [--forget-only] [--skip-generate]" >&2
  exit 2
fi
RUN_ID=$1
OUT=$2
shift 2
FORGET_ONLY=0
SKIP_GENERATE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --forget-only) FORGET_ONLY=1; shift ;;
    --skip-generate) SKIP_GENERATE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPO=/usr/local/workspace/tracks/lrm-unlearning/redact
export HF_HOME=/jizhicfs/chenyliao/cache/huggingface
export TRANSFORMERS_OFFLINE=1
RESULTS=${UNL1127_RESULTS:-/usr/local/workspace/tracks/lrm-unlearning/R-TOFU/workspace/results/unl1127}
TOKENIZER=/jizhicfs/chenyliao/models/sangyon/LRM-target
mkdir -p "$OUT/config" "$OUT/logs"

ADAPTER="$RESULTS/$RUN_ID/adapter"
MERGED="$RESULTS/$RUN_ID/merged"

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
  prompt_suffix: "<｜Assistant｜>"
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
fi

echo "=== HY4 AFE $RUN_ID $(date -Iseconds) ==="
python -m scripts.hy4_judge_afe_rtofu \
  --csv "$OUT/forget10.csv" --split forget10 \
  --output_jsonl "$OUT/afe.hy4.jsonl" \
  --workers 4 --max_retries 2 --timeout 180 \
  2>&1 | tee -a "$OUT/logs/afe.log"

echo "=== HY4 CFE $RUN_ID $(date -Iseconds) ==="
python -m scripts.hy4_judge_cfe_rtofu \
  --csv "$OUT/forget10.csv" --split forget10 \
  --output_jsonl "$OUT/cfe.hy4.jsonl" \
  --workers 4 --max_retries 2 --timeout 180 \
  2>&1 | tee -a "$OUT/logs/cfe.log"

if [ "$FORGET_ONLY" -eq 1 ]; then
  echo "=== AFE/CFE ONLY (no MU) $RUN_ID $(date -Iseconds) ==="
  python -m scripts.evaluate_rtofu_from_csv \
    --csv "$OUT/forget10.csv" --split forget10 \
    --judge_afe_scores "$OUT/afe.hy4.jsonl" \
    --rtofu_judge_scores "$OUT/cfe.hy4.jsonl" \
    --output_json "$OUT/eval.json" \
    2>&1 | tee -a "$OUT/logs/afe_cfe.log"
else
  echo "=== MU $RUN_ID $(date -Iseconds) ==="
  python -m scripts.evaluate_rtofu_from_csv \
    --csv "$OUT/forget10.csv" --split forget10 \
    --judge_afe_scores "$OUT/afe.hy4.jsonl" \
    --rtofu_judge_scores "$OUT/cfe.hy4.jsonl" \
    --mu_retain_csv "$OUT/forget10_retain.csv" \
    --mu_real_authors_csv "$OUT/real_authors.csv" \
    --mu_world_facts_csv "$OUT/world_facts.csv" \
    --output_json "$OUT/eval.json" \
    2>&1 | tee -a "$OUT/logs/mu.log"
fi

echo "=== EVAL DONE $RUN_ID mode=$([ "$FORGET_ONLY" -eq 1 ] && echo forget_only || echo full) ==="
