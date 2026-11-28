#!/bin/bash
# End-to-end eval for one model: 4-split generation -> Hy4 AFE/CFE -> MU aggregate.
#
# Usage:
#   scripts/upbd/eval_one_model.sh <model_name> <out_dir> [--skip-generate]
#
# <model_name> is a YAML basename under $MODEL_CONFIG_DIR (default:
# config/rtofu_model_config). Output CSVs, judge JSONLs, and eval.json all land
# in <out_dir>. Hy4 is the only judge/review backend used here.
set -u
if [ $# -lt 2 ]; then
  echo "usage: $0 <model_name> <out_dir> [--skip-generate]" >&2
  exit 2
fi
MODEL_NAME=$1
OUT=$2
shift 2

SKIP_GENERATE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-generate) SKIP_GENERATE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd /usr/local/workspace/tracks/lrm-unlearning/redact
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/jizhicfs/chenyliao/cache/huggingface
export TRANSFORMERS_OFFLINE=1
MODEL_CONFIG_DIR=${MODEL_CONFIG_DIR:-config/rtofu_model_config}
mkdir -p "$OUT"

if [ "$SKIP_GENERATE" -eq 0 ]; then
  for SPLIT in forget10 forget10_retain real_authors world_facts; do
    echo "=== GENERATE $MODEL_NAME $SPLIT $(date -Iseconds) ==="
    T0=$(date +%s)
    python -m scripts.generate_rtofu_responses \
      --model_name "$MODEL_NAME" \
      --model_config_dir "$MODEL_CONFIG_DIR" \
      --split "$SPLIT" \
      --output "$OUT/$SPLIT.csv" \
      --max_new_tokens 1024 --batch_size 4
    RC=$?
    T1=$(date +%s)
    echo "=== GENERATE DONE $SPLIT rc=$RC elapsed_s=$((T1-T0)) ==="
  done
fi

echo "=== HY4 AFE $MODEL_NAME $(date -Iseconds) ==="
python -m scripts.hy4_judge_afe_rtofu \
  --csv "$OUT/forget10.csv" --split forget10 \
  --output_jsonl "$OUT/afe.hy4.jsonl" \
  --workers 4 --max_retries 2 --timeout 180

echo "=== HY4 CFE $MODEL_NAME $(date -Iseconds) ==="
python -m scripts.hy4_judge_cfe_rtofu \
  --csv "$OUT/forget10.csv" --split forget10 \
  --output_jsonl "$OUT/cfe.hy4.jsonl" \
  --workers 4 --max_retries 2 --timeout 180

# Hy4 intermittently returns HTTP 500 / times out. Retry unresolved uids and
# refuse to aggregate until both JSONLs cover every CSV row; otherwise the
# evaluator raises and no eval.json is written while the chain keeps going.
echo "=== ENSURE JUDGES $MODEL_NAME $(date -Iseconds) ==="
python -m scripts.ensure_hy4_judges \
  --csv "$OUT/forget10.csv" --split forget10 \
  --afe "$OUT/afe.hy4.jsonl" --cfe "$OUT/cfe.hy4.jsonl" \
  --rounds 3 --workers 4 --timeout 180
ENSURE_RC=$?
if [ "$ENSURE_RC" -ne 0 ]; then
  echo "=== EVAL_FAILED $MODEL_NAME (incomplete Hy4 judges) ==="
  exit 5
fi

echo "=== AGGREGATE $MODEL_NAME $(date -Iseconds) ==="
python -m scripts.evaluate_rtofu_from_csv \
  --csv "$OUT/forget10.csv" --split forget10 \
  --judge_afe_scores "$OUT/afe.hy4.jsonl" \
  --rtofu_judge_scores "$OUT/cfe.hy4.jsonl" \
  --mu_retain_csv "$OUT/forget10_retain.csv" \
  --mu_real_authors_csv "$OUT/real_authors.csv" \
  --mu_world_facts_csv "$OUT/world_facts.csv" \
  --output_json "$OUT/eval.json" \
  2>&1 | tee -a "$OUT/logs/aggregate.log"
AGG_RC=${PIPESTATUS[0]}
if [ "$AGG_RC" -ne 0 ] || [ ! -s "$OUT/eval.json" ]; then
  echo "=== EVAL_FAILED $MODEL_NAME (aggregate rc=$AGG_RC) ==="
  exit 6
fi
echo "=== ALL DONE $MODEL_NAME $(date -Iseconds) ==="
