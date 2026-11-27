#!/bin/bash
# UPBD full-split reproduction (400 forget + 400 retain + 217 aux).
#
# Protocol: frozen R2MU forget10 checkpoint, LoRA r=8, 1 epoch, fp32,
# holdout=0 (train on all rows), seeds 1001 and 2002. Generation/eval are
# separate steps against the merged checkpoint.
#
# If the in-process fp32 merge OOMs, add --skip-merge and merge offline; the
# adapter is still written under $OUT/adapter_run/adapter.
set -u
cd /usr/local/workspace/tracks/lrm-unlearning/redact
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/jizhicfs/chenyliao/cache/huggingface
export TRANSFORMERS_OFFLINE=1

ROOT=/jizhicfs/chenyliao/lrm-unlearning/results
MODEL=/jizhicfs/chenyliao/checkpoints/r2mu-forget10/checkpoint-last

# Target-behavior CSVs must come from the transformers-4.43 regeneration
# (eval_tf443); the older eval/ dir has degenerate CoT from tf 4.48.2.
TARGET_EVAL=${UNL_UPBD_TARGET_EVAL:-/jizhicfs/chenyliao/runs/target/forget10/eval_tf443}

for SEED in 1001 2002; do
  OUT="$ROOT/upbd_full_s${SEED}"
  mkdir -p "$OUT"
  echo "=== UPBD full seed=$SEED start $(date -Iseconds) ==="
  T0=$(date +%s)
  python -m scripts.upbd.train_upbd \
    --model "$MODEL" \
    --forget-csv /jizhicfs/chenyliao/runs/r2mu/forget10/eval/forget10.csv \
    --retain-csv "$TARGET_EVAL/forget10_retain.csv" \
    --authors-csv "$TARGET_EVAL/real_authors.csv" \
    --world-csv "$TARGET_EVAL/world_facts.csv" \
    --seed "$SEED" \
    --holdout 0 \
    --epochs 1 \
    --lr 2e-4 \
    --batch-size 1 \
    --grad-accum 4 \
    --max-len 1024 \
    --no-tf32 \
    --max-minutes 180 \
    --output-dir "$OUT/adapter_run" \
    --save-merged "$OUT/merged"
  RC=$?
  T1=$(date +%s)
  echo "=== UPBD full seed=$SEED done rc=$RC elapsed_s=$((T1-T0)) $(date -Iseconds) ==="
done
echo "=== UPBD full ALL DONE $(date -Iseconds) ==="
