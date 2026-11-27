#!/usr/bin/env bash
# PHASE 3 queue, scheme B: after the 1127 training queue finishes, evaluate
# the 20 decisive adapters with Hy4 AFE/CFE. The primary seed 1001 gets the
# full 4-split + MU treatment; the secondary seed 2002 is a forget10-only
# AFE/CFE spot-check (no MU), which avoids 10 x 617 extra generations.
#
# Usage:
#   bash scripts/queue_1127_eval_gpu1.sh --dry-run   # print plan, start nothing
#   bash scripts/queue_1127_eval_gpu1.sh             # wait, then run all 20
#
# Gate (dependency, not a preference): the training queue MUST have written
# "QUEUE ALL DONE" to its queue.log, otherwise adapters do not exist yet.
# Additionally GPU1 must be free (3 consecutive 60s polls), because generation
# uses fp32 8B and must not contend with any remaining UPBD-chain job.
#
# This script is intentionally separate from queue_1127_gpu1.sh: that file is
# already running in the background and must not be edited while bash is
# reading it.
set -uo pipefail

ROOT=/usr/local/workspace/tracks/lrm-unlearning/redact/workspace/implementations-1127
cd "$ROOT" || exit 1

export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/jizhicfs/chenyliao/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export UNL1127_ASU_ROOT=/jizhicfs/chenyliao
export UNL1127_RTOFU_REPO=/usr/local/workspace/tracks/lrm-unlearning/R-TOFU
export UNL1127_REDACT_ROOT=/usr/local/workspace/tracks/lrm-unlearning/redact
export UNL1127_WORKSPACE=/usr/local/workspace/tracks/lrm-unlearning/R-TOFU/workspace
export UNL1127_RESULTS="$UNL1127_WORKSPACE/results/unl1127"
export UNL1127_LOGS="$UNL1127_WORKSPACE/logs/unl1127"
export UNL1127_DTYPE=fp32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_LOG="$UNL1127_LOGS/queue_1127_gpu1/queue.log"
LOG="$UNL1127_LOGS/queue_1127_eval_gpu1"
EVAL_ROOT="$UNL1127_RESULTS/eval1127"
mkdir -p "$LOG" "$EVAL_ROOT"

SLUGS="budget pivot consensus drift reverse faith graph attrib layer probe"
PRIMARY_SEED=1001
SPOT_SEED=2002
MAX_WAIT_SEC=$((40 * 3600))
DEBOUNCE=3

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG/queue.log"; }

gpu1_used() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v g=1 '$1==g {gsub(/ /,"",$2); print $2}'
}

wait_gate() {
  local ok=0 start=$SECONDS marker=0
  while true; do
    if grep -q 'QUEUE ALL DONE' "$TRAIN_LOG" 2>/dev/null; then marker=1; fi
    local used
    used=$(gpu1_used); used=${used:-0}
    if [ "$marker" -eq 1 ] && [ "$used" -lt 3000 ]; then
      ok=$((ok + 1))
      log "gate poll ${ok}/${DEBOUNCE}: training_done=1 GPU1_used=${used}MiB"
    else
      ok=0
      log "gate waiting: training_done=$marker GPU1_used=${used}MiB; sleeping 60s"
    fi
    if [ "$ok" -ge "$DEBOUNCE" ]; then
      log "GATE RELEASED: training queue finished and GPU1 is free"
      return 0
    fi
    if [ $((SECONDS - start)) -ge "$MAX_WAIT_SEC" ]; then
      log "QUEUE_TIMEOUT: gate not satisfied within 40h; no eval started"
      return 3
    fi
    sleep 60
  done
}

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [ "$DRY_RUN" -eq 1 ]; then
  echo "=== queue_1127_eval_gpu1.sh DRY RUN (no wait, no GPU work) ==="
  echo "ROOT=$ROOT"
  echo "TRAIN_LOG=$TRAIN_LOG (wait for 'QUEUE ALL DONE')"
  echo "GATE: training_done marker + GPU1 memory.used<3000MiB for 3 consecutive 60s polls; max wait 40h"
  echo "EVAL_ROOT=$EVAL_ROOT"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "JOBS: 10 methods x seed 1001 full(4-split+MU) + 10 methods x seed 2002 forget-only(AFE/CFE)"
  for slug in $SLUGS; do
    echo "  [full]       scripts/eval_one_1127.sh unl1127_${slug}_decisive_s1001 $EVAL_ROOT/unl1127_${slug}_decisive_s1001"
  done
  for slug in $SLUGS; do
    echo "  [forget-only] scripts/eval_one_1127.sh unl1127_${slug}_decisive_s2002 $EVAL_ROOT/unl1127_${slug}_decisive_s2002 --forget-only"
  done
  echo "=== dry-run complete; nothing launched ==="
  exit 0
fi

: > "$LOG/status.tsv"
log "=== EVAL QUEUE START pid=$$ ==="
wait_gate || exit $?

eval_run() {
  local slug=$1 seed=$2 mode=$3
  local run="unl1127_${slug}_decisive_s${seed}"
  local out="$EVAL_ROOT/$run"
  local -a flags=()
  [ "$mode" = "forget-only" ] && flags+=(--forget-only)
  mkdir -p "$out"
  log "EVAL START $run mode=$mode"
  local t0 rc status
  t0=$(date +%s)
  bash "$ROOT/scripts/eval_one_1127.sh" "$run" "$out" "${flags[@]}" \
    2>&1 | tee -a "$LOG/${run}.log"
  rc=${PIPESTATUS[0]}
  if [ -s "$out/eval.json" ]; then status=DONE; else status=INCOMPLETE; fi
  printf '%s\t%s\tmode=%s\trc=%s\telapsed_s=%s\n' \
    "$run" "$status" "$mode" "$rc" "$(( $(date +%s) - t0 ))" >> "$LOG/status.tsv"
  log "EVAL DONE $run status=$status mode=$mode rc=$rc elapsed_s=$(( $(date +%s) - t0 ))"
}

log "=== PHASE 3 EVAL START (scheme B) ==="
log "stage A: 10 full 4-split+MU runs on primary seed $PRIMARY_SEED"
for slug in $SLUGS; do
  eval_run "$slug" "$PRIMARY_SEED" full
done
log "stage B: 10 forget-only AFE/CFE spot checks on secondary seed $SPOT_SEED"
for slug in $SLUGS; do
  eval_run "$slug" "$SPOT_SEED" forget-only
done

log "=== PHASE 3 ALL DONE ==="
