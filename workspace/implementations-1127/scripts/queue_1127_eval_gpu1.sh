#!/usr/bin/env bash
# PHASE 3 queue, scheme B: after the 1127 training queue finishes, evaluate
# the 20 decisive adapters with Hy4 AFE/CFE. The primary seed 1001 gets the
# full 4-split + MU treatment; the secondary seed 2002 is a forget10-only
# AFE/CFE spot-check (no MU), which avoids 10 x 617 extra generations.
#
# Pipelined: each run = GPU stage (merge + generation, owns GPU1) followed by
# a CPU stage (Hy4 AFE/CFE + MU, CUDA_VISIBLE_DEVICES=""). As soon as a run's
# GPU stage finishes, the next run's generation starts immediately while this
# run's CPU stage runs in the background. MAX_CPU_JOBS bounds concurrent Hy4
# request bursts so the endpoint is not hammered.
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
MAX_CPU_JOBS=2
CPU_PIDS=()
CPU_RUNS=()
CPU_MODES=()

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
  echo "PIPELINE: GPU stage (merge+generate) serial on GPU1; CPU stage (Hy4+MU, CUDA_VISIBLE_DEVICES=\"\") background, max $MAX_CPU_JOBS concurrent"
  for slug in $SLUGS; do
    echo "  [full]       gpu+cpu unl1127_${slug}_decisive_s1001 $EVAL_ROOT/unl1127_${slug}_decisive_s1001"
  done
  for slug in $SLUGS; do
    echo "  [forget-only] gpu+cpu unl1127_${slug}_decisive_s2002 $EVAL_ROOT/unl1127_${slug}_decisive_s2002 --forget-only"
  done
  echo "=== dry-run complete; nothing launched ==="
  exit 0
fi

: > "$LOG/status.tsv"
log "=== EVAL QUEUE START pid=$$ ==="
wait_gate || exit $?

CPU_N=0
REAP_IDX=0
CPU_OUTS=()

# The CPU stage is fired in the background so GPU1 is free for the NEXT run's
# generation immediately. It hides all GPUs (see eval_one_1127.sh --stage cpu).
launch_cpu() {
  local run=$1 mode=$2 out=$3
  local -a flags=()
  [ "$mode" = "forget-only" ] && flags+=(--forget-only)
  (
    bash "$ROOT/scripts/eval_one_1127.sh" "$run" "$out" --stage cpu "${flags[@]}" \
      > "$LOG/${run}.cpu.log" 2>&1
    echo $? > "$out/cpu_rc"
    touch "$out/cpu_done"
  ) &
  CPU_PIDS[$CPU_N]=$!
  CPU_RUNS[$CPU_N]="$run"
  CPU_MODES[$CPU_N]="$mode"
  CPU_OUTS[$CPU_N]="$out"
  CPU_N=$((CPU_N + 1))
  log "CPU STAGE LAUNCHED $run (in flight=$((CPU_N - REAP_IDX)))"
  # First reap everything already finished (non-blocking). If the endpoint
  # is slow and CPU stages pile up past the cap, wait for the oldest one as
  # backpressure; in the normal case CPU stages are far shorter than a GPU
  # stage (~2.7h full / ~1h forget-only), so this wait returns immediately.
  reap_finished
  while [ $((CPU_N - REAP_IDX)) -gt "$MAX_CPU_JOBS" ]; do
    log "backpressure: $((CPU_N - REAP_IDX)) CPU stages in flight (cap $MAX_CPU_JOBS)"
    reap_cpu
  done
}

# Non-blocking, order-preserving reap: a CPU stage signals completion with a
# cpu_done marker so we can tell "finished" from "still running" without wait.
reap_finished() {
  while [ "$REAP_IDX" -lt "$CPU_N" ] && \
        [ -f "${CPU_OUTS[$REAP_IDX]}/cpu_done" ]; do
    reap_cpu
  done
}

reap_cpu() {
  local i=$REAP_IDX pid run mode out rc status
  pid=${CPU_PIDS[$i]}; run=${CPU_RUNS[$i]}
  mode=${CPU_MODES[$i]}; out=${CPU_OUTS[$i]}
  wait "$pid" 2>/dev/null || true
  rc=$(cat "$out/cpu_rc" 2>/dev/null || echo "?")
  if [ -s "$out/eval.json" ]; then status=DONE; else status=INCOMPLETE; fi
  printf '%s\t%s\tmode=%s\tcpu_rc=%s\n' "$run" "$status" "$mode" "$rc" \
    >> "$LOG/status.tsv"
  log "CPU STAGE DONE $run status=$status mode=$mode rc=$rc"
  REAP_IDX=$((REAP_IDX + 1))
}

gpu_stage() {
  local slug=$1 seed=$2 mode=$3
  local run="unl1127_${slug}_decisive_s${seed}"
  local out="$EVAL_ROOT/$run"
  local -a flags=()
  [ "$mode" = "forget-only" ] && flags+=(--forget-only)
  mkdir -p "$out"
  local t0 rc
  t0=$(date +%s)
  log "GPU STAGE START $run mode=$mode"
  bash "$ROOT/scripts/eval_one_1127.sh" "$run" "$out" --stage gpu "${flags[@]}" \
    2>&1 | tee -a "$LOG/${run}.gpu.log"
  rc=${PIPESTATUS[0]}
  log "GPU STAGE DONE $run rc=$rc elapsed_s=$(( $(date +%s) - t0 ))"
  if [ "$rc" -ne 0 ] || [ ! -s "$out/forget10.csv" ]; then
    printf '%s\tGPU_FAILED\tmode=%s\trc=%s\n' "$run" "$mode" "$rc" >> "$LOG/status.tsv"
    log "GPU STAGE FAILED $run; skipping its CPU stage"
    return 0
  fi
  launch_cpu "$run" "$mode" "$out"
}

log "=== PHASE 3 EVAL START (scheme B, pipelined) ==="
log "stage A: 10 full 4-split+MU runs on primary seed $PRIMARY_SEED"
for slug in $SLUGS; do
  gpu_stage "$slug" "$PRIMARY_SEED" full
done
log "stage B: 10 forget-only AFE/CFE spot checks on secondary seed $SPOT_SEED"
for slug in $SLUGS; do
  gpu_stage "$slug" "$SPOT_SEED" forget-only
done

log "=== all GPU stages done; draining CPU stages ==="
while [ "$REAP_IDX" -lt "$CPU_N" ]; do
  if [ ! -f "${CPU_OUTS[$REAP_IDX]}/cpu_done" ]; then
    wait "${CPU_PIDS[$REAP_IDX]}" 2>/dev/null || true
  fi
  reap_cpu
done

log "=== PHASE 3 ALL DONE ==="
