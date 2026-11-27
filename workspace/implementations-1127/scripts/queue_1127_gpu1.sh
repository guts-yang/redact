#!/usr/bin/env bash
# GPU1 queue for the ten 1127 methods.
#
# Usage:
#   bash scripts/queue_1127_gpu1.sh --dry-run   # print gate/env/commands, start nothing
#   bash scripts/queue_1127_gpu1.sh             # wait for GPU1, then smoke -> decisive
#
# The gate is GPU1-only (per project decision): 3 consecutive 60s polls with
# memory.used < 3000 MiB and no compute app on GPU1. It deliberately does NOT
# check the UPBD chain's completion marker, so there is a residual risk of
# starting during a short idle gap between UPBD stages.
#
# All training output is diagnostic_only / non-claim. --decisive is a larger
# training run, NOT a T0-T3 result (eval_decisive.py does not exist yet).
set -uo pipefail

ROOT=/usr/local/workspace/tracks/lrm-unlearning/redact/workspace/implementations-1127
cd "$ROOT" || exit 1

# ---- Local R-TOFU cluster layout (the cluster prompt's /usr/local/workspace/R-TOFU
#      does not exist on this machine). ----
export CUDA_VISIBLE_DEVICES=1
export UNL1127_ASU_ROOT=/jizhicfs/chenyliao
export UNL1127_RTOFU_REPO=/usr/local/workspace/tracks/lrm-unlearning/R-TOFU
export UNL1127_REDACT_ROOT=/usr/local/workspace/tracks/lrm-unlearning/redact
export UNL1127_WORKSPACE=/usr/local/workspace/tracks/lrm-unlearning/R-TOFU/workspace
export UNL1127_RESULTS="$UNL1127_WORKSPACE/results/unl1127"
export UNL1127_LOGS="$UNL1127_WORKSPACE/logs/unl1127"
export UNL1127_TARGET_MODEL="$UNL1127_ASU_ROOT/models/sangyon/LRM-target"
export UNL1127_FORGET_FILE="$UNL1127_ASU_ROOT/data/rtofu/forget10.json"
export UNL1127_RETAIN_FILE="$UNL1127_ASU_ROOT/data/rtofu/retain90.json"
export UNL1127_REFUSAL_FILE="$UNL1127_RTOFU_REPO/data/idontknow.jsonl"
export UNL1127_UNLEARNED_ADAPTER=/jizhicfs/chenyliao/lrm-unlearning/checkpoints/r2mu-forget10/checkpoint-last
export UNL1127_DTYPE=fp32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

LOG="$UNL1127_LOGS/queue_1127_gpu1"
mkdir -p "$LOG" "$UNL1127_RESULTS"

SLUGS="budget pivot consensus drift reverse faith graph attrib layer probe"
SEEDS="1001 2002"
MAX_WAIT_SEC=$((30 * 3600))
DEBOUNCE=3

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG/queue.log"; }

gpu1_used() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v g=1 '$1==g {gsub(/ /,"",$2); print $2}'
}

gpu1_compute_apps() {
  local uuid
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
    | awk -F, -v g=1 '$1==g {gsub(/ /,"",$2); print $2}')
  [ -z "$uuid" ] && { echo 0; return; }
  nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits \
    | awk -F, -v u="$uuid" 'tolower($1)==tolower(u) {n++} END {print n+0}'
}

monitor_gpu() {
  while true; do
    { date -u +%Y-%m-%dT%H:%M:%SZ
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
      echo '---'
    } >> "$LOG/gpu_monitor.log" 2>/dev/null
    sleep 60
  done
}

wait_gpu1() {
  local ok=0 start=$SECONDS
  # Dependency gate: the UPBD chain writes "CHAIN ALL DONE" only after all its
  # training/eval stages finish. Without this, a short idle gap between UPBD
  # stages (e.g. generation -> model load) can release this queue early and
  # cause two fp32 8B jobs to contend for GPU1 (observed 2026-11-27).
  UPBD_DONE_LOG=/tmp/upbd_chain/chain.log
  while true; do
    local used apps done
    used=$(gpu1_used); used=${used:-0}
    apps=$(gpu1_compute_apps); apps=${apps:-0}
    done=0
    grep -q 'CHAIN ALL DONE' "$UPBD_DONE_LOG" 2>/dev/null && done=1
    if [ "$done" -eq 1 ] && [ "$used" -lt 3000 ] && [ "$apps" -eq 0 ]; then
      ok=$((ok + 1))
      log "GPU1 idle poll ${ok}/${DEBOUNCE} (upbd_done=1 used=${used}MiB apps=${apps})"
    else
      ok=0
      log "GPU1 busy or UPBD chain not done (upbd_done=$done used=${used}MiB apps=${apps}); waiting 60s"
    fi
    if [ "$ok" -ge "$DEBOUNCE" ]; then
      log "GPU1 free and UPBD chain done for ${DEBOUNCE} consecutive polls; queue released"
      return 0
    fi
    if [ $((SECONDS - start)) -ge "$MAX_WAIT_SEC" ]; then
      log "QUEUE_TIMEOUT: GPU1 did not free within 30h; exiting without training"
      return 3
    fi
    sleep 60
  done
}

record_status() {
  local tag=$1 slug seed dir
  touch "$LOG/status.tsv"
  for slug in $SLUGS; do
    for seed in $SEEDS; do
      dir="$UNL1127_RESULTS/unl1127_${slug}_${tag}_s${seed}"
      if [ -s "$dir/train_summary.json" ]; then
        printf '%s\t%s\t%s\tDONE\n' "$tag" "$slug" "$seed" >> "$LOG/status.tsv"
      else
        printf '%s\t%s\t%s\tMISSING\n' "$tag" "$slug" "$seed" >> "$LOG/status.tsv"
      fi
    done
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
  echo "=== queue_1127_gpu1.sh DRY RUN (no wait, no GPU work) ==="
  echo "ROOT=$ROOT"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "UNL1127_RESULTS=$UNL1127_RESULTS"
  echo "UNL1127_LOGS=$LOG"
  echo "UNL1127_DTYPE=$UNL1127_DTYPE"
  echo "UNL1127_UNLEARNED_ADAPTER=$UNL1127_UNLEARNED_ADAPTER"
  echo "GATE: UPBD chain 'CHAIN ALL DONE' marker + GPU1 memory.used<3000MiB and 0 compute apps for 3 consecutive 60s polls; max wait 30h"
  echo "NOTE: completion marker added after an idle-gap collision was observed on 2026-11-27"
  echo "PHASE 1: python scripts/run_all.py --all"
  echo "PHASE 2: python scripts/run_all.py --all --decisive"
  echo "JOBS: 10 methods x 2 seeds x 2 phases = 40; run-id unl1127_<slug>_<tag>_s<seed>"
  echo "=== dry-run complete; nothing launched ==="
  exit 0
fi

: > "$LOG/status.tsv"
log "=== QUEUE START pid=$$ ==="
log "WARNING: gate only checks GPU1 idle; UPBD chain completion marker is NOT checked (user choice). Short idle gaps may cause contention."
log "env: RESULTS=$UNL1127_RESULTS DTYPE=$UNL1127_DTYPE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

monitor_gpu &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT

wait_gpu1 || exit $?

log "=== PHASE 1 SMOKE START ==="
python scripts/run_all.py --all 2>&1 | tee -a "$LOG/phase_smoke.log"
SMOKE_RC=${PIPESTATUS[0]}
log "=== PHASE 1 SMOKE DONE rc=$SMOKE_RC ==="
record_status smoke

log "=== PHASE 2 DECISIVE START ==="
python scripts/run_all.py --all --decisive 2>&1 | tee -a "$LOG/phase_decisive.log"
DECISIVE_RC=${PIPESTATUS[0]}
log "=== PHASE 2 DECISIVE DONE rc=$DECISIVE_RC ==="
record_status decisive

python - <<'PY' 2>&1 | tee -a "$LOG/queue.log"
import json, pathlib
root = pathlib.Path("/usr/local/workspace/tracks/lrm-unlearning/R-TOFU/workspace/results/unl1127")
p = root / "unl1127_budget_decisive_s1001" / "train_summary.json"
if p.is_file():
    d = json.loads(p.read_text())
    print(f"[calibration] first decisive job elapsed_sec={d.get('elapsed_sec')} steps={d.get('steps')}")
else:
    print("[calibration] first decisive job summary missing; check phase_decisive.log")
PY

log "=== QUEUE ALL DONE smoke_rc=$SMOKE_RC decisive_rc=$DECISIVE_RC ==="
