#!/usr/bin/env bash
# Battery 2 -- the SPEED levers and the recommended BUNDLE against the same U12 reference
# (battery 1's launch-code run), every arm 12 steps, CPU, same argv recipe:
#   U12L5: launch code logged every 5th step (lever 2's reference)
#   L12 : lever 2 only (apply_lever2.py on the launch trainer), logged every 5th step
#   K12 : launch code, --conflict-every 50 (the argv-only cadence lever)
#   B12 : the BUNDLE tree (freeze + lever 2 + getstate + I3), --conflict-every 50, every 5th step logged
#   O12 : launch code, --map-hires-grad-ckpt off (informative: CPU determinism only)
# usage: run_ab_batch2.sh <launch tree> <lever2 tree> <bundle tree> <work dir> [wait-for-file]
set -u
CLEAN="$1"; L2="$2"; COMBO="$3"; W="$4"; WAIT="${5:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
mkdir -p "$W"
if [ -n "$WAIT" ]; then
  until grep -q "BATCH DONE" "$WAIT" 2>/dev/null; do sleep 30; done
fi
run() {
  local name="$1" tree="$2" steps="$3"; shift 3
  echo "$(date +%T) START $name" >> "$W/batch.log"
  "$PY" "$HERE/numerics_ab.py" --tree "$tree" --out "$W/$name" --steps "$steps" "$@" > "$W/$name.log" 2>&1
  local rc=$?   # ⛔ captured FIRST: "$(date)" inside the echo resets $? to 0 (MEASURED: F6N)
  echo "$(date +%T) END $name rc=$rc (the verdict is digests.json 'error', never this rc)" >> "$W/batch.log"
}
# ⛔ one instance per work dir: two batteries writing one run dir corrupt each other (MEASURED
# 2026-09-28: a TaskStop'd wrapper left its script bash alive and it ran beside the relaunch)
# (no flock in Git Bash: an atomic mkdir is the lock; remove it by hand after a hard kill)
if ! mkdir "$W/.battery.lock" 2>/dev/null; then
  echo "another battery holds $W/.battery.lock -- exiting"; exit 3
fi
# lever 2 is a no-op when EVERY step is logged, so its arms log every 5th step (rows at 5, 10)
# against a launch-code reference logged the same way (U12L5; U12L5 vs U12 = log-cadence control)
run U12L5 "$CLEAN" 12 --log-every 5
run L12 "$L2" 12 --log-every 5
run K12 "$CLEAN" 12 --conflict-every 50
run B12 "$COMBO" 12 --conflict-every 50 --log-every 5
echo "$(date +%T) BATCH2 CORE DONE" >> "$W/batch.log"
run O12 "$CLEAN" 12 --map-hires-grad-ckpt off
echo "$(date +%T) BATCH2 DONE" >> "$W/batch.log"
