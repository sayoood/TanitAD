#!/usr/bin/env bash
# Direct launch, 2026-09-21: W7's navhard scoring SUCCEEDED on content (4/4 arms OK, run 859e25) and
# handed extraction scheduling to the orchestrator; no scoring process is alive; RAM 12 GB free.
# The waiter's gates are therefore satisfied by measurement, so this runs the same two steps it would.
set -u
REPO="D:/Projects/TanitAD"; PKG="$REPO/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-20-navhard-stage1-extract"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
INPUTS="C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json"
ARCH="D:/Archive/devbox-C/navsim/data/openscene-v1.1/openscene_sensor_test_camera"
DEST="C:/Users/Admin/tanitad-caches/navhard-stage1-frames-20260920"
LOG="$PKG/raw/extract_waiter.log"
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
say "DIRECT LAUNCH (pid $$): W7 run 859e25 4/4 OK; extraction scheduled by orchestrator"
t0=$(date +%s)
"$PY" "$PKG/code/extract_navhard_stage1.py" --inputs "$INPUTS" --archives "$ARCH" --dest "$DEST" \
  --receipts "$PKG/raw/receipts" --shards 0-31 >> "$PKG/raw/extract_run.log" 2>&1
say "extraction finished rc=$? wall_s=$(( $(date +%s) - t0 ))"
"$PY" "$PKG/code/extract_navhard_stage1.py" --inputs "$INPUTS" --archives "$ARCH" --dest "$DEST" \
  --receipts "$PKG/raw/receipts" --verify > "$PKG/raw/VERIFY.txt" 2>&1
say "verify rc=$? verdict=$(head -1 "$PKG/raw/VERIFY.txt" 2>/dev/null)"
