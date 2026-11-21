#!/bin/bash
# LF0 — decoded-BEV lead read-off (JEPA_PHYSICS_SURVEY §LF0). Pod chain.
#
# ⛔ NO GIT IN THIS CHAIN. Pods have no git credentials: `git fetch` HANGS, and a
# failed fetch followed by a successful checkout RESETS the tree to an ancient
# commit and destroys shipped files. Everything arrives by md5-verified file-ship
# and is verified here by grep + a real import before anything launches.
#
# Every branch emits LF0_EXIT= so a caller can never mistake silence for success.
set -u
REPO="${REPO:-/workspace/TanitAD_head}"
S="$REPO/stack"
CKPT="${CKPT:-/workspace/experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt}"
P8RUN="${P8RUN:-/workspace/experiments/p8-occupancy-c}"
CACHE="${CACHE:-/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl}"
JOIN="${JOIN:-/workspace/data/p8_join/combined140.jsonl}"
OUT="${OUT:-/workspace/experiments/lf0-bev-lead}"
K="${K:-10}"
NWIN="${NWIN:-900}"
LOG="${LOG:-/tmp/lf0.log}"
: > "$LOG"

export PYTHONPATH="$S"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
# torch spawns ~113 threads PER PROCESS and concurrent arms then make NO
# progress while looking exactly like a hang (MEASURED: 7 arms at sm 0-6% for
# 50 minutes). OMP_NUM_THREADS is set above for that reason, not for speed.

# ---- gate 10: the files this needs actually exist ------------------------- #
for f in "$S/scripts/lf0_bev_lead.py" "$S/scripts/train_p8_occupancy.py" \
         "$S/scripts/p8_bev_reel.py" "$CKPT" "$P8RUN/p8_head.pt" "$JOIN"; do
  if [ ! -e "$f" ]; then
    echo "[lf0-chain] 10 MISSING $f — ship it (xz+b64 PTY push, per-file md5)." >> "$LOG"
    echo "LF0_EXIT=10" >> "$LOG"; exit 1
  fi
done

# ---- gate 11: the shipped file is the CURRENT one -------------------------- #
# `git log` on a pod is not proof of anything (its HEAD is weeks stale while the
# working tree is current), so verify the specific fix by content.
for tok in "read_lead_range" "reader_sanity" "INHERITED from the P8 gate"; do
  if ! grep -q "$tok" "$S/scripts/lf0_bev_lead.py"; then
    echo "[lf0-chain] 11 STALE lf0_bev_lead.py — missing '$tok'" >> "$LOG"
    echo "LF0_EXIT=11" >> "$LOG"; exit 1
  fi
done

# ---- gate 12: a REAL import, not a grep ----------------------------------- #
# The greps passing while the imports failed is exactly how the T1 run burned
# 11 minutes per arm and then died in analyze().
cd "$S" || { echo "LF0_EXIT=12_NO_STACK" >> "$LOG"; exit 1; }
python3 - >> "$LOG" 2>&1 <<'PY'
import sys
try:
    from tanitad.data.bev_raster import GRID_DEFAULT
    from train_p8_occupancy import (BEVOccupancyHead, batch_rasters,  # noqa: F401
                                    build_args, build_raster_source, p8_latents)
    sys.path.insert(0, "scripts")
    from lf0_bev_lead import corridor_cols, read_lead_range
    # maths smoke: an agent 20.25 m ahead in the corridor must read 20.25 m
    import numpy as np
    nx, ny = GRID_DEFAULT.shape
    cols = corridor_cols(ny, GRID_DEFAULT.y_half_m, GRID_DEFAULT.cell_m, 1.5)
    r = np.zeros((nx, ny), np.float32); r[40, int(cols[2])] = 1.0
    got = read_lead_range(r, tau=0.7, cols=cols, cell_m=GRID_DEFAULT.cell_m)
    assert abs(got - 20.25) < 1e-6, got
    print("LF0_IMPORTS_OK")
except Exception as e:                                   # noqa: BLE001
    print(f"LF0_IMPORT_FAIL {type(e).__name__}: {e}")
PY
if ! grep -q LF0_IMPORTS_OK "$LOG"; then
  echo "LF0_EXIT=12" >> "$LOG"; exit 1
fi

# ---- gate 15: real disk, never df ----------------------------------------- #
# df reports the 965 TB cluster and hides the per-pod MooseFS quota; a full
# quota killed the flagship mid-checkpoint.
mkdir -p "$OUT"
if ! dd if=/dev/zero of="$OUT/.ddtest" bs=1M count=64 >/dev/null 2>&1; then
  echo "[lf0-chain] 15 DISK — 64 MB dd write FAILED (quota, not df)" >> "$LOG"
  echo "LF0_EXIT=15" >> "$LOG"; rm -f "$OUT/.ddtest"; exit 1
fi
rm -f "$OUT/.ddtest"

# ---- run ------------------------------------------------------------------ #
# --v2-cache is required by the TRAINER's shared arg surface; LF0 never builds a
# train loader, so it points at the val cache (as p8_bev_reel does).
python3 -u scripts/lf0_bev_lead.py \
  --p8-run "$P8RUN" --ckpt "$CKPT" --out "$OUT" \
  --k "$K" --n-windows "$NWIN" \
  --raster-source join-file --join-file "$JOIN" \
  --v2-val-cache "$CACHE" --v2-cache "$CACHE" \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 >> "$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then echo "LF0_EXIT=RUN_$rc" >> "$LOG"; exit "$rc"; fi
grep -q LF0_DONE "$LOG" || { echo "LF0_EXIT=NO_DONE_MARKER" >> "$LOG"; exit 1; }
echo "LF0_EXIT=0" >> "$LOG"
