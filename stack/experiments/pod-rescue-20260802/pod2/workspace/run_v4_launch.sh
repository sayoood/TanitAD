#!/usr/bin/env bash
# ============================================================================
# flagship v4 30k launch — pod2 (tanitad-pod2), 2026-07-22.
# Interpreter: /usr/bin/python3  (torch 2.4.1+cu124, CUDA True on A40 — MEASURED;
#   NOTE the brief's assumed /workspace/venv/bin/python DOES NOT EXIST on pod2).
# Modes:  preflight | smoke | launch
#   preflight -> --print-launch (imports whole v4 graph + config invariants -> PREFLIGHT: OK)
#   smoke     -> --real-smoke on 4 real windows, warm-started from the real trunk (CPU)
#   launch    -> detached nohup training; writes PID to $OUT/train.pid and $OUT/train.log
# ============================================================================
set -uo pipefail
MODE="${1:-preflight}"
cd /workspace/TanitAD/stack || { echo "FATAL: no /workspace/TanitAD/stack"; exit 1; }
export PYTHONPATH=/workspace/TanitAD/stack
PY=/usr/bin/python3
OUT=/workspace/experiments/flagship-v4-30k
TRAIN=/workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894
VAL=/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11
TRUNK=/workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt
ANCH=/workspace/experiments/flagship_v4_anchors_dense.pt

ARGS=(
  --train-cache "$TRAIN"
  --val-cache "$VAL"
  --trunk "$TRUNK"
  --anchors-dense "$ANCH"
  --out "$OUT"
  --labels v3 --lambda-plan sched --phase-a-steps 2000 --phase-b-steps 8000
  --strategic full --long-horizon-k 50 --steps 30000 --gate-step 10000
  --batch 16 --lr-head 1e-4 --lr-trunk 3e-4 --eval-every 500 --save-every 1000
  --eval-episodes 40 --rollout-k 4 --seed 0 --device cuda
)

case "$MODE" in
  preflight)
    "$PY" scripts/train_flagship_v4.py --print-launch "${ARGS[@]}"
    ;;
  smoke)
    "$PY" scripts/train_flagship_v4.py --real-smoke \
      --train-cache "$TRAIN" --trunk "$TRUNK" --n-windows 4 --seed 0
    ;;
  launch)
    mkdir -p "$OUT"
    nohup "$PY" scripts/train_flagship_v4.py "${ARGS[@]}" </dev/null > "$OUT/train.log" 2>&1 &
    PID=$!
    echo "$PID" > "$OUT/train.pid"
    disown "$PID" 2>/dev/null || true
    echo "V4_PID=$PID"
    echo "launched_utc=$(date -u +%FT%TZ)"
    ;;
  *) echo "usage: $0 {preflight|smoke|launch}"; exit 64 ;;
esac
