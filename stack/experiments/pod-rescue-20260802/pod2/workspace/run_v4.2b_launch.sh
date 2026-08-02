#!/usr/bin/env bash
# ============================================================================
# flagship v4.2b 30k launch - pod2 (tanitad-pod2), 2026-07-23.
#
# v4.2b = v4.2 with the pre-registered FLOOR-TOO-HIGH fix: --lam-mult-floor 0.15
#   (down from v4.2's 0.25). EVERYTHING ELSE IS IDENTICAL to v4.2.
#
# Why: v4.2 (floor 0.25) confirmed decisively that 0.25 is TOO HIGH. Its WM-integrity
#   canary ran AWAY under the held-at-floor planner gradient (MEASURED,
#   /tmp/flagship-v4.2-train.log): baseline 0.421 -> 0.520@1500 -> 0.860@2000 ->
#   0.722@4000 -> 0.768@5000, stuck in hard_breach/held_at_floor the whole of Phase B.
#   Interim eval v4.2@4000: 4wp ade_0_2s 0.987 + canary 0.722 — WORSE than v4.1@10k
#   (0.852/0.460) at fewer than half the steps, both still degrading. At floor 0.25 the
#   planner's 0.25x pull on the shared trunk degrades the world model FASTER than the
#   planner benefits. Lower the floor: 0.15 keeps a real (un-starved) planner gradient
#   — 5 orders of magnitude above v4.1's ~5e-7 starvation — while cutting the WM-
#   degrading pull ~40%. Cap-and-hold controller + gate are unchanged.
#
# Warm-start: FRESH from the V1 trunk flagship4b-speedjerk-30k (NOT v4.2's degraded
#   ckpt). Fresh --out => no resume => clean warm-start from --trunk.
# Interpreter: /usr/bin/python3 (torch 2.4.1+cu124, CUDA True on A40).
# Modes: preflight | smoke | launch
# ============================================================================
set -uo pipefail
MODE="${1:-preflight}"
cd /workspace/TanitAD/stack || { echo "FATAL: no /workspace/TanitAD/stack"; exit 1; }
export PYTHONPATH=/workspace/TanitAD/stack
PY=/usr/bin/python3
OUT=/workspace/experiments/flagship-v4.2b-30k
TRAIN=/workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894
VAL=/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11
TRUNK=/workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt
ANCH=/workspace/experiments/flagship_v4_anchors_dense.pt
# /workspace logs get SWALLOWED on death (CLAUDE.md pod2 constraints) -> ALSO /tmp.
TMPLOG=/tmp/flagship-v4.2b-train.log

ARGS=(
  --train-cache "$TRAIN"
  --val-cache "$VAL"
  --trunk "$TRUNK"
  --anchors-dense "$ANCH"
  --out "$OUT"
  --labels v3 --lambda-plan sched --phase-a-steps 2000 --phase-b-steps 8000
  --strategic full --long-horizon-k 50 --steps 30000 --gate-step 10000
  --batch 16 --accum 4 --lr-head 1e-4 --lr-trunk 1e-4 --lam-mult-floor 0.15
  --eval-every 500 --save-every 1000
  --eval-episodes 40 --rollout-k 4 --seed 0 --device cuda
)
# ⭐ ONLY change vs v4.2: --lam-mult-floor 0.15 (was 0.25). Same cap-and-hold
#   controller, lr_trunk 1e-4, effective batch 64 (16x4), v3 labels, dense anchors,
#   parity cache, gate-step 10000, milestones, not-frozen gate + gnorm split.

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
    # PRIMARY stdout -> /tmp (durable; survives a /workspace swallow; trainer prints
    # flush=True so per-line durable). Plain nohup + </dev/null + disown survives SSH
    # close (NO process substitution). Trainer ALSO writes train_log.jsonl/config.json/
    # metrics.json into $OUT on /workspace, flushed per row.
    nohup "$PY" scripts/train_flagship_v4.py "${ARGS[@]}" </dev/null > "$TMPLOG" 2>&1 &
    PID=$!
    echo "$PID" > "$OUT/train.pid"
    disown "$PID" 2>/dev/null || true
    ln -sf "$TMPLOG" "$OUT/train.log"
    echo "V42B_PID=$PID"
    echo "launched_utc=$(date -u +%FT%TZ)"
    echo "stdout_log=$TMPLOG (durable /tmp)"
    echo "structured_log=$OUT/train_log.jsonl (trainer-written, flushed per row)"
    ;;
  *) echo "usage: $0 {preflight|smoke|launch}"; exit 64 ;;
esac
