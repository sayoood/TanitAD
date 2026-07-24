#!/usr/bin/env bash
# ============================================================================
# flagship v4.2 30k launch - pod2 (tanitad-pod2), 2026-07-23.
#
# v4.2 = the SAME joint architecture as v4/v4.1 (v1 WM trunk trained-from-scratch
#   ViT encoder + action-conditioned predictor + strategic + tactical + operative
#   anchored-diffusion, ALL TRAINED JOINTLY, NO FROZEN PART, warm-started from the
#   v1 trunk). TWO schedule changes ONLY vs v4.1 (Sayed's decision, 2026-07-23):
#
#   1. CAP-AND-HOLD lambda_plan controller (--lam-mult-floor 0.25). v4.1 ran a naive
#      HALVE-TO-ZERO controller: on its cut lr_trunk any planner gradient briefly
#      breached the noisy canary, the down-only ratchet fired every eval, and
#      lam_mult decayed to ~1.5e-5 by step 10k => the planner->trunk coupling was OFF
#      from ~step 2000 => planner STARVED (held-out ade_0_2s 0.8522, FAIL vs 0.60).
#      The fix HOLDS lam_mult at a floor so the planner ALWAYS keeps a real gradient.
#      (tanitad/train/v4_curriculum.py CanaryController; V4_DESIGN §14.4 O-14 / §5.5;
#      2026-07-23 synthesis §4/§7.)
#   2. --lr-trunk 1e-4 : BETWEEN v4's 3e-4 (degraded the WM: canary 0.42->1.3 by
#      step 3500) and v4.1's 3e-5 (starved the planner). Deviates from O-14's 3e-4
#      by Sayed's decision.
#
# The WM stayed HEALTHY under v4.1 (canary 0.4599 PASS) — only the schedule was wrong;
#   the architecture is UNCHANGED. Warm-start: the V1 trunk flagship4b-speedjerk-30k
#   (NOT the v4/v4.1 ckpts). Fresh --out => no resume => clean warm-start from --trunk.
# Interpreter: /usr/bin/python3 (torch 2.4.1+cu124, CUDA True on A40).
# Modes: preflight | smoke | launch
# ============================================================================
set -uo pipefail
MODE="${1:-preflight}"
cd /workspace/TanitAD/stack || { echo "FATAL: no /workspace/TanitAD/stack"; exit 1; }
export PYTHONPATH=/workspace/TanitAD/stack
PY=/usr/bin/python3
OUT=/workspace/experiments/flagship-v4.2-30k
TRAIN=/workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894
VAL=/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11
TRUNK=/workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt
ANCH=/workspace/experiments/flagship_v4_anchors_dense.pt
# /workspace logs get SWALLOWED on death (CLAUDE.md pod2 constraints) -> ALSO /tmp.
TMPLOG=/tmp/flagship-v4.2-train.log

ARGS=(
  --train-cache "$TRAIN"
  --val-cache "$VAL"
  --trunk "$TRUNK"
  --anchors-dense "$ANCH"
  --out "$OUT"
  --labels v3 --lambda-plan sched --phase-a-steps 2000 --phase-b-steps 8000
  --strategic full --long-horizon-k 50 --steps 30000 --gate-step 10000
  --batch 16 --accum 4 --lr-head 1e-4 --lr-trunk 1e-4 --lam-mult-floor 0.25
  --eval-every 500 --save-every 1000
  --eval-episodes 40 --rollout-k 4 --seed 0 --device cuda
)
# ⭐ --batch 16 --accum 4 = EFFECTIVE batch 64, matching v1 (registry §1.2:
#   flagship4b-speedjerk was --batch-size 16 --accum 4). v4.1 ran accum 1 (=16),
#   4x too small => 4x less data/step + noisier gradients (a second contributor to
#   v4.1's weak planner). Micro-batch 16 fits ~34.7 GB on the A40 WITHOUT
#   grad-checkpoint; expect ~4x v4.1's step time (~8-10 s/step) — the CORRECT signal
#   the whole chain (encoder+predictor+planners) trains at v1's data volume.

case "$MODE" in
  preflight)
    "$PY" scripts/train_flagship_v4.py --print-launch "${ARGS[@]}"
    ;;
  smoke)
    # real-smoke: proves v4_loss_step trains the factorised + strategic heads on REAL
    # parity windows on a fresh trunk (CPU-safe, tiny). No GPU load, no run started.
    "$PY" scripts/train_flagship_v4.py --real-smoke \
      --train-cache "$TRAIN" --trunk "$TRUNK" --n-windows 4 --seed 0
    ;;
  launch)
    mkdir -p "$OUT"
    # PRIMARY stdout -> /tmp (local, durable: survives a /workspace MooseFS swallow on
    # death; the trainer prints with flush=True so it is per-line durable). Plain
    # nohup redirect + </dev/null + disown = detaches cleanly and survives SSH close
    # (NO process substitution: a >(tee) child gets SIGHUP on disconnect and breaks
    # the trainer's stdout). The trainer ALSO writes train_log.jsonl / config.json /
    # metrics.json into $OUT on /workspace, flushed per row — the structured record.
    nohup "$PY" scripts/train_flagship_v4.py "${ARGS[@]}" </dev/null > "$TMPLOG" 2>&1 &
    PID=$!
    echo "$PID" > "$OUT/train.pid"
    disown "$PID" 2>/dev/null || true
    ln -sf "$TMPLOG" "$OUT/train.log"   # conventional path -> the durable /tmp log
    echo "V42_PID=$PID"
    echo "launched_utc=$(date -u +%FT%TZ)"
    echo "stdout_log=$TMPLOG (durable /tmp; survives a /workspace swallow)"
    echo "structured_log=$OUT/train_log.jsonl (trainer-written, flushed per row)"
    ;;
  *) echo "usage: $0 {preflight|smoke|launch}"; exit 64 ;;
esac
