#!/bin/bash
# E1b full run: MINE recoverable pre-failure states (once) -> failure-gated
# CL-SFT of REF-C base. Detached-safe (survives ssh close). Logs to e1b_run.log.
set -u
cd /workspace/e1b
export PYTHONPATH=/workspace/TanitAD/stack
PY=/workspace/venv/bin/python

TRAIN=/workspace/pai_epcache/physicalai-train-e438721ae894
HELDOUT=/workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6
BASE=/workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt
BUF=/workspace/e1b/mined_buffer.pt
OUT=/workspace/e1b/refc-base-e1b-clsft

echo "[run_e1b] START $(date -u +%FT%TZ)"

# --- 1. mine (idempotent: skip if buffer already present) ---
if [ ! -f "$BUF" ]; then
  echo "[run_e1b] MINING ..."
  $PY e1b_mine.py --train-dir "$TRAIN" --refc-ckpt "$BASE" \
      --K 185 --episodes 600 --out "$BUF" || { echo "[run_e1b] MINE FAILED"; exit 1; }
else
  echo "[run_e1b] buffer exists, skip mining: $BUF"
fi

# --- 2. failure-gated CL-SFT ---
echo "[run_e1b] CL-SFT ..."
$PY e1b_clsft.py --base-ckpt "$BASE" --buffer "$BUF" --parity-dir "$TRAIN" \
    --out "$OUT" --steps 4000 --lr 2e-5 --warmup 100 \
    --cl-batch 32 --replay-batch 32 --replay-episodes 0 --workers 4 \
    --lam-cl 1.0 --lam-replay 1.0 --freeze-encoder 1 \
    --assert-disjoint-heldout "$HELDOUT" \
    || { echo "[run_e1b] CLSFT FAILED"; exit 1; }

echo "[run_e1b] DONE $(date -u +%FT%TZ)"
echo "E1B_RUN_DONE"
