#!/bin/bash
# E1c — instrumented CL-SFT re-run of E1b with the CORRECTED (held-out) guard.
# The mined buffer is REUSED from E1b (md5-verified), never regenerated.
# Detached-safe. Logs to e1c_run.log. Terminal marker: E1C_RUN_DONE.
set -u
cd /workspace/e1c
export PYTHONPATH=/workspace/TanitAD/stack
PY=/workspace/venv/bin/python

TRAIN=/workspace/pai_epcache/physicalai-train-e438721ae894
HELDOUT=/workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6
BASE=/workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt
BUF=/workspace/e1b/mined_buffer.pt
BUFMD5=a32cfe9bfea4b1b5c196d3bb7f71fa5f
OUT=/workspace/e1c/refc-base-e1c-clsft

echo "[run_e1c] START $(date -u +%FT%TZ)"
$PY e1c_clsft.py --base-ckpt "$BASE" --buffer "$BUF" --buffer-md5 "$BUFMD5" \
    --parity-dir "$TRAIN" --heldout-dir "$HELDOUT" \
    --out "$OUT" --steps 4000 --lr 2e-5 --warmup 100 \
    --cl-batch 16 --replay-batch 16 --replay-episodes 0 --workers 4 \
    --lam-cl 1.0 --lam-replay 1.0 --freeze-encoder 1 \
    --assert-disjoint-heldout "$HELDOUT" \
    || { echo "[run_e1c] CLSFT FAILED"; echo "E1C_RUN_FAILED"; exit 1; }

echo "[run_e1c] DONE $(date -u +%FT%TZ)"
echo "E1C_RUN_DONE"
