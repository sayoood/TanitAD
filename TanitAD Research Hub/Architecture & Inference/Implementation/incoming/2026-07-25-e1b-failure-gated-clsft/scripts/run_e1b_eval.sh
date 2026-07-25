#!/bin/bash
# E1b paired verdict — the PRE-REGISTERED run. All 44 held-out episodes, K=20 & K=185.
cd /workspace/e1b || exit 1
export PYTHONPATH=/workspace/TanitAD/stack
echo "[run_e1b_eval] START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
/workspace/venv/bin/python e1b_eval.py \
  --base-ckpt /workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt \
  --ft-ckpt   /workspace/e1b/refc-base-e1b-clsft/ckpt.pt \
  --val-dir   /workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6 \
  --horizons  20,185 \
  --out       /workspace/e1b/e1b_eval_result.json
rc=$?
echo "[run_e1b_eval] rc=$rc DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ $rc -ne 0 ]; then echo "E1B_EVAL_FAILED"; fi
echo "E1B_EVAL_RUN_DONE"
