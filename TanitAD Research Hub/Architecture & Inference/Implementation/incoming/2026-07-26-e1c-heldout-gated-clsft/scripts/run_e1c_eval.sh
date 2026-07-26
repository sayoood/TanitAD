#!/bin/bash
# E1c FRONTIER EVAL — base + every pre-registered checkpoint on the SAME 44
# held-out episodes. Incremental: --resume skips points already banked.
# Terminal marker: E1C_EVAL_RUN_DONE.
cd /workspace/e1c || exit 1
export PYTHONPATH=/workspace/TanitAD/stack
echo "[run_e1c_eval] START $(date -u +%FT%TZ)"
/workspace/venv/bin/python e1c_eval.py \
  --base-ckpt /workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt \
  --ft-dir    /workspace/e1c/refc-base-e1c-clsft \
  --val-dir   /workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6 \
  --k20-steps 4000 \
  --order-by-gate /workspace/e1c/refc-base-e1c-clsft/heldout_gate.jsonl \
  --out       /workspace/e1c/e1c_frontier_result.json \
  --resume
rc=$?
echo "[run_e1c_eval] rc=$rc DONE $(date -u +%FT%TZ)"
if [ $rc -ne 0 ]; then echo "E1C_EVAL_FAILED"; fi
echo "E1C_EVAL_RUN_DONE"
