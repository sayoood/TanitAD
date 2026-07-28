#!/bin/bash
# E1d — WiSE-FT alpha frontier over the E1c CL-SFT delta_step04000.
# Same evaluator, same estimator, same verdict logic as E1c: alpha is encoded in
# the step number so e1c_eval.py adjudicates it UNMODIFIED.
# CONTROL: alpha=1.00 (step 100) must reproduce E1c frontier row 4000
#          (dep_overall -0.4274, dep_junction -0.4270, ade +0.1947).
# Terminal marker: E1D_ALPHA_RUN_DONE.
cd /workspace/e1c || exit 1
export PYTHONPATH=/workspace/TanitAD/stack
export OMP_NUM_THREADS=6
echo "[run_e1d] START $(date -u +%FT%TZ)"
/workspace/venv/bin/python e1c_eval.py \
  --base-ckpt /workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt \
  --ft-dir    /workspace/e1c/alpha_sweep \
  --val-dir   /workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6 \
  --steps     10,20,30,40,50,60,70,85,100 \
  --k20-steps 100 \
  --out       /workspace/e1c/e1d_alpha_result.json \
  --resume
rc=$?
echo "[run_e1d] rc=$rc DONE $(date -u +%FT%TZ)"
[ $rc -ne 0 ] && echo "E1D_ALPHA_FAILED"
echo "E1D_ALPHA_RUN_DONE"
