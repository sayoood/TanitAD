#!/bin/bash
# E1e-A — is the CL/OL conflict a WEIGHTING artifact? lam_replay 1.0 -> 3.0.
#
# EVERYTHING ELSE BYTE-IDENTICAL to run_e1c.sh: same base ckpt, same mined buffer
# (md5 asserted by the trainer), same parity + heldout dirs, same
# steps/lr/warmup/batches/seed, encoder frozen. lam_cl stays 1.0, so the ONLY
# change is the anti-forgetting weight that e1c_clsft.py's own header calls
# "deliberately NOT a lever here".
#
# WHY THIS AND NOT ANOTHER ALPHA POINT: E1d moved along a segment between two
# already-trained endpoints and found a BARRIER (dep_overall separated-WORSE at
# five consecutive interior alpha). A point off that segment can only be reached
# by changing the OBJECTIVE, which is where E1c §4.2 and E2a both pointed.
#
# Pre-registration: PRE_REGISTRATION_E1E.md, committed BEFORE this ran.
# Terminal marker: E1E_A_RUN_DONE.
set -u
cd /workspace/e1c || exit 1
export PYTHONPATH=/workspace/TanitAD/stack
export OMP_NUM_THREADS=6
PY=/workspace/venv/bin/python

TRAIN=/workspace/pai_epcache/physicalai-train-e438721ae894
HELDOUT=/workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6
BASE=/workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt
BUF=/workspace/e1b/mined_buffer.pt
BUFMD5=a32cfe9bfea4b1b5c196d3bb7f71fa5f
OUT=/workspace/e1e/refc-base-e1e-lamrep3

mkdir -p /workspace/e1e
echo "[run_e1e_A] START $(date -u +%FT%TZ)"

$PY e1c_clsft.py --base-ckpt "$BASE" --buffer "$BUF" --buffer-md5 "$BUFMD5" \
    --parity-dir "$TRAIN" --heldout-dir "$HELDOUT" \
    --out "$OUT" --steps 4000 --lr 2e-5 --warmup 100 \
    --cl-batch 16 --replay-batch 16 --replay-episodes 0 --workers 4 \
    --lam-cl 1.0 --lam-replay 3.0 --freeze-encoder 1 \
    --assert-disjoint-heldout "$HELDOUT"
RC=$?
echo "[run_e1e_A] TRAIN rc=$RC $(date -u +%FT%TZ)"
if [ $RC -ne 0 ]; then echo "E1E_A_RUN_FAILED"; exit 1; fi

# Frontier eval with the UNMODIFIED evaluator: same estimator, same
# evaluate_point/render_verdict, same six conditions as E1c and E1d.
$PY e1c_eval.py \
  --base-ckpt "$BASE" --ft-dir "$OUT" --val-dir "$HELDOUT" \
  --steps 1000,2000,3000,4000 --k20-steps 4000 \
  --out /workspace/e1e/e1e_A_frontier.json --resume
echo "[run_e1e_A] EVAL rc=$? $(date -u +%FT%TZ)"
echo "E1E_A_RUN_DONE"
