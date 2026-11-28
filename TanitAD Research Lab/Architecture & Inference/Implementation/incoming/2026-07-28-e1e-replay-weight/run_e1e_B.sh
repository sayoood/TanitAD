#!/bin/bash
# E1e-B — lam_replay 8.0. THE SECOND AND LAST POINT ON THIS AXIS.
#
# ⛔ DO NOT LAUNCH THIS UNTIL E1e-A's FRONTIER HAS BEEN READ. The pre-registration
# fixes the skip rule in advance: B is SKIPPED iff A shows the closed-loop gain
# already destroyed (P1 false at every step) AND Ga still fails — because more
# replay can then only push further along the same losing direction. A's open-loop
# improvement alone does NOT license B; the closed-loop side must be seen first.
#
# ⚠️ AND NEVER ALONGSIDE ANOTHER JOB ON THIS POD. C53: a "light" concurrent job ran
# at 477 % CPU and cost the trainer >=4x throughput. Run this only when pod3 is
# otherwise idle.
#
# Everything except --lam-replay is byte-identical to run_e1e_A.sh, which is itself
# byte-identical to run_e1c.sh apart from that one flag: same base ckpt, same mined
# buffer (md5 asserted by the trainer), same parity + heldout dirs, same
# steps/lr/warmup/batches/seed, encoder frozen.
#
# HONEST BOUND, restated from the pre-registration: lam_replay and lam_cl are NOT
# independent — only their ratio matters up to the LR schedule — so this is the
# SAME AXIS as A, one point further along, not a second lever.
# Terminal marker: E1E_B_RUN_DONE.
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
OUT=/workspace/e1e/refc-base-e1e-lamrep8

# refuse to start if anything else is on the GPU (the C53 guard, mechanical)
if pgrep -f "[e]1c_clsft|[e]1c_eval|[h]arvest_scaleup" >/dev/null 2>&1; then
  echo "[run_e1e_B] REFUSING: another job is running on pod3 (C53). Aborting."
  echo "E1E_B_RUN_REFUSED"; exit 2
fi

mkdir -p /workspace/e1e
echo "[run_e1e_B] START $(date -u +%FT%TZ)"
$PY e1c_clsft.py --base-ckpt "$BASE" --buffer "$BUF" --buffer-md5 "$BUFMD5" \
    --parity-dir "$TRAIN" --heldout-dir "$HELDOUT" \
    --out "$OUT" --steps 4000 --lr 2e-5 --warmup 100 \
    --cl-batch 16 --replay-batch 16 --replay-episodes 0 --workers 4 \
    --lam-cl 1.0 --lam-replay 8.0 --freeze-encoder 1 \
    --assert-disjoint-heldout "$HELDOUT"
RC=$?
echo "[run_e1e_B] TRAIN rc=$RC $(date -u +%FT%TZ)"
if [ $RC -ne 0 ]; then echo "E1E_B_RUN_FAILED"; exit 1; fi

$PY e1c_eval.py \
  --base-ckpt "$BASE" --ft-dir "$OUT" --val-dir "$HELDOUT" \
  --steps 1000,2000,3000,4000 --k20-steps 4000 \
  --out /workspace/e1e/e1e_B_frontier.json --resume
echo "[run_e1e_B] EVAL rc=$? $(date -u +%FT%TZ)"
echo "E1E_B_RUN_DONE"
