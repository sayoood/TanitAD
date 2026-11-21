#!/bin/bash
# E1f — junction-restricted buffer. Is the TARGET wrong, rather than its weight?
#
# Byte-identical to run_e1e_A.sh EXCEPT --buffer / --buffer-md5 / --out.
# lam_replay stays 3.0 (E1e-A's setting, which has a full frontier to compare
# against), so exactly ONE thing changes: WHAT is supervised.
#
# Buffer: 733 of 3,537 records (20.7%) across 102 of 362 episodes, filtered at
# |dpsi| >= radians(10.0) — the evaluator's OWN --junction-deg, not a threshold
# invented for this arm. md5 asserted by the trainer at run time.
#
# ⚠️ NOT shortened for reuse: the trainer's leak-guard proves buffer episodes and
# the held-out 44 are DISJOINT, so over-reuse cannot inflate held-out metrics — it
# would show as UNDERperformance. Step-matched to E1c/E1e-A/E1e-B instead, which is
# what makes the comparison clean. (Corrects E1F_FEASIBILITY.md, before running.)
#
# Pre-registration: PRE_REGISTRATION_E1F.md, committed BEFORE this ran.
# Terminal marker: E1F_RUN_DONE.
set -u
cd /workspace/e1c || exit 1
export PYTHONPATH=/workspace/TanitAD/stack
export OMP_NUM_THREADS=6
PY=/workspace/venv/bin/python

TRAIN=/workspace/pai_epcache/physicalai-train-e438721ae894
HELDOUT=/workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6
BASE=/workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt
BUF=/workspace/e1f/junction_buffer.pt
BUFMD5=35fe24a2787c2afbf72888aeb23c525f
OUT=/workspace/e1f/refc-base-e1f-junction

# C53 guard, mechanical: never share the pod with another job.
if pgrep -f "[e]1c_clsft|[e]1c_eval|[h]arvest_scaleup|[p]seudo_label" >/dev/null 2>&1; then
  echo "[run_e1f] REFUSING: another job is running on pod3 (C53)."
  echo "E1F_RUN_REFUSED"; exit 2
fi

mkdir -p /workspace/e1f
echo "[run_e1f] START $(date -u +%FT%TZ)"
$PY e1c_clsft.py --base-ckpt "$BASE" --buffer "$BUF" --buffer-md5 "$BUFMD5" \
    --parity-dir "$TRAIN" --heldout-dir "$HELDOUT" \
    --out "$OUT" --steps 4000 --lr 2e-5 --warmup 100 \
    --cl-batch 16 --replay-batch 16 --replay-episodes 0 --workers 4 \
    --lam-cl 1.0 --lam-replay 3.0 --freeze-encoder 1 \
    --assert-disjoint-heldout "$HELDOUT"
RC=$?
echo "[run_e1f] TRAIN rc=$RC $(date -u +%FT%TZ)"
if [ $RC -ne 0 ]; then echo "E1F_RUN_FAILED"; exit 1; fi

$PY e1c_eval.py \
  --base-ckpt "$BASE" --ft-dir "$OUT" --val-dir "$HELDOUT" \
  --steps 1000,2000,3000,4000 --k20-steps 4000 \
  --out /workspace/e1f/e1f_frontier.json --resume
echo "[run_e1f] EVAL rc=$? $(date -u +%FT%TZ)"
echo "E1F_RUN_DONE"
