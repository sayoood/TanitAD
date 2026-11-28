#!/bin/bash
# P2 chain (eval/PREREG_P2.md): the joint fine-tune starts the moment the A40 is free.
# Waits for: SFT-4's RUN EXIT, the PDM training relabel's exit (the labels must be complete) and the P2 gate's PASS.
# Then runs train.py --preflight on the REAL configuration and trains under a resume supervisor (summary.json done=true
# is the off-switch; every child is launched with the chain's fds closed).
set -u
source /workspace/teacher_env.sh
PY="$DRIVERL_EVAL_PYTHON"
RUN=/workspace/data/refe_runs/p2_joint_pdm
L=/workspace/data/refe_p2/chain.log; mkdir -p /workspace/data/refe_p2
log() { echo "$(date -u +%FT%T) $*" >> $L; }
log "chain armed"
until grep -q "RUN EXIT" /workspace/data/refe_sft4/chain.log 2>/dev/null || grep -q "REFUSED" /workspace/data/refe_sft4/chain.log 2>/dev/null; do sleep 120 200>&-; done
log "SFT-4 done: $(grep -E 'RUN EXIT|REFUSED' /workspace/data/refe_sft4/chain.log | tail -1)"
until grep -q "ZZPDM_TRAIN_EXIT" /workspace/data/refe_sft2/pdm_train.log 2>/dev/null; do sleep 120 200>&-; done
log "PDM labels complete: $(cat /workspace/data/refe_sft2/train_pdm/*.jsonl | wc -l) rows"
until grep -q "^ZZEXIT" /workspace/data/refe_p2/gate.log 2>/dev/null; do sleep 120 200>&-; done
grep -q '"verdict": "PASS"' /workspace/data/refe_p2/gate/gate.json || { log "REFUSED: the P2 gate did not PASS"; exit 1; }
log "gate PASS"
cd /workspace/refe-p2/run
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
ARGS="--backbone vitl16 --targets /workspace/data/refe_navtrain/train_grow --scorer-targets /workspace/data/refe_navtrain/train_grow --images /workspace/data/navtrain_pixels --calib /workspace/data/refe_navtrain/train_grow/calib_table.json --init-from /workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt --scorer-mode onpolicy --onpolicy-targets /workspace/data/refe_navtrain/onpolicy/sets --pdm-labels /workspace/data/refe_sft2/train_pdm --pdm-only --scorer-sees-trunk --epochs 3 --epoch-unit scenes --batch 4 --accum 64 --lr 5e-5 --amp bf16 --tf32 --compile --workers 2 --log-every 1 --ckpt-every-min 20 --out $RUN --resume"
"$PY" train.py $ARGS --preflight > /workspace/data/refe_p2/preflight.log 2>&1 200>&-
grep -q "^PREFLIGHT_OK" /workspace/data/refe_p2/preflight.log || { log "REFUSED: train.py --preflight did not print PREFLIGHT_OK"; exit 1; }
log "train.py preflight OK; launching"
for att in 1 2 3 4 5 6; do
  if [ -f $RUN/summary.json ] && grep -q '"done": true' $RUN/summary.json; then break; fi
  "$PY" train.py $ARGS >> /workspace/data/refe_p2/train.log 2>&1 200>&-
  log "attempt $att exited rc $?"
  sleep 60 200>&-
done
if [ -f $RUN/summary.json ] && grep -q '"done": true' $RUN/summary.json; then log "ZZP2_DONE"; else log "ZZP2_FAILED (no done=true after the attempts)"; fi
