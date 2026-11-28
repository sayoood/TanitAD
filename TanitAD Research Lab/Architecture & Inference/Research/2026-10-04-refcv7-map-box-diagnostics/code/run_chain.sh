#!/bin/bash
# The diagnostic chain on Thor (SPEC.md sec. 6). Every GPU pass takes the shared Thor GPU lock for ITSELF only,
# so the video renderer can interleave between passes. Each step's exit code is written to status/<tag>.exit
# (never read through a pipe); the ARTIFACT (<tag>.json) is the evidence a pass ran, not the code.
set -u
D=/home/nvidia/refcv7_post/diag
RUN=/home/nvidia/refcv7_run/runs/refcv7-r101-s0
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
export REFCV6_REPO=/home/nvidia/refcv7_run/fec3a0dccf REFCV6_KIT=/home/nvidia
export PYTHONPATH=/home/nvidia/refcv7_run/fec3a0dccf/stack OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
mkdir -p $D/status $D/logs $D/out
cd $D/code

pass () {   # tag split ckpt [extra args...]  (REMAP set by caller for train splits)
  local tag=$1 split=$2 ck=$3; shift 3
  if [ -s $D/out/$tag.json ]; then echo "[chain] $tag already banked -- skip"; return 0; fi
  echo "[chain] $(date -u +%H:%M:%S) start $tag"
  flock $LOCK $PY run_diag.py --split $split --ckpt $ck --tag $tag --out $D/out "$@" \
      > $D/logs/$tag.log 2>&1 < /dev/null
  local rc=$?
  echo $rc > $D/status/$tag.exit
  echo "[chain] $(date -u +%H:%M:%S) end $tag rc=$rc artifact=$( [ -s $D/out/$tag.json ] && echo yes || echo NO )"
}

TRAIN_REMAP=$D/cache/remap_train_diag.json
CAL_REMAP=$D/cache/remap_train_cal64.json

# 1. TRAIN-DIAG fit pass (histograms + 1 % cell subsample + box packs)
export REFCV6_REMAP_OVERRIDES=$TRAIN_REMAP; pass train_fitpass train_diag $RUN/ckpt.pt --subsample 0.01; unset REFCV6_REMAP_OVERRIDES
# 2. the TRAIN fit (CPU/GPU light)
if [ -s $D/out/train_fitpass.acc.pt ] && [ ! -s $D/out/fit_map.json ]; then
  flock $LOCK $PY fit_map.py --train-acc $D/out/train_fitpass.acc.pt --out $D/out/fit_map.json \
      > $D/logs/fit_map.log 2>&1; echo $? > $D/status/fit_map.exit
fi
[ -s $D/out/fit_map.json ] || { echo "[chain] NO fit_map.json -- stopping"; exit 3; }
# 3. EVAL-DIAG at the final checkpoint, scored with the TRAIN fit
pass eval_final eval_diag $RUN/ckpt.pt --fit $D/out/fit_map.json
# 4. TRAIN-DIAG with the fit (exact per-episode counts for every decision on TRAIN)
export REFCV6_REMAP_OVERRIDES=$TRAIN_REMAP; pass train_final train_diag $RUN/ckpt.pt --fit $D/out/fit_map.json --no-mf; unset REFCV6_REMAP_OVERRIDES
# 5. M-e: EVAL-DIAG at the milestones (no fit)
for s in 5000 15000 20000 30000; do
  pass eval_s$s eval_diag $RUN/ckpt_$s.pt --no-mf
done
# 6. box sensitivity fit set (A10's banked 256 TRAIN windows)
export REFCV6_REMAP_OVERRIDES=$CAL_REMAP; pass cal256_final train_calib256 $RUN/ckpt.pt --no-mf; unset REFCV6_REMAP_OVERRIDES
# 7. determinism control: the in-run subset again
pass inrun_final_rep eval_inrun $RUN/ckpt.pt --no-mf
echo "[chain] ALLDONE $(date -u +%H:%M:%S)"
