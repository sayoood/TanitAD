#!/bin/bash
# route-following capture chain on Thor; each pass takes the shared GPU lock for ITSELF only.
# exit codes go to status/<tag>.exit; the ARTIFACT (<tag>.json) is the evidence a pass ran.
set -u
R=/home/nvidia/refcv7_post/route
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
export REFCV6_REPO=/home/nvidia/refcv7_run/fec3a0dccf REFCV6_KIT=/home/nvidia
export PYTHONPATH=/home/nvidia/refcv7_run/fec3a0dccf/stack OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd $R/code
pass () {   # tag split seed
  local tag=$1 split=$2 seed=$3
  if [ -s $R/out/$tag.json ]; then echo "[chain] $tag banked -- skip"; return 0; fi
  echo "[chain] $(date -u +%H:%M:%S) start $tag"
  flock $LOCK $PY run_route.py --split $split --seed $seed --tag $tag --out $R/out > $R/logs/$tag.log 2>&1 < /dev/null
  local rc=$?
  echo $rc > $R/status/$tag.exit
  echo "[chain] $(date -u +%H:%M:%S) end $tag rc=$rc artifact=$( [ -s $R/out/$tag.json ] && echo yes || echo NO )"
}
pass eval_s0 eval_diag 0
export REFCV6_REMAP_OVERRIDES=/home/nvidia/refcv7_post/diag/cache/remap_train_diag.json
pass train_s0 train_diag 0
unset REFCV6_REMAP_OVERRIDES
pass reel_s0 reel 0
pass eval_s1 eval_diag 1
echo "[chain] ALLDONE $(date -u +%H:%M:%S)"
