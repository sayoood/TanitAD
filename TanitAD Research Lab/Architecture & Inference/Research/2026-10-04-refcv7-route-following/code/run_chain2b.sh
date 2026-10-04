#!/bin/bash
# eval seed 0 again WITH the g_tac capture (train_s0 / reel_s0 / eval_s1 already start with the g_tac code);
# its fan must be bit-identical to eval_s0 (same per-window seed)
set -u
R=/home/nvidia/refcv7_post/route
until grep -q ALLDONE $R/logs/chain.out; do sleep 30; done
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
export REFCV6_REPO=/home/nvidia/refcv7_run/fec3a0dccf REFCV6_KIT=/home/nvidia
export PYTHONPATH=/home/nvidia/refcv7_run/fec3a0dccf/stack OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd $R/code
tag=eval_s0g
if [ ! -s $R/out/$tag.json ]; then
  echo "[chain2b] $(date -u +%H:%M:%S) start $tag"
  flock $LOCK $PY run_route.py --split eval_diag --seed 0 --tag $tag --out $R/out > $R/logs/$tag.log 2>&1 < /dev/null
  echo $? > $R/status/$tag.exit
  echo "[chain2b] $(date -u +%H:%M:%S) end $tag artifact=$( [ -s $R/out/$tag.json ] && echo yes || echo NO )"
fi
echo "[chain2b] ALLDONE $(date -u +%H:%M:%S)"
