#!/bin/bash
# dry_mint_n1dm.sh: run the arm's OWN mint for N1DM_s0 by hand (the exact n1dm_arm.sh command and env), under the GPU queue lock, BEFORE the launch (N1DH Amendment 3 lesson: a mint that refuses must refuse here, not in the chain).
# On PASS the gate token stays in runs/N1DM_s0/gate and the chain's arm skips its mint (it only mints when no token exists). Needs the same go as the chain (it holds the GPU lock for ~5 min).
R=/root/strat1; N=$R/n1dm; TREE=$N/tree; INP=$R/inputs; RUNS=$N/runs; CACHE=/root/data/cache_pod
PY=/root/venv/bin/python; CHK="$TREE/stack/scripts/strat1_check.py"
COMMIT=@@N1DM_COMMIT@@
SUITE=$N/gsuite_result_n1dm.json
ps -eo args | grep -q "[n]1dm_chain.sh" && { echo "an n1dm_chain.sh is running; refusing"; exit 1; }
[ -f $RUNS/N1DM_s0/gate/gate_token.json ] && { echo "a token exists; refusing"; exit 1; }
unset CUDA_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1
COMMON="--root $TREE --inputs $INP --cache-dir $CACHE --runs $RUNS --commit $COMMIT --suite $SUITE --py $PY --gpu 0"
exec 9>/root/gpu_queue.lock
flock -w 60 9 || { echo "GPU queue lock busy"; exit 1; }
u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); echo "gpu mem used $u MiB"; [ "${u:-99999}" -lt 3500 ] || { echo "GPU not idle"; exit 1; }
echo "$(date -u +%FT%TZ) dry mint N1DM_s0"
$PY $CHK mint N1DM s0 $COMMON --steps 3000 > $RUNS/mint_N1DM_s0.log 2>&1; rc=$?
echo "$(date -u +%FT%TZ) mint rc=$rc"
echo "ZZ DRYMINT rc=$rc ZZ"
