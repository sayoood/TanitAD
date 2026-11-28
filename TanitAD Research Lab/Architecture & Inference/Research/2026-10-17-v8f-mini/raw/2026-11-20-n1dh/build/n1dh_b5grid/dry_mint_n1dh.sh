#!/bin/bash
# MM 2026-11-21 (N1DH Amendment 3): run the arm's OWN mint for N1DH_s0 by hand (the exact n1dh_arm.sh command and env), under the GPU queue lock,
# before relaunching the chain. On PASS the gate token stays in runs/N1DH_s0/gate and the chain's arm skips its mint (it only mints when no token exists).
R=/root/strat1; N=$R/n1dh; TREE=$N/tree; INP=$R/inputs; RUNS=$N/runs; CACHE=/root/data/cache_pod
PY=/root/venv/bin/python; CHK="$TREE/stack/scripts/strat1_check.py"
COMMIT=1f217bf015ebd6d69b888e424a2675970ff72075
SUITE=$N/gsuite_result_n1dh.json
ps -eo args | grep -q "[n]1dh_chain.sh" && { echo "an n1dh_chain.sh is running; refusing"; exit 1; }
[ -f $RUNS/N1DH_s0/gate/gate_token.json ] && { echo "a token exists; refusing"; exit 1; }
unset CUDA_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1
COMMON="--root $TREE --inputs $INP --cache-dir $CACHE --runs $RUNS --commit $COMMIT --suite $SUITE --py $PY --gpu 0"
exec 9>/root/gpu_queue.lock
flock -w 60 9 || { echo "GPU queue lock busy"; exit 1; }
u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); echo "gpu mem used $u MiB"; [ "${u:-99999}" -lt 3500 ] || { echo "GPU not idle"; exit 1; }
echo "$(date -u +%FT%TZ) dry mint N1DH_s0"
$PY $CHK mint N1DH s0 $COMMON --steps 3000 > $RUNS/mint_N1DH_s0.log 2>&1; rc=$?
echo "$(date -u +%FT%TZ) mint rc=$rc"
echo "ZZ DRYMINT rc=$rc ZZ"
