#!/bin/bash
# n1dw_arm.sh KIND SEED -- the GPU part of ONE N1DW arm. Called by n1dw_chain.sh as `flock /root/gpu_queue.lock bash n1dw_arm.sh KIND SEED`, so it holds the pod GPU lock for the whole arm.
# gpu idle wait -> mint (skipped when a token exists) -> pair on the mint argv -> run (refuses without the token) -> post -> zerograd gate (control e_nav_rb.weight) -> ZZ ARM <tag> DONE ZZ. Exit 0 = arm done.
R=/root/strat1; N=$R/n1dw; TREE=$N/tree; INP=$R/inputs; RUNS=$N/runs; CACHE=/root/data/cache_pod
PY=/root/venv/bin/python; CHK="$TREE/stack/scripts/strat1_check.py"; ZG="$TREE/stack/scripts/zerograd_gate.py"
COMMIT=134f5da1ee50a3c2fa242bb340a26dd6e4072ade
SUITE=$N/gsuite_result_n1dw.json
kind=$1; seed=$2; tag=${kind}_${seed}
REF=$R/navrbd/runs/N1D_$seed/argv.json
fail() { echo "ZZ ARM $tag FAIL ZZ $1"; exit 1; }
unset CUDA_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1
COMMON="--root $TREE --inputs $INP --cache-dir $CACHE --runs $RUNS --commit $COMMIT --suite $SUITE --py $PY --gpu 0"
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-99999}" -lt 3500 ]; }
mkdir -p $RUNS/$tag
n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle (>= 3500 MiB used)"; sleep 10; done
if [ ! -f $RUNS/$tag/gate/gate_token.json ]; then
  echo "$(date -u +%T) mint $tag"
  $PY $CHK mint $kind $seed $COMMON --steps 3000 > $RUNS/mint_$tag.log 2>&1 || fail "mint refused: $(tail -c 300 $RUNS/mint_$tag.log)"
fi
$PY $CHK pair $kind --argv-json $RUNS/$tag/argv.json --ref-argv-json $REF > $RUNS/pair_$tag.log 2>&1 || fail "pairing: $(tail -c 300 $RUNS/pair_$tag.log)"
echo "$(date -u +%T) run $tag"
$PY $CHK run $kind $seed $COMMON --steps 3000 > $RUNS/run_$tag.log 2>&1; rc=$?
[ $rc -eq 0 ] || fail "run rc=$rc: $(tail -c 300 $RUNS/run_$tag.log)"
init=$INP/M2b_s0/ckpt_last.pt; case $seed in s1) init=$INP/M2b_s1/ckpt_last.pt;; esac
$PY $CHK post $kind --init $init --final $RUNS/$tag/train/ckpt_last.pt > $RUNS/post_$tag.log 2>&1 || fail "post check: $(tail -c 300 $RUNS/post_$tag.log)"
PYTHONPATH=$TREE/stack:$TREE $PY $ZG --ckpt $RUNS/$tag/train/ckpt_last.pt --allow '.*' --control 'e_nav_rb\.weight' --json-out $RUNS/$tag/zerograd_n1dw.json > $RUNS/zerograd_$tag.log 2>&1 || fail "zerograd gate: $(tail -c 300 $RUNS/zerograd_$tag.log)"
echo "ZZ ARM $tag DONE ZZ" | tee $RUNS/$tag/ARM_DONE
