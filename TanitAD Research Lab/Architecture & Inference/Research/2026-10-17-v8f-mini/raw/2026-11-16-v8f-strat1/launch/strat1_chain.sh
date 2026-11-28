#!/bin/bash
# STRAT1 7-arm chain on the pod. NOT LAUNCHED by the builder: the MM freezes the card first, then launches it detached:
#   nohup setsid bash /root/strat1/strat1_chain.sh < /dev/null > /root/strat1/logs/chain.log 2>&1 &
# One arm at a time. Each arm: mint (if no token), run THROUGH THE RUNNER (it refuses without a matching token), post check (moved identity), DONE marker.
# Ends with `ZZ STRAT1 CHAIN DONE ZZ` or `ZZ STRAT1 CHAIN FAIL ZZ <why>`. A STOP file at /root/strat1/chains/STOP ends it between arms.
set -u
R=/root/strat1; TREE=$R/tree; INP=$R/inputs; RUNS=$R/runs; CACHE=/root/data/cache_pod
PY=/root/venv/bin/python; CHK="$TREE/stack/scripts/strat1_check.py"
COMMIT=$(cat $R/COMMIT)
SUITE=$R/gsuite_result.json
ARMS="C0:s0 N1:s0 S1:s0 C0:s1 N1:s1 S1:s1 C0:rep"
export PYTHONPATH="$TREE/stack:$TREE" HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p $R/chains $RUNS
fail() { echo "ZZ STRAT1 CHAIN FAIL ZZ $1"; exit 1; }
COMMON="--root $TREE --inputs $INP --cache-dir $CACHE --runs $RUNS --commit $COMMIT --suite $SUITE --py $PY --gpu 0"
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-99999}" -lt 3500 ]; }
echo "$(date -u +%FT%TZ) chain start commit=$COMMIT"
for arm in $ARMS; do
  kind=${arm%%:*}; seed=${arm##*:}; tag=${kind}_${seed}
  [ -e $R/chains/STOP ] && fail "STOP file before $tag"
  [ -e $RUNS/$tag/ARM_DONE ] && { echo "skip $tag (ARM_DONE)"; continue; }
  n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle (>= 3500 MiB used) before $tag"; sleep 10; done
  if [ ! -f $RUNS/$tag/gate/gate_token.json ]; then
    echo "$(date -u +%T) mint $tag"
    $PY $CHK mint $kind $seed $COMMON --steps 3000 > $RUNS/mint_$tag.log 2>&1 || fail "mint $tag refused: $(tail -c 300 $RUNS/mint_$tag.log)"
  fi
  echo "$(date -u +%T) run $tag"
  $PY $CHK run $kind $seed $COMMON --steps 3000 > $RUNS/run_$tag.log 2>&1; rc=$?
  [ $rc -eq 0 ] || fail "run $tag rc=$rc: $(tail -c 300 $RUNS/run_$tag.log)"
  init=$INP/M2b_s0/ckpt_last.pt; case $seed in s1) init=$INP/M2b_s1/ckpt_last.pt;; esac
  $PY $CHK post $kind --init $init --final $RUNS/$tag/train/ckpt_last.pt > $RUNS/post_$tag.log 2>&1 || fail "post check $tag: $(tail -c 300 $RUNS/post_$tag.log)"
  echo "ZZ ARM $tag DONE ZZ" | tee $RUNS/$tag/ARM_DONE
done
echo "ZZ STRAT1 CHAIN DONE ZZ"
