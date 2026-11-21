#!/bin/bash
# STRAT1 N1 at seeds 10 / 11 (MM 2026-11-16 ~22:50Z; PREREG addendum PREREG_ADDENDUM_N1S10.md, written before either arm trains): N1_s10 (init M2b_s0), N1_s11 (init M2b_s1).
# Same tree, inputs, checker, mint, runner and post check as strat1_s10.sh (tree new4 = a4357e79, G-SUITE result gsuite_result6.json for that tree).
# Waits first for the DSCALE10K block-1 gate's GPU work to end (ZZ FINISH-GATE DONE|FAIL ZZ) and for an idle GPU, so the pod runs one GPU job at a time.
# usage: nohup setsid bash /root/strat1/strat1_n1s10.sh < /dev/null > /root/strat1/logs/n1s10.log 2>&1 &
# Ends with `ZZ STRAT1 N1S10 DONE ZZ` or `ZZ STRAT1 N1S10 FAIL ZZ <why>`. A STOP file at /root/strat1/chains/STOP ends it between arms.
set -u
R=/root/strat1; TREE=$R/new4; INP=$R/inputs; RUNS=$R/runs; CACHE=/root/data/cache_pod
PY=/root/venv/bin/python; CHK="$TREE/stack/scripts/strat1_check.py"
COMMIT=a4357e7986f3aa30dab2ed83fde2351fda8d5662
SUITE=$R/gsuite_result6.json
ARMS="N1:s10 N1:s11"
fail() { echo "ZZ STRAT1 N1S10 FAIL ZZ $1"; exit 1; }
echo "$(date -u +%FT%TZ) waiting for the DSCALE10K gate's GPU work"
n=0; until tail -c 400 /root/dscale10k/gate/finish_gate.log 2>/dev/null | grep -qE "ZZ FINISH-GATE (DONE|FAIL) ZZ"; do n=$((n+1)); [ $n -gt 720 ] && fail "no FINISH-GATE sentinel after 6 h"; sleep 30; done
echo "$(date -u +%FT%TZ) gate sentinel: $(tail -c 400 /root/dscale10k/gate/finish_gate.log | grep -oE 'ZZ FINISH-GATE (DONE|FAIL) ZZ' | tail -1)"
# 0. the tree is still the a4357e79 tree the suite result binds (the two seeds-commit files equal their blobs)
sha256sum -c $R/fix6.sha256 > $R/logs/fix6_verify_n1.log 2>&1 || fail "new4 files are not the a4357e79 blobs"
[ -s "$SUITE" ] || fail "no G-SUITE result $SUITE"
# 2. inputs still byte-equal to their sources
sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify_n1.log 2>&1 || fail "inputs sha256 mismatch"
unset CUDA_VISIBLE_DEVICES
mkdir -p $R/chains $RUNS
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
  init=$INP/M2b_s0/ckpt_last.pt; case $seed in s1|s11) init=$INP/M2b_s1/ckpt_last.pt;; esac
  $PY $CHK post $kind --init $init --final $RUNS/$tag/train/ckpt_last.pt > $RUNS/post_$tag.log 2>&1 || fail "post check $tag: $(tail -c 300 $RUNS/post_$tag.log)"
  echo "ZZ ARM $tag DONE ZZ" | tee $RUNS/$tag/ARM_DONE
done
echo "ZZ STRAT1 N1S10 DONE ZZ"
