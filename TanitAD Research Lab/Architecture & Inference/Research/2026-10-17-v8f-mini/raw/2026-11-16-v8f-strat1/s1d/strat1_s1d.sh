#!/bin/bash
# STRAT1 S1d (PREREG_V8F_STRAT1, the committed next step of the SEEDS read outcome COSTS-HEADS): S1 with --ctx-detach on, seeds s0, s1, s10, s11
# (inits M2b_s0 for s0/s10, M2b_s1 for s1/s11).  Tree new9 = new4 + the 4dedac45 checker (s1d2.sha256); G-SUITE result gsuite_result_s1d.json for that commit.
# Same inputs, mint, runner and post check as strat1_s10.sh / strat1_n1s10.sh.  STAGED, NOT LAUNCHED: it needs the PI's training go.
# Launch ONLY under the pod GPU queue lock (one GPU job at a time; DSCALE10K block 1 takes the same lock):
#   nohup setsid flock /root/gpu_queue.lock bash /root/strat1/strat1_s1d.sh < /dev/null > /root/strat1/logs/s1d.log 2>&1 &
# Ends with `ZZ STRAT1 S1D DONE ZZ` or `ZZ STRAT1 S1D FAIL ZZ <why>`.  A STOP file at /root/strat1/chains/STOP ends it between arms.
set -u
R=/root/strat1; TREE=$R/new9; INP=$R/inputs; RUNS=$R/runs; CACHE=/root/data/cache_pod
PY=/root/venv/bin/python; CHK="$TREE/stack/scripts/strat1_check.py"
COMMIT=4dedac45dc8d4f6922711ed9af63e0ab54e435be
SUITE=$R/gsuite_result_s1d.json
ARMS="S1d:s0 S1d:s1 S1d:s10 S1d:s11"
fail() { echo "ZZ STRAT1 S1D FAIL ZZ $1"; exit 1; }
tail -c 600 $R/logs/s1d_gsuite.log 2>/dev/null | grep -q "ZZ S1D GSUITE DONE ZZ" || fail "the 4dedac45 G-SUITE has not passed (logs/s1d_gsuite.log)"
[ -s "$SUITE" ] || fail "no G-SUITE result $SUITE"
(cd $TREE && sha256sum -c $R/tmp/s1dsuite/s1d2.sha256) > $R/logs/s1d_tree_verify.log 2>&1 || fail "new9 files are not the 4dedac45 blobs"
sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify_s1d.log 2>&1 || fail "inputs sha256 mismatch"
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
echo "ZZ STRAT1 S1D DONE ZZ"
