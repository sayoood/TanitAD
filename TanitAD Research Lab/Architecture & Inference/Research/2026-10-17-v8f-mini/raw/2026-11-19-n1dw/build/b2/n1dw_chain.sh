#!/bin/bash
# N1DW chain (PREREG_V8F_N1DW section 4, card sha256 59f2b7f9 + Amendment 1). Pattern: navrbd_chain.sh.
# 4 arms in the card order: N1DW4_s0 N1DW4_s1 N1DW2_s0 N1DW2_s1. Each arm GPU part (mint if no token, run, post, zerograd gate) runs under `flock /root/gpu_queue.lock` (n1dw_arm.sh): one job at a time, released between arms.
# After each arm the D1-HO T1-kin dumps (follow, then the zero-channel twin for Z-USE) run on pod CPU in a detached worker, so the GPU lock is not held for them. EVAL-141 dumps are NOT run here: in NAVRBD they were
# a dev-box step (navrbd_dumps.py / navrbd_t0.py on the pulled checkpoints); this chain leaves the four checkpoints in runs/<arm>/train/ckpt_last.pt for it.
# Idempotent: an arm with ARM_DONE + checkpoint is skipped; a partial train/ or logs/ without ARM_DONE is moved aside (runs/aside_<tag>_<time>/), the gate token is kept.
# Needs: the N1DW G-SUITE sentinel DONE, the tree sha256, the inputs sha256, no STOP file at /root/strat1/chains/STOP.
# STAGED, NOT LAUNCHED. By hand (the MM sets the priority marker first):
#   nohup setsid bash /root/strat1/n1dw/n1dw_chain.sh < /dev/null > /root/strat1/logs/n1dw.log 2>&1 &
# Ends with `ZZ N1DW CHAIN DONE ZZ` or `ZZ N1DW CHAIN FAIL ZZ <reason>`.
cd /root/strat1/n1dw || exit 1
set -u
R=/root/strat1; N=$R/n1dw; TREE=$N/tree; INP=$R/inputs; RUNS=$N/runs
ARMS="N1DW4:s0 N1DW4:s1 N1DW2:s0 N1DW2:s1"
fail() { echo "ZZ N1DW CHAIN FAIL ZZ $1"; exit 1; }
tail -c 600 $R/logs/n1dw_gsuite.log 2>/dev/null | grep -q "ZZ N1DW GSUITE DONE ZZ" || fail "the N1DW G-SUITE has not passed (logs/n1dw_gsuite.log)"
[ -s $N/gsuite_result_n1dw.json ] || fail "no G-SUITE result"
(cd $TREE && sha256sum -c $N/n1dw_tree.sha256) > $R/logs/n1dw_tree_verify.log 2>&1 || fail "the 10 N1DW files are not the 134f5da1 blobs (logs/n1dw_tree_verify.log)"
/root/venv/bin/python $N/blobcheck.py $TREE $N/all_blobs.txt > $R/logs/n1dw_blobcheck.log 2>&1 || fail "tree blob check (logs/n1dw_blobcheck.log)"
sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify_n1dw.log 2>&1 || fail "inputs sha256 mismatch"
(cd $INP/strat1 && sha256sum -c $R/navrb2/navrb2_inputs.sha256 && sha256sum -c $N/navrbd_inputs.sha256) >> $R/logs/inputs_verify_n1dw.log 2>&1 || fail "navtext_nav_rb(d) sha256 mismatch"
mkdir -p $R/chains $RUNS $N/out
echo "$(date -u +%FT%TZ) chain start commit=134f5da1"
for arm in $ARMS; do
  kind=${arm%%:*}; seed=${arm##*:}; tag=${kind}_${seed}
  [ -e $R/chains/STOP ] && fail "STOP file before $tag"
  if [ -e $RUNS/$tag/ARM_DONE ] && [ -s $RUNS/$tag/train/ckpt_last.pt ]; then
    echo "skip $tag (ARM_DONE, ckpt present)"
  else
    if [ -d $RUNS/$tag/train ] || [ -d $RUNS/$tag/logs ]; then
      A=$RUNS/aside_${tag}_$(date -u +%H%M%S); mkdir -p $A
      for d in train logs; do [ -e $RUNS/$tag/$d ] && mv $RUNS/$tag/$d $A/; done
      rm -f $RUNS/$tag/ARM_DONE
      echo "partial $tag moved aside to $A"
    fi
    echo "$(date -u +%T) waiting for the GPU lock for $tag"
    flock /root/gpu_queue.lock bash $N/n1dw_arm.sh $kind $seed || fail "arm $tag (see runs/*_$tag.log)"
  fi
  # CPU dumps in a detached worker (own CPU lock), started once per arm
  if [ ! -e $N/out/DUMPS_DONE_$tag ] && [ ! -e $N/out/dumps_started_$tag ]; then
    touch $N/out/dumps_started_$tag
    nohup setsid bash $N/n1dw_dumps_worker.sh $tag < /dev/null > $N/out/dumps_worker_$tag.log 2>&1 &
  fi
done
# wait for every dump worker (cap 8 h); a failed dump fails the chain
d=0
for i in $(seq 1 960); do
  d=0; for arm in $ARMS; do tag=${arm%%:*}_${arm##*:}; [ -e $N/out/DUMPS_DONE_$tag ] && d=$((d+1)); done
  [ $d -eq 4 ] && break
  ls $N/out/t1/FAIL_* $N/out/t1_zero/FAIL_* > /dev/null 2>&1 && fail "a D1-HO dump failed ($(ls $N/out/t1/FAIL_* $N/out/t1_zero/FAIL_* 2>/dev/null | tr '\n' ' '))"
  sleep 30
done
[ $d -eq 4 ] || fail "dumps not finished within 8 h"
echo "ZZ N1DW CHAIN DONE ZZ"
