#!/bin/bash
# n1dm_chain_early.sh -- MM 2026-11-21, PI: "Dont let the pod idle, use it effectively". The N1DM chain's steps tree, inputs, G-SUITE, pairing and the
# two arms, run WHILE the ROLLHIST-K chain is in its seed-1 bank step (4 CPU-bound shards, ~2.2 of 20 GB GPU, GPU util <= 4 %), on a PRIVATE lock
# instead of /root/gpu_queue.lock (RHK holds that one). Same step functions and done-markers as n1dm_chain.sh, so the main chain, launched after RHK
# ends, skips what this finished and runs the rest (any arm left + the CPU dumps) under the shared lock. Arms start only while RHK is in a bank step;
# each arm keeps n1dm_arm.sh's GPU-idle wait (< 3500 MiB), and RHK's arm has the same wait, so two trainings never overlap. A watchdog kills THIS
# chain's trainer if the GPU passes 19000 MiB (protects the bank shards). No dumps here (no eval on a pod where an arm may train). Log n1dm_chain_early.log.
# Last line: ZZ N1DM EARLY DONE ZZ | ZZ N1DM EARLY STOP ZZ <why> | ZZ N1DM FAIL ZZ <why>.
R=/root/strat1; N=$R/n1dm; TREE=$N/tree; INP=$R/inputs; RUNS=$N/runs; PY=/root/venv/bin/python
LOG=$N/n1dm_chain_early.log; D=$N/done; RL=/root/rollhist/rh_k_chain.log
mkdir -p $N $D $RUNS $N/out $R/logs
exec >> $LOG 2>&1
say() { echo "$(date -u +%FT%TZ) $*"; }
fail() { echo "ZZ N1DM FAIL ZZ $1"; exit 1; }
step() { local n=$1; shift; [ -e $D/$n.done ] && { say "skip $n (done)"; return 0; }; say "step $n"; "$@" || fail "step $n"; date -u +%FT%TZ > $D/$n.done; }
COMMIT=573ad839b03f3f97ac160a399d6fd19077f7ad00
case "$COMMIT" in @@*) fail "COMMIT placeholder not substituted (run n1dm_stage.sh)";; esac
say "EARLY chain start commit=$COMMIT (n1dm, private lock, during the RHK seed-1 bank)"
rhk_in_bank() { case "$(grep -E ' step | arm ' $RL | tail -n1)" in *' step bank_k_'*) return 0;; *) return 1;; esac; }
[ -e /root/rollhist/runs/RHK_s0/ARM_DONE ] && rhk_in_bank || fail "RHK is not in a bank step (last: $(grep -E ' step | arm ' $RL | tail -n1))"
exec 9>$N/early.lock
flock -n 9 || fail "another early chain holds $N/early.lock"
say "private early lock held; RHK holds /root/gpu_queue.lock (bank step)"
[ -e $R/chains/STOP ] && fail "STOP file at $R/chains/STOP"
# 1. tree + kit + inputs
chk_tree() {
  (cd $TREE && sha256sum -c $N/n1dm_tree.sha256) > $R/logs/n1dm_tree_verify.log 2>&1 || return 1
  $PY $N/blobcheck.py $TREE $N/all_blobs.txt > $R/logs/n1dm_blobcheck.log 2>&1 || return 1
  [ "$(find $TREE -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $N/all_blobs.txt)" ] || return 1      # no extra file; all_blobs.txt lists the eval-141 grid
  [ "$(find "$TREE/TanitAD Research Lab" -type f | wc -l)" = 1 ] || return 1      # exactly one Research Lab file: the eval-141 grid (K-B3), listed in all_blobs.txt
  (cd $N/hb && echo "74fd8445e377e763d17b1fa4c60753f23b8240eb94d0bdf7f4633a9e3ef80608 *v8f_t1_eval_hb.py" | sha256sum -c) >> $R/logs/n1dm_tree_verify.log 2>&1 || return 1
  (cd $N && echo "8e78e69e202e6fa1422ae680d6bbda3c24af0786ae5234b9f744fefb4004f53c *kclip.py" | sha256sum -c) >> $R/logs/n1dm_tree_verify.log 2>&1 || return 1
}
step tree chk_tree
chk_inputs() {
  sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify_n1dm.log 2>&1 || return 1
  (cd $INP/strat1 && sha256sum -c $R/navrb2/navrb2_inputs.sha256 && sha256sum -c $R/n1dw/navrbd_inputs.sha256) >> $R/logs/inputs_verify_n1dm.log 2>&1 || return 1
}
step inputs chk_inputs
# 2. G-SUITE (CPU) on the clean tree, against the 1f217bf0 junit of N1DH's own run (junit/n1dh); it holds no GPU, but the lock is ours
gs() { bash $N/n1dm_gsuite.sh > $R/logs/n1dm_gsuite.driver.log 2>&1; tail -n1 $R/logs/n1dm_gsuite.log | grep -q "ZZ N1DM GSUITE DONE ZZ"; }
step gsuite gs
[ -s $N/gsuite_result_n1dm.json ] || fail "no G-SUITE result"
# 3. argv + pairing + negative controls
pr() { bash $N/pairing_n1dm.sh > $N/b2/pairing.out 2>&1; }
mkdir -p $N/b2
step pairing pr
# 4. the arms, one at a time
for seed in s0 s1; do
  tag=N1DM_$seed
  [ -e $R/chains/STOP ] && fail "STOP file before $tag"
  if [ -e $RUNS/$tag/ARM_DONE ] && [ -s $RUNS/$tag/train/ckpt_last.pt ]; then say "skip $tag (ARM_DONE, ckpt present)"; continue; fi
  if [ -d $RUNS/$tag/train ] || [ -d $RUNS/$tag/logs ]; then
    A=$RUNS/aside_${tag}_$(date -u +%H%M%S); mkdir -p $A
    for d in train logs; do [ -e $RUNS/$tag/$d ] && mv $RUNS/$tag/$d $A/; done
    rm -f $RUNS/$tag/ARM_DONE; say "partial $tag moved aside to $A"
  fi
  rhk_in_bank || { say "RHK has left its bank step: $tag is left to the main chain"; echo "ZZ N1DM EARLY STOP ZZ RHK left its bank step before $tag"; exit 0; }
  say "arm $tag"
  ( while sleep 15; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d " "); if [ "${u:-0}" -gt 19000 ]; then for p in $(grep -l "runs/$tag/train" /proc/[0-9]*/cmdline 2>/dev/null | cut -d/ -f3); do kill $p 2>/dev/null; done; echo "$(date -u +%FT%TZ) WATCHDOG GPU ${u} MiB > 19000: killed the $tag trainer" >> $N/early_watchdog.log; fi; done ) &
  WD=$!
  bash $N/n1dm_arm.sh $seed; arc=$?
  kill $WD 2>/dev/null
  [ $arc = 0 ] || fail "arm $tag (see $RUNS/*_$tag.log)"
done
echo "ZZ N1DM EARLY DONE ZZ"
