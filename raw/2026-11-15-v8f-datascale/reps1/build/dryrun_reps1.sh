#!/bin/bash
# dryrun_reps1.sh -- control-flow dry run of blockreps1.sh and queue_reps1.sh (fake leaf operations, the REAL chainlib.py and dscale_banker.py, a directory as the hub). Pod, CPU only: no GPU, no trainer, no real data.
# Writes ONLY under /root/dscale10k/blockreps1/ (fake roots fake_root and fake_q) and deletes them at the end. One PASS / FAIL line per scenario, then `ZZ DRYRUN-REPS1 DONE ZZ`.
R=/root/dscale10k; B=$R/blockreps1; F=$B/fake_root; Q=$B/fake_q; PY=/root/venv/bin/python
ok=0; bad=0
chk() { if [ "$2" = "$3" ]; then echo "PASS $1"; ok=$((ok+1)); else echo "FAIL $1: got [$2] want [$3]"; bad=$((bad+1)); fi; }
has() { if echo "$2" | grep -q -- "$3"; then echo "PASS $1"; ok=$((ok+1)); else echo "FAIL $1: [$3] not in [$(echo "$2" | tail -c 300)]"; bad=$((bad+1)); fi; }
mk() {
  rm -rf "$F"; mkdir -p $F/argv $F/dry $F/gate/tokens $F/chains
  cp $R/chainlib.py $F/chainlib.py; cp $R/dscale_banker.py $F/dscale_banker.py; cp $B/fake_leaf_reps1.sh $F/fake_leaf.sh
  for a in UE_s1 UM_s1; do echo '{"argv": ["--seed", "1", "--steps", "61568"]}' > $F/argv/$a.A.json; done
}
run() { DS_FAKE=1 DS_ROOT=$F DS_PY=$PY DS_POLL=0 DS_GPU_TRIES=2 DS_RP=$B bash $B/blockreps1.sh 2>&1; }
mk; out=$(run); last=$(echo "$out" | tail -n 1)
chk "1 sentinel DONE" "$last" "ZZ DSCALE10K REPS1 DONE ZZ"
order=$(grep -E "fake (prep|trainer|um_check)" $F/fake_order.log | sed -E 's/^[0-9:]+ fake //' | tr '\n' ',')
chk "1 order: prep, UE_s1, um pre, UM_s1, um post" "$order" "prep,trainer UE_s1 A,um_check pre,trainer UM_s1 A,um_check post,"
chk "1 two markers written by the banker" "$(ls $F/bank_ok | tr '\n' ' ')" "UE_s1.A.sha256 UM_s1.A.sha256 "
chk "1 hub holds the two checkpoints" "$(ls $F/fakehub/*/*/ckpt_last.pt | wc -l)" "2"
chk "1 ARM_DONE of both" "$(ls $F/runs/UE_s1/ARM_DONE $F/runs/UM_s1/ARM_DONE | wc -l)" "2"
chk "1 checkpoints kept at step 61568" "$($PY -c "import torch,glob;print(sorted({torch.load(p,weights_only=False)['step'] for p in glob.glob('$F/runs/*/A/ckpt_last.pt')}))")" "[61568]"
chk "1 nothing pending" "$(echo "$out" | grep -c 'banking pending (no bank_ok marker): $')" "1"
out=$(run); chk "2 rerun DONE" "$(echo "$out" | tail -n 1)" "ZZ DSCALE10K REPS1 DONE ZZ"; has "2 skip" "$out" "skip UE_s1 (ARM_DONE)"; chk "2 no second trainer run" "$(grep -c 'fake trainer' $F/fake_order.log)" "2"
mk; out=$(DS_FAKE_FAIL="prep:-.-" run); has "3 prep fail" "$(echo "$out" | tail -n 1)" "REPS1 prep"; chk "3 nothing trained" "$(cat $F/fake_order.log 2>/dev/null | grep -c "fake trainer")" "0"
mk; for a in UE_s1 UM_s1; do mkdir -p $F/gate/tokens/$a.A; echo '{}' > $F/gate/tokens/$a.A/gate_token.json; done; out=$(run); chk "4 DONE without prep" "$(echo "$out" | tail -n 1)" "ZZ DSCALE10K REPS1 DONE ZZ"; chk "4 no prep call" "$(grep -c 'fake prep' $F/fake_order.log)" "0"
mk; out=$(DS_FREE_GB=10 run); has "5 disk guard" "$(echo "$out" | tail -n 1)" "free disk 10 GB < 60 GB"
mk; touch $F/chains/STOP; out=$(run); has "6 STOP before anything" "$(echo "$out" | tail -n 1)" "STOP file before"
mk; out=$(DS_FAKE_FAIL="run:UE_s1.A" run); has "7 failed UE" "$(echo "$out" | tail -n 1)" "UE_s1 stage A rc=7"; chk "7 UM never ran" "$(grep -c 'trainer UM_s1' $F/fake_order.log)" "0"
mk; out=$(DS_FAKE_FAIL="run:UM_s1.A" run); has "8 failed UM" "$(echo "$out" | tail -n 1)" "UM_s1 stage A rc=7"; chk "8 UE banked before" "$(ls $F/bank_ok | tr '\n' ' ')" "UE_s1.A.sha256 "
mk; out=$(DS_FAKE_FAIL="um:post." run); has "9 um post FAIL ends the chain" "$(echo "$out" | tail -n 1)" "UM_s1 post check refused"
chk "9 UM checkpoint banked BEFORE the post check" "$(ls $F/bank_ok | tr '\n' ' ')" "UE_s1.A.sha256 UM_s1.A.sha256 "
chk "9 no ARM_A_DONE for UM_s1 after a post FAIL" "$([ -e $F/runs/UM_s1/A/ARM_A_DONE ] && echo yes || echo no)" "no"
out=$(run); has "10 rerun after the post FAIL refuses to retrain over the output" "$(echo "$out" | tail -n 1)" "output exists without ARM_A_DONE"; chk "10 no extra trainer run" "$(grep -c 'trainer UM_s1' $F/fake_order.log)" "1"
mk; out=$(DS_FAKE_FAIL="um:pre." run); has "11 um pre FAIL refuses before UM trains" "$(echo "$out" | tail -n 1)" "UM_s1 pre check refused"; chk "11 UM never trained" "$(grep -c 'trainer UM_s1' $F/fake_order.log)" "0"
chk "12 chain checks the pairing proof" "$(grep -c 'PAIRING_PROOF.json' $B/blockreps1.sh)" "2"
chk "12 runner is the reps1 checker wrapper" "$(grep -c 'chk_arm_reps1.sh run' $B/blockreps1.sh)" "1"
# the queue wrapper: execs flock + chain; the lock is a fake file; a held lock blocks until released
rm -rf "$Q"; mkdir -p $Q; echo "echo CHAIN-RAN-UNDER-LOCK" > $Q/chain.sh
qrun() { QUEUE_LOCK=$Q/lock QUEUE_CHAIN=$Q/chain.sh QUEUE_POLL=1 QUEUE_MAXN=${1:-3} bash $B/queue_reps1.sh 2>&1; }
out=$(qrun); has "13 queue runs the chain" "$out" "CHAIN-RAN-UNDER-LOCK"
( flock $Q/lock sleep 4 ) & sleep 1; t0=$(date +%s); out=$(qrun); t1=$(date +%s); wait
has "14 chain ran after the lock" "$out" "CHAIN-RAN-UNDER-LOCK"; chk "14 it waited for the held lock (>= 2 s)" "$([ $((t1-t0)) -ge 2 ] && echo yes || echo no)" "yes"
echo "x" > $Q/log; out=$(QUEUE_LOG=$Q/log QUEUE_PATTERN='^ZZ X DONE ZZ' QUEUE_LOCK=$Q/lock QUEUE_CHAIN=$Q/chain.sh QUEUE_POLL=1 QUEUE_MAXN=2 bash $B/queue_reps1.sh 2>&1); has "15 queue wait times out without the line" "$out" "ZZ DSCALE10K REPS1 FAIL ZZ queue"
rm -rf "$F" "$Q"
echo "scenarios PASS $ok FAIL $bad"
echo "ZZ DRYRUN-REPS1 DONE ZZ"
