#!/bin/bash
# dryrun_block3.sh -- control-flow dry run of block3.sh (fake leaf operations, the REAL chainlib.py and dscale_banker.py, a directory as the hub) and of queue_block3.sh. Pod, CPU only: no GPU, no trainer, no real data.
# Writes ONLY under /root/dscale10k/block3/ (fake roots fake_root3 and fake_q) and deletes them at the end. One PASS / FAIL line per scenario, then `ZZ DRYRUN3 DONE ZZ`.
R=/root/dscale10k; B3=$R/block3; F=$B3/fake_root3; Q=$B3/fake_q; PY=/root/venv/bin/python
ok=0; bad=0
chk() { if [ "$2" = "$3" ]; then echo "PASS $1"; ok=$((ok+1)); else echo "FAIL $1: got [$2] want [$3]"; bad=$((bad+1)); fi; }
has() { if echo "$2" | grep -q -- "$3"; then echo "PASS $1"; ok=$((ok+1)); else echo "FAIL $1: [$3] not in [$(echo "$2" | tail -c 300)]"; bad=$((bad+1)); fi; }
mk() {
  rm -rf "$F"; mkdir -p $F/argv $F/dry $F/gate/tokens $F/chains
  cp $R/chainlib.py $F/chainlib.py; cp $R/dscale_banker.py $F/dscale_banker.py; cp $B3/fake_leaf3.sh $F/fake_leaf.sh
  for a in UE_s0 BL_s0; do echo '{"argv": ["--cache-dir", "/c", "--tracks", "/t", "--join-train", "/j", "--seed", "0", "--steps", "61568"]}' > $F/argv/$a.A.json; echo '{"n_clips": 4512, "n_windows": 306566}' > $F/dry/$a.A.dry.json; done
}
run() { DS_FAKE=1 DS_ROOT=$F DS_PY=$PY DS_POLL=0 DS_GPU_TRIES=2 bash $B3/block3.sh 2>&1; }
mk; out=$(run); last=$(echo "$out" | tail -n 1)
chk "1 sentinel DONE" "$last" "ZZ DSCALE10K BLOCK3 DONE ZZ"
order=$(grep -E "fake (prep|trainer|bank|mint|selfcheck)" $F/fake_order.log | sed -E 's/^[0-9:]+ fake //' | tr '\n' ',')
chk "1 order: prep, UE_s0 A, BL_s0 A, nothing else (no bank, no stage B, no mint)" "$order" "prep,trainer UE_s0 A,trainer BL_s0 A,"
chk "1 two markers written by the banker" "$(ls $F/bank_ok | tr '\n' ' ')" "BL_s0.A.sha256 UE_s0.A.sha256 "
chk "1 hub holds the two checkpoints" "$(ls $F/fakehub/*/*/ckpt_last.pt | wc -l)" "2"
chk "1 ARM_DONE of both" "$(ls $F/runs/UE_s0/ARM_DONE $F/runs/BL_s0/ARM_DONE | wc -l)" "2"
chk "1 checkpoints kept at step 61568" "$($PY -c "import torch,glob;print(sorted({torch.load(p,weights_only=False)['step'] for p in glob.glob('$F/runs/*/A/ckpt_last.pt')}))")" "[61568]"
has "1 ARM lines" "$out" "ZZ ARM BL_s0 DONE ZZ"
chk "1 nothing pending" "$(echo "$out" | grep -c 'banking pending (no bank_ok marker): $')" "1"
out=$(run); chk "2 rerun DONE" "$(echo "$out" | tail -n 1)" "ZZ DSCALE10K BLOCK3 DONE ZZ"; has "2 skip" "$out" "skip UE_s0 (ARM_DONE)"; chk "2 no second trainer run" "$(grep -c 'fake trainer' $F/fake_order.log)" "2"
mk; out=$(DS_FAKE_FAIL="prep:-.-" run); has "3 prep fail" "$(echo "$out" | tail -n 1)" "block 3 prep"; chk "3 nothing trained" "$(cat $F/fake_order.log 2>/dev/null | grep -c "fake trainer")" "0"
mk; for a in UE_s0 BL_s0; do mkdir -p $F/gate/tokens/$a.A; echo '{}' > $F/gate/tokens/$a.A/gate_token.json; done; out=$(run); chk "4 DONE without prep" "$(echo "$out" | tail -n 1)" "ZZ DSCALE10K BLOCK3 DONE ZZ"; chk "4 no prep call" "$(grep -c 'fake prep' $F/fake_order.log)" "0"
mk; out=$(DS_FREE_GB=10 run); has "5 disk guard" "$(echo "$out" | tail -n 1)" "free disk 10 GB < 60 GB"
mk; touch $F/chains/STOP; out=$(run); has "6 STOP" "$(echo "$out" | tail -n 1)" "STOP file before"
mk; out=$(DS_FAKE_FAIL="run:UE_s0.A" run); has "7 failed UE" "$(echo "$out" | tail -n 1)" "UE_s0 stage A rc=7"; chk "7 BL never ran" "$(grep -c 'trainer BL_s0' $F/fake_order.log)" "0"
mk; out=$(DS_FAKE_FAIL="run:BL_s0.A" run); has "8 failed BL" "$(echo "$out" | tail -n 1)" "BL_s0 stage A rc=7"; chk "8 UE banked before" "$(ls $F/bank_ok | tr '\n' ' ')" "UE_s0.A.sha256 "
# the real runner refuses without a token (the fake emulates it); the real chk_arm3 selects the MM-ruled checker and the union-dry flag
chk "9 chk_arm3 checker path" "$(grep -c 'CHK=${DS_CHK:-$B3/study/dscale10k_check.py}' $B3/chk_arm3.sh)" "1"
chk "9 chk_arm3 passes --ue-dry for UE and BL" "$(grep -c 'UE_s0|BL_s0) MORE="$MORE --ue-dry' $B3/chk_arm3.sh)" "1"
# the queue wrapper: waits, ends on a C0REP line, then execs flock + chain; the lock is a fake file; a held lock blocks until released
rm -rf "$Q"; mkdir -p $Q; echo "echo CHAIN-RAN-UNDER-LOCK; echo \"\$(date +%s)\" > $Q/ran" > $Q/chain.sh
qrun() { QUEUE_LOG=$Q/c0rep.log QUEUE_LOCK=$Q/lock QUEUE_CHAIN=$Q/chain.sh QUEUE_POLL=1 QUEUE_MAXN=${1:-3} bash $B3/queue_block3.sh 2>&1; }
echo "2026-11-17T19:34:02Z C0REP queued behind the CKPT120 fix3 chain" > $Q/c0rep.log
out=$(qrun 2); has "10 queue waits then times out on a log without the end line" "$out" "ZZ DSCALE10K BLOCK3 FAIL ZZ queue"
echo "note: ZZ STRAT1 C0REP DONE ZZ quoted mid-line" >> $Q/c0rep.log; out=$(qrun 2); has "11 a mid-line quote does not release it" "$out" "ZZ DSCALE10K BLOCK3 FAIL ZZ queue"
echo "ZZ STRAT1 C0REP DONE ZZ" >> $Q/c0rep.log; out=$(qrun 3); has "12 DONE line releases it" "$out" "CHAIN-RAN-UNDER-LOCK"
rm -f $Q/ran; echo "ZZ STRAT1 C0REP FAIL ZZ x" > $Q/c0rep.log; out=$(qrun 3); has "13 FAIL line releases it too" "$out" "CHAIN-RAN-UNDER-LOCK"
rm -f $Q/ran; ( flock $Q/lock sleep 4 ) & sleep 1; t0=$(date +%s); out=$(qrun 3); t1=$(date +%s); wait
has "14 chain ran" "$out" "CHAIN-RAN-UNDER-LOCK"; chk "14 it waited for the held lock (>= 2 s)" "$([ $((t1-t0)) -ge 2 ] && echo yes || echo no)" "yes"
rm -rf "$F" "$Q"
echo "scenarios PASS $ok FAIL $bad"
echo "ZZ DRYRUN3 DONE ZZ"
