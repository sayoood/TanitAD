#!/bin/bash
# dryrun_blockx.sh -- control-flow dry run of blockx.sh (fake leaf operations, the REAL chainlib.py and dscale_banker.py, a directory as the hub) and of queue_blockx.sh.  Forked from dryrun_block3.sh.
# Pod, CPU only: no GPU, no trainer, no real data (DS_FAKE=1 on a fake root).  Writes ONLY under /root/dscale10k/blockx/ (fake roots fake_rootx and fake_qx) and deletes them at the end.
# One PASS / FAIL line per scenario, then `ZZ DRYRUNX DONE ZZ`.
R=/root/dscale10k; BX=$R/blockx; F=$BX/fake_rootx; Q=$BX/fake_qx; PY=/root/venv/bin/python
ok=0; bad=0
chk() { if [ "$2" = "$3" ]; then echo "PASS $1"; ok=$((ok+1)); else echo "FAIL $1: got [$2] want [$3]"; bad=$((bad+1)); fi; }
has() { if echo "$2" | grep -q -- "$3"; then echo "PASS $1"; ok=$((ok+1)); else echo "FAIL $1: [$3] not in [$(echo "$2" | tail -c 300)]"; bad=$((bad+1)); fi; }
mk() {
  rm -rf "$F"; mkdir -p $F/argv $F/dry $F/gate/tokens $F/chains
  cp $R/chainlib.py $F/chainlib.py; cp $R/dscale_banker.py $F/dscale_banker.py; cp $BX/fake_leaf_x.sh $F/fake_leaf.sh
  echo '{"argv": ["--cache-dir", "/c", "--tracks", "/t", "--join-train", "/j", "--seed", "0", "--steps", "19164"]}' > $F/argv/X_s0.A.json
  echo '{"argv": ["--cache-dir", "/c", "--tracks", "/t", "--join-train", "/j", "--seed", "0", "--init-ckpt", "@INIT@", "--steps", "3000", "--dagger-bank", "@BANK@", "--dagger-bank-sha256", "@BANKSHA@"]}' > $F/argv/X_s0.B.tmpl.json
  echo '{"n_clips": 10000, "n_windows": 700000}' > $F/dry/X_s0.A.dry.json
}
run() { DS_FAKE=1 DS_ROOT=$F DS_BX=$BX DS_PY=$PY DS_POLL=0 DS_GPU_TRIES=2 bash $BX/blockx.sh 2>&1; }
mk; out=$(run); last=$(echo "$out" | tail -n 1)
chk "1 sentinel DONE" "$last" "ZZ DSCALE10K BLOCKX DONE ZZ"
order=$(grep -E "fake (prep|trainer|bank|mint|selfcheck)" $F/fake_order.log | sed -E 's/^[0-9:]+ fake //' | sed -E 's/ \(n_shards.*//' | tr '\n' ',')
chk "1 order: prep, X_s0 A, bank, selfcheck, mint B, X_s0 B" "$order" "prep,trainer X_s0 A,bank X_s0,selfcheck X_s0,mint X_s0 B,trainer X_s0 B,"
chk "1 stage A ckpt at step 19164, stage B at 3000" "$($PY -c "import torch;print([torch.load('$F/runs/X_s0/%s/ckpt_last.pt'%s,weights_only=False)['step'] for s in 'AB'])")" "[19164, 3000]"
has "1 ARM line" "$out" "ZZ ARM X_s0 DONE ZZ"
chk "1 ARM_DONE" "$(ls $F/runs/X_s0/ARM_DONE | wc -l)" "1"
chk "1 stage-B argv has no open placeholder" "$(grep -c '@' $F/runs/X_s0/argv_B.json)" "0"
chk "1 bank markers for both stages" "$(ls $F/bank_ok | tr '\n' ' ')" "X_s0.A.sha256 X_s0.B.sha256 "
out=$(run); chk "2 rerun DONE" "$(echo "$out" | tail -n 1)" "ZZ DSCALE10K BLOCKX DONE ZZ"; has "2 skip" "$out" "skip X_s0 (ARM_DONE)"; chk "2 no second trainer run" "$(grep -c 'fake trainer' $F/fake_order.log)" "2"
mk; out=$(DS_FAKE_FAIL="prep:-.-" run); has "3 prep fail" "$(echo "$out" | tail -n 1)" "block X prep"; chk "3 nothing trained" "$(cat $F/fake_order.log 2>/dev/null | grep -c "fake trainer")" "0"
mk; out=$(DS_FREE_GB=10 run); has "4 disk guard" "$(echo "$out" | tail -n 1)" "free disk 10 GB < 60 GB"
mk; touch $F/chains/STOP; out=$(run); has "5 STOP" "$(echo "$out" | tail -n 1)" "STOP file before"
mk; out=$(DS_FAKE_FAIL="run:X_s0.A" run); has "6 failed stage A" "$(echo "$out" | tail -n 1)" "X_s0 stage A rc=7"; chk "6 no bank, no stage B" "$(grep -c 'fake bank\|trainer X_s0 B' $F/fake_order.log)" "0"
mk; out=$(DS_FAKE_FAIL="run:X_s0.B" run); has "7 failed stage B" "$(echo "$out" | tail -n 1)" "X_s0 stage B rc=7"; chk "7 stage A banked before" "$(ls $F/bank_ok | tr '\n' ' ')" "X_s0.A.sha256 "
mk; out=$(DS_FAKE_FAIL="mint:X_s0.B" run); has "8 stage-B mint refusal stops it" "$(echo "$out" | tail -n 1)" "stage-B mint refused"; chk "8 stage B never ran" "$(grep -c 'trainer X_s0 B' $F/fake_order.log)" "0"
# real-file checks (the fork must name the X pieces and none of the block-3 ones)
chk "9 chk_armx selects the blockx checker" "$(grep -c 'B3=${DS_BX:-$D/blockx}' $BX/chk_armx.sh)" "1"
chk "9 chk_armx X join slots" "$(grep -c 'X_s0.*--corpus-join $D/merged/join_union.npz --union-join $D/merged/join_X.npz' $BX/chk_armx.sh)" "1"
chk "9 blockx.sh stage A 19164 / B 3000" "$(grep -c 'STEPS_A=19164; STEPS_B=3000;' $BX/blockx.sh)" "1"
chk "9 blockx.sh runs the X arm with stage B" "$(grep -c '^do_arm X_s0 yes' $BX/blockx.sh)" "1"
chk "9 no UE_s0 / BL_s0 left in blockx.sh code (comments excluded)" "$(grep -v '^#' $BX/blockx.sh | grep -c 'UE_s0\|BL_s0')" "0"
# the queue wrapper: waits for NAVRB2's end line, then execs flock + chain
rm -rf "$Q"; mkdir -p $Q; echo "echo CHAIN-RAN-UNDER-LOCK; echo \"\$(date +%s)\" > $Q/ran" > $Q/chain.sh
qrun() { QUEUE_LOG=$Q/navrb2.log QUEUE_LOCK=$Q/lock QUEUE_CHAIN=$Q/chain.sh QUEUE_POLL=1 QUEUE_MAXN=${1:-3} bash $BX/queue_blockx.sh 2>&1; }
echo "2026-11-17T19:34:02Z NAVRB2 queued behind block 3" > $Q/navrb2.log
out=$(qrun 2); has "10 queue times out on a log without the end line" "$out" "ZZ DSCALE10K BLOCKX FAIL ZZ queue"
echo "note: ZZ STRAT1 NAVRB2 DONE ZZ quoted mid-line" >> $Q/navrb2.log; out=$(qrun 2); has "11 a mid-line quote does not release it" "$out" "ZZ DSCALE10K BLOCKX FAIL ZZ queue"
echo "ZZ STRAT1 NAVRB2 DONE ZZ" >> $Q/navrb2.log; out=$(qrun 3); has "12 DONE line releases it" "$out" "CHAIN-RAN-UNDER-LOCK"
rm -f $Q/ran; echo "ZZ STRAT1 NAVRB2 FAIL ZZ x" > $Q/navrb2.log; out=$(qrun 3); has "13 FAIL line releases it too" "$out" "CHAIN-RAN-UNDER-LOCK"
echo "2026-11-17T19:34:02Z a C0REP line ZZ STRAT1 C0REP DONE ZZ" > $Q/navrb2.log; echo "ZZ STRAT1 C0REP DONE ZZ" >> $Q/navrb2.log; out=$(qrun 2); has "14 a C0REP line does not release the NAVRB2 wait" "$out" "ZZ DSCALE10K BLOCKX FAIL ZZ queue"
rm -f $Q/ran; echo "ZZ STRAT1 NAVRB2 DONE ZZ" > $Q/navrb2.log; ( flock $Q/lock sleep 4 ) & sleep 1; t0=$(date +%s); out=$(qrun 3); t1=$(date +%s); wait
has "15 chain ran" "$out" "CHAIN-RAN-UNDER-LOCK"; chk "15 it waited for the held lock (>= 2 s)" "$([ $((t1-t0)) -ge 2 ] && echo yes || echo no)" "yes"
rm -rf "$F" "$Q"
echo "scenarios PASS $ok FAIL $bad"
echo "ZZ DRYRUNX DONE ZZ"
