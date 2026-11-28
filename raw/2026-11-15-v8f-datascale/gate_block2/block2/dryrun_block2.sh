#!/bin/bash
# dryrun_block2.sh -- control-flow dry run of block2.sh with fake leaf operations and the REAL chainlib.py and dscale_banker.py (a directory plays the hub) (pod, CPU only, no GPU, no trainer, no real data).
# Writes ONLY under the fake root /root/dscale10k/fake_block2 and deletes it at the end (own temporary data). One PASS / FAIL line per scenario, then `ZZ DRYRUN2 DONE ZZ`.
R=/root/dscale10k; F=$R/fake_block2; PY=/root/venv/bin/python; B=$R/block2_files
ok=0; bad=0
chk() { if [ "$2" = "$3" ]; then echo "PASS $1"; ok=$((ok+1)); else echo "FAIL $1: got [$2] want [$3]"; bad=$((bad+1)); fi; }
has() { if echo "$2" | grep -q -- "$3"; then echo "PASS $1"; ok=$((ok+1)); else echo "FAIL $1: [$3] not in [$(echo "$2" | tail -c 300)]"; bad=$((bad+1)); fi; }
mk() {   # mk : a fresh fake root, block-1 arms absent, no block-2 tokens
  rm -rf "$F"; mkdir -p $F/argv $F/dry $F/gate/tokens $F/chains
  cp $R/chainlib.py $F/chainlib.py; cp $R/dscale_banker.py $F/dscale_banker.py; cp $B/fake_leaf2.sh $F/fake_leaf.sh
  for a in B_s1 U_s1; do
    echo '{"argv": ["--cache-dir", "/c", "--tracks", "/t", "--join-train", "/j", "--seed", "1"]}' > $F/argv/$a.A.json
    echo '{"argv": ["--cache-dir", "/c", "--init-ckpt", "@INIT@", "--dagger-bank", "@BANK@", "--dagger-bank-sha256", "@BANKSHA@"]}' > $F/argv/$a.B.tmpl.json
    echo '{"n_clips": 4512, "n_windows": 306566}' > $F/dry/$a.A.dry.json
  done
}
run() { DS_FAKE=1 DS_ROOT=$F DS_PY=$PY DS_POLL=0 DS_GPU_TRIES=2 bash $B/block2.sh 2>&1; }
# 1. no tokens: the prep runs first, then the card's order, the banker banks after every stage, the sweep deletes banked bank shards, nothing else
mk; out=$(run); last=$(echo "$out" | tail -n 1)
chk "1 sentinel DONE" "$last" "ZZ DSCALE10K BLOCK2 DONE ZZ"
order=$(grep -E "fake (prep|trainer|bank|mint|selfcheck)" $F/fake_order.log | sed -E 's/^[0-9:]+ fake //; s/ \(n_shards.*//' | tr '\n' ',')
chk "1 order" "$order" "prep,trainer B_s1 A,bank B_s1,selfcheck B_s1,mint B_s1 B,trainer B_s1 B,trainer U_s1 A,bank U_s1,selfcheck U_s1,mint U_s1 B,trainer U_s1 B,"
chk "1 four markers written by the banker" "$(ls $F/bank_ok | tr '\n' ' ')" "B_s1.A.sha256 B_s1.B.sha256 U_s1.A.sha256 U_s1.B.sha256 "
chk "1 hub holds the four checkpoints" "$(ls $F/fakehub/*/*/ckpt_last.pt | wc -l)" "4"
chk "1 bank shards of both arms swept" "$(ls $F/runs/B_s1/bank $F/runs/U_s1/bank | grep -c shard_)" "0"
chk "1 bank.json kept" "$(ls $F/runs/B_s1/bank/bank.json $F/runs/U_s1/bank/bank.json | wc -l)" "2"
chk "1 deletions logged (8 shards)" "$(grep -c shard_ $F/deletions_block1.log)" "8"
chk "1 checkpoints kept" "$(ls $F/runs/*/*/ckpt_last.pt | wc -l)" "4"
chk "1 nothing pending" "$(echo "$out" | grep -c 'banking pending (no bank_ok marker): $')" "1"
# 2. rerun: the arms are skipped, no second training, no second upload
n=$(ls $F/fakehub/*/*/ckpt_last.pt | wc -l); out=$(run); chk "2 rerun DONE" "$(echo "$out" | tail -n 1)" "ZZ DSCALE10K BLOCK2 DONE ZZ"; has "2 skip" "$out" "skip B_s1 (ARM_DONE)"
chk "2 no second trainer run" "$(grep -c 'fake trainer' $F/fake_order.log)" "4"
# 3. the prep fails: nothing runs
mk; out=$(DS_FAKE_FAIL="prep:-.-" run); has "3 prep fail" "$(echo "$out" | tail -n 1)" "block 2 prep"; chk "3 nothing trained" "$(cat $F/fake_order.log 2>/dev/null | grep -c "fake trainer")" "0"
# 4. tokens present: the prep is skipped
mk; for a in B_s1 U_s1; do mkdir -p $F/gate/tokens/$a.A; echo '{}' > $F/gate/tokens/$a.A/gate_token.json; done; out=$(run); chk "4 DONE without prep" "$(echo "$out" | tail -n 1)" "ZZ DSCALE10K BLOCK2 DONE ZZ"; chk "4 no prep call" "$(grep -c 'fake prep' $F/fake_order.log)" "0"
# 5. a banker conflict (a marker for other bytes) defers the deletion, the chain goes on, the shards stay
mk; for a in B_s1 U_s1; do mkdir -p $F/gate/tokens/$a.A; echo '{}' > $F/gate/tokens/$a.A/gate_token.json; done
mkdir -p $F/bank_ok; echo "$(printf 'c%.0s' $(seq 64)) old" > $F/bank_ok/B_s1.B.sha256
out=$(run); has "5 banker deferred" "$out" "BANK-DEFERRED B_s1.B"; has "5 delete deferred" "$out" "DELETE-DEFERRED B_s1"
chk "5 B_s1 shards kept" "$(ls $F/runs/B_s1/bank | grep -c shard_)" "4"; chk "5 U_s1 shards swept" "$(ls $F/runs/U_s1/bank | grep -c shard_)" "0"
chk "5 marker not overwritten" "$(cut -c1-3 $F/bank_ok/B_s1.B.sha256)" "ccc"; chk "5 chain finished" "$(echo "$out" | tail -n 1)" "ZZ DSCALE10K BLOCK2 DONE ZZ"
# 6. disk guard, STOP, failed leaf, refused mint
mk; out=$(DS_FREE_GB=10 run); has "6 disk guard" "$(echo "$out" | tail -n 1)" "free disk 10 GB < 60 GB"
mk; touch $F/chains/STOP; out=$(run); has "7 STOP" "$(echo "$out" | tail -n 1)" "STOP file before"
mk; out=$(DS_FAKE_FAIL="run:U_s1.A" run); has "8 failed arm" "$(echo "$out" | tail -n 1)" "U_s1 stage A rc=7"
mk; out=$(DS_FAKE_FAIL="mint:B_s1.B" run); has "9 refused mint" "$(echo "$out" | tail -n 1)" "B_s1 stage-B mint refused"; chk "9 stage B never ran" "$(grep -c 'trainer B_s1 B' $F/fake_order.log)" "0"
rm -rf "$F"
echo "scenarios PASS $ok FAIL $bad"
echo "ZZ DRYRUN2 DONE ZZ"
