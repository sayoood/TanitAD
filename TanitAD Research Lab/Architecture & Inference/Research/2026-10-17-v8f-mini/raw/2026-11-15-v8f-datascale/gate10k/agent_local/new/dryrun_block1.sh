#!/bin/bash
# dryrun_block1.sh -- control-flow dry run of block1.sh with fake leaf operations (pod, CPU only, no GPU, no trainer, no real data). Writes ONLY under the fake root
# /root/dscale10k/fake_block1 and deletes it at the end (own temporary data). Prints one PASS / FAIL line per scenario and `ZZ DRYRUN DONE ZZ`.
R=/root/dscale10k; F=$R/fake_block1; PY=/root/venv/bin/python
ok=0; bad=0
chk() { if [ "$2" = "$3" ]; then echo "PASS $1"; ok=$((ok+1)); else echo "FAIL $1: got [$2] want [$3]"; bad=$((bad+1)); fi; }
has() { if echo "$2" | grep -q -- "$3"; then echo "PASS $1"; ok=$((ok+1)); else echo "FAIL $1: [$3] not in [$(echo "$2" | tail -c 300)]"; bad=$((bad+1)); fi; }
mk() {   # mk : a fresh fake root with the stage-A tokens of B_s0 U_s0 B_rep (and R_s0 when $1 = r)
  rm -rf $F; mkdir -p $F/argv $F/dry $F/gate/tokens $F/chains
  cp $R/chainlib.py $F/chainlib.py; cp $R/gate/fake_leaf.sh $F/fake_leaf.sh
  for a in B_s0 U_s0 B_rep R_s0; do
    echo '{"argv": ["--cache-dir", "/c", "--tracks", "/t", "--join-train", "/j", "--seed", "0"]}' > $F/argv/$a.A.json
    echo '{"argv": ["--cache-dir", "/c", "--init-ckpt", "@INIT@", "--dagger-bank", "@BANK@", "--dagger-bank-sha256", "@BANKSHA@"]}' > $F/argv/$a.B.tmpl.json
    echo '{"n_clips": 4512, "n_windows": 306566}' > $F/dry/$a.A.dry.json
  done
  for a in B_s0 U_s0 B_rep; do mkdir -p $F/gate/tokens/$a.A; echo '{}' > $F/gate/tokens/$a.A/gate_token.json; done
  [ "${1:-}" = r ] && { mkdir -p $F/gate/tokens/R_s0.A; echo '{}' > $F/gate/tokens/R_s0.A/gate_token.json; }
}
run() { DS_FAKE=1 DS_ROOT=$F DS_PY=$PY DS_POLL=0 DS_GPU_TRIES=2 bash $R/block1.sh 2>&1; }
# 1. every token: DONE, the card's order, 4-shard banks, R_s0 stage A only and last, nothing deleted (nothing banked)
mk r; out=$(run); last=$(echo "$out" | tail -n 1)
chk "1 sentinel DONE" "$last" "ZZ DSCALE10K BLOCK1 DONE ZZ"
order=$(grep -E "fake (trainer|bank|mint|selfcheck)" $F/fake_order.log | sed -E 's/^[0-9:]+ fake //; s/ \(n_shards.*//' | tr '\n' ',')
chk "1 order" "$order" "trainer B_s0 A,bank B_s0,selfcheck B_s0,mint B_s0 B,trainer B_s0 B,trainer U_s0 A,bank U_s0,selfcheck U_s0,mint U_s0 B,trainer U_s0 B,trainer B_rep A,bank B_rep,selfcheck B_rep,mint B_rep B,trainer B_rep B,trainer R_s0 A,"
has "1 bank in 4 parallel shards" "$(cat $F/fake_order.log)" "n_shards 4 in parallel"
has "1 deletes deferred (nothing banked)" "$out" "DELETE-DEFERRED B_s0"
has "1 banker told" "$(ls $F/bank_ready)" "B_s0.B.txt"
chk "1 nothing deleted" "$(ls $F/deletions_block1.log 2>/dev/null | wc -l)" "0"
# 2. rerun: every arm skipped
out=$(run); chk "2 rerun sentinel" "$(echo "$out" | tail -n 1)" "ZZ DSCALE10K BLOCK1 DONE ZZ"; has "2 skip" "$out" "skip B_s0 (ARM_DONE)"
chk "2 no second trainer run" "$(grep -c 'fake trainer' $F/fake_order.log)" "7"
# 3. banked read checkpoints: the sweep deletes the bank shards (bank.json stays) and logs each deletion
for a in B_s0 U_s0 B_rep; do echo "$(sha256sum $F/runs/$a/B/ckpt_last.pt | cut -d' ' -f1)  x" > $F/bank_ok_$a; mkdir -p $F/bank_ok; cp $F/bank_ok_$a $F/bank_ok/$a.B.sha256; done
echo "$(sha256sum $F/runs/R_s0/A/ckpt_last.pt | cut -d' ' -f1)  x" > $F/bank_ok/R_s0.A.sha256
rm -f $F/runs/B_s0/ARM_DONE; out=$(run)
chk "3 shards deleted" "$(ls $F/runs/B_s0/bank | grep -c shard_)" "0"
chk "3 bank.json kept" "$(ls $F/runs/B_s0/bank/bank.json | wc -l)" "1"
chk "3 deletions logged (4 shards)" "$(grep -c shard_ $F/deletions_block1.log)" "12"
has "3 ckpt kept" "$(ls $F/runs/B_s0/B)" "ckpt_last.pt"
# 4. a banked marker for a DIFFERENT file is not accepted
echo "0000  x" > $F/bank_ok/B_s0.B.sha256; $PY $F/chainlib.py banked B_s0.B $F/runs/B_s0/B/ckpt_last.pt; chk "4 wrong marker refused" "$?" "1"
# 5. no R token: B_s0 U_s0 B_rep run, then FAIL naming R_s0
mk; out=$(run); has "5 R_s0 FAIL" "$(echo "$out" | tail -n 1)" "ZZ DSCALE10K BLOCK1 FAIL ZZ R_s0 not run"; has "5 other arms ran" "$out" "ZZ ARM B_rep DONE ZZ"
# 6. a missing stage-A token stops the chain before any run
mk r; rm -rf $F/gate/tokens/U_s0.A; out=$(run); has "6 no U_s0 token" "$(echo "$out" | tail -n 1)" "no stage-A gate token for U_s0"; chk "6 nothing ran" "$(ls $F/fake_order.log 2>/dev/null | wc -l)" "0"
# 7. disk guard
mk r; out=$(DS_FREE_GB=10 run); has "7 disk guard" "$(echo "$out" | tail -n 1)" "free disk 10 GB < 60 GB"
# 8. STOP file
mk r; touch $F/chains/STOP; out=$(run); has "8 STOP" "$(echo "$out" | tail -n 1)" "STOP file before B_s0 stage A"
# 9. a failed leaf stops the chain with its reason and no later arm runs
mk r; out=$(DS_FAKE_FAIL="run:U_s0.A" run); has "9 failed arm" "$(echo "$out" | tail -n 1)" "U_s0 stage A rc=7"; chk "9 B_rep never ran" "$(grep -c 'trainer B_rep' $F/fake_order.log)" "0"
# 10. a refused runner (rc 2) is a FAIL: the stage-B token is minted in the chain, so remove the minted token's effect by failing the mint
mk r; out=$(DS_FAKE_FAIL="mint:B_s0.B" run); has "10 refused mint" "$(echo "$out" | tail -n 1)" "B_s0 stage-B mint refused"; chk "10 stage B never ran" "$(grep -c 'trainer B_s0 B' $F/fake_order.log)" "0"
# 11 / 12. the post-run reads on a checkpoint that stopped at the wrong step and on one with a parameter that never received a gradient
rm -rf $F; mkdir -p $F/x; $PY - <<'E'
import torch, json, os, subprocess, sys
d = "/root/dscale10k/fake_block1/x"
torch.save({"step": 100, "model": {}, "opt": {"param_groups": [{"params": [0, 1]}], "state": {0: {"a": 1}}}}, d + "/ckpt_last.pt")
json.dump({"done": True, "steps": 100}, open(d + "/summary.json", "w"))
env = dict(os.environ, DS_ROOT="/root/dscale10k/fake_block1")
r1 = subprocess.run(["/root/venv/bin/python", "/root/dscale10k/chainlib.py", "summary", d, "19164"], env=env, capture_output=True, text=True)
r2 = subprocess.run(["/root/venv/bin/python", "/root/dscale10k/chainlib.py", "post", d + "/ckpt_last.pt"], env=env, capture_output=True, text=True)
print("PASS 11 summary refuses a wrong step" if r1.returncode == 1 else "FAIL 11 summary", "|", r1.stderr.strip()[:120])
print("PASS 12 post gate refuses a parameter with no Adam state" if r2.returncode == 1 else "FAIL 12 post", "|", r2.stderr.strip()[:160])
E
rm -rf $F
echo "scenarios PASS $ok FAIL $bad (+ the two python checks above)"
echo "ZZ DRYRUN DONE ZZ"
