#!/bin/bash
# gpu_gate.sh -- the GPU half of the block-1 gate (pod). Waits for the B pilot bank (bank_B_pilot.log "bank4 done"), then, one GPU job at a time and only while the GPU is idle (< 3,500 MiB):
#   1. reach probes B_s0.A and B_rep.A (corpus-only batch + the TL-positive control)               -> probe2/<arm>.json
#   2. the trainer's --selfcheck-only on the REAL stage-B argv of B_s0 (real complete bank, real --init-ckpt = the pilot checkpoint)   -> sb/B_s0.B/selfcheck.json
#   3. the same with the DAgger swap MUTATED (apply returns the logged batch): must be REFUSED     -> sb/B_s0.B.mut.json
#   4. a U-set smoke bank (--max-clips 200, 4 shards parallel) + the trainer's stage-B selfcheck on the U_s0 stage-B argv  -> sb/U_s0.B/selfcheck.json
#   5. G-CKPT resume smoke (crash hook, resume, uninterrupted)                                      -> resume.json
# Ends with `ZZ GPU-GATE DONE ZZ` or `ZZ GPU-GATE FAIL ZZ <why>` in /root/dscale10k/gate/gpu_gate.log.
D=/root/dscale10k; G=$D/gate; PY=/root/venv/bin/python; TREE=$D/tree
export PYTHONPATH="$TREE/stack:$TREE/stack/scripts:$TREE:$D/study" HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
fail() { echo "ZZ GPU-GATE FAIL ZZ $1"; exit 1; }
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${u:-99999}" -lt 3500 ]; }
wait_gpu() { n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle before $1"; sleep 10; done; }
n=0; until grep -q "bank4 done" $G/bank_B_pilot.log 2>/dev/null; do grep -q "BANK4 FAIL" $G/bank_B_pilot.log 2>/dev/null && fail "B pilot bank failed"; n=$((n+1)); [ $n -gt 720 ] && fail "B pilot bank not done in 2 h"; sleep 10; done
echo "$(date -u +%T) B pilot bank done"
mkdir -p $G/probe2 $G/sb
sub() {   # sub TEMPLATE OUT INIT BANKDIR : fill the stage-B placeholders
  $PY - "$1" "$2" "$3" "$4" <<'E'
import json, sys, hashlib, os
t, out, init, bank = sys.argv[1:5]
sha = hashlib.sha256(open(os.path.join(bank, "bank.json"), "rb").read()).hexdigest()
a = json.load(open(t))["argv"]
a = [init if x == "@INIT@" else bank if x == "@BANK@" else sha if x == "@BANKSHA@" else x for x in a]
json.dump({"argv": a}, open(out, "w"), indent=1)
E
}
# 1
for nm in B_s0.A B_rep.A; do
  wait_gpu "reach $nm"
  $PY $G/reach_probe2.py $D/argv/$nm.json $G/probe2/$nm.json $D/merged/v9_labels_union.manifest.json --reach > $G/probe2/$nm.log 2>&1 || fail "reach probe $nm: $(tail -c 300 $G/probe2/$nm.log)"
  echo "$(date -u +%T) reach probe $nm done: $(tail -n1 $G/probe2/$nm.log | cut -c1-200)"
done
# 2, 3
sub $D/argv/B_s0.B.tmpl.json $G/sb/B_s0.B.argv.json $D/pilot/pB/ckpt_last.pt $G/bank_B_pilot || fail "sub B"
wait_gpu "stage-B selfcheck B_s0"
$PY $G/sb_selfcheck.py $G/sb/B_s0.B.argv.json $G/sb/B_s0.B $G/sb/B_s0.B.result.json > $G/sb/B_s0.B.log 2>&1; echo "$(date -u +%T) B_s0.B selfcheck: $(tail -n1 $G/sb/B_s0.B.log | cut -c1-300)"
wait_gpu "mutated selfcheck"
$PY $G/sb_selfcheck.py $G/sb/B_s0.B.argv.json $G/sb/B_s0.B.mut $G/sb/B_s0.B.mut.json --mutate-apply > $G/sb/B_s0.B.mut.log 2>&1; echo "$(date -u +%T) mutated selfcheck: $(tail -n1 $G/sb/B_s0.B.mut.log | cut -c1-300)"
# 4: U smoke bank + selfcheck
wait_gpu "U smoke bank"
$PY - <<'E'
import json
D = "/root/dscale10k"
a = json.load(open(D + "/argv/U_s0.A.json"))["argv"]
a = [x for x in a if x != "--fit-join"]
i = a.index("--out"); a[i + 1] = D + "/gate/sb/U_smoke_A"
a += ["--max-clips", "200"]
json.dump({"argv": a}, open(D + "/gate/sb/U_smoke_m2b_argv.json", "w"), indent=1)
E
bash $D/bank4.sh $D/pilot/pU/ckpt_last.pt $G/sb/U_smoke_m2b_argv.json /root/data/cache_union_dscale $D/merged/native_tracks_union.pt $D/merged/join_union.npz $G/sb/U_smoke_bank 200 0 > $G/sb/U_smoke_bank.log 2>&1 || fail "U smoke bank: $(tail -c 300 $G/sb/U_smoke_bank.log)"
sub $D/argv/U_s0.B.tmpl.json $G/sb/U_s0.B.argv.json $D/pilot/pU/ckpt_last.pt $G/sb/U_smoke_bank || fail "sub U"
wait_gpu "U stage-B selfcheck"
$PY $G/sb_selfcheck.py $G/sb/U_s0.B.argv.json $G/sb/U_s0.B $G/sb/U_s0.B.result.json --max-clips 200 > $G/sb/U_s0.B.log 2>&1; echo "$(date -u +%T) U_s0.B smoke selfcheck: $(tail -n1 $G/sb/U_s0.B.log | cut -c1-300)"
# 5
wait_gpu "resume smoke"
$PY $G/resume_smoke.py > $G/resume.log 2>&1 || fail "resume smoke: $(tail -c 300 $G/resume.log)"
echo "$(date -u +%T) $(tail -n1 $G/resume.log | cut -c1-300)"
echo "ZZ GPU-GATE DONE ZZ"
