#!/bin/bash
# s3_arm.sh TAG -- ONE stage-B arm of SEED3 (TAG = B0_s2 | RH_s2 | KI_s2; card PREREG_V8F_SEED3). Called by rh_s3_chain.sh, which ALREADY HOLDS /root/gpu_queue.lock (fd 9): takes no lock.
# THIN UNION of rh_arm.sh (B0 / RH) and ki_arm.sh (KI): idle GPU -> s3_argv.py verify (exactly the card's substitutions vs the s0 argv) -> pair check -> the trainer's --selfcheck-only of THIS argv (real bank, real init)
#   -> gate record mint + verify (rh_gate_record_s3.py for B0 / RH, ki_gate_record_s3.py for KI: the KINPUT items) -> MINT THE STUDY-SIDE GATE TOKEN (chk_arm_s3.sh) -> train 3,000 steps THROUGH THE TOKEN
#   -> lever-acts (RH, KI: dagger_rows summed over the log > 0) -> summary -> post gate (every tensor moved) -> zerograd_gate.py (report, as the s0 / s1 arms) -> K-CLIP.
# Last line: `ZZ ARM TAG DONE ZZ` or `ZZ ARM TAG FAIL ZZ <why>`.
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/s3_env.sh
tag=$1; seed=${tag##*_s}; kind=${tag%%_*}; D=$R/runs/$tag; OUT=$D/train
afail() { echo "ZZ ARM $tag FAIL ZZ $1"; exit 1; }
s3_arm_env $tag || afail "unknown tag (B0_s2 | RH_s2 | KI_s2 only)"
rh_readout_flags; export_py; unset CUDA_VISIBLE_DEVICES
INIT=$UE2
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${u:-99999}" -lt 3500 ]; }
n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && afail "GPU not idle (>= 3500 MiB used)"; sleep 10; done
[ -s $D/argv.json ] || afail "no $D/argv.json (s3_argv.py render first)"
[ "$($PY -c "import hashlib;print(hashlib.sha256(open('$INIT','rb').read()).hexdigest())")" = $UE2_SHA ] || afail "UE_s2.A sha256 is not $UE2_SHA"
mkdir -p $D/gate
$PY $HERE/s3_argv.py verify --arms $tag > $D/argv_verify.log 2>&1 || afail "argv verify (the card's substitutions only): $(tail -c 300 $D/argv_verify.log | tr '\n' ' ')"
sha=$($PY -c "import hashlib;print(hashlib.sha256(open('$R/bank/s2/bank.json','rb').read()).hexdigest())" 2>/dev/null)
case $kind in
  B0) $PY $TREE/stack/scripts/rollhist_pair_check.py --rh $D/pair_rh_skeleton.json --b0 $D/argv.json > $D/pair_$tag.log 2>&1 || afail "pair check (B0 vs RH_s2's skeleton): $(tail -c 300 $D/pair_$tag.log)";;
  RH) [ -n "$sha" ] || afail "no bank.json"
      $PY $TREE/stack/scripts/rollhist_pair_check.py --rh $D/argv.json --b0 $R/runs/B0_s2/argv.json --bank-sha256 $sha > $D/pair_$tag.log 2>&1 || afail "pair check: $(tail -c 300 $D/pair_$tag.log)";;
  KI) [ -n "$sha" ] || afail "no bank.json"
      $PY - $D/argv.json $D/argv_nobase.json <<'PYE'
import json, sys
t = json.load(open(sys.argv[1]))["argv"]
i = t.index("--dagger-base")
assert t[i + 1] == "logged" and i + 2 == len(t)          # the base pair, as the LAST two tokens
json.dump({"argv": t[:i]}, open(sys.argv[2], "w"), indent=1)
PYE
      $PY $TREE/stack/scripts/rollhist_pair_check.py --rh $D/argv_nobase.json --b0 $R/runs/B0_s2/argv.json --bank-sha256 $sha > $D/pair_$tag.log 2>&1 || afail "pair check: $(tail -c 300 $D/pair_$tag.log)";;
esac
if [ ! -s $D/selfcheck/selfcheck.json ]; then
  echo "$(date -u +%T) selfcheck $tag"
  RH_TREE=$TREE $PY $R/rh_selfcheck.py $D/argv.json $D/selfcheck $D/selfcheck.result.json > $D/selfcheck.log 2>&1 || afail "selfcheck refused: $(tail -c 400 $D/selfcheck.log | tr '\n' ' ')"
fi
[ -s $D/selfcheck/selfcheck.json ] || afail "selfcheck wrote no selfcheck.json: $(tail -c 300 $D/selfcheck.log | tr '\n' ' ')"
$PY $GR mint --tag $tag --readout $RH_READOUT > $D/mint.log 2>&1 || afail "gate record refused: $(tail -c 500 $D/mint.log | tr '\n' ' ')"
grep "^probe:" $D/mint.log
$PY $GR verify --tag $tag > $D/verify.log 2>&1 || afail "gate record verify: $(tail -c 300 $D/verify.log | tr '\n' ' ')"
bash $HERE/chk_arm_s3.sh mint $tag > $D/mint_token.log 2>&1 || afail "gate token mint refused: $(tail -c 500 $D/mint_token.log | tr '\n' ' ')"
grep -q "^TOKEN-WRITTEN $tag" $D/mint_token.log && [ -s $D/gate/gate_token.json ] || afail "no gate token written: $(tail -c 300 $D/mint_token.log | tr '\n' ' ')"
echo "$(date -u +%T) train $tag"
mkdir -p $OUT
cd $TREE || afail "cd tree"
bash $HERE/chk_arm_s3.sh run $tag > $D/train.log 2>&1      # REFUSED (rc 2, nothing started) unless the token binds exactly this argv, commit, data, tree, trainer, checker, template and pair
rc=$?
[ $rc -eq 0 ] || afail "trainer / gate runner rc=$rc: $(tail -c 400 $D/train.log | tr '\n' ' ')"
if [ $kind != B0 ]; then
$PY - $OUT/train_log.jsonl $D/lever_acts_$tag.json $kind <<'PYE' || afail "lever-acts: the train log's dagger_rows is absent or 0 (card section 3): $(cat $D/lever_acts_$tag.json 2>/dev/null | tail -c 300)"
import json, sys
n, rows = 0, 0
for line in open(sys.argv[1], encoding="utf-8"):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if "dagger_rows" in r:
        n += 1
        rows += int(r["dagger_rows"])
json.dump({"records_with_key": n, "dagger_rows_sum": rows, "kind": sys.argv[3]}, open(sys.argv[2], "w"))
sys.exit(0 if rows > 0 else 1)
PYE
fi
$PY $DS/chainlib.py summary $OUT 3000 > $OUT/summary_check.json 2> $OUT/summary_check.err || afail "summary: $(cat $OUT/summary_check.err | tail -c 300)"
$PY $DS/chainlib.py post $OUT/ckpt_last.pt $INIT > $OUT/post.json 2> $OUT/post.err || afail "post gate (zero-gradient / unmoved tensors): $(tail -c 300 $OUT/post.err)"
PYTHONPATH=$TREE/stack:$TREE $PY $TREE/stack/scripts/zerograd_gate.py --ckpt $OUT/ckpt_last.pt --json-out $D/zerograd_$tag.json > $D/zerograd_$tag.log 2>&1 || echo "zerograd_gate.py rc!=0 (report only; the fatal gate is chainlib post): $(tail -n 2 $D/zerograd_$tag.log | tr '\n' ' ')"
$PY $S1/n1dc/kclip.py $OUT/train_log.jsonl --json > $D/kclip_$tag.json 2> $D/kclip_$tag.err || afail "K-CLIP: $(tail -c 300 $D/kclip_$tag.err)"
echo "ZZ ARM $tag DONE ZZ" | tee $D/ARM_DONE
