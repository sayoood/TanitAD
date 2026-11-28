#!/bin/bash
# ki_arm.sh TAG -- ONE stage-B arm of K-INPUT (TAG = KI_s0 | KI_s1; card PREREG_V8F_KINPUT). THIN COPY of rk_arm.sh revision 2 (itself of ra_arm.sh / rh_arm.sh sha256 0e794f68...) for kind KI.
# Called by rh_ki_chain.sh, which ALREADY HOLDS /root/gpu_queue.lock (fd 9): takes no lock.
# Differences from rk_arm.sh, and only these:
#   * the accepted kind is KI; the argv is RH's with `--dagger-base logged` appended (ki_argv.py verify re-checks the two differences against RH_s$seed just before the pair check);
#   * NO LABEL CACHE: no cache or CPU-precompute file is required or read (the label is the human's); the gate record is ki_gate_record.py (the RH branch + the base-lever checks of card section 3);
#   * the pair check is `KI argv minus the last two tokens (--dagger-base logged) vs the EXISTING B0_s$seed argv` (the base flag is the arm's own, checked by the study-side checker dscale10k_check_ki.py in the kit);
#   * the study-side calls go to chk_arm_ki.sh; after training the LEVER-ACTS check reads the train log's dagger_rows (summed over the log intervals; must be > 0: rows were swapped, card section 3).
# Order is rk_arm.sh's: idle GPU -> ki_argv.py verify -> pair check -> the trainer's --selfcheck-only of THIS argv (real bank, real init) -> ki_gate_record.py mint + verify -> MINT THE STUDY-SIDE GATE TOKEN
#   -> train (3,000 steps) THROUGH THE TOKEN -> lever-acts -> summary -> post gate -> zerograd_gate.py (report) -> K-CLIP.
# Last line: `ZZ ARM TAG DONE ZZ` or `ZZ ARM TAG FAIL ZZ <why>`.
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/ki_env.sh
tag=$1; seed=${tag##*_s}; kind=${tag%%_*}; D=$R/runs/$tag; OUT=$D/train
afail() { echo "ZZ ARM $tag FAIL ZZ $1"; exit 1; }
[ "$kind" = KI ] || afail "unknown tag (KI_s0 | KI_s1 only)"
rh_readout_flags; export_py; unset CUDA_VISIBLE_DEVICES
INIT=$DS/runs/UE_s$seed/A/ckpt_last.pt
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${u:-99999}" -lt 3500 ]; }
n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && afail "GPU not idle (>= 3500 MiB used)"; sleep 10; done
[ -s $D/argv.json ] || afail "no $D/argv.json (ki_argv.py render first)"
mkdir -p $D/gate
$PY $HERE/ki_argv.py verify --seeds $seed > $D/argv_verify.log 2>&1 || afail "argv verify (KI vs RH, two differences only): $(tail -c 300 $D/argv_verify.log | tr '\n' ' ')"
sha=$($PY -c "import hashlib;print(hashlib.sha256(open('$R/bank/s$seed/bank.json','rb').read()).hexdigest())")
$PY - $D/argv.json $D/argv_nobase.json <<'PYE'
import json, sys
t = json.load(open(sys.argv[1]))["argv"]
i = t.index("--dagger-base")
assert t[i + 1] == "logged" and i + 2 == len(t)          # the base pair, as the LAST two tokens
json.dump({"argv": t[:i]}, open(sys.argv[2], "w"), indent=1)
PYE
  $PY $TREE/stack/scripts/rollhist_pair_check.py --rh $D/argv_nobase.json --b0 $R/runs/B0_s$seed/argv.json --bank-sha256 $sha > $D/pair_$tag.log 2>&1 || afail "pair check: $(tail -c 300 $D/pair_$tag.log)"
if [ ! -s $D/selfcheck/selfcheck.json ]; then
  echo "$(date -u +%T) selfcheck $tag"
  RH_TREE=$TREE $PY $R/rh_selfcheck.py $D/argv.json $D/selfcheck $D/selfcheck.result.json > $D/selfcheck.log 2>&1 || afail "selfcheck refused: $(tail -c 400 $D/selfcheck.log | tr '\n' ' ')"
fi
[ -s $D/selfcheck/selfcheck.json ] || afail "selfcheck wrote no selfcheck.json: $(tail -c 300 $D/selfcheck.log | tr '\n' ' ')"
$PY $HERE/ki_gate_record.py mint --tag $tag --readout $RH_READOUT > $D/mint.log 2>&1 || afail "gate record refused: $(tail -c 500 $D/mint.log | tr '\n' ' ')"
grep "^probe:" $D/mint.log
$PY $HERE/ki_gate_record.py verify --tag $tag > $D/verify.log 2>&1 || afail "gate record verify: $(tail -c 300 $D/verify.log | tr '\n' ' ')"
bash $HERE/chk_arm_ki.sh mint $tag > $D/mint_token.log 2>&1 || afail "gate token mint refused: $(tail -c 500 $D/mint_token.log | tr '\n' ' ')"
grep -q "^TOKEN-WRITTEN $tag" $D/mint_token.log && [ -s $D/gate/gate_token.json ] || afail "no gate token written: $(tail -c 300 $D/mint_token.log | tr '\n' ' ')"
echo "$(date -u +%T) train $tag"
mkdir -p $OUT
cd $TREE || afail "cd tree"
bash $HERE/chk_arm_ki.sh run $tag > $D/train.log 2>&1      # REFUSED (rc 2, nothing started) unless the token binds exactly this argv, commit, data, tree, trainer, checker, template and pair
rc=$?
[ $rc -eq 0 ] || afail "trainer / gate runner rc=$rc: $(tail -c 400 $D/train.log | tr '\n' ' ')"
$PY - $OUT/train_log.jsonl $D/lever_acts_$tag.json <<'PYE' || afail "lever-acts: the train log's dagger_rows is absent or 0 (card section 3): $(cat $D/lever_acts_$tag.json 2>/dev/null | tail -c 300)"
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
json.dump({"records_with_key": n, "dagger_rows_sum": rows, "dagger_base": "logged"}, open(sys.argv[2], "w"))
sys.exit(0 if rows > 0 else 1)
PYE
$PY $DS/chainlib.py summary $OUT 3000 > $OUT/summary_check.json 2> $OUT/summary_check.err || afail "summary: $(cat $OUT/summary_check.err | tail -c 300)"
$PY $DS/chainlib.py post $OUT/ckpt_last.pt $INIT > $OUT/post.json 2> $OUT/post.err || afail "post gate (zero-gradient / unmoved tensors): $(tail -c 300 $OUT/post.err)"
PYTHONPATH=$TREE/stack:$TREE $PY $TREE/stack/scripts/zerograd_gate.py --ckpt $OUT/ckpt_last.pt --json-out $D/zerograd_$tag.json > $D/zerograd_$tag.log 2>&1 || echo "zerograd_gate.py rc!=0 (report only; the fatal gate is chainlib post): $(tail -n 2 $D/zerograd_$tag.log | tr '\n' ' ')"
$PY $S1/n1dc/kclip.py $OUT/train_log.jsonl --json > $D/kclip_$tag.json 2> $D/kclip_$tag.err || afail "K-CLIP: $(tail -c 300 $D/kclip_$tag.err)"
echo "ZZ ARM $tag DONE ZZ" | tee $D/ARM_DONE
