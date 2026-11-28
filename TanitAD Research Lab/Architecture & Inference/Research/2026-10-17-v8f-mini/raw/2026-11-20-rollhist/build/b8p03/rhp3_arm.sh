#!/bin/bash
# rhp3_arm.sh TAG -- ONE stage-B arm of the ROLLHIST next arm (TAG = RHp3_s0 | RHp3_s1). THIN COPY of rh_arm.sh (sha256 0e794f68...) for kind RHp3. Called by rh_p03_chain.sh, which ALREADY HOLDS /root/gpu_queue.lock (fd 9): takes no lock.
# Differences from rh_arm.sh, and only these: the accepted kind is RHp3; the pair check is always `RHp3 argv vs the EXISTING B0_s$seed argv` (no new B0); the study-side calls go to chk_arm_rhp3.sh.
# Order is rh_arm.sh's: idle GPU -> pair check (stock rollhist_pair_check.py on the filled argv; p > 0 only) -> the trainer's --selfcheck-only of THIS argv (real bank, real init) -> rh_gate_record.py mint + verify
#   -> MINT THE STUDY-SIDE GATE TOKEN (chk_arm_rhp3.sh mint: dscale10k_check_rhp3.py, p pinned to 0.3) -> train (3,000 steps) THROUGH THE TOKEN (chk_arm_rhp3.sh run) -> summary -> post gate -> zerograd_gate.py (report) -> K-CLIP.
# Last line: `ZZ ARM TAG DONE ZZ` or `ZZ ARM TAG FAIL ZZ <why>`.
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/rh_env.sh
tag=$1; seed=${tag##*_s}; kind=${tag%%_*}; D=$R/runs/$tag; OUT=$D/train
afail() { echo "ZZ ARM $tag FAIL ZZ $1"; exit 1; }
[ "$kind" = RHp3 ] || afail "unknown tag (RHp3_s0 | RHp3_s1 only)"
rh_readout_flags; export_py; unset CUDA_VISIBLE_DEVICES
INIT=$DS/runs/UE_s$seed/A/ckpt_last.pt
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${u:-99999}" -lt 3500 ]; }
n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && afail "GPU not idle (>= 3500 MiB used)"; sleep 10; done
[ -s $D/argv.json ] || afail "no $D/argv.json (rhp3_argv.py render first)"
mkdir -p $D/gate
sha=$($PY -c "import hashlib;print(hashlib.sha256(open('$R/bank/s$seed/bank.json','rb').read()).hexdigest())")
$PY $TREE/stack/scripts/rollhist_pair_check.py --rh $D/argv.json --b0 $R/runs/B0_s$seed/argv.json --bank-sha256 $sha > $D/pair_$tag.log 2>&1 || afail "pair check: $(tail -c 300 $D/pair_$tag.log)"
if [ ! -s $D/selfcheck/selfcheck.json ]; then
  echo "$(date -u +%T) selfcheck $tag"
  RH_TREE=$TREE $PY $HERE/rh_selfcheck.py $D/argv.json $D/selfcheck $D/selfcheck.result.json > $D/selfcheck.log 2>&1 || afail "selfcheck refused: $(tail -c 400 $D/selfcheck.log | tr '\n' ' ')"
fi
[ -s $D/selfcheck/selfcheck.json ] || afail "selfcheck wrote no selfcheck.json: $(tail -c 300 $D/selfcheck.log | tr '\n' ' ')"
$PY $HERE/rh_gate_record.py mint --tag $tag --readout $RH_READOUT > $D/mint.log 2>&1 || afail "gate record refused: $(tail -c 400 $D/mint.log | tr '\n' ' ')"
$PY $HERE/rh_gate_record.py verify --tag $tag > $D/verify.log 2>&1 || afail "gate record verify: $(tail -c 300 $D/verify.log | tr '\n' ' ')"
bash $HERE/chk_arm_rhp3.sh mint $tag > $D/mint_token.log 2>&1 || afail "gate token mint refused: $(tail -c 500 $D/mint_token.log | tr '\n' ' ')"
grep -q "^TOKEN-WRITTEN $tag" $D/mint_token.log && [ -s $D/gate/gate_token.json ] || afail "no gate token written: $(tail -c 300 $D/mint_token.log | tr '\n' ' ')"
echo "$(date -u +%T) train $tag"
mkdir -p $OUT
cd $TREE || afail "cd tree"
bash $HERE/chk_arm_rhp3.sh run $tag > $D/train.log 2>&1      # REFUSED (rc 2, nothing started) unless the token binds exactly this argv, commit, data, tree, trainer, checker, template and pair
rc=$?
[ $rc -eq 0 ] || afail "trainer / gate runner rc=$rc: $(tail -c 400 $D/train.log | tr '\n' ' ')"
$PY $DS/chainlib.py summary $OUT 3000 > $OUT/summary_check.json 2> $OUT/summary_check.err || afail "summary: $(cat $OUT/summary_check.err | tail -c 300)"
$PY $DS/chainlib.py post $OUT/ckpt_last.pt $INIT > $OUT/post.json 2> $OUT/post.err || afail "post gate (zero-gradient / unmoved tensors): $(tail -c 300 $OUT/post.err)"
PYTHONPATH=$TREE/stack:$TREE $PY $TREE/stack/scripts/zerograd_gate.py --ckpt $OUT/ckpt_last.pt --json-out $D/zerograd_$tag.json > $D/zerograd_$tag.log 2>&1 || echo "zerograd_gate.py rc!=0 (report only; the fatal gate is chainlib post): $(tail -n 2 $D/zerograd_$tag.log | tr '\n' ' ')"
$PY $S1/n1dc/kclip.py $OUT/train_log.jsonl --json > $D/kclip_$tag.json 2> $D/kclip_$tag.err || afail "K-CLIP: $(tail -c 300 $D/kclip_$tag.err)"
echo "ZZ ARM $tag DONE ZZ" | tee $D/ARM_DONE
