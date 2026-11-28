#!/bin/bash
# rh_arm.sh TAG -- ONE stage-B arm of ROLLHIST (TAG = RH_s0 | B0_s0 | RH_s1 | B0_s1). Called by rh_chain.sh, which ALREADY HOLDS /root/gpu_queue.lock (fd 9): this script takes no lock.
# Order, as STAGEB (chk_arm.sh / block2.sh) and N1DC (n1dc_arm.sh) do it: idle GPU -> pair check (on the filled argv) -> trainer --selfcheck-only on THIS argv (real bank for RH, real init) -> mint the chain's gate record
#   -> verify it -> MINT THE STUDY-SIDE GATE TOKEN (chk_arm_rh.sh mint -> dscale10k_check_rh.py, kinds RH / B0) -> train (3,000 steps) THROUGH THE TOKEN (chk_arm_rh.sh run: rc 2 without it) -> summary check -> post gate (chainlib `post`: every registered tensor has Adam state = no zero-gradient head, and every one moved against the init) -> zerograd_gate.py (report) -> K-CLIP.
# Needs $R/runs/TAG/argv.json (render_argv_rh.py fill SEED) and, for RH, $R/bank/sK/bank.json + BANK_CHECK.json. Last line: `ZZ ARM TAG DONE ZZ` or `ZZ ARM TAG FAIL ZZ <why>`.
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/rh_env.sh
tag=$1; seed=${tag##*_s}; kind=${tag%%_*}; D=$R/runs/$tag; OUT=$D/train
afail() { echo "ZZ ARM $tag FAIL ZZ $1"; exit 1; }
[ "$kind" = RH ] || [ "$kind" = B0 ] || afail "unknown tag"
rh_readout_flags; export_py; unset CUDA_VISIBLE_DEVICES
INIT=$DS/runs/UE_s$seed/A/ckpt_last.pt
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${u:-99999}" -lt 3500 ]; }
n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && afail "GPU not idle (>= 3500 MiB used)"; sleep 10; done
[ -s $D/argv.json ] || afail "no $D/argv.json (render_argv_rh.py fill $seed first)"
mkdir -p $D/gate
# the pair, on the filled argv files
if [ $kind = RH ]; then
  sha=$($PY -c "import hashlib;print(hashlib.sha256(open('$R/bank/s$seed/bank.json','rb').read()).hexdigest())")
  $PY $TREE/stack/scripts/rollhist_pair_check.py --rh $D/argv.json --b0 $R/runs/B0_s$seed/argv.json --bank-sha256 $sha > $D/pair_$tag.log 2>&1 || afail "pair check (stock rollhist_pair_check.py; it hard-codes causal_g1x): $(tail -c 300 $D/pair_$tag.log)"
else
  sha=$($PY -c "import hashlib;print(hashlib.sha256(open('$R/bank/s$seed/bank.json','rb').read()).hexdigest())" 2>/dev/null)
  $PY $TREE/stack/scripts/rollhist_pair_check.py --rh $R/runs/RH_s$seed/argv.json --b0 $D/argv.json ${sha:+--bank-sha256 $sha} > $D/pair_$tag.log 2>&1 || afail "pair check (stock rollhist_pair_check.py; it hard-codes causal_g1x): $(tail -c 300 $D/pair_$tag.log)"
fi
# the trainer's own selfcheck of THIS argv (declared_vs_built, acc_probe, dagger_probe with its p = 0 controls)
if [ ! -s $D/selfcheck/selfcheck.json ]; then
  echo "$(date -u +%T) selfcheck $tag"
  RH_TREE=$TREE $PY $HERE/rh_selfcheck.py $D/argv.json $D/selfcheck $D/selfcheck.result.json > $D/selfcheck.log 2>&1 || afail "selfcheck refused: $(tail -c 400 $D/selfcheck.log | tr '\n' ' ')"
fi
[ -s $D/selfcheck/selfcheck.json ] || afail "selfcheck wrote no selfcheck.json: $(tail -c 300 $D/selfcheck.log | tr '\n' ' ')"
# mint (chain-local record), then verify right before the launch
$PY $HERE/rh_gate_record.py mint --tag $tag --readout $RH_READOUT > $D/mint.log 2>&1 || afail "gate record refused: $(tail -c 400 $D/mint.log | tr '\n' ' ')"
$PY $HERE/rh_gate_record.py verify --tag $tag > $D/verify.log 2>&1 || afail "gate record verify: $(tail -c 300 $D/verify.log | tr '\n' ' ')"
# the study-side gate token (dscale10k_check_rh.py mint: every check the stage-B mint runs + the RH / B0 argv, bank generator record and pair checks); bound to the commit, argv sha256 and data sha256s
bash $HERE/chk_arm_rh.sh mint $tag > $D/mint_token.log 2>&1 || afail "gate token mint refused: $(tail -c 500 $D/mint_token.log | tr '\n' ' ')"
grep -q "^TOKEN-WRITTEN $tag" $D/mint_token.log && [ -s $D/gate/gate_token.json ] || afail "no gate token written: $(tail -c 300 $D/mint_token.log | tr '\n' ' ')"
# train (cwd = the tree; the argv is the one the record binds)
echo "$(date -u +%T) train $tag"
mkdir -p $OUT
cd $TREE || afail "cd tree"
bash $HERE/chk_arm_rh.sh run $tag > $D/train.log 2>&1      # the checker's runner: REFUSED (rc 2, nothing started) unless the token binds exactly this argv, commit, data, tree, trainer, checker, template and pair
rc=$?
[ $rc -eq 0 ] || afail "trainer / gate runner rc=$rc: $(tail -c 400 $D/train.log | tr '\n' ' ')"
$PY $DS/chainlib.py summary $OUT 3000 > $OUT/summary_check.json 2> $OUT/summary_check.err || afail "summary: $(cat $OUT/summary_check.err | tail -c 300)"
$PY $DS/chainlib.py post $OUT/ckpt_last.pt $INIT > $OUT/post.json 2> $OUT/post.err || afail "post gate (zero-gradient / unmoved tensors): $(tail -c 300 $OUT/post.err)"
PYTHONPATH=$TREE/stack:$TREE $PY $TREE/stack/scripts/zerograd_gate.py --ckpt $OUT/ckpt_last.pt --json-out $D/zerograd_$tag.json > $D/zerograd_$tag.log 2>&1 || echo "zerograd_gate.py rc!=0 (report only; the fatal gate is chainlib post): $(tail -n 2 $D/zerograd_$tag.log | tr '\n' ' ')"
$PY $S1/n1dc/kclip.py $OUT/train_log.jsonl --json > $D/kclip_$tag.json 2> $D/kclip_$tag.err || afail "K-CLIP: $(tail -c 300 $D/kclip_$tag.err)"
echo "ZZ ARM $tag DONE ZZ" | tee $D/ARM_DONE
