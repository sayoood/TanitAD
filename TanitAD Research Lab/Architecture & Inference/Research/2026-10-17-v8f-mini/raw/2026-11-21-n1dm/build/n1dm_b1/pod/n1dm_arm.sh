#!/bin/bash
# n1dm_arm.sh SEED -- the GPU part of ONE N1DM arm. Called by n1dm_chain.sh, which ALREADY HOLDS /root/gpu_queue.lock (fd 9) for the whole chain: this script takes no lock.
# Adapted from n1dh_arm.sh (which ran): gpu idle wait -> mint (selfcheck + gate token; skipped when a token exists) -> pair on the mint argv (N1D argv + exactly --aux-route modes --w-aux-route 0.5) -> run (refuses without the token)
# -> post (route_head moved, the 3 branches moved off their zero init, no route_emb) -> zerograd gate (controls route_head.weight AND route_br.0/1/2.weight must have TRAINED) -> K-CLIP -> ZZ ARM <tag> DONE ZZ.
# COMMIT is the commit the MM landed the N1DM patch at; it is substituted by n1dm_stage.sh (the placeholder below refuses to run).
# The three post-training lines (post, zerograd, K-CLIP) were each run once by hand on a real finished checkpoint before this script was written (N1DH Amendments 4 and 5): see build/n1dm_b1/logs/dry_post_lines.log.
R=/root/strat1; N=$R/n1dm; TREE=$N/tree; INP=$R/inputs; RUNS=$N/runs; CACHE=/root/data/cache_pod
PY=/root/venv/bin/python; CHK="$TREE/stack/scripts/strat1_check.py"; ZG="$TREE/stack/scripts/zerograd_gate.py"
COMMIT=@@N1DM_COMMIT@@
SUITE=$N/gsuite_result_n1dm.json
kind=N1DM; seed=$1; tag=${kind}_${seed}
REF=$R/navrbd/runs/N1D_$seed/argv.json
fail() { echo "ZZ ARM $tag FAIL ZZ $1"; exit 1; }
case "$COMMIT" in @@*) fail "COMMIT placeholder not substituted (run n1dm_stage.sh)";; esac
unset CUDA_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1
COMMON="--root $TREE --inputs $INP --cache-dir $CACHE --runs $RUNS --commit $COMMIT --suite $SUITE --py $PY --gpu 0"
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-99999}" -lt 3500 ]; }
mkdir -p $RUNS/$tag
n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle (>= 3500 MiB used)"; sleep 10; done
if [ ! -f $RUNS/$tag/gate/gate_token.json ]; then
  echo "$(date -u +%T) mint $tag"
  $PY $CHK mint $kind $seed $COMMON --steps 3000 > $RUNS/mint_$tag.log 2>&1 || fail "mint refused: $(tail -c 300 $RUNS/mint_$tag.log)"
fi
$PY $CHK pair $kind --argv-json $RUNS/$tag/argv.json --ref-argv-json $REF > $RUNS/pair_$tag.log 2>&1 || fail "pairing: $(tail -c 300 $RUNS/pair_$tag.log)"
echo "$(date -u +%T) run $tag"
$PY $CHK run $kind $seed $COMMON --steps 3000 > $RUNS/run_$tag.log 2>&1; rc=$?
[ $rc -eq 0 ] || fail "run rc=$rc: $(tail -c 300 $RUNS/run_$tag.log)"
init=$INP/M2b_s0/ckpt_last.pt; case $seed in s1) init=$INP/M2b_s1/ckpt_last.pt;; esac
$PY $CHK post $kind --init $init --final $RUNS/$tag/train/ckpt_last.pt > $RUNS/post_$tag.log 2>&1 || fail "post check: $(tail -c 300 $RUNS/post_$tag.log)"
PYTHONPATH=$TREE/stack:$TREE $PY $ZG --ckpt $RUNS/$tag/train/ckpt_last.pt --allow '.*' --control 'route_head\.weight' --control 'route_br\.0\.weight' --control 'route_br\.1\.weight' --control 'route_br\.2\.weight' --json-out $RUNS/$tag/zerograd_n1dm.json > $RUNS/zerograd_$tag.log 2>&1 || fail "zerograd gate (route_head.weight and route_br.0-2.weight must have trained): $(tail -c 300 $RUNS/zerograd_$tag.log)"
$PY $N/kclip.py $RUNS/$tag/train/train_log.jsonl --json > $RUNS/$tag/kclip_$tag.json 2> $RUNS/kclip_$tag.err || fail "K-CLIP: $(tail -c 300 $RUNS/kclip_$tag.err)"
echo "ZZ ARM $tag DONE ZZ" | tee $RUNS/$tag/ARM_DONE
