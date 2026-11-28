#!/bin/bash
# STRAT1 NAVRB2 chain (PREREG_V8F_NAVRB2 section 4.1, frozen card sha256 e238ba8d...; PI go "add the channel for the roadabout" 2026-11-17).
# 4 base arms in priority order: N1R:s0 S1R:s0 N1R:s1 S1R:s1 (N1 / S1 + the five --nav-rb flags, lr x10). Conditional arms (S1dR, N1Z, s10 / s11) are NOT in this chain (PI go on cost first).
# Tree = /root/strat1/navrb2/tree (4dedac45 + navrb2.patch, verified against navrb2_tree.sha256); G-SUITE result gsuite_result_navrb2.json (checker `suite`, for this commit).
# Per arm: strat1_check.py mint (token) -> run (refuses without the token) -> post (moved identity incl. e_nav_rb.weight) -> zerograd_gate on e_nav_rb.weight -> `ZZ ARM <tag> DONE ZZ`.
# STAGED, NOT LAUNCHED. The queue script exec's this under the pod GPU lock; or by hand:
#   nohup setsid flock /root/gpu_queue.lock bash /root/strat1/navrb2/navrb2_chain.sh < /dev/null > /root/strat1/logs/navrb2.log 2>&1 &
# Ends with `ZZ STRAT1 NAVRB2 DONE ZZ` or `ZZ STRAT1 NAVRB2 FAIL ZZ <why>`. A STOP file at /root/strat1/chains/STOP ends it between arms.
# Edit COMMIT below to the commit the MM lands BEFORE launch (the token binds it); no other line changes.
set -u
R=/root/strat1; N=$R/navrb2; TREE=$N/tree; INP=$R/inputs; RUNS=$N/runs; CACHE=/root/data/cache_pod
PY=/root/venv/bin/python; CHK="$TREE/stack/scripts/strat1_check.py"; ZG="$TREE/stack/scripts/zerograd_gate.py"
COMMIT=dfea5c9d8ebc3d2b616dd8077cca8717b94b9ab6
SUITE=$R/gsuite_result_navrb2.json
ARMS="N1R:s0 S1R:s0 N1R:s1 S1R:s1"
fail() { echo "ZZ STRAT1 NAVRB2 FAIL ZZ $1"; exit 1; }
case "$COMMIT" in __MM*) fail "COMMIT not set";; esac
tail -c 600 $R/logs/navrb2_gsuite.log 2>/dev/null | grep -q "ZZ NAVRB2 GSUITE DONE ZZ" || fail "the NAVRB2 G-SUITE has not passed (logs/navrb2_gsuite.log)"
[ -s "$SUITE" ] || fail "no G-SUITE result $SUITE"
(cd $TREE && sha256sum -c $N/navrb2_tree.sha256) > $R/logs/navrb2_tree_verify.log 2>&1 || fail "tree files are not the NAVRB2 blobs (logs/navrb2_tree_verify.log)"
sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify_navrb2.log 2>&1 || fail "inputs sha256 mismatch"
(cd $INP/strat1 && sha256sum -c $N/navrb2_inputs.sha256) >> $R/logs/inputs_verify_navrb2.log 2>&1 || fail "navtext_nav_rb.npz / report sha256 mismatch"
unset CUDA_VISIBLE_DEVICES
mkdir -p $R/chains $RUNS
COMMON="--root $TREE --inputs $INP --cache-dir $CACHE --runs $RUNS --commit $COMMIT --suite $SUITE --py $PY --gpu 0"
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-99999}" -lt 3500 ]; }
echo "$(date -u +%FT%TZ) chain start commit=$COMMIT"
for arm in $ARMS; do
  kind=${arm%%:*}; seed=${arm##*:}; tag=${kind}_${seed}
  [ -e $R/chains/STOP ] && fail "STOP file before $tag"
  [ -e $RUNS/$tag/ARM_DONE ] && [ -s $RUNS/$tag/train/ckpt_last.pt ] && { echo "skip $tag (ARM_DONE, ckpt present)"; continue; }
  n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle (>= 3500 MiB used) before $tag"; sleep 10; done
  if [ ! -f $RUNS/$tag/gate/gate_token.json ]; then
    echo "$(date -u +%T) mint $tag"
    $PY $CHK mint $kind $seed $COMMON --steps 3000 > $RUNS/mint_$tag.log 2>&1 || fail "mint $tag refused: $(tail -c 300 $RUNS/mint_$tag.log)"
  fi
  echo "$(date -u +%T) run $tag"
  $PY $CHK run $kind $seed $COMMON --steps 3000 > $RUNS/run_$tag.log 2>&1; rc=$?
  [ $rc -eq 0 ] || fail "run $tag rc=$rc: $(tail -c 300 $RUNS/run_$tag.log)"
  init=$INP/M2b_s0/ckpt_last.pt; case $seed in s1) init=$INP/M2b_s1/ckpt_last.pt;; esac
  $PY $CHK post $kind --init $init --final $RUNS/$tag/train/ckpt_last.pt > $RUNS/post_$tag.log 2>&1 || fail "post check $tag: $(tail -c 300 $RUNS/post_$tag.log)"
  # the channel weight must have TRAINED (Adam state, no zero moments): --allow '.*' exempts every other parameter (this gate reads only e_nav_rb.weight), --control requires it among the trained ones
  PYTHONPATH=$TREE/stack:$TREE $PY $ZG --ckpt $RUNS/$tag/train/ckpt_last.pt --allow '.*' --control 'e_nav_rb\.weight' --json-out $RUNS/$tag/zerograd_navrb.json > $RUNS/zerograd_$tag.log 2>&1 || fail "zerograd gate $tag: $(tail -c 300 $RUNS/zerograd_$tag.log)"
  echo "ZZ ARM $tag DONE ZZ" | tee $RUNS/$tag/ARM_DONE
done
echo "ZZ STRAT1 NAVRB2 DONE ZZ"
