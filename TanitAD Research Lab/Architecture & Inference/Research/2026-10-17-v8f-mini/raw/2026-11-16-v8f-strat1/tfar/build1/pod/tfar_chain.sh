#!/bin/bash
# STRAT1 TFAR chain (PREREG_V8F_TFAR section 5, frozen card sha256 816d4357...; the PI's go on Q1-Q3 is the MM's to wait for: this script is STAGED, NOT LAUNCHED).
# Arms in the card's priority order: H2:s10 H2:s11 (CKPT120's 4-seed comparators, skipped when the CKPT120 chain already trained them) TF:s0 TF:s1 TF:rep R3:s0 TZ:s0 TN:s0   (about 1.6 pod GPU-h for TF..TN, E).
# Conditional arms (TF:s10 TF:s11 TN:s1 TFnt TFd TFc/H2c) are NOT in this chain: each runs only on its committed trigger and the PI's go on cost first (set ARMS by hand, or re-run with ARMS="...").
# Tree = /root/strat1/tfar/tree (aedf8716 = /root/strat1/new11 + the files of tfar_tree.sha256, verified); G-SUITE result gsuite_result_tfar.json (tfar_check.py suite, for this commit).
# Per arm: tfar_check.py mint (token) -> run (refuses without the token) -> post (moved identity incl. e_ck_far / e_nav_s_t) -> tfar_reach.py (K-REACH-e, names ids from the checkpoint's param_names) -> `ZZ ARM <tag> DONE ZZ`.
# Launch ONLY under the pod GPU queue lock (the queue script does it):
#   nohup setsid flock /root/gpu_queue.lock bash /root/strat1/tfar/tfar_chain.sh < /dev/null > /root/strat1/logs/tfar.log 2>&1 &
# Ends with `ZZ STRAT1 TFAR DONE ZZ` or `ZZ STRAT1 TFAR FAIL ZZ <why>`. A STOP file at /root/strat1/chains/STOP ends it between arms.
# K-REACH-e: HEAD-DEAD (rc 1) FAILS the chain; INCONCLUSIVE (rc 2) is logged as `ZZ ARM <tag> REACH-INCONCLUSIVE ZZ` and the chain goes on (the gate is re-runnable on the checkpoint without the GPU).
# The MM sets COMMIT below to the commit it lands BEFORE launch (the token binds it); no other line changes.
set -u
R=/root/strat1; N=$R/tfar; TREE=$N/tree; INP=$R/inputs; RUNS=$N/runs; CACHE=/root/data/cache_pod; PRIOR=$R/runs
PY=/root/venv/bin/python; CHK="$TREE/stack/scripts/tfar_check.py"; REACH="$TREE/stack/scripts/tfar_reach.py"
COMMIT=__MM_FILLS_THE_TFAR_COMMIT__
SUITE=$R/gsuite_result_tfar.json
ARMS=${ARMS:-"H2:s10 H2:s11 TF:s0 TF:s1 TF:rep R3:s0 TZ:s0 TN:s0"}
fail() { echo "ZZ STRAT1 TFAR FAIL ZZ $1"; exit 1; }
case "$COMMIT" in __MM*) fail "COMMIT not set";; esac
tail -c 600 $R/logs/tfar_gsuite.log 2>/dev/null | grep -q "ZZ TFAR GSUITE DONE ZZ" || fail "the TFAR G-SUITE has not passed (logs/tfar_gsuite.log)"
[ -s "$SUITE" ] || fail "no G-SUITE result $SUITE"
(cd $TREE && sha256sum -c $N/tfar_tree.sha256) > $R/logs/tfar_tree_verify.log 2>&1 || fail "tree files are not the TFAR blobs (logs/tfar_tree_verify.log)"
sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify_tfar.log 2>&1 || fail "inputs sha256 mismatch"
sha256sum -c $R/inputs_ckpt120.sha256 >> $R/logs/inputs_verify_tfar.log 2>&1 || fail "ckpt120 inputs (RC-A120 table, nav_s table, reports) sha256 mismatch"
unset CUDA_VISIBLE_DEVICES
mkdir -p $R/chains $RUNS
COMMON="--root $TREE --inputs $INP --cache-dir $CACHE --runs $RUNS --commit $COMMIT --suite $SUITE --py $PY --gpu 0"
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-99999}" -lt 3500 ]; }
echo "$(date -u +%FT%TZ) chain start commit=$COMMIT"
for arm in $ARMS; do
  kind=${arm%%:*}; seed=${arm##*:}; tag=${kind}_${seed}
  [ -e $R/chains/STOP ] && fail "STOP file before $tag"
  [ -e $RUNS/$tag/ARM_DONE ] && [ -s $RUNS/$tag/train/ckpt_last.pt ] && { echo "skip $tag (ARM_DONE, ckpt present)"; continue; }
  case $kind in H2) [ -e $PRIOR/$tag/ARM_DONE ] && [ -s $PRIOR/$tag/train/ckpt_last.pt ] && { echo "skip $tag (trained by the CKPT120 chain: $PRIOR/$tag)"; continue; };; esac
  n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle (>= 3500 MiB used) before $tag"; sleep 10; done
  if [ ! -f $RUNS/$tag/gate/gate_token.json ]; then
    echo "$(date -u +%T) mint $tag"
    $PY $CHK mint $kind $seed $COMMON --steps 3000 > $RUNS/mint_$tag.log 2>&1 || fail "mint $tag refused: $(tail -c 300 $RUNS/mint_$tag.log)"
  fi
  echo "$(date -u +%T) run $tag"
  $PY $CHK run $kind $seed $COMMON --steps 3000 > $RUNS/run_$tag.log 2>&1; rc=$?
  [ $rc -eq 0 ] || fail "run $tag rc=$rc: $(tail -c 300 $RUNS/run_$tag.log)"
  init=$INP/M2b_s0/ckpt_last.pt; case $seed in s1|s11) init=$INP/M2b_s1/ckpt_last.pt;; esac
  case $kind in TFc|H2c) init=$INP/tfar/h2_init_$seed.pt;; esac
  $PY $CHK post $kind --init $init --final $RUNS/$tag/train/ckpt_last.pt > $RUNS/post_$tag.log 2>&1 || fail "post check $tag: $(tail -c 300 $RUNS/post_$tag.log)"
  rf="--t-far off --nav-strat-t off"
  case $kind in TF|R3|TFnt|TFd|TFc) rf="--t-far on --t-far-input pred --nav-strat-t off";; TZ) rf="--t-far on --t-far-input zero --nav-strat-t off";; TN) rf="--t-far off --nav-strat-t on";; esac
  PYTHONPATH=$TREE/stack:$TREE $PY $REACH --ckpt $RUNS/$tag/train/ckpt_last.pt --out $RUNS/$tag/reach.json --expect-nav-s on $rf > $RUNS/reach_$tag.log 2>&1; rr=$?
  [ $rr -eq 1 ] && fail "K-REACH-e HEAD-DEAD $tag: $(tail -c 300 $RUNS/reach_$tag.log)"
  echo "ZZ ARM $tag DONE ZZ" | tee $RUNS/$tag/ARM_DONE
  [ $rr -eq 0 ] || echo "ZZ ARM $tag REACH-INCONCLUSIVE ZZ rc=$rr: $(tail -c 300 $RUNS/reach_$tag.log)"
done
echo "ZZ STRAT1 TFAR DONE ZZ"
