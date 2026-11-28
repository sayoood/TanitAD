#!/bin/bash
# STRAT1 CKPT120 chain (PREREG_V8F_CKPT120 section 5; PI go 2026-11-17: 120 m S output, H2, 7 arms, base N1, ~1.75 pod GPU-h).
# Arms in priority order: H1_s0 H1_s1 H2_s0 H2_s1 H1_rep R1_s0 R2_s0   (H1 = N1 + the far-point levers; H2 adds nav_s; R1 = label-permuted far head; R2 = donor nav_s; H1_rep = H1_s0's flags, --seed 2).
# Inits (the checker renders them, base N1's argv of that seed): s0 and rep = M2b_s0, s1 = M2b_s1.
# Tree new11 = new4 (a4357e79) + the aedf8716 files (sha256 list ckpt120_new11.sha256); G-SUITE result gsuite_result_ckpt120_fix3.json (written by ckpt120_check.py suite for that commit).
# Per arm: ckpt120_check.py mint (token) -> run (the runner refuses without the token) -> post (moved identity) -> ckpt120_reach.py (K-REACH-e) -> `ZZ ARM <tag> DONE ZZ`.
# STAGED, NOT LAUNCHED: needs the MM's launch.  Launch ONLY under the pod GPU queue lock:
#   nohup setsid flock /root/gpu_queue.lock bash /root/strat1/strat1_ckpt120_fix3.sh < /dev/null > /root/strat1/logs/ckpt120_fix3.log 2>&1 &
# Ends with `ZZ STRAT1 CKPT120 DONE ZZ` or `ZZ STRAT1 CKPT120 FAIL ZZ <why>`.  A STOP file at /root/strat1/chains/STOP ends it between arms.
# K-REACH-e: HEAD-DEAD (rc 1) FAILS the chain; INCONCLUSIVE (rc 2, e.g. an ambiguous shape alignment: the checkpoint is not touched) is logged as `ZZ ARM <tag> REACH-INCONCLUSIVE ZZ`
# and the chain goes on, the gate is re-runnable on the checkpoint without the GPU: ckpt120_reach.py --ckpt <run>/train/ckpt_last.pt --out <run>/reach.json [--expect-nav-s on].
set -u
R=/root/strat1; TREE=$R/new11; INP=$R/inputs; RUNS=$R/runs; CACHE=/root/data/cache_pod
PY=/root/venv/bin/python; CHK="$TREE/stack/scripts/ckpt120_check.py"; REACH="$TREE/stack/scripts/ckpt120_reach.py"
COMMIT=aedf8716989494464277e50ec4decbbf1a2e65dc
SUITE=$R/gsuite_result_ckpt120_fix3.json
ARMS="H1:s0 H1:s1 H2:s0 H2:s1 H1:rep R1:s0 R2:s0"
fail() { echo "ZZ STRAT1 CKPT120 FAIL ZZ $1"; exit 1; }
tail -c 600 $R/logs/ckpt120_gsuite_fix3.log 2>/dev/null | grep -q "ZZ CKPT120 GSUITE FIX3 DONE ZZ" || fail "the aedf8716 G-SUITE has not passed (logs/ckpt120_gsuite_fix3.log)"
[ -s "$SUITE" ] || fail "no G-SUITE result $SUITE"
(cd $TREE && sha256sum -c $R/ckpt120_new11.sha256) > $R/logs/ckpt120_tree_verify_fix3.log 2>&1 || fail "new11 files are not the aedf8716 blobs (logs/ckpt120_tree_verify_fix3.log)"
D=$(diff -rq -x __pycache__ $R/new4/stack $TREE/stack; diff -rq -x __pycache__ $R/new4/taniteval $TREE/taniteval)
[ "$(printf '%s\n' "$D" | grep -v '^$' | wc -l)" -eq 20 ] || fail "new11 differs from new4 beyond the 20 commit files"
sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify_ckpt120_fix3.log 2>&1 || fail "inputs sha256 mismatch"
sha256sum -c $R/inputs_ckpt120.sha256 >> $R/logs/inputs_verify_ckpt120_fix3.log 2>&1 || fail "ckpt120 inputs (RC-A120 table, nav_s table, reports) sha256 mismatch"
unset CUDA_VISIBLE_DEVICES
mkdir -p $R/chains $RUNS
# fix3: H2_s0 gate/ left by the mint refused at 213250ce is kept aside (not deleted) so the mint runs fresh at aedf8716
if [ -d $RUNS/H2_s0/gate ] && [ ! -f $RUNS/H2_s0/gate/gate_token.json ] && [ ! -e $RUNS/H2_s0/gate.refused_213250ce ]; then mv $RUNS/H2_s0/gate $RUNS/H2_s0/gate.refused_213250ce || fail "cannot move aside H2_s0/gate"; fi
COMMON="--root $TREE --inputs $INP --cache-dir $CACHE --runs $RUNS --commit $COMMIT --suite $SUITE --py $PY --gpu 0"
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-99999}" -lt 3500 ]; }
echo "$(date -u +%FT%TZ) chain start commit=$COMMIT"
for arm in $ARMS; do
  kind=${arm%%:*}; seed=${arm##*:}; tag=${kind}_${seed}
  [ -e $R/chains/STOP ] && fail "STOP file before $tag"
  [ -e $RUNS/$tag/ARM_DONE ] && [ -s $RUNS/$tag/train/ckpt_last.pt ] && { echo "skip $tag (ARM_DONE, trained at 213250ce, ckpt present)"; continue; }
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
  ens=off; case $kind in H2|R2) ens=on;; esac
  PYTHONPATH=$TREE/stack:$TREE $PY $REACH --ckpt $RUNS/$tag/train/ckpt_last.pt --out $RUNS/$tag/reach.json --expect-nav-s $ens > $RUNS/reach_$tag.log 2>&1; rr=$?
  [ $rr -eq 1 ] && fail "K-REACH-e HEAD-DEAD $tag: $(tail -c 300 $RUNS/reach_$tag.log)"
  echo "ZZ ARM $tag DONE ZZ" | tee $RUNS/$tag/ARM_DONE
  [ $rr -eq 0 ] || echo "ZZ ARM $tag REACH-INCONCLUSIVE ZZ rc=$rr: $(tail -c 300 $RUNS/reach_$tag.log)"
done
echo "ZZ STRAT1 CKPT120 DONE ZZ"
