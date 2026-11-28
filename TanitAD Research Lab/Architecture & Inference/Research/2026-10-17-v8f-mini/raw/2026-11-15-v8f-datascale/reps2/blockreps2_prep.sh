#!/bin/bash
# blockreps2_prep.sh -- DSCALE10K REPS2: everything the stage-A gate tokens of UE_s2 and UE_s3 need (pod). FORK of blockreps1_prep.sh (UE arm only). Idempotent: a step whose output exists is kept. Runs INSIDE blockreps2.sh
# (under /root/gpu_queue.lock; GPU steps = two reach probes about 15 s each) or by hand under the same lock. `cd $D` is the FIRST command. Both arms on $D/tree = cdbff971, as UE_s0 and UE_s1.
#   0 gen_argv_reps2.py                     argv/{UE_s2,UE_s3}.A.json + blockreps2/PAIRING_PROOF.json   (= UE_s0's argv with --seed and --out only; refuses otherwise)
#   1 dry build (CPU)                       dry/UE_s<k>.A.dry.json     (merged/join_union.npz sha256 before / after EQUAL)
#   2 init_reps2.py ue_s<k> (CPU)           gate/init_ue_s<k>.json     (G-HYG against B_s0; C-INIT: cfg equal to UE_s0 / UE_s1, state different)
#   3 reach probe (GPU)                     gate/probe2/UE_s<k>.A.json (gate/reach_probe2.py, unchanged)
#   4 assemble_evidence_reps2_ue.py         gate/evidence_UE_s<k>.json
#   5 chk_arm_reps2.sh mint, verify_token_reps2.py   gate/tokens/UE_s<k>.A/ , gate/verify_UE_s<k>.A.json
# gate records reused unchanged (INHERITED from seed 0 / 1, same tree): init.json, init_b3.json, init_ue_s1.json, clock, eval, ckptproc, resume, suite, cachecheck, stageb, sess.
# Ends with `ZZ REPS2-PREP DONE ZZ` or `ZZ REPS2-PREP FAIL ZZ <why>` (last line of the output).
D=${DS_ROOT:-/root/dscale10k}
cd $D || { echo "ZZ REPS2-PREP FAIL ZZ cd $D"; exit 1; }
set -u
G=$D/gate; PY=${DS_PY:-/root/venv/bin/python}; RP=${DS_RP:-$D/blockreps2}; TREE_U=$D/tree
fail() { echo "ZZ REPS2-PREP FAIL ZZ $1"; exit 1; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_XET=1 PYTHONSAFEPATH=1
pp() { echo "$1/stack:$1/stack/scripts:$1:$RP/study:$D/study"; }
gpu_idle() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -lt 3500 ]; }
wait_idle() { n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle before $1"; sleep 10; done; }
for f in gen_argv_reps2.py init_reps2.py assemble_evidence_reps2_ue.py chk_arm_reps2.sh verify_token_reps2.py study/dscale10k_check.py; do [ -s $RP/$f ] || fail "missing $RP/$f"; done
for f in gate/suite.json gate/init.json gate/init_b3.json gate/init_ue_s1.json gate/reach_probe2.py gate/gate_items.py dry/U_s0.A.dry.json argv/UE_s0.A.json argv/B_s0.A.json; do [ -s $D/$f ] || fail "missing $D/$f"; done
[ -s $G/tokens/UE_s2.A/gate_token.json ] && [ -s $G/tokens/UE_s3.A/gate_token.json ] && { echo "tokens of UE_s2 and UE_s3 stage A exist (kept)"; echo "ZZ REPS2-PREP DONE ZZ"; exit 0; }
$PY $RP/gen_argv_reps2.py > $G/gen_argv_reps2.log 2>&1 || fail "gen_argv_reps2: $(tail -c 300 $G/gen_argv_reps2.log)"
dry() {   # dry ARM TREE
  if [ -s dry/$1.A.dry.json ]; then echo "$(date -u +%T) dry $1 exists (kept)"; return 0; fi
  echo "$(date -u +%T) dry $1"
  h0=$(sha256sum merged/join_union.npz | cut -d' ' -f1)
  PYTHONPATH=$(pp $2) CUDA_VISIBLE_DEVICES= $PY $RP/study/dscale10k_check.py dry --argv argv/$1.A.json --root $2 --out dry/$1.A.dry.json > dry/$1.A.dry.log 2>&1 || fail "dry $1: $(tail -c 300 dry/$1.A.dry.log)"
  h1=$(sha256sum merged/join_union.npz | cut -d' ' -f1)
  [ "$h0" = "$h1" ] || fail "merged/join_union.npz changed under the $1 dry build (--fit-join): $h0 -> $h1"
}
mkdir -p $G/probe2
for arm in UE_s2 UE_s3; do
  k=$(echo $arm | tr 'A-Z' 'a-z')
  [ -s $G/tokens/$arm.A/gate_token.json ] && { echo "$(date -u +%T) token of $arm exists (kept)"; continue; }
  dry $arm $TREE_U
  if [ ! -s $G/init_$k.json ]; then PYTHONPATH=$(pp $TREE_U) CUDA_VISIBLE_DEVICES= $PY $RP/init_reps2.py $k > $G/init_$k.log 2>&1 || fail "init_reps2 $k: $(tail -c 400 $G/init_$k.log)"; fi
  echo "$(date -u +%T) init_$k: $(tail -n 1 $G/init_$k.log | cut -c1-200)"
  if [ ! -s $G/probe2/$arm.A.json ]; then
    wait_idle "the $arm reach probe"
    PYTHONPATH=$(pp $TREE_U) $PY $G/reach_probe2.py $D/argv/$arm.A.json $G/probe2/$arm.A.json $D/merged/v9_labels_union.manifest.json --reach > $G/probe2/$arm.A.log 2>&1 || fail "reach probe $arm: $(tail -c 400 $G/probe2/$arm.A.log)"
  fi
  echo "$(date -u +%T) reach probe $arm: $(tail -n 1 $G/probe2/$arm.A.log | cut -c1-200)"
  PYTHONPATH=$(pp $TREE_U) $PY $RP/assemble_evidence_reps2_ue.py $arm > $G/assemble_$k.log 2>&1 || fail "assemble_evidence_reps2_ue $arm: $(tail -c 400 $G/assemble_$k.log)"
  echo "$(date -u +%T) evidence $arm assembled"
  bash $RP/chk_arm_reps2.sh mint $arm A > $G/mint_$arm.A.log 2>&1 || fail "mint $arm A refused: $(tail -c 700 $G/mint_$arm.A.log | tr '\n' ' ')"
  echo "$(date -u +%T) $(tail -n 1 $G/mint_$arm.A.log | cut -c1-200)"
  PYTHONPATH=$(pp $TREE_U) $PY $RP/verify_token_reps2.py $arm A > $G/verify_$arm.A.log 2>&1 || fail "verify_token $arm A: $(tail -c 300 $G/verify_$arm.A.log)"
done
echo "ZZ REPS2-PREP DONE ZZ"
