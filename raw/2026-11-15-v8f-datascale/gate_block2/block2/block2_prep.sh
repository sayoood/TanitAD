#!/bin/bash
# block2_prep.sh -- DSCALE10K block 2: everything the stage-A gate tokens of B_s1 and U_s1 need, written like mmfix/r_mint.sh (pod). Idempotent: a step whose output exists is kept. Runs INSIDE block2.sh
# (under /root/gpu_queue.lock; the only GPU steps are the two reach probes, about 15 s each in block 1) or by hand under the same lock. Steps:
#   1 gen_argv_s1.py                            argv/{B_s1,U_s1}.{A,B.tmpl}.json   (seed 1; stage B --seed 1 --dagger-seed 1; nothing else differs from B_s0 / U_s0)
#   2 dry record per arm (CPU)                  dry/<ARM>.A.dry.json               (the trainer's own build_data via dscale10k_check.py dry)
#   3 init_s1.py (CPU)                          gate/init_s1.json                  (G-HYG + C-INIT; refuses if the pod tree is not the one G-SUITE ran on)
#   4 reach probes (GPU, only while idle)       gate/probe2/<ARM>.A.json           (G-LIVE; B_s1 corpus + control, U_s1 corpus / 10k / mixed + control)
#   5 assemble_evidence_s1.py                   gate/evidence_<ARM>.json           (refuses on any failed item or a regression arm not RED)
#   6 chk_arm.sh mint ARM A, verify_token_b2.py ARM A                              (the checker writes a token only when every required check reads PASS)
# Ends with `ZZ BLOCK2-PREP DONE ZZ` or `ZZ BLOCK2-PREP FAIL ZZ <why>` (last line of the output).
set -u
D=${DS_ROOT:-/root/dscale10k}; G=$D/gate; PY=${DS_PY:-/root/venv/bin/python}; B2=$D/block2_files
fail() { echo "ZZ BLOCK2-PREP FAIL ZZ $1"; exit 1; }
cd $D || fail "cd"
export PYTHONPATH="$D/tree/stack:$D/tree/stack/scripts:$D/tree:$D/study" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_XET=1 PYTHONSAFEPATH=1
gpu_idle() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -lt 3500 ]; }
for f in gen_argv_s1.py init_s1.py assemble_evidence_s1.py verify_token.py; do [ -s $B2/$f ] || fail "missing $B2/$f"; done
cp -n $B2/assemble_evidence_s1.py $G/assemble_evidence_s1.py; cp -n $B2/init_s1.py $G/init_s1.py; cp -n $B2/verify_token.py $G/verify_token_b2.py
ARMS="B_s1 U_s1"
for a in $ARMS; do [ -s $G/tokens/$a.A/gate_token.json ] || NEED=1; done
[ -z "${NEED:-}" ] && { echo "tokens of $ARMS exist (kept)"; echo "ZZ BLOCK2-PREP DONE ZZ"; exit 0; }
[ -s argv/B_s1.A.json ] && [ -s argv/U_s1.B.tmpl.json ] || { DS_ROOT=$D $PY $B2/gen_argv_s1.py > $G/gen_argv_s1.log 2>&1 || fail "gen_argv_s1: $(tail -c 300 $G/gen_argv_s1.log)"; }
for a in $ARMS; do
  if [ -s dry/$a.A.dry.json ]; then echo "$(date -u +%T) dry $a exists (kept)"; else
    echo "$(date -u +%T) dry $a"
    CUDA_VISIBLE_DEVICES= $PY study/dscale10k_check.py dry --argv argv/$a.A.json --root $D/tree --out dry/$a.A.dry.json > dry/$a.A.dry.log 2>&1 || fail "dry $a: $(tail -c 300 dry/$a.A.dry.log)"
  fi
done
if [ ! -s $G/init_s1.json ]; then CUDA_VISIBLE_DEVICES= $PY $G/init_s1.py > $G/init_s1.log 2>&1 || fail "init_s1: $(tail -c 400 $G/init_s1.log)"; fi
echo "$(date -u +%T) init_s1: $(tail -n 1 $G/init_s1.log | cut -c1-200)"
mkdir -p $G/probe2
for a in $ARMS; do
  [ -s $G/probe2/$a.A.json ] && continue
  n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle before the $a reach probe"; sleep 10; done
  $PY $G/reach_probe2.py $D/argv/$a.A.json $G/probe2/$a.A.json $D/merged/v9_labels_union.manifest.json --reach > $G/probe2/$a.A.log 2>&1 || fail "reach probe $a: $(tail -c 400 $G/probe2/$a.A.log)"
  echo "$(date -u +%T) reach probe $a: $(tail -n 1 $G/probe2/$a.A.log | cut -c1-200)"
done
$PY $G/assemble_evidence_s1.py $ARMS > $G/assemble_s1.log 2>&1 || fail "assemble_evidence_s1: $(tail -c 400 $G/assemble_s1.log)"
echo "$(date -u +%T) evidence assembled"
for a in $ARMS; do
  [ -s $G/tokens/$a.A/gate_token.json ] && continue
  bash $D/chk_arm.sh mint $a A > $G/mint_$a.A.log 2>&1 || fail "mint $a A refused: $(tail -c 600 $G/mint_$a.A.log | tr '\n' ' ')"
  echo "$(date -u +%T) $(tail -n 1 $G/mint_$a.A.log | cut -c1-200)"
  $PY $G/verify_token_b2.py $a A > $G/verify_$a.A.log 2>&1 || fail "verify_token $a A: $(tail -c 300 $G/verify_$a.A.log)"
done
echo "ZZ BLOCK2-PREP DONE ZZ"
