#!/bin/bash
# FORK of block3/block3_prep.sh for the conditional arm X_s0 (the 10,000 data-scale clips alone; card 2.3 row X_s0, section 6 DATA-HURTS). Changes: blockx/FORK_NOTES.md. NOT launched: the MM asks the PI first.
# block3_prep.sh -- DSCALE10K block 3: everything the stage-A gate tokens of UE_s0 and BL_s0 need, as block2_prep.sh did for B_s1 / U_s1 (pod). Idempotent: a step whose output exists is kept. Runs INSIDE block3.sh
# (under /root/gpu_queue.lock via queue_block3.sh; the only GPU steps are the two reach probes, about 15 s each in block 1) or by hand under the same lock. Steps:
#   1 gen_argv_b3.py                            argv/{UE_s0,BL_s0}.A.json          (seed 0; --steps 4 x S_U of dry/U_s0.A.dry.json = 61,568 on both; nothing else differs from U_s0 / B_s0)
#   2 dry record per arm (CPU)                  dry/<ARM>.A.dry.json               (the trainer's own build_data via the block3/study checker `dry`)
#   3 init_b3.py (CPU)                          gate/init_b3.json                  (G-HYG + C-INIT; refuses if the pod tree is not the one G-SUITE ran on)
#   4 reach probes (GPU, only while idle)       gate/probe2/<ARM>.A.json           (G-LIVE; UE_s0 corpus / 10k / mixed + control, BL_s0 corpus + control)
#   5 assemble_evidence_b3.py                   gate/evidence_<ARM>.json           (adds ue_steps = 4 x S_U of U_s0's dry record; refuses on any failed item or a regression arm not RED)
#   6 chk_arm3.sh mint ARM A, verify_token3.py ARM A                               (the MM-ruled checker writes a token only when every required check reads PASS)
# Ends with `ZZ BLOCKX-PREP DONE ZZ` or `ZZ BLOCKX-PREP FAIL ZZ <why>` (last line of the output).
set -u
D=${DS_ROOT:-/root/dscale10k}; G=$D/gate; PY=${DS_PY:-/root/venv/bin/python}; B3=${DS_BX:-$D/blockx}
fail() { echo "ZZ BLOCKX-PREP FAIL ZZ $1"; exit 1; }
cd $D || fail "cd"
export PYTHONPATH="$D/tree/stack:$D/tree/stack/scripts:$D/tree:$B3/study:$D/study" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_XET=1 PYTHONSAFEPATH=1
gpu_idle() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -lt 3500 ]; }
for f in xview.py gen_argv_x.py init_x.py assemble_evidence_x.py verify_token_x.py chk_armx.sh study/dscale10k_check.py; do [ -s $B3/$f ] || fail "missing $B3/$f"; done
ARMS="X_s0"
for a in $ARMS; do [ -s $G/tokens/$a.A/gate_token.json ] || NEED=1; done
[ -z "${NEED:-}" ] && { echo "tokens of $ARMS exist (kept)"; echo "ZZ BLOCKX-PREP DONE ZZ"; exit 0; }
[ -d /root/data/cache_x_dscale/train/clips ] && [ -s /root/data/cache_x_dscale/train/manifest.jsonl ] || { $PY $B3/xview.py > $G/xview.log 2>&1 || fail "xview: $(tail -c 300 $G/xview.log)"; }
[ -s argv/X_s0.A.json ] && [ -s argv/X_s0.B.tmpl.json ] || { DS_ROOT=$D $PY $B3/gen_argv_x.py > $G/gen_argv_x.log 2>&1 || fail "gen_argv_x: $(tail -c 300 $G/gen_argv_x.log)"; }
for a in $ARMS; do
  if [ -s dry/$a.A.dry.json ]; then echo "$(date -u +%T) dry $a exists (kept)"; else
    echo "$(date -u +%T) dry $a"
    CUDA_VISIBLE_DEVICES= $PY $B3/study/dscale10k_check.py dry --argv argv/$a.A.json --root $D/tree --out dry/$a.A.dry.json > dry/$a.A.dry.log 2>&1 || fail "dry $a: $(tail -c 300 dry/$a.A.dry.log)"
  fi
done
if [ ! -s $G/init_x.json ]; then CUDA_VISIBLE_DEVICES= $PY $B3/init_x.py > $G/init_x.log 2>&1 || fail "init_x: $(tail -c 400 $G/init_x.log)"; fi
echo "$(date -u +%T) init_x: $(tail -n 1 $G/init_x.log | cut -c1-200)"
[ -s $G/probe2/UE_s0.A.json ] || fail "no gate/probe2/UE_s0.A.json: the X evidence takes its tenk record and control from UE_s0 (no GPU probe is run for X)"
$PY $B3/assemble_evidence_x.py $ARMS > $G/assemble_x.log 2>&1 || fail "assemble_evidence_x: $(tail -c 400 $G/assemble_x.log)"
echo "$(date -u +%T) evidence assembled"
for a in $ARMS; do
  [ -s $G/tokens/$a.A/gate_token.json ] && continue
  bash $B3/chk_armx.sh mint $a A > $G/mint_$a.A.log 2>&1 || fail "mint $a A refused: $(tail -c 600 $G/mint_$a.A.log | tr '\n' ' ')"
  echo "$(date -u +%T) $(tail -n 1 $G/mint_$a.A.log | cut -c1-200)"
  $PY $B3/verify_token_x.py $a A > $G/verify_$a.A.log 2>&1 || fail "verify_token $a A: $(tail -c 300 $G/verify_$a.A.log)"
done
echo "ZZ BLOCKX-PREP DONE ZZ"
