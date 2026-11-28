#!/bin/bash
# r_mint.sh -- MM 2026-11-17: after the gate re-run passes, prepare and mint the stage-A token of R_s0 (MM_RULING_R.md: the R release stays as built).
#   dry record (CPU, the trainer's own build_data on argv/R_s0.A.json) -> assemble_evidence.py R_s0 -> chk_arm.sh mint R_s0 A -> verify_token.py R_s0 A
# Waits for ZZ GATE-RERUN DONE ZZ (a FAIL there ends this script).  Sentinel (last line of gate/r_mint.log): ZZ R-MINT DONE|FAIL ZZ.
set -u
D=/root/dscale10k; G=$D/gate; PY=/root/venv/bin/python
fail() { echo "ZZ R-MINT FAIL ZZ $1"; exit 1; }
n=0; until tail -c 800 $G/gate_rerun_mm.log 2>/dev/null | grep -qE "ZZ GATE-RERUN (DONE|FAIL) ZZ"; do n=$((n+1)); [ $n -gt 360 ] && fail "no gate re-run sentinel after 6 h"; sleep 30; done
tail -c 800 $G/gate_rerun_mm.log | grep -q "ZZ GATE-RERUN DONE ZZ" || fail "the gate re-run did not pass: $(tail -n 1 $G/gate_rerun_mm.log | cut -c1-300)"
cd $D || fail "cd"
[ -s $G/MM_RULING_R.md ] || fail "no MM ruling on R_COLUMNS.md"
export PYTHONPATH="$D/tree/stack:$D/tree/stack/scripts:$D/tree:$D/study" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_XET=1
export PYTHONSAFEPATH=1                                       # no script-folder import: gate/ holds a stale dscale10k_check.py copy (2026-11-17)
if [ -s dry/R_s0.A.dry.json ]; then echo "$(date -u +%T) dry R_s0 exists (kept)"; else
echo "$(date -u +%T) dry R_s0"
CUDA_VISIBLE_DEVICES= $PY study/dscale10k_check.py dry --argv argv/R_s0.A.json --root $D/tree --out dry/R_s0.A.dry.json > dry/R_s0.A.dry.log 2>&1 || fail "dry R_s0: $(tail -c 300 dry/R_s0.A.dry.log)"
fi
n=0; until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -lt 3500 ]; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle before the R_s0 reach probe"; sleep 10; done
$PY $G/reach_probe2.py $D/argv/R_s0.A.json $G/probe2/R_s0.A.json $D/merged/v9_labels_R_s0.manifest.json --reach > $G/probe2/R_s0.A.log 2>&1 || fail "reach probe R_s0: $(tail -c 400 $G/probe2/R_s0.A.log)"
echo "$(date -u +%T) reach probe R_s0: $(tail -n 1 $G/probe2/R_s0.A.log | cut -c1-300)"
echo "$(date -u +%T) dry: $(tail -n 1 dry/R_s0.A.dry.log | cut -c1-300)"
[ -f $G/assemble_evidence.py.pre_mm ] || cp $G/assemble_evidence.py $G/assemble_evidence.py.pre_mm
echo "504fcdcb3bef0c0da529949fa01141d2839256971b18772aa3819eb9b89b2d81  $G/assemble_evidence.py.pre_mm" | sha256sum -c > /dev/null || fail "assemble_evidence.py.pre_mm is not the original"
cp $G/mmfix/assemble_evidence.py $G/assemble_evidence.py && echo "c5e176fe8d8fcdf333418a6c77dcf033e5bf80c92027e92fbf508f2d388522ec  $G/assemble_evidence.py" | sha256sum -c > /dev/null || fail "install assemble_evidence.py"
if [ ! -s $G/init_r.json ]; then CUDA_VISIBLE_DEVICES= $PY $G/mmfix/init_r.py > $G/init_r.log 2>&1 || fail "init_r: $(tail -c 300 $G/init_r.log)"; fi
echo "$(date -u +%T) init_r: $(tail -n 1 $G/init_r.log | cut -c1-200)"
$PY $G/assemble_evidence.py R_s0 > $G/assemble_R_s0.log 2>&1 || fail "assemble_evidence R_s0: $(tail -c 400 $G/assemble_R_s0.log)"
echo "$(date -u +%T) evidence R_s0 assembled"
n=0; until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -lt 3500 ]; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle before the R_s0 mint"; sleep 10; done
bash $D/chk_arm.sh mint R_s0 A > $G/mint_R_s0.A.log 2>&1 || fail "mint R_s0 A refused: $(tail -c 600 $G/mint_R_s0.A.log | tr '\n' ' ')"
echo "$(date -u +%T) $(tail -n 1 $G/mint_R_s0.A.log | cut -c1-200)"
$PY $G/verify_token.py R_s0 A > $G/verify_R_s0.A.log 2>&1 || fail "verify_token R_s0 A: $(tail -c 300 $G/verify_R_s0.A.log)"
echo "ZZ R-MINT DONE ZZ"
