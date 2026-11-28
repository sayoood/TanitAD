#!/bin/bash
# The read of record (PREREG_V8F_LAUNCH_INTENT Amendment 1): the amended reader and the amended PREREG, extracted from the landed commit e28516e8, on lit1's manifest.
set -u
C=e28516e8af282ab094b23b8caae4b4d3c77b2b5b
O=/f/TanitAD-scratch/launch_intent_t1_out/read_A1
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-28-launch-intent-amend1"
G="git -C /c/Users/Admin/tanitad-push -c safe.directory=* -c core.autocrlf=false"
mkdir -p $O/code/read/ref
$G show "$C:$PK/read/launch_intent_read.py" > $O/code/read/launch_intent_read.py
$G show "$C:$PK/read/ref/rep_s0_speedcells.json" > $O/code/read/ref/rep_s0_speedcells.json
$G show "$C:Project Steering/PREREG_V8F_LAUNCH_INTENT.md" > $O/code/PREREG_V8F_LAUNCH_INTENT.md
RS=$(sha256sum $O/code/read/launch_intent_read.py | cut -c1-64); PS=$(sha256sum $O/code/PREREG_V8F_LAUNCH_INTENT.md | cut -c1-64)
[ "$RS" = fb3d2d7d945b54209b62d5f906d4dcea30925a24f0b807459b8cbf3d7f0747a9 ] || { echo "READER SHA $RS"; exit 9; }
[ "$PS" = 6ae79f2f451868e8da9d270214d6132f2fdc7c9f6ff608c0ec05c35a58ac1d34 ] || { echo "PREREG SHA $PS"; exit 9; }
echo "reader $RS prereg $PS"
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I F:/TanitAD-scratch/launch_intent_t1_out/read_A1/code/read/launch_intent_read.py \
  --manifest F:/TanitAD-scratch/launch_intent_t1_out/runs.json --out F:/TanitAD-scratch/launch_intent_t1_out/read_A1/LAUNCH_INTENT_READ.json \
  --prereg F:/TanitAD-scratch/launch_intent_t1_out/read_A1/code/PREREG_V8F_LAUNCH_INTENT.md --prereg-sha $PS \
  --orc-dir E:/Projects/TanitAD-artifacts/dagorc_rebase/read
echo "ZZREAD-A1-RC $?"
