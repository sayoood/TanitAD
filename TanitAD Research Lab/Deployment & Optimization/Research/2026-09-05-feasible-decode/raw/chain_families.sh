#!/bin/sh
# P3 -- the FOUR FAMILIES at T1 for the feasibility-aware arms, paired against the banked
# base over the shared `ha0` floor. ZERO GPU: every arm is a deterministic geometric
# function of the banked base rollout, so no model is called.
#
# T1 = self-action OPEN LOOP (PI ruling 2026-09-02). Never a closed-loop claim.
set -u
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
REPO="/c/Users/Admin/refcv4b_repo"
BASE="/c/Users/Admin/_wp56/dump/refcv3_40284_dump"
RUN="/c/Users/Admin/feasdec/run"
RAW="/c/Users/Admin/feasdec/raw"
export TANITAD_REPO="$REPO" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=3
export PYTHONPATH="$REPO/stack;$REPO/taniteval;$REPO"

for arm in projoff proj07 proj07e; do
  echo "ZZPAIR-START-$arm-$(date -u +%H:%M:%S)Z-ZZ"
  "$PY" -u "$REPO/taniteval/tools/paired_openloop.py" \
     --a-dump "$BASE" --a-name base --a-arm os \
     --b-dump "$RUN/${arm}_dump" --b-name "$arm" --b-arm os \
     --floor ha0 --n-boot 2000 --seed 0 \
     --out "$RAW/paired_${arm}_vs_base.json" \
     --md  "$RAW/paired_${arm}_vs_base.md" > "$RAW/paired_${arm}.log" 2>&1
  echo "ZZPAIR-DONE-$arm-rc$?-$(date -u +%H:%M:%S)Z-ZZ"
done
echo "ZZFAMILIES-COMPLETE-$(date -u +%H:%M:%S)Z-ZZ"
