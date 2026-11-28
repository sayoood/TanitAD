#!/bin/sh
# v2 -- corrected sequencing. Runs AFTER the coll200 panel, in PRIORITY ORDER so a killed
# chain still yields the primary endpoint:
#
#   1. as soon as BOTH coll200 seeds land: analyze_veto.py -> the PRIMARY endpoint
#      (fan_contact) against BOTH floors. CPU ONLY, ~1 min, safe to run beside training.
#   2. WAIT FOR ALL FOUR ARMS TO FINISH before touching the GPU again.
#      ⛔ v1 launched the T1 evals while ctrl0 and ctrl_null s1 were still training. torch
#      spawns ~113 threads per process and CONCURRENT ARMS MAKE NO PROGRESS (MEASURED: 7
#      arms at 0-6 % sm for 50 min; one arm 232 s). Two sibling refav1 arms are already on
#      this 4060, so adding a third and fourth of my own is exactly that trap.
#   3. the T1 four-family evals (GPU, expensive), then read_t1_families.py.
#
# Markers are OPAQUE (ZZ...ZZ) and disjoint from the searched tokens, so a client-side
# filter cannot match this script's own echoed command line.
set -u
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
REPO="/c/Users/Admin/refcv4b_repo"
OUT="/c/Users/Admin/veto_run/run"
WORK="/c/Users/Admin/tanitad-rlgen"
export TANITAD_REPO="$REPO" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6
export PYTHONPATH="$REPO/stack;$REPO"

# ---- 1. the PRIMARY endpoint, as soon as the lever + its replicate exist ----
i=0
while [ $i -lt 360 ]; do
  [ -s "$OUT/s0/coll200/arm_summary.json" ] && [ -s "$OUT/s1/coll200/arm_summary.json" ] && break
  i=$((i + 1)); sleep 30
done
if [ ! -s "$OUT/s1/coll200/arm_summary.json" ]; then
  echo "ZZCHAIN-TIMEOUT-NO-S1-$(date -u +%H:%M:%S)Z-ZZ"; exit 2
fi
echo "ZZCHAIN-ARMS-READY-$(date -u +%H:%M:%S)Z-ZZ"

"$PY" -u "$WORK/raw/analyze_veto.py" --run-dir "$OUT" --arm coll200 \
    --null-arm ctrl_null --n-boot 4000 --seed 11 \
    --out "$WORK/raw/coll_verdict_coll200.json" > "$WORK/raw/coll_verdict_coll200.log" 2>&1
echo "ZZVERDICT-rc$?-$(date -u +%H:%M:%S)Z-ZZ"
if [ -s "$WORK/raw/coll_verdict_coll200.json" ]; then
  echo "ZZPRIMARY-$("$PY" -c "import json;d=json.load(open(r'$WORK/raw/coll_verdict_coll200.json'))['metrics']['fan_contact'];print('%+.6f_%s_%+.6f_%+.6f'%(d['lever_s0']['delta'],d['lever_s0']['sep'],d['lever_s1']['delta'],d['replicate_floor']['delta']))" 2>/dev/null)-ZZ"
else
  echo "ZZVERDICT-NO-OUTPUT-ZZ"
fi

# ---- 2. do NOT touch the GPU until every arm has finished ----
j=0
while [ $j -lt 360 ]; do
  if [ -s "$OUT/s0/ctrl0/arm_summary.json" ] && [ -s "$OUT/s1/ctrl_null/arm_summary.json" ]; then
    break
  fi
  # a dead runner is also a terminal state -- do not wait forever on a corpse
  if ! ps -ef 2>/dev/null | grep -q "[r]un_coll_arms"; then
    echo "ZZRUNNER-GONE-$(date -u +%H:%M:%S)Z-ZZ"; break
  fi
  j=$((j + 1)); sleep 30
done
echo "ZZALL-ARMS-SETTLED-$(date -u +%H:%M:%S)Z-ZZ"

# ---- 3. the T1 four-family evals (GPU) ----
ARMS="s0/coll200 s1/coll200" sh "$WORK/raw/run_eval_veto.sh" >> "$WORK/raw/t1_coll.log" 2>&1
echo "ZZT1EVAL-rc$?-$(date -u +%H:%M:%S)Z-ZZ"

if [ -s "$OUT/paired_s0-coll200_vs_base.json" ]; then
  "$PY" -u "$WORK/raw/read_t1_families.py" \
      --s0 "$OUT/paired_s0-coll200_vs_base.json" \
      --s1 "$OUT/paired_s1-coll200_vs_base.json" \
      --out "$WORK/raw/t1_families_coll200.json" >> "$WORK/raw/t1_coll.log" 2>&1
  echo "ZZFAMILIES-rc$?-$(date -u +%H:%M:%S)Z-ZZ"
else
  echo "ZZFAMILIES-NO-PAIRED-INPUT-ZZ"
fi
echo "ZZCHAIN-COMPLETE-$(date -u +%H:%M:%S)Z-ZZ"
