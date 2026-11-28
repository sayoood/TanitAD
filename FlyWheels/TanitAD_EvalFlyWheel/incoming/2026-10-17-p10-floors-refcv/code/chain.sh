#!/usr/bin/env bash
# S1 chain: wait for the two P10 fits -> 4 floor/analysis jobs in parallel -> 2 fallback CV fits -> finalize.
cd "$(dirname "$0")"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export OMP_NUM_THREADS=2 S1_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONPATH="E:/Projects/TanitAD/stack;E:/Projects/TanitAD/taniteval"
R=../raw
until [ -f "$R/train_S2_meta.json" ] && [ -f "$R/train_S6_meta.json" ]; do
  if grep -q Traceback "$R/fit_p10_S2.log" "$R/fit_p10_S6.log" 2>/dev/null; then echo "ZZCHAIN-FIT-FAILZZ"; exit 3; fi
  sleep 5
done
echo "ZZCHAIN-FITS-OKZZ $(date +%T)"
for H in S2 S6; do for s in 0 1; do
  $PY run_refcv7_floors.py $H $s > $R/run_${H}_s$s.log 2>&1 &
done; done
wait
for H in S2 S6; do grep -q "ZZFLOORS-$H-s0-DONEZZ" $R/run_${H}_s0.log && grep -q "ZZFLOORS-$H-s1-DONEZZ" $R/run_${H}_s1.log || { echo "ZZCHAIN-RUN-FAIL-$H"; }; done
for H in S2 S6; do $PY run_fallback_cv.py $H > $R/fallback_$H.log 2>&1 & done
wait
$PY finalize.py > $R/finalize.log 2>&1
echo "ZZCHAIN-ALL-DONEZZ $(date +%T)"
