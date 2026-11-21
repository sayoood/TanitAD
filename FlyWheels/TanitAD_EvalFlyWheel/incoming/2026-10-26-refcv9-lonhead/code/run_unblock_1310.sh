#!/usr/bin/env bash
# MM 2026-10-26 13:11: user GO for the fresh obstacle.offline download (AskUserQuestion answer "Yes, download").
cd /c/Users/Admin/refcv8_pf/refcv9_lonhead || exit 9
export TMP=F:/c_overflow_2026-10-26/tmp TEMP=F:/c_overflow_2026-10-26/tmp CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY -X utf8 code/fetch_fresh_obstacle.py > logs/fetch_fresh.log 2>&1; r1=$?
echo "ZZFETCH-$r1-ZZ"
[ $r1 -eq 0 ] || exit 1
bash code/chain_label_pipeline.sh fresh > logs/chain_fresh_withlead.log 2>&1; r2=$?
echo "ZZCHAIN-$r2-ZZ"
[ $r2 -eq 0 ] || exit 2
$PY -X utf8 code/lh_fit.py > logs/lh_fit_phase1.log 2>&1; r3=$?
echo "ZZFIT-$r3-ZZ"
exit $r3
