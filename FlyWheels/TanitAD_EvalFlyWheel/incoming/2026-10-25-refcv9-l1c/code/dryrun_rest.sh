#!/bin/bash
# dryrun_rest.sh -- the rest of the analysis-chain regression test on L1's own data (see dryrun_repro_l1.sh): braking AUC, verdict, next-lever arms.  Not part of the verdict.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_l1c; CODE=$W/code; D=F:/l1c/dry
T=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58
R8=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval;$R8" PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TMP=F:/l1c/tmp TEMP=F:/l1c/tmp
cd "$W"
"$PY" -X utf8 "$CODE/l1c_roc.py" "$D/result" > "$D/roc_dry.log" 2>&1
echo "DRY_ROC_DONE $(date +%FT%T%z)" >> "$D/roc_dry.log"
"$PY" -X utf8 "$CODE/l1c_verdict.py" "$D/result" > "$D/verdict_dry.log" 2>&1
echo "DRY_VERDICT_DONE $(date +%FT%T%z)" >> "$D/verdict_dry.log"
"$PY" -X utf8 "$CODE/l1c_next.py" --out "$D/result" --bank-fresh "$D" --tree-root E:/refcv8_eval_kit/trainfit --tree step30000ema_train139 > "$D/next_dry.log" 2>&1
echo "DRY_NEXT_DONE $(date +%FT%T%z)" >> "$D/next_dry.log"
