#!/bin/bash
# dryrun_repro_l1.sh -- REGRESSION TEST of the L1C analysis chain on L1's own data (not part of the verdict).
# The "fresh" slot is filled with L1's TRAIN139 tree + L1's train z_tac bank, so H2c / H2cs must REPRODUCE L1's H2 / H2s (same rows, same recipe):
#   L1C S2 paired H2c-H2cs  ==  L1 S2 paired H2-H2s ;  H2c-H0 == H2-H0 ;  CF2-CF2s == followup CF2-CF2s.
# K4_fresh is EXPECTED to fail here (the train139 dump carries the 30k model's os plan, the bank the 48,852 plan): it proves the control can go RED.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_l1c; CODE=$W/code; D=F:/l1c/dry
T=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58
R8=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
mkdir -p "$D/result" F:/l1c/tmp
cp C:/Users/Admin/refcv8_pf/refcv9_l1/bank/ztac_train_s0.npz "$D/ztac_fresh.npz"
export R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval;$R8" PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TMP=F:/l1c/tmp TEMP=F:/l1c/tmp
cd "$W"
"$PY" -X utf8 "$CODE/l1c_fit_score.py" --out "$D/result" --bank-fresh "$D" --tree-root E:/refcv8_eval_kit/trainfit --tree step30000ema_train139 --no-sens > "$D/fit_dry.log" 2>&1
echo "DRY_FIT_DONE $(date +%FT%T%z)" >> "$D/fit_dry.log"
