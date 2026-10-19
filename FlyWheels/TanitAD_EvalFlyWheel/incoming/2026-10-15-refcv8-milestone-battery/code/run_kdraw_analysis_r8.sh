#!/bin/sh
# run_kdraw_analysis_r8.sh [tree] -- the ONE command for the K-draw read.  CPU only; works with whatever seeds are rolled (incomplete sets => PENDING).
#   sh <PKG>/code/run_kdraw_analysis_r8.sh                       # EMA-30k (exploratory)
#   sh <PKG>/code/run_kdraw_analysis_r8.sh step48852ema_unc      # the registered (final) read
# Writes <PKG>/raw/kdraw/KDRAW_E1.json + KDRAW_E1.md (final tree: pass --out-dir to keep the 30k files).  Judge by the JSON, not the exit code.
TREE="${1:-step30000ema_unc}"
PKGW="${R8_PKGW:-C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
PY="${R8_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$PKGW/code"
OUTD="$PKGW/raw/kdraw"; [ "$TREE" = "step30000ema_unc" ] || OUTD="$PKGW/raw/kdraw/$TREE"
"$PY" -X utf8 "$PKGW/code/kdraw_e1_r8.py" run --tree "$TREE" --out-dir "$OUTD"
