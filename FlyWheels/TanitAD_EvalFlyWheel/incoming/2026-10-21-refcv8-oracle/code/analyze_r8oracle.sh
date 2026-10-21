#!/bin/sh
# analyze_r8oracle.sh -- CPU-only (re)analysis of a finished R8-ORACLE run: R8ORACLE.json + RESULT.md, mechanical from the sealed prereg.
# No GPU, no lock, no model.  The run lands on the dev box itself (run_r8oracle.sh), so there is nothing to pull; if the roll is ever
# moved to another host, copy oracle_s{0,1}.npz + dump_s{0,1}/ (md5-verified) into $ROOT first, then run this.
#   sh analyze_r8oracle.sh [ROOT]          (default C:/Users/Admin/refcv8_pf/r8oracle/run_48852ema_pf)
#   sh analyze_r8oracle.sh --selftest      (C9 decision-function mutations + C8 calibration on the BANKED trees; needs no run)
HERE=C:/Users/Admin/refcv8_pf/r8oracle/code
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=6 R8_NO_POSE_SYNC=1
if [ "$1" = "--selftest" ]; then exec "$PY" -X utf8 "$HERE/r8oracle_analyze.py" --selftest; fi
ROOT=${1:-C:/Users/Admin/refcv8_pf/r8oracle/run_48852ema_pf}
for s in 0 1; do [ -s "$ROOT/oracle_s$s.npz" ] || { echo "missing $ROOT/oracle_s$s.npz"; exit 2; }; done
"$PY" -X utf8 "$HERE/r8oracle_analyze.py" --root "$ROOT" --seeds 0,1 > "$ROOT/analyze.log" 2>&1
[ -s "$ROOT/R8ORACLE.json" ] && [ -s "$ROOT/RESULT.md" ] && grep -m1 'Verdict' "$ROOT/RESULT.md" || { echo "analysis produced no artifact; see $ROOT/analyze.log"; exit 3; }
