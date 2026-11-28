#!/usr/bin/env bash
# post_lh.sh -- after chain_lh.sh banks all three rolls: the full SHIP-R8 Amendment-5 forward on the LHS traces (code/sub_fx_lhs/run_lhs_fact8.sh),
# then the bars (code/lh_bars.py).  CPU only.  Markers ZZPOST<TAG>ZZ in logs/post_lh.log.  Judged on artifacts, never on exit codes through pipes.
set -u
PK=C:/Users/Admin/refcv8_pf/refcv9_lonhead; LOG=$PK/logs/post_lh.log; CH=$PK/logs/chain_lh.log
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
T=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58; BAT=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code
mk() { echo "ZZPOST$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
trap 'mk END "stage=$STAGE"' EXIT
STAGE=wait; mk START "pid $$"
live() { sed -n '/chain_lh restarted for the live run/,$p' "$CH"; }
until live | grep -q "ZZLHALLROLLSZZ"; do
  if live | grep -q "ZZLHABORTZZ\|ZZLHSTOPZZ\|ZZLHNOLOCKZZ"; then mk ABORT "the roll chain stopped (see chain_lh.log)"; exit 3; fi
  sleep 60
done
mk ROLLSDONE "all three rolls banked"
STAGE=forward
bash "$PK/code/sub_fx_lhs/run_lhs_fact8.sh"
grep -q "ZZRUNENDZZ" "$PK/fwd_lhs/logs/run.log" || { mk FWDFAIL "see fwd_lhs/logs/run.log"; exit 4; }
mk FWD "$(grep ZZRUNENDZZ "$PK/fwd_lhs/logs/run.log" | tail -1)"
STAGE=bars
( export TMP=F:/c_overflow_2026-10-26/tmp TEMP=F:/c_overflow_2026-10-26/tmp R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval;$BAT" CUDA_VISIBLE_DEVICES=-1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 R8_NO_POSE_SYNC=1 PYTHONIOENCODING=utf-8
  "$PY" -X utf8 "$PK/code/lh_bars.py" > "$PK/logs/lh_bars.log" 2>&1 )
if [ -s "$PK/result/LONHEAD_VERDICT.json" ] && grep -q ZZLHVERDICTENDZZ "$PK/result/LONHEAD_VERDICT.md"; then mk BARS "$(head -1 "$PK/result/LONHEAD_VERDICT.md")"; else mk BARSFAIL "see logs/lh_bars.log"; fi
STAGE=done
