#!/bin/bash
# read_r2.sh <out folder under actcons/> <pairs> <seeds> -- dev box. Repeats the family read and the headway read of a stage of card ADD37 with the
# second-version readers (headway_read_r2.py, q3_read_r3.py), on the rollouts that the stage waiter already pulled. One Python job at a time; each job starts
# only with 5.5 GB of commit memory free (waits at most 30 min). Starts nothing on the pod. Deletes nothing. Last line: ZZ READ-R2 <folder> DONE.
set -u
H=/f/TanitAD-scratch/v8proof; W=F:/TanitAD-scratch/v8proof
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
OUT=$1; PAIRS=$2; SEEDS=$3
ROOTS="${ROOTS:-$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull}"
R=$H/actcons/$OUT; RW=$W/actcons/$OUT
mkdir -p "$R"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
freemb() { powershell.exe -NoProfile -Command "\$o = Get-CimInstance Win32_OperatingSystem; [int](\$o.FreeVirtualMemory / 1024)" 2>/dev/null | tr -d '\r'; }
gate() { local n=0 f; while true; do f=$(freemb); [ "${f:-0}" -ge 5632 ] && { echo "[$(ts)] commit free $f MB"; return 0; }; n=$((n+1)); [ "$n" -gt 30 ] && { echo "[$(ts)] ZZ READ-R2 $OUT STOP: under 5.5 GB commit free for 30 min"; exit 8; }; sleep 60; done; }
gate
( cd $H/headway && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds "$SEEDS" --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$RW/hw_r2" > "$R/hw_r2.log" 2>&1 ); echo "[$(ts)] headway r2 rc $? : $(tail -n 1 "$R/hw_r2.log" | cut -c1-240)"
gate
( cd $H/spatial/q3 && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds "$SEEDS" --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$RW/q3_r3" > "$R/q3_r3.log" 2>&1 ); echo "[$(ts)] families r2 rc $? : $(tail -n 1 "$R/q3_r3.log" | cut -c1-240)"
echo "ZZ READ-R2 $OUT DONE $(ts)"
