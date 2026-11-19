#!/bin/bash
# read_imag_r2.sh <out folder under actcons/> <seed1> <seed2> "<arms>" "<a:b:dir-of-b-read> ..." -- dev box. Repeats the imagination read and its pair reads of a
# stage of card ADD37 with the second-version reader (imaghead/imag_read_r2.py) on the dumps that the stage waiter already pulled. Outputs go to
# actcons/<folder>/imag_r2/. One Python job at a time; each starts only with 5.5 GB of commit memory free. Starts nothing on the pod. Deletes nothing.
# Last line: ZZ READ-IMAG-R2 <folder> DONE.
set -u
H=/f/TanitAD-scratch/v8proof; W=F:/TanitAD-scratch/v8proof
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
OUT=$1; S1=$2; S2=$3; ARMS=$4; PAIRS=$5
R=$H/actcons/$OUT/imag_r2; RW=$W/actcons/$OUT/imag_r2; DW=$W/imaghead/dump
mkdir -p "$R"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
freemb() { powershell.exe -NoProfile -Command "\$o = Get-CimInstance Win32_OperatingSystem; [int](\$o.FreeVirtualMemory / 1024)" 2>/dev/null | tr -d '\r'; }
gate() { local n=0 f; while true; do f=$(freemb); [ "${f:-0}" -ge 5632 ] && return 0; n=$((n+1)); [ "$n" -gt 30 ] && { echo "[$(ts)] ZZ READ-IMAG-R2 $OUT STOP: under 5.5 GB commit free for 30 min"; exit 8; }; sleep 60; done; }
for a in $ARMS; do
  [ -f "$H/imaghead/dump/DUMP_${a}_s$S1.npz" ] && [ -f "$H/imaghead/dump/DUMP_${a}_s$S2.npz" ] || { echo "[$(ts)] $a: a dump is missing, no read"; continue; }
  gate
  ( cd $H/imaghead && "$PY" imag_read_r2.py read --dump "$DW/DUMP_${a}_s$S1.npz" "$DW/DUMP_${a}_s$S2.npz" --arm "$a" --seeds $S1,$S2 --out-dir "$RW/$a" > "$R/$a.log" 2>&1 )
  echo "[$(ts)] read $a rc $? : $(tail -n 1 "$R/$a.log" | cut -c1-260)"
done
for p in $PAIRS; do
  a=${p%%:*}; rest=${p#*:}; b=${rest%%:*}; bd=${rest#*:}
  [ -f "$R/$a/RESULT_IMAG_$a.json" ] && [ -f "$bd/RESULT_IMAG_$b.json" ] || { echo "[$(ts)] pair $a minus $b: a read is missing"; continue; }
  gate
  ( cd $H/imaghead && "$PY" imag_read_r2.py pair --a-dir "$RW/$a" --b-dir "$bd" --a-arm "$a" --b-arm "$b" --out-dir "$RW/PAIR_${a}_$b" > "$R/PAIR_${a}_$b.log" 2>&1 )
  echo "[$(ts)] pair $a minus $b rc $? : $(tail -n 1 "$R/PAIR_${a}_$b.log" | cut -c1-300)"
done
echo "ZZ READ-IMAG-R2 $OUT DONE $(ts)"
