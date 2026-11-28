#!/bin/bash
# usage: run_one.sh <name>   -- guard, then one eval of X on CPU. Writes logs/eval_<name>.log ; exit 3 if the guard fails (write HANDOFF.md and end)
set -u
OUT="C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/kindx"
N=$1
G=$(powershell -NoProfile -ExecutionPolicy Bypass -File "$OUT/guard.ps1"); RC=$?
echo "GUARD $N: $G"
if [ $RC -ne 0 ]; then echo "GUARD-FAIL $N: $G" >> "$OUT/logs/guard_fail.txt"; exit 3; fi
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
echo "START $(date '+%H:%M:%S') $N"
"$PY" -I "$OUT/kindx_eval.py" "$N" > "$OUT/logs/eval_$N.log" 2>&1
RC=$?
echo "END $(date '+%H:%M:%S') $N rc=$RC"
tail -3 "$OUT/logs/eval_$N.log"
exit $RC
