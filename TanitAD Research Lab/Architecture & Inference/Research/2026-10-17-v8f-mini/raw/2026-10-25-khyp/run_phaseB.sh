#!/bin/bash
# usage: run_phaseB.sh <stage>      stage = eval0 | train0 | eval1 | train1 | read     (ONE data run at a time; run them in this order)
# CPU only, 6 torch threads. Each stage is resumable (hyps shards) and idempotent.
#
# GUARD (replaced 2026-10-25 by the Phase-B brief; the old one was "refuse under 3 GB free on C:"):
#   before each run require  C: free >= 0.7 GB  AND  commit headroom (Win32_OperatingSystem.FreeVirtualMemory) >= 5 GB.
#   If not met: re-check once a minute, at most 4 min (5 checks in all); then exit 3 (caller writes HANDOFF.md and ends).
#   Overrides (tests only): KHYP_MIN_FREE_C_GB, KHYP_MIN_COMMIT_GB.
# PEAK MEMORY: each run appends its peak private commit (psutil peak_pagefile) and peak working set to logs/peak_memory.jsonl.
set -u
O="C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/khyp"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SCRIPT="$O/khyp.py"
MIN_C_GB="${KHYP_MIN_FREE_C_GB:-0.7}"
MIN_VM_GB="${KHYP_MIN_COMMIT_GB:-5}"
case "$1" in
  eval0)  ARGS=(hyps --seed 0 --split eval  --threads 6 --batch 16 --lru 2 --shard 640);;
  train0) ARGS=(hyps --seed 0 --split train --threads 6 --batch 16 --lru 2 --shard 640);;
  eval1)  ARGS=(hyps --seed 1 --split eval  --threads 6 --batch 16 --lru 2 --shard 640);;
  train1) ARGS=(hyps --seed 1 --split train --threads 6 --batch 16 --lru 2 --shard 640);;
  read)   ARGS=(read --threads 6 --n-boot 2000);;
  post)   ARGS=(); SCRIPT="$O/post_read.py";;   # post-hoc descriptive pass (added after the outcome was fixed; see post_read.py)
  *) echo "unknown stage $1"; exit 2;;
esac
mkdir -p "$O/logs"

probe() {  # prints "<C free GB> <commit headroom GB>"
  powershell.exe -NoProfile -NonInteractive -Command '$os=Get-CimInstance Win32_OperatingSystem; "{0:N3} {1:N3}" -f ((Get-PSDrive C).Free/1GB), ($os.FreeVirtualMemory/1MB)' | tr -d '\r' | tail -1
}
OK=0
for i in 1 2 3 4 5; do
  read -r CFREE VMFREE <<< "$(probe | tr ',' '.')"
  if awk -v c="$CFREE" -v v="$VMFREE" -v mc="$MIN_C_GB" -v mv="$MIN_VM_GB" 'BEGIN{exit !(c+0>=mc+0 && v+0>=mv+0)}'; then OK=1; break; fi
  echo "GUARD wait $i/5 $(date '+%H:%M:%S'): C free ${CFREE} GB (need ${MIN_C_GB}), commit headroom ${VMFREE} GB (need ${MIN_VM_GB})"
  [ "$i" -lt 5 ] && sleep 60
done
if [ "$OK" -ne 1 ]; then echo "REFUSED $1: guard not met after 4 min (C free ${CFREE} GB, commit headroom ${VMFREE} GB)"; exit 3; fi

echo "START $(date '+%H:%M:%S') $1 C_free_gb=$CFREE commit_headroom_gb=$VMFREE"
STAGE="$1" PEAKLOG="$O/logs/peak_memory.jsonl" KHYP_PY="$SCRIPT" "$PY" -I -c '
import atexit, json, os, runpy, sys, time, psutil
T0 = time.time()
def _peak():
    m = psutil.Process().memory_info()
    with open(os.environ["PEAKLOG"], "a") as fh:
        fh.write(json.dumps({"stage": os.environ["STAGE"], "peak_private_commit_gb": round(m.peak_pagefile / 2**30, 3), "peak_working_set_gb": round(m.peak_wset / 2**30, 3), "seconds": round(time.time() - T0, 1)}) + "\n")
atexit.register(_peak)
sys.argv = [os.environ["KHYP_PY"]] + sys.argv[1:]
runpy.run_path(os.environ["KHYP_PY"], run_name="__main__")
' "${ARGS[@]}" > "$O/logs/$1.log" 2>&1
RC=$?
echo "END $(date '+%H:%M:%S') $1 rc=$RC"
exit $RC
