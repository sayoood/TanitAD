#!/bin/bash
export PYTHONUTF8=1 TMP=C:/Users/Admin/AppData/Local/Temp TEMP=C:/Users/Admin/AppData/Local/Temp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
cd "$(dirname "$0")"
run_one() {
  args="$1"; id=$(echo "$args" | tr -d ' -' | cut -c1-60)
  $PY phead_closed.py $args --regime P --out-dir closed_out --threads 6 > "logs/closed_${id}.log" 2>&1
}
export -f run_one; export PY
cat logs/queue4.txt | xargs -P 3 -I{} bash -c 'run_one "{}"'
echo ZZQUEUE4_DONE > logs/queue4_done.txt
