#!/bin/bash
# queue 3 (2026-10-16 16:30): replaces the tail of queue 2 with 3-way parallelism once the LAD_2 pair finishes.  PH_C lam 1/0 (queue 2) are already running.
export PYTHONUTF8=1 TMP=C:/Users/Admin/AppData/Local/Temp TEMP=C:/Users/Admin/AppData/Local/Temp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
cd "$(dirname "$0")"
cat > logs/queue3.txt <<'EOQ'
--group PH_B_SIX --kind R --lam 1.0
--group PH_B_SIX --kind R --lam 0.0
--group PH_A_EPROW --kind R --lam 1.0
--group PH_A_EPROW --kind R --lam 0.0
--group EMPTY --kind R --lam 1.0 --K 3 --snaps 3 --tag _b3
--group EMPTY --kind R --lam 0.0 --K 3 --snaps 3 --tag _b3
--group PH_C_HEAD --kind R --lam 1.0 --lr-mult 4
--group PH_C_HEAD --kind R --lam 0.0 --lr-mult 4
--group PH_C_HEAD --kind R --lam 1.0 --lr-mult 16
--group PH_C_HEAD --kind R --lam 0.0 --lr-mult 16
--group LAD_1 --kind R --lam 1.0
--group LAD_1 --kind R --lam 0.0
--group PH_A_EPROW --kind R --lam 1.0 --lr-mult 4
--group PH_A_EPROW --kind R --lam 0.0 --lr-mult 4
--group PH_A_EPROW --kind R --lam 1.0 --lr-mult 16
--group PH_A_EPROW --kind R --lam 0.0 --lr-mult 16
--group PH_C_HEAD --kind E --lam 1.0
EOQ
run_one() {
  args="$1"; id=$(echo "$args" | tr -d ' -' | cut -c1-60)
  $PY phead_closed.py $args --regime P --out-dir closed_out --threads 6 > "logs/closed_${id}.log" 2>&1
}
export -f run_one; export PY
cat logs/queue3.txt | xargs -P 3 -I{} bash -c 'run_one "{}"'
echo ZZQUEUE3_DONE > logs/queue3_done.txt
