#!/bin/bash
# runs the closed-loop job list, 3 at a time, 7 threads each.  Heavy jobs first.  Output: closed_out/*.pkl, logs/*.log
export PYTHONUTF8=1 TMP=C:/Users/Admin/AppData/Local/Temp TEMP=C:/Users/Admin/AppData/Local/Temp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
cd "$(dirname "$0")"
cat > logs/queue.txt <<'EOQ'
--group FULL --kind R --lam 1.0 --K 25 --snaps 25 --identity-vs-banked --tag _ident
--group LAD_2 --kind R --lam 1.0
--group LAD_2 --kind R --lam 0.0
--group LAD_1 --kind R --lam 1.0
--group LAD_1 --kind R --lam 0.0
--group PH_C_HEAD --kind E --lam 1.0
--group PH_C_HEAD --kind R --lam 1.0
--group PH_C_HEAD --kind R --lam 0.0
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
--group PH_A_EPROW --kind R --lam 1.0 --lr-mult 4
--group PH_A_EPROW --kind R --lam 0.0 --lr-mult 4
--group PH_A_EPROW --kind R --lam 1.0 --lr-mult 16
--group PH_A_EPROW --kind R --lam 0.0 --lr-mult 16
EOQ
run_one() {
  args="$1"; id=$(echo "$args" | tr -d ' -' | cut -c1-60)
  $PY phead_closed.py $args --regime P --out-dir closed_out --threads 7 > "logs/closed_${id}.log" 2>&1
}
export -f run_one; export PY
cat logs/queue.txt | xargs -P 3 -I{} bash -c 'run_one "{}"'
echo ZZQUEUE_DONE > logs/queue_done.txt
