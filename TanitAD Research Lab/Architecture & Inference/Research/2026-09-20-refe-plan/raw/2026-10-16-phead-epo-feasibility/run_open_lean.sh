#!/bin/bash
# lean open-loop J evaluation (the first, fuller chain was stopped at 16:55 for CPU: the closed-loop head arms decide, the open loop only bounds data-order sensitivity)
export PYTHONUTF8=1 TMP=C:/Users/Admin/AppData/Local/Temp TEMP=C:/Users/Admin/AppData/Local/Temp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
cd "$(dirname "$0")"
G="FULL PH_A_EPROW PH_B_SIX PH_C_HEAD LAD_1 LAD_2"
$PY phead_open.py --streams 0 1 2 --j-streams 0 1 2 --groups $G --kinds R --regimes P --lams 0 1 --jP 50 --jT --threads 4 --out open_out/open_P50.pkl > logs/open_P50.log 2>&1
$PY phead_open.py --streams 0 --j-streams 0 --groups $G --kinds R --regimes T --lams 0 1 --jP --jT 250 --threads 4 --out open_out/open_T250.pkl > logs/open_T250.log 2>&1
echo ZZOPENLEAN_DONE > logs/openlean_done.txt
