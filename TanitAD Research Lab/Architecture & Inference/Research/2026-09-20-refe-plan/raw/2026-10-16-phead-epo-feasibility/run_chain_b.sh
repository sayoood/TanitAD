#!/bin/bash
export PYTHONUTF8=1 TMP=C:/Users/Admin/AppData/Local/Temp TEMP=C:/Users/Admin/AppData/Local/Temp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
cd "$(dirname "$0")"
PH_THREADS=4 $PY phead_proxy_validity.py > logs/proxy_validity.log 2>&1
$PY phead_open.py --streams 0 --j-streams 0 --groups FULL PH_A_EPROW PH_B_SIX PH_C_HEAD LAD_1 LAD_2 --kinds R --regimes P T --lams 0 0.25 1 --jP 50 100 --jT 250 --threads 4 --out open_out/open_s0_R.pkl > logs/open_s0_R.log 2>&1
$PY phead_open.py --streams 1 2 --j-streams 1 2 --groups FULL PH_A_EPROW PH_B_SIX PH_C_HEAD LAD_1 LAD_2 --kinds R --regimes P --lams 0 1 --jP 50 --jT --threads 4 --out open_out/open_s12_R.pkl > logs/open_s12_R.log 2>&1
$PY phead_open.py --streams 0 --j-streams 0 --groups PH_C_HEAD PH_A_EPROW --kinds E --regimes P T --lams 0 0.25 1 --jP 50 100 --jT 250 --threads 4 --out open_out/open_s0_E.pkl > logs/open_s0_E.log 2>&1
echo ZZCHAINB_DONE > logs/chainb_done.txt
