@echo off
rem PREREG_V8F_DAGGER on box 2. NOT RUN: it needs the PI's go (box 2 is the PI's machine). The chain refuses to start without AFTER_OUT (the log of the chain that holds box 2) and, before touching anything, reads PREREG_V8F_ORECOVER's read:
rem ROBUST / PARTIAL park the arm, REFUSED / UNPLACED stop it. Set AFTER_OUT to the ORC chain's log (it ended with ZZV8FMINI-DONE).
"C:\Program Files\Git\bin\bash.exe" -lc "AFTER_OUT=/c/Users/Admin/box2_ship/logs/chain_v8f_orc.out bash /c/Users/Admin/box2_ship/chain_v8f_dag.sh < /dev/null > /c/Users/Admin/box2_ship/logs/chain_v8f_dag.out 2>&1" < NUL
