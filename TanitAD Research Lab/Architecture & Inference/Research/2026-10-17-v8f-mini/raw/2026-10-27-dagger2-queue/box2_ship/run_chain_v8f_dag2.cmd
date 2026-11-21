@echo off
rem PREREG_V8F_DAGGER2 on box 2. NOT RUN: it needs the PI's go (box 2 is the PI's machine). The chain refuses to start without AFTER_OUT (the log of the chain that holds box 2: the DAG chain, which ended with ZZV8FMINI-DONE)
rem and, before touching anything, requires DAG_READ.json to be the pinned file (sha256 3393fdd5...) with outcome ROBUST.
"C:\Program Files\Git\bin\bash.exe" -lc "AFTER_OUT=/c/Users/Admin/box2_ship/logs/chain_v8f_dag.out bash /c/Users/Admin/box2_ship/chain_v8f_dag2.sh < /dev/null > /c/Users/Admin/box2_ship/logs/chain_v8f_dag2.out 2>&1" < NUL
