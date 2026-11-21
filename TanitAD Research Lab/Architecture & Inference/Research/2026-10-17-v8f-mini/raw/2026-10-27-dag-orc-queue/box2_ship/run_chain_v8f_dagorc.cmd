@echo off
rem PREREG_V8F_DAG_ORC on box 2. NOT RUN: needs the PI's go. AFTER_OUT = the DAG-OACC r2 chain's log (it holds box 2 now, behind FLAT-DAG); a predecessor STOP of its own with BOX2_BUSY released does not stop this chain, a ZZGUARD-STOP or a held BOX2_BUSY does.
"C:\Program Files\Git\bin\bash.exe" -lc "AFTER_OUT=/c/Users/Admin/box2_ship/logs/chain_v8f_dagoacc_r2.out bash /c/Users/Admin/box2_ship/chain_v8f_dagorc.sh < /dev/null > /c/Users/Admin/box2_ship/logs/chain_v8f_dagorc.out 2>&1" < NUL
