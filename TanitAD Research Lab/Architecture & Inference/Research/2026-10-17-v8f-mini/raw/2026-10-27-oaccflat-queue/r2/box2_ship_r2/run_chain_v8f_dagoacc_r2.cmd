@echo off
rem PREREG_V8F_DAG_OACC r2 (the re-run after the first attempt stopped at the gate asserter's read_o control) on box 2. NOT RUN: needs the PI's go. AFTER_OUT = the FLAT-DAG chain's log (FLAT-DAG holds box 2 now); a predecessor STOP of its own with BOX2_BUSY released does not stop this chain, a ZZGUARD-STOP or a held BOX2_BUSY does.
"C:\Program Files\Git\bin\bash.exe" -lc "AFTER_OUT=/c/Users/Admin/box2_ship/logs/chain_v8f_flatdag.out bash /c/Users/Admin/box2_ship/chain_v8f_dagoacc_r2.sh < /dev/null > /c/Users/Admin/box2_ship/logs/chain_v8f_dagoacc_r2.out 2>&1" < NUL
