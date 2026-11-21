@echo off
rem FEAS2-T1 chain launcher: start through WMI Win32_Process.Create so it survives the caller.  RESUME = launch again (finished runs are skipped).
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagland/feas2_t1/feas2_t1_chain.sh < /dev/null > /e/Projects/TanitAD-artifacts/dagland/feas2_t1/logs/chain.launch.out 2>&1" < NUL
