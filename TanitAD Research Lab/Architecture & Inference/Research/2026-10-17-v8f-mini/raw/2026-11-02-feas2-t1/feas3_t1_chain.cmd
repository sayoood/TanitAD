@echo off
rem FEAS3 chain launcher: start through WMI Win32_Process.Create so it survives the caller.  RESUME = launch again (finished runs are skipped).
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagland/feas2_t1/feas3_t1_chain.sh < /dev/null > /e/Projects/TanitAD-artifacts/dagland/feas2_t1/logs/chain_feas3.launch.out 2>&1" < NUL
