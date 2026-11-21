@echo off
rem HISTSWAP chain launcher: start through WMI Win32_Process.Create so it survives the caller.  RESUME = launch again (finished runs are skipped).  The chain itself waits for the FEAS3 marker.
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagland/histswap/histswap_chain.sh < /dev/null > /e/Projects/TanitAD-artifacts/dagland/histswap/logs/chain.launch.out 2>&1" < NUL
