@echo off
rem FRAMEFREEZE chain launcher: start through WMI Win32_Process.Create so it survives the caller.  RESUME = launch again (finished runs are skipped).  The chain itself waits for the RECOVER-FT card's FROZEN_SHA.txt.
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagland/framefreeze/framefreeze_chain.sh < /dev/null > /e/Projects/TanitAD-artifacts/dagland/framefreeze/logs/chain.launch.out 2>&1" < NUL
