@echo off
rem GAP48 chain launcher: start it through WMI Win32_Process.Create so it survives the caller. RESUME = GAP48_RESUME=1 in the environment of a new launch (finished scenes are skipped).
"C:\Program Files\Git\bin\bash.exe" -c "bash /e/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/gap48/gap48_chain.sh < /dev/null > /e/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/gap48/logs/chain.out 2>&1" < NUL
