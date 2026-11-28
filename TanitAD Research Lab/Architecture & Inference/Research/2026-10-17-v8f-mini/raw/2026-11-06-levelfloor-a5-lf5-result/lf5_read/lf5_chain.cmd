@echo off
rem LF5 chain launcher: start through WMI Win32_Process.Create so it survives the caller.  RESUME = launch again (finished rollouts are skipped).
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagland/motion_c/lf5_read/lf5_chain.sh < /dev/null > /e/Projects/TanitAD-artifacts/dagland/motion_c/lf5_read/logs/chain.launch.out 2>&1" < NUL
