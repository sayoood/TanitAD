@echo off
rem CLR read chain launcher: start through WMI Win32_Process.Create so it survives the caller.  RESUME = launch again (finished runs are skipped).
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagland/motion_c/clr_read/clr_read_chain.sh < /dev/null > /e/Projects/TanitAD-artifacts/dagland/motion_c/clr_read/logs/chain.launch.out 2>&1" < NUL
