@echo off
rem EXEC-STANDSTILL chain launcher: start through WMI Win32_Process.Create so it survives the caller.  RESUME = launch again (finished rollouts are skipped).
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagland/motion_c/exec_standstill/code/exec_chain.sh < /dev/null > /e/Projects/TanitAD-artifacts/dagland/motion_c/exec_standstill/logs/chain.launch.out 2>&1" < NUL
