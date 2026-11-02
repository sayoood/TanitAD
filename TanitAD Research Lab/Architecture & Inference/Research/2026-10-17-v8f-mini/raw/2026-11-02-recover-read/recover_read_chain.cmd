@echo off
rem RECOVER-READ chain launcher: start through WMI Win32_Process.Create so it survives the caller.  RESUME = launch again (finished rollouts are skipped).  The chain itself waits (6 h max) for the pulled box-2 runs; start it after pull_recft.sh.
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagland/recover_read/recover_read_chain.sh < /dev/null > /e/Projects/TanitAD-artifacts/dagland/recover_read/logs/chain.launch.out 2>&1" < NUL
