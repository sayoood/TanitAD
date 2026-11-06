@echo off
rem EXEC-STANDSTILL post-chain launcher (WMI Win32_Process.Create): waits for the rollout chain, then extract, G-ID full grid, read, tables.
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagland/motion_c/exec_standstill/code/post_chain.sh < /dev/null > /e/Projects/TanitAD-artifacts/dagland/motion_c/exec_standstill/logs/post.launch.out 2>&1" < NUL
