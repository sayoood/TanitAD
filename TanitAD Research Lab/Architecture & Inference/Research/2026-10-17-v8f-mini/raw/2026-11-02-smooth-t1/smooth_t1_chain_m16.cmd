@echo off
rem SMOOTH-T1 chain launcher: start through WMI Win32_Process.Create so it survives the caller.  RESUME = launch again (finished runs are skipped).
set SMT_THREADS=16
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagland/smooth_t1/smooth_t1_chain_m16.sh < /dev/null > /e/Projects/TanitAD-artifacts/dagland/smooth_t1/logs/chain_m16.launch.out 2>&1" < NUL
