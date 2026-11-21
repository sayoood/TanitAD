@echo off
rem SMOOTH-T1 thread-noise replicate launcher (WMI Win32_Process.Create).
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagland/smooth_t1/smooth_t1_threadrep_chain.sh < /dev/null > /e/Projects/TanitAD-artifacts/dagland/smooth_t1/logs/threadrep.launch.out 2>&1" < NUL
