@echo off
rem Detached launcher for run_speedread.sh (the pull is DONE; resume = run again: finished stages are skipped by their .done files).
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/speedread_run/run_speedread.sh < /dev/null >> /f/TanitAD-scratch/speedread/run.out 2>&1" < NUL
