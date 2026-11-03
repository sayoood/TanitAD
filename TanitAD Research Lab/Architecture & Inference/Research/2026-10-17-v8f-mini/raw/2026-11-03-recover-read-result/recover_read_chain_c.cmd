@echo off
rem RECOVER-READ chain launcher with the write root on C: (READER_FIX_RREAD2: F: full on 2026-11-03). Same chain; RESUME = launch again.
"C:\Program Files\Git\bin\bash.exe" -lc "RREAD_W=/c/Users/Admin/rr_scratch bash /e/Projects/TanitAD-artifacts/dagland/recover_read/recover_read_chain.sh < /dev/null > /e/Projects/TanitAD-artifacts/dagland/recover_read/logs/chain.launch.out 2>&1" < NUL
