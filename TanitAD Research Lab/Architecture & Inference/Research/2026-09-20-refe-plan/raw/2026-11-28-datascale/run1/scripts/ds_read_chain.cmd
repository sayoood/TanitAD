@echo off
rem MM 2026-11-28: DATA-SCALE decision read chain (ds_read_chain.sh), started DETACHED via WMI, stdin from NUL. Waits for run1\STEP2.DONE.
"C:\Program Files\Git\usr\bin\bash.exe" "E:/Projects/TanitAD-artifacts/datascale/run1/scripts/ds_read_chain.sh" > "E:\Projects\TanitAD-artifacts\datascale\run1\dsread_stdout.log" 2>&1 < NUL
exit /b %ERRORLEVEL%
