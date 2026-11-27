@echo off
rem MM 2026-11-27: L2 controls-only read (NULL-HELDOUT case), detached; waits on F:/safehead_cache/markers/navtest.DONE.
"C:\Program Files\Git\bin\bash.exe" E:/Projects/TanitAD-artifacts/safehead/run1/scripts/chain3_nh.sh > E:\Projects\TanitAD-artifacts\safehead\run1\chain3_nh_stdout.log 2>&1 < NUL
exit /b %ERRORLEVEL%
