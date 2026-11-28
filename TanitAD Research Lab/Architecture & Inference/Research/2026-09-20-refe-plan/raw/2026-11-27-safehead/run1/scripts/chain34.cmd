@echo off
rem MM 2026-11-27: L2 steps 3-4 (read + families), detached; waits on F:/safehead_cache/markers/navtest.DONE. Rewritten: the agent's copy lost "\b" of Git\bin to printf.
"C:\Program Files\Git\bin\bash.exe" E:/Projects/TanitAD-artifacts/safehead/run1/scripts/chain34.sh > E:\Projects\TanitAD-artifacts\safehead\run1\chain34_stdout.log 2>&1 < NUL
exit /b %ERRORLEVEL%
