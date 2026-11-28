@echo off
rem MM 2026-11-28: EP-HEAD navtest decision read + families + confirm, detached.
"C:\Program Files\Git\bin\bash.exe" E:/Projects/TanitAD-artifacts/l2next/run1/scripts/eph_read_chain.sh > E:\Projects\TanitAD-artifacts\l2next\run1\eph_read_chain_stdout.log 2>&1 < NUL
exit /b %ERRORLEVEL%
