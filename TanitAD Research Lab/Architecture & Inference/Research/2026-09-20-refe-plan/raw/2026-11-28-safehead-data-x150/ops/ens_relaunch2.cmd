@echo off
rem MM 2026-11-28: ENS relaunch supervisor (ens_relaunch2.sh), started DETACHED via WMI, stdin from NUL.
"C:\Program Files\Git\usr\bin\bash.exe" "E:/Projects/TanitAD-artifacts/mm/ens_relaunch2.sh" > "E:\Projects\TanitAD-artifacts\mm\ens_relaunch2_stdout.log" 2>&1 < NUL
exit /b %ERRORLEVEL%
