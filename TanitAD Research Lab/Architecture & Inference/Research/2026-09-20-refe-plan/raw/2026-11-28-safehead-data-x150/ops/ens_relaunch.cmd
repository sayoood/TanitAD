@echo off
rem MM 2026-11-28: ENS relaunch supervisor (ens_relaunch.sh), started DETACHED via WMI, stdin from NUL.
"C:\Program Files\Git\usr\bin\bash.exe" "E:/Projects/TanitAD-artifacts/mm/ens_relaunch.sh" > "E:\Projects\TanitAD-artifacts\mm\ens_relaunch_stdout.log" 2>&1 < NUL
exit /b %ERRORLEVEL%
