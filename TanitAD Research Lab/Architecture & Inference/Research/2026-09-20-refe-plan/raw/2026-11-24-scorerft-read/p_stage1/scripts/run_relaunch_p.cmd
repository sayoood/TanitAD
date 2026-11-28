@echo off
rem MM 2026-11-26: memory-gated relaunch of P's seam + read waiter (relaunch_p_memgate.sh), started DETACHED (WMI), stdin from NUL.
set "PATH=C:\Program Files\Git\usr\bin;C:\Program Files\Git\bin;%PATH%"
"C:\Program Files\Git\usr\bin\bash.exe" "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorerft_read/relaunch_p_memgate.sh" > "E:\Projects\TanitAD-artifacts\scorerft_read\relaunch_p_stdout.log" 2>&1 < NUL
exit /b %ERRORLEVEL%
