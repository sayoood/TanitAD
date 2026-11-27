@echo off
rem MM 2026-11-25: P's read-input puller (pull_p_auto.sh), started DETACHED (WMI, CreateFlags 0x08000200), stdin from NUL.
set "PATH=C:\Program Files\Git\usr\bin;C:\Program Files\Git\bin;%PATH%"
"C:\Program Files\Git\usr\bin\bash.exe" "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorerft_read/pull_p_auto.sh" > "E:\Projects\TanitAD-artifacts\scorerft_read\pull_p_stdout.log" 2>&1 < NUL
echo pullp exit %ERRORLEVEL% >> "E:\Projects\TanitAD-artifacts\scorerft_read\pull_p_stdout.log"
exit /b %ERRORLEVEL%
