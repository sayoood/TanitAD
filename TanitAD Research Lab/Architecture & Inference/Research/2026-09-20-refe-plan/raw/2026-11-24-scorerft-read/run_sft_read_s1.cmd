@echo off
rem SCORER-FT Stage 1 read launcher (MM 2026-11-23). Start DETACHED (WMI Win32_Process.Create, CreateFlags 0x08000200), as run_sft_seam.cmd.
rem Git\usr\bin\bash.exe, stdin from NUL. The script waits for T's seam chain itself; see sft_read_s1.sh.
set "PATH=C:\Program Files\Git\usr\bin;C:\Program Files\Git\bin;%PATH%"
"C:\Program Files\Git\usr\bin\bash.exe" "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorerft_read/sft_read_s1.sh" > "E:\Projects\TanitAD-artifacts\scorerft_read\read1_stdout.log" 2>&1 < NUL
echo read1 exit %ERRORLEVEL% >> "E:\Projects\TanitAD-artifacts\scorerft_read\read1_stdout.log"
exit /b %ERRORLEVEL%
