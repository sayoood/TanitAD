@echo off
rem SCORER-FT arm-P-vs-C read launcher (MM 2026-11-24). Start DETACHED (WMI Win32_Process.Create, CreateFlags 0x08000200), as run_sft_read_s1.cmd.
rem Git\usr\bin\bash.exe, stdin from NUL. The script waits for P's seam chain (markers_sftp\chain.DONE) and for read_p\pod\PULLED.txt itself; see sft_read_p.sh.
rem Rehearsal: set READP_REHEARSE=1 before calling this file (dry run on Stage 1 stand-ins). A later tip commit goes in scorerft_read\readp_tip.txt.
set "PATH=C:\Program Files\Git\usr\bin;C:\Program Files\Git\bin;%PATH%"
"C:\Program Files\Git\usr\bin\bash.exe" "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorerft_read/sft_read_p.sh" > "E:\Projects\TanitAD-artifacts\scorerft_read\readp_stdout.log" 2>&1 < NUL
echo readp exit %ERRORLEVEL% >> "E:\Projects\TanitAD-artifacts\scorerft_read\readp_stdout.log"
exit /b %ERRORLEVEL%
