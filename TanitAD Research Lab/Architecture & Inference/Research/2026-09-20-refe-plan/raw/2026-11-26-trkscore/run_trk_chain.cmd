@echo off
rem TRK-SCORE full navtest chain (trk_chain.sh), started DETACHED via WMI (CreateFlags 0x08000200), stdin from NUL.
set "PATH=C:\Program Files\Git\usr\bin;C:\Program Files\Git\bin;%PATH%"
"C:\Program Files\Git\usr\bin\bash.exe" "E:/Projects/TanitAD-artifacts/trkscore/stage/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-11-26-trkscore/trk_chain.sh" > "E:\Projects\TanitAD-artifacts\trkscore\work\chain_stdout.log" 2>&1 < NUL
exit /b %ERRORLEVEL%
