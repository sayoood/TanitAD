@echo off
rem ENS-SCORE chain (ens_chain.sh), started DETACHED via WMI (CreateFlags 0x08000200), stdin from NUL. Gated inside on the TRK chain.DONE + GPU lock + memory.
set "PATH=C:\Program Files\Git\usr\bin;C:\Program Files\Git\bin;%PATH%"
"C:\Program Files\Git\usr\bin\bash.exe" "E:/Projects/TanitAD-artifacts/ensscore/stage/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-11-27-ensscore/ens_chain.sh" > "E:\Projects\TanitAD-artifacts\ensscore\work\chain_stdout.log" 2>&1 < NUL
exit /b %ERRORLEVEL%
