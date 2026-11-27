@echo off
rem NAVTEST feature dump chain (refe-safehead data agent 2026-11-27). Detached via WMI CreateFlags 0x08000200.
set "SH_NEED_MB=2000"
"C:\Users\Admin\venvs\tanitad\Scripts\python.exe" "F:\safehead_cache\code\nt_chain.py" >> "F:\safehead_cache\navtest\chain_stdout.log" 2>&1 < NUL
exit /b %ERRORLEVEL%
