@echo off
rem MM 2026-11-27: L2 cache chain relaunch with RAM floor 2000 MB (SH_NEED_MB) and 2 label workers; commit floor stays 8 GB. Resumable.
set "SH_NEED_MB=2000"
"C:\Users\Admin\venvs\tanitad\Scripts\python.exe" "F:\safehead_cache\code\sh_run.py" --workers 2 --passive-fetch >> "F:\safehead_cache\logs\chain_stdout.log" 2>&1 < NUL
exit /b %ERRORLEVEL%
