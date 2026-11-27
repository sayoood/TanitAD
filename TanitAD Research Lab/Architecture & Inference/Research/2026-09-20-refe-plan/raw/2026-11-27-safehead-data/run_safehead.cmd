@echo off
"C:\Users\Admin\venvs\tanitad\Scripts\python.exe" "F:\safehead_cache\code\sh_run.py" --workers 3 --passive-fetch > "F:\safehead_cache\logs\chain_stdout.log" 2>&1 < NUL
exit /b %ERRORLEVEL%
