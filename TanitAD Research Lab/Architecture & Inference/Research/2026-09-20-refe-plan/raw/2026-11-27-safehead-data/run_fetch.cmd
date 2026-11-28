@echo off
"C:\Users\Admin\venvs\tanitad\Scripts\python.exe" "F:\safehead_cache\code\sh_run.py" --fetch-only > "F:\safehead_cache\logs\fetch_stdout.log" 2>&1 < NUL
exit /b %ERRORLEVEL%
