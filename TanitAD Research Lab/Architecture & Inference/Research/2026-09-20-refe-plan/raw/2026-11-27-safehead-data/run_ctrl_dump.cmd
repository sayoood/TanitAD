@echo off
"C:\Users\Admin\venvs\tanitad\Scripts\python.exe" "F:\safehead_cache\code\ctrl_dump_launch.py" > "F:\safehead_cache\logs\ctrl_dump_stdout.log" 2>&1 < NUL
exit /b %ERRORLEVEL%
