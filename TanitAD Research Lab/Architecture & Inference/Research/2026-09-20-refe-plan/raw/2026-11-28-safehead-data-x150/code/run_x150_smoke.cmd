@echo off
rem x150 smoke: 1 log end-to-end (resumable; the DONE marker carries into the full run). Root F:\safehead_cache\x150 only.
set "SH_WORK=F:/safehead_cache/x150"
set "SH_NEED_MB=2000"
set "SH_COMMIT_MB=6000"
"C:\Users\Admin\venvs\tanitad\Scripts\python.exe" "F:\safehead_cache\code_x150\sh_run.py" --workers 2 --only 2021.06.14.13.27.42_veh-35_01342_01461 >> "F:\safehead_cache\x150\logs\smoke_stdout.log" 2>&1 < NUL
exit /b %ERRORLEVEL%
