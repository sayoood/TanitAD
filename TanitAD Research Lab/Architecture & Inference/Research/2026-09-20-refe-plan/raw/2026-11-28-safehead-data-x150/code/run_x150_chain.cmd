@echo off
rem x150 chain: +150 logs on a SEPARATE root; never writes F:\safehead_cache\{out,markers,logs,plan,navtest,ckpt,code}. Resumable (per-log DONE markers).
set "SH_WORK=F:/safehead_cache/x150"
set "SH_NEED_MB=2000"
set "SH_COMMIT_MB=6000"
set "SH_F_FLOOR_GB=5"
"C:\Users\Admin\venvs\tanitad\Scripts\python.exe" "F:\safehead_cache\code_x150\sh_run.py" --workers 2 >> "F:\safehead_cache\x150\logs\chain_x150_stdout.log" 2>&1 < NUL
exit /b %ERRORLEVEL%
