@echo off
call D:\tanitad\jobs\rj5_env.cmd
echo ZZ fetch start %TIME%
"%PY%" "%PK%\code\fetch_navtrain_dbs.py" --out D:\tanitad\rj5\db --logs-file D:\tanitad\rj5\bank\smoke_logs.txt --held-root D:\nonexistent --jobs 8
echo ZZ fetch rc=%ERRORLEVEL% %TIME%
