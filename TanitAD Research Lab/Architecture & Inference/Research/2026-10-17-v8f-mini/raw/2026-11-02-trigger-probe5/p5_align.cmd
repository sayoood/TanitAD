@echo off
setlocal enabledelayedexpansion
call D:\tanitad\env.cmd
echo ===== P5 align %DATE% %TIME%
set TRY=0
:again
set /a TRY+=1
%RUNSAFE% D:\tanitad\venv\Scripts\python.exe -u D:\tanitad\jobs\probe5\encode_crops.py --mode align
set RC=!ERRORLEVEL!
echo ===== ATTEMPT !TRY! rc=!RC! %TIME%
if !RC! NEQ 0 if !TRY! LSS 3 (ping -n 31 127.0.0.1 >nul & goto again)
echo ===== ZZJOB-END %DATE% %TIME% rc=!RC!
