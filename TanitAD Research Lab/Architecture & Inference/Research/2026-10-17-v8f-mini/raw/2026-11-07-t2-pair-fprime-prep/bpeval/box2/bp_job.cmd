@echo off
setlocal enabledelayedexpansion
call C:\lorapod_eval\env_bp.cmd
echo ===== BPEVAL %DATE% %TIME%
cd /d C:\lorapod_eval
%RUNSAFE% cmd /c C:\lorapod_eval\bp_evals.cmd
set RC=!ERRORLEVEL!
echo ===== ZZJOB-END %DATE% %TIME% rc=!RC!
