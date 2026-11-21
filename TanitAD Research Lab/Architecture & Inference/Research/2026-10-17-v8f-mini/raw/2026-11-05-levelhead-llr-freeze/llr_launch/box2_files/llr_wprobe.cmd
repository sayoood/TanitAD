@echo off
rem LEVELHEAD W_LEAD PROBE (step 0 of arm LLR): LLR_0's argv with --selfcheck-only --w-lead 1, the frozen wiring and its own --out (M2b_s0_wleadprobe_LLR). It trains nothing; it writes selfcheck.json, from which the pull reads
rem lead_head_probe.w_lead_for_10pct_share. Markers WLPROBE.done / WLPROBE.FAILED / WLPROBE.STOPPED in D:\tanitad\runs\v8f_mini_full\llr.
rem Launch: powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\llr\llr_wprobe.cmd -Log D:\tanitad\logs\llr_wprobe.log
rem %RUNSAFE% (env.cmd) = start /b /wait /belownormal /affinity FF; env.cmd also sets the 4 threads, AVX2 and TEMP (SAFE MODE, box-2 memory note).
setlocal enabledelayedexpansion
call D:\tanitad\env.cmd
set "RUNSM=D:\tanitad\runs\v8f_mini_full"
set "RCD=%RUNSM%\llr"
set "CHK=D:\tanitad\llr\llr_check.py"
set "RUNCMD=D:\tanitad\llr\llr_run.cmd"
echo ===== LLR WPROBE START %DATE% %TIME%
if not exist "%RCD%" mkdir "%RCD%"
if not exist "%RCD%\logs" mkdir "%RCD%\logs"
rem ONE job at a time (box 2 is the PI's machine and has blue-screened under load): refuse to start while ANY python process exists
set "NPY=0"
for /f %%N in ('tasklist /fi "imagename eq python.exe" /fo csv /nh ^| find /c "python.exe"') do set "NPY=%%N"
if not "!NPY!"=="0" (echo ZZLLR-STOP a python process is already running on box 2 ^(!NPY!^) & "%PY%" "%CHK%" wprobe-stopped python process present & echo stop> D:\tanitad\logs\telemetry.stop & exit /b 3)
"%PY%" "%CHK%" preflight wprobe
set "RC=!ERRORLEVEL!"
if !RC! NEQ 0 (echo ZZLLR-STOP preflight rc !RC! & "%PY%" "%CHK%" wprobe-stopped preflight failed rc !RC! & echo stop> D:\tanitad\logs\telemetry.stop & exit /b 3)
"%PY%" "%CHK%" pin WLPROBE_0
set "RC=!ERRORLEVEL!"
if !RC! NEQ 0 ( "%PY%" "%CHK%" fail WLPROBE_0 pin refused rc !RC! & echo stop> D:\tanitad\logs\telemetry.stop & exit /b 3 )
echo [%DATE% %TIME%] GATE WLPROBE_0
%RUNSAFE% cmd /c %RUNCMD% gate WLPROBE_0
set "RC=!ERRORLEVEL!"
"%PY%" "%CHK%" gate WLPROBE_0 !RC!
set "GRC=!ERRORLEVEL!"
echo ===== LLR WPROBE END gate-verdict-rc=!GRC! trainer-rc=!RC! %DATE% %TIME%
echo stop> D:\tanitad\logs\telemetry.stop
exit /b !GRC!
