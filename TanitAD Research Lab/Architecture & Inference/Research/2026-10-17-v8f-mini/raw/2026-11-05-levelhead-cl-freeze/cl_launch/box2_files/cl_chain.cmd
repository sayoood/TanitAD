@echo off
rem LEVELHEAD arm CL chain (PREREG_LEVELHEAD_CL.md section 2): CL_0 then CL_1 = C0's argv + --level-head on --w-level W (the frozen W, FROZEN_W.txt), 3,000 steps each. Needs the PI's box-2 go (the launcher refuses without CL_PI_GO).
rem Per run: argv audit + pin -> in-process launch gate (the trainer's --selfcheck-only, asserted by cl_check.py) -> training -> post checks -> marker RUN_<tag>.done / RUN_<tag>.FAILED, all in D:\tanitad\runs\v8f_mini_full\cl.
rem FAIL CLOSED, NEVER STALL THE GPU: a refusal / failed gate / failed run is marked FAILED and the chain proceeds to the NEXT run. Only a SAFETY stop (guard temperature stop, disk floor) halts the chain (CHAIN_CL.STOPPED).
rem Launch: powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\cl\cl_chain.cmd -Log D:\tanitad\logs\cl_chain.log
rem %RUNSAFE% (env.cmd) = start /b /wait /belownormal /affinity FF; env.cmd also sets the 4 threads, AVX2 and TEMP (SAFE MODE, box-2 memory note).
setlocal enabledelayedexpansion
call D:\tanitad\env.cmd
set "RUNSM=D:\tanitad\runs\v8f_mini_full"
set "RCD=%RUNSM%\cl"
set "CHK=D:\tanitad\cl\cl_check.py"
set "RUNCMD=D:\tanitad\cl\cl_run.cmd"
echo ===== CL CHAIN START %DATE% %TIME%
if not exist "%RCD%" mkdir "%RCD%"
if not exist "%RCD%\logs" mkdir "%RCD%\logs"
rem ONE job at a time (box 2 is the PI's machine and has blue-screened under load): refuse to start while ANY python process exists
set "NPY=0"
for /f %%N in ('tasklist /fi "imagename eq python.exe" /fo csv /nh ^| find /c "python.exe"') do set "NPY=%%N"
if not "!NPY!"=="0" (echo ZZCL-STOP a python process is already running on box 2 ^(!NPY!^) & "%PY%" "%CHK%" stopped python process present & exit /b 3)
"%PY%" "%CHK%" preflight chain
set "RC=!ERRORLEVEL!"
if !RC! NEQ 0 (echo ZZCL-STOP preflight rc !RC! & "%PY%" "%CHK%" stopped preflight failed rc !RC! & exit /b 3)
for %%T in (CL_0 CL_1) do (
  call :one %%T
  if !ERRORLEVEL! EQU 9 goto :halt
)
"%PY%" "%CHK%" finish
echo ===== CL CHAIN END %DATE% %TIME%
echo stop> D:\tanitad\logs\telemetry.stop
exit /b 0

:halt
echo ===== CL CHAIN HALTED %DATE% %TIME%
"%PY%" "%CHK%" stopped safety stop, see the chain log
echo stop> D:\tanitad\logs\telemetry.stop
exit /b 9

:one
set "TAG=%~1"
if exist "%RCD%\RUN_%TAG%.done" (echo [skip] %TAG% already done & exit /b 0)
if exist "%RCD%\GUARD_STOP.txt" (echo ZZCL-STOP guard stop marker present & exit /b 9)
"%PY%" "%CHK%" preflight run %TAG%
set "RC=!ERRORLEVEL!"
if !RC! NEQ 0 exit /b 9
"%PY%" "%CHK%" pin %TAG%
set "RC=!ERRORLEVEL!"
if !RC! NEQ 0 ( "%PY%" "%CHK%" fail %TAG% pin refused rc !RC! & exit /b 0 )
"%PY%" "%CHK%" gate-cached %TAG%
set "RC=!ERRORLEVEL!"
if !RC! EQU 0 ( echo [%DATE% %TIME%] GATE %TAG% PASS ^(cached: the gate token is bound to this commit, argv, selfcheck and shipped bytes^) & goto :traintime )
echo [%DATE% %TIME%] GATE %TAG%
%RUNSAFE% cmd /c %RUNCMD% gate %TAG%
set "RC=!ERRORLEVEL!"
"%PY%" "%CHK%" gate %TAG% !RC!
set "GRC=!ERRORLEVEL!"
if !GRC! NEQ 0 ( echo [FAIL] gate %TAG% - see %RCD%\logs\%TAG%.gate.log & exit /b 0 )
if exist "%RCD%\GUARD_STOP.txt" exit /b 9
:traintime
echo [%DATE% %TIME%] TRAIN %TAG%
%RUNSAFE% cmd /c %RUNCMD% train %TAG%
set "RC=!ERRORLEVEL!"
"%PY%" "%CHK%" post %TAG% !RC!
echo [%DATE% %TIME%] END %TAG% trainer rc !RC!
exit /b 0
