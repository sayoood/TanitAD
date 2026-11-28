@echo off
rem OPTB: ONE box-2 job (instantiated by optb_run.sh with optb_check.py render: @ROOT@ @COMMIT@ @GID@ @ARGS@ @MARKDIR@ @MARK@). Pattern of the ladder job files.
rem env.cmd gives PY and RUNSAFE. All logic is in optb_check.py; if it dies without writing its own marker, the fallback below writes the fail marker so the dev-box waiter never hangs. No labels, no goto.
setlocal enabledelayedexpansion
call @ROOT@\env.cmd
set "OPTB_COMMIT=@COMMIT@"
set "CHK=@ROOT@\optb\check\optb_check.py"
if not exist "@MARKDIR@" mkdir "@MARKDIR@"
echo ===== OPTB-JOB @ARGS@ commit=%OPTB_COMMIT% %DATE% %TIME%
%RUNSAFE% "%PY%" "%CHK%" @ARGS@
set "RC=!ERRORLEVEL!"
echo ===== OPTB-JOB-END rc=!RC! %DATE% %TIME%
if !RC! NEQ 0 if not exist "@MARKDIR@\@MARK@_@GID@.fail" (echo crash rc=!RC!> "@MARKDIR@\@MARK@_@GID@.fail")
exit /b !RC!
