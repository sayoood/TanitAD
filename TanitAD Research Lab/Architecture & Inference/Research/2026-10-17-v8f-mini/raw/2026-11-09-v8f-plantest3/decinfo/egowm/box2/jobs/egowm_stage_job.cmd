@echo off
rem EGOWM: ONE box-2 job (instantiated by egowm_run.sh with egowm_check.py render: @ROOT@ @COMMIT@ @GID@ @ARGS@ @MARKDIR@ @MARK@). Pattern of the ladder job files.
rem env.cmd gives PY and RUNSAFE. All logic is in egowm_check.py; if it dies without writing its own marker, the fallback below writes the fail marker so the dev-box waiter never hangs. No labels, no goto.
setlocal enabledelayedexpansion
call @ROOT@\env.cmd
set "EGOWM_COMMIT=@COMMIT@"
set "CHK=@ROOT@\egowm\check\egowm_check.py"
if not exist "@MARKDIR@" mkdir "@MARKDIR@"
echo ===== EGOWM-JOB @ARGS@ commit=%EGOWM_COMMIT% %DATE% %TIME%
%RUNSAFE% "%PY%" "%CHK%" @ARGS@
set "RC=!ERRORLEVEL!"
echo ===== EGOWM-JOB-END rc=!RC! %DATE% %TIME%
if !RC! NEQ 0 if not exist "@MARKDIR@\@MARK@_@GID@.fail" (echo crash rc=!RC!> "@MARKDIR@\@MARK@_@GID@.fail")
exit /b !RC!
