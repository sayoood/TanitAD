@echo off
rem OTBAND: ONE box-2 job (instantiated by otband_run.sh with otband_check.py render: @ROOT@ @COMMIT@ @GID@ @ARGS@ @MARKDIR@ @MARK@). Pattern of the ladder job files.
rem env.cmd gives PY and RUNSAFE. All logic is in otband_check.py; if it dies without writing its own marker, the fallback below writes the fail marker so the dev-box waiter never hangs. No labels, no goto.
setlocal enabledelayedexpansion
call @ROOT@\env.cmd
set "OTBAND_COMMIT=@COMMIT@"
set "CHK=@ROOT@\otband\check\otband_check.py"
if not exist "@MARKDIR@" mkdir "@MARKDIR@"
echo ===== OTBAND-JOB @ARGS@ commit=%OTBAND_COMMIT% %DATE% %TIME%
%RUNSAFE% "%PY%" "%CHK%" @ARGS@
set "RC=!ERRORLEVEL!"
echo ===== OTBAND-JOB-END rc=!RC! %DATE% %TIME%
if !RC! NEQ 0 if not exist "@MARKDIR@\@MARK@_@GID@.fail" (echo crash rc=!RC!> "@MARKDIR@\@MARK@_@GID@.fail")
exit /b !RC!
