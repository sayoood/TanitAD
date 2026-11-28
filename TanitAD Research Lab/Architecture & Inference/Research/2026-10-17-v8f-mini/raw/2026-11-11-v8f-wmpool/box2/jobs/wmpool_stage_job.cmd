@echo off
rem WMPOOL: ONE box-2 job (instantiated by wmpool_run.sh with wmpool_check.py render: @ROOT@ @COMMIT@ @GID@ @ARGS@ @MARKDIR@ @MARK@). Pattern of the ladder job files.
rem env.cmd gives PY and RUNSAFE. All logic is in wmpool_check.py; if it dies without writing its own marker, the fallback below writes the fail marker so the dev-box waiter never hangs. No labels, no goto.
setlocal enabledelayedexpansion
call @ROOT@\env.cmd
set "WMPOOL_COMMIT=@COMMIT@"
set "CHK=@ROOT@\wmpool\check\wmpool_check.py"
if not exist "@MARKDIR@" mkdir "@MARKDIR@"
echo ===== WMPOOL-JOB @ARGS@ commit=%WMPOOL_COMMIT% %DATE% %TIME%
%RUNSAFE% "%PY%" "%CHK%" @ARGS@
set "RC=!ERRORLEVEL!"
echo ===== WMPOOL-JOB-END rc=!RC! %DATE% %TIME%
if !RC! NEQ 0 if not exist "@MARKDIR@\@MARK@_@GID@.fail" (echo crash rc=!RC!> "@MARKDIR@\@MARK@_@GID@.fail")
exit /b !RC!
