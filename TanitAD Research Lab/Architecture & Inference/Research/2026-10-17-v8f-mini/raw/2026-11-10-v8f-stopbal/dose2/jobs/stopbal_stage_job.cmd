@echo off
rem STOPBAL: ONE box-2 job (instantiated by stopbal_run.sh with stopbal_check.py render: @ROOT@ @COMMIT@ @GID@ @ARGS@ @MARKDIR@ @MARK@). Pattern of the ladder job files.
rem env.cmd gives PY and RUNSAFE. All logic is in stopbal_check.py; if it dies without writing its own marker, the fallback below writes the fail marker so the dev-box waiter never hangs. No labels, no goto.
setlocal enabledelayedexpansion
call @ROOT@\env.cmd
set "STOPBAL_COMMIT=@COMMIT@"
set "CHK=@ROOT@\stopbal\check\stopbal_check.py"
if not exist "@MARKDIR@" mkdir "@MARKDIR@"
echo ===== STOPBAL-JOB @ARGS@ commit=%STOPBAL_COMMIT% %DATE% %TIME%
%RUNSAFE% "%PY%" "%CHK%" @ARGS@
set "RC=!ERRORLEVEL!"
echo ===== STOPBAL-JSTOPBAL-END rc=!RC! %DATE% %TIME%
if !RC! NEQ 0 if not exist "@MARKDIR@\@MARK@_@GID@.fail" (echo crash rc=!RC!> "@MARKDIR@\@MARK@_@GID@.fail")
exit /b !RC!
