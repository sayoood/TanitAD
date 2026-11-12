@echo off
rem STOPPERC: ONE box-2 job (rendered by stopperc_chain.sh with stopperc_check.py render: @ROOT@ @COMMIT@ @GID@ @ARGS@ @MARKDIR@ @MARK@ @W@ @REUSE_FAILED@). Pattern of the STOPBAL job file.
rem env.cmd gives PY and RUNSAFE. All logic is in stopperc_check.py; if it dies without writing its own marker, the fallback below writes the fail marker so the dev-box waiter never hangs. No labels, no goto.
setlocal enabledelayedexpansion
call @ROOT@\env.cmd
set "STOPBAL_COMMIT=@COMMIT@"
set "STOPPERC_W=@W@"
set "STOPPERC_NULL_REUSE_FAILED=@REUSE_FAILED@"
set "CHK=@ROOT@\stopperc\check\stopperc_check.py"
if not exist "@MARKDIR@" mkdir "@MARKDIR@"
echo ===== STOPPERC-JOB @ARGS@ commit=%STOPBAL_COMMIT% %DATE% %TIME%
%RUNSAFE% "%PY%" "%CHK%" @ARGS@
set "RC=!ERRORLEVEL!"
echo ===== STOPPERC-JOB-END rc=!RC! %DATE% %TIME%
if !RC! NEQ 0 if not exist "@MARKDIR@\@MARK@_@GID@.fail" (echo crash rc=!RC!> "@MARKDIR@\@MARK@_@GID@.fail")
exit /b !RC!
