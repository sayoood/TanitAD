@echo off
rem OBD seed 3: ONE box-2 job (instantiated by the stage scripts with obd3_check.py render: @ROOT@ @COMMIT@ @GID@ @ARGS@ @MARKDIR@ @MARK@). Pattern of the OBD job file.
rem env.cmd gives PY and RUNSAFE. All logic is in obd3_check.py (which imports the pinned OBD checker); if it dies without writing its own marker, the fallback below writes the fail marker so the dev-box waiter never hangs.
rem PYTHONDONTWRITEBYTECODE keeps the shared OBD code tree D:\tanitad\obd\repo byte-identical (it is read only here). No labels, no goto.
setlocal enabledelayedexpansion
call @ROOT@\env.cmd
set "PYTHONDONTWRITEBYTECODE=1"
set "OBD_COMMIT=@COMMIT@"
set "CHK=@ROOT@\obd3\check\obd3_check.py"
if not exist "@MARKDIR@" mkdir "@MARKDIR@"
echo ===== OBD3-JOB @ARGS@ commit=%OBD_COMMIT% %DATE% %TIME%
%RUNSAFE% "%PY%" "%CHK%" @ARGS@
set "RC=!ERRORLEVEL!"
echo ===== OBD3-JOB-END rc=!RC! %DATE% %TIME%
if !RC! NEQ 0 if not exist "@MARKDIR@\@MARK@_@GID@.fail" (echo crash rc=!RC!> "@MARKDIR@\@MARK@_@GID@.fail")
exit /b !RC!
