@echo off
rem ACTSENS tree refresh on box 2 (instantiated by actsens_ship.sh, actsens_check.py render: @ROOT@ @COMMIT@ @GID@): actsens_refresh.py overlays the shipped tar onto the ACTSENS-OWNED tree D:\tanitad\actsens\repo
rem with every changed file copied aside first, verifies every member's sha256, writes TIP_ACTSENS.txt and actsens\tree_manifest.json. Nothing is deleted. The shared D:\tanitad\repo is never touched. No labels, no goto.
setlocal enabledelayedexpansion
call @ROOT@\env.cmd
set "SD=@ROOT@\actsens\ship"
echo ===== SHIP START @COMMIT@ @GID@ %DATE% %TIME%
"%PY%" "%SD%\actsens_refresh.py" --commit @COMMIT@
set "RC=!ERRORLEVEL!"
echo ===== SHIP END rc=!RC! %DATE% %TIME%
if !RC! NEQ 0 (echo rc=!RC!> "%SD%\SHIP_@GID@.fail" & exit /b !RC!)
echo done> "%SD%\SHIP_@GID@.done"
exit /b 0
