@echo off
rem EGOWM tree refresh on box 2 (instantiated by egowm_ship.sh, egowm_check.py render: @ROOT@ @COMMIT@ @GID@): egowm_refresh.py overlays the shipped tar onto the EGOWM-OWNED tree D:\tanitad\egowm\repo
rem with every changed file copied aside first, verifies every member's sha256, writes TIP_EGOWM.txt and egowm\tree_manifest.json. Nothing is deleted. The shared D:\tanitad\repo is never touched. No labels, no goto.
setlocal enabledelayedexpansion
call @ROOT@\env.cmd
set "SD=@ROOT@\egowm\ship"
echo ===== SHIP START @COMMIT@ @GID@ %DATE% %TIME%
"%PY%" "%SD%\egowm_refresh.py" --commit @COMMIT@
set "RC=!ERRORLEVEL!"
echo ===== SHIP END rc=!RC! %DATE% %TIME%
if !RC! NEQ 0 (echo rc=!RC!> "%SD%\SHIP_@GID@.fail" & exit /b !RC!)
echo done> "%SD%\SHIP_@GID@.done"
exit /b 0
