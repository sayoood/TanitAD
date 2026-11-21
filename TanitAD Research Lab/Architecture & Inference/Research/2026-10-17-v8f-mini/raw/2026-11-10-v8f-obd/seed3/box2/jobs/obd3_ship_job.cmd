@echo off
rem OBD seed 3, FALLBACK tree refresh on box 2 (instantiated by obd3_ship.sh only when shipcheck could not prove the tree; obd3_check.py render: @ROOT@ @COMMIT@ @GID@).
rem The OBD chain's own obd_refresh.py (sha256-pinned, unchanged) overlays the shipped tar onto the SAME tree D:\tanitad\obd\repo (the commit is the same one): every changed file copied aside first,
rem every member's sha256 verified, TIP_OBD.txt and obd\tree_manifest.json written last. Nothing is deleted. The shared D:\tanitad\repo is never touched. No labels, no goto.
setlocal enabledelayedexpansion
call @ROOT@\env.cmd
set "PYTHONDONTWRITEBYTECODE=1"
set "SD=@ROOT@\obd3\ship"
echo ===== OBD3-SHIP START @COMMIT@ @GID@ %DATE% %TIME%
"%PY%" "%SD%\obd_refresh.py" --commit @COMMIT@ --tar "%SD%\obd_tree.tar" --manifest "%SD%\obd_tree_manifest.json" --result "%SD%\SHIP_RESULT.json"
set "RC=!ERRORLEVEL!"
echo ===== OBD3-SHIP END rc=!RC! %DATE% %TIME%
if !RC! NEQ 0 (echo rc=!RC!> "%SD%\SHIP_@GID@.fail" & exit /b !RC!)
echo done> "%SD%\SHIP_@GID@.done"
exit /b 0
