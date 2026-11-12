@echo off
rem STOPPERC tree refresh on box 2 (rendered by stopperc_chain.sh: @ROOT@ @COMMIT@ @GID@): stopbal_refresh.py (the STOPBAL one, unchanged, parametrised by STOPBAL_HOME / STOPBAL_REPO / STOPBAL_ASIDE_ROOT) overlays the shipped tar onto the
rem STOPPERC-OWNED tree D:\tanitad\stopperc\repo. Changed files are copied aside first; every member's sha256 is verified; TIP_STOPBAL.txt and stopperc\tree_manifest.json are written. Nothing is deleted. The STOPBAL tree and the shared repo are never touched.
setlocal enabledelayedexpansion
call @ROOT@\env.cmd
set "STOPBAL_HOME=@ROOT@\stopperc"
set "STOPBAL_REPO=@ROOT@\stopperc\repo"
set "STOPBAL_ASIDE_ROOT=@ROOT@\stopperc"
set "SD=@ROOT@\stopperc\ship"
echo ===== SHIP START @COMMIT@ @GID@ %DATE% %TIME%
"%PY%" "%SD%\stopbal_refresh.py" --commit @COMMIT@
set "RC=!ERRORLEVEL!"
echo ===== SHIP END rc=!RC! %DATE% %TIME%
if !RC! NEQ 0 (echo rc=!RC!> "%SD%\SHIP_@GID@.fail" & exit /b !RC!)
echo done> "%SD%\SHIP_@GID@.done"
exit /b 0
