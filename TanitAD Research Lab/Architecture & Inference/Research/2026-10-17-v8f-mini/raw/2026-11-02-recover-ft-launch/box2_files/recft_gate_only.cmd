@echo off
rem RECOVER-FT: the launch gate for ONE run before the chain starts (R_0, then C0_0): preflight + pin + the trainer's --selfcheck-only + recft_check.py's assertions. Usage: recft_gate_only.cmd <tag>
setlocal enabledelayedexpansion
call D:\tanitad\env.cmd
set "TAG=%~1"
set "CHK=D:\tanitad\recft\recft_check.py"
echo ===== GATE-ONLY %TAG% %DATE% %TIME%
"%PY%" "%CHK%" preflight
set "RC=!ERRORLEVEL!"
if !RC! NEQ 0 (echo ===== GATE-ONLY-END %TAG% PREFLIGHT-FAIL rc=!RC! & exit /b 3)
"%PY%" "%CHK%" pin %TAG%
set "RC=!ERRORLEVEL!"
if !RC! NEQ 0 (echo ===== GATE-ONLY-END %TAG% PIN-FAIL rc=!RC! & exit /b 3)
%RUNSAFE% cmd /c D:\tanitad\recft\recft_run.cmd gate %TAG%
set "RC=!ERRORLEVEL!"
"%PY%" "%CHK%" gate %TAG% !RC!
set "GRC=!ERRORLEVEL!"
echo ===== GATE-ONLY-END %TAG% gate-verdict-rc=!GRC! trainer-rc=!RC! %DATE% %TIME%
exit /b !GRC!
