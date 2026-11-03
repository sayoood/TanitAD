@echo off
rem RECOVER-NEXT: the launch gate for ONE run before (or instead of) the chain: preflight + argv audit/pin + the trainer's --selfcheck-only + rnext_check.py's assertions. Usage: rnext_gate_only.cmd <tag>
rem A PASS leaves gate_token_<tag>.json; the chain then reuses it (gate-cached) when it reaches the tag. RTL / RP01 build the deviation table here (~42 min); RAPR builds none.
setlocal enabledelayedexpansion
call D:\tanitad\env.cmd
set "TAG=%~1"
set "CHK=D:\tanitad\rnext\rnext_check.py"
echo ===== GATE-ONLY %TAG% %DATE% %TIME%
"%PY%" "%CHK%" preflight
set "RC=!ERRORLEVEL!"
if !RC! NEQ 0 (echo ===== GATE-ONLY-END %TAG% PREFLIGHT-FAIL rc=!RC! & exit /b 3)
"%PY%" "%CHK%" pin %TAG%
set "RC=!ERRORLEVEL!"
if !RC! NEQ 0 (echo ===== GATE-ONLY-END %TAG% PIN-FAIL rc=!RC! & exit /b 3)
%RUNSAFE% cmd /c D:\tanitad\rnext\rnext_run.cmd gate %TAG%
set "RC=!ERRORLEVEL!"
"%PY%" "%CHK%" gate %TAG% !RC!
set "GRC=!ERRORLEVEL!"
echo ===== GATE-ONLY-END %TAG% gate-verdict-rc=!GRC! trainer-rc=!RC! %DATE% %TIME%
exit /b !GRC!
