@echo off
rem amend7.cmd [fit|score] -- launched detached by launch7.ps1 (WMI Win32_Process.Create, CreateFlags 0x08000200, stdin from NUL). Calls the Git usr bash, not the Git bin wrapper.
set MODE=%1
if "%MODE%"=="" set MODE=fit
cd /d F:\TanitAD-scratch\actrsp\lonplan\amend7\build
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/actrsp/lonplan/amend7/build/run_amend7.sh %MODE% >> F:\TanitAD-scratch\actrsp\lonplan\amend7\amend7_stdout.log 2>&1
echo rc=%ERRORLEVEL% finished=%DATE% %TIME%>> F:\TanitAD-scratch\actrsp\lonplan\amend7\amend7_stdout.log
