@echo off
rem amend6.cmd -- launched detached by launch6.ps1 -Job amend6.cmd (WMI Win32_Process.Create, CreateFlags 0x08000200, stdin from NUL). Calls Git\usr\bin\bash.exe, NOT Git\bin\bash.exe.
cd /d F:\TanitAD-scratch\actrsp\lonplan\amend6\build
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/actrsp/lonplan/amend6/build/run_amend6.sh >> F:\TanitAD-scratch\actrsp\lonplan\amend6\amend6_stdout.log 2>&1
echo rc=%ERRORLEVEL% finished=%DATE% %TIME%>> F:\TanitAD-scratch\actrsp\lonplan\amend6\amend6_stdout.log
