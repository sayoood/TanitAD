@echo off
rem amend4_readout2.cmd (MM, readout only after fix 2) -- launched detached by launch_amend4.ps1 (WMI, CreateFlags 0x08000200, stdin from NUL). Calls Git\usr\bin\bash.exe.
cd /d F:\TanitAD-scratch\actrsp\lonplan\amend4\build
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/actrsp/lonplan/amend4/build/run_amend4_readout2.sh >> F:\TanitAD-scratch\actrsp\lonplan\amend4\amend4_stdout.log 2>&1
echo rc=%ERRORLEVEL% finished=%DATE% %TIME%>> F:\TanitAD-scratch\actrsp\lonplan\amend4\amend4_stdout.log
