@echo off
rem job_dscale10k_b1.cmd -- launched detached by launch_wmi.ps1 (WMI Win32_Process.Create, CreateFlags 0x08000200, stdin from NUL).  Calls Git\usr\bin\bash.exe, NOT Git\bin\bash.exe.
cd /d F:\TanitAD-scratch\datascale\read
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/datascale/read/launch_dscale10k_b1.sh >> out\dscale10k_b1_stdout.log 2>&1
echo rc=%ERRORLEVEL% finished=%DATE% %TIME%>> out\dscale10k_b1_stdout.log
