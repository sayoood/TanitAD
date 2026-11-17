@echo off
rem job_dscale10k_b2.cmd -- launched detached by launch_wmi_b2.ps1 (WMI Win32_Process.Create, CreateFlags 0x08000200, stdin from NUL).  Calls Git\usr\bin\bash.exe, NOT Git\bin\bash.exe.
cd /d F:\TanitAD-scratch\datascale\read2
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/datascale/read2/launch_dscale10k_b2.sh >> out\dscale10k_b2_stdout.log 2>&1
echo rc=%ERRORLEVEL% finished=%DATE% %TIME%>> out\dscale10k_b2_stdout.log
