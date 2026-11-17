@echo off
rem job_c0rep_read.cmd -- launched detached by launch_wmi.ps1 (WMI Win32_Process.Create, CreateFlags 0x08000200, stdin from NUL).  Calls Git\usr\bin\bash.exe, NOT Git\bin\bash.exe.
cd /d F:\TanitAD-scratch\strat1\read\c0rep
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/strat1/read/c0rep/launch_c0rep_read.sh >> out\c0rep_read_stdout.log 2>&1
echo rc=%ERRORLEVEL% finished=%DATE% %TIME%>> out\c0rep_read_stdout.log
