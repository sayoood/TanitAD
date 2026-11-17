@echo off
rem job_ckpt120_read.cmd -- launched detached by launch_wmi.ps1 (WMI Win32_Process.Create, CreateFlags 0x08000200, stdin from NUL).  Calls Git\usr\bin\bash.exe, NOT Git\bin\bash.exe.
cd /d F:\TanitAD-scratch\strat1\read\ckpt120
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/strat1/read/ckpt120/launch_ckpt120_read.sh >> out\ckpt120_read_stdout.log 2>&1
echo rc=%ERRORLEVEL% finished=%DATE% %TIME%>> out\ckpt120_read_stdout.log
