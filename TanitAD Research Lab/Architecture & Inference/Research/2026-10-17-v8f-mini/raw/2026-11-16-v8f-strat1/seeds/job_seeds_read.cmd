@echo off
rem job_seeds_read.cmd -- launched detached by launch_wmi.ps1 (WMI Win32_Process.Create, CreateFlags 0x08000200, stdin from NUL).  Calls Git\usr\bin\bash.exe, NOT Git\bin\bash.exe
rem (the wrapper ^C-kills the batch at exit; memory 11-11).  Sentinel: last line of out\seeds_read.log.
cd /d F:\TanitAD-scratch\strat1\read\seeds
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/strat1/read/seeds/launch_seeds_read.sh >> out\seeds_read_stdout.log 2>&1
echo rc=%ERRORLEVEL% finished=%DATE% %TIME%>> out\seeds_read_stdout.log
