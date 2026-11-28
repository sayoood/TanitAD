@echo off
rem job_strat1_read.cmd -- launched detached by launch_wmi.ps1 (WMI Win32_Process.Create, CreateFlags 0x08000200, stdin from NUL).  Calls Git\usr\bin\bash.exe, NOT Git\bin\bash.exe
rem (the wrapper ^C-kills the batch at exit; memory 11-11).  The script prints the sentinel ZZ STRAT1 READ DONE ZZ / ZZ STRAT1 READ FAIL ZZ <why> as the last line of out\strat1_read.log.
cd /d F:\TanitAD-scratch\strat1\read
if exist out\STRAT1_READ.done del out\STRAT1_READ.done
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/strat1/read/launch_strat1_read.sh >> out\strat1_read_stdout.log 2>&1
echo rc=%ERRORLEVEL% finished=%DATE% %TIME%>> out\strat1_read_stdout.log
