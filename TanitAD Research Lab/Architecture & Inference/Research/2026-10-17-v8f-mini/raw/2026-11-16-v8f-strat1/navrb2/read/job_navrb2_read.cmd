@echo off
rem job_navrb2_read.cmd -- launched detached by launch_wmi.ps1 (WMI Win32_Process.Create, CreateFlags 0x08000200, stdin from NUL).  Calls Git\usr\bin\bash.exe, NOT Git\bin\bash.exe
rem (the wrapper ^C-kills the batch at exit; memory 11-11).  Set NAVRB2_STANDIN_ROOT in the environment of the WMI call for the dry run.  Sentinel: ZZ NAVRB2 READ DONE ZZ / ZZ NAVRB2 READ FAIL ZZ <why>
rem as the last line of out\navrb2_read.log.
cd /d F:\TanitAD-scratch\strat1\navrb2\read
if not exist out mkdir out
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/strat1/navrb2/read/launch_navrb2_read.sh >> out\navrb2_read_stdout.log 2>&1
echo rc=%ERRORLEVEL% finished=%DATE% %TIME%>> out\navrb2_read_stdout.log
