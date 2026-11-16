@echo off
rem job_wh.cmd -- launched detached by launch_wh_wmi.ps1 (WMI Win32_Process.Create, CreateFlags 0x08000200, stdin from NUL).  Calls Git\usr\bin\bash.exe, NOT Git\bin\bash.exe
rem (the wrapper ^C-kills the batch at exit; memory 11-11).  The script prints ZZ WH CHAIN DONE ZZ / ZZ WH CHAIN FAIL ZZ <why> as the last line of run\wh_launch.log.
cd /d F:\TanitAD-scratch\strat1\whead
if not exist run mkdir run
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/strat1/whead/launch_wh.sh >> run\wh_stdout.log 2>&1
echo rc=%ERRORLEVEL% finished=%DATE% %TIME%>> run\wh_stdout.log
