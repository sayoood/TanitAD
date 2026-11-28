@echo off
rem job_wh2.cmd -- (runs folder run2: the c02bf3a6 tree, tokens minted there) launched detached by launch_wh_wmi.ps1 (WMI Win32_Process.Create, CreateFlags 0x08000200, stdin from NUL).  Calls Git\usr\bin\bash.exe, NOT Git\bin\bash.exe
rem (the wrapper ^C-kills the batch at exit; memory 11-11).  The script prints ZZ WH CHAIN DONE ZZ / ZZ WH CHAIN FAIL ZZ <why> as the last line of run2\wh_launch.log.
cd /d F:\TanitAD-scratch\strat1\whead
if not exist run2 mkdir run2
set WH_RUNS=F:/TanitAD-scratch/strat1/whead/run2
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/strat1/whead/launch_wh.sh >> run2\wh_stdout.log 2>&1
echo rc=%ERRORLEVEL% finished=%DATE% %TIME%>> run2\wh_stdout.log
