@echo off
rem job_navrbd_chain.cmd -- launched detached by launch_wmi.ps1 (WMI Win32_Process.Create, CreateFlags 0x08000200, stdin from NUL).  Calls Git\usr\bin\bash.exe, NOT Git\bin\bash.exe.
rem Sentinel (last line of out\navrbd_read_chain.log): ZZ NAVRBD READ DONE ZZ / ZZ NAVRBD READ FAIL ZZ <step and why>
cd /d F:\TanitAD-scratch\strat1\navrbd\read
if not exist out mkdir out
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/strat1/navrbd/read/navrbd_chain.sh >> out\navrbd_chain_stdout.log 2>&1
echo rc=%ERRORLEVEL% finished=%DATE% %TIME%>> out\navrbd_chain_stdout.log
