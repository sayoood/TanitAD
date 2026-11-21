@echo off
rem amend4.cmd -- launched detached by launch_amend4.ps1 (WMI Win32_Process.Create, CreateFlags 0x08000200, stdin from NUL). Calls Git\usr\bin\bash.exe, NOT Git\bin\bash.exe (the wrapper ^C-kills the batch at exit; memory 11-11).
rem The script prints ZZ AMEND4 READ DONE ZZ / ZZ AMEND4 READ FAIL ZZ <why> as the last line of F:\TanitAD-scratch\actrsp\lonplan\amend4\read.log.
cd /d F:\TanitAD-scratch\actrsp\lonplan\amend4\build
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/actrsp/lonplan/amend4/build/run_amend4.sh >> F:\TanitAD-scratch\actrsp\lonplan\amend4\amend4_stdout.log 2>&1
echo rc=%ERRORLEVEL% finished=%DATE% %TIME%>> F:\TanitAD-scratch\actrsp\lonplan\amend4\amend4_stdout.log
