@echo off
rem job_dscale_mix.cmd -- launched detached by launch_wmi_mix.ps1 (WMI Win32_Process.Create, CreateFlags 0x08000200, stdin from NUL).  Calls Git\usr\bin\bash.exe, NOT Git\bin\bash.exe.
cd /d F:\TanitAD-scratch\datascale\read_mix
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/datascale/read_mix/launch_dscale_mix.sh >> out\dscale_mix_stdout.log 2>&1
echo rc=%ERRORLEVEL% finished=%DATE% %TIME%>> out\dscale_mix_stdout.log
