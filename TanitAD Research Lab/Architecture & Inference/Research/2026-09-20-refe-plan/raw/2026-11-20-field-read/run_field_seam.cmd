@echo off
rem Field-arm navtest pre-work launcher. Start DETACHED with launch_field_seam.ps1 (WMI Win32_Process.Create, CreateFlags 0x08000200).
rem %1 = arm (C, F, Cs, Fs). Git\usr\bin\bash.exe and stdin from NUL (the P5-C launcher's pattern).
set "PATH=C:\Program Files\Git\usr\bin;C:\Program Files\Git\bin;%PATH%"
if not exist "E:\Projects\TanitAD-artifacts\field_read" mkdir "E:\Projects\TanitAD-artifacts\field_read"
"C:\Program Files\Git\usr\bin\bash.exe" "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/field_read/field_seam_chain.sh" %1 > "E:\Projects\TanitAD-artifacts\field_read\chain_stdout_%1.log" 2>&1 < NUL
echo chain exit %ERRORLEVEL% >> "E:\Projects\TanitAD-artifacts\field_read\chain_stdout_%1.log"
exit /b %ERRORLEVEL%
