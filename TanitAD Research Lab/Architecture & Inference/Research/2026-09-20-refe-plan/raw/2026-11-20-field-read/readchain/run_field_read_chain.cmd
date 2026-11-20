@echo off
rem Field-read chain launcher. Start DETACHED (WMI Win32_Process.Create, CreateFlags 0x08000200), like run_field_seam.cmd.
rem %1 = mode (pre-C | full). Optional env: FR_DIR, FR_WORK, FR_DRY, FR_TIP (write them with set "X=..." in a launch_env.cmd and pass it as %2, or set them in the launching shell for a WMI-less start).
set "PATH=C:\Program Files\Git\usr\bin;C:\Program Files\Git\bin;%PATH%"
if not "%~2"=="" call "%~2"
if "%FR_WORK%"=="" set "FR_WORK=E:/Projects/TanitAD-artifacts/field_verdict"
if not exist "%FR_WORK%" mkdir "%FR_WORK%"
"C:\Program Files\Git\usr\bin\bash.exe" "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/field_read/readchain/field_read_chain.sh" %1 > "%FR_WORK%\chain_stdout_%1.log" 2>&1 < NUL
echo chain exit %ERRORLEVEL% >> "%FR_WORK%\chain_stdout_%1.log"
exit /b %ERRORLEVEL%
