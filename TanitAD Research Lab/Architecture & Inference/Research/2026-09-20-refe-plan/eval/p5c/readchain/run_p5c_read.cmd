@echo off
rem P5-C read chain launcher. Start it DETACHED with launch_p5c_read.ps1 (WMI Win32_Process.Create, CreateFlags 0x08000200); never from an agent shell.
rem %1 = launch_env.cmd written by the launcher (WMI children do not inherit the caller's environment).
rem Real run: launch_p5c_read.ps1    Dry run: dry_run_p5c.ps1 (see RUNBOOK_P5C_READ.md section 7).
rem Uses Git\usr\bin\bash.exe (the Git\bin wrapper ^C-kills the batch at exit) and stdin from NUL.
set "HOLD=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain"
set "PATH=C:\Program Files\Git\usr\bin;C:\Program Files\Git\bin;%PATH%"
if not "%~1"=="" call "%~1"
if "%P5C_WORK%"=="" set "P5C_WORK=E:/Projects/TanitAD-artifacts/p5c_read"
if not exist "%P5C_WORK%" mkdir "%P5C_WORK%"
if "%P5C_SCRIPT%"=="" set "P5C_SCRIPT=%HOLD%/p5c_read_chain.sh"
"C:\Program Files\Git\usr\bin\bash.exe" "%P5C_SCRIPT%" > "%P5C_WORK%\chain_stdout.log" 2>&1 < NUL
echo chain exit %ERRORLEVEL% >> "%P5C_WORK%\chain_stdout.log"
exit /b %ERRORLEVEL%
