@echo off
rem run_speedsub_t1.cmd -- launcher of run_speedsub_t1.sh for a detached start on the dev box (WMI Win32_Process Create, stdin from NUL).
rem   usage:  cmd /c run_speedsub_t1.cmd [--dry-run | --stage <name>]
rem   log:    %SST1_LOGDIR%\chain_speedsub_t1.out   (default F:\TanitAD-scratch\speedsub_t1\logs)
rem   A bare WMI shell has no coreutils on PATH: this puts Git's usr\bin and bin first. The shell script sets CUDA_VISIBLE_DEVICES=-1 and OMP/MKL 8 itself.
set PATH=C:\Program Files\Git\usr\bin;C:\Program Files\Git\bin;%PATH%
set CUDA_VISIBLE_DEVICES=-1
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
if "%SST1_LOGDIR%"=="" set SST1_LOGDIR=F:\TanitAD-scratch\speedsub_t1\logs
if not exist "%SST1_LOGDIR%" mkdir "%SST1_LOGDIR%"
"C:\Program Files\Git\bin\bash.exe" "%~dp0run_speedsub_t1.sh" %* > "%SST1_LOGDIR%\chain_speedsub_t1.out" 2>&1
echo ZZSST1-CMD-EXIT %ERRORLEVEL% >> "%SST1_LOGDIR%\chain_speedsub_t1.out"
