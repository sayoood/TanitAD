@echo off
rem AMENDMENT A1 calibration (REAL features, SYNTHETIC labels only).  CPU only.  Writes raw\calib_a1.json, raw\calib_a1.partial.jsonl, raw\calib_a1.log; the last log line is ZZCALIB_DONE.
rem Launch detached (it outlives the shell):  powershell: Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c ""C:\Users\Admin\par18\spess0a1\run_calib_a1.cmd" < NUL"'; CurrentDirectory='C:\Users\Admin\par18\spess0a1'}
setlocal
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set PYTHONHASHSEED=0
set PYTHONDONTWRITEBYTECODE=1
if "%TANITAD_PY%"=="" set TANITAD_PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe
if "%CALIB_WORKERS%"=="" set CALIB_WORKERS=12
set HERE=%~dp0
"%TANITAD_PY%" "%HERE%code\calib_a1.py" --workers %CALIB_WORKERS% --out "%HERE%raw\calib_a1.json" > "%HERE%raw\calib_a1.log" 2>&1
exit /b %errorlevel%
