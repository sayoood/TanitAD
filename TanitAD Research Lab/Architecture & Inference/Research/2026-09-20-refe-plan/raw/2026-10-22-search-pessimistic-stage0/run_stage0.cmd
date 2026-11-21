@echo off
rem SEARCH-PESSIMISTIC Stage 0: the REAL read, in one command.  CPU only.  Writes raw\RESULT_STAGE0.json.
rem Needs the SEALED prereg next to this file:  PREREG_STAGE0.md  and  PREREG_STAGE0.sha256  (its sha256, hex, first token of the first line).  Refuses to start without them.
rem   1. the test suite must be green (G-SUITE) -> raw\pytest_before_read.txt
rem   2. stage0.py checks: CUDA_VISIBLE_DEVICES=-1, the prereg's sha256, that it quotes this code's constants_sha256, and the sha256+size of every input
setlocal
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set PYTHONHASHSEED=0
if "%TANITAD_PY%"=="" set TANITAD_PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe
set HERE=%~dp0
if not exist "%HERE%PREREG_STAGE0.md" ( echo ABORT: no sealed PREREG_STAGE0.md next to this script & exit /b 2 )
if not exist "%HERE%PREREG_STAGE0.sha256" ( echo ABORT: no PREREG_STAGE0.sha256 next to this script & exit /b 2 )
set SEAL=
for /f "usebackq tokens=1" %%a in ("%HERE%PREREG_STAGE0.sha256") do if not defined SEAL set SEAL=%%a
if not exist "%HERE%raw" mkdir "%HERE%raw"
"%TANITAD_PY%" -m pytest "%HERE%code\test_stage0.py" -q --no-header -p no:cacheprovider > "%HERE%raw\pytest_before_read.txt" 2>&1
if errorlevel 1 ( echo ABORT: the test suite is red, see raw\pytest_before_read.txt & exit /b 3 )
"%TANITAD_PY%" "%HERE%code\stage0.py" run --prereg "%HERE%PREREG_STAGE0.md" --prereg-sha256 %SEAL% --out "%HERE%raw\RESULT_STAGE0.json"
exit /b %errorlevel%
