@echo off
rem RA-1 job queue for Box 2 (Windows, RTX 4060 8 GB).  See ..\RA1_RUN_ON_BOX2.md.
rem
rem   ra1_box2_queue.cmd gate                    G-HYG / G-DVB / G-LIVE + VRAM estimate for every arm (no training, ~2 min)
rem   ra1_box2_queue.cmd stage1                  R0 x2 seeds, A0 (beta 0.04, 0.08) x2 seeds, the A1 2x2 grid at seed 0
rem   ra1_box2_queue.cmd stage1verdict           prints the best A1 cell (lowest seed-0 ADE@6s on the withheld 60-clip TRAIN holdout) from the stage-1 evals
rem   ra1_box2_queue.cmd stage2 BETA GAMMA       the best cell: A1 seed 1, A2 x2, A1x seed 0, A1h (EXPLORATORY) seed 0
rem   ra1_box2_queue.cmd final BETA GAMMA        the verdict.json over everything that finished
rem
rem HOLDOUT (PREREG_HJ_STAGE0 Amendment 1): every trainer call carries --holdout-n 60 (the trainer REFUSES to run without it): the last 60 of the
rem 600 stage-A clips in sha12 order are withheld from EVERY arm, the evals score them (--train-cache), and the A1 cell is chosen on them, NEVER on
rem eval-141, which scores K once for the chosen cell.
rem Every job is skipped when its eval.json exists, so a killed queue is simply re-run.  RA1_DRYRUN=1 prints instead of running.
rem The repo path contains an ampersand ("Architecture & Inference"): every path below is set with set "VAR=value" and
rem expanded only inside double quotes.  Never expand %P10% / %GRID% unquoted.
setlocal EnableExtensions
if not defined REPO set "REPO=D:\tanitad\repo"
if not defined PY set "PY=D:\tanitad\venv\Scripts\python.exe"
if not defined CACHE set "CACHE=D:\tanitad\cache"
if not defined RUNS set "RUNS=D:\tanitad\runs\ra1"
if not defined STEPS set "STEPS=6000"
set "PKG=%REPO%\TanitAD Research Lab\Architecture & Inference\Research\2026-10-17-v8f-mini\raw"
set "P10=%PKG%\p10\p10_prior.pt"
set "P10REF=%PKG%\p10\p10_eval_pred6.npy"
set "GRID=%PKG%\eval_grid_windows.json"
set "HOLDOUT=--holdout-n 60"
set "TRAIN=%REPO%\stack\scripts\ra1_hjepa_train.py"
set "EVAL=%REPO%\stack\scripts\ra1_hjepa_eval.py"
set "PYTHONPATH=%REPO%\stack;%REPO%\taniteval"
if not exist "%RUNS%" mkdir "%RUNS%"

if /i "%~1"=="gate" goto :gate
if /i "%~1"=="stage1" goto :stage1
if /i "%~1"=="stage1verdict" goto :stage1verdict
if /i "%~1"=="stage2" goto :stage2
if /i "%~1"=="final" goto :final
echo usage: ra1_box2_queue.cmd gate^|stage1^|stage1verdict^|stage2 BETA GAMMA^|final BETA GAMMA
exit /b 2

:gate
call :chk R0 - - ""
call :chk A0 0.08 - ""
call :chk A1 0.08 100 ""
call :chk A2 0.08 100 ""
call :chk A1x 0.08 100 ""
call :chk A1h 0.08 100 "--hist-drop 0.5"
exit /b 0

:stage1
for %%S in (0 1) do call :run R0_s%%S R0 %%S - - ""
for %%S in (0 1) do for %%B in (0.04 0.08) do call :run A0_b%%B_s%%S A0 %%S %%B - ""
for %%B in (0.04 0.08) do for %%G in (50 100) do call :run A1_b%%Bg%%G_s0 A1 0 %%B %%G ""
exit /b 0

:stage1verdict
set "VN=stage1_verdict.json"
call :verdict --run "R0@0=%RUNS%\R0_s0\eval" --run "A0:b0.04@0=%RUNS%\A0_b0.04_s0\eval" --run "A0:b0.08@0=%RUNS%\A0_b0.08_s0\eval" ^
  --run "A1:b0.04g50@0=%RUNS%\A1_b0.04g50_s0\eval" --run "A1:b0.04g100@0=%RUNS%\A1_b0.04g100_s0\eval" ^
  --run "A1:b0.08g50@0=%RUNS%\A1_b0.08g50_s0\eval" --run "A1:b0.08g100@0=%RUNS%\A1_b0.08g100_s0\eval"
exit /b 0

:stage2
if "%~2"=="" (echo stage2 needs BETA GAMMA of the best A1 cell & exit /b 2)
call :run A1_b%~2g%~3_s1 A1 1 %~2 %~3 ""
for %%S in (0 1) do call :run A2_b%~2g%~3_s%%S A2 %%S %~2 %~3 ""
call :run A1x_b%~2g%~3_s0 A1x 0 %~2 %~3 ""
call :run A1h_b%~2g%~3_s0 A1h 0 %~2 %~3 "--hist-drop 0.5"
exit /b 0

:final
set "C=b%~2g%~3"
set "VN=verdict.json"
call :verdict --run "R0@0=%RUNS%\R0_s0\eval" --run "R0@1=%RUNS%\R0_s1\eval" ^
  --run "A0:b%~2@0=%RUNS%\A0_b%~2_s0\eval" --run "A0:b%~2@1=%RUNS%\A0_b%~2_s1\eval" ^
  --run "A1:%C%@0=%RUNS%\A1_%C%_s0\eval" --run "A1:%C%@1=%RUNS%\A1_%C%_s1\eval" ^
  --run "A2:%C%@0=%RUNS%\A2_%C%_s0\eval" --run "A2:%C%@1=%RUNS%\A2_%C%_s1\eval" ^
  --run "A1x:%C%@0=%RUNS%\A1x_%C%_s0\eval" --run "A1h:%C%@0=%RUNS%\A1h_%C%_s0\eval"
exit /b 0

rem ---------------------------------------------------------------------------------------------------------------
rem :run TAG ARM SEED BETA|- GAMMA|- "EXTRA"      train (resumable) then eval one arm
:run
set "TAG=%~1"
set "OUT=%RUNS%\%~1"
if exist "%OUT%\eval\eval.json" (echo [skip] %~1 & exit /b 0)
set "ARGS=--arm %~2 --seed %~3 %HOLDOUT%"
if not "%~4"=="-" set "ARGS=%ARGS% --beta %~4"
if not "%~5"=="-" set "ARGS=%ARGS% --gamma %~5"
if not "%~6"=="" set "ARGS=%ARGS% %~6"
if defined RA1_DRYRUN (
  echo "%PY%" "%TRAIN%" %ARGS% --grid 4x10 --cache "%CACHE%" --p10-prior "%P10%" --out "%OUT%" --steps %STEPS% --eval-cache-disjoint "%CACHE%" --device cuda
  echo "%PY%" "%EVAL%" eval --ckpt "%OUT%\final.pt" --eval-cache "%CACHE%" --train-cache "%CACHE%" --p10-prior "%P10%" --grid-json "%GRID%" --p10-ref-pred "%P10REF%" --out "%OUT%\eval"
  exit /b 0
)
if not exist "%OUT%" mkdir "%OUT%"
echo [%date% %time%] train %~1
"%PY%" "%TRAIN%" %ARGS% --grid 4x10 --cache "%CACHE%" --p10-prior "%P10%" --out "%OUT%" --steps %STEPS% --eval-cache-disjoint "%CACHE%" --device cuda >> "%OUT%\train.log" 2>&1
if errorlevel 1 (echo [FAIL] train %~1 rc=%errorlevel% - see %OUT%\train.log & exit /b 1)
echo [%date% %time%] eval %~1
"%PY%" "%EVAL%" eval --ckpt "%OUT%\final.pt" --eval-cache "%CACHE%" --train-cache "%CACHE%" --p10-prior "%P10%" --grid-json "%GRID%" --p10-ref-pred "%P10REF%" --out "%OUT%\eval" >> "%OUT%\eval.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %~1 rc=%errorlevel% - an instrument control failed, read %OUT%\eval\eval.json & exit /b 1)
exit /b 0

rem :chk ARM BETA|- GAMMA|- "EXTRA"      the launch gate only
:chk
set "ARGS=--arm %~1 %HOLDOUT%"
if not "%~2"=="-" set "ARGS=%ARGS% --beta %~2"
if not "%~3"=="-" set "ARGS=%ARGS% --gamma %~3"
if not "%~4"=="" set "ARGS=%ARGS% %~4"
if defined RA1_DRYRUN (echo "%PY%" "%TRAIN%" %ARGS% --cache "%CACHE%" --p10-prior "%P10%" --out "%RUNS%\_gate_%~1" --check-only --no-resume --device cuda & exit /b 0)
"%PY%" "%TRAIN%" %ARGS% --cache "%CACHE%" --p10-prior "%P10%" --out "%RUNS%\_gate_%~1" --check-only --no-resume --device cuda
exit /b %errorlevel%

rem :verdict --run ...        (output name in %VN%; %* is NOT affected by shift, so the name is not an argument)
:verdict
set "VOUT=%RUNS%\%VN%"
if defined RA1_DRYRUN (echo "%PY%" "%EVAL%" verdict --out "%VOUT%" %* & exit /b 0)
"%PY%" "%EVAL%" verdict --out "%VOUT%" %*
exit /b %errorlevel%
