@echo off
rem RA-1 job queue for Box 2 (Windows, RTX 4060 8 GB).  See ..\RA1_RUN_ON_BOX2.md.
rem
rem   ra1_box2_queue.cmd gate                    G-HYG / G-DVB / G-LIVE + VRAM estimate for every arm (no training, ~2 min)
rem   ra1_box2_queue.cmd stage1                  R0 x2 seeds, A0 (beta 0.04, 0.08) x2 seeds, the A1 2x2 grid at seed 0
rem   ra1_box2_queue.cmd stage1verdict           prints the best A1 cell (lowest seed-0 ADE@6s on the withheld 60-clip TRAIN holdout) from the stage-1 evals
rem   ra1_box2_queue.cmd stage2 BETA GAMMA       the best cell: A1 seed 1, A2 x2, A1x seed 0, A1h (EXPLORATORY) seed 0
rem   ra1_box2_queue.cmd final BETA GAMMA        the verdict.json over everything that finished
rem   ra1_box2_queue.cmd es "S1 S2 ..."         PREREG_RA1_ES ES-1: R0 x2 seeds at each --steps budget, in %RUNS%_es\st<S>
rem   ra1_box2_queue.cmd es2 S BETA GAMMA       PREREG_RA1_ES ES-2: at budget S, A1 x2 seeds + A1x seed 0 at one cell, same root
rem   ra1_box2_queue.cmd es3 S BETA             PREREG_RA1_ES Addendum A: A0 x2 seeds at budget S, + the verdict over R0/A0/A1/A1x at S
rem   ra1_box2_queue.cmd ridge                  PREREG_RA1_RIDGE: the CPU ridge probe of the pooled features (no training)
rem
rem HOLDOUT (PREREG_HJ_STAGE0 Amendment 1): every trainer call carries --holdout-n 60 (the trainer REFUSES to run without it): the last 60 of the
rem 600 stage-A clips in sha12 order are withheld from EVERY arm, the evals score them (--train-cache), and the A1 cell is chosen on them, NEVER on
rem eval-141, which scores K once for the chosen cell.
rem Every job is skipped when its eval.json exists, so a killed queue is simply re-run.  RA1_DRYRUN=1 prints instead of running.
rem
rem PI RULING D-PI-YAWRATE-1 (2026-10-17), PREREG_HJ_STAGE0 Amendment 3: *"the model should get measured ego motion and its history, instead
rem of steering angle ... the model should get the measured yaw rate"*.  This queue runs the COMPLIANT representation: every trainer / eval call
rem carries --kin-mode yaw and --tracks native_tracks.pt (the raw 10 Hz pose tracks: the cache holds only the even frames, the one-frame yaw
rem difference needs the odd ones; build with stack\scripts\build_native_tracks.py and copy next to the cache), the prior is P10-yaw-540, and the
rem runs go to ...\runs\ra1_yaw.  The 2026-10-17 22:35 steer-representation runs live in ...\runs\ra1 and are NON-DECIDING (representation-
rem equivalence check only); this queue must NEVER share their directory (a steer eval.json would make a compliant job "skip as done").
rem The repo path contains an ampersand ("Architecture & Inference"): every path below is set with set "VAR=value" and
rem expanded only inside double quotes.  Never expand %P10% / %GRID% unquoted.
setlocal EnableExtensions
if not defined REPO set "REPO=D:\tanitad\repo"
if not defined PY set "PY=D:\tanitad\venv\Scripts\python.exe"
if not defined CACHE set "CACHE=D:\tanitad\cache"
if not defined RUNS set "RUNS=D:\tanitad\runs\ra1_yaw"
if not defined STEPS set "STEPS=6000"
set "PKG=%REPO%\TanitAD Research Lab\Architecture & Inference\Research\2026-10-17-v8f-mini\raw"
rem PREREG_HJ_STAGE0 Amendment 2 + 3: RA-1 uses P10-yaw-540 (fit on the 540 TRAIN clips only, so the holdout is out of sample for the prior too;
rem its features are the MEASURED-yaw-rate ones, units pose_kappa -- a steer-unit prior is REFUSED by the trainer under --kin-mode yaw).
set "P10DIR=%REPO%\TanitAD Research Lab\Architecture & Inference\Research\2026-10-17-refav1-hj-mini\raw\p10_yaw540"
set "TRACKS=%CACHE%\native_tracks.pt"
set "KIN=--kin-mode yaw"
set "P10=%P10DIR%\p10_prior.pt"
set "P10REF=%P10DIR%\p10_eval_pred6.npy"
set "GRID=%PKG%\eval_grid_windows.json"
set "HOLDOUT=--holdout-n 60"
set "TRAIN=%REPO%\stack\scripts\ra1_hjepa_train.py"
set "EVAL=%REPO%\stack\scripts\ra1_hjepa_eval.py"
set "RIDGE=%REPO%\stack\scripts\ra1_ridge_probe.py"
set "PYTHONPATH=%REPO%\stack;%REPO%\taniteval"
if not exist "%RUNS%" mkdir "%RUNS%"
if not defined RA1_DRYRUN if not exist "%TRACKS%" (echo REFUSED: %TRACKS% is missing - build it with stack\scripts\build_native_tracks.py and copy it next to the cache & exit /b 2)

if /i "%~1"=="gate" goto :gate
if /i "%~1"=="stage1" goto :stage1
if /i "%~1"=="stage1verdict" goto :stage1verdict
if /i "%~1"=="stage2" goto :stage2
if /i "%~1"=="final" goto :final
if /i "%~1"=="es" goto :es
if /i "%~1"=="es2" goto :es2
if /i "%~1"=="es3" goto :es3
if /i "%~1"=="ridge" goto :ridge
echo usage: ra1_box2_queue.cmd gate^|stage1^|stage1verdict^|stage2 BETA GAMMA^|final BETA GAMMA
echo        ra1_box2_queue.cmd es "S1 S2 ..."^|es2 S BETA GAMMA   (PREREG_RA1_ES)
echo        ra1_box2_queue.cmd es3 S BETA^|ridge   (PREREG_RA1_ES Addendum A, PREREG_RA1_RIDGE)
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
:es
rem PREREG_RA1_ES ES-1. Each budget is a FULL run (the cosine schedule is compressed to --steps; the trainer keeps only latest.pt,
rem so intermediate checkpoints of the 6000-step runs do not exist). Every other constant is the :run block's, unchanged.
if "%~2"=="" (echo es needs a quoted step list, e.g. es "250 500 1000 2000" & exit /b 2)
set "ESROOT=%RUNS%_es"
for %%N in (%~2) do (
  set "STEPS=%%N"
  set "RUNS=%ESROOT%\st%%N"
  if not exist "%ESROOT%\st%%N" mkdir "%ESROOT%\st%%N"
  for %%S in (0 1) do call :run R0_s%%S R0 %%S - - "" || exit /b 1
)
exit /b 0

:es2
rem PREREG_RA1_ES ES-2: the budget S* is R0's (holdout-selected), applied unchanged to A1 and A1x; the cell is stage 1's holdout pick.
if "%~4"=="" (echo es2 needs S BETA GAMMA & exit /b 2)
set "STEPS=%~2"
set "RUNS=%RUNS%_es\st%~2"
if not exist "%RUNS%" mkdir "%RUNS%"
for %%S in (0 1) do call :run A1_b%~3g%~4_s%%S A1 %%S %~3 %~4 "" || exit /b 1
call :run A1x_b%~3g%~4_s0 A1x 0 %~3 %~4 "" || exit /b 1
rem The queue's own verdict over this budget's runs (R0 from ES-1): A1x gate, then K on {R0, A1}. Its SENS/branch text assumes A0 runs,
rem which this sweep does not have -- code\ra1_es_verdict.py applies PREREG_RA1_ES's readings instead.
set "VN=es2_verdict.json"
set "C=b%~3g%~4"
call :verdict --run "R0@0=%RUNS%\R0_s0\eval" --run "R0@1=%RUNS%\R0_s1\eval" --run "A1:%C%@0=%RUNS%\A1_%C%_s0\eval" --run "A1:%C%@1=%RUNS%\A1_%C%_s1\eval" --run "A1x:%C%@0=%RUNS%\A1x_%C%_s0\eval"
exit /b %errorlevel%

:es3
rem PREREG_RA1_ES Addendum A: A0 (SIGReg + prediction, gamma 0) at the ES-2 budget, then the queue's verdict over every arm at that
rem budget. A0's cell b<BETA> pairs with A1's b<BETA>g50 (the verdict's a0_for rule); the A1 cell is ES-2's.
if "%~3"=="" (echo es3 needs S BETA & exit /b 2)
set "STEPS=%~2"
set "RUNS=%RUNS%_es\st%~2"
if not exist "%RUNS%" mkdir "%RUNS%"
for %%S in (0 1) do call :run A0_b%~3_s%%S A0 %%S %~3 - "" || exit /b 1
set "VN=es3_verdict.json"
set "C=b%~3g50"
call :verdict --run "R0@0=%RUNS%\R0_s0\eval" --run "R0@1=%RUNS%\R0_s1\eval" --run "A0:b%~3@0=%RUNS%\A0_b%~3_s0\eval" --run "A0:b%~3@1=%RUNS%\A0_b%~3_s1\eval" --run "A1:%C%@0=%RUNS%\A1_%C%_s0\eval" --run "A1:%C%@1=%RUNS%\A1_%C%_s1\eval" --run "A1x:%C%@0=%RUNS%\A1x_%C%_s0\eval"
exit /b %errorlevel%

:ridge
rem PREREG_RA1_RIDGE: CPU only (no --device), no training. Output next to the ES runs.
if not exist "%RUNS%_es" mkdir "%RUNS%_es"
if defined RA1_DRYRUN (echo "%PY%" "%RIDGE%" --cache "%CACHE%" --tracks "%TRACKS%" --p10-prior "%P10%" --grid-json "%GRID%" %HOLDOUT% --out "%RUNS%_es\ridge_probe.json" & exit /b 0)
"%PY%" "%RIDGE%" --cache "%CACHE%" --tracks "%TRACKS%" --p10-prior "%P10%" --grid-json "%GRID%" %HOLDOUT% --out "%RUNS%_es\ridge_probe.json" > "%RUNS%_es\ridge_probe.log" 2>&1
exit /b %errorlevel%

:run
set "TAG=%~1"
set "OUT=%RUNS%\%~1"
if exist "%OUT%\eval\eval.json" (echo [skip] %~1 & exit /b 0)
set "ARGS=--arm %~2 --seed %~3 %HOLDOUT% %KIN%"
if not "%~4"=="-" set "ARGS=%ARGS% --beta %~4"
if not "%~5"=="-" set "ARGS=%ARGS% --gamma %~5"
if not "%~6"=="" set "ARGS=%ARGS% %~6"
if defined RA1_DRYRUN (
  echo "%PY%" "%TRAIN%" %ARGS% --grid 4x10 --cache "%CACHE%" --tracks "%TRACKS%" --p10-prior "%P10%" --out "%OUT%" --steps %STEPS% --eval-cache-disjoint "%CACHE%" --device cuda
  echo "%PY%" "%EVAL%" eval --ckpt "%OUT%\final.pt" --eval-cache "%CACHE%" --train-cache "%CACHE%" --tracks "%TRACKS%" --p10-prior "%P10%" --grid-json "%GRID%" --p10-ref-pred "%P10REF%" --out "%OUT%\eval"
  exit /b 0
)
if not exist "%OUT%" mkdir "%OUT%"
echo [%date% %time%] train %~1
"%PY%" "%TRAIN%" %ARGS% --grid 4x10 --cache "%CACHE%" --tracks "%TRACKS%" --p10-prior "%P10%" --out "%OUT%" --steps %STEPS% --eval-cache-disjoint "%CACHE%" --device cuda >> "%OUT%\train.log" 2>&1
if errorlevel 1 (echo [FAIL] train %~1 rc=%errorlevel% - see %OUT%\train.log & exit /b 1)
echo [%date% %time%] eval %~1
"%PY%" "%EVAL%" eval --ckpt "%OUT%\final.pt" --eval-cache "%CACHE%" --train-cache "%CACHE%" --tracks "%TRACKS%" --p10-prior "%P10%" --grid-json "%GRID%" --p10-ref-pred "%P10REF%" --out "%OUT%\eval" >> "%OUT%\eval.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %~1 rc=%errorlevel% - an instrument control failed, read %OUT%\eval\eval.json & exit /b 1)
exit /b 0

rem :chk ARM BETA|- GAMMA|- "EXTRA"      the launch gate only
:chk
set "ARGS=--arm %~1 %HOLDOUT% %KIN%"
if not "%~2"=="-" set "ARGS=%ARGS% --beta %~2"
if not "%~3"=="-" set "ARGS=%ARGS% --gamma %~3"
if not "%~4"=="" set "ARGS=%ARGS% %~4"
if defined RA1_DRYRUN (echo "%PY%" "%TRAIN%" %ARGS% --cache "%CACHE%" --tracks "%TRACKS%" --p10-prior "%P10%" --out "%RUNS%\_gate_%~1" --check-only --no-resume --device cuda & exit /b 0)
"%PY%" "%TRAIN%" %ARGS% --cache "%CACHE%" --tracks "%TRACKS%" --p10-prior "%P10%" --out "%RUNS%\_gate_%~1" --check-only --no-resume --device cuda
exit /b %errorlevel%

rem :verdict --run ...        (output name in %VN%; %* is NOT affected by shift, so the name is not an argument)
:verdict
set "VOUT=%RUNS%\%VN%"
if defined RA1_DRYRUN (echo "%PY%" "%EVAL%" verdict --out "%VOUT%" %* & exit /b 0)
"%PY%" "%EVAL%" verdict --out "%VOUT%" %*
exit /b %errorlevel%
