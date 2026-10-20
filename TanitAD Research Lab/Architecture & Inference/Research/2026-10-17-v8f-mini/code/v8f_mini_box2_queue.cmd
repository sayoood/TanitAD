@echo off
rem v8F-mini job queue for Box 2 (Windows, RTX 4060 8 GB). PREREG_V8F_MINI Amendment 4: the full corpus (cache_full), the RA-1
rem 60-clip holdout withheld (--exclude 540:600, so TRAIN = RA-1 full's 4,512 clips), the residual base and reference = P10-yaw-4512.
rem
rem   v8f_mini_box2_queue.cmd gate              the trainer's in-process launch gate (G-DVB + G-LIVE + admissibility) for M1 and M2b, no
rem                                             training; the first call builds the v9 join once (%RUNSM%\join_train.npz)
rem   v8f_mini_box2_queue.cmd train ARM SEED [GRID]   one 4S run (resumes from ckpt_last.pt; skipped once summary.json exists)
rem   v8f_mini_box2_queue.cmd eval ARM SEED [GRID]    eval-141 of that run (+ the nav-shuffle control for M1); skipped once its npz exists
rem   v8f_mini_box2_queue.cmd floors            P10-yaw-4512 and the kinematic floors on the eval grid, into %RUNSM%\eval
rem   v8f_mini_box2_queue.cmd compare           the PREREG bars (K-1, CONS-1, STRAT-1, TAC-1) over the M1 / M2b / navshuf evals
rem   v8f_mini_box2_queue.cmd tac               PREREG_V8F_TAC (eval only): the T -> O intervention -- O reads T's own code,
rem                                             a donor episode's code, or the GT v9 code -- for M1 / M2b at seeds 0 and 1
rem
rem GRID is pool4x10 (default) or pool8x20 (plan 13.6, the first K-FAIL lever: run only on the PI's go). An 8x20 run lives in
rem %RUNSM%\<ARM>_s<SEED>_8x20, so it can never be mistaken for, or resume, a 4x10 run.
rem
rem PI RULING D-PI-YAWRATE-1: every trainer / eval call carries --kin-mode yaw and --tracks native_tracks.pt (4,572 / 4,572 train clips).
rem The repo path contains an ampersand ("Architecture & Inference"): the prior is COPIED to %RUNSM%\p10 first, so no argument string
rem below ever carries the ampersand. The eval-side defaults (grid, kin, join) are resolved inside Python, never on this command line.
setlocal EnableExtensions
if not defined REPO set "REPO=D:\tanitad\repo"
if not defined PY set "PY=D:\tanitad\venv\Scripts\python.exe"
if not defined CACHEFULL set "CACHEFULL=D:\tanitad\cache_full"
if not defined V9 set "V9=D:\tanitad\data\v9"
if not defined RUNSM set "RUNSM=D:\tanitad\runs\v8f_mini_full"
set "HJRAW=%REPO%\TanitAD Research Lab\Architecture & Inference\Research\2026-10-17-refav1-hj-mini\raw\p10_yaw4512"
set "TRAIN=%REPO%\stack\scripts\v8f_mini_train.py"
set "EVAL=%REPO%\stack\scripts\v8f_mini_eval.py"
set "PYTHONPATH=%REPO%\stack;%REPO%\taniteval"
set "TRACKS=%CACHEFULL%\native_tracks.pt"
set "P10=%RUNSM%\p10\p10_prior.pt"
set "P10PRED=%RUNSM%\p10\p10_eval_pred6.npy"
set "EV=%RUNSM%\eval"
set "A=--cache-dir %CACHEFULL% --v9-train %V9%\v9_labels_train.npz --v9-train-md5 83e30bd3791f0767707eade5f4671759 --lead-train %V9%\lead_train4572.npz --speed-train %V9%\refcv6_speed_max_v8_train.jsonl --join-train %RUNSM%\join_train.npz --p10 %P10% --kin-mode yaw --tracks %TRACKS% --exclude 540:600 --batch 64 --steps-mult 4 --device cuda"
set "EC=--v9-eval %V9%\v9_labels_eval139.npz --lead-eval %V9%\lead_eval139.npz --speed-eval %V9%\refcv6_speed_max_v8_eval.jsonl --kin-mode yaw --p10 %P10% --out-dir %EV%"
set "ER=--eval-cache %CACHEFULL% --tracks %TRACKS%"

if /i "%~1"=="gate" goto :gate
if /i "%~1"=="train" goto :d_train
if /i "%~1"=="eval" goto :d_eval
if /i "%~1"=="floors" goto :floors
if /i "%~1"=="compare" goto :compare
if /i "%~1"=="tac" goto :tac
echo usage: v8f_mini_box2_queue.cmd gate^|train ARM SEED [GRID]^|eval ARM SEED [GRID]^|floors^|compare^|tac
exit /b 2

rem train / eval are CALLED with the stage name dropped, so inside them %~1 = ARM, %~2 = SEED, %~3 = GRID
:d_train
call :train %2 %3 %4
exit /b %errorlevel%
:d_eval
call :eval %2 %3 %4
exit /b %errorlevel%

:prep
if defined V8F_DRYRUN exit /b 0
if not exist "%CACHEFULL%\train\manifest.jsonl" (echo REFUSED: %CACHEFULL% has no train manifest & exit /b 2)
if not exist "%CACHEFULL%\eval\manifest.jsonl" (echo REFUSED: %CACHEFULL% has no eval manifest & exit /b 2)
if not exist "%TRACKS%" (echo REFUSED: %TRACKS% is missing & exit /b 2)
if not exist "%RUNSM%\p10" mkdir "%RUNSM%\p10"
copy /y "%HJRAW%\p10_prior.pt" "%P10%" >nul || (echo REFUSED: the P10-yaw-4512 prior did not copy & exit /b 2)
copy /y "%HJRAW%\p10_eval_pred6.npy" "%P10PRED%" >nul || (echo REFUSED: the P10-yaw-4512 eval predictions did not copy & exit /b 2)
if not exist "%EV%" mkdir "%EV%"
exit /b 0

:gate
call :prep || exit /b 2
for %%R in (M1 M2b) do (
  if defined V8F_DRYRUN (echo "%PY%" "%TRAIN%" --arm %%R --out %RUNSM%\gate_%%R %A% --selfcheck-only) else (
    "%PY%" "%TRAIN%" --arm %%R --out "%RUNSM%\gate_%%R" %A% --selfcheck-only > "%RUNSM%\gate_%%R.log" 2>&1
    if errorlevel 1 (echo [FAIL] gate %%R - see %RUNSM%\gate_%%R.log & exit /b 1)
    echo [ok] gate %%R
  )
)
exit /b 0

:train
call :prep || exit /b 2
set "G=%~3"
if "%G%"=="" set "G=pool4x10"
set "SUF="
if /i "%G%"=="pool8x20" set "SUF=_8x20"
set "OUT=%RUNSM%\%~1_s%~2%SUF%"
if exist "%OUT%\summary.json" (echo [skip] %~1_s%~2%SUF% & exit /b 0)
if defined V8F_DRYRUN (echo "%PY%" "%TRAIN%" --arm %~1 --seed %~2 --grid %G% --out %OUT% %A% & exit /b 0)
if not exist "%OUT%" mkdir "%OUT%"
echo [%date% %time%] train %~1_s%~2%SUF%
"%PY%" "%TRAIN%" --arm %~1 --seed %~2 --grid %G% --out "%OUT%" %A% >> "%OUT%\train.log" 2>&1
if errorlevel 1 (echo [FAIL] train %~1_s%~2%SUF% - see %OUT%\train.log & exit /b 1)
if not exist "%OUT%\summary.json" (echo [FAIL] train %~1_s%~2%SUF% exited 0 without summary.json & exit /b 1)
exit /b 0

:eval
call :prep || exit /b 2
set "G=%~3"
if "%G%"=="" set "G=pool4x10"
set "SUF="
if /i "%G%"=="pool8x20" set "SUF=_8x20"
set "TAG=%~1%SUF%_s%~2"
set "OUT=%RUNSM%\%~1_s%~2%SUF%"
if defined V8F_DRYRUN (echo "%PY%" "%EVAL%" run --ckpt %OUT%\ckpt_last.pt --tag %TAG% %EC% %ER% --project --families & exit /b 0)
if not exist "%OUT%\summary.json" (echo REFUSED: %TAG% has not finished training & exit /b 2)
if exist "%EV%\eval_%TAG%.npz" (echo [skip] eval %TAG%) else (
  "%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TAG% %EC% %ER% --project --families >> "%EV%\eval_%TAG%.log" 2>&1
  if errorlevel 1 (echo [FAIL] eval %TAG% - see %EV%\eval_%TAG%.log & exit /b 1)
  echo [ok] eval %TAG%
)
if /i not "%~1"=="M1" exit /b 0
if exist "%EV%\eval_M1-navshuf%SUF%_s%~2.npz" (echo [skip] eval M1-navshuf%SUF%_s%~2 & exit /b 0)
"%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag M1-navshuf%SUF%_s%~2 --nav-shuffle %EC% %ER% >> "%EV%\eval_M1-navshuf%SUF%_s%~2.log" 2>&1
if errorlevel 1 (echo [FAIL] eval M1-navshuf%SUF%_s%~2 - see %EV%\eval_M1-navshuf%SUF%_s%~2.log & exit /b 1)
echo [ok] eval M1-navshuf%SUF%_s%~2
exit /b 0

:tac
rem PREREG_V8F_TAC (eval only, ZERO training): for M1 and M2b at seeds 0 and 1, O reads T's own code (tacpred), a DONOR window's
rem predicted code from another episode (tacshuf), or the GT v9 code where the window has a row (tacorc, a T0 oracle). Into %EV%
rem beside the banked eval_<ARM>_s<SEED>.npz, which code\v8f_tac_read.py reads for its reproducibility check. No --project,
rem no --families: the reader rebuilds the families from the predictions. Each eval is skipped once its npz exists.
call :prep || exit /b 2
for %%A in (M1 M2b) do for %%S in (0 1) do (
  call :tacone %%A %%S pred tacpred || exit /b 1
  call :tacone %%A %%S shuffle tacshuf || exit /b 1
  call :tacone %%A %%S oracle tacorc || exit /b 1
)
exit /b 0

:tacone
set "OUT=%RUNSM%\%~1_s%~2"
set "TAG=%~1_s%~2-%~4"
if defined V8F_DRYRUN (echo "%PY%" "%EVAL%" run --ckpt %OUT%\ckpt_last.pt --tag %TAG% --tac %~3 %EC% %ER% & exit /b 0)
if not exist "%OUT%\summary.json" (echo REFUSED: %~1_s%~2 has not finished training & exit /b 2)
if not exist "%EV%\eval_%~1_s%~2.npz" (echo REFUSED: the banked eval_%~1_s%~2.npz is missing from %EV% & exit /b 2)
if exist "%EV%\eval_%TAG%.npz" (echo [skip] eval %TAG% & exit /b 0)
"%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TAG% --tac %~3 %EC% %ER% >> "%EV%\eval_%TAG%.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TAG% - see %EV%\eval_%TAG%.log & exit /b 1)
echo [ok] eval %TAG%
exit /b 0

:floors
call :prep || exit /b 2
if defined V8F_DRYRUN (echo "%PY%" "%EVAL%" floors --p10-pred %P10PRED% %EC% --families & exit /b 0)
"%PY%" "%EVAL%" floors --p10-pred "%P10PRED%" %EC% --families > "%EV%\floors.log" 2>&1
if errorlevel 1 (echo [FAIL] floors - see %EV%\floors.log & exit /b 1)
echo [ok] floors
exit /b 0

:compare
call :prep || exit /b 2
set "RUNSARG=M1_s0=%EV%\eval_M1_s0.npz M1_s1=%EV%\eval_M1_s1.npz M2b_s0=%EV%\eval_M2b_s0.npz M2b_s1=%EV%\eval_M2b_s1.npz M1-navshuf_s0=%EV%\eval_M1-navshuf_s0.npz M1-navshuf_s1=%EV%\eval_M1-navshuf_s1.npz"
if defined V8F_DRYRUN (echo "%PY%" "%EVAL%" compare --runs %RUNSARG% --floors-npz %EV%\floors_perwindow.npz --out %EV%\compare.json %EC% & exit /b 0)
for %%T in (M1_s0 M1_s1 M2b_s0 M2b_s1 M1-navshuf_s0 M1-navshuf_s1) do if not exist "%EV%\eval_%%T.npz" (echo REFUSED: compare needs all six evals - eval_%%T.npz is missing & exit /b 2)
if not exist "%EV%\floors_perwindow.npz" (echo REFUSED: run floors first & exit /b 2)
"%PY%" "%EVAL%" compare --runs %RUNSARG% --floors-npz "%EV%\floors_perwindow.npz" --out "%EV%\compare.json" %EC% > "%EV%\compare.log" 2>&1
if errorlevel 1 (echo [FAIL] compare - see %EV%\compare.log & exit /b 1)
echo [ok] compare
exit /b 0
