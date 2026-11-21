@echo off
rem Rung-beta gate on Box 2 (PREREG_V8F_BETA Amendment 1): EVAL ONLY, no training. It reads the v8F-mini M2b checkpoints that
rem PREREG_V8F_MINI Amendment 4 trains on cache_full with the RA-1 holdout withheld (%RUNSM%\M2b_s0, M2b_s1).
rem
rem   v8f_beta_box2_queue.cmd run SEED     v8f_mini_eval.py run --plan beta on M2b_s<SEED>: M2 itself, beta (the WM ranks the 117-candidate
rem                                        fan), the shuffled-goal control and the oracle-goal diagnostic, on the same 2,399 windows;
rem                                        skipped once its JSON exists
rem   v8f_beta_box2_queue.cmd compare      compare-beta over the two seeds: the verdict and the named next lever
rem
rem The fan is the pinned 13 x 9 grid (no --fan-artifact): Amendment 1 banks its check against refcv4b's trained bank offline.
rem PI RULING D-PI-YAWRATE-1: --kin-mode yaw and cache_full's native_tracks.pt on every call. Paths never carry the repo's ampersand.
setlocal EnableExtensions
if not defined REPO set "REPO=D:\tanitad\repo"
if not defined PY set "PY=D:\tanitad\venv\Scripts\python.exe"
if not defined CACHEFULL set "CACHEFULL=D:\tanitad\cache_full"
if not defined V9 set "V9=D:\tanitad\data\v9"
if not defined RUNSM set "RUNSM=D:\tanitad\runs\v8f_mini_full"
set "EVAL=%REPO%\stack\scripts\v8f_mini_eval.py"
set "PYTHONPATH=%REPO%\stack;%REPO%\taniteval;%REPO%\stack\scripts"
set "TRACKS=%CACHEFULL%\native_tracks.pt"
set "P10=%RUNSM%\p10\p10_prior.pt"
set "EVB=%RUNSM%\eval_beta"
set "EC=--v9-eval %V9%\v9_labels_eval139.npz --lead-eval %V9%\lead_eval139.npz --speed-eval %V9%\refcv6_speed_max_v8_eval.jsonl --kin-mode yaw --p10 %P10% --out-dir %EVB% --n-boot 2000"

if /i "%~1"=="run" goto :d_run
if /i "%~1"=="compare" goto :compare
echo usage: v8f_beta_box2_queue.cmd run SEED^|compare
exit /b 2

rem run is CALLED with the stage name dropped, so inside it %~1 = SEED
:d_run
call :run %2
exit /b %errorlevel%

:prep
if defined V8F_DRYRUN exit /b 0
if not exist "%TRACKS%" (echo REFUSED: %TRACKS% is missing & exit /b 2)
if not exist "%P10%" (echo REFUSED: %P10% is missing - the v8F-mini queue copies it & exit /b 2)
if not exist "%EVB%" mkdir "%EVB%"
exit /b 0

:run
call :prep || exit /b 2
set "CK=%RUNSM%\M2b_s%~1\ckpt_last.pt"
if defined V8F_DRYRUN (echo "%PY%" "%EVAL%" run --ckpt %CK% --tag M2b_s%~1 --plan beta --families --eval-cache %CACHEFULL% --tracks %TRACKS% %EC% & exit /b 0)
if not exist "%RUNSM%\M2b_s%~1\summary.json" (echo REFUSED: M2b_s%~1 has not finished training & exit /b 2)
if exist "%EVB%\eval_M2b_s%~1.json" (echo [skip] beta M2b_s%~1 & exit /b 0)
echo [%date% %time%] beta M2b_s%~1
"%PY%" "%EVAL%" run --ckpt "%CK%" --tag M2b_s%~1 --plan beta --families --eval-cache %CACHEFULL% --tracks %TRACKS% %EC% >> "%EVB%\eval_M2b_s%~1.log" 2>&1
if errorlevel 1 (echo [FAIL] beta M2b_s%~1 - see %EVB%\eval_M2b_s%~1.log & exit /b 1)
if not exist "%EVB%\eval_M2b_s%~1.json" (echo [FAIL] beta M2b_s%~1 exited 0 without its JSON & exit /b 1)
echo [ok] beta M2b_s%~1
exit /b 0

:compare
call :prep || exit /b 2
if defined V8F_DRYRUN (echo "%PY%" "%EVAL%" compare-beta --runs M2b_s0=%EVB%\eval_M2b_s0.json M2b_s1=%EVB%\eval_M2b_s1.json --out %EVB%\verdict_M2b.json %EC% & exit /b 0)
for %%S in (0 1) do if not exist "%EVB%\eval_M2b_s%%S.json" (echo REFUSED: compare-beta needs both seeds - eval_M2b_s%%S.json is missing & exit /b 2)
"%PY%" "%EVAL%" compare-beta --runs M2b_s0=%EVB%\eval_M2b_s0.json M2b_s1=%EVB%\eval_M2b_s1.json --out "%EVB%\verdict_M2b.json" %EC% > "%EVB%\compare_beta.log" 2>&1
if errorlevel 1 (echo [FAIL] compare-beta - see %EVB%\compare_beta.log & exit /b 1)
echo [ok] compare-beta
exit /b 0
