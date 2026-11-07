@echo off
rem The three evals, each the invocation of record (queue.cmd :eval, tip 6e5d3768 chain_v8f_mini): run --ckpt .. --tag .. %EC% %ER% --project --families
rem   identity: banked M2b_s0 on box 2's ORIGINAL eval cache and tracks; B-prime s0 / s1: the pod eval cache. Same code, same inputs otherwise. Each skipped once its npz exists.
setlocal EnableExtensions
call C:\lorapod_eval\env_bp.cmd
set "EVAL=C:\lorapod_eval\code\stack\scripts\v8f_mini_eval.py"
set "V9=D:\tanitad\data\v9"
set "ORIG=D:\tanitad\cache_full"
set "OUT=C:\lorapod_eval\out"
set "EC=--v9-eval %V9%\v9_labels_eval139.npz --lead-eval %V9%\lead_eval139.npz --speed-eval %V9%\refcv6_speed_max_v8_eval.jsonl --kin-mode yaw --p10 D:\tanitad\runs\v8f_mini_full\p10\p10_prior.pt --out-dir %OUT%"
set "TRK_POD=C:\lorapod_eval\cache_pod\native_tracks.pt"
if not exist "%TRK_POD%" set "TRK_POD=%ORIG%\native_tracks.pt"
if not exist "%OUT%" mkdir "%OUT%"
call :one M2b_s0-idcheck D:\tanitad\runs\v8f_mini_full\M2b_s0\ckpt_last.pt %ORIG% %ORIG%\native_tracks.pt || exit /b 1
call :one M2bp_s0 C:\lorapod_eval\ckpt\M2bp_s0\ckpt_last.pt C:\lorapod_eval\cache_pod "%TRK_POD%" || exit /b 1
call :one M2bp_s1 C:\lorapod_eval\ckpt\M2bp_s1\ckpt_last.pt C:\lorapod_eval\cache_pod "%TRK_POD%" || exit /b 1
echo [ok] all three evals
exit /b 0

:one
if exist "%OUT%\eval_%~1.npz" (echo [skip] eval %~1 & exit /b 0)
echo [%date% %time%] eval %~1
"%PY%" "%EVAL%" run --ckpt "%~2" --tag %~1 %EC% --eval-cache "%~3" --tracks "%~4" --project --families >> "%OUT%\eval_%~1.log" 2>&1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (echo [FAIL] eval %~1 rc=%RC% - see %OUT%\eval_%~1.log & exit /b 1)
if not exist "%OUT%\eval_%~1.npz" (echo [FAIL] eval %~1 exited 0 without its npz & exit /b 1)
echo [ok] eval %~1
exit /b 0
