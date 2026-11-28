@echo off
setlocal
set "KILL=C:\Users\Admin\mixo_kill"
set "T=%KILL%\tree"
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
set "LOG=%KILL%\out\mutation_proof_kill.out"
set "CUDA_VISIBLE_DEVICES=-1"
set "PYTHONPATH=%T%\stack;%T%\taniteval;%T%\stack\scripts"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
set "MIXO_MUT_BASE=E:\Projects\TanitAD-artifacts\mixo_kill_e\tmp"
set "TMP=E:\Projects\TanitAD-artifacts\mixo_kill_e\systmp"
set "TEMP=E:\Projects\TanitAD-artifacts\mixo_kill_e\systmp"
if not exist "E:\Projects\TanitAD-artifacts\mixo_kill_e\systmp" mkdir "E:\Projects\TanitAD-artifacts\mixo_kill_e\systmp"
echo # PREREG_V8F_MIXO mutation proof over the tip 6df75ad202b70aa22c3acadc99dfba4de94ef92f + the 14 killing tests; CPU only. cmd started %DATE% %TIME% > "%LOG%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%KILL%\wait_no_pytest.ps1" < NUL
if errorlevel 1 (echo # NOT STARTED: another pytest kept running, or C: had under 3 GB free, for 90 minutes - see wait_no_pytest.ps1 >> "%LOG%" & exit /b 3)
echo # coupled check (the two label-pin tests of test_v8f_flat / test_v8f_oacc, which load test_v8f_mixo.py as a module) started %DATE% %TIME% >> "%LOG%"
pushd "%T%"
"%PY%" -m pytest -q -p no:cacheprovider --basetemp "E:\Projects\TanitAD-artifacts\mixo_kill_e\tmp\coupled" stack/tests/test_v8f_flat.py stack/tests/test_v8f_oacc.py -k every_existing_label_is_byte_identical > "%KILL%\out\coupled_check.out" 2>&1 < NUL
echo # coupled check exit code %ERRORLEVEL% (log: out\coupled_check.out) >> "%LOG%"
popd
powershell -NoProfile -ExecutionPolicy Bypass -File "%KILL%\wait_no_pytest.ps1" < NUL
if errorlevel 1 (echo # NOT STARTED, proof: another pytest kept running, or C: had under 3 GB free, for 90 minutes >> "%LOG%" & exit /b 3)
echo # mutate_mixo.py started %DATE% %TIME% >> "%LOG%"
"%PY%" "%T%\TanitAD Research Lab\Architecture & Inference\Research\2026-10-17-v8f-mini\raw\2026-10-25-mixo\mutate_mixo.py" "%T%" "%PY%" >> "%LOG%" 2>&1 < NUL
echo # mutate_mixo.py exit code %ERRORLEVEL% >> "%LOG%"
if errorlevel 1 (pushd "%T%" & "%PY%" -m pytest -q -p no:cacheprovider --basetemp "E:\Projects\TanitAD-artifacts\mixo_kill_e\tmp\detail" stack/tests/test_v8f_mixo.py -k test_chain_mixo_syntax -vv --tb=long > "%KILL%\out\baseline_fail_detail.out" 2>&1 < NUL & popd)
echo # ended %DATE% %TIME% >> "%LOG%"
exit /b 0
