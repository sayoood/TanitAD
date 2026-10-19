@echo off
set L1_TREE=C:/Users/Admin/par18/l1run/tree_hybrid
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1
cd /d C:\Users\Admin\par18\l1run
echo STEP3_START %DATE% %TIME% L1_TREE=%L1_TREE%
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u C:\Users\Admin\par18\l1run\code\launch_stepA_ctype.py run --max-wait 21600 -- C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe -u C:\Users\Admin\par18\l1run\code\make_stepA_bundle.py --ckpt D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt --plan C:/Users/Admin/p2heldout/plan/heldout_tokens.json --sample D:/Projects/TanitAD-artifacts/l1_stepA_run/sample.json --table D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/table_p2_heldout.npz --out-dir D:/Projects/TanitAD-artifacts/l1_stepA_run/bundle
> C:\Users\Admin\par18\l1run\STEP3_RC echo %ERRORLEVEL%
echo STEP3_END rc=%ERRORLEVEL% %DATE% %TIME%
