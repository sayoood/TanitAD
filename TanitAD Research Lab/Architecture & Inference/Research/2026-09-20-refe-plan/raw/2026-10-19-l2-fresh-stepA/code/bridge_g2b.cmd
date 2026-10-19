@echo off
rem READ 2 G2b BRIDGE (identity of the bundle forward against the banked fan on 3 B tokens + a one-pixel RED arm).  ONE GPU phase, ~5 min (ESTIMATED: 2 x model load + 3 forwards each).  NOT RUN by the package.
rem It takes the dev-box GPU lock through launch_stepA_ctype.py (waits for chain_active, RAM >= 8 GB, a free GPU; never touches another owner's lock).  Launch DETACHED (WMI CreateFlags 0x08000200, "< NUL >> <log> 2>&1");
rem a Master Mind / agent decision, after READ 1's chain has released the lock.  Markers: ZZGPUWAIT -> ZZGPULOCKED -> ZZBRIDGE_IDENTITY_OK 3 -> ZZBRIDGE_RED_OK 3 -> ZZBRIDGE_PASS | ZZBRIDGE_FAIL ; then BRIDGE_DONE.
rem   usage: bridge_g2b.cmd [<bundle script, default the L1 one>]      (a different script must be passed with its own sha256 recorded in the result json)
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1
set CUDA_VISIBLE_DEVICES=
set L1_TREE=C:/Users/Admin/par18/l1run/tree_hybrid
set C=C:/Users/Admin/par18/l1run/code
set F=C:/Users/Admin/par18/l2fresh
set PYCPU=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
set PYGPU=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe
set TAB=D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/table_p2_heldout.npz
set CK=D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt
set PLAN=C:/Users/Admin/p2heldout/plan/heldout_tokens.json
set BS=%~1
if "%BS%"=="" set BS=%C%/make_stepA_bundle.py
echo BRIDGE_START %DATE% %TIME% SCRIPT=%BS%
%PYCPU% -u %C%/launch_stepA_ctype.py run --max-wait 21600 -- %PYGPU% -u %F%/code/bridge_g2b.py --ckpt %CK% --plan %PLAN% --sample %F%/raw/bridge3_sample.json --table %TAB% --out-dir %F%/bridge_run --bundle-script %BS%
echo BRIDGE_RC=%ERRORLEVEL%
echo BRIDGE_DONE %TIME%   (result: %F%/bridge_run/bridge_g2b.json)
