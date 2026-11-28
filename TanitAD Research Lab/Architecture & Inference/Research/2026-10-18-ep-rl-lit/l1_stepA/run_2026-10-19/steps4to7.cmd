@echo off
rem L1 Step A, steps 4-7 (all CPU). Launch DETACHED (WMI, CreateFlags 0x08000200, "< NUL >> steps4to7.log 2>&1"). Stops at the first failure and writes STEP<n>_FAIL.
rem Refuses to start unless step 3 is clean: G2b (bundle forward == banked fan, bit for bit) is a HARD gate.
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1
set W=D:/Projects/TanitAD-artifacts/l1_stepA_run
set C=C:/Users/Admin/par18/l1run/code
set PYCPU=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
set TAB=D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/table_p2_heldout.npz
set CK=D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt
set PLAN=C:/Users/Admin/p2heldout/plan/heldout_tokens.json
cd /d C:\Users\Admin\par18\l1run
findstr /C:"ZZBUNDLE_OK" step3.log >NUL || (echo G2B_GATE_NOT_PASSED: no ZZBUNDLE_OK in step3.log & exit /b 9)
if not exist "%W%/bundle/bundle.npz" (echo G2B_GATE_NOT_PASSED: no bundle.npz & exit /b 9)
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=4
echo STEP4_START %TIME%
%PYCPU% -u %C%/stepA.py perturb --bundle %W%/bundle/bundle.npz --out %W%/perts.npz || (echo STEP4_FAIL & exit /b 4)
echo STEP4_DONE %TIME%
set "PYTHONPATH=D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896;C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe"
set NUPLAN_MAPS_ROOT=D:/Archive/devbox-C/navsim/data/maps
set NUPLAN_MAP_VERSION=nuplan-maps-v1.0
set REFE_SIM_HZ=10
set OMP_NUM_THREADS=1
set PYTHONIOENCODING=utf-8
echo STEP5_START %TIME%
C:/Users/Admin/navsim-crun/venv/Scripts/python.exe -u %C%/label_perturbations.py --bundle %W%/bundle/bundle.npz --perts %W%/perts.npz --plan %PLAN% --table %TAB% --out %W%/labels.npz --g1 20 || (echo STEP5_FAIL & exit /b 5)
echo STEP5_DONE %TIME%
set PYTHONPATH=
set OMP_NUM_THREADS=4
set L1_REFE_DIR=C:/Users/Admin/par18/l1run/tree_hybrid/refe
echo STEP6_START %TIME%
%PYCPU% -u %C%/stepA.py score --bundle %W%/bundle/bundle.npz --perts %W%/perts.npz --ckpt %CK% --ctx %W%/bundle/ctx.npy --device cpu --out %W%/scores.npz || (echo STEP6_FAIL & exit /b 6)
echo STEP6_DONE %TIME%
echo STEP7_START %TIME%
%PYCPU% -u %C%/stepA.py analyse --bundle %W%/bundle/bundle.npz --perts %W%/perts.npz --scores %W%/scores.npz --labels %W%/labels.npz --out %W%/RESULT.json || (echo STEP7_FAIL & exit /b 7)
echo STEP7_DONE %TIME% ALL_DONE
