@echo off
rem L2-FRESH-STEPA chain: step 3 (GPU, waits for the dev-box lock) -> 4 perturb -> 5 label -> 6 P2 scores -> 6b EP-head-token export -> merge -> 6c EP scores of P2 + both final L2 heads (+ C-FEAT).
rem STOPS BEFORE THE READ: analyse_fresh.py (step 7) is SEALED-ONLY and is never called here.  Launch DETACHED (WMI, CreateFlags 0x08000200, "< NUL >> <log> 2>&1").
rem   usage: fresh_chain.cmd <SAMPLE.json> <W dir, forward slashes> <TAG>      e.g.  ...fresh_sample_token_disjoint.json D:/Projects/TanitAD-artifacts/l2_fresh_stepA B
rem Markers (stdout): ZZGPUWAIT -> ZZGPULOCKED -> ZZBUNDLE_OK|ZZBUNDLE_FAIL -> ZZGPURELEASED ; STEP4_DONE STEP5_DONE STEP6_DONE EXPORT_DONE MERGE_DONE SCORES_DONE ; CHAIN_DONE_NO_READ.  Failure: STEP<n>_FAIL / G2B_GATE_NOT_PASSED (exit 9).
set SAMPLE=%~1
set W=%~2
set TAG=%~3
set "WB=%W:/=\%"
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1
set C=C:/Users/Admin/par18/l1run/code
set F=C:/Users/Admin/par18/l2fresh/code
set L2=C:/Users/Admin/par18/l2/code
set PYCPU=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
set PYGPU=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe
set TAB=D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/table_p2_heldout.npz
set CK=D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt
set PLAN=C:/Users/Admin/p2heldout/plan/heldout_tokens.json
set HEADS=D:/Projects/TanitAD-artifacts/l2_fresh_stepA/heads
if not exist "%SAMPLE%" (echo NO_SAMPLE %SAMPLE% & exit /b 9)
for /f "tokens=1,2" %%a in ('%PYCPU% %F%/nsplit.py "%SAMPLE%"') do (set N=%%a& set H=%%b)
if "%N%"=="" (echo NO_N & exit /b 9)
if "%N%"=="0" (echo EMPTY_SAMPLE_REFUSED & exit /b 9)
mkdir "%WB%" 2>NUL
echo CHAIN_START %DATE% %TIME% TAG=%TAG% N=%N% SAMPLE=%SAMPLE% W=%W%
rem ---------------- step 3: the ONE GPU forward, under the dev-box lock (the launcher takes it, waits for it, never touches another owner's lock)
set L1_TREE=C:/Users/Admin/par18/l1run/tree_hybrid
set CUDA_VISIBLE_DEVICES=
%PYCPU% -u %C%/launch_stepA_ctype.py run --max-wait 21600 -- %PYGPU% -u %C%/make_stepA_bundle.py --ckpt %CK% --plan %PLAN% --sample "%SAMPLE%" --table %TAB% --out-dir %W%/bundle > "%WB%\step3.log" 2>&1
set RC3=%ERRORLEVEL%
echo STEP3_RC=%RC3%   (step 3 log: %WB%\step3.log; markers ZZGPUWAIT ZZGPULOCKED ZZBUNDLE_OK ZZGPURELEASED)
if not "%RC3%"=="0" (echo STEP3_FAIL rc=%RC3% & exit /b 3)
rem G2b HARD GATE: ZZBUNDLE_OK is printed by make_stepA_bundle.py only when the forward reproduced the banked fan, logits and pick bit for bit on every token
if not exist "%WB%\bundle\bundle.npz" (echo G2B_GATE_NOT_PASSED: no bundle.npz & exit /b 9)
findstr /C:"ZZBUNDLE_OK" "%WB%\step3.log" >NUL || (echo G2B_GATE_NOT_PASSED: no ZZBUNDLE_OK in step3.log & exit /b 9)
echo STEP3_DONE %TIME%
rem ---------------- steps 4-6: CPU
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=4
echo STEP4_START %TIME%
%PYCPU% -u %C%/stepA.py perturb --bundle %W%/bundle/bundle.npz --out %W%/perts.npz || (echo STEP4_FAIL & exit /b 4)
echo STEP4_DONE %TIME%
%PYCPU% -u %F%/ram_gate.py --min-gb 8 --max-wait 7200 || (echo STEP5_RAM_GATE_FAIL & exit /b 5)
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
rem ---------------- 6b: the EP head's INPUT tokens (L2's own exporter, 2 workers = the 2-process limit), then merge
set OMP_NUM_THREADS=4
echo EXPORT_START %TIME%
start "" /B cmd /c "%PYCPU% -u %L2%/export_feats.py --bundle %W%/bundle/bundle.npz --perts %W%/perts.npz --ctx %W%/bundle/ctx.npy --out-dir %W%/shards --start %H% --end %N% > %WB%\export_w2.log 2>&1 & echo EXPORT_W2_DONE >> %WB%\export_w2.log"
%PYCPU% -u %L2%/export_feats.py --bundle %W%/bundle/bundle.npz --perts %W%/perts.npz --ctx %W%/bundle/ctx.npy --out-dir %W%/shards --start 0 --end %H% || (echo EXPORT_FAIL w1 & exit /b 7)
:waitw2
findstr /C:"EXPORT_W2_DONE" "%WB%\export_w2.log" >NUL && goto w2done
findstr /C:"Traceback" "%WB%\export_w2.log" >NUL && (echo EXPORT_FAIL w2 & exit /b 7)
ping -n 21 127.0.0.1 >NUL
goto waitw2
:w2done
findstr /C:"ZZEXPORT_DONE" "%WB%\export_w2.log" >NUL || (echo EXPORT_FAIL w2 no ZZEXPORT_DONE & exit /b 7)
echo EXPORT_DONE %TIME%
%PYCPU% -u %L2%/merge_feats.py --shards %W%/shards --out-dir %W%/feats --run-dir %W% --n %N% || (echo MERGE_FAIL & exit /b 8)
echo MERGE_DONE %TIME%
%PYCPU% -u %F%/score_fresh.py --run-dir %W% --heads %HEADS% || (echo SCORES_FAIL_or_CFEAT_FAIL & exit /b 3)
echo SCORES_DONE %TIME%
echo CHAIN_DONE_NO_READ %TIME%   (step 7 = analyse_fresh.py is SEALED-ONLY)
