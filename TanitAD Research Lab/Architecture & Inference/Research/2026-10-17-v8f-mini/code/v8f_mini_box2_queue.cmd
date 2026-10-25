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
rem   v8f_mini_box2_queue.cmd tacgrp            PREREG_V8F_TAC Amendment 1 (eval only): the per-group oracle -- GT in ONE
rem                                             part of T's code (6 groups + 2 composites) -- for M1 / M2b at seeds 0 and 1
rem   v8f_mini_box2_queue.cmd tacgrp2           PREREG_V8F_TAC Amendment 2 (eval only): lonspeed+londec, the six leave-one-out
rem                                             oracles (all-<g>) and the constraints dose-response (alpha 0.25 / 0.5 / 0.75)
rem   v8f_mini_box2_queue.cmd tacabs            PREREG_V8F_CZ stage 1 (eval only): T's own code with the GT-ABSENT constraint
rem                                             slots set to 0 (the value O is teacher-forced on), M1 / M2b at seeds 0 and 1
rem   v8f_mini_box2_queue.cmd gatearm ARM       the in-process launch gate (G-DVB + G-LIVE + admissibility) for ONE arm
rem   v8f_mini_box2_queue.cmd tacabsarm ARM SEED  the absent-slot oracle on ONE trained arm (PREREG_V8F_CZ's mechanism check)
rem   v8f_mini_box2_queue.cmd gatewc            PREREG_V8F_WC: the in-process launch gate for M2b-wc = M2b with ONE lever moved,
rem                                             T's constraint-loss weight --w-cons %WC% (M2b: the default 0.5), no training
rem   v8f_mini_box2_queue.cmd trainwc SEED      PREREG_V8F_WC: one M2b-wc run into %RUNSM%\M2b-wc_s<SEED> (resumes; skipped once done)
rem   v8f_mini_box2_queue.cmd evalwc SEED       PREREG_V8F_WC: that run's eval-141 (--project --families), its absent-slot oracle
rem                                             (T's own present-slot values) and its GT oracle (the GT code); each skipped once its npz exists
rem   v8f_mini_box2_queue.cmd gatepgt           PREREG_V8F_PGT: the in-process launch gate for M2b-pgt = M2b with ONE lever moved, the scheduled-sampling
rem                                             schedule --p-gt-start %PGTS% --p-gt-end %PGTE% (M2b: 1.0 -> 0.2), no training
rem   v8f_mini_box2_queue.cmd trainpgt SEED     PREREG_V8F_PGT: one M2b-pgt run into %RUNSM%\M2b-pgt_s<SEED> (resumes; skipped once done)
rem   v8f_mini_box2_queue.cmd evalpgt SEED      PREREG_V8F_PGT: that run's eval-141 (--project --families), its absent-slot oracle (absmask), its GT-code
rem                                             oracle (tacorc) and its all-latcons oracle (GT everywhere but T's latcons); each skipped once its npz exists
rem   v8f_mini_box2_queue.cmd gategrid BASE        PREREG_V8F_GRID: the in-process launch gate for G = BASE + --grid pool8x20, no training. BASE is M2b or M2b-pgt (the
rem                                             base is decided by PREREG_V8F_PGT's reads, not here): M2b-pgt carries the PGT dose (PGTS / PGTE, defined ONCE below)
rem   v8f_mini_box2_queue.cmd traingrid SEED BASE  PREREG_V8F_GRID: one G run into %RUNSM%\<BASE>_s<SEED>_8x20 (resumes; skipped once done)
rem   v8f_mini_box2_queue.cmd evalgrid SEED BASE   PREREG_V8F_GRID: that run's eval-141 (--project --families), its absmask eval, its GT-code oracle and its all-latcons oracle,
rem                                             tagged <BASE>-8x20_s<SEED>; each skipped once its npz exists
rem   v8f_mini_box2_queue.cmd gatehist              PREREG_V8F_HIST: the in-process launch gate for H = M2b + --hist 10 --window-start 3 (Amendment 1), no training. B is M2b (fixed in the PREREG): there is no BASE argument
rem   v8f_mini_box2_queue.cmd trainhist SEED       PREREG_V8F_HIST: one H run into %RUNSM%\M2b_s<SEED>_h10 (resumes; skipped once done)
rem   v8f_mini_box2_queue.cmd evalhist SEED        PREREG_V8F_HIST: that run's eval-141 (--project --families), its absmask eval, its GT-code oracle and its all-latcons oracle,
rem                                             tagged M2b-h10_s<SEED>; each skipped once its npz exists
rem   v8f_mini_box2_queue.cmd gatedscale           PREREG_V8F_DSCALE: the in-process launch gate for Q = M2b + --data-frac 0.5 (the first half of B's clips in sha12 order), no training. B is M2b: no BASE argument
rem   v8f_mini_box2_queue.cmd traindscale SEED    PREREG_V8F_DSCALE: one Q run into %RUNSM%\M2b_s<SEED>_df50 (resumes; skipped once done). Seed 0 only is pre-registered
rem   v8f_mini_box2_queue.cmd evaldscale SEED     PREREG_V8F_DSCALE: that run's eval-141 (--project --families), its absmask eval, its GT-code oracle and its all-latcons oracle,
rem                                             tagged M2b-df50_s<SEED>; each skipped once its npz exists
rem   v8f_mini_box2_queue.cmd identitylora    PREREG_V8F_LORA Amendment 2: the SAME-DEVICE identity control. v8f_lora_identity_dev.py encodes the declared train and eval clips' frames (production route, shipped to
rem                                             %LORAFRAMES%) with the arm's encoder at B = 0 in the builder's pure bf16 on THIS box's cuda device and compares the pooled fp8 tokens with cache_full's STORED ones: PASS iff
rem                                             rel-L2 <= 1e-3 on every clip. Then the same with --mutant (one preprocessing constant changed), which must FAIL. Both write a json under %RUNSM%\lora_ar2\identity;
rem                                             the stage exits 0 once both readings exist (the CHAIN judges them) and 1 if either driver crashed or refused to write one. Skipped once both jsons exist
rem   v8f_mini_box2_queue.cmd smokelora       PREREG_V8F_LORA proof 4 (AR-2): v8f_lora_smoke.py, 200 real-data steps at batch 1 through the trainer itself (--encoder lora --enc-ckpt, bf16 weights, --amp OFF) on windows drawn ONLY
rem                                             from the declared train clips (--max-clips; frames shipped for exactly those), into %RUNSM%\lora_ar2\smoke. Exits 0 once smoke_report.json exists: the report carries its own
rem                                             verdict, step time, peak memory and the projected hours of one 19,164-step seed (ESTIMATED), and the CHAIN reads it
rem   v8f_mini_box2_queue.cmd kind    PREREG_V8F_KIND (eval only, ZERO training): which KIND of T's constraint cells is worth ADE6 to O. Per seed of the banked M2b (0 and 1), 15 evals into %EV%, tagged
rem                                             M2b_s<SEED>-kind-<name>: own (T's own code, the standard eval: identity control 1), all (the full oracle: identity control 2), only-<k> and loo-<k> (GT in kind k's cells only /
rem                                             everywhere EXCEPT kind k) for the six kinds speed angle tte timegap geom leadgap, and union (GT in all 26 constraint cells). Read by code8f_kind_read.py. Skipped once its npz exists
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
rem PREREG_V8F_WC: the ONE lever M2b-wc moves, written once so the gate, the runs and the dry-run test read one value
set "WC=4.0"
rem PREREG_V8F_PGT: the ONE lever M2b-pgt moves (two flags, one dose), written once so the gate, the runs, the chain and the tests read one value
set "PGTS=0.5"
set "PGTE=0.0"

if /i "%~1"=="gate" goto :gate
if /i "%~1"=="train" goto :d_train
if /i "%~1"=="eval" goto :d_eval
if /i "%~1"=="floors" goto :floors
if /i "%~1"=="compare" goto :compare
if /i "%~1"=="tac" goto :tac
if /i "%~1"=="tacgrp" goto :tacgrp
if /i "%~1"=="tacgrp2" goto :tacgrp2
if /i "%~1"=="tacabs" goto :tacabs
if /i "%~1"=="gatearm" goto :gatearm
if /i "%~1"=="tacabsarm" goto :tacabsarm
if /i "%~1"=="gatewc" goto :gatewc
if /i "%~1"=="trainwc" goto :trainwc
if /i "%~1"=="evalwc" goto :evalwc
if /i "%~1"=="gatepgt" goto :gatepgt
if /i "%~1"=="trainpgt" goto :trainpgt
if /i "%~1"=="evalpgt" goto :evalpgt
if /i "%~1"=="gategrid" goto :gategrid
if /i "%~1"=="traingrid" goto :traingrid
if /i "%~1"=="evalgrid" goto :evalgrid
if /i "%~1"=="gatehist" goto :gatehist
if /i "%~1"=="trainhist" goto :trainhist
if /i "%~1"=="evalhist" goto :evalhist
if /i "%~1"=="gatedscale" goto :gatedscale
if /i "%~1"=="traindscale" goto :traindscale
if /i "%~1"=="evaldscale" goto :evaldscale
if /i "%~1"=="identitylora" goto :identitylora
if /i "%~1"=="smokelora" goto :smokelora
if /i "%~1"=="kind" goto :kind
echo usage: v8f_mini_box2_queue.cmd gate^|train ARM SEED [GRID]^|eval ARM SEED [GRID]^|floors^|compare^|tac^|tacgrp^|tacgrp2^|kind^|tacabs^|gatearm ARM^|tacabsarm ARM SEED^|gatewc^|trainwc SEED^|evalwc SEED^|gatepgt^|trainpgt SEED^|evalpgt SEED^|gategrid BASE^|traingrid SEED BASE^|evalgrid SEED BASE^|gatehist^|trainhist SEED^|evalhist SEED^|gatedscale^|traindscale SEED^|evaldscale SEED^|identitylora^|smokelora
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

:tacgrp
rem PREREG_V8F_TAC Amendment 1 (eval only, ZERO training): O reads the GT v9 code in ONE group of T's code and T's own code elsewhere,
rem for the six groups that partition the code and the two composites (goal latdec londec latcons lonspeed stoplead decisions constraints). Into %EV% beside
rem PREREG_V8F_TAC's tacpred / tacorc runs, which code\v8f_tac_groups_read.py reads with them. Skipped once its npz exists.
call :prep || exit /b 2
for %%A in (M1 M2b) do for %%S in (0 1) do for %%G in (goal latdec londec latcons lonspeed stoplead decisions constraints) do call :tacgone %%A %%S %%G || exit /b 1
exit /b 0

:tacgone
set "OUT=%RUNSM%\%~1_s%~2"
set "TAG=%~1_s%~2-tacorc-%~3"
if defined V8F_DRYRUN (echo "%PY%" "%EVAL%" run --ckpt %OUT%\ckpt_last.pt --tag %TAG% --tac oracle --tac-groups %~3 %EC% %ER% & exit /b 0)
if not exist "%OUT%\summary.json" (echo REFUSED: %~1_s%~2 has not finished training & exit /b 2)
if not exist "%EV%\eval_%~1_s%~2-tacpred.npz" (echo REFUSED: PREREG_V8F_TAC's eval_%~1_s%~2-tacpred.npz is missing from %EV% & exit /b 2)
if exist "%EV%\eval_%TAG%.npz" (echo [skip] eval %TAG% & exit /b 0)
"%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TAG% --tac oracle --tac-groups %~3 %EC% %ER% >> "%EV%\eval_%TAG%.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TAG% - see %EV%\eval_%TAG%.log & exit /b 1)
echo [ok] eval %TAG%
exit /b 0

:tacgrp2
rem PREREG_V8F_TAC Amendment 2 (eval only, ZERO training), beside Amendment 1's runs in %EV%: (A) the joint oracle lonspeed+londec that
rem Amendment 1's MIXED reading committed to; (B) leave-one-out, GT everywhere EXCEPT one group (all-<g>); (C) the constraints moved only a
rem fraction alpha of the way from T's code to the GT. Read by code\v8f_tac_groups2_read.py. Skipped once its npz exists.
call :prep || exit /b 2
for %%A in (M1 M2b) do for %%S in (0 1) do for %%G in (lonspeed+londec all-goal all-latdec all-londec all-latcons all-lonspeed all-stoplead) do call :tacgtwo %%A %%S %%G %%G 1 || exit /b 1
for %%A in (M1 M2b) do for %%S in (0 1) do for %%L in (25 50 75) do call :tacgtwo %%A %%S constraints constraints-a%%L 0.%%L || exit /b 1
exit /b 0

:tacgtwo
rem args: ARM SEED GROUPSPEC TAGSUFFIX ALPHA
set "OUT=%RUNSM%\%~1_s%~2"
set "TAG=%~1_s%~2-tacorc-%~4"
if defined V8F_DRYRUN (echo "%PY%" "%EVAL%" run --ckpt %OUT%\ckpt_last.pt --tag %TAG% --tac oracle --tac-groups %~3 --tac-alpha %~5 %EC% %ER% & exit /b 0)
if not exist "%OUT%\summary.json" (echo REFUSED: %~1_s%~2 has not finished training & exit /b 2)
if not exist "%EV%\eval_%~1_s%~2-tacpred.npz" (echo REFUSED: PREREG_V8F_TAC's eval_%~1_s%~2-tacpred.npz is missing from %EV% & exit /b 2)
if not exist "%EV%\eval_%~1_s%~2-tacorc-constraints.npz" (echo REFUSED: Amendment 1's eval_%~1_s%~2-tacorc-constraints.npz is missing from %EV% & exit /b 2)
if exist "%EV%\eval_%TAG%.npz" (echo [skip] eval %TAG% & exit /b 0)
"%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TAG% --tac oracle --tac-groups %~3 --tac-alpha %~5 %EC% %ER% >> "%EV%\eval_%TAG%.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TAG% - see %EV%\eval_%TAG%.log & exit /b 1)
echo [ok] eval %TAG%
exit /b 0

:tacabs
rem PREREG_V8F_CZ stage 1 (eval only, ZERO training): O reads T's own code with every constraint slot the GT marks ABSENT set to 0.
rem Read by code\v8f_cz_gate_read.py beside PREREG_V8F_TAC's tacpred / tacorc runs. Skipped once its npz exists.
call :prep || exit /b 2
for %%A in (M1 M2b) do for %%S in (0 1) do call :tacone %%A %%S absmask tacabs || exit /b 1
exit /b 0

:tacabsarm
call :prep || exit /b 2
if "%~3"=="" (echo REFUSED: tacabsarm needs ARM SEED & exit /b 2)
call :tacone %~2 %~3 absmask tacabs
exit /b %errorlevel%

:gatewc
rem PREREG_V8F_WC: M2b's levers with --w-cons %WC%. The trainer's G-DVB checks the built w_cons against this argv and selfcheck.json
rem records it (levers.w.w_cons); the chain asserts both before any training.
call :prep || exit /b 2
rem ONE command string: the dry run prints exactly what the real run executes
set "GW="%PY%" "%TRAIN%" --arm M2b --w-cons %WC% --out "%RUNSM%\gate_M2b-wc" %A% --selfcheck-only"
if defined V8F_DRYRUN (echo %GW% & exit /b 0)
%GW% > "%RUNSM%\gate_M2b-wc.log" 2>&1
if errorlevel 1 (echo [FAIL] gate M2b-wc - see %RUNSM%\gate_M2b-wc.log & exit /b 1)
echo [ok] gate M2b-wc
exit /b 0

:trainwc
rem PREREG_V8F_WC: one M2b-wc run. Its summary.json says arm M2b (the trainer's arm table is unchanged); its argv.json carries
rem --w-cons %WC% and its levers.w.w_cons the built value: code\v8f_wc_read.py refuses a run where either is not %WC%.
call :prep || exit /b 2
if "%~2"=="" (echo REFUSED: trainwc needs a SEED & exit /b 2)
set "OUT=%RUNSM%\M2b-wc_s%~2"
if exist "%OUT%\summary.json" (echo [skip] M2b-wc_s%~2 & exit /b 0)
rem ONE command string: the dry run prints exactly what the real run executes
set "TW="%PY%" "%TRAIN%" --arm M2b --w-cons %WC% --seed %~2 --grid pool4x10 --out "%OUT%" %A%"
if defined V8F_DRYRUN (echo %TW% & exit /b 0)
if not exist "%OUT%" mkdir "%OUT%"
echo [%date% %time%] train M2b-wc_s%~2
%TW% >> "%OUT%\train.log" 2>&1
if errorlevel 1 (echo [FAIL] train M2b-wc_s%~2 - see %OUT%\train.log & exit /b 1)
if not exist "%OUT%\summary.json" (echo [FAIL] train M2b-wc_s%~2 exited 0 without summary.json & exit /b 1)
exit /b 0

:evalwc
rem PREREG_V8F_WC: the standard eval of M2b-wc_s<SEED>, then absmask (its present slots are T's own values) and the GT oracle (its code
rem is the GT): code\v8f_wc_read.py reads T's present-slot error from the pair, exactly as for the banked M2b runs.
call :prep || exit /b 2
if "%~2"=="" (echo REFUSED: evalwc needs a SEED & exit /b 2)
call :eval M2b-wc %~2 || exit /b 1
call :tacone M2b-wc %~2 absmask tacabs || exit /b 1
call :tacone M2b-wc %~2 oracle tacorc
exit /b %errorlevel%

:gatepgt
rem PREREG_V8F_PGT: M2b's levers with --p-gt-start %PGTS% --p-gt-end %PGTE%. The trainer's G-DVB checks the built p_gt_start / p_gt_end against this
rem argv and selfcheck.json records them (levers.p_gt_start / p_gt_end); the chain asserts both before any training.
call :prep || exit /b 2
rem ONE command string: the dry run prints exactly what the real run executes
set "GP="%PY%" "%TRAIN%" --arm M2b --p-gt-start %PGTS% --p-gt-end %PGTE% --out "%RUNSM%\gate_M2b-pgt" %A% --selfcheck-only"
if defined V8F_DRYRUN (echo %GP% & exit /b 0)
%GP% > "%RUNSM%\gate_M2b-pgt.log" 2>&1
if errorlevel 1 (echo [FAIL] gate M2b-pgt - see %RUNSM%\gate_M2b-pgt.log & exit /b 1)
echo [ok] gate M2b-pgt
exit /b 0

:trainpgt
rem PREREG_V8F_PGT: one M2b-pgt run. Its summary.json says arm M2b (the trainer's arm table is unchanged); its argv.json carries --p-gt-start %PGTS%
rem --p-gt-end %PGTE%, its levers the BUILT schedule and its train_log.jsonl the p_gt the loop used: code\v8f_pgt_read.py refuses a run where any is not.
call :prep || exit /b 2
if "%~2"=="" (echo REFUSED: trainpgt needs a SEED & exit /b 2)
set "OUT=%RUNSM%\M2b-pgt_s%~2"
if exist "%OUT%\summary.json" (echo [skip] M2b-pgt_s%~2 & exit /b 0)
rem ONE command string: the dry run prints exactly what the real run executes
set "TP="%PY%" "%TRAIN%" --arm M2b --p-gt-start %PGTS% --p-gt-end %PGTE% --seed %~2 --grid pool4x10 --out "%OUT%" %A%"
if defined V8F_DRYRUN (echo %TP% & exit /b 0)
if not exist "%OUT%" mkdir "%OUT%"
echo [%date% %time%] train M2b-pgt_s%~2
%TP% >> "%OUT%\train.log" 2>&1
if errorlevel 1 (echo [FAIL] train M2b-pgt_s%~2 - see %OUT%\train.log & exit /b 1)
if not exist "%OUT%\summary.json" (echo [FAIL] train M2b-pgt_s%~2 exited 0 without summary.json & exit /b 1)
exit /b 0

:evalpgt
rem PREREG_V8F_PGT: the four evals its reader needs, EACH ONE command string that the dry run prints and the real run executes:
rem   E0 the standard eval (--project --families)   E1 absmask (T's own present slots)   E2 the GT-code oracle (tacorc)
rem   E3 the all-latcons oracle (GT everywhere but T's latcons, T's own there; the same flags as :tacgtwo's all-<g> runs, but guarded on THIS arm's own
rem      eval rather than on PREREG_V8F_TAC's tacpred / constraints runs, which an M2b-pgt run does not have). Each is skipped once its npz exists.
rem   FLAT on purpose (no parenthesised else-block): MEASURED, an `exit /b 1` inside a multi-line block returns rc 0 from `cmd /c queue.cmd`, so a failed eval
rem   would read as a finished job; a single-line `if errorlevel 1 (... & exit /b 1)` at the top level returns 1. tests pin it (real stages, fake eval).
call :prep || exit /b 2
if "%~2"=="" (echo REFUSED: evalpgt needs a SEED & exit /b 2)
set "OUT=%RUNSM%\M2b-pgt_s%~2"
set "TG=M2b-pgt_s%~2"
set "E0="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG% %EC% %ER% --project --families"
set "E1="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacabs --tac absmask %EC% %ER%"
set "E2="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacorc --tac oracle %EC% %ER%"
set "E3="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacorc-all-latcons --tac oracle --tac-groups all-latcons --tac-alpha 1 %EC% %ER%"
if defined V8F_DRYRUN (echo %E0% & echo %E1% & echo %E2% & echo %E3% & exit /b 0)
if not exist "%OUT%\summary.json" (echo REFUSED: %TG% has not finished training & exit /b 2)
if exist "%EV%\eval_%TG%.npz" (echo [skip] eval %TG% & goto :evalpgt_1)
%E0% >> "%EV%\eval_%TG%.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG% - see %EV%\eval_%TG%.log & exit /b 1)
echo [ok] eval %TG%
:evalpgt_1
if exist "%EV%\eval_%TG%-tacabs.npz" (echo [skip] eval %TG%-tacabs & goto :evalpgt_2)
%E1% >> "%EV%\eval_%TG%-tacabs.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacabs - see %EV%\eval_%TG%-tacabs.log & exit /b 1)
echo [ok] eval %TG%-tacabs
:evalpgt_2
if exist "%EV%\eval_%TG%-tacorc.npz" (echo [skip] eval %TG%-tacorc & goto :evalpgt_3)
%E2% >> "%EV%\eval_%TG%-tacorc.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacorc - see %EV%\eval_%TG%-tacorc.log & exit /b 1)
echo [ok] eval %TG%-tacorc
:evalpgt_3
if exist "%EV%\eval_%TG%-tacorc-all-latcons.npz" (echo [skip] eval %TG%-tacorc-all-latcons & goto :evalpgt_4)
%E3% >> "%EV%\eval_%TG%-tacorc-all-latcons.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacorc-all-latcons - see %EV%\eval_%TG%-tacorc-all-latcons.log & exit /b 1)
echo [ok] eval %TG%-tacorc-all-latcons
:evalpgt_4
exit /b 0

:gridbase
rem PREREG_V8F_GRID: the base B of the arm, passed as BASE. M2b-pgt carries the PGT dose (PGTS / PGTE, defined once above); M2b carries no schedule flag. Sets GBASE and GLEV (the base's
rem lever flags WITH A LEADING SPACE, or empty), so ONE command string serves both bases. Any other BASE (or none) is refused with rc 2.
set "GBASE=%~1"
set "GLEV="
if /i "%GBASE%"=="M2b-pgt" set "GLEV= --p-gt-start %PGTS% --p-gt-end %PGTE%"
if /i not "%GBASE%"=="M2b" if /i not "%GBASE%"=="M2b-pgt" (echo REFUSED: BASE must be M2b or M2b-pgt, got "%GBASE%" & exit /b 2)
exit /b 0

:gategrid
rem PREREG_V8F_GRID: G = BASE's levers with --grid pool8x20 and nothing else. The trainer's G-DVB checks the built n_tok against this argv; selfcheck.json records the built parameter
rem groups (the position table is the only per-token parameter) and the LRU the 160-token frames imply; the chain asserts them before any training.
call :prep || exit /b 2
call :gridbase %~2 || exit /b 2
rem ONE command string: the dry run prints exactly what the real run executes
set "GG="%PY%" "%TRAIN%" --arm M2b%GLEV% --grid pool8x20 --out "%RUNSM%\gate_%GBASE%-8x20" %A% --selfcheck-only"
if defined V8F_DRYRUN (echo %GG% & exit /b 0)
%GG% > "%RUNSM%\gate_%GBASE%-8x20.log" 2>&1
if errorlevel 1 (echo [FAIL] gate %GBASE%-8x20 - see %RUNSM%\gate_%GBASE%-8x20.log & exit /b 1)
echo [ok] gate %GBASE%-8x20
exit /b 0

:traingrid
rem PREREG_V8F_GRID: one G run. Its summary.json says arm M2b (the trainer's arm table is unchanged); its argv.json carries --grid pool8x20 (and, for BASE M2b-pgt, the PGT dose), its
rem checkpoint the grid and the built n_tok: code\v8f_grid_read.py refuses a run where any is not. The run lives in <BASE>_s<SEED>_8x20, so it can never resume or be mistaken for a 4x10 run.
call :prep || exit /b 2
if "%~2"=="" (echo REFUSED: traingrid needs a SEED and a BASE & exit /b 2)
call :gridbase %~3 || exit /b 2
set "OUT=%RUNSM%\%GBASE%_s%~2_8x20"
if exist "%OUT%\summary.json" (echo [skip] %GBASE%_s%~2_8x20 & exit /b 0)
rem ONE command string: the dry run prints exactly what the real run executes
set "TGR="%PY%" "%TRAIN%" --arm M2b%GLEV% --seed %~2 --grid pool8x20 --out "%OUT%" %A%"
if defined V8F_DRYRUN (echo %TGR% & exit /b 0)
if not exist "%OUT%" mkdir "%OUT%"
echo [%date% %time%] train %GBASE%_s%~2_8x20
%TGR% >> "%OUT%\train.log" 2>&1
if errorlevel 1 (echo [FAIL] train %GBASE%_s%~2_8x20 - see %OUT%\train.log & exit /b 1)
if not exist "%OUT%\summary.json" (echo [FAIL] train %GBASE%_s%~2_8x20 exited 0 without summary.json & exit /b 1)
exit /b 0

:evalgrid
rem PREREG_V8F_GRID: the four evals its reader needs, EACH ONE command string that the dry run prints and the real run executes (the PGT recipe):
rem   E0 the standard eval (--project --families)   E1 absmask (T's own present slots)   E2 the GT-code oracle (tacorc)   E3 the all-latcons oracle (GT everywhere but T's latcons)
rem   All four read THIS run's last checkpoint, which names its own grid (pool8x20): v8f_mini_eval.py takes the grid from the checkpoint. Each is skipped once its npz exists.
rem   FLAT on purpose (no parenthesised else-block): an `exit /b 1` inside a multi-line block returns rc 0 from `cmd /c queue.cmd` (MEASURED by the PGT build).
call :prep || exit /b 2
if "%~2"=="" (echo REFUSED: evalgrid needs a SEED and a BASE & exit /b 2)
call :gridbase %~3 || exit /b 2
set "OUT=%RUNSM%\%GBASE%_s%~2_8x20"
set "TG=%GBASE%-8x20_s%~2"
set "E0="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG% %EC% %ER% --project --families"
set "E1="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacabs --tac absmask %EC% %ER%"
set "E2="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacorc --tac oracle %EC% %ER%"
set "E3="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacorc-all-latcons --tac oracle --tac-groups all-latcons --tac-alpha 1 %EC% %ER%"
if defined V8F_DRYRUN (echo %E0% & echo %E1% & echo %E2% & echo %E3% & exit /b 0)
if not exist "%OUT%\summary.json" (echo REFUSED: %TG% has not finished training & exit /b 2)
if exist "%EV%\eval_%TG%.npz" (echo [skip] eval %TG% & goto :evalgrid_1)
%E0% >> "%EV%\eval_%TG%.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG% - see %EV%\eval_%TG%.log & exit /b 1)
echo [ok] eval %TG%
:evalgrid_1
if exist "%EV%\eval_%TG%-tacabs.npz" (echo [skip] eval %TG%-tacabs & goto :evalgrid_2)
%E1% >> "%EV%\eval_%TG%-tacabs.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacabs - see %EV%\eval_%TG%-tacabs.log & exit /b 1)
echo [ok] eval %TG%-tacabs
:evalgrid_2
if exist "%EV%\eval_%TG%-tacorc.npz" (echo [skip] eval %TG%-tacorc & goto :evalgrid_3)
%E2% >> "%EV%\eval_%TG%-tacorc.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacorc - see %EV%\eval_%TG%-tacorc.log & exit /b 1)
echo [ok] eval %TG%-tacorc
:evalgrid_3
if exist "%EV%\eval_%TG%-tacorc-all-latcons.npz" (echo [skip] eval %TG%-tacorc-all-latcons & goto :evalgrid_4)
%E3% >> "%EV%\eval_%TG%-tacorc-all-latcons.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacorc-all-latcons - see %EV%\eval_%TG%-tacorc-all-latcons.log & exit /b 1)
echo [ok] eval %TG%-tacorc-all-latcons
:evalgrid_4
exit /b 0

:gatehist
rem PREREG_V8F_HIST: H = M2b's levers with --hist 10 --window-start 3 and nothing else (hmax stays at the arm's auto 16, so the batch is B's). Amendment 1: --hist 10 ALONE would start every clip's
rem training windows at t = 9 (306,566 windows down to 279,494, 19,164 steps down to 17,472); --window-start 3 fixes the first window AND the held-age clamp at B's, so the window list and the budget are B's.
rem The trainer's G-DVB checks the built cfg.hist, the built window_start and the window list the data built against this argv; selfcheck.json records the built parameter groups (the per-frame
rem time table is the only per-history parameter), hist / hmax / window_start and the steps and windows; the chain asserts them before any training.
call :prep || exit /b 2
rem ONE command string: the dry run prints exactly what the real run executes
set "GH="%PY%" "%TRAIN%" --arm M2b --hist 10 --window-start 3 --out "%RUNSM%\gate_M2b-h10" %A% --selfcheck-only"
if defined V8F_DRYRUN (echo %GH% & exit /b 0)
%GH% > "%RUNSM%\gate_M2b-h10.log" 2>&1
if errorlevel 1 (echo [FAIL] gate M2b-h10 - see %RUNSM%\gate_M2b-h10.log & exit /b 1)
echo [ok] gate M2b-h10
exit /b 0

:trainhist
rem PREREG_V8F_HIST: one H run. Its summary.json says arm M2b (the trainer's arm table is unchanged); its argv.json carries --hist 10 --window-start 3, its checkpoint the built history, window_start 3
rem and a 10-row time table: code\v8f_hist_read.py refuses a run where any is not. The run lives in M2b_s<SEED>_h10, so it can never resume or be mistaken for a 4-frame run.
call :prep || exit /b 2
if "%~2"=="" (echo REFUSED: trainhist needs a SEED & exit /b 2)
set "OUT=%RUNSM%\M2b_s%~2_h10"
if exist "%OUT%\summary.json" (echo [skip] M2b_s%~2_h10 & exit /b 0)
rem ONE command string: the dry run prints exactly what the real run executes
set "TH="%PY%" "%TRAIN%" --arm M2b --seed %~2 --grid pool4x10 --hist 10 --window-start 3 --out "%OUT%" %A%"
if defined V8F_DRYRUN (echo %TH% & exit /b 0)
if not exist "%OUT%" mkdir "%OUT%"
echo [%date% %time%] train M2b_s%~2_h10
%TH% >> "%OUT%\train.log" 2>&1
if errorlevel 1 (echo [FAIL] train M2b_s%~2_h10 - see %OUT%\train.log & exit /b 1)
if not exist "%OUT%\summary.json" (echo [FAIL] train M2b_s%~2_h10 exited 0 without summary.json & exit /b 1)
exit /b 0

:evalhist
rem PREREG_V8F_HIST: the four evals its reader needs, EACH ONE command string that the dry run prints and the real run executes (the PGT / GRID recipe):
rem   E0 the standard eval (--project --families)   E1 absmask (T's own present slots)   E2 the GT-code oracle (tacorc)   E3 the all-latcons oracle (GT everywhere but T's latcons)
rem   All four read THIS run's last checkpoint, which names its own history (cfg.hist 10): v8f_mini_eval.py builds the model from the checkpoint's cfg and its strict load_state_dict
rem   refuses a time table of another size. Each is skipped once its npz exists.
rem   FLAT on purpose (no parenthesised else-block): an `exit /b 1` inside a multi-line block returns rc 0 from `cmd /c queue.cmd` (MEASURED by the PGT build).
call :prep || exit /b 2
if "%~2"=="" (echo REFUSED: evalhist needs a SEED & exit /b 2)
set "OUT=%RUNSM%\M2b_s%~2_h10"
set "TG=M2b-h10_s%~2"
set "E0="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG% %EC% %ER% --project --families"
set "E1="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacabs --tac absmask %EC% %ER%"
set "E2="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacorc --tac oracle %EC% %ER%"
set "E3="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacorc-all-latcons --tac oracle --tac-groups all-latcons --tac-alpha 1 %EC% %ER%"
if defined V8F_DRYRUN (echo %E0% & echo %E1% & echo %E2% & echo %E3% & exit /b 0)
if not exist "%OUT%\summary.json" (echo REFUSED: %TG% has not finished training & exit /b 2)
if exist "%EV%\eval_%TG%.npz" (echo [skip] eval %TG% & goto :evalhist_1)
%E0% >> "%EV%\eval_%TG%.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG% - see %EV%\eval_%TG%.log & exit /b 1)
echo [ok] eval %TG%
:evalhist_1
if exist "%EV%\eval_%TG%-tacabs.npz" (echo [skip] eval %TG%-tacabs & goto :evalhist_2)
%E1% >> "%EV%\eval_%TG%-tacabs.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacabs - see %EV%\eval_%TG%-tacabs.log & exit /b 1)
echo [ok] eval %TG%-tacabs
:evalhist_2
if exist "%EV%\eval_%TG%-tacorc.npz" (echo [skip] eval %TG%-tacorc & goto :evalhist_3)
%E2% >> "%EV%\eval_%TG%-tacorc.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacorc - see %EV%\eval_%TG%-tacorc.log & exit /b 1)
echo [ok] eval %TG%-tacorc
:evalhist_3
if exist "%EV%\eval_%TG%-tacorc-all-latcons.npz" (echo [skip] eval %TG%-tacorc-all-latcons & goto :evalhist_4)
%E3% >> "%EV%\eval_%TG%-tacorc-all-latcons.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacorc-all-latcons - see %EV%\eval_%TG%-tacorc-all-latcons.log & exit /b 1)
echo [ok] eval %TG%-tacorc-all-latcons
:evalhist_4
exit /b 0

:gatedscale
rem PREREG_V8F_DSCALE: Q = M2b's levers with --data-frac 0.5 and nothing else. select_train_clips takes the first round(0.5 x N) of B's clips in sha12 order (a nested half); the step budget stays B's 19,164
rem because the trainer sizes S from the windows of ALL clips (n_windows_all_clips), not from the half. The trainer's G-DVB reads back the built clip count (data_frac -> data:n_clips);
rem selfcheck.json records n_clips / n_clips_all / train_sha256 / steps_total / the window counts, and the chain asserts them before any training.
call :prep || exit /b 2
rem ONE command string: the dry run prints exactly what the real run executes
set "GD="%PY%" "%TRAIN%" --arm M2b --data-frac 0.5 --out "%RUNSM%\gate_M2b-df50" %A% --selfcheck-only"
if defined V8F_DRYRUN (echo %GD% & exit /b 0)
%GD% > "%RUNSM%\gate_M2b-df50.log" 2>&1
if errorlevel 1 (echo [FAIL] gate M2b-df50 - see %RUNSM%\gate_M2b-df50.log & exit /b 1)
echo [ok] gate M2b-df50
exit /b 0

:traindscale
rem PREREG_V8F_DSCALE: one Q run. Its summary.json says arm M2b and data_frac 0.5 with the half's clip count; its argv.json carries --data-frac 0.5. code\v8f_dscale_read.py refuses a run where any is not.
rem The run lives in M2b_s<SEED>_df50, so it can never resume or be mistaken for a full-data run.
call :prep || exit /b 2
if "%~2"=="" (echo REFUSED: traindscale needs a SEED & exit /b 2)
set "OUT=%RUNSM%\M2b_s%~2_df50"
if exist "%OUT%\summary.json" (echo [skip] M2b_s%~2_df50 & exit /b 0)
rem ONE command string: the dry run prints exactly what the real run executes
set "TD="%PY%" "%TRAIN%" --arm M2b --seed %~2 --grid pool4x10 --data-frac 0.5 --out "%OUT%" %A%"
if defined V8F_DRYRUN (echo %TD% & exit /b 0)
if not exist "%OUT%" mkdir "%OUT%"
echo [%date% %time%] train M2b_s%~2_df50
%TD% >> "%OUT%\train.log" 2>&1
if errorlevel 1 (echo [FAIL] train M2b_s%~2_df50 - see %OUT%\train.log & exit /b 1)
if not exist "%OUT%\summary.json" (echo [FAIL] train M2b_s%~2_df50 exited 0 without summary.json & exit /b 1)
exit /b 0

:evaldscale
rem PREREG_V8F_DSCALE: the four evals its reader needs, EACH ONE command string that the dry run prints and the real run executes (the HIST recipe):
rem   E0 the standard eval (--project --families)   E1 absmask (T's own present slots)   E2 the GT-code oracle (tacorc)   E3 the all-latcons oracle (GT everywhere but T's latcons)
rem   All four read THIS run's last checkpoint; the eval path never reads data_frac (a static test pins it). Each is skipped once its npz exists.
rem   FLAT on purpose (no parenthesised else-block): an `exit /b 1` inside a multi-line block returns rc 0 from `cmd /c queue.cmd` (MEASURED by the PGT build).
call :prep || exit /b 2
if "%~2"=="" (echo REFUSED: evaldscale needs a SEED & exit /b 2)
set "OUT=%RUNSM%\M2b_s%~2_df50"
set "TG=M2b-df50_s%~2"
set "E0="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG% %EC% %ER% --project --families"
set "E1="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacabs --tac absmask %EC% %ER%"
set "E2="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacorc --tac oracle %EC% %ER%"
set "E3="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacorc-all-latcons --tac oracle --tac-groups all-latcons --tac-alpha 1 %EC% %ER%"
if defined V8F_DRYRUN (echo %E0% & echo %E1% & echo %E2% & echo %E3% & exit /b 0)
if not exist "%OUT%\summary.json" (echo REFUSED: %TG% has not finished training & exit /b 2)
if exist "%EV%\eval_%TG%.npz" (echo [skip] eval %TG% & goto :evaldscale_1)
%E0% >> "%EV%\eval_%TG%.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG% - see %EV%\eval_%TG%.log & exit /b 1)
echo [ok] eval %TG%
:evaldscale_1
if exist "%EV%\eval_%TG%-tacabs.npz" (echo [skip] eval %TG%-tacabs & goto :evaldscale_2)
%E1% >> "%EV%\eval_%TG%-tacabs.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacabs - see %EV%\eval_%TG%-tacabs.log & exit /b 1)
echo [ok] eval %TG%-tacabs
:evaldscale_2
if exist "%EV%\eval_%TG%-tacorc.npz" (echo [skip] eval %TG%-tacorc & goto :evaldscale_3)
%E2% >> "%EV%\eval_%TG%-tacorc.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacorc - see %EV%\eval_%TG%-tacorc.log & exit /b 1)
echo [ok] eval %TG%-tacorc
:evaldscale_3
if exist "%EV%\eval_%TG%-tacorc-all-latcons.npz" (echo [skip] eval %TG%-tacorc-all-latcons & goto :evaldscale_4)
%E3% >> "%EV%\eval_%TG%-tacorc-all-latcons.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacorc-all-latcons - see %EV%\eval_%TG%-tacorc-all-latcons.log & exit /b 1)
echo [ok] eval %TG%-tacorc-all-latcons
:evaldscale_4
exit /b 0

:loraenv
rem PREREG_V8F_LORA Amendment 2 (the AR-2 chain, chain_v8f_lora_ar2.sh): the DECLARED clips and paths of identitylora / smokelora, written ONCE so the two stages, the chain and the tests read one list.
rem   LORA_TRAIN = the first LORA_NTRAIN train clips of B's list (the cache manifest in sha12 order, --exclude 540:600 applied): all are inside the first 600, whose production frames exist; --max-clips LORA_NTRAIN in smokelora selects exactly these
rem   LORA_EVAL  = the first 2 eval clips of the 141 in sha12 order
rem   LORAFRAMES = where the chain ships their v2ep files (built by the PRODUCTION route, pod_build_b1_epcache.py): train and eval subdirectories
rem   the two scripts are named RELATIVE to the code directory: mk_v8f_job.py runs this queue from there (cd /d), and the repo path carries an ampersand, which must never enter a command string
rem   (inside set "X="%PY%" "..."" it splits the line at the ampersand: MEASURED by the first dry run of these stages)
set "LORA_TRAIN=8f387977-4086-466d-bba4-b04f8cdc0743 bf5ec76e-e8b5-4f0d-a9cc-4fe1687d3d3a 0e1526b3-d2b6-46b2-95fa-2903bb34e2ec 83e32d0f-ed19-4116-8c1c-9e4c843de993"
set "LORA_NTRAIN=4"
set "LORA_EVAL=dca157a0-20a9-464b-a74c-084b678e7be1 e6cb7cd4-8349-4b9d-9674-e3ebf789e611"
if not defined LORAFRAMES set "LORAFRAMES=D:\tanitad\data\lora_ar2_frames"
set "LORAID=v8f_lora_identity_dev.py"
set "LORASMOKE=v8f_lora_smoke.py"
set "LORAOUT=%RUNSM%\lora_ar2"
if defined V8F_DRYRUN exit /b 0
if not exist "%LORAID%" (echo REFUSED: %LORAID% is not in the current directory - run the queue from the v8f-mini code directory, as mk_v8f_job.py does & exit /b 2)
if not exist "%LORASMOKE%" (echo REFUSED: %LORASMOKE% is not in the current directory - run the queue from the v8f-mini code directory, as mk_v8f_job.py does & exit /b 2)
exit /b 0

:identitylora
rem PREREG_V8F_LORA Amendment 2: the same-device identity control (see the header). ID0 is the control, ID1 the same with ONE preprocessing constant changed (--mutant): the control is only evidence if ID1 FAILS.
rem Exit codes of the driver: 0 PASS / mutant caught, 1 FAIL / mutant missed, 2 REFUSED. A FAIL is a READING (the chain pulls it and stops with the numbers); a crash writes no json and is [FAIL] here.
rem ONE command string each: the dry run prints exactly what the real run executes. FLAT (no parenthesised else-block): an exit /b 1 inside a multi-line block returns rc 0 from cmd /c queue.cmd (MEASURED by the PGT build).
call :prep || exit /b 2
call :loraenv || exit /b 2
set "OUT=%LORAOUT%\identity"
set "ID0="%PY%" %LORAID% --cache-root "%CACHEFULL%" --device cuda --train-frames "%LORAFRAMES%\train" --train-clips %LORA_TRAIN% --eval-frames "%LORAFRAMES%\eval" --eval-clips %LORA_EVAL% --out "%OUT%\identity_lora.json""
set "ID1="%PY%" %LORAID% --cache-root "%CACHEFULL%" --device cuda --train-frames "%LORAFRAMES%\train" --train-clips %LORA_TRAIN% --eval-frames "%LORAFRAMES%\eval" --eval-clips %LORA_EVAL% --mutant --out "%OUT%\identity_lora_mutant.json""
if defined V8F_DRYRUN (echo %ID0% & echo %ID1% & exit /b 0)
if exist "%OUT%\identity_lora.json" if exist "%OUT%\identity_lora_mutant.json" (echo [skip] identitylora & exit /b 0)
if not exist "%OUT%" mkdir "%OUT%"
if exist "%OUT%\identity_lora.json" del "%OUT%\identity_lora.json"
if exist "%OUT%\identity_lora_mutant.json" del "%OUT%\identity_lora_mutant.json"
echo [%date% %time%] identitylora
%ID0% > "%OUT%\identity_lora.log" 2>&1
if not exist "%OUT%\identity_lora.json" (echo [FAIL] identitylora wrote no json - see %OUT%\identity_lora.log & exit /b 1)
echo [ok] identitylora reading written
%ID1% > "%OUT%\identity_lora_mutant.log" 2>&1
if not exist "%OUT%\identity_lora_mutant.json" (echo [FAIL] identitylora mutant wrote no json - see %OUT%\identity_lora_mutant.log & exit /b 1)
echo [ok] identitylora mutant reading written
exit /b 0

:smokelora
rem PREREG_V8F_LORA proof 4 (AR-2): the 200-step smoke at batch 1 (see the header). The trainer argv is B's (A) plus --max-clips LORA_NTRAIN --frames-root; the smoke appends --encoder lora --batch 1 --accum 1 --steps 200
rem --enc-ckpt (and NO --amp: Amendment 2 pins autocast OFF; the encoder runs in the builder's pure bf16). Its census and heads-at-64 measurement drop --max-clips themselves, so the projected hours come from the real clip list.
rem ONE command string: the dry run prints exactly what the real run executes. A partial OUT (no smoke_report.json) is REFUSED, not resumed: it would double-count smoke_steps.jsonl.
call :prep || exit /b 2
call :loraenv || exit /b 2
set "OUT=%LORAOUT%\smoke"
set "SM="%PY%" %LORASMOKE% --steps 200 --out "%OUT%" -- --arm M2b --seed 0 --grid pool4x10 --max-clips %LORA_NTRAIN% --frames-root "%LORAFRAMES%\train" %A%"
if defined V8F_DRYRUN (echo %SM% & exit /b 0)
if exist "%OUT%\smoke_report.json" (echo [skip] smokelora & exit /b 0)
if exist "%OUT%" (echo REFUSED: %OUT% exists without a smoke_report.json - a partial run would resume and double-count smoke_steps.jsonl: move it aside & exit /b 2)
mkdir "%OUT%"
echo [%date% %time%] smokelora
%SM% > "%OUT%\smoke.log" 2>&1
if not exist "%OUT%\smoke_report.json" (echo [FAIL] smokelora wrote no smoke_report.json - see %OUT%\smoke.log & exit /b 1)
echo [ok] smokelora report written
exit /b 0

:gatearm
call :prep || exit /b 2
if "%~2"=="" (echo REFUSED: gatearm needs an ARM & exit /b 2)
if defined V8F_DRYRUN (echo "%PY%" "%TRAIN%" --arm %~2 --out %RUNSM%\gate_%~2 %A% --selfcheck-only & exit /b 0)
"%PY%" "%TRAIN%" --arm %~2 --out "%RUNSM%\gate_%~2" %A% --selfcheck-only > "%RUNSM%\gate_%~2.log" 2>&1
if errorlevel 1 (echo [FAIL] gate %~2 - see %RUNSM%\gate_%~2.log & exit /b 1)
echo [ok] gate %~2
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
:kind
rem PREREG_V8F_KIND (eval only, ZERO training), the banked M2b at seeds 0 and 1: 15 evals per seed, each ONE command string that the dry run prints and the real run executes (the DSCALE recipe).
rem Order per seed: own, all (the two identity controls), the six single-kind oracles, the six leave-one-out oracles, the union. Read by code\v8f_kind_read.py. Each eval is skipped once its npz exists.
call :prep || exit /b 2
for %%S in (0 1) do call :kindseed %%S || exit /b 1
exit /b 0

:kindseed
call :kindown M2b %~1 || exit /b 1
call :kindorc M2b %~1 all all || exit /b 1
for %%K in (speed angle tte timegap geom leadgap) do call :kindorc M2b %~1 %%K only-%%K || exit /b 1
for %%K in (speed angle tte timegap geom leadgap) do call :kindorc M2b %~1 all-%%K loo-%%K || exit /b 1
call :kindorc M2b %~1 constraints union || exit /b 1
exit /b 0

:kindown
rem args: ARM SEED. T's own code (--tac off) with the SAME flags as the banked standard eval (:eval): identity control 1 compares the two per window.
set "OUT=%RUNSM%\%~1_s%~2"
set "TG=%~1_s%~2-kind-own"
set "EK="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG% %EC% %ER% --project --families"
if defined V8F_DRYRUN (echo %EK% & exit /b 0)
if not exist "%OUT%\summary.json" (echo REFUSED: %~1_s%~2 has not finished training & exit /b 2)
if not exist "%EV%\eval_%~1_s%~2.npz" (echo REFUSED: the banked eval_%~1_s%~2.npz is missing from %EV% & exit /b 2)
if exist "%EV%\eval_%TG%.npz" (echo [skip] eval %TG% & exit /b 0)
%EK% >> "%EV%\eval_%TG%.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG% - see %EV%\eval_%TG%.log & exit /b 1)
echo [ok] eval %TG%
exit /b 0

:kindorc
rem args: ARM SEED GROUPSPEC NAME. The GT code in the cells GROUPSPEC names (all | <kind> | all-<kind> | constraints), T's own code elsewhere, tagged <ARM>_s<SEED>-kind-<NAME>.
set "OUT=%RUNSM%\%~1_s%~2"
set "TG=%~1_s%~2-kind-%~4"
set "EK="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG% --tac oracle --tac-groups %~3 --tac-alpha 1 %EC% %ER%"
if defined V8F_DRYRUN (echo %EK% & exit /b 0)
if not exist "%OUT%\summary.json" (echo REFUSED: %~1_s%~2 has not finished training & exit /b 2)
if not exist "%EV%\eval_%~1_s%~2-tacorc.npz" (echo REFUSED: the banked eval_%~1_s%~2-tacorc.npz is missing from %EV% & exit /b 2)
if exist "%EV%\eval_%TG%.npz" (echo [skip] eval %TG% & exit /b 0)
%EK% >> "%EV%\eval_%TG%.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG% - see %EV%\eval_%TG%.log & exit /b 1)
echo [ok] eval %TG%
exit /b 0
