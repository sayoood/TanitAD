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
rem   v8f_mini_box2_queue.cmd gatemixo              PREREG_V8F_MIXO: the in-process launch gate for X = M2b + --code-mix kind (O's teacher forcing drawn per sample AND per unit), no training. B is M2b (fixed in the PREREG): there is no BASE argument
rem   v8f_mini_box2_queue.cmd trainmixo SEED        PREREG_V8F_MIXO: one X run into %RUNSM%\M2b_s<SEED>_mixo (resumes; skipped once done)
rem   v8f_mini_box2_queue.cmd evalmixo SEED         PREREG_V8F_MIXO: that run's seven evals, tagged M2b-mixo_s<SEED>: the standard eval (--project --families), absmask, the GT-code oracle, the all-latcons oracle, and TAC_A2's dose chain on the
rem                                             constraints (T's own code --tac pred, the constraints oracle at alpha 1, at alpha 0.50); each skipped once its npz exists
rem
rem   v8f_mini_box2_queue.cmd gateflat              PREREG_V8F_FLAT: the in-process launch gate for FLAT = M2b + --o-interface cut (O reads a CONSTANT in place of T's code and S's checkpoints), no training. B is M2b (fixed in the PREREG): there is no BASE argument
rem   v8f_mini_box2_queue.cmd trainflat SEED       PREREG_V8F_FLAT: one FLAT run into %RUNSM%\M2b_s<SEED>_flat (resumes; skipped once done). BOTH seeds are always trained
rem   v8f_mini_box2_queue.cmd evalflat SEED        PREREG_V8F_FLAT: that run's four evals, tagged M2b-flat_s<SEED>: the standard eval (--project --families), absmask, the GT-code oracle (the identity read: FLAT's oracle ADE6 minus
rem                                             its own must be exactly 0) and the all-latcons oracle; each skipped once its npz exists
rem
rem   v8f_mini_box2_queue.cmd waccoacc              PREREG_V8F_OACC: the w_acc RULE, computed once before launch: w_acc = 0.5 * L_traj(B_s0) / L_acc(B_s0) over HBLEND's fit windows (141 clips, 9,582 windows) with B's s0 checkpoint
rem                                             (v8f_mini_eval_oacc.py wacc, CPU, cross-checked against HBLEND's banked arrays). Writes %RUNSM%\oacc\w_acc.json and w_acc.txt (the bare number the next stages read). Skipped once both exist
rem   v8f_mini_box2_queue.cmd gateoacc              PREREG_V8F_OACC: the in-process launch gate for X = M2b + --w-acc <the rule's value> (the acceleration term on O's plan), no training. B is M2b (fixed in the PREREG): there is no BASE argument
rem   v8f_mini_box2_queue.cmd trainoacc SEED        PREREG_V8F_OACC: one X run into %RUNSM%\M2b_s<SEED>_oacc (resumes; skipped once done). Seed 1 runs only if seed 0 passes PRIMARY-ACC and PRIMARY-KEEP (the chain decides)
rem   v8f_mini_box2_queue.cmd evaloacc SEED         PREREG_V8F_OACC: that run's standard eval (--project --families), tagged M2b-oacc_s<SEED>; skipped once its npz exists
rem   v8f_mini_box2_queue.cmd fitoacc SEED          PREREG_V8F_OACC: HBLEND's per-step weights RE-FITTED on X's own training windows (v8f_mini_eval_oacc.py hx, CPU): %RUNSM%\oacc\weights_M2b-oacc_s<SEED>.json (HX, the deployed form)
rem
rem   v8f_mini_box2_queue.cmd gateorc ARM SEED        PREREG_V8F_ORECOVER (Amendments 1 and 2): the in-process launch gate for ONE fine-tune run, ARM orc (--ego-perturb-p 0.3) or orc0 (--ego-perturb-p 0), SEED 0 or 1: the argv of
rem                                             its training run with --out and --selfcheck-only changed, nothing else. Four gates in all (orc/orc0 x seed 0/1), no training. The fine-tune starts from the BANKED M2b_s<SEED> checkpoint
rem   v8f_mini_box2_queue.cmd trainorc ARM SEED       PREREG_V8F_ORECOVER: one fine-tune run into %RUNSM%\M2b_s<SEED>_orc / _orc0 (resumes; skipped once done). ALL FOUR runs are always trained: the control ORC-0 is not optional
rem   v8f_mini_box2_queue.cmd evalorc ARM SEED        PREREG_V8F_ORECOVER: that run's standard T0 eval (--project --families), tagged M2b-orc_s<SEED> / M2b-orc0_s<SEED>; skipped once its npz exists. The T1-kin rollouts run on the dev box, not here
rem
rem   v8f_mini_box2_queue.cmd bankdag SEED              PREREG_V8F_DAGGER section 1: the own-state bank of the BANKED M2b_s<SEED> (frozen, PLAIN, one pass, TRAIN only): four clip-sharded generators (v8f_dagger_bank.py gen) then the merge, into %RUNSM%\dagger_bank\s<SEED>. Depends only on M2b and the TRAIN cache. Skipped once bank.json and PROVENANCE.json exist
rem   v8f_mini_box2_queue.cmd gatedag ARM SEED [P]     PREREG_V8F_DAGGER: the in-process launch gate for ONE fine-tune run, ARM dag (--dagger-p P, P = 0.1 after ORC read HARMS, 0.3 after NO EFFECT; the bank's sha256 is computed HERE from bank.json) or dag0 (--dagger-p 0, no P): the argv of its training run with --out and --selfcheck-only changed, nothing else. The TRAINING argv is PINNED to %RUNSM%\dagpin\ by this stage
rem   v8f_mini_box2_queue.cmd traindag ARM SEED [P]    PREREG_V8F_DAGGER: one fine-tune run into %RUNSM%\M2b_s<SEED>_dag / _dag0 (resumes; skipped once done). It runs the PINNED line the gate tokenised, after checking that the line it would build now is byte-identical to it
rem   v8f_mini_box2_queue.cmd evaldag ARM SEED         PREREG_V8F_DAGGER: that run's standard T0 eval (--project --families), tagged M2b-dag_s<SEED> / M2b-dag0_s<SEED>; skipped once its npz exists
rem   v8f_mini_box2_queue.cmd bankdag2 SEED             PREREG_V8F_DAGGER2 section 1: the own-state bank of DAG_s<SEED> (M2b_s<SEED>_dag\ckpt_last.pt, frozen, PLAIN, one pass, TRAIN only), the SAME rollout configuration and expectations as bankdag: four clip-sharded generators then the merge, into %RUNSM%\dagger_bank_dag2\s<SEED> (never the iteration-1 dagger_bank). Skipped once bank.json and PROVENANCE.json exist
rem   gatedag / traindag / evaldag also take ARM dag2 (DAG_s<SEED> fine-tuned with --dagger-p 0.1 on the iteration-2 bank; P is fixed at 0.1, any other P is REFUSED) and dag20 (the same argv with --dagger-p 0; takes no P): PREREG_V8F_DAGGER2 section 4. Init = %RUNSM%\M2b_s<SEED>_dag\ckpt_last.pt; run M2b_s<SEED>_dag2 / _dag20; tag M2b-dag2_s<SEED> / M2b-dag20_s<SEED>; pin dagpin\M2b_s<SEED>_dag2.train.argv.txt
rem   v8f_mini_box2_queue.cmd waccdag              PREREG_V8F_DAG_OACC section 1: the w_acc RULE of waccoacc with B = DAG: w_acc = 0.5 * L_traj(DAG_s0) / L_acc(DAG_s0) over HBLEND's fit windows, DAG_s0's checkpoint (--expect-sha256 pinned), CPU, cross-checked against DAG_s0's banked fit windows (rel 1e-3, else REFUSED). Writes %RUNSM%\dagoacc\w_acc.json and w_acc.txt (its OWN directory, never OACC's %RUNSM%\oacc). Skipped once both exist
rem   v8f_mini_box2_queue.cmd bankflat SEED         PREREG_V8F_FLATDAG section 1: the own-state bank of FLAT_s<SEED> (M2b_s<SEED>_flat\ckpt_last.pt, frozen, PLAIN, one pass, TRAIN only), the SAME rollout configuration and expectations as bankdag: four clip-sharded generators then the merge, into %RUNSM%\dagger_bank_flat\s<SEED>. Needs FLAT_s<SEED> FINISHED (summary.json)
rem   gatedag / traindag / evaldag also take ARM dagoacc (DAG's fine-tune argv verbatim + --w-acc <the waccdag value>; P is fixed at 0.1, any other P is REFUSED; same init M2b_s<SEED>, same iteration-1 bank and sha256; own run directory M2b_s<SEED>_dagoacc): PREREG_V8F_DAG_OACC section 1
rem   gatedag / traindag / evaldag also take ARM flatdag (FLAT_s<SEED> fine-tuned with DAG's recipe, --o-interface cut named, --dagger-p 0.1 on the FLAT bank; P fixed at 0.1) and flatdag0 (the same argv with --dagger-p 0; takes no P); evaldag runs the standard eval AND the GT-code oracle (the cut identity read) for them: PREREG_V8F_FLATDAG section 1
rem   gatedag / traindag / evaldag also take ARM dagorc (DAG_s<SEED> fine-tuned with DAG2-0's argv plus O-RECOVER's ego perturbation at p 0.3 and its five law flags), dagorcrep (DAG2-0's argv with --seed 10 / 11, the replicate; no perturbation) and dagorc01 (dagorc at p 0.1); none takes a P (any extra argument is REFUSED): PREREG_V8F_DAG_ORC section 1. Init = %RUNSM%\M2b_s<SEED>_dag\ckpt_last.pt; run M2b_s<SEED>_<ARM>; tag M2b-<ARM>_s<SEED>; pin dagpin\M2b_s<SEED>_<ARM>.train.argv.txt
rem   gatedag / traindag / evaldag also take ARM dagoaccorc (M2b_s<SEED> fine-tuned with DAG-OACC's argv verbatim plus O-RECOVER's ego perturbation at p 0.3 with its five law flags, composed with the DAgger swap by --ego-perturb-with-dagger) and dagoaccorcrep (DAG-OACC's argv with --seed 10 / 11, the replicate; no perturbation); neither takes a P (any extra argument is REFUSED): PREREG_V8F_DAG_ORC Amendment 1 section 1. Init = %RUNSM%\M2b_s<SEED>\ckpt_last.pt; run M2b_s<SEED>_<ARM>; tag M2b-<ARM>_s<SEED>; pin dagpin\M2b_s<SEED>_<ARM>.train.argv.txt
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
if /i "%~1"=="gatemixo" goto :gatemixo
if /i "%~1"=="trainmixo" goto :trainmixo
if /i "%~1"=="evalmixo" goto :evalmixo
if /i "%~1"=="gateflat" goto :gateflat
if /i "%~1"=="trainflat" goto :trainflat
if /i "%~1"=="evalflat" goto :evalflat
if /i "%~1"=="waccoacc" goto :waccoacc
if /i "%~1"=="gateoacc" goto :gateoacc
if /i "%~1"=="trainoacc" goto :trainoacc
if /i "%~1"=="evaloacc" goto :evaloacc
if /i "%~1"=="fitoacc" goto :fitoacc
if /i "%~1"=="gateorc" goto :gateorc
if /i "%~1"=="trainorc" goto :trainorc
if /i "%~1"=="evalorc" goto :evalorc
if /i "%~1"=="bankdag" goto :bankdag
if /i "%~1"=="gatedag" goto :gatedag
if /i "%~1"=="traindag" goto :traindag
if /i "%~1"=="evaldag" goto :evaldag
if /i "%~1"=="bankdag2" goto :bankdag2
if /i "%~1"=="waccdag" goto :waccdag
if /i "%~1"=="bankflat" goto :bankflat
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
:gatemixo
rem PREREG_V8F_MIXO: X = M2b's levers with --code-mix kind and nothing else (O's teacher forcing is drawn per sample AND per unit -- the decisions and the six constraint kinds -- at B's p_gt schedule 1.0 -> 0.2; the S-level switch,
rem T's losses, the windows and the 19,164 steps are B's). The trainer's G-DVB checks the built code_mix against this argv, its G-LIVE probe reads the code O actually receives (a partly-GT code must occur), and selfcheck.json
rem records both (levers.code_mix, code_mix_probe); the chain asserts them before any training.
call :prep || exit /b 2
rem ONE command string: the dry run prints exactly what the real run executes
set "GX="%PY%" "%TRAIN%" --arm M2b --code-mix kind --out "%RUNSM%\gate_M2b-mixo" %A% --selfcheck-only"
if defined V8F_DRYRUN (echo %GX% & exit /b 0)
%GX% > "%RUNSM%\gate_M2b-mixo.log" 2>&1
if errorlevel 1 (echo [FAIL] gate M2b-mixo - see %RUNSM%\gate_M2b-mixo.log & exit /b 1)
echo [ok] gate M2b-mixo
exit /b 0

:trainmixo
rem PREREG_V8F_MIXO: one X run. Its summary.json says arm M2b (the trainer's arm table is unchanged); its argv.json carries --code-mix kind, its levers and its checkpoint the BUILT code_mix: code\v8f_mixo_read.py refuses a run where any is not.
rem The run lives in M2b_s<SEED>_mixo, so it can never resume or be mistaken for B.
call :prep || exit /b 2
if "%~2"=="" (echo REFUSED: trainmixo needs a SEED & exit /b 2)
set "OUT=%RUNSM%\M2b_s%~2_mixo"
if exist "%OUT%\summary.json" (echo [skip] M2b_s%~2_mixo & exit /b 0)
rem ONE command string: the dry run prints exactly what the real run executes
set "TX="%PY%" "%TRAIN%" --arm M2b --seed %~2 --grid pool4x10 --code-mix kind --out "%OUT%" %A%"
if defined V8F_DRYRUN (echo %TX% & exit /b 0)
if not exist "%OUT%" mkdir "%OUT%"
echo [%date% %time%] train M2b_s%~2_mixo
%TX% >> "%OUT%\train.log" 2>&1
if errorlevel 1 (echo [FAIL] train M2b_s%~2_mixo - see %OUT%\train.log & exit /b 1)
if not exist "%OUT%\summary.json" (echo [FAIL] train M2b_s%~2_mixo exited 0 without summary.json & exit /b 1)
exit /b 0

:evalmixo
rem PREREG_V8F_MIXO: the seven evals its reader needs, EACH ONE command string that the dry run prints and the real run executes (the HIST recipe, plus TAC_A2's dose chain):
rem   E0 the standard eval (--project --families)   E1 absmask (T's own present slots)   E2 the GT-code oracle (tacorc: HARM)   E3 the all-latcons oracle (GT everywhere but T's latcons: LAMBDA)
rem   E4 T's own code (--tac pred: the dose chain's base, as PREREG_V8F_TAC's tacpred)   E5 the constraints oracle at alpha 1 (d_cons(1))   E6 the constraints oracle at alpha 0.50 (d_cons(0.5); r(0.5) = d_cons(0.5) / d_cons(1))
rem   E5 and E6 are the flags of PREREG_V8F_TAC Amendments 1 and 2 (`--tac oracle --tac-groups constraints --tac-alpha 1` / `0.50`), so r(0.5) on X is TAC_A2's own definition. All seven read THIS run's last checkpoint. Each is skipped once its npz exists.
rem   FLAT on purpose (no parenthesised else-block): an `exit /b 1` inside a multi-line block returns rc 0 from `cmd /c queue.cmd` (MEASURED by the PGT build).
call :prep || exit /b 2
if "%~2"=="" (echo REFUSED: evalmixo needs a SEED & exit /b 2)
set "OUT=%RUNSM%\M2b_s%~2_mixo"
set "TG=M2b-mixo_s%~2"
set "E0="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG% %EC% %ER% --project --families"
set "E1="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacabs --tac absmask %EC% %ER%"
set "E2="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacorc --tac oracle %EC% %ER%"
set "E3="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacorc-all-latcons --tac oracle --tac-groups all-latcons --tac-alpha 1 %EC% %ER%"
set "E4="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacpred --tac pred %EC% %ER%"
set "E5="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacorc-constraints --tac oracle --tac-groups constraints --tac-alpha 1 %EC% %ER%"
set "E6="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacorc-constraints-a50 --tac oracle --tac-groups constraints --tac-alpha 0.50 %EC% %ER%"
if defined V8F_DRYRUN (echo %E0% & echo %E1% & echo %E2% & echo %E3% & echo %E4% & echo %E5% & echo %E6% & exit /b 0)
if not exist "%OUT%\summary.json" (echo REFUSED: %TG% has not finished training & exit /b 2)
if exist "%EV%\eval_%TG%.npz" (echo [skip] eval %TG% & goto :evalmixo_1)
%E0% >> "%EV%\eval_%TG%.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG% - see %EV%\eval_%TG%.log & exit /b 1)
echo [ok] eval %TG%
:evalmixo_1
if exist "%EV%\eval_%TG%-tacabs.npz" (echo [skip] eval %TG%-tacabs & goto :evalmixo_2)
%E1% >> "%EV%\eval_%TG%-tacabs.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacabs - see %EV%\eval_%TG%-tacabs.log & exit /b 1)
echo [ok] eval %TG%-tacabs
:evalmixo_2
if exist "%EV%\eval_%TG%-tacorc.npz" (echo [skip] eval %TG%-tacorc & goto :evalmixo_3)
%E2% >> "%EV%\eval_%TG%-tacorc.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacorc - see %EV%\eval_%TG%-tacorc.log & exit /b 1)
echo [ok] eval %TG%-tacorc
:evalmixo_3
if exist "%EV%\eval_%TG%-tacorc-all-latcons.npz" (echo [skip] eval %TG%-tacorc-all-latcons & goto :evalmixo_4)
%E3% >> "%EV%\eval_%TG%-tacorc-all-latcons.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacorc-all-latcons - see %EV%\eval_%TG%-tacorc-all-latcons.log & exit /b 1)
echo [ok] eval %TG%-tacorc-all-latcons
:evalmixo_4
if exist "%EV%\eval_%TG%-tacpred.npz" (echo [skip] eval %TG%-tacpred & goto :evalmixo_5)
%E4% >> "%EV%\eval_%TG%-tacpred.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacpred - see %EV%\eval_%TG%-tacpred.log & exit /b 1)
echo [ok] eval %TG%-tacpred
:evalmixo_5
if exist "%EV%\eval_%TG%-tacorc-constraints.npz" (echo [skip] eval %TG%-tacorc-constraints & goto :evalmixo_6)
%E5% >> "%EV%\eval_%TG%-tacorc-constraints.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacorc-constraints - see %EV%\eval_%TG%-tacorc-constraints.log & exit /b 1)
echo [ok] eval %TG%-tacorc-constraints
:evalmixo_6
if exist "%EV%\eval_%TG%-tacorc-constraints-a50.npz" (echo [skip] eval %TG%-tacorc-constraints-a50 & goto :evalmixo_7)
%E6% >> "%EV%\eval_%TG%-tacorc-constraints-a50.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacorc-constraints-a50 - see %EV%\eval_%TG%-tacorc-constraints-a50.log & exit /b 1)
echo [ok] eval %TG%-tacorc-constraints-a50
:evalmixo_7
exit /b 0
:gateflat
rem PREREG_V8F_FLAT: FLAT = M2b's levers with --o-interface cut and nothing else (O reads a CONSTANT in place of T's code -- zeros, after every override -- and of S's checkpoints -- the nominal arcs, ok = 0; S and T are still
rem built, trained on their own losses and run, and T still reads S; the data, the 19,164 steps, the p_gt schedule 1.0 -> 0.2 and the eval are B's). The trainer's G-DVB checks the built o_interface against this argv, its G-LIVE
rem requires e_code.weight to read exactly 0.0, its o_interface probe reads what O is HANDED under a perturbation of everything above it, and selfcheck.json records them (levers.o_interface, o_interface_probe); the chain asserts
rem them before any training.
call :prep || exit /b 2
rem ONE command string: the dry run prints exactly what the real run executes
set "GX="%PY%" "%TRAIN%" --arm M2b --o-interface cut --out "%RUNSM%\gate_M2b-flat" %A% --selfcheck-only"
if defined V8F_DRYRUN (echo %GX% & exit /b 0)
%GX% > "%RUNSM%\gate_M2b-flat.log" 2>&1
if errorlevel 1 (echo [FAIL] gate M2b-flat - see %RUNSM%\gate_M2b-flat.log & exit /b 1)
echo [ok] gate M2b-flat
exit /b 0

:trainflat
rem PREREG_V8F_FLAT: one FLAT run. Its summary.json says arm M2b (the trainer's arm table is unchanged); its argv.json carries --o-interface cut, its levers and its checkpoint the BUILT o_interface: code\v8f_flat_read.py refuses a run
rem where any is not. The run lives in M2b_s<SEED>_flat, so it can never resume or be mistaken for B.
call :prep || exit /b 2
if "%~2"=="" (echo REFUSED: trainflat needs a SEED & exit /b 2)
set "OUT=%RUNSM%\M2b_s%~2_flat"
if exist "%OUT%\summary.json" (echo [skip] M2b_s%~2_flat & exit /b 0)
rem ONE command string: the dry run prints exactly what the real run executes
set "TX="%PY%" "%TRAIN%" --arm M2b --seed %~2 --grid pool4x10 --o-interface cut --out "%OUT%" %A%"
if defined V8F_DRYRUN (echo %TX% & exit /b 0)
if not exist "%OUT%" mkdir "%OUT%"
echo [%date% %time%] train M2b_s%~2_flat
%TX% >> "%OUT%\train.log" 2>&1
if errorlevel 1 (echo [FAIL] train M2b_s%~2_flat - see %OUT%\train.log & exit /b 1)
if not exist "%OUT%\summary.json" (echo [FAIL] train M2b_s%~2_flat exited 0 without summary.json & exit /b 1)
exit /b 0

:evalflat
rem PREREG_V8F_FLAT: the four evals its reader needs, EACH ONE command string that the dry run prints and the real run executes (the HIST recipe):
rem   E0 the standard eval (--project --families)   E1 absmask (T's own present slots)   E2 the GT-code oracle (tacorc: the PREREG's identity read, ADE6 minus own = exactly 0 under the cut)   E3 the all-latcons oracle
rem   All four read THIS run's last checkpoint, which names its own interface (cfg.o_interface cut): v8f_mini_eval.py builds the model from the checkpoint's cfg, so there is no flag to give it. Each is skipped once its npz exists.
rem   FLAT on purpose (no parenthesised else-block): an `exit /b 1` inside a multi-line block returns rc 0 from `cmd /c queue.cmd` (MEASURED by the PGT build).
call :prep || exit /b 2
if "%~2"=="" (echo REFUSED: evalflat needs a SEED & exit /b 2)
set "OUT=%RUNSM%\M2b_s%~2_flat"
set "TG=M2b-flat_s%~2"
set "E0="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG% %EC% %ER% --project --families"
set "E1="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacabs --tac absmask %EC% %ER%"
set "E2="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacorc --tac oracle %EC% %ER%"
set "E3="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG%-tacorc-all-latcons --tac oracle --tac-groups all-latcons --tac-alpha 1 %EC% %ER%"
if defined V8F_DRYRUN (echo %E0% & echo %E1% & echo %E2% & echo %E3% & exit /b 0)
if not exist "%OUT%\summary.json" (echo REFUSED: %TG% has not finished training & exit /b 2)
if exist "%EV%\eval_%TG%.npz" (echo [skip] eval %TG% & goto :evalflat_1)
%E0% >> "%EV%\eval_%TG%.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG% - see %EV%\eval_%TG%.log & exit /b 1)
echo [ok] eval %TG%
:evalflat_1
if exist "%EV%\eval_%TG%-tacabs.npz" (echo [skip] eval %TG%-tacabs & goto :evalflat_2)
%E1% >> "%EV%\eval_%TG%-tacabs.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacabs - see %EV%\eval_%TG%-tacabs.log & exit /b 1)
echo [ok] eval %TG%-tacabs
:evalflat_2
if exist "%EV%\eval_%TG%-tacorc.npz" (echo [skip] eval %TG%-tacorc & goto :evalflat_3)
%E2% >> "%EV%\eval_%TG%-tacorc.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacorc - see %EV%\eval_%TG%-tacorc.log & exit /b 1)
echo [ok] eval %TG%-tacorc
:evalflat_3
if exist "%EV%\eval_%TG%-tacorc-all-latcons.npz" (echo [skip] eval %TG%-tacorc-all-latcons & goto :evalflat_4)
%E3% >> "%EV%\eval_%TG%-tacorc-all-latcons.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG%-tacorc-all-latcons - see %EV%\eval_%TG%-tacorc-all-latcons.log & exit /b 1)
echo [ok] eval %TG%-tacorc-all-latcons
:evalflat_4
exit /b 0
:waccoacc
rem PREREG_V8F_OACC: the w_acc RULE, computed ONCE before launch, on the box that will train X. B's s0 checkpoint is the BANKED run's (the chain asserts its sha256 before this stage). The fit set is HBLEND's (fit_clips.json, k = 32), its
rem banked arrays are the cross-check (rel 1e-3, else REFUSED rc 2). The two HBLEND files are COPIED to %OD% first so that no argument string carries the ampersand of the repo path.
call :prep || exit /b 2
set "OD=%RUNSM%\oacc"
set "HB=%REPO%\TanitAD Research Lab\Architecture & Inference\Research\2026-10-17-v8f-mini\raw\2026-10-25-hblend"
set "EVALOACC=%REPO%\stack\scripts\v8f_mini_eval_oacc.py"
set "WX="%PY%" "%EVALOACC%" wacc --ckpt "%RUNSM%\M2b_s0\ckpt_last.pt" --fit-clips "%OD%\fit_clips.json" --cache-dir "%CACHEFULL%" --join-train "%RUNSM%\join_train.npz" --tracks "%TRACKS%" --banked-fit-windows "%OD%\fit_windows_M2b_s0.npz" --out "%OD%\w_acc.json" --out-txt "%OD%\w_acc.txt""
if defined V8F_DRYRUN (echo %WX% & exit /b 0)
if exist "%OD%\w_acc.json" if exist "%OD%\w_acc.txt" (echo [skip] w_acc & exit /b 0)
if not exist "%OD%" mkdir "%OD%"
copy /y "%HB%\fit_clips.json" "%OD%\fit_clips.json" >nul || (echo REFUSED: HBLEND's fit_clips.json did not copy & exit /b 2)
copy /y "%HB%\fit_windows_M2b_s0.npz" "%OD%\fit_windows_M2b_s0.npz" >nul || (echo REFUSED: HBLEND's banked fit windows did not copy & exit /b 2)
%WX% >> "%OD%\waccoacc.log" 2>&1
if errorlevel 1 (echo [FAIL] w_acc rule - see %OD%\waccoacc.log & exit /b 1)
if not exist "%OD%\w_acc.txt" (echo [FAIL] w_acc rule exited 0 without w_acc.txt & exit /b 1)
echo [ok] w_acc
exit /b 0

:oaccwacc
rem PREREG_V8F_OACC: the rule's value for the stages below, from the file the waccoacc stage wrote (an env WACC overrides it: the tests). The dry run of a box that has not computed it yet prints the placeholder WACC_FROM_RULE.
if not defined OD set "OD=%RUNSM%\oacc"
if not defined WACC if exist "%OD%\w_acc.txt" set /p WACC=<"%OD%\w_acc.txt"
if not defined WACC if defined V8F_DRYRUN set "WACC=WACC_FROM_RULE"
if not defined WACC (echo REFUSED: no w_acc - run the waccoacc stage first & exit /b 2)
exit /b 0

:gateoacc
rem PREREG_V8F_OACC: X = M2b's levers with --w-acc <the rule's value> and nothing else (the acceleration term on O's plan: the second difference on the 0.2 s grid, origin knot included, plan minus GT, squared, mean over the GT-valid steps; the
rem data, the 19,164 steps, the p_gt schedule 1.0 -> 0.2 and the eval are B's). The trainer's G-DVB checks the built w_acc against this argv, its G-LIVE requires the term in the loss dict and every weight alive, its acc_probe reads that the BUILT
rem loss's total moves by exactly w_acc x the term and that O's head receives the term's gradient, and selfcheck.json records them (levers.w_acc, acc_probe); the chain asserts them before any training.
call :prep || exit /b 2
call :oaccwacc || exit /b 2
rem ONE command string: the dry run prints exactly what the real run executes
set "GX="%PY%" "%TRAIN%" --arm M2b --w-acc %WACC% --out "%RUNSM%\gate_M2b-oacc" %A% --selfcheck-only"
if defined V8F_DRYRUN (echo %GX% & exit /b 0)
%GX% > "%RUNSM%\gate_M2b-oacc.log" 2>&1
if errorlevel 1 (echo [FAIL] gate M2b-oacc - see %RUNSM%\gate_M2b-oacc.log & exit /b 1)
echo [ok] gate M2b-oacc
exit /b 0

:trainoacc
rem PREREG_V8F_OACC: one X run. Its summary.json says arm M2b (the trainer's arm table is unchanged); its argv.json carries --w-acc <the rule's value>, its levers and its checkpoint the BUILT w_acc: code\v8f_oacc_read.py refuses a run where any is
rem not. The run lives in M2b_s<SEED>_oacc, so it can never resume or be mistaken for B.
call :prep || exit /b 2
call :oaccwacc || exit /b 2
if "%~2"=="" (echo REFUSED: trainoacc needs a SEED & exit /b 2)
set "OUT=%RUNSM%\M2b_s%~2_oacc"
if exist "%OUT%\summary.json" (echo [skip] M2b_s%~2_oacc & exit /b 0)
rem ONE command string: the dry run prints exactly what the real run executes
set "TX="%PY%" "%TRAIN%" --arm M2b --seed %~2 --grid pool4x10 --w-acc %WACC% --out "%OUT%" %A%"
if defined V8F_DRYRUN (echo %TX% & exit /b 0)
if not exist "%OUT%" mkdir "%OUT%"
echo [%date% %time%] train M2b_s%~2_oacc
%TX% >> "%OUT%\train.log" 2>&1
if errorlevel 1 (echo [FAIL] train M2b_s%~2_oacc - see %OUT%\train.log & exit /b 1)
if not exist "%OUT%\summary.json" (echo [FAIL] train M2b_s%~2_oacc exited 0 without summary.json & exit /b 1)
exit /b 0

:evaloacc
rem PREREG_V8F_OACC: the ONE eval its reader needs, ONE command string that the dry run prints and the real run executes: the standard eval (--project --families) of THIS run's last checkpoint, which names its own config (cfg.w_acc):
rem v8f_mini_eval.py builds the model from the checkpoint's cfg, so there is no flag to give it. Skipped once its npz exists.
call :prep || exit /b 2
if "%~2"=="" (echo REFUSED: evaloacc needs a SEED & exit /b 2)
set "OUT=%RUNSM%\M2b_s%~2_oacc"
set "TG=M2b-oacc_s%~2"
set "E0="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG% %EC% %ER% --project --families"
if defined V8F_DRYRUN (echo %E0% & exit /b 0)
if not exist "%OUT%\summary.json" (echo REFUSED: %TG% has not finished training & exit /b 2)
if exist "%EV%\eval_%TG%.npz" (echo [skip] eval %TG% & exit /b 0)
%E0% >> "%EV%\eval_%TG%.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG% - see %EV%\eval_%TG%.log & exit /b 1)
echo [ok] eval %TG%
exit /b 0

:fitoacc
rem PREREG_V8F_OACC: HX = X with HBLEND's per-step weights RE-FITTED on X's own training windows (the deployed form): HBLEND's own closed form + clip + isotonic step (v8f_hblend.fit_weights) on THIS checkpoint's held outputs over HBLEND's fit
rem windows (141 clips, 9,582 windows). CPU, forward only. The two HBLEND files were copied to %OD% by waccoacc (no ampersand in any argument). Skipped once the weights exist.
call :prep || exit /b 2
if "%~2"=="" (echo REFUSED: fitoacc needs a SEED & exit /b 2)
set "OD=%RUNSM%\oacc"
set "OUT=%RUNSM%\M2b_s%~2_oacc"
set "EVALOACC=%REPO%\stack\scripts\v8f_mini_eval_oacc.py"
set "FX="%PY%" "%EVALOACC%" hx --ckpt "%OUT%\ckpt_last.pt" --fit-clips "%OD%\fit_clips.json" --cache-dir "%CACHEFULL%" --join-train "%RUNSM%\join_train.npz" --tracks "%TRACKS%" --tag M2b-oacc_s%~2 --out "%OD%\weights_M2b-oacc_s%~2.json""
if defined V8F_DRYRUN (echo %FX% & exit /b 0)
if not exist "%OUT%\summary.json" (echo REFUSED: M2b-oacc_s%~2 has not finished training & exit /b 2)
if exist "%OD%\weights_M2b-oacc_s%~2.json" (echo [skip] HX weights M2b-oacc_s%~2 & exit /b 0)
if not exist "%OD%\fit_clips.json" (echo REFUSED: %OD%\fit_clips.json is missing - run waccoacc first & exit /b 2)
%FX% >> "%OD%\fitoacc_s%~2.log" 2>&1
if errorlevel 1 (echo [FAIL] HX fit M2b-oacc_s%~2 - see %OD%\fitoacc_s%~2.log & exit /b 1)
echo [ok] HX weights M2b-oacc_s%~2
exit /b 0
:orcarm
rem PREREG_V8F_ORECOVER Amendments 1 and 3: the helper of the three stages below (called by them, never dispatched). %1 = ARM (orc | orc0), %2 = SEED (0 | 1). ORC-0 = ORC's argv MINUS the five ego-perturbation LAW flags (vscale, vmax, adrop,
rem yawrate-std, seed), with --ego-perturb-p 0 and its own --out (Amendment 3): the trainer's own gate refuses a law flag set while p is 0, and at p = 0 they are inert (vmax's default 8.0 is ORC's, so the eligible denominator is identical).
rem The one value that differs (ORCP), the law flags (ORCL: ORC only, empty for ORC-0), the run's name suffix (ORCS) and the fine-tune's levers (ORCF, written ONCE, read by the gate AND the run) are decided HERE and nowhere else. Anything else is REFUSED (rc 2).
rem The fine-tune starts from the BANKED M2b_s<SEED> checkpoint (the chain asserts its sha256 first); --seed is that checkpoint's seed; --ego-perturb-p is always NAMED, because the flag's default disables the lever.
set "ORCP="
set "ORCS="
if /i "%~1"=="orc" (set "ORCP=0.3" & set "ORCS=orc")
if /i "%~1"=="orc0" (set "ORCP=0" & set "ORCS=orc0")
if not defined ORCP (echo REFUSED: ARM must be orc or orc0, got "%~1" & exit /b 2)
if "%~2"=="" (echo REFUSED: a SEED is needed & exit /b 2)
if not "%~2"=="0" if not "%~2"=="1" (echo REFUSED: SEED must be 0 or 1, got "%~2" & exit /b 2)
set "ORCSEED=%~2"
set "ORCL="
if /i "%~1"=="orc" set "ORCL= --ego-perturb-vscale 0.5 1.0 --ego-perturb-vmax 8.0 --ego-perturb-adrop 1.5 --ego-perturb-yawrate-std 2.0 --ego-perturb-seed %~2"
set "ORCF=--init-ckpt "%RUNSM%\M2b_s%~2\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %ORCP%%ORCL%"
exit /b 0

:gateorc
rem PREREG_V8F_ORECOVER (Amendment 2, "Which launch gate"): for this trainer the launch gate IS its own in-process one: v8f_mini_train.py --selfcheck-only (G-DVB on every flag incl. the perturbation, --init-ckpt and --warmup, and G-LIVE incl. the
rem ego_perturb_probe on a PERTURBED batch). The gate's argv is the run's argv with --out and --selfcheck-only changed and nothing else (the chain asserts that, and binds the run's launch-gate token to the TRAINING argv).
call :prep || exit /b 2
call :orcarm %~2 %~3 || exit /b 2
rem ONE command string: the dry run prints exactly what the real run executes
set "GX="%PY%" "%TRAIN%" --arm M2b --seed %ORCSEED% --grid pool4x10 --out "%RUNSM%\gate_M2b-%ORCS%_s%ORCSEED%" %A% %ORCF% --selfcheck-only"
if defined V8F_DRYRUN (echo %GX% & exit /b 0)
%GX% > "%RUNSM%\gate_M2b-%ORCS%_s%ORCSEED%.log" 2>&1
if errorlevel 1 (echo [FAIL] gate M2b-%ORCS%_s%ORCSEED% - see %RUNSM%\gate_M2b-%ORCS%_s%ORCSEED%.log & exit /b 1)
echo [ok] gate M2b-%ORCS%_s%ORCSEED%
exit /b 0

:trainorc
rem PREREG_V8F_ORECOVER: one fine-tune run. Its summary.json says arm M2b (the trainer's arm table is unchanged); its argv.json carries the whole fine-tune (--init-ckpt, --steps 3000, --ego-perturb-p), its levers and its checkpoint the BUILT
rem values, and summary.json the run totals ego_perturb_n / ego_perturb_eligible: code\raw\2026-10-26-orcread\orc_read.py refuses a run where any is not. The run lives in M2b_s<SEED>_orc / _orc0, so it can never resume or be mistaken for B.
call :prep || exit /b 2
call :orcarm %~2 %~3 || exit /b 2
set "OUT=%RUNSM%\M2b_s%ORCSEED%_%ORCS%"
rem ONE command string: the dry run prints exactly what the real run executes
set "TX="%PY%" "%TRAIN%" --arm M2b --seed %ORCSEED% --grid pool4x10 --out "%OUT%" %A% %ORCF%"
if defined V8F_DRYRUN (echo %TX% & exit /b 0)
if exist "%OUT%\summary.json" (echo [skip] M2b_s%ORCSEED%_%ORCS% & exit /b 0)
if not exist "%OUT%" mkdir "%OUT%"
echo [%date% %time%] train M2b_s%ORCSEED%_%ORCS%
%TX% >> "%OUT%\train.log" 2>&1
if errorlevel 1 (echo [FAIL] train M2b_s%ORCSEED%_%ORCS% - see %OUT%\train.log & exit /b 1)
if not exist "%OUT%\summary.json" (echo [FAIL] train M2b_s%ORCSEED%_%ORCS% exited 0 without summary.json & exit /b 1)
exit /b 0

:evalorc
rem PREREG_V8F_ORECOVER: the ONE T0 eval its reader needs (the T0 guard R-T0 and the four families), ONE command string that the dry run prints and the real run executes: the standard eval (--project --families) of THIS run's last
rem checkpoint, which names its own config (v8f_mini_eval.py builds the model from it). Skipped once its npz exists. FLAT on purpose (no parenthesised multi-line block).
call :prep || exit /b 2
call :orcarm %~2 %~3 || exit /b 2
set "OUT=%RUNSM%\M2b_s%ORCSEED%_%ORCS%"
set "TG=M2b-%ORCS%_s%ORCSEED%"
set "E0="%PY%" "%EVAL%" run --ckpt "%OUT%\ckpt_last.pt" --tag %TG% %EC% %ER% --project --families"
if defined V8F_DRYRUN (echo %E0% & exit /b 0)
if not exist "%OUT%\summary.json" (echo REFUSED: %TG% has not finished training & exit /b 2)
if exist "%EV%\eval_%TG%.npz" (echo [skip] eval %TG% & exit /b 0)
%E0% >> "%EV%\eval_%TG%.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %TG% - see %EV%\eval_%TG%.log & exit /b 1)
echo [ok] eval %TG%
exit /b 0

:dagname
rem PREREG_V8F_DAGGER: the helper of the stages below (called by them, never dispatched). %1 = ARM (dag | dag0), %2 = SEED (0 | 1). Sets the names ONCE: DAGS (arm), DAGSEED, DAGOUT (the run directory), DAGTAG (the T0 eval tag), DAGGATE (the gate directory), DAGBANK (the seed's own-state bank),
rem DAGPIN (the pinned TRAINING argv), DAGGRID (eval-141's grid, COPIED out of the repo path: its ampersand must never reach an argument string), DAGTOOL. Anything else is REFUSED (rc 2).
set "DAGS="
if /i "%~1"=="dag" set "DAGS=dag"
if /i "%~1"=="dag0" set "DAGS=dag0"
if /i "%~1"=="dag2" set "DAGS=dag2"
if /i "%~1"=="dag20" set "DAGS=dag20"
if /i "%~1"=="dagoacc" set "DAGS=dagoacc"
if /i "%~1"=="flatdag" set "DAGS=flatdag"
if /i "%~1"=="flatdag0" set "DAGS=flatdag0"
if /i "%~1"=="dagorc" set "DAGS=dagorc"
if /i "%~1"=="dagorcrep" set "DAGS=dagorcrep"
if /i "%~1"=="dagorc01" set "DAGS=dagorc01"
if /i "%~1"=="dagoaccorc" set "DAGS=dagoaccorc"
if /i "%~1"=="dagoaccorcrep" set "DAGS=dagoaccorcrep"
if not defined DAGS (echo REFUSED: ARM must be dag or dag0, got "%~1" & exit /b 2)
if "%~2"=="" (echo REFUSED: a SEED is needed & exit /b 2)
if not "%~2"=="0" if not "%~2"=="1" (echo REFUSED: SEED must be 0 or 1, got "%~2" & exit /b 2)
set "DAGSEED=%~2"
set "DAGOUT=%RUNSM%\M2b_s%~2_%DAGS%"
set "DAGTAG=M2b-%DAGS%_s%~2"
set "DAGGATE=%RUNSM%\gate_M2b-%DAGS%_s%~2"
set "DAGBANK=%RUNSM%\dagger_bank\s%~2"
if /i "%DAGS%"=="dag2" set "DAGBANK=%RUNSM%\dagger_bank_dag2\s%~2"
if /i "%DAGS%"=="dag20" set "DAGBANK=%RUNSM%\dagger_bank_dag2\s%~2"
if /i "%DAGS%"=="flatdag" set "DAGBANK=%RUNSM%\dagger_bank_flat\s%~2"
if /i "%DAGS%"=="flatdag0" set "DAGBANK=%RUNSM%\dagger_bank_flat\s%~2"
set "DAGPIN=%RUNSM%\dagpin\M2b_s%~2_%DAGS%.train.argv.txt"
set "DAGGRID=%RUNSM%\dagpin\eval_grid_windows.json"
set "DAGTOOL=%REPO%\stack\scripts\v8f_dagger_queue_tool.py"
exit /b 0

:prepdag
rem PREREG_V8F_DAGGER: eval-141's grid (the bank must be disjoint from it) is COPIED to a path without an ampersand, as :prep copies the P10 prior. Dry run: nothing is touched.
if defined V8F_DRYRUN exit /b 0
if not defined DAGGRID (echo REFUSED: :prepdag needs :dagname first & exit /b 2)
if not exist "%RUNSM%\dagpin" mkdir "%RUNSM%\dagpin"
copy /y "%REPO%\TanitAD Research Lab\Architecture & Inference\Research\2026-10-17-v8f-mini\raw\eval_grid_windows.json" "%DAGGRID%" >nul || (echo REFUSED: eval_grid_windows.json did not copy & exit /b 2)
exit /b 0

:dagarm
rem PREREG_V8F_DAGGER section 4 and Amendment 1: the helper of the gate and train stages (called by them, never dispatched). %1 = ARM (dag | dag0), %2 = SEED (0 | 1), %3 = P (dag only: 0.1 after ORC read HARMS, 0.3 after NO EFFECT; dag0 takes none).
rem The fine-tune is ORC Amendment 1's, unchanged (DAGF0), with --ego-perturb-p NAMED as 0 on both arms and --dagger-p always NAMED (the flag's default disables the lever). DAG-0 = DAGF0 + --dagger-p 0 and NOTHING else: the trainer's gate refuses a bank, a bank hash or a seed declared while p is 0.
rem DAG = DAGF0 + --dagger-p P --dagger-bank <the seed's bank> --dagger-bank-sha256 <sha256 of bank.json, computed HERE from the file> --dagger-seed <SEED>. The pair rule is therefore: the two argv differ in --out, --dagger-p's value and the three DAG-only flags, and nowhere else.
call :dagname %~1 %~2 || exit /b 2
if /i "%DAGS%"=="dag2" goto :dag2arm
if /i "%DAGS%"=="dag20" goto :dag2arm
if /i "%DAGS%"=="dagoacc" goto :dagoaccarm
if /i "%DAGS%"=="flatdag" goto :flatdagarm
if /i "%DAGS%"=="flatdag0" goto :flatdagarm
if /i "%DAGS%"=="dagorc" goto :dagorcarm
if /i "%DAGS%"=="dagorcrep" goto :dagorcarm
if /i "%DAGS%"=="dagorc01" goto :dagorcarm
if /i "%DAGS%"=="dagoaccorc" goto :dagoaccorcarm
if /i "%DAGS%"=="dagoaccorcrep" goto :dagoaccorcarm
set "DAGP=0"
set "DAGSHA="
set "DAGF0=--init-ckpt "%RUNSM%\M2b_s%~2\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0"
if /i "%DAGS%"=="dag0" (
  if not "%~3"=="" (echo REFUSED: dag0 takes no P, got "%~3" - a lever declared and inert & exit /b 2)
  set "DAGF=%DAGF0% --dagger-p 0"
  exit /b 0
)
if "%~3"=="" (echo REFUSED: dag needs P, 0.1 after ORC read HARMS or 0.3 after NO EFFECT & exit /b 2)
if not "%~3"=="0.1" if not "%~3"=="0.3" (echo REFUSED: P must be 0.1 or 0.3, got "%~3" & exit /b 2)
set "DAGP=%~3"
if defined V8F_DRYRUN goto :dagarm_dry
if not exist "%DAGBANK%\bank.json" (echo REFUSED: no merged bank at %DAGBANK% - run bankdag %~2 first & exit /b 2)
if not exist "%RUNSM%\dagpin" mkdir "%RUNSM%\dagpin"
"%PY%" "%DAGTOOL%" sha "%DAGBANK%\bank.json" > "%RUNSM%\dagpin\bank_sha_s%~2.txt"
if errorlevel 1 (echo REFUSED: could not hash %DAGBANK%\bank.json & exit /b 2)
set /p DAGSHA=<"%RUNSM%\dagpin\bank_sha_s%~2.txt"
goto :dagarm_have
:dagarm_dry
rem the dry run of a box that has no bank yet prints a placeholder; an env DAGSHA_DRY (the chain, which hashed the PULLED bank.json, and the tests) supplies the value. It is read ONLY here, never by a real run.
if defined DAGSHA_DRY (set "DAGSHA=%DAGSHA_DRY%") else (set "DAGSHA=BANK_SHA256_OF_BANK_JSON")
:dagarm_have
if not defined DAGSHA (echo REFUSED: no bank sha256 & exit /b 2)
set "DAGF=%DAGF0% --dagger-p %DAGP% --dagger-bank "%DAGBANK%" --dagger-bank-sha256 %DAGSHA% --dagger-seed %~2"
exit /b 0

:bankdag
rem PREREG_V8F_DAGGER section 1 and 8: the own-state bank of the BANKED M2b_s<SEED> checkpoint (frozen, PLAIN, one pass, ticks 1 and 2, TRAIN only), FOUR clip-sharded generator processes then the merge, ONE command string that the dry run prints and the real run executes.
rem It depends only on M2b and the TRAIN cache, so it may run before ORC's read. v8f_dagger_queue_tool.py bank fans out v8f_dagger_bank.py gen --shard K --n-shards 4 (a finished shard is skipped), then runs merge; rc != 0 if any shard or the merge failed.
rem --threads 2 per process: the safe-mode affinity mask is 0xFF (8 logical CPUs). --device cuda: the forwards run on the GPU (DESIGN_NOTES section 4). The bank is written beside the M2b runs, never into a run directory.
call :prep || exit /b 2
call :dagname dag %~2 || exit /b 2
call :prepdag || exit /b 2
set "BX="%PY%" "%DAGTOOL%" bank --n-shards 4 --out-dir "%DAGBANK%" -- --ckpt "%RUNSM%\M2b_s%~2\ckpt_last.pt" --m2b-argv "%RUNSM%\M2b_s%~2\argv.json" --cache-dir "%CACHEFULL%" --tracks "%TRACKS%" --join-train "%RUNSM%\join_train.npz" --eval-grid "%DAGGRID%" --batch 32 --threads 2 --device cuda --expect-clips 4512 --expect-windows 306566"
if defined V8F_DRYRUN (echo %BX% & exit /b 0)
if exist "%DAGBANK%\bank.json" if exist "%DAGBANK%\PROVENANCE.json" (echo [skip] bank s%~2 & exit /b 0)
if not exist "%RUNSM%\M2b_s%~2\ckpt_last.pt" (echo REFUSED: %RUNSM%\M2b_s%~2\ckpt_last.pt is missing & exit /b 2)
if not exist "%RUNSM%\M2b_s%~2\argv.json" (echo REFUSED: %RUNSM%\M2b_s%~2\argv.json is missing & exit /b 2)
if not exist "%DAGBANK%" mkdir "%DAGBANK%"
echo [%date% %time%] bank s%~2
%BX% >> "%DAGBANK%\bank_stage.log" 2>&1
if errorlevel 1 (echo [FAIL] bank s%~2 - see %DAGBANK%\bank_stage.log & exit /b 1)
if not exist "%DAGBANK%\bank.json" (echo [FAIL] bank s%~2 exited 0 without bank.json & exit /b 1)
echo [ok] bank s%~2
exit /b 0

:gatedag
rem PREREG_V8F_DAGGER section 6 ("Launch gate"): the trainer's own in-process gate (v8f_mini_train.py --selfcheck-only: G-DVB incl. bad_dagger, --init-ckpt and --warmup; G-LIVE incl. dagger_probe on a SWAPPED batch). The gate's argv is the run's argv with --out and --selfcheck-only changed and nothing else.
rem THE TOKEN BINDS THE TRAINING ARGV, SO THE TRAINING ARGV IS PINNED HERE: this stage writes the exact TRAINING line it derives from to %DAGPIN% (a re-pin that differs from an existing pin is REFUSED, never silently replaced), and traindag runs THAT FILE. The bank's sha256 is computed here, from bank.json.
call :prep || exit /b 2
call :dagarm %~2 %~3 %~4 || exit /b 2
set "TX="%PY%" "%TRAIN%" --arm M2b --seed %DAGSEED% --grid pool4x10 --out "%DAGOUT%" %A% %DAGF%"
set "GX="%PY%" "%TRAIN%" --arm M2b --seed %DAGSEED% --grid pool4x10 --out "%DAGGATE%" %A% %DAGF% --selfcheck-only"
if defined V8F_DRYRUN (echo %GX% & exit /b 0)
if not exist "%RUNSM%\dagpin" mkdir "%RUNSM%\dagpin"
>"%DAGPIN%.new" echo %TX%
if exist "%DAGPIN%" (
  fc /b "%DAGPIN%" "%DAGPIN%.new" >nul
  if errorlevel 1 (echo REFUSED: %DAGPIN% pins another training argv than the one derived now - delete it only if the bank or the levers were deliberately changed & exit /b 2)
) else (
  copy /y "%DAGPIN%.new" "%DAGPIN%" >nul || (echo REFUSED: could not pin the training argv & exit /b 2)
)
%GX% > "%DAGGATE%.log" 2>&1
if errorlevel 1 (echo [FAIL] gate M2b-%DAGS%_s%DAGSEED% - see %DAGGATE%.log & exit /b 1)
echo [ok] gate M2b-%DAGS%_s%DAGSEED%
exit /b 0

:traindag
rem PREREG_V8F_DAGGER: one fine-tune run. Its summary.json says arm M2b (the trainer's arm table is unchanged); its argv.json carries the whole fine-tune and its levers and checkpoint the BUILT values.
rem The line that runs is the PINNED one: the stage re-derives the training line from the current bank.json and the current files, REFUSES unless that is byte-identical (fc /b) to the pin the gate tokenised, then executes the line READ FROM THE PIN FILE. The run lives in M2b_s<SEED>_dag / _dag0, so it can never resume or be mistaken for B.
call :prep || exit /b 2
call :dagarm %~2 %~3 %~4 || exit /b 2
set "TX="%PY%" "%TRAIN%" --arm M2b --seed %DAGSEED% --grid pool4x10 --out "%DAGOUT%" %A% %DAGF%"
if defined V8F_DRYRUN (echo %TX% & exit /b 0)
if exist "%DAGOUT%\summary.json" (echo [skip] M2b_s%DAGSEED%_%DAGS% & exit /b 0)
if not exist "%DAGPIN%" (echo REFUSED: no pinned training argv at %DAGPIN% - gatedag %DAGS% %DAGSEED% must pass first & exit /b 2)
>"%DAGPIN%.chk" echo %TX%
fc /b "%DAGPIN%" "%DAGPIN%.chk" >nul
if errorlevel 1 (echo REFUSED: the training argv derived now differs from the one the gate tokenised - bank.json or a lever changed since the gate & exit /b 2)
set "TXP="
for /f "usebackq delims=" %%L in ("%DAGPIN%") do set "TXP=%%L"
if not defined TXP (echo REFUSED: the pin file is empty & exit /b 2)
if not exist "%DAGOUT%" mkdir "%DAGOUT%"
echo [%date% %time%] train M2b_s%DAGSEED%_%DAGS%
%TXP% >> "%DAGOUT%\train.log" 2>&1
if errorlevel 1 (echo [FAIL] train M2b_s%DAGSEED%_%DAGS% - see %DAGOUT%\train.log & exit /b 1)
if not exist "%DAGOUT%\summary.json" (echo [FAIL] train M2b_s%DAGSEED%_%DAGS% exited 0 without summary.json & exit /b 1)
exit /b 0

:evaldag
rem PREREG_V8F_DAGGER section 5: the T0 eval its reader needs for R-T0, T0-CKPT and the four families at T0: the standard eval (--project --families) of THIS run's last checkpoint (--ages cycle, v8f_mini_eval.py's default), ONE command string. Skipped once its npz exists. FLAT on purpose.
call :prep || exit /b 2
call :dagname %~2 %~3 || exit /b 2
if /i "%DAGS%"=="flatdag" goto :evalflatdag
if /i "%DAGS%"=="flatdag0" goto :evalflatdag
set "E0="%PY%" "%EVAL%" run --ckpt "%DAGOUT%\ckpt_last.pt" --tag %DAGTAG% %EC% %ER% --project --families"
if defined V8F_DRYRUN (echo %E0% & exit /b 0)
if not exist "%DAGOUT%\summary.json" (echo REFUSED: %DAGTAG% has not finished training & exit /b 2)
if exist "%EV%\eval_%DAGTAG%.npz" (echo [skip] eval %DAGTAG% & exit /b 0)
%E0% >> "%EV%\eval_%DAGTAG%.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %DAGTAG% - see %EV%\eval_%DAGTAG%.log & exit /b 1)
echo [ok] eval %DAGTAG%
exit /b 0
:dag2arm
rem PREREG_V8F_DAGGER2 section 4: reached ONLY from :dagarm (never called, never dispatched) with DAGS = dag2 | dag20 and :dagname's names set. %1 = ARM, %2 = SEED (0 | 1), %3 = P (dag2: 0.1 only; dag20 takes none).
rem The fine-tune is DAG's, unchanged, except that it starts from DAG_s<SEED> (M2b_s<SEED>_dag), the checkpoint that GENERATED the iteration-2 bank: --init-ckpt "%RUNSM%\M2b_s<SEED>_dag\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0.
rem DAG2 = that + --dagger-p 0.1 --dagger-bank <%RUNSM%\dagger_bank_dag2\s<SEED>> --dagger-bank-sha256 <sha256 of its bank.json, computed HERE> --dagger-seed <SEED>. DAG2-0 = that + --dagger-p 0 and NOTHING else (the trainer's gate refuses a bank, a bank hash or a seed declared while p is 0).
set "DAGP=0"
set "DAGSHA="
set "DAGF0=--init-ckpt "%RUNSM%\M2b_s%~2_dag\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0"
if /i "%DAGS%"=="dag20" (
  if not "%~3"=="" (echo REFUSED: dag20 takes no P, got "%~3" - a lever declared and inert & exit /b 2)
  set "DAGF=%DAGF0% --dagger-p 0"
  exit /b 0
)
if "%~3"=="" (echo REFUSED: dag2 needs P, which is fixed at 0.1 by PREREG_V8F_DAGGER2 & exit /b 2)
if not "%~3"=="0.1" (echo REFUSED: dag2's P is fixed at 0.1 by PREREG_V8F_DAGGER2, got "%~3" & exit /b 2)
set "DAGP=%~3"
if defined V8F_DRYRUN goto :dag2arm_dry
if not exist "%DAGBANK%\bank.json" (echo REFUSED: no merged bank at %DAGBANK% - run bankdag2 %~2 first & exit /b 2)
if not exist "%RUNSM%\dagpin" mkdir "%RUNSM%\dagpin"
"%PY%" "%DAGTOOL%" sha "%DAGBANK%\bank.json" > "%RUNSM%\dagpin\bank_sha_dag2_s%~2.txt"
if errorlevel 1 (echo REFUSED: could not hash %DAGBANK%\bank.json & exit /b 2)
set /p DAGSHA=<"%RUNSM%\dagpin\bank_sha_dag2_s%~2.txt"
goto :dag2arm_have
:dag2arm_dry
rem the dry run of a box that has no iteration-2 bank yet prints a placeholder; an env DAGSHA_DRY (the chain, which hashed the PULLED bank.json, and the tests) supplies the value. It is read ONLY here, never by a real run.
if defined DAGSHA_DRY (set "DAGSHA=%DAGSHA_DRY%") else (set "DAGSHA=BANK_SHA256_OF_BANK_JSON")
:dag2arm_have
if not defined DAGSHA (echo REFUSED: no bank sha256 & exit /b 2)
set "DAGF=%DAGF0% --dagger-p %DAGP% --dagger-bank "%DAGBANK%" --dagger-bank-sha256 %DAGSHA% --dagger-seed %~2"
exit /b 0

:bankdag2
rem PREREG_V8F_DAGGER2 section 1: the own-state bank of DAG_s<SEED> (M2b_s<SEED>_dag\ckpt_last.pt, frozen, PLAIN, one pass, ticks 1 and 2, TRAIN only), FOUR clip-sharded generator processes then the merge, ONE command string that the dry run prints and the real run executes.
rem It is bankdag's line with ONE change of generator: --ckpt and --m2b-argv are DAG's (the TRAIN list is rebuilt from DAG's argv.json; DAG's data flags are M2b's, only its fine-tune and --dagger-* flags are added), and --out-dir is the iteration-2 directory. The rollout configuration and the --expect-* numbers are bankdag's.
rem It needs DAG FINISHED (summary.json beside the checkpoint): the trainer rewrites ckpt_last.pt during training, and a bank of a half-trained DAG would be a different generator.
call :prep || exit /b 2
call :dagname dag2 %~2 || exit /b 2
call :prepdag || exit /b 2
set "DAGGEN=%RUNSM%\M2b_s%~2_dag"
set "BX="%PY%" "%DAGTOOL%" bank --n-shards 4 --out-dir "%DAGBANK%" -- --ckpt "%DAGGEN%\ckpt_last.pt" --m2b-argv "%DAGGEN%\argv.json" --cache-dir "%CACHEFULL%" --tracks "%TRACKS%" --join-train "%RUNSM%\join_train.npz" --eval-grid "%DAGGRID%" --batch 32 --threads 2 --device cuda --expect-clips 4512 --expect-windows 306566"
if defined V8F_DRYRUN (echo %BX% & exit /b 0)
if exist "%DAGBANK%\bank.json" if exist "%DAGBANK%\PROVENANCE.json" (echo [skip] bank dag2 s%~2 & exit /b 0)
if not exist "%DAGGEN%\summary.json" (echo REFUSED: %DAGGEN%\summary.json is missing - DAG_s%~2 has not finished training & exit /b 2)
if not exist "%DAGGEN%\ckpt_last.pt" (echo REFUSED: %DAGGEN%\ckpt_last.pt is missing & exit /b 2)
if not exist "%DAGGEN%\argv.json" (echo REFUSED: %DAGGEN%\argv.json is missing & exit /b 2)
if not exist "%DAGBANK%" mkdir "%DAGBANK%"
echo [%date% %time%] bank dag2 s%~2
%BX% >> "%DAGBANK%\bank_stage.log" 2>&1
if errorlevel 1 (echo [FAIL] bank dag2 s%~2 - see %DAGBANK%\bank_stage.log & exit /b 1)
if not exist "%DAGBANK%\bank.json" (echo [FAIL] bank dag2 s%~2 exited 0 without bank.json & exit /b 1)
echo [ok] bank dag2 s%~2
exit /b 0
:dagoaccarm
rem PREREG_V8F_DAG_OACC section 1: reached ONLY from :dagarm (never called, never dispatched) with DAGS = dagoacc and :dagname's names set. %1 = ARM, %2 = SEED (0 | 1), %3 = P (0.1 only).
rem The arm is DAG's fine-tune argv VERBATIM (init M2b_s<SEED>, 3,000 steps, lr 1e-4, warmup 100, p_gt 0.2 -> 0.2, --ego-perturb-p 0, --dagger-p 0.1, the ITERATION-1 bank %RUNSM%\dagger_bank\s<SEED> with its sha256 computed HERE from bank.json, --dagger-seed <SEED>)
rem plus --w-acc <the waccdag value> at the END, and nothing else. The bank's sha256 is written to its own file (bank_sha_dagoacc_s<SEED>.txt); the pin is dagpin\M2b_s<SEED>_dagoacc.train.argv.txt (DAG's pin is never touched).
set "DAGP=0"
set "DAGSHA="
set "DAGF0=--init-ckpt "%RUNSM%\M2b_s%~2\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0"
if "%~3"=="" (echo REFUSED: dagoacc needs P, which is fixed at 0.1 by PREREG_V8F_DAG_OACC & exit /b 2)
if not "%~3"=="0.1" (echo REFUSED: dagoacc's P is fixed at 0.1 by PREREG_V8F_DAG_OACC, got "%~3" & exit /b 2)
set "DAGP=%~3"
call :dagoaccwacc || exit /b 2
if defined V8F_DRYRUN goto :dagoaccarm_dry
if not exist "%DAGBANK%\bank.json" (echo REFUSED: no merged bank at %DAGBANK% - DAG's iteration-1 bank, run bankdag %~2 first & exit /b 2)
if not exist "%RUNSM%\dagpin" mkdir "%RUNSM%\dagpin"
"%PY%" "%DAGTOOL%" sha "%DAGBANK%\bank.json" > "%RUNSM%\dagpin\bank_sha_dagoacc_s%~2.txt"
if errorlevel 1 (echo REFUSED: could not hash %DAGBANK%\bank.json & exit /b 2)
set /p DAGSHA=<"%RUNSM%\dagpin\bank_sha_dagoacc_s%~2.txt"
goto :dagoaccarm_have
:dagoaccarm_dry
rem the dry run of a box that has no bank hash yet prints a placeholder; an env DAGSHA_DRY (the chain, which hashed the PULLED bank.json, and the tests) supplies the value. It is read ONLY here, never by a real run.
if defined DAGSHA_DRY (set "DAGSHA=%DAGSHA_DRY%") else (set "DAGSHA=BANK_SHA256_OF_BANK_JSON")
:dagoaccarm_have
if not defined DAGSHA (echo REFUSED: no bank sha256 & exit /b 2)
set "DAGF=%DAGF0% --dagger-p %DAGP% --dagger-bank "%DAGBANK%" --dagger-bank-sha256 %DAGSHA% --dagger-seed %~2 --w-acc %WACCDAG%"
exit /b 0

:dagoaccwacc
rem PREREG_V8F_DAG_OACC: the rule's value for the dagoacc arm (called by :dagoaccarm, never dispatched). A real run reads ONLY the file the waccdag stage wrote (an inherited WACCDAG is cleared first, so a stale environment cannot stand in for the rule);
rem the dry run of a box that has not computed it yet prints the placeholder WACCDAG_FROM_RULE, or the value of WACCDAG_DRY (the chain, which pulled the box's file and checked it against the offline recomputation, and the tests).
set "WACCDAG="
if defined V8F_DRYRUN goto :dagoaccwacc_dry
if not exist "%RUNSM%\dagoacc\w_acc.txt" (echo REFUSED: no w_acc for DAG-OACC - run the waccdag stage first & exit /b 2)
set /p WACCDAG=<"%RUNSM%\dagoacc\w_acc.txt"
goto :dagoaccwacc_have
:dagoaccwacc_dry
if defined WACCDAG_DRY (set "WACCDAG=%WACCDAG_DRY%") else (set "WACCDAG=WACCDAG_FROM_RULE")
:dagoaccwacc_have
if not defined WACCDAG (echo REFUSED: no w_acc & exit /b 2)
exit /b 0

:flatdagarm
rem PREREG_V8F_FLATDAG section 1: reached ONLY from :dagarm (never called, never dispatched) with DAGS = flatdag | flatdag0 and :dagname's names set. %1 = ARM, %2 = SEED (0 | 1), %3 = P (flatdag: 0.1 only; flatdag0 takes none).
rem The fine-tune is DAG's, unchanged, except that it starts from FLAT_s<SEED> (M2b_s<SEED>_flat) and KEEPS FLAT's lever --o-interface cut, NAMED (the flag's default is full, and INIT_CKPT_MATCH did not compare it before the F1 fix):
rem --init-ckpt "%RUNSM%\M2b_s<SEED>_flat\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0 --o-interface cut.
rem FLATDAG = that + --dagger-p 0.1 --dagger-bank <%RUNSM%\dagger_bank_flat\s<SEED>> --dagger-bank-sha256 <sha256 of its bank.json, computed HERE> --dagger-seed <SEED>. FLATDAG-0 = that + --dagger-p 0 and NOTHING else (C-F2: the pair differs only in --dagger-p, the bank flags and --out).
set "DAGP=0"
set "DAGSHA="
set "DAGF0=--init-ckpt "%RUNSM%\M2b_s%~2_flat\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0 --o-interface cut"
if /i "%DAGS%"=="flatdag0" (
  if not "%~3"=="" (echo REFUSED: flatdag0 takes no P, got "%~3" - a lever declared and inert & exit /b 2)
  set "DAGF=%DAGF0% --dagger-p 0"
  exit /b 0
)
if "%~3"=="" (echo REFUSED: flatdag needs P, which is fixed at 0.1 by PREREG_V8F_FLATDAG & exit /b 2)
if not "%~3"=="0.1" (echo REFUSED: flatdag's P is fixed at 0.1 by PREREG_V8F_FLATDAG, got "%~3" & exit /b 2)
set "DAGP=%~3"
if defined V8F_DRYRUN goto :flatdagarm_dry
if not exist "%DAGBANK%\bank.json" (echo REFUSED: no merged bank at %DAGBANK% - run bankflat %~2 first & exit /b 2)
if not exist "%RUNSM%\dagpin" mkdir "%RUNSM%\dagpin"
"%PY%" "%DAGTOOL%" sha "%DAGBANK%\bank.json" > "%RUNSM%\dagpin\bank_sha_flatdag_s%~2.txt"
if errorlevel 1 (echo REFUSED: could not hash %DAGBANK%\bank.json & exit /b 2)
set /p DAGSHA=<"%RUNSM%\dagpin\bank_sha_flatdag_s%~2.txt"
goto :flatdagarm_have
:flatdagarm_dry
rem the dry run of a box that has no FLAT bank yet prints a placeholder; an env DAGSHA_DRY (the chain, which hashed the PULLED bank.json, and the tests) supplies the value. It is read ONLY here, never by a real run.
if defined DAGSHA_DRY (set "DAGSHA=%DAGSHA_DRY%") else (set "DAGSHA=BANK_SHA256_OF_BANK_JSON")
:flatdagarm_have
if not defined DAGSHA (echo REFUSED: no bank sha256 & exit /b 2)
set "DAGF=%DAGF0% --dagger-p %DAGP% --dagger-bank "%DAGBANK%" --dagger-bank-sha256 %DAGSHA% --dagger-seed %~2"
exit /b 0

:evalflatdag
rem PREREG_V8F_FLATDAG: reached ONLY from :evaldag (never called, never dispatched) with DAGS = flatdag | flatdag0 and :dagname's names set. The two evals its reader needs, EACH ONE command string that the dry run prints and the real run executes (the evalflat recipe):
rem   E0 the standard T0 eval (--project --families), tagged M2b-flatdag_s<SEED> / M2b-flatdag0_s<SEED>      E2 the GT-code oracle (--tac oracle), tagged <tag>-tacorc: C-F3's identity read (oracle minus own = exactly 0 under the cut)
rem Both read THIS run's last checkpoint, which names its own interface (cfg.o_interface); each is skipped once its npz exists. FLAT on purpose (no parenthesised else-block).
call :prep || exit /b 2
set "E0="%PY%" "%EVAL%" run --ckpt "%DAGOUT%\ckpt_last.pt" --tag %DAGTAG% %EC% %ER% --project --families"
set "E2="%PY%" "%EVAL%" run --ckpt "%DAGOUT%\ckpt_last.pt" --tag %DAGTAG%-tacorc --tac oracle %EC% %ER%"
if defined V8F_DRYRUN (echo %E0% & echo %E2% & exit /b 0)
if not exist "%DAGOUT%\summary.json" (echo REFUSED: %DAGTAG% has not finished training & exit /b 2)
if exist "%EV%\eval_%DAGTAG%.npz" (echo [skip] eval %DAGTAG% & goto :evalflatdag_1)
%E0% >> "%EV%\eval_%DAGTAG%.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %DAGTAG% - see %EV%\eval_%DAGTAG%.log & exit /b 1)
echo [ok] eval %DAGTAG%
:evalflatdag_1
if exist "%EV%\eval_%DAGTAG%-tacorc.npz" (echo [skip] eval %DAGTAG%-tacorc & goto :evalflatdag_2)
%E2% >> "%EV%\eval_%DAGTAG%-tacorc.log" 2>&1
if errorlevel 1 (echo [FAIL] eval %DAGTAG%-tacorc - see %EV%\eval_%DAGTAG%-tacorc.log & exit /b 1)
echo [ok] eval %DAGTAG%-tacorc
:evalflatdag_2
exit /b 0

:waccdag
rem PREREG_V8F_DAG_OACC section 1 and C-O3: the w_acc RULE of waccoacc (PREREG_V8F_OACC and its Amendment 1, R1-R3) with B = DAG, computed ONCE before the gates, on the box that will train DAG-OACC:
rem w_acc = 0.5 * L_traj(DAG_s0) / L_acc(DAG_s0) over HBLEND's fit windows (141 clips, 9,582 windows). DAG_s0's checkpoint is the BANKED run's (--expect-sha256 refuses any other file; the chain also asserts it before this stage).
rem The cross-check is DAG_s0's banked fit windows (fitplans_DAG_s0.npz, shipped by the chain to %OD%\fit_windows_DAG_s0.npz): rel 1e-3 and the same windows, else REFUSED rc 2. Everything is written into %RUNSM%\dagoacc (OACC's %RUNSM%\oacc is never read or written).
rem The fit-clips file is COPIED from the repo first so that no argument string carries the ampersand of the repo path.
call :prep || exit /b 2
set "OD=%RUNSM%\dagoacc"
set "HB=%REPO%\TanitAD Research Lab\Architecture & Inference\Research\2026-10-17-v8f-mini\raw\2026-10-25-hblend"
set "EVALOACC=%REPO%\stack\scripts\v8f_mini_eval_oacc.py"
set "WX="%PY%" "%EVALOACC%" wacc --ckpt "%RUNSM%\M2b_s0_dag\ckpt_last.pt" --expect-sha256 d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1 --fit-clips "%OD%\fit_clips.json" --cache-dir "%CACHEFULL%" --join-train "%RUNSM%\join_train.npz" --tracks "%TRACKS%" --banked-fit-windows "%OD%\fit_windows_DAG_s0.npz" --out "%OD%\w_acc.json" --out-txt "%OD%\w_acc.txt""
if defined V8F_DRYRUN (echo %WX% & exit /b 0)
if exist "%OD%\w_acc.json" if exist "%OD%\w_acc.txt" (echo [skip] w_acc DAG-OACC & exit /b 0)
if not exist "%RUNSM%\M2b_s0_dag\summary.json" (echo REFUSED: %RUNSM%\M2b_s0_dag\summary.json is missing - DAG_s0 has not finished training & exit /b 2)
if not exist "%RUNSM%\M2b_s0_dag\ckpt_last.pt" (echo REFUSED: %RUNSM%\M2b_s0_dag\ckpt_last.pt is missing & exit /b 2)
if not exist "%OD%" mkdir "%OD%"
if not exist "%OD%\fit_windows_DAG_s0.npz" (echo REFUSED: %OD%\fit_windows_DAG_s0.npz is missing - the chain ships DAG_s0's banked fit windows & exit /b 2)
copy /y "%HB%\fit_clips.json" "%OD%\fit_clips.json" >nul || (echo REFUSED: HBLEND's fit_clips.json did not copy & exit /b 2)
echo [%date% %time%] w_acc DAG-OACC
%WX% >> "%OD%\waccdag.log" 2>&1
if errorlevel 1 (echo [FAIL] w_acc rule on DAG_s0 - see %OD%\waccdag.log & exit /b 1)
if not exist "%OD%\w_acc.txt" (echo [FAIL] w_acc rule exited 0 without w_acc.txt & exit /b 1)
echo [ok] w_acc DAG-OACC
exit /b 0

:bankflat
rem PREREG_V8F_FLATDAG section 1: the own-state bank of FLAT_s<SEED> (M2b_s<SEED>_flat\ckpt_last.pt, frozen, PLAIN, one pass, ticks 1 and 2, TRAIN only), FOUR clip-sharded generator processes then the merge, ONE command string that the dry run prints and the real run executes.
rem It is bankdag's line with ONE change of generator: --ckpt and --m2b-argv are FLAT's (the TRAIN list is rebuilt from FLAT's argv.json, which carries --o-interface cut: the generator builds the CUT policy), and --out-dir is the FLAT bank directory. The rollout configuration and the --expect-* numbers are bankdag's.
rem It needs FLAT FINISHED (summary.json beside the checkpoint): a bank of a half-trained generator would be a different generator.
call :prep || exit /b 2
call :dagname flatdag %~2 || exit /b 2
call :prepdag || exit /b 2
set "DAGGEN=%RUNSM%\M2b_s%~2_flat"
set "BX="%PY%" "%DAGTOOL%" bank --n-shards 4 --out-dir "%DAGBANK%" -- --ckpt "%DAGGEN%\ckpt_last.pt" --m2b-argv "%DAGGEN%\argv.json" --cache-dir "%CACHEFULL%" --tracks "%TRACKS%" --join-train "%RUNSM%\join_train.npz" --eval-grid "%DAGGRID%" --batch 32 --threads 2 --device cuda --expect-clips 4512 --expect-windows 306566"
if defined V8F_DRYRUN (echo %BX% & exit /b 0)
if exist "%DAGBANK%\bank.json" if exist "%DAGBANK%\PROVENANCE.json" (echo [skip] bank flat s%~2 & exit /b 0)
if not exist "%DAGGEN%\summary.json" (echo REFUSED: %DAGGEN%\summary.json is missing - FLAT_s%~2 has not finished training & exit /b 2)
if not exist "%DAGGEN%\ckpt_last.pt" (echo REFUSED: %DAGGEN%\ckpt_last.pt is missing & exit /b 2)
if not exist "%DAGGEN%\argv.json" (echo REFUSED: %DAGGEN%\argv.json is missing & exit /b 2)
if not exist "%DAGBANK%" mkdir "%DAGBANK%"
echo [%date% %time%] bank flat s%~2
%BX% >> "%DAGBANK%\bank_stage.log" 2>&1
if errorlevel 1 (echo [FAIL] bank flat s%~2 - see %DAGBANK%\bank_stage.log & exit /b 1)
if not exist "%DAGBANK%\bank.json" (echo [FAIL] bank flat s%~2 exited 0 without bank.json & exit /b 1)
echo [ok] bank flat s%~2
exit /b 0
:dagorcarm
rem PREREG_V8F_DAG_ORC section 1: reached ONLY from :dagarm (never called, never dispatched) with DAGS = dagorc | dagorcrep | dagorc01 and :dagname's names set. %1 = ARM, %2 = SEED (0 | 1: the seed of DAG_s<SEED>, the init), %3 = nothing: the doses are fixed here and any extra argument is REFUSED (rc 2).
rem The fine-tune is DAG2-0's banked argv (PREREG_V8F_DAGGER2's dag20 arm): init DAG_s<SEED> (M2b_s<SEED>_dag), --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2, --ego-perturb-p NAMED and --dagger-p 0 NAMED (the swap is off; the trainer's gate refuses a DAgger flag set while it is 0). No bank is read, hashed or named.
rem dagorc (A) = that with --ego-perturb-p 0.3 followed by O-RECOVER's five law flags at their registered values (vscale 0.5 1.0, vmax 8.0, adrop 1.5, yawrate-std 2.0, seed = SEED: the :orcarm law). dagorc01 (B) = A with --ego-perturb-p 0.1. dagorcrep (REP) = DAG2-0's argv with --seed 1<SEED> (10 / 11) and p 0: the replicate, still initialised from DAG_s<SEED>.
rem Names (all from :dagname, indexed by the init seed SEED, REP too): run M2b_s<SEED>_<ARM>, tag M2b-<ARM>_s<SEED>, gate directory gate_M2b-<ARM>_s<SEED>, pin dagpin\M2b_s<SEED>_<ARM>.train.argv.txt. REP's trainer seed is DAGSEED, set here: the gate and train lines carry --seed 10 / 11 (their progress messages print that seed).
set "DAGORCP=0"
set "DAGORCL="
if not "%~3"=="" (echo REFUSED: %DAGS% takes no P, the dose is fixed by PREREG_V8F_DAG_ORC, got "%~3" & exit /b 2)
if /i "%DAGS%"=="dagorc" (set "DAGORCP=0.3" & set "DAGORCL= --ego-perturb-vscale 0.5 1.0 --ego-perturb-vmax 8.0 --ego-perturb-adrop 1.5 --ego-perturb-yawrate-std 2.0 --ego-perturb-seed %~2")
if /i "%DAGS%"=="dagorc01" (set "DAGORCP=0.1" & set "DAGORCL= --ego-perturb-vscale 0.5 1.0 --ego-perturb-vmax 8.0 --ego-perturb-adrop 1.5 --ego-perturb-yawrate-std 2.0 --ego-perturb-seed %~2")
if /i "%DAGS%"=="dagorcrep" set "DAGSEED=1%~2"
set "DAGF=--init-ckpt "%RUNSM%\M2b_s%~2_dag\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %DAGORCP%%DAGORCL% --dagger-p 0"
exit /b 0
:dagoaccorcarm
rem PREREG_V8F_DAG_ORC Amendment 1 section 1: reached ONLY from :dagarm (never called, never dispatched) with DAGS = dagoaccorc | dagoaccorcrep and :dagname's names set. %1 = ARM, %2 = SEED (0 | 1: the seed of M2b_s<SEED>, the init), %3 = nothing: the dose is fixed (B is not queued).
rem The fine-tune is DAG-OACC's banked argv VERBATIM (PREREG_V8F_DAG_OACC: init M2b_s<SEED>, 3,000 steps, lr 1e-4, warmup 100, p_gt 0.2 -> 0.2, --ego-perturb-p NAMED, --dagger-p 0.1, the ITERATION-1 bank %RUNSM%\dagger_bank\s<SEED> with its sha256 computed HERE, --dagger-seed <SEED>, --w-acc <the waccdag value>), obtained by CALLING :dagoaccarm (untouched): DAGP, DAGSHA, WACCDAG and DAGBANK come from it.
rem dagoaccorc (A) = that with --ego-perturb-p 0.3 followed by O-RECOVER's five law flags at their registered values (vscale 0.5 1.0, vmax 8.0, adrop 1.5, yawrate-std 2.0, seed = SEED) and the bare flag --ego-perturb-with-dagger at the END (the swap is drawn first, the perturbation only on samples the swap did not replace). dagoaccorcrep (REP) = DAG-OACC's argv with the trainer seed 10 / 11 (DAGSEED = 1<SEED>; --dagger-seed stays SEED), no perturbation, no flag.
rem Names (all from :dagname, indexed by the init seed SEED, REP too): run M2b_s<SEED>_<ARM>, tag M2b-<ARM>_s<SEED>, gate directory gate_M2b-<ARM>_s<SEED>, pin dagpin\M2b_s<SEED>_<ARM>.train.argv.txt.
set "DAGORCP=0"
set "DAGORCL="
set "DAGORCX="
if not "%~3"=="" (echo REFUSED: %DAGS% takes no P, the dose is fixed by PREREG_V8F_DAG_ORC Amendment 1, got "%~3" & exit /b 2)
call :dagoaccarm %~1 %~2 0.1 || exit /b 2
if /i "%DAGS%"=="dagoaccorc" (set "DAGORCP=0.3" & set "DAGORCL= --ego-perturb-vscale 0.5 1.0 --ego-perturb-vmax 8.0 --ego-perturb-adrop 1.5 --ego-perturb-yawrate-std 2.0 --ego-perturb-seed %~2" & set "DAGORCX= --ego-perturb-with-dagger")
if /i "%DAGS%"=="dagoaccorcrep" set "DAGSEED=1%~2"
set "DAGF=--init-ckpt "%RUNSM%\M2b_s%~2\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %DAGORCP%%DAGORCL% --dagger-p %DAGP% --dagger-bank "%DAGBANK%" --dagger-bank-sha256 %DAGSHA% --dagger-seed %~2 --w-acc %WACCDAG%%DAGORCX%"
exit /b 0
