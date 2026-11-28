"""Apply the O-RECOVER stages to v8f_mini_box2_queue.cmd (CRLF-safe). Usage: apply_queue_edit.py <queue file>  (edits in place; refuses a file that already has them)."""
import sys

p = sys.argv[1]
with open(p, "rb") as fh:
    raw = fh.read()
assert b"\r\n" in raw and raw.count(b"\r\n") == raw.count(b"\n"), "the queue is CRLF"
t = raw.decode("utf-8").replace("\r\n", "\n")
assert "gateorc" not in t, "already applied"

# ---- 1. header: three lines after fitoacc's
H_ANCHOR = "rem   v8f_mini_box2_queue.cmd fitoacc SEED"
i = t.index(H_ANCHOR)
j = t.index("\n", i) + 1
header = (
    "rem\n"
    "rem   v8f_mini_box2_queue.cmd gateorc ARM SEED        PREREG_V8F_ORECOVER (Amendments 1 and 2): the in-process launch gate for ONE fine-tune run, ARM orc (--ego-perturb-p 0.3) or orc0 (--ego-perturb-p 0), SEED 0 or 1: the argv of\n"
    "rem                                             its training run with --out and --selfcheck-only changed, nothing else. Four gates in all (orc/orc0 x seed 0/1), no training. The fine-tune starts from the BANKED M2b_s<SEED> checkpoint\n"
    "rem   v8f_mini_box2_queue.cmd trainorc ARM SEED       PREREG_V8F_ORECOVER: one fine-tune run into %RUNSM%\\M2b_s<SEED>_orc / _orc0 (resumes; skipped once done). ALL FOUR runs are always trained: the control ORC-0 is not optional\n"
    "rem   v8f_mini_box2_queue.cmd evalorc ARM SEED        PREREG_V8F_ORECOVER: that run's standard T0 eval (--project --families), tagged M2b-orc_s<SEED> / M2b-orc0_s<SEED>; skipped once its npz exists. The T1-kin rollouts run on the dev box, not here\n"
)
t = t[:j] + header + t[j:]

# ---- 2. dispatch: three lines after fitoacc's
D_ANCHOR = 'if /i "%~1"=="fitoacc" goto :fitoacc\n'
assert t.count(D_ANCHOR) == 1
t = t.replace(D_ANCHOR, D_ANCHOR + 'if /i "%~1"=="gateorc" goto :gateorc\nif /i "%~1"=="trainorc" goto :trainorc\nif /i "%~1"=="evalorc" goto :evalorc\n')

# ---- 3. the stages, appended after the file's last line
assert t.endswith("echo [ok] HX weights M2b-oacc_s%~2\nexit /b 0\n")
stages = r''':orcarm
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
'''
t = t + stages
with open(p, "wb") as fh:
    fh.write(t.replace("\n", "\r\n").encode("utf-8"))
print("ok", len(t))
