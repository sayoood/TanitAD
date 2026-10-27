#!/usr/bin/env python3
"""Build the staged box-2 queue for PREREG_V8F_DAG_OACC and PREREG_V8F_FLATDAG: the repo tip's v8f_mini_box2_queue.cmd (1d4f4fbb, which already carries PREREG_V8F_DAGGER's and DAGGER2's stages) + the additions below,
INSERTIONS ONLY.

    tip/queue_1d4f4fbb.cmd  (git show 1d4f4fbb:<PK>/code/v8f_mini_box2_queue.cmd, CRLF)
  + 4 doc lines          after the `gatedag / traindag / evaldag also take ARM dag2` documentation line  (waccdag, bankflat, arms dagoacc, arms flatdag / flatdag0)
  + 2 dispatch lines     after the `bankdag2` dispatch line                                              (waccdag, bankflat)
  + 3 lines in :dagname  accept ARM dagoacc / flatdag / flatdag0                                         (after the `dag20` acceptance line)
  + 2 lines in :dagname  the FLAT bank directory for flatdag / flatdag0                                  (after the `dag20` DAGBANK line)
  + 3 lines in :dagarm   hand dagoacc / flatdag / flatdag0 to :dagoaccarm / :flatdagarm                  (after the `dag20` hand-off line)
  + 2 lines in :evaldag  hand flatdag / flatdag0 to :evalflatdag                                         (after its `call :dagname` line)
  + :dagoaccarm :dagoaccarm_dry :dagoaccarm_have :dagoaccwacc :dagoaccwacc_dry :dagoaccwacc_have :flatdagarm :flatdagarm_dry :flatdagarm_have :evalflatdag :evalflatdag_1 :evalflatdag_2 :waccdag :bankflat
    appended at the end of the file

No existing line is changed or removed (queue.diff has 0 removed lines; the selftest re-derives the file and asserts it). Every anchor must occur EXACTLY once; every output line keeps the CRLF the tip uses.
"""
from __future__ import annotations

import difflib
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
TIP = HERE / "tip" / "queue_1d4f4fbb.cmd"
OUT = HERE / "stage" / PK / "code" / "v8f_mini_box2_queue.cmd"
TIP_SHA256 = "1e038f8bca8d016d5799ee2888d4c15eac8684a200085e4480ca45153e4015c0"      # the tip's file (1d4f4fbb; also the DAGGER2 build's staged output)

DAG_S0_CKPT_SHA = "d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1"      # v8f-dagger/M2b_s0_dag/SHA256SUMS.txt: the banked DAG_s0 (the rule's B)

DOC = [
    'rem   v8f_mini_box2_queue.cmd waccdag              PREREG_V8F_DAG_OACC section 1: the w_acc RULE of waccoacc with B = DAG: w_acc = 0.5 * L_traj(DAG_s0) / L_acc(DAG_s0) over HBLEND\'s fit windows, DAG_s0\'s checkpoint (--expect-sha256 pinned), CPU, cross-checked against DAG_s0\'s banked fit windows (rel 1e-3, else REFUSED). Writes %RUNSM%\\dagoacc\\w_acc.json and w_acc.txt (its OWN directory, never OACC\'s %RUNSM%\\oacc). Skipped once both exist',
    'rem   v8f_mini_box2_queue.cmd bankflat SEED         PREREG_V8F_FLATDAG section 1: the own-state bank of FLAT_s<SEED> (M2b_s<SEED>_flat\\ckpt_last.pt, frozen, PLAIN, one pass, TRAIN only), the SAME rollout configuration and expectations as bankdag: four clip-sharded generators then the merge, into %RUNSM%\\dagger_bank_flat\\s<SEED>. Needs FLAT_s<SEED> FINISHED (summary.json)',
    'rem   gatedag / traindag / evaldag also take ARM dagoacc (DAG\'s fine-tune argv verbatim + --w-acc <the waccdag value>; P is fixed at 0.1, any other P is REFUSED; same init M2b_s<SEED>, same iteration-1 bank and sha256; own run directory M2b_s<SEED>_dagoacc): PREREG_V8F_DAG_OACC section 1',
    'rem   gatedag / traindag / evaldag also take ARM flatdag (FLAT_s<SEED> fine-tuned with DAG\'s recipe, --o-interface cut named, --dagger-p 0.1 on the FLAT bank; P fixed at 0.1) and flatdag0 (the same argv with --dagger-p 0; takes no P); evaldag runs the standard eval AND the GT-code oracle (the cut identity read) for them: PREREG_V8F_FLATDAG section 1',
]
DISPATCH = ['if /i "%~1"=="waccdag" goto :waccdag', 'if /i "%~1"=="bankflat" goto :bankflat']
DAGNAME_ARMS = ['if /i "%~1"=="dagoacc" set "DAGS=dagoacc"', 'if /i "%~1"=="flatdag" set "DAGS=flatdag"', 'if /i "%~1"=="flatdag0" set "DAGS=flatdag0"']
DAGNAME_BANK = ['if /i "%DAGS%"=="flatdag" set "DAGBANK=%RUNSM%\\dagger_bank_flat\\s%~2"', 'if /i "%DAGS%"=="flatdag0" set "DAGBANK=%RUNSM%\\dagger_bank_flat\\s%~2"']
DAGARM_HAND = ['if /i "%DAGS%"=="dagoacc" goto :dagoaccarm', 'if /i "%DAGS%"=="flatdag" goto :flatdagarm', 'if /i "%DAGS%"=="flatdag0" goto :flatdagarm']
EVALDAG_HAND = ['if /i "%DAGS%"=="flatdag" goto :evalflatdag', 'if /i "%DAGS%"=="flatdag0" goto :evalflatdag']

STAGES = r'''
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
set "WX="%PY%" "%EVALOACC%" wacc --ckpt "%RUNSM%\M2b_s0_dag\ckpt_last.pt" --expect-sha256 @@DAGSHA@@ --fit-clips "%OD%\fit_clips.json" --cache-dir "%CACHEFULL%" --join-train "%RUNSM%\join_train.npz" --tracks "%TRACKS%" --banked-fit-windows "%OD%\fit_windows_DAG_s0.npz" --out "%OD%\w_acc.json" --out-txt "%OD%\w_acc.txt""
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
'''.replace("@@DAGSHA@@", DAG_S0_CKPT_SHA)


def build() -> bytes:
    tip = TIP.read_bytes()
    if hashlib.sha256(tip).hexdigest() != TIP_SHA256:
        sys.exit("the tip queue is not the pinned 1d4f4fbb file")
    text = tip.decode("utf-8")
    assert "\r\n" in text and text.endswith("\r\n") and text.count("\n") == text.count("\r\n")
    L = text[:-2].split("\r\n")

    def only(pred, what):
        idx = [i for i, x in enumerate(L) if pred(x)]
        if len(idx) != 1:
            sys.exit(f"anchor {what}: {idx} - each must occur exactly once")
        return idx[0]

    ins = {
        only(lambda x: x.startswith("rem   gatedag / traindag / evaldag also take ARM dag2 "), "doc"): DOC,
        only(lambda x: x == 'if /i "%~1"=="bankdag2" goto :bankdag2', "dispatch"): DISPATCH,
        only(lambda x: x == 'if /i "%~1"=="dag20" set "DAGS=dag20"', "dagname arms"): DAGNAME_ARMS,
        only(lambda x: x == 'if /i "%DAGS%"=="dag20" set "DAGBANK=%RUNSM%\\dagger_bank_dag2\\s%~2"', "dagname bank"): DAGNAME_BANK,
        only(lambda x: x == 'if /i "%DAGS%"=="dag20" goto :dag2arm', "dagarm hand-off"): DAGARM_HAND,
        only(lambda x: x == 'call :dagname %~2 %~3 || exit /b 2', "evaldag hand-off"): EVALDAG_HAND,
    }
    new = []
    for i, x in enumerate(L):
        new.append(x)
        new.extend(ins.get(i, []))
    new.extend(STAGES.strip("\n").split("\n"))
    return ("\r\n".join(new) + "\r\n").encode("utf-8")


def main() -> int:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)
    a = TIP.read_bytes().decode("utf-8")[:-2].split("\r\n")
    b = data.decode("utf-8")[:-2].split("\r\n")
    diff = list(difflib.unified_diff(a, b, "a/" + PK + "/code/v8f_mini_box2_queue.cmd", "b/" + PK + "/code/v8f_mini_box2_queue.cmd", lineterm="", n=2))
    (HERE / "queue.diff").write_bytes(("\n".join(diff) + "\n").encode("utf-8"))
    removed = [x for x in diff if x.startswith("-") and not x.startswith("---")]
    added = [x for x in diff if x.startswith("+") and not x.startswith("+++")]
    ncrlf, nlf = data.count(b"\r\n"), data.count(b"\n")
    print(f"wrote {OUT}  sha256 {hashlib.sha256(data).hexdigest()}  lines {len(b)} (tip {len(a)})  added {len(added)}  removed {len(removed)}  CRLF lines {ncrlf} of {nlf}")
    return 1 if removed else 0


if __name__ == "__main__":
    sys.exit(main())
