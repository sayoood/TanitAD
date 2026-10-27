#!/usr/bin/env python3
"""Build the staged box-2 queue for PREREG_V8F_DAGGER2: the repo tip's v8f_mini_box2_queue.cmd (8e121ea8, which already carries PREREG_V8F_DAGGER's stages) + the DAGGER2 additions, INSERTIONS ONLY.

    tip/queue_8e121ea8.cmd  (git show 8e121ea8:<PK>/code/v8f_mini_box2_queue.cmd, CRLF)
  + 2 doc lines          after the `evaldag` documentation line
  + 1 dispatch line      after the `evaldag` dispatch line         (bankdag2)
  + 2 lines in :dagname  accept ARM dag2 / dag20                  (after the `dag0` acceptance line)
  + 2 lines in :dagname  the iteration-2 bank directory for them  (after the DAGBANK line)
  + 2 lines in :dagarm   hand dag2 / dag20 to :dag2arm            (after its `call :dagname` line)
  + :dag2arm and :bankdag2 appended at the end of the file

No existing line is changed or removed (queue.diff has 0 removed lines; the selftest re-derives the file and asserts it). Every anchor must occur EXACTLY once; every output line keeps the CRLF the tip uses.
"""
from __future__ import annotations

import difflib
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
TIP = HERE / "tip" / "queue_8e121ea8.cmd"
OUT = HERE / "stage" / PK / "code" / "v8f_mini_box2_queue.cmd"
TIP_SHA256 = "1519260af238ea0da601fcd5bca10d30fc974786d748f03d3bc3eeb73fd1d67e"      # the tip's file (also the DAG build's staged output: nothing landed after it changed the queue)

DOC = [
    'rem   v8f_mini_box2_queue.cmd bankdag2 SEED             PREREG_V8F_DAGGER2 section 1: the own-state bank of DAG_s<SEED> (M2b_s<SEED>_dag\\ckpt_last.pt, frozen, PLAIN, one pass, TRAIN only), the SAME rollout configuration and expectations as bankdag: four clip-sharded generators then the merge, into %RUNSM%\\dagger_bank_dag2\\s<SEED> (never the iteration-1 dagger_bank). Skipped once bank.json and PROVENANCE.json exist',
    'rem   gatedag / traindag / evaldag also take ARM dag2 (DAG_s<SEED> fine-tuned with --dagger-p 0.1 on the iteration-2 bank; P is fixed at 0.1, any other P is REFUSED) and dag20 (the same argv with --dagger-p 0; takes no P): PREREG_V8F_DAGGER2 section 4. Init = %RUNSM%\\M2b_s<SEED>_dag\\ckpt_last.pt; run M2b_s<SEED>_dag2 / _dag20; tag M2b-dag2_s<SEED> / M2b-dag20_s<SEED>; pin dagpin\\M2b_s<SEED>_dag2.train.argv.txt',
]
DISPATCH = ['if /i "%~1"=="bankdag2" goto :bankdag2']
DAGNAME_ARMS = ['if /i "%~1"=="dag2" set "DAGS=dag2"', 'if /i "%~1"=="dag20" set "DAGS=dag20"']
DAGNAME_BANK = ['if /i "%DAGS%"=="dag2" set "DAGBANK=%RUNSM%\\dagger_bank_dag2\\s%~2"', 'if /i "%DAGS%"=="dag20" set "DAGBANK=%RUNSM%\\dagger_bank_dag2\\s%~2"']
DAGARM_HAND = ['if /i "%DAGS%"=="dag2" goto :dag2arm', 'if /i "%DAGS%"=="dag20" goto :dag2arm']

STAGES = r'''
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
'''


def build() -> bytes:
    tip = TIP.read_bytes()
    if hashlib.sha256(tip).hexdigest() != TIP_SHA256:
        sys.exit("the tip queue is not the pinned 8e121ea8 file")
    text = tip.decode("utf-8")
    assert "\r\n" in text and text.endswith("\r\n") and text.count("\n") == text.count("\r\n")
    L = text[:-2].split("\r\n")

    def only(pred, what):
        idx = [i for i, x in enumerate(L) if pred(x)]
        if len(idx) != 1:
            sys.exit(f"anchor {what}: {idx} - each must occur exactly once")
        return idx[0]

    ins = {
        only(lambda x: x.startswith("rem   v8f_mini_box2_queue.cmd evaldag ARM SEED "), "doc"): DOC,
        only(lambda x: x == 'if /i "%~1"=="evaldag" goto :evaldag', "dispatch"): DISPATCH,
        only(lambda x: x == 'if /i "%~1"=="dag0" set "DAGS=dag0"', "dagname arms"): DAGNAME_ARMS,
        only(lambda x: x == 'set "DAGBANK=%RUNSM%\\dagger_bank\\s%~2"', "dagname bank"): DAGNAME_BANK,
        only(lambda x: x == 'call :dagname %~1 %~2 || exit /b 2', "dagarm hand-off"): DAGARM_HAND,
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
