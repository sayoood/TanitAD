#!/usr/bin/env python3
"""Build the staged box-2 queue for PREREG_V8F_DAG_ORC: the repo tip's v8f_mini_box2_queue.cmd (3b7b404e: it already carries PREREG_V8F_DAGGER2's, DAG-OACC's and FLAT-DAG's stages) + the DAG-ORC additions, INSERTIONS ONLY.

    tip/queue_3b7b404e.cmd  (git show 3b7b404e:<PK>/code/v8f_mini_box2_queue.cmd, CRLF, sha256 6d26a613...)
  + 1 doc line           after the `gatedag / traindag / evaldag also take ARM flatdag` documentation line (the later edits' usage-line block)
  + 3 lines in :dagname  accept ARM dagorc / dagorcrep / dagorc01   (after the `flatdag0` acceptance line)
  + 3 lines in :dagarm   hand the three arms to :dagorcarm          (after the `flatdag0` hand-off line)
  + :dagorcarm appended at the end of the file

No existing line is changed or removed (queue.diff has 0 removed lines; the selftest re-derives the file and asserts it). Every anchor must occur EXACTLY once; every output line keeps the CRLF the tip uses.
The three arms ride the existing gatedag / traindag / evaldag stages (no new dispatched stage): the run directory, tag, gate directory and pin are :dagname's, the standard eval is :evaldag's.
"""
from __future__ import annotations

import difflib
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
TIP = HERE / "tip" / "queue_3b7b404e.cmd"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "stage" / PK / "code" / "v8f_mini_box2_queue.cmd"
TIP_SHA256 = "6d26a613c1bcd6c1c7cc018be65904d3b456623e3a049718a038a0a31a9d2262"      # the tip's file (a later commit that changed the queue means: rebuild against it, do not land this file)

DOC = [
    'rem   gatedag / traindag / evaldag also take ARM dagorc (DAG_s<SEED> fine-tuned with DAG2-0\'s argv plus O-RECOVER\'s ego perturbation at p 0.3 and its five law flags), dagorcrep (DAG2-0\'s argv with --seed 10 / 11, the replicate; no perturbation) and dagorc01 (dagorc at p 0.1); none takes a P (any extra argument is REFUSED): PREREG_V8F_DAG_ORC section 1. Init = %RUNSM%\\M2b_s<SEED>_dag\\ckpt_last.pt; run M2b_s<SEED>_<ARM>; tag M2b-<ARM>_s<SEED>; pin dagpin\\M2b_s<SEED>_<ARM>.train.argv.txt',
]
DAGNAME_ARMS = ['if /i "%~1"=="dagorc" set "DAGS=dagorc"', 'if /i "%~1"=="dagorcrep" set "DAGS=dagorcrep"', 'if /i "%~1"=="dagorc01" set "DAGS=dagorc01"']
DAGARM_HAND = ['if /i "%DAGS%"=="dagorc" goto :dagorcarm', 'if /i "%DAGS%"=="dagorcrep" goto :dagorcarm', 'if /i "%DAGS%"=="dagorc01" goto :dagorcarm']
LAW = 'set "DAGORCL= --ego-perturb-vscale 0.5 1.0 --ego-perturb-vmax 8.0 --ego-perturb-adrop 1.5 --ego-perturb-yawrate-std 2.0 --ego-perturb-seed %~2"'

STAGES = r'''
:dagorcarm
rem PREREG_V8F_DAG_ORC section 1: reached ONLY from :dagarm (never called, never dispatched) with DAGS = dagorc | dagorcrep | dagorc01 and :dagname's names set. %1 = ARM, %2 = SEED (0 | 1: the seed of DAG_s<SEED>, the init), %3 = nothing: the doses are fixed here and any extra argument is REFUSED (rc 2).
rem The fine-tune is DAG2-0's banked argv (PREREG_V8F_DAGGER2's dag20 arm): init DAG_s<SEED> (M2b_s<SEED>_dag), --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2, --ego-perturb-p NAMED and --dagger-p 0 NAMED (the swap is off; the trainer's gate refuses a DAgger flag set while it is 0). No bank is read, hashed or named.
rem dagorc (A) = that with --ego-perturb-p 0.3 followed by O-RECOVER's five law flags at their registered values (vscale 0.5 1.0, vmax 8.0, adrop 1.5, yawrate-std 2.0, seed = SEED: the :orcarm law). dagorc01 (B) = A with --ego-perturb-p 0.1. dagorcrep (REP) = DAG2-0's argv with --seed 1<SEED> (10 / 11) and p 0: the replicate, still initialised from DAG_s<SEED>.
rem Names (all from :dagname, indexed by the init seed SEED, REP too): run M2b_s<SEED>_<ARM>, tag M2b-<ARM>_s<SEED>, gate directory gate_M2b-<ARM>_s<SEED>, pin dagpin\M2b_s<SEED>_<ARM>.train.argv.txt. REP's trainer seed is DAGSEED, set here: the gate and train lines carry --seed 10 / 11 (their progress messages print that seed).
set "DAGORCP=0"
set "DAGORCL="
if not "%~3"=="" (echo REFUSED: %DAGS% takes no P, the dose is fixed by PREREG_V8F_DAG_ORC, got "%~3" & exit /b 2)
if /i "%DAGS%"=="dagorc" (set "DAGORCP=0.3" & LAWLINE)
if /i "%DAGS%"=="dagorc01" (set "DAGORCP=0.1" & LAWLINE)
if /i "%DAGS%"=="dagorcrep" set "DAGSEED=1%~2"
set "DAGF=--init-ckpt "%RUNSM%\M2b_s%~2_dag\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %DAGORCP%%DAGORCL% --dagger-p 0"
exit /b 0
'''.replace("LAWLINE", LAW)


def build() -> bytes:
    tip = TIP.read_bytes()
    if hashlib.sha256(tip).hexdigest() != TIP_SHA256:
        sys.exit("the tip queue is not the pinned 3b7b404e file")
    text = tip.decode("utf-8")
    assert "\r\n" in text and text.endswith("\r\n") and text.count("\n") == text.count("\r\n")
    L = text[:-2].split("\r\n")

    def only(pred, what):
        idx = [i for i, x in enumerate(L) if pred(x)]
        if len(idx) != 1:
            sys.exit(f"anchor {what}: {idx} - each must occur exactly once")
        return idx[0]

    ins = {
        only(lambda x: x.startswith("rem   gatedag / traindag / evaldag also take ARM flatdag "), "doc"): DOC,
        only(lambda x: x == 'if /i "%~1"=="flatdag0" set "DAGS=flatdag0"', "dagname arms"): DAGNAME_ARMS,
        only(lambda x: x == 'if /i "%DAGS%"=="flatdag0" goto :flatdagarm', "dagarm hand-off"): DAGARM_HAND,
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
    if OUT == HERE / "stage" / PK / "code" / "v8f_mini_box2_queue.cmd":
        (HERE / "queue.diff").write_bytes(("\n".join(diff) + "\n").encode("utf-8"))
    removed = [x for x in diff if x.startswith("-") and not x.startswith("---")]
    added = [x for x in diff if x.startswith("+") and not x.startswith("+++")]
    ncrlf, nlf = data.count(b"\r\n"), data.count(b"\n")
    print(f"wrote {OUT}  sha256 {hashlib.sha256(data).hexdigest()}  lines {len(b)} (tip {len(a)})  added {len(added)}  removed {len(removed)}  CRLF lines {ncrlf} of {nlf}")
    return 1 if removed else 0


if __name__ == "__main__":
    sys.exit(main())
