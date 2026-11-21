#!/usr/bin/env python3
"""Build the staged box-2 queue for PREREG_V8F_DAG_ORC re-based on DAG-OACC (Amendment 1): the repo tip's v8f_mini_box2_queue.cmd + the DAG-OACC-ORC additions. INSERTION ONLY.

    python build_queue.py [OUT]       writes ./stage/<PK>/code/v8f_mini_box2_queue.cmd (CRLF) and ./queue.diff

    tip/queue_bf404bfe.cmd  (git show 6cd741b6:<PK>/code/v8f_mini_box2_queue.cmd, CRLF, sha256 bf404bfe...; unchanged at 9ba6b6a1)
  + 1 doc line           after the `gatedag / traindag / evaldag also take ARM dagorc` documentation line
  + 2 lines in :dagname  accept ARM dagoaccorc / dagoaccorcrep      (after the `dagorc01` acceptance line)
  + 2 lines in :dagarm   hand the two arms to :dagoaccorcarm        (after the `dagorc01` hand-off line)
  + :dagoaccorcarm appended at the end of the file

No existing line is changed or removed (queue.diff has 0 removed lines; the chain self-test re-derives the file and asserts it). Every anchor must occur EXACTLY once; every output line keeps the CRLF the tip has.
The two arms ride the existing gatedag / traindag / evaldag stages. B (p 0.1) is NOT queued (Amendment 1 section 4), so there is no third arm and no P argument.
:dagoaccorcarm CALLS :dagoaccarm (DAG-OACC's own stage helper, untouched) for the bank sha256, the w_acc and DAGP, and only then writes DAGF in the SAME layout; the one-lever rule of the gate asserter
compares the result against DAG-OACC's banked argv.json token for token, so a layout drift is a refusal, not a silent change.
"""
from __future__ import annotations

import difflib
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
TIP = HERE / "tip" / "queue_bf404bfe.cmd"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "stage" / PK / "code" / "v8f_mini_box2_queue.cmd"
TIP_SHA256 = "bf404bfe6a6d75ad5b14615802396faa70e625e743306b9df37b3e77fe2694ab"      # the tip's file (a later commit that changed the queue means: rebuild against it, do not land this file)

DOC = [
    "rem   gatedag / traindag / evaldag also take ARM dagoaccorc (M2b_s<SEED> fine-tuned with DAG-OACC's argv verbatim plus O-RECOVER's ego perturbation at p 0.3 with its five law flags, composed with the DAgger swap by "
    "--ego-perturb-with-dagger) and dagoaccorcrep (DAG-OACC's argv with --seed 10 / 11, the replicate; no perturbation); neither takes a P (any extra argument is REFUSED): PREREG_V8F_DAG_ORC Amendment 1 section 1. "
    "Init = %RUNSM%\\M2b_s<SEED>\\ckpt_last.pt; run M2b_s<SEED>_<ARM>; tag M2b-<ARM>_s<SEED>; pin dagpin\\M2b_s<SEED>_<ARM>.train.argv.txt",
]
DAGNAME_ARMS = ['if /i "%~1"=="dagoaccorc" set "DAGS=dagoaccorc"', 'if /i "%~1"=="dagoaccorcrep" set "DAGS=dagoaccorcrep"']
DAGARM_HAND = ['if /i "%DAGS%"=="dagoaccorc" goto :dagoaccorcarm', 'if /i "%DAGS%"=="dagoaccorcrep" goto :dagoaccorcarm']
LAW = 'set "DAGORCL= --ego-perturb-vscale 0.5 1.0 --ego-perturb-vmax 8.0 --ego-perturb-adrop 1.5 --ego-perturb-yawrate-std 2.0 --ego-perturb-seed %~2"'

STAGES = r'''
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
if /i "%DAGS%"=="dagoaccorc" (set "DAGORCP=0.3" & LAWLINE & set "DAGORCX= --ego-perturb-with-dagger")
if /i "%DAGS%"=="dagoaccorcrep" set "DAGSEED=1%~2"
set "DAGF=--init-ckpt "%RUNSM%\M2b_s%~2\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %DAGORCP%%DAGORCL% --dagger-p %DAGP% --dagger-bank "%DAGBANK%" --dagger-bank-sha256 %DAGSHA% --dagger-seed %~2 --w-acc %WACCDAG%%DAGORCX%"
exit /b 0
'''.replace("LAWLINE", LAW)


def build() -> bytes:
    tip = TIP.read_bytes()
    if hashlib.sha256(tip).hexdigest() != TIP_SHA256:
        sys.exit("the tip queue is not the pinned 6cd741b6 file")
    text = tip.decode("utf-8")
    assert "\r\n" in text and text.endswith("\r\n") and text.count("\n") == text.count("\r\n")
    L = text[:-2].split("\r\n")

    def only(pred, what):
        idx = [i for i, x in enumerate(L) if pred(x)]
        if len(idx) != 1:
            sys.exit(f"anchor {what}: {idx} - each must occur exactly once")
        return idx[0]

    ins = {
        only(lambda x: x.startswith("rem   gatedag / traindag / evaldag also take ARM dagorc "), "doc"): DOC,
        only(lambda x: x == 'if /i "%~1"=="dagorc01" set "DAGS=dagorc01"', "dagname arms"): DAGNAME_ARMS,
        only(lambda x: x == 'if /i "%DAGS%"=="dagorc01" goto :dagorcarm', "dagarm hand-off"): DAGARM_HAND,
    }
    new = []
    for i, x in enumerate(L):
        new.append(x)
        new.extend(ins.get(i, []))
    new.extend(STAGES.strip("\n").split("\n"))
    out = ("\r\n".join(new) + "\r\n").encode("utf-8")
    return out


def main() -> int:
    out = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(out)
    tip = TIP.read_bytes().decode("utf-8").split("\r\n")
    mine = out.decode("utf-8").split("\r\n")
    tip, mine = tip[:-1], mine[:-1]      # the empty element after the final CRLF
    diff = list(difflib.unified_diff(tip, mine, "a/" + PK + "/code/v8f_mini_box2_queue.cmd", "b/" + PK + "/code/v8f_mini_box2_queue.cmd", lineterm="", n=1))
    removed = [d for d in diff if d.startswith("-") and not d.startswith("---")]
    assert not removed, removed
    (HERE / "queue.diff").write_text("\n".join(diff) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT} sha256 {hashlib.sha256(out).hexdigest()} ({len(mine)} lines; +{sum(1 for d in diff if d.startswith('+') and not d.startswith('+++'))} -0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
