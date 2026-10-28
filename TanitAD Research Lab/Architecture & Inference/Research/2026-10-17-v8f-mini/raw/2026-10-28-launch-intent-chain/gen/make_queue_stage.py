#!/usr/bin/env python3
"""Stage the box-2 queue with the arms `launchintent` and `launchintentrep` (PREREG_V8F_LAUNCH_INTENT section 4 as REDESIGNED by MM_REDESIGN_2026-10-28 section 3: four runs).

INPUT   the LANDED commit's queue blob  `git show b09ac89f:<PK>/code/v8f_mini_box2_queue.cmd`  (read-only on the mirror; byte-identical to the pre-lever tip 94a47f4b's: blob a45dedf2).
OUTPUT  queue_stage/v8f_mini_box2_queue.cmd   (CRLF, the file to LAND at that repo path)  +  queue_stage/queue.diff  +  queue_stage/queue_tip_<commit>.cmd (the input, for `diff`).

THREE EDITS, each an exact-count replacement, and NOTHING else (every other stage is byte-identical; the self-test proves it line by line):
  1. :dagname   + `if /i "%~1"=="launchintent" set "DAGS=launchintent"` and the same for launchintentrep   (the arm-name whitelist: anything else is REFUSED rc 2)
  2. :dagarm    + `if /i "%DAGS%"=="launchintent" goto :launchintentarm` and the same for launchintentrep (the dispatch; :launchintentarm is reached ONLY from here)
  3. appended   :launchintentarm  = CALL :dagoaccarm (untouched: DAG-OACC's banked argv verbatim), `set "DAGSEED=1%~2"` for launchintentrep ONLY (REP's rule: the trainer seed is 10 + S, --dagger-seed stays S),
                then `--launch-intent-w 1.58926` at the END, nothing else.
The ARM LABEL carries the run: launchintent 0 / launchintent 1 are A_0 / A_1 (pair control DAG-OACC_s), launchintentrep 0 / launchintentrep 1 are A_rep,0 / A_rep,1 (pair control REP_s = DAG-ORC r2's M2b_s<S>_dagoaccorcrep).
W is COMPUTED from the measured launch fraction f = 49,618 / 306,566 (f_measure.json) by PREREG section 3's rule (1 + W = (1 - f) / (2 f), 6 significant digits) and asserted equal to the registered 1.58926.

    python make_queue_stage.py          writes queue_stage/ (CRLF)
"""
from __future__ import annotations

import difflib
import sys

import common as C

W = C.w_from_rule(1, 3, C.f_measured())
assert W == C.LI_W, (W, C.LI_W)
assert C.w_from_rule(1, 4, C.f_measured()) == C.LI_W_B and C.w_from_rule(1, 2, C.f_measured()) == C.LI_W_HI

NAME_ANCHOR = 'if /i "%~1"=="dagoaccorcrep" set "DAGS=dagoaccorcrep"\r\n'
DISP_ANCHOR = 'if /i "%DAGS%"=="dagoaccorcrep" goto :dagoaccorcarm\r\n'
NAME_ADD = ''.join(f'if /i "%~1"=="{a}" set "DAGS={a}"\r\n' for a in C.LI_ARMS)
DISP_ADD = ''.join(f'if /i "%DAGS%"=="{a}" goto :launchintentarm\r\n' for a in C.LI_ARMS)

STAGE_TEXT = f'''\
:launchintentarm
rem PREREG_V8F_LAUNCH_INTENT section 4 as redesigned (MM_REDESIGN_2026-10-28 section 3), the decisive arm: reached ONLY from :dagarm (never called, never dispatched) with DAGS = launchintent | launchintentrep and :dagname's names set. %1 = ARM, %2 = SEED (0 | 1: the seed of M2b_s<SEED>, the init), %3 = nothing: the dose is fixed (arm B is NOT queued: it is trained only after A's HARMS or NOT ADOPTED, with its own PI go).
rem The fine-tune is DAG-OACC's banked argv VERBATIM, obtained by CALLING :dagoaccarm (untouched: init M2b_s<SEED>, 3,000 steps, lr 1e-4, warmup 100, p_gt 0.2 to 0.2, --ego-perturb-p 0, --dagger-p 0.1, the ITERATION-1 bank %RUNSM%\\dagger_bank\\s<SEED> with its sha256 computed HERE, --dagger-seed <SEED>, --w-acc <the waccdag value>), plus --launch-intent-w {W} at the END and nothing else.
rem THE ARM LABEL IS THE RUN: launchintent = A_s (trainer seed SEED; its pair control is the banked DAG-OACC_s); launchintentrep = A_rep,s (trainer seed 10 + SEED, set below exactly as :dagoaccorcarm sets REP's: DAGSEED = 1<SEED>, --dagger-seed stays SEED; its pair control is the banked REP_s = DAG-ORC r2's M2b_s<SEED>_dagoaccorcrep, whose argv is DAG-OACC_s's at --seed 10 + SEED). Four runs: launchintent 0, launchintent 1, launchintentrep 0, launchintentrep 1.
rem {W} = W_A of PREREG section 3: 1 + W = (1 - f) / (2 f) with f = 49,618 / 306,566 (MEASURED, tools/measure_f.py), so launch samples hold one third of the along-track weight. The chain's one-lever asserter allows exactly this flag at exactly this value, and --out may differ.
rem Names (all from :dagname, indexed by the init seed SEED): run M2b_s<SEED>_<ARM>, tag M2b-<ARM>_s<SEED>, gate directory gate_M2b-<ARM>_s<SEED>, pin dagpin\\M2b_s<SEED>_<ARM>.train.argv.txt. The bank-hash file bank_sha_dagoacc_s<SEED>.txt is DAG-OACC's own (:dagoaccarm writes the sha256 of the SAME bank.json into it, for A_s and A_rep,s alike: one file per seed, the same bytes; the chain pins that bank sha256 independently).
if not "%~3"=="" (echo REFUSED: %DAGS% takes no P, the dose is fixed by PREREG_V8F_LAUNCH_INTENT, got "%~3" & exit /b 2)
call :dagoaccarm %~1 %~2 0.1 || exit /b 2
if /i "%DAGS%"=="launchintentrep" set "DAGSEED=1%~2"
set "DAGF=%DAGF% --launch-intent-w {W}"
exit /b 0
'''.replace("\n", "\r\n")


def derive(tip_queue: bytes) -> bytes:
    t = tip_queue.decode("utf-8")
    assert t.count("\r\n") == t.count("\n"), "the tip's queue is not pure CRLF"
    assert t.endswith("exit /b 0\r\n")
    assert "launchintent" not in t, "the tip's queue already names launchintent"
    t = C.rep(t, NAME_ANCHOR, NAME_ANCHOR + NAME_ADD, 1)
    t = C.rep(t, DISP_ANCHOR, DISP_ANCHOR + DISP_ADD, 1)
    return (t + STAGE_TEXT).encode("utf-8")


def main() -> int:
    tipq = C.git("show", f"{C.tip_full()}:{C.QUEUE_PATH}", binary=True)
    out = derive(tipq)
    C.QSTAGE.mkdir(parents=True, exist_ok=True)
    C.QUEUE_TIP_FILE.write_bytes(tipq)
    (C.QSTAGE / "v8f_mini_box2_queue.cmd").write_bytes(out)
    a, b = tipq.decode("utf-8").split("\r\n"), out.decode("utf-8").split("\r\n")
    d = "\n".join(difflib.unified_diff(a, b, "tip:" + C.QUEUE_PATH, "staged:" + C.QUEUE_PATH, lineterm="", n=2))
    (C.QSTAGE / "queue.diff").write_bytes((d + "\n").encode("utf-8"))
    print("wrote queue_stage/v8f_mini_box2_queue.cmd", len(out), "bytes (CRLF) sha256", C.sha_b(out))
    print("tip", C.tip_full(), "queue blob sha256", C.sha_b(tipq), "| added lines:", sum(1 for x in d.split("\n") if x.startswith("+") and not x.startswith("+++")), "| removed:", sum(1 for x in d.split("\n") if x.startswith("-") and not x.startswith("---")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
