#!/usr/bin/env python3
"""make_queue_dd.py -- the DAGDOSE queue stage (PREREG_V8F_DAGDOSE section 4), written into a COPY of the tip's
v8f_mini_box2_queue.cmd by exact anchored insertions (each anchor must occur exactly once, else the build aborts).

    python make_queue_dd.py --mirror C:/Users/Admin/tanitad-push --tip <40 hex> --out ../queue/v8f_mini_box2_queue.cmd

Adds, and changes nothing else (the diff is written beside the output and asserted to be insertions only):
  * two whitelist lines in :dagname (dagdose, dagdoserep), two dispatch lines in :dagarm, one usage rem line;
  * :dagdosearm = CALL :dagoaccarm (DAG-OACC's recipe, untouched, P 0.1) then DAGF rebuilt from :dagoaccarm's own DAGF0 /
    DAGBANK / DAGSHA / WACCDAG with DAGP = 0.3: the same tokens in the same order as DAG-OACC's line, --dagger-p's VALUE the only
    change.  dagdoserep sets DAGSEED = 1<SEED> exactly as :dagoaccorcarm sets REP's (trainer seed 10 + s; --dagger-seed stays s).
The file keeps its CRLF bytes.  Every backslash in the inserted text is chr(92)."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import subprocess
import sys

BS = chr(92)
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
QPATH = PK + "/code/v8f_mini_box2_queue.cmd"
DD_P = "0.3"

A_NAME = 'if /i "%~1"=="launchintentrep" set "DAGS=launchintentrep"'
A_DISP = 'if /i "%DAGS%"=="launchintentrep" goto :launchintentarm'
A_REM = "rem   gatedag / traindag / evaldag also take ARM dagoaccorc ("
A_END = 'set "DAGF=%DAGF% --launch-intent-w 1.58926"' + "\r\nexit /b 0\r\n"

NAME_LINES = ['if /i "%~1"=="dagdose" set "DAGS=dagdose"', 'if /i "%~1"=="dagdoserep" set "DAGS=dagdoserep"']
DISP_LINES = ['if /i "%DAGS%"=="dagdose" goto :dagdosearm', 'if /i "%DAGS%"=="dagdoserep" goto :dagdosearm']
REM_LINE = ("rem   gatedag / traindag / evaldag also take ARM dagdose (DAG-OACC's argv verbatim with --dagger-p 0.3 in place of 0.1) and dagdoserep "
            "(the same at --seed 10 / 11, as REP); neither takes a P (any extra argument is REFUSED): PREREG_V8F_DAGDOSE section 4. Init = %RUNSM%"
            + BS + "M2b_s<SEED>" + BS + "ckpt_last.pt; run M2b_s<SEED>_<ARM>; tag M2b-<ARM>_s<SEED>; pin dagpin" + BS + "M2b_s<SEED>_<ARM>.train.argv.txt")
ARM_LINES = [
    ":dagdosearm",
    "rem PREREG_V8F_DAGDOSE section 4: reached ONLY from :dagarm (never called, never dispatched) with DAGS = dagdose | dagdoserep and :dagname's names set. %1 = ARM, %2 = SEED (0 | 1), %3 = nothing: the dose is fixed at " + DD_P + ".",
    "rem The fine-tune is DAG-OACC's banked argv, obtained by CALLING :dagoaccarm (untouched, P 0.1: it sets DAGF0, DAGBANK, DAGSHA and WACCDAG and hashes the ITERATION-1 bank HERE), then DAGF is rebuilt from those same values",
    "rem in :dagoaccarm's own token order with DAGP = " + DD_P + ": the only token that differs from DAG-OACC's line is --dagger-p's value (and --out, from :dagname). dagdoserep = the same at trainer seed 10 + SEED (DAGSEED = 1<SEED>, REP's rule; --dagger-seed stays SEED).",
    'if not "%~3"=="" (echo REFUSED: %DAGS% takes no P, the dose is fixed at ' + DD_P + ' by PREREG_V8F_DAGDOSE, got "%~3" & exit /b 2)',
    "call :dagoaccarm %~1 %~2 0.1 || exit /b 2",
    'if /i "%DAGS%"=="dagdoserep" set "DAGSEED=1%~2"',
    'set "DAGP=' + DD_P + '"',
    'set "DAGF=%DAGF0% --dagger-p %DAGP% --dagger-bank "%DAGBANK%" --dagger-bank-sha256 %DAGSHA% --dagger-seed %~2 --w-acc %WACCDAG%"',
    "exit /b 0",
]
# the line :dagoaccarm builds (DAG-OACC's): :dagdosearm's DAGF must be it with DAGP the only variable that differs
OACC_DAGF = 'set "DAGF=%DAGF0% --dagger-p %DAGP% --dagger-bank "%DAGBANK%" --dagger-bank-sha256 %DAGSHA% --dagger-seed %~2 --w-acc %WACCDAG%"'


def insert_after(text: str, anchor: str, lines: list) -> str:
    n = text.count(anchor + "\r\n")
    if n != 1:
        raise SystemExit(f"ABORT: anchor occurs {n} times (must be 1): {anchor[:80]!r}")
    return text.replace(anchor + "\r\n", anchor + "\r\n" + "".join(x + "\r\n" for x in lines), 1)


def build(src: bytes) -> bytes:
    t = src.decode("ascii")
    if t.count(OACC_DAGF + "\r\n") != 1:
        raise SystemExit("ABORT: :dagoaccarm's DAGF line is not the reviewed one (the dose arm copies its token order)")
    if ":dagdosearm" in t or "dagdose" in t:
        raise SystemExit("ABORT: the source queue already names dagdose")
    t = insert_after(t, A_NAME, NAME_LINES)
    t = insert_after(t, A_DISP, DISP_LINES)
    rem = [ln for ln in t.split("\r\n") if ln.startswith(A_REM)]
    if len(rem) != 1:
        raise SystemExit(f"ABORT: the usage anchor occurs {len(rem)} times")
    t = insert_after(t, rem[0], [REM_LINE])
    if not t.endswith(A_END) or t.count(A_END) != 1:
        raise SystemExit("ABORT: the queue does not end with :launchintentarm's last two lines")
    t = t + "".join(x + "\r\n" for x in ARM_LINES)
    if ARM_LINES[-2] != OACC_DAGF:
        raise SystemExit("ABORT: :dagdosearm's DAGF is not :dagoaccarm's line")
    return t.encode("ascii")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--tip", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src = subprocess.run(["git", "-c", "safe.directory=*", "-C", a.mirror, "show", f"{a.tip}:{QPATH}"], capture_output=True, check=True).stdout
    out = build(src)
    open(a.out, "wb").write(out)
    s_l, o_l = src.decode("ascii").split("\r\n")[:-1], out.decode("ascii").split("\r\n")[:-1]      # both end in CRLF: drop the empty tail
    diff = list(difflib.unified_diff(s_l, o_l, f"a/{QPATH}", f"b/{QPATH}", lineterm="", n=1))
    bad = [d for d in diff if d.startswith("-") and not d.startswith("---")]
    if bad:
        raise SystemExit(f"ABORT: the patch removes lines: {bad[:3]}")
    open(a.out + ".diff", "w", encoding="ascii", newline="\n").write("\n".join(diff) + "\n")
    print(f"source {a.tip}:{QPATH} sha256 {hashlib.sha256(src).hexdigest()}")
    print(f"wrote {a.out} sha256 {hashlib.sha256(out).hexdigest()} ({len(o_l) - len(s_l)} lines added, 0 removed, CRLF line ends {out.count(bytes([13, 10]))}, lone LF {out.count(bytes([10])) - out.count(bytes([13, 10]))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
