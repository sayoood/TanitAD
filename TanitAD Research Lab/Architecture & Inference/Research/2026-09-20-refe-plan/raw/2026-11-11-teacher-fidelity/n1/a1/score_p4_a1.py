#!/usr/bin/env python3
"""Score ONE P4 seam with W3's harness through the drive-repointed wrapper score_e.py (eval/PREREG_P4 §3), retried while
the harness's own RAM guard fires, exactly as eval/stage2_sft.py does it. The harness's OWN status line must read PASS;
a CSV existing is not evidence (stage2_sft.py: a run without --tokens scored all 12,146, failed guard C2, and still
wrote a CSV).

    python eval/score_p4.py --label refe_p4_teacher_navtest --seam <npz> --tokens <W3-format json> --log-dir <dir>
Writes <log-dir>/<label>_status.json and prints ZZSCORE <label> <PASS|FAIL> <PDMS>.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# A1 (TF-N1 amendment 1): the harness working directory is the BANKED, sha256-pinned eval/harness/ of the tree this file sits in
# (was: a deleted %TEMP% scratchpad). The score output root is an argument (was: E:/Projects/TanitAD/data/refe_navtest).
HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.join(HERE, "harness")
PY_D = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
D = os.environ.get("TFN1_A1_SCORE_ROOT", "")


def _lf_sha(path):
    import hashlib
    b = open(path, "rb").read()
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def preflight() -> None:
    """A1 guards: (1) harness bytes == HARNESS_PINS (harness_pins.verify); (2) score_e.py loads its caller from the hard-coded
    E:/Projects/TanitAD/... PKG path, so that file must be byte-identical (LF-normalised) to THIS tree's copy; (3) interpreter exists."""
    sys.path.insert(0, HERE)
    import harness_pins
    v = harness_pins.verify(root=harness_pins.Path(SP), caller=harness_pins.Path(HERE) / "score_navtest_refe.py")
    if not v.get("ok"):
        sys.exit("A1-PREFLIGHT REFUSED: harness pins do not hold: " + json.dumps(v, default=str)[:600])
    src = open(os.path.join(SP, "score_e.py"), encoding="utf-8").read()
    import re
    m = re.search(r'^PKG = "([^"]+)"', src, re.M)
    ec = (m.group(1) + "/eval/score_navtest_refe.py") if m else None
    if not ec or not os.path.isfile(ec) or _lf_sha(ec) != _lf_sha(os.path.join(HERE, "score_navtest_refe.py")):
        sys.exit(f"A1-PREFLIGHT REFUSED: score_e.py's caller {ec} is missing or differs from {HERE}/score_navtest_refe.py")
    if not os.path.isfile(PY_D):
        sys.exit(f"A1-PREFLIGHT REFUSED: {PY_D} missing")


def free_mb() -> int:
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)"],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--seam", required=True)
    ap.add_argument("--tokens", required=True, help="W3-format subset json, or 'all' for the full navtest")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--score-root", default=None, help="A1: output root; the CSV lands in <root>/score/<label>/<label>.csv "
                                                         "(default $TFN1_A1_SCORE_ROOT)")
    ap.add_argument("--min-free-mb", type=int, default=6500, help="free RAM before starting (harness peak ~2.3 GB "
                                                                    "+ the 4 GB the box keeps free)")
    a = ap.parse_args()
    if not a.label.startswith("refe"):
        sys.exit("labels must start with 'refe'")
    global D
    a.seam = os.path.abspath(a.seam)                      # A1: the harness runs with cwd = eval/harness, so relative paths would break
    a.tokens = a.tokens if a.tokens == "all" else os.path.abspath(a.tokens)
    a.log_dir = os.path.abspath(a.log_dir)
    D = a.score_root or D
    if not D:
        sys.exit("A1: --score-root (or $TFN1_A1_SCORE_ROOT) is required")
    preflight()
    os.makedirs(a.log_dir, exist_ok=True)
    stp = os.path.join(a.log_dir, f"{a.label}_status.json")
    status = json.load(open(stp, encoding="utf-8")) if os.path.exists(stp) else {}
    for att in range(1, 7):
        if status.get("status") == "PASS":
            break
        while free_mb() < a.min_free_mb:
            print(f"waiting for free RAM >= {a.min_free_mb} MB {time.strftime('%T')}", flush=True)
            time.sleep(60)
        lp = os.path.join(a.log_dir, f"{a.label}_score_{att}.log")
        t0 = time.time()
        # ⛔ MEASURED 2026-10-05: W3's run_v1 passes the token list ON THE COMMAND LINE, so --tokens with all 12,146
        # navtest tokens dies on WinError 206 (Windows' 32 KB command-line limit) before the harness starts. For the
        # FULL set the token filter is unnecessary -- the navtest scene filter IS the 12,146 (the student's full read,
        # refe_navtest_final, was scored exactly this way). `--tokens all` = omit it; C2's count guard then checks
        # that the seam covers all of them.
        tok_args = [] if a.tokens == "all" else ["--tokens", a.tokens]
        subprocess.call([PY_D, "score_e.py", "--label", a.label, "--seam", a.seam] + tok_args +
                        ["--out", f"{D}/score"], cwd=SP, env=dict(os.environ, PYTHONIOENCODING="utf-8", RUN_V1_FIX1_ALLOW_DRIFT="1"),   # A1: as stage2_sft.py:191; the repointed run_v1.py differs from W3's original by the drive roots only
                        stdout=open(lp, "w"), stderr=subprocess.STDOUT)
        lines = open(lp, encoding="utf-8", errors="replace").read().splitlines()
        status = next((json.loads(l) for l in reversed(lines) if l.startswith("{") and '"status"' in l), {})
        status["_attempt"], status["_seconds"] = att, round(time.time() - t0)
        json.dump(status, open(stp, "w"), indent=1)
        print(f"attempt {att}: status {status.get('status')} in {status['_seconds']} s", flush=True)
        if status.get("status") != "PASS":
            time.sleep(120)
    csv_p = f"{D}/score/{a.label}/{a.label}.csv"
    ok = status.get("status") == "PASS" and os.path.exists(csv_p)
    pd = (status.get("summary_x100_4dp") or {}).get("PDMS")
    print(f"ZZSCORE {a.label} {'PASS' if ok else 'FAIL'} {pd}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
