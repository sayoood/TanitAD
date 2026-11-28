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

SP = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/a085b71c-4fb3-409b-ae6b-65f605ed575e/scratchpad"
PY_D = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
D = "E:/Projects/TanitAD/data/refe_navtest"


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
    ap.add_argument("--min-free-mb", type=int, default=6500, help="free RAM before starting (harness peak ~2.3 GB "
                                                                    "+ the 4 GB the box keeps free)")
    a = ap.parse_args()
    if not a.label.startswith("refe"):
        sys.exit("labels must start with 'refe'")
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
                        ["--out", f"{D}/score"], cwd=SP, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
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
