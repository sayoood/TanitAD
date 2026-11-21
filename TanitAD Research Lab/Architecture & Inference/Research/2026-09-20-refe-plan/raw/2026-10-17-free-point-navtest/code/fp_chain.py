#!/usr/bin/env python3
"""The dev-box tail of the free-point read: picks -> seams -> harness (4 points + the C0A subset) -> controls -> the rows.  CPU only, resumable, no model turns.

    python fp_chain.py --work W [--wait-for-pull] [--parallel 2]

* `--wait-for-pull` : first waits until W/ship.log contains ZZSHIP_ALL_PULLED (so it can be started BEFORE the pod stage ends and run on its own afterwards).
* Writes W/chain.log lines  `ZZCHAIN <stage> <OK|FAIL> <seconds>`; the last line is `ZZCHAIN_DONE <status>` where status is the gates_summary status.
* The pod's `ic5_forward.json` is copied from the pulled chunk's meta by this script (chunk_000.npz meta_json["ic5_forward"]).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"


def run(stage, argv, work, log):
    t0 = time.time()
    rc = subprocess.call([PY, "-I", "-X", "utf8", *argv], stdout=open(f"{work}/chain_{stage}.out", "w", encoding="utf-8"), stderr=subprocess.STDOUT)
    line = f"ZZCHAIN {stage} {'OK' if rc == 0 else 'FAIL rc=' + str(rc)} {time.time() - t0:.0f}s"
    print(line, flush=True)
    open(log, "a").write(line + "\n")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--wait-for-pull", action="store_true")
    ap.add_argument("--parallel", type=int, default=2)
    a = ap.parse_args()
    w = a.work
    log = f"{w}/chain.log"
    if a.wait_for_pull:
        while not (os.path.exists(f"{w}/ship.log") and "ZZSHIP_ALL_PULLED" in open(f"{w}/ship.log", errors="replace").read()):
            time.sleep(60)
    # ic5 forward arm: the first chunk's meta carries it
    import numpy as np
    z = np.load(f"{w}/out/chunk_000.npz", allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    json.dump(meta.get("ic5_forward", {"missing": True}), open(f"{w}/ic5_forward.json", "w"), indent=1)
    steps = [("picks", [f"{HERE}/fp_picks.py", "build", "--out-dir", w, "--chunks", f"{w}/out"]),
             ("score", [f"{HERE}/fp_score.py", "score", "--work", w, "--points", "BASE,C0A,R0B,M25", "--parallel", str(a.parallel)]),
             ("subset", [f"{HERE}/fp_score.py", "subset", "--work", w, "--point", "C0A", "--n", "1000"]),
             ("checks", [f"{HERE}/fp_checks.py", "all", "--work", w]),
             ("read", [f"{HERE}/fp_read.py", "read", "--work", w, "--out-dir", f"{w}/result"])]
    for st, argv in steps:
        if st == "picks" and os.path.exists(f"{w}/picks_report.json"):
            continue
        if run(st, argv, w, log) != 0:
            open(log, "a").write(f"ZZCHAIN_DONE FAILED_AT_{st}\n")
            return 1
    status = json.load(open(f"{w}/gates_summary.json"))["status"]
    open(log, "a").write(f"ZZCHAIN_DONE {status}\n")
    print(f"ZZCHAIN_DONE {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
