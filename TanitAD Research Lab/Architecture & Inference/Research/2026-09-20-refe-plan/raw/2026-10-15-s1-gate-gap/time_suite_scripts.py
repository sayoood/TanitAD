#!/usr/bin/env python3
"""Times the five G-SUITE scripts of profile refe_aux_s1 the way gate_checks.check_suite runs them (same argv, same env incl. REFE_GATE_NESTED=1, cwd = refe/),
first ONE AT A TIME (today's behaviour), then ALL AT ONCE (--parallel), and writes a JSON.

    python time_suite_scripts.py --refe <package>/refe --python <interpreter> --out suite_timing.json [--parallel] [--repeat N]

Environment for the aux scripts (test_aux_drivable_head.py reads the real label store; without them it prints ZZTEST_SKIP and the timing is not representative):
    AUX_LABEL_STORE=<label_store>  AUX_NAVTEST=<navtest.json>

No GPU, no network, nothing written outside --out.  Runnable on the pod too (CPU only) with the pod's interpreter.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SUITE = [("test_rl_reward.py", [], "ZZTEST_PASS"), ("test_rl_epo.py", [], "ZZTEST_PASS"), ("test_aux_drivable_head.py", [], "ZZTEST_PASS"),
         ("test_aux_s1_wiring.py", [], "ZZTEST_PASS"), ("test_launch_gate.py", ["--static", "--no-banked"], "ZZLG_STATIC_PASS")]


def run_one(refe: Path, py: str, script: str, args: list, token: str) -> dict:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", REFE_GATE_NESTED="1")
    t0 = time.perf_counter()
    r = subprocess.run([py, str(refe / script), *args], cwd=str(refe), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    skips = [ln for ln in r.stdout.splitlines() if ln.startswith("ZZTEST_SKIP")]
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith(token)), "")
    return {"script": script, "rc": r.returncode, "token_seen": token in r.stdout, "token_line": line, "skips": len(skips), "seconds": round(dt, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refe", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", action="store_true", help="also run the five at once (a thread per subprocess)")
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    refe = Path(a.refe).resolve()
    res = {"refe": str(refe), "python": a.python, "cpu_count": os.cpu_count(), "aux_env": {k: bool(os.environ.get(k)) for k in ("AUX_LABEL_STORE", "AUX_NAVTEST")}, "runs": []}
    for k in range(a.repeat):
        t0 = time.perf_counter()
        seq = [run_one(refe, a.python, s, args, tok) for s, args, tok in SUITE]
        row = {"sequential": seq, "sequential_wall_s": round(time.perf_counter() - t0, 2)}
        if a.parallel:
            t0 = time.perf_counter()
            with cf.ThreadPoolExecutor(max_workers=len(SUITE)) as ex:
                par = list(ex.map(lambda e: run_one(refe, a.python, e[0], e[1], e[2]), SUITE))
            row["parallel"] = par
            row["parallel_wall_s"] = round(time.perf_counter() - t0, 2)
            row["same_outcome"] = [(x["rc"], x["token_seen"], x["token_line"], x["skips"]) for x in seq] == [(x["rc"], x["token_seen"], x["token_line"], x["skips"]) for x in par]
        res["runs"].append(row)
        print(json.dumps({k2: v for k2, v in row.items() if k2 not in ("sequential", "parallel")}), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
