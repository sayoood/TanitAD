#!/usr/bin/env python3
"""Runs the two suites the brief names on a refe/ directory and records counts, return codes and wall time as JSON (a ZZTEST_SKIP anywhere is recorded: on the pod a skip FAILS G-SUITE).

    python record_suites.py --refe <package>/refe --python <interpreter> --label patched --out suite_results_patched.json [--chain-log <a finished test_s1aux_chain.py log>]

  test_launch_gate.py --static              -> ZZLG_STATIC_PASS <n>   (dev box: includes the banked generic-gate cross-check)
  test_launch_gate.py --static --no-banked  -> ZZLG_STATIC_PASS <n>   (what the pod's G-SUITE runs)
  test_s1aux_chain.py                       -> ZZTEST_PASS <n>        (about 5-20 minutes: pass --chain-log to record an already finished run instead of running it again)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def run(py: str, refe: Path, args: list) -> dict:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.time()
    r = subprocess.run([py, str(refe / args[0]), *args[1:]], cwd=str(refe), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = r.stdout + r.stderr
    m = re.findall(r"^(ZZLG_STATIC_PASS|ZZTEST_PASS) (\d+)", out, re.M)
    return {"cmd": " ".join(args), "rc": r.returncode, "token": m[-1][0] if m else None, "checks": int(m[-1][1]) if m else None, "skips": len(re.findall(r"^ZZTEST_SKIP", out, re.M)),
            "seconds": round(time.time() - t0, 1), "tail": [ln[:200] for ln in out.splitlines()[-3:]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refe", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chain-log")
    a = ap.parse_args()
    refe = Path(a.refe).resolve()
    res = {"label": a.label, "refe": str(refe), "python": a.python, "runs": []}
    res["runs"].append(run(a.python, refe, ["test_launch_gate.py", "--static"]))
    res["runs"].append(run(a.python, refe, ["test_launch_gate.py", "--static", "--no-banked"]))
    if a.chain_log:
        txt = Path(a.chain_log).read_text(encoding="utf-8", errors="replace")
        m = re.findall(r"^ZZTEST_PASS (\d+)", txt, re.M)
        rc = re.findall(r"^rc=(\d+)", txt, re.M)
        wall = re.findall(r"wall=(\d+)s", txt)
        res["runs"].append({"cmd": "test_s1aux_chain.py (recorded from " + Path(a.chain_log).name + ")", "rc": int(rc[-1]) if rc else None, "token": "ZZTEST_PASS" if m else None,
                            "checks": int(m[-1]) if m else None, "skips": len(re.findall(r"^ZZTEST_SKIP", txt, re.M)), "seconds": int(wall[-1]) if wall else None})
    else:
        res["runs"].append(run(a.python, refe, ["test_s1aux_chain.py"]))
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps([{k: r[k] for k in ("cmd", "rc", "checks", "skips", "seconds")} for r in res["runs"]], indent=1))
    return 0 if all(r["rc"] == 0 and r["checks"] for r in res["runs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
