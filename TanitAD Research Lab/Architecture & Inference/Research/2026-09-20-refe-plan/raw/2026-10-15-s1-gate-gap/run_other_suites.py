#!/usr/bin/env python3
"""The patches touch modules every chain's launch gate shares (gate_token.py, launch_gate.py, gate_checks.py).  This runs the OTHER chains' suites that import them, on the BASE refe/ and on
the PATCHED refe/, and records the pass tokens and counts side by side.  A suite that passes on BASE and fails or changes its count on PATCHED is a regression of the patch.

    python run_other_suites.py --base <base package root> --new <patched package root> --python <interpreter> --out other_suites.json [--suites a.py,b.py]

Each suite prints its own pass token: `ZZTEST_PASS <n>` / `ZZLG_*` / `ZZP6*`.  (Both package roots keep refe/ plus eval raw code splits beside it.)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DEFAULT = ["test_p2r_chain.py", "test_p2rep_chain.py", "test_rls1_chain.py", "test_rl_s1b.py", "test_p6_chain.py", "test_aux_s1_wiring.py", "test_rl_reward.py", "test_rl_epo.py"]
TOKEN = re.compile(r"^(ZZ[A-Z0-9_]*PASS[A-Z0-9_]*)\s+(\d+)", re.M)


def run(py: str, root: Path, suite: str) -> dict:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.time()
    try:
        r = subprocess.run([py, str(root / "refe" / suite)], cwd=str(root / "refe"), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3000)
        out = r.stdout + r.stderr
        m = TOKEN.findall(out)
        return {"suite": suite, "rc": r.returncode, "token": m[-1][0] if m else None, "checks": int(m[-1][1]) if m else None, "skips": len(re.findall(r"^ZZTEST_SKIP", out, re.M)),
                "seconds": round(time.time() - t0, 1), "tail": [ln[:200] for ln in out.splitlines()[-2:]] if r.returncode else []}
    except subprocess.TimeoutExpired:
        return {"suite": suite, "rc": "timeout", "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suites", default=",".join(DEFAULT))
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    suites = [s for s in a.suites.split(",") if s]
    base, new = Path(a.base).resolve(), Path(a.new).resolve()
    jobs = [(root, s) for s in suites for root in (base, new)]
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(lambda j: (str(j[0]) == str(base), run(a.python, j[0], j[1])), jobs))
    rows = []
    for s in suites:
        b = next(r for isb, r in res if isb and r["suite"] == s)
        n = next(r for isb, r in res if not isb and r["suite"] == s)
        rows.append({"suite": s, "base": {k: b.get(k) for k in ("rc", "token", "checks", "skips", "seconds")}, "patched": {k: n.get(k) for k in ("rc", "token", "checks", "skips", "seconds")},
                     "same_outcome": (b.get("rc"), b.get("token"), b.get("checks")) == (n.get("rc"), n.get("token"), n.get("checks")), "patched_tail": n.get("tail", [])})
        print(json.dumps({k: rows[-1][k] for k in ("suite", "base", "patched", "same_outcome")}), flush=True)
    Path(a.out).write_text(json.dumps({"base": str(base), "patched": str(new), "python": a.python, "suites": rows, "all_same": all(r["same_outcome"] for r in rows)}, indent=1), encoding="utf-8")
    print("all same outcome:", all(r["same_outcome"] for r in rows))
    return 0 if all(r["same_outcome"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
