#!/usr/bin/env python3
"""Amendment 8: an END-TO-END CPU smoke of the patched seam on 2 SELECTION tokens (never a confirmation token) --
one the trigger fires on (3efebf87894a552e, the traced 373.5 m case) and one covering control (014369205e025f0c).
Four runs through eval/refe_navtest_seam.py: OFF, ON (--sanitize-goal), clamp150, straight, all with --record-inputs.
Asserts: 2 rows each; the ego vector and the frames are IDENTICAL across the four; the covering token's goal is
identical across all four and its executed pose too; the triggered token's goal differs from OFF in ON / clamp150 /
straight; clamp150's p2 is 150 m on it; straight's goal is (s/2, 0, s, 0); the ON report says sanitize_goal True.
    python a8_seam_smoke.py --root <package root> [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
TRIG, COVER = "3efebf87894a552e", "014369205e025f0c"
LOGS = {TRIG: "2021.08.30.13.45.25_veh-40_01116_01336", COVER: "2021.05.25.14.16.10_veh-35_00083_00485"}
SNAP = "D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch015.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--work", default="D:/Projects/TanitAD/data/refe_proxy/goal_clamp/a8_smoke")
    a = ap.parse_args()
    root, work = Path(a.root).resolve(), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PKG / "eval"))
    import eval_checkpoint as EC
    tj = work / "tokens.json"
    json.dump({"tokens": [TRIG, COVER], "token_log": LOGS}, open(tj, "w"))
    env = EC.env_driverl()
    env.update(OMP_NUM_THREADS="6", PYTHONIOENCODING="utf-8")
    arms = {"off": [], "on": ["--sanitize-goal"], "clamp150": ["--goal-variant", "clamp150"],
            "straight": ["--goal-variant", "straight"]}
    rec, rep, seam = {}, {}, {}
    for arm, extra in arms.items():
        out = work / f"a8_smoke_{arm}.npz"
        cmd = [EC.DRIVERL_PY, "refe_navtest_seam.py", "--ckpt", SNAP, "--tokens", str(tj), "--out", str(out),
               "--device", a.device, "--record-inputs", str(work / f"a8_smoke_{arm}.inputs.json"), *extra]
        with open(work / f"a8_smoke_{arm}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, cwd=str(root / "eval"), env=env, stdout=fh, stderr=subprocess.STDOUT)
        print(f"  {arm}: rc {rc}", flush=True)
        rec[arm] = json.load(open(work / f"a8_smoke_{arm}.inputs.json")) if rc == 0 else {}
        rep[arm] = json.load(open(work / f"a8_smoke_{arm}.report.json")) if rc == 0 else {}
        z = np.load(out) if rc == 0 else None
        seam[arm] = {str(t): z["poses"][i] for i, t in enumerate(z["token"])} if z is not None else {}
    ok = all(len(seam[k]) == 2 for k in arms)
    checks = {"every arm wrote 2 rows": ok}
    if ok:
        g = {k: {t: np.array(rec[k][t]["goal"]) for t in (TRIG, COVER)} for k in arms}
        checks.update({
            "ego vector identical across the 4 arms": all(rec[k][t]["ego"] == rec["off"][t]["ego"]
                                                          for k in arms for t in (TRIG, COVER)),
            "frames identical across the 4 arms": all(rec[k][t]["frames"] == rec["off"][t]["frames"]
                                                      for k in arms for t in (TRIG, COVER)),
            "covering token: goal identical in all 4 arms": all(np.array_equal(g[k][COVER], g["off"][COVER]) for k in arms),
            "covering token: executed pose identical in all 4 arms": all(np.array_equal(seam[k][COVER], seam["off"][COVER])
                                                                        for k in arms),
            "triggered token: goal differs from OFF in on / clamp150 / straight": all(
                not np.array_equal(g[k][TRIG], g["off"][TRIG]) for k in ("on", "clamp150", "straight")),
            "clamp150: p2 is 150 m on the triggered token": abs(float(np.hypot(*g["clamp150"][TRIG][2:4])) - 150.0) < 1e-3,
            "straight: goal is (s/2, 0, s, 0)": abs(g["straight"][TRIG][1]) < 1e-9 and abs(g["straight"][TRIG][3]) < 1e-9
            and abs(g["straight"][TRIG][2] - 2 * g["straight"][TRIG][0]) < 1e-4 and g["straight"][TRIG][2] >= 60.0 - 1e-4,
            "ON: goal_diag triggered on the far token, not on the covering one": bool(
                (rec["on"][TRIG]["goal_diag"] or {}).get("triggered")) and not (rec["on"][COVER]["goal_diag"] or {}).get("triggered"),
            "reports: sanitize_goal False / True / False / False": [rep[k].get("sanitize_goal") for k in arms] == [False, True, False, False],
        })
        print("  goals on the triggered token:", {k: [round(float(x), 1) for x in g[k][TRIG]] for k in arms})
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    json.dump({"checks": checks, "failed": [k for k, v in checks.items() if not v], "reports": rep},
              open(HERE / "a8_seam_smoke.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    print("ZZA8_SMOKE_" + ("OK" if all(checks.values()) else "FAIL"))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
