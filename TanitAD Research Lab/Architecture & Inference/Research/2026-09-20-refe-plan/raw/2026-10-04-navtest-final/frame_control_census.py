#!/usr/bin/env python3
"""CPU only, NO model: the seam's frame/time control PER TOKEN, for the tokens of a seam whose control failed.

Same quantity as refe_navtest_seam.py (the log future on NAVSIM's 0.5 s grid in the ego rear-axle frame vs W3's
exported human_future_poses), but it keeps every token's value plus the diagnostics needed to say WHY a token fails:
the future time offsets, the anchor timestamp, and the token's position inside its log.
The drive letter moved D: -> E:, so every D: constant is re-pointed through DRIVE (no repo file is edited).
"""
from __future__ import annotations

import gzip
import json
import math
import os
import subprocess
import sys

import numpy as np

DRIVE = os.environ.get("REFE_DRIVE", "E:")
PKG = f"{DRIVE}/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TOKENS = f"{DRIVE}/Projects/TanitAD/data/refe_navtest/navtest_all_tokens.json"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "frame_control_navtest_full.json")
sys.path[:0] = [f"{PKG}/eval", f"{PKG}/refe", f"{PKG}/code"]


def fix(s: str) -> str:
    return s.replace("D:/", f"{DRIVE}/").replace("D:\\", f"{DRIVE}\\")


def main() -> int:
    import eval_checkpoint as EC
    if os.environ.get("REFE_FC_CHILD") != "1":
        env = {k: (fix(v) if isinstance(v, str) else v) for k, v in EC.env_driverl().items()}
        env["REFE_FC_CHILD"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([env.get("PYTHONPATH", ""), f"{PKG}/refe", f"{PKG}/eval", f"{PKG}/code"])
        return subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__), OUT], cwd=f"{PKG}/eval", env=env)
    import navtrain_scenarios as NS
    import refe_navtest_seam as SEAM
    SEAM.W3_EXPORT, SEAM.TEST_DB_DIR = fix(SEAM.W3_EXPORT), fix(SEAM.TEST_DB_DIR)
    tj = json.load(open(TOKENS, encoding="utf-8"))
    toks = tj["tokens"] if isinstance(tj, dict) else tj
    exp = json.load(gzip.open(SEAM.W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
    by_log: dict = {}
    for t in toks:
        by_log.setdefault(exp[t]["log_name"], []).append(t)
    rows, n_ctrl = {}, 0
    for li, (lg, ts) in enumerate(sorted(by_log.items())):
        try:
            for sc in NS.build_scenarios_for_log(os.path.join(SEAM.TEST_DB_DIR, f"{lg}.db"), ts,
                                                 history_rows=1, future_rows=80):
                tok = sc._initial_lidar_token
                ego = sc.get_ego_state_at_iteration(0)
                px, py, pyaw = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
                c, s = math.cos(-pyaw), math.sin(-pyaw)
                t0 = int(ego.time_point.time_us)
                f8 = list(sc.get_ego_future_trajectory(0, 4.0, SEAM.NAVSIM_N))[:SEAM.NAVSIM_N]
                theirs = np.asarray(exp[tok]["human_future_poses"], dtype=np.float64)
                r = {"log": lg, "t0_us": t0, "n_future": len(f8)}
                if len(f8) == SEAM.NAVSIM_N:
                    mine = np.array([[(f.rear_axle.x - px) * c - (f.rear_axle.y - py) * s,
                                      (f.rear_axle.x - px) * s + (f.rear_axle.y - py) * c] for f in f8])
                    err = np.linalg.norm(mine - theirs[:, :2], axis=1)
                    r.update(err_max=float(err.max()), err_per_pose=[round(float(x), 4) for x in err],
                             dt_s=[round((int(f.time_point.time_us) - t0) / 1e6, 3) for f in f8],
                             mine_end=[round(float(x), 3) for x in mine[-1]],
                             theirs_end=[round(float(x), 3) for x in theirs[-1, :2]])
                    n_ctrl += 1
                rows[tok] = r
        except Exception as e:  # noqa: BLE001
            for t in ts:
                rows.setdefault(t, {"log": lg, "error": repr(e)[:300]})
        if (li + 1) % 10 == 0 or li + 1 == len(by_log):
            bad = sum(1 for r in rows.values() if r.get("err_max", 0) > 1e-3)
            print(f"  [{li + 1}/{len(by_log)}] rows {len(rows)} ctrl {n_ctrl} over-bar {bad}", flush=True)
    errs = np.array([r["err_max"] for r in rows.values() if "err_max" in r])
    bad = sorted(((t, r) for t, r in rows.items() if r.get("err_max", 0) > 1e-3), key=lambda x: -x[1]["err_max"])
    json.dump({"n_tokens": len(toks), "n_rows": len(rows), "n_ctrl": n_ctrl,
               "n_error": sum(1 for r in rows.values() if "error" in r),
               "bar_m": 1e-3, "max_m": float(errs.max()), "median_m": float(np.median(errs)),
               "n_over_bar": len(bad), "over_bar": {t: r for t, r in bad}, "rows": rows},
              open(OUT, "w", encoding="utf-8", newline="\n"), indent=0)
    print("ZZFC_DONE", len(rows), n_ctrl, len(bad), f"{errs.max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
