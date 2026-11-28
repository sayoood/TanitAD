#!/usr/bin/env python3
"""G-VIOL feasibility probe (read-only): does a T1-kin rollout carry a rule-violation measure, and what does it cost to compute one from the dumps?
Uses the T0 eval's OWN function chain (v8f_mini_eval.py:98-100, 149-157): D.eval_context -> D.v9_label_dict -> C.constraints_from_v9 -> C.violations.
Input: a banked T1-kin dump dir (ep*.npz, arms cl / ol / fl / ha0, key g = GT in the t0 ego frame). CPU, 2 threads."""
import glob
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "2"
sys.dont_write_bytecode = True
TREE = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a"
PK = f"{TREE}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
for p in (f"{TREE}/stack", f"{TREE}/taniteval"):
    sys.path.insert(0, p)
import numpy as np  # noqa: E402

from tanitad.data import v8f_mini_data as D  # noqa: E402
from taniteval import v8f_consistency as C  # noqa: E402


def main(dump_dir):
    t0 = time.time()
    grid = json.load(open(f"{PK}/eval_grid_windows.json"))
    J = D.load_join(f"{PK}/join_eval141.npz")
    X = D.eval_context(grid, f"{PK}/eval141_kin.npz", J, kin_mode="yaw")
    gtc = C.constraints_from_v9(D.v9_label_dict(X), X["limit_ms"], X["lead_gap_h"], X["s_h"])
    t_ctx = time.time() - t0
    files = sorted(glob.glob(os.path.join(dump_dir, "ep*.npz")))
    arms = {}
    for f in files:
        with np.load(f) as z:
            for k in ("cl", "ol", "fl", "g", "eid", "ws"):
                arms.setdefault(k, []).append(z[k])
    arms = {k: np.concatenate(v) for k, v in arms.items()}
    out = {"n_files": len(files), "n_windows": int(len(arms["cl"])), "ctx_seconds": round(t_ctx, 2), "keys_in_dump": sorted(np.load(files[0]).files)}
    out["grid_aligned"] = bool(np.array_equal(arms["ws"], X["t"]) and len(arms["cl"]) == len(X["t"]))
    out["gt_equal_ctx_gt_maxabs"] = float(np.abs(arms["g"] - X["gt"]).max())
    t1 = time.time()
    for arm in ("cl", "ol", "fl", "g"):
        V = C.violations(arms[arm], gtc)
        out[arm] = {"viol_any_rate": float(V["any_violated"].mean()), "any_active_rate": float(V["any_active"].mean()),
                    **{r: {"active": float(V[r]["active"].mean()), "violated": float(V[r]["violated"].mean())} for r in C.RULE_NAMES}}
    out["violations_seconds_4_arms"] = round(time.time() - t1, 2)
    out["total_seconds"] = round(time.time() - t0, 2)
    return out


if __name__ == "__main__":
    r = main(sys.argv[1])
    print(json.dumps(r, indent=1))
