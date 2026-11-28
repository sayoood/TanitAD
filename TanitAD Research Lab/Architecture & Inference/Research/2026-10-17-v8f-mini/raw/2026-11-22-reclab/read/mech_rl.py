"""mech_rl.py (MM 2026-11-22): the RECLAB card's MECH read (section 3): RL - RH at T0 ADE6, paired, per seed.
MECH-YES when the paired interval sits below 0 at both seeds.  Same instrument and estimator as rh_t0.py's G-T0
(per-window `ade6` and `eid` of out/t0/eval_<ARM>.npz; taniteval.ci.paired_episode_cluster_bootstrap B 2,000 seed 0).
Tier T0 (a WM / tactical diagnostic, never driving performance).  Beside: RL - B0 (the guard, re-read) and RH - B0.
Usage: python -I mech_rl.py <read dir> <out json>"""
import json
import os
import sys

import numpy as np

RD = sys.argv[1]
OUTF = sys.argv[2]
sys.path.insert(0, RD + "/tree/taniteval")
sys.path.insert(0, RD + "/tree/stack")
from taniteval.ci import paired_episode_cluster_bootstrap as pb  # noqa: E402
import taniteval.ci as ci  # noqa: E402

T0 = RD + "/out/t0"


def load(arm):
    z = np.load(f"{T0}/eval_{arm}.npz", allow_pickle=True)
    return z["ade6"], z["eid"]


def pair(a, b):
    A, ea = load(a)
    B, eb = load(b)
    assert len(A) == len(B) == 2399 and np.array_equal(ea, eb), (a, b)
    r = pb(A, B, ea, n_boot=2000, seed=0)
    return {"A": a, "B": b, "point_m": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]),
            "mean_A": float(np.mean(A)), "mean_B": float(np.mean(B)), "n_windows": int(len(A)), "n_episodes": int(len(np.unique(ea)))}


out = {"read": "MECH (PREREG_V8F_RECLAB.md section 3): RL - RH at T0 ADE6, paired, per seed; MECH-YES when the interval sits below 0 at both seeds",
       "tier": "T0", "estimator": "taniteval.ci.paired_episode_cluster_bootstrap B 2,000 seed 0 (answers 'another draw of EPISODES'; the two seeds answer 'another TRAINING run')",
       "ci_module": os.path.abspath(ci.__file__).replace("\\", "/"), "per_seed": {}, "beside": {}}
below = []
for s in (0, 1):
    m = pair(f"RL_s{s}", f"RH_s{s}")
    m["below_zero"] = bool(m["hi"] < 0)
    below.append(m["below_zero"])
    out["per_seed"][str(s)] = m
    out["beside"][f"RL_minus_B0_s{s}"] = pair(f"RL_s{s}", f"B0_s{s}")
    out["beside"][f"RH_minus_B0_s{s}"] = pair(f"RH_s{s}", f"B0_s{s}")
out["MECH"] = "MECH-YES" if all(below) else "MECH-NO"
json.dump(out, open(OUTF, "w", encoding="utf-8"), indent=1)
print(out["MECH"])
for s in ("0", "1"):
    m = out["per_seed"][s]
    print(f"s{s} RL-RH {m['point_m']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}]  RL-B0 {out['beside']['RL_minus_B0_s' + s]['point_m']:+.4f} [{out['beside']['RL_minus_B0_s' + s]['lo']:+.4f}, {out['beside']['RL_minus_B0_s' + s]['hi']:+.4f}]  RH-B0 {out['beside']['RH_minus_B0_s' + s]['point_m']:+.4f} [{out['beside']['RH_minus_B0_s' + s]['lo']:+.4f}, {out['beside']['RH_minus_B0_s' + s]['hi']:+.4f}]")
