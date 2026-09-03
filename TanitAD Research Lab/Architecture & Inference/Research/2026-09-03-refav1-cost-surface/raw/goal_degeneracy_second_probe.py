"""SECOND, MODEL-FREE PROBE of factor (0) — the degenerate imagined goal.

`cost_surface_probe.py` reads `goal_kappa_max` off the LIVE model. This probe reads the
BANKED decoded tokens (`goal_lat_cl` / `goal_lon_cl`) and the banked `v0` out of the T1
dumps and re-derives `canonical_controls` with NO model in the loop. Two mechanisms, one
answer — which is what CLAUDE.md's *"absence found at ONE location is not absence"* rule
asks for on a load-bearing structural claim.

⛔ The claim it must settle: on windows where `canonical_controls` is EXACTLY zero, the
planner's imagined goal rollout and the `cv` candidate's rollout are the SAME forward
pass, so the goal term is `1 - cos(x, x) = 0` exactly; with every term non-negative the
straight plan is then the GLOBAL MINIMUM by construction, whatever the lateral potency,
the ×2.9 boundary or the κ² penalty do.
"""
import glob
import json
import os
from collections import Counter

import numpy as np
import torch

from tanitad.models.v6 import tactical_lat_actions, tactical_lon_actions_v
from tanitad.refs.refa_v1 import canonical_controls

BASE = r"C:\Users\Admin\refav1_eval_slice"
VOCAB = "v7.0"
OP_STEPS, OP_DT = 30, 0.2

lat_v = list(tactical_lat_actions(VOCAB))
lon_v = list(tactical_lon_actions_v(VOCAB))
out = {"vocab": VOCAB, "op_steps": OP_STEPS, "op_dt": OP_DT,
       "lat_vocab": lat_v, "lon_vocab": lon_v, "dumps": {}}

for dd in ("t1_dump", "t1_dump_ep2"):
    rows = []
    for f in sorted(glob.glob(os.path.join(BASE, dd, "ep*.npz"))):
        e = np.load(f)
        d = np.load(os.path.join(os.path.dirname(f), "decisions",
                                 os.path.basename(f)), allow_pickle=True)
        v0, lat, lon = e["v0"], d["goal_lat_cl"], d["goal_lon_cl"]
        for i in range(v0.size):
            c = canonical_controls(lat_v[int(lat[i])], lon_v[int(lon[i])],
                                   float(v0[i]), OP_STEPS, OP_DT)
            rows.append({"lat": lat_v[int(lat[i])], "lon": lon_v[int(lon[i])],
                         "v0": float(v0[i]),
                         "a_max": float(c[:, 0].abs().max()),
                         "k_max": float(c[:, 1].abs().max())})
    n = len(rows)
    zero = sum(1 for r in rows if max(r["a_max"], r["k_max"]) == 0.0)
    blk = {"n": n, "n_zero_canonical": zero, "frac_zero": round(zero / n, 4),
           "n_kappa_nonzero": sum(1 for r in rows if r["k_max"] > 0),
           "n_accel_nonzero": sum(1 for r in rows if r["a_max"] > 0),
           "lat_counts": dict(Counter(r["lat"] for r in rows)),
           "lon_counts": dict(Counter(r["lon"] for r in rows)),
           "nonzero_by_token": {f"{a}|{b}": c for (a, b), c in Counter(
               (r["lat"], r["lon"]) for r in rows
               if max(r["a_max"], r["k_max"]) > 0).items()}}
    out["dumps"][dd] = blk
    print(f"== {dd}: n={n}  EXACTLY-ZERO canonical controls = {zero} "
          f"({zero / n:.1%})  kappa-nonzero={blk['n_kappa_nonzero']}  "
          f"accel-nonzero={blk['n_accel_nonzero']}")
    print(f"   lat: {blk['lat_counts']}")
    print(f"   lon: {blk['lon_counts']}")
    print(f"   non-zero windows by (lat|lon): {blk['nonzero_by_token']}")

p = os.path.join(BASE, "costsurface", "goal_degeneracy_second_probe.json")
with open(p, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("\nwritten:", p)
