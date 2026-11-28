#!/usr/bin/env python3
"""Lateral progress of each re-rolled twin toward its twin lane: f(t) = d_logged / (d_logged + d_twin) per pose (0 = on the logged
centreline, 1 = on the twin centreline), at 1/2/3/4 s, plus the ego's initial speed. Diagnoses WHY a twin is not closer to its lane.
    cd /workspace/ops/d7fix/pkg/refe && PYTHONPATH=../code $PY lateral_profile.py <probe dir> <arm> [<mc>] [<bank subdir>]
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import shapely

import online_labeller as OL

R, ARM = Path(sys.argv[1]), sys.argv[2]
MC = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/workspace/ops/d7fix/mc_final")
BANKDIR = sys.argv[4] if len(sys.argv) > 4 else f"vbank_{ARM}"
rows = [json.loads(l) for l in open(R / BANKDIR / "targets_rank1.jsonl", encoding="utf-8")]
out = []
for r in rows:
    k = int(r["aug"]["value"])
    tw_p, lg_p = OL.parts_path(MC, r["log_name"], r["token"], r["aug"]), OL.parts_path(MC, r["log_name"], r["token"], None)
    if not tw_p.is_file() or not lg_p.is_file():
        continue
    tw, lg = OL.load_parts(tw_p)[3].linestring, OL.load_parts(lg_p)[3].linestring
    xy = shapely.points(np.asarray(r["traj_world"], dtype=np.float64)[:, :2])
    dl, dt = shapely.distance(lg, xy), shapely.distance(tw, xy)
    f = dl / np.maximum(dl + dt, 1e-6)
    o0 = shapely.points(np.asarray(r["origin_world"][:2], dtype=np.float64))
    sep = float(shapely.distance(tw, o0) + shapely.distance(lg, o0))      # lane separation seen from the start pose
    out.append({"k": k, "div": r["aug"]["divergence_m"], "v0": r["ego"][6], "sep0": round(sep, 2),
                "f1": float(f[4]), "f2": float(f[9]), "f3": float(f[14]), "f4": float(f[19]),
                "closer": bool(dt.mean() < dl.mean()), "end_closer": bool(dt[-1] < dl[-1]),
                "monotone": bool(np.all(np.diff(f) >= -0.02))})
res = {}
for k in (1, 2, 3):
    o = [x for x in out if x["k"] == k]
    hi = [x for x in o if x["div"] >= 0.5]
    if not o:
        continue
    res[k] = {"n": len(o), "n_div_ge_0_5": len(hi),
              "median_f_at_1_2_3_4s_div_ge_0_5": [round(st.median(x[f"f{i}"] for x in hi), 3) for i in (1, 2, 3, 4)] if hi else None,
              "frac_f4_ge_0_5_div_ge_0_5": round(sum(x["f4"] >= 0.5 for x in hi) / len(hi), 3) if hi else None,
              "frac_monotone_div_ge_0_5": round(sum(x["monotone"] for x in hi) / len(hi), 3) if hi else None,
              "median_v0_mps_div_ge_0_5": round(st.median(x["v0"] for x in hi), 2) if hi else None,
              "median_lane_sep_m": round(st.median(x["sep0"] for x in o), 2)}
print(json.dumps(res, indent=1))
(R / "results" / f"lateral_profile_{ARM}{'' if BANKDIR == f'vbank_{ARM}' else '_' + BANKDIR}.json").write_text(json.dumps({"summary": res, "twins": out}, indent=1))
