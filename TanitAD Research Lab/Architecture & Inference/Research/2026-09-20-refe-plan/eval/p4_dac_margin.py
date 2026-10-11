#!/usr/bin/env python3
"""P4 diagnostic D3 (EXPLORATORY): drivable-area MARGIN of the teacher's own path vs the human's, on D1's token sets.

D1: 87 % of the teacher's NAVSIM DAC failures are compliant in the teacher's own rollout and fail only after NAVSIM's
tracker. D2: the tracker deviates from the HUMAN plan just as much (median 0.79 vs 0.81 m on those tokens), yet the human
never fails. The remaining explanation is margin -- this measures it: per plan, the minimum over steps and the 4
footprint corners of the signed distance to the drivable-area boundary (positive = inside), on the plan's OWN poses
(0.1 s interpolation of the 8 poses, no tracker), same polygons and footprint as refe/navsim_dac.py.

    python eval/run_p4.py --script p4_dac_margin.py --log <log> -- --teacher-seam <npz> --out <json>
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "refe"))
EXPORT = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
TEST_DB = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
D1 = os.path.join(PKG, "raw", "2026-10-05-teacher-navtest", "d1_dac_anatomy.json")


def margin(polys, corners) -> float:
    """min signed distance (m) of all corners [T, 4, 2] to the union of polys; > 0 inside, < 0 outside."""
    import shapely
    from shapely.ops import unary_union
    u = unary_union(polys)
    pts = corners.reshape(-1, 2)
    geo = shapely.points(pts[:, 0], pts[:, 1])
    inside = shapely.contains(u, geo)
    d = shapely.distance(u.boundary, geo)
    return float(np.min(np.where(inside, d, -d)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-seam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    import navsim_dac as ND
    import navtrain_scenarios as NS
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    d1 = {r["token"]: r for r in json.load(open(D1, encoding="utf-8"))["rows"]}
    z = np.load(a.teacher_seam)
    pos = {str(t): i for i, t in enumerate(z["token"])}
    by: dict = {}
    for t in d1:
        by.setdefault(exp[t]["log_name"], []).append(t)
    rows, t0 = [], time.time()
    for lg, lt in sorted(by.items()):
        for sc in NS.build_scenarios_for_log(os.path.join(TEST_DB, f"{lg}.db"), lt, history_rows=1, future_rows=80):
            tok = sc._initial_lidar_token
            ego = sc.get_ego_state_at_iteration(0)
            ra = ego.rear_axle
            polys = ND.drivable_polygons(sc.map_api, ego)
            r = d1[tok]
            cls = ("A" if r.get("raw20") else "B" if r.get("interp8") else "C" if r.get("sim8") else "pass") \
                if r["set"] == "dac_fail" else "pass_control"
            out = {"token": tok, "set": r["set"], "class": cls}
            for nm, p8 in (("teacher", np.asarray(z["poses"][pos[tok]], np.float64)),
                           ("human", np.asarray(exp[tok]["human_future_poses"], np.float64))):
                out[nm + "_margin_m"] = margin(polys, ND.corners_global(ND.at_10hz(p8), ra.x, ra.y, ra.heading))
            rows.append(out)
    res = {"what": "D3: min signed distance of the plan's own footprint (0.1 s interpolation, no tracker) to the "
                   "drivable-area boundary; > 0 inside", "n": len(rows), "seconds": round(time.time() - t0, 1)}
    for c in ("C", "A", "pass_control"):
        sel = [r for r in rows if r["class"] == c]
        if not sel:
            continue
        tm = np.array([r["teacher_margin_m"] for r in sel])
        hm = np.array([r["human_margin_m"] for r in sel])
        res[c] = {"n": len(sel), "teacher_margin_median_m": round(float(np.median(tm)), 3),
                  "human_margin_median_m": round(float(np.median(hm)), 3),
                  "teacher_margin_p10_m": round(float(np.percentile(tm, 10)), 3),
                  "human_margin_p10_m": round(float(np.percentile(hm, 10)), 3),
                  "share_teacher_margin_lt_human": round(float(np.mean(tm < hm)), 3),
                  "share_teacher_margin_lt_0p5m": round(float(np.mean(tm < 0.5)), 3),
                  "share_human_margin_lt_0p5m": round(float(np.mean(hm < 0.5)), 3)}
    res["rows"] = rows
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    print("ZZD3_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
