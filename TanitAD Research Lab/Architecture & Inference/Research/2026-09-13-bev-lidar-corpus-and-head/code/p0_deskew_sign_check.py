#!/usr/bin/env python3
"""P0 - is the deskew ROTATION sign in `lidar_bev.deskew_rigid` right? MEASURED, not argued.

The suspicion (from reading the code): `deskew_rigid` maps a point captured at t_pt into
the ego frame at t_ref with `dth = yaw_rate * (t_pt - t_ref)` and then rotates by `-dth`.
For a left-turning ego (+yaw, +y LEFT), a static point dead ahead at t_ref appears to the
RIGHT at a later t_pt, so mapping it back needs a rotation by `+dth`. If that is right,
the 09-11 deskew DOUBLES the yaw smear instead of cancelling it.

⭐ The discriminating experiment: TWO CONSECUTIVE SPINS deskewed to the SAME instant must
put static structure in the same place. Every azimuth of spin k+1 is sampled exactly
~100 ms after the same azimuth of spin k, so the yaw term is maximally exercised.
Agreement = median xy nearest-neighbour distance from spin-k obstacle points to spin-(k+1)
obstacle points (r in [5, 40] m, z-band [0.3, 3.0] m), under three variants that share
the SAME translation term:
  * `rot_off`   -- translation only
  * `rot_code`  -- rotate by -dth (the 09-11 code)
  * `rot_flip`  -- rotate by +dth (the derivation)
⭐ Control with a KNOWN answer: on low-|yaw-rate| instants the three variants must agree
to within noise (the rotation term is ~0). A variant ranking that also appears on straight
driving is a defect of the probe, not a sign.

Usage:  python p0_deskew_sign_check.py --clips <id,id> --out ../raw/p0_deskew_sign_check.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parents[1] / "2026-09-11-lidar-bev-gt" / "code"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PRIOR))
from lidar_bev import DRACO_ATTR_POINT_TS_US, lidar_to_rig  # noqa: E402
from p2_build_bev_gt import ego_state_at, load_ego, load_extrinsics  # noqa: E402
from lidar_fetch import LIDAR_CACHE, sha12  # noqa: E402


def deskew(pts, pt_ts, t_ref, v, r, sign):
    dt = (pt_ts.astype(np.float64) - float(t_ref)) * 1e-6
    dth = sign * r * dt
    x = pts[:, 0] + v * dt
    y = pts[:, 1]
    c, s = np.cos(dth), np.sin(dth)
    out = pts.copy()
    out[:, 0] = x * c - y * s
    out[:, 1] = x * s + y * c
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", default=str(HERE.parent / "raw" / "p0_deskew_sign_check.json"))
    ap.add_argument("--yaw-hi", type=float, default=0.15)
    ap.add_argument("--yaw-lo", type=float, default=0.02)
    ap.add_argument("--max-per-class", type=int, default=12)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import DracoPy
    from scipy.spatial import cKDTree

    res = {"schema": "tanitad.deskew_sign_check/1",
           "evidence_class": "MEASURED (ours, dev-box CPU)",
           "variants": {"rot_off": "translation only", "rot_code": "rotate by -yaw_rate*dt "
                        "(lidar_bev.deskew_rigid as shipped 2026-09-11)",
                        "rot_flip": "rotate by +yaw_rate*dt"},
           "metric": "median xy nearest-neighbour distance [m], spin k -> spin k+1, both "
                     "deskewed to the boundary instant; obstacle z-band points, r in [5,40] m",
           "instants": []}
    for cid in [c for c in args.clips.split(",") if c]:
        lp = Path(LIDAR_CACHE) / f"{cid}.lidar_top_360fov.parquet"
        pf = pq.ParquetFile(lp)
        sp = pq.read_table(lp, columns=["spin_start_timestamp", "spin_end_timestamp"]).to_pydict()
        s0 = np.asarray(sp["spin_start_timestamp"], dtype=np.int64)
        s1 = np.asarray(sp["spin_end_timestamp"], dtype=np.int64)
        mid = (s0 + s1) // 2
        ext = load_extrinsics(cid, "lidar_top_360fov")
        ego = load_ego(cid)
        # candidate boundaries between spin k and k+1, classified by |yaw rate|
        cands = []
        for k in range(len(mid) - 1):
            t = int((mid[k] + mid[k + 1]) // 2)
            v, r = ego_state_at(ego, t)
            cands.append((k, t, v, r))
        hi = [c for c in cands if abs(c[3]) >= args.yaw_hi and c[2] > 2.0]
        lo = [c for c in cands if abs(c[3]) <= args.yaw_lo and c[2] > 2.0]
        # spread the picks over the clip rather than taking a contiguous run
        def spread(lst):
            if len(lst) <= args.max_per_class:
                return lst
            idx = np.linspace(0, len(lst) - 1, args.max_per_class).round().astype(int)
            return [lst[i] for i in idx]
        for cls, picks in (("high_yaw", spread(hi)), ("low_yaw", spread(lo))):
            for k, t, v, r in picks:
                clouds = []
                for kk in (k, k + 1):
                    blob = pf.read_row_group(kk, columns=["draco_encoded_pointcloud"]).column(0)[0].as_py()
                    pc = DracoPy.decode(blob)
                    pts = np.asarray(pc.points)
                    attrs = {a["unique_id"]: a["data"] for a in pc.attributes}
                    clouds.append((lidar_to_rig(pts, ext), attrs[DRACO_ATTR_POINT_TS_US][:, 0].astype(np.int64)))
                rec = {"clip": sha12(cid), "class": cls, "k": k, "v_ms": round(v, 3),
                       "yaw_rate_rps": round(r, 4)}
                for name, sign in (("rot_off", 0.0), ("rot_code", -1.0), ("rot_flip", +1.0)):
                    sel = []
                    for rig, ts in clouds:
                        d = deskew(rig, ts, t, v, r, sign)
                        rr = np.hypot(d[:, 0], d[:, 1])
                        m = (d[:, 2] >= 0.3) & (d[:, 2] <= 3.0) & (rr >= 5) & (rr <= 40)
                        sel.append(d[m, :2])
                    tree = cKDTree(sel[1])
                    dist, _ = tree.query(sel[0], k=1)
                    rec[name] = round(float(np.median(dist)), 4)
                    rec[name + "_p90"] = round(float(np.percentile(dist, 90)), 4)
                res["instants"].append(rec)
                print(json.dumps(rec), flush=True)

    summ = {}
    for cls in ("high_yaw", "low_yaw"):
        rows = [r for r in res["instants"] if r["class"] == cls]
        if not rows:
            continue
        summ[cls] = {
            "n": len(rows),
            **{f"median_{n}": round(float(np.median([r[n] for r in rows])), 4)
               for n in ("rot_off", "rot_code", "rot_flip")},
            "n_flip_better_than_code": sum(r["rot_flip"] < r["rot_code"] for r in rows),
            "n_flip_better_than_off": sum(r["rot_flip"] < r["rot_off"] for r in rows),
            "n_code_better_than_off": sum(r["rot_code"] < r["rot_off"] for r in rows),
        }
    res["summary"] = summ
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
