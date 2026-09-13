#!/usr/bin/env python3
"""P2b - is the POLAR `observed` rule of `lidar_bev.rasterise_polar` biased? MEASURED.

Trigger: P2's content band on the polar48 marginal (declared [0.005, 0.50] BEFORE the build)
quarantined 3 of the first 24 clips at 0.501 / 0.553 / 0.692, and the 21 that passed read
0.194-0.496 -- against Cartesian occupancy of 3.5-11.6 % of the grid on the same clips.

Hypothesis from reading the code: the Cartesian rule is
    observed = ~shadow | occ | (n_pts > 0)
but the polar rule is
    observed = ~shadow | occ
so BEHIND the first occupied cell of a column, an OCCUPIED cell stays "observed" while a
FREE cell with ground returns becomes "occluded". The scored set then keeps the positives
and drops the negatives behind the first hit -- a label-side bias that inflates the base
rate and would reward a head for "everything past the first obstacle is occupied".

Discriminating measurement, same spins, same deskew, three rules on polar48:
  * A `shipped`   ~shadow | occ                       (lidar_bev.rasterise_polar as shipped)
  * B `with_npts` ~shadow | occ | n_pts > 0           (the Cartesian rule, ported)
  * C `no_occ_or` ~shadow                             (control: pure geometric shadow)
and the share of scored positives that lie BEHIND the first hit of their column.
⭐ Control with a known answer: IN FRONT of the first hit, A and B must agree exactly --
the rules differ only behind it (shadow is False there, so both are True).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lidar_bev import (  # noqa: E402
    PolarBEVSpec, ZBand, DRACO_ATTR_POINT_TS_US, deskew_rigid, lidar_to_rig, rasterise_polar,
)
from lidar_fetch import LIDAR_CACHE, sha12  # noqa: E402
import p2_build_corpus as B  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--n-instants", type=int, default=24)
    ap.add_argument("--out", default=str(HERE.parent / "raw" / "p2b_observed_rule_probe.json"))
    args = ap.parse_args()
    import pyarrow.parquet as pq
    import DracoPy

    spec = PolarBEVSpec(n_az=40, n_rng=48)
    zb = ZBand()
    r_c = spec.r_min_m + (np.arange(spec.n_rng) + 0.5) * spec.cell_rng_m
    res = {"schema": "tanitad.observed_rule_probe/1", "evidence_class": "MEASURED (ours)",
           "grid": "polar48 48x40, 1.25 m x 3 deg", "clips": []}
    for cid in [c for c in args.clips.split(",") if c]:
        lp = Path(LIDAR_CACHE) / f"{cid}.lidar_top_360fov.parquet"
        pf = pq.ParquetFile(lp)
        sp = pq.read_table(lp, columns=["spin_start_timestamp", "spin_end_timestamp"]).to_pydict()
        mid = (np.asarray(sp["spin_start_timestamp"]) + np.asarray(sp["spin_end_timestamp"])) // 2
        ext = B.extrinsics(cid, "lidar_top_360fov")
        ego = B.load_ego(cid)
        acc = {k: [0, 0] for k in ("A_shipped", "B_with_npts", "C_shadow_only")}
        front_disagree = 0
        behind_pos = [0, 0]      # [positives behind first hit, all scored positives] under A
        free_behind_with_returns = 0
        for k in np.linspace(5, len(mid) - 6, args.n_instants).round().astype(int):
            blob = pf.read_row_group(int(k), columns=["draco_encoded_pointcloud"]).column(0)[0].as_py()
            pc = DracoPy.decode(blob)
            attrs = {a["unique_id"]: a["data"] for a in pc.attributes}
            ts = attrs[DRACO_ATTR_POINT_TS_US][:, 0].astype(np.int64)
            v, r = B.ego_state_at(ego, int(mid[k]))
            rig = deskew_rigid(lidar_to_rig(np.asarray(pc.points), ext), ts, int(mid[k]), v, r)
            P = rasterise_polar(rig, spec, zb)
            occ = P["occ"] > 0
            npts = P["n_pts"] > 0
            first = np.full(spec.n_az, np.inf)
            for c in range(spec.n_az):
                h = np.nonzero(occ[:, c])[0]
                if h.size:
                    first[c] = r_c[h[0]]
            shadow = r_c[:, None] > (first[None, :] + spec.cell_rng_m)
            rules = {"A_shipped": (~shadow) | occ,
                     "B_with_npts": (~shadow) | occ | npts,
                     "C_shadow_only": ~shadow}
            if not np.array_equal(rules["A_shipped"] & ~shadow, rules["B_with_npts"] & ~shadow):
                front_disagree += 1
            for name, obs in rules.items():
                acc[name][0] += int(occ[obs].sum())
                acc[name][1] += int(obs.sum())
            behind_pos[0] += int((occ & shadow).sum())
            behind_pos[1] += int(occ[rules["A_shipped"]].sum())
            free_behind_with_returns += int((shadow & ~occ & npts).sum())
        rec = {"clip": sha12(cid), "n_instants": args.n_instants,
               "control_rules_agree_in_front_of_first_hit": front_disagree == 0,
               "share_of_scored_positives_behind_first_hit_A": round(behind_pos[0] / max(behind_pos[1], 1), 4),
               "free_cells_behind_first_hit_WITH_returns_dropped_by_A": free_behind_with_returns}
        for name, (o, n) in acc.items():
            rec[f"marginal_{name}"] = round(o / max(n, 1), 4)
            rec[f"n_observed_{name}"] = n
        res["clips"].append(rec)
        print(json.dumps(rec), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
