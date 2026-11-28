#!/usr/bin/env python3
"""P5 - the registration check over EVERY labelled frame of a clip, not a handful.

Four frames is an anecdote. This sweeps all LiDAR spins that have an `obstacle.offline`
label within tolerance and pools the result, so "our 3D boxes land on the LiDAR returns"
becomes a measurement with an n.

⛔ THE CONTROLS ARE THE POINT, AND THEY ARE DERIVED INDEPENDENTLY OF THE THING THEY CHECK:

  * `marginal`  - the occupancy rate of an OBSERVED cell. A no-information predictor.
                  Any hit rate at or below this has measured nothing.
  * `mirror_y`  - y -> -y. `R-2026-09-08-wpa-mirror` is the programme's most expensive
                  geometry defect: an unasserted azimuth sense inverted the reading of
                  whether the trunk localises agents at all. If mirroring does not
                  collapse the hit rate, this pipeline's sense is unfalsified.
  * `shift_y5`  - 5 m left (10 cells). A translation the scene's own structure could mask.
  * `shift_x10` - 10 m forward (20 cells).
  * `rot90`     - (x, y) -> (-y, x). A rotation, which a shift cannot detect.

⭐ A positive assertion alone ("the boxes overlap returns") passes on a mirrored geometry
in a dense scene. Only a control that MUST read a known-bad value discriminates.

Usage: python p5_registration_sweep.py --clip <uuid> --out raw/registration_sweep.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lidar_bev import (  # noqa: E402
    CartesianBEVSpec, ZBand, DRACO_ATTR_INTENSITY, DRACO_ATTR_POINT_TS_US,
    deskew_rigid, lidar_to_rig, rasterise_cartesian,
)
from p2_build_bev_gt import (  # noqa: E402
    CAM_DIR, JOIN, LIDAR_CACHE, box_cells, ego_state_at, load_ego, load_extrinsics,
)

ARMS = {
    "real": {},
    "mirror_y": {"mirror": True},
    "shift_y5": {"dy": 5.0},
    "shift_x10": {"dx": 10.0},
}


def rot90_cells(agents, spec):
    """(x, y) -> (-y, x): a rotation no translation control can stand in for."""
    rot = [{**a, "cx": -float(a["cy"]), "cy": float(a["cx"]),
            "yaw": float(a["yaw"]) + np.pi / 2} for a in agents]
    return box_cells(rot, spec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="raw/registration_sweep.json")
    ap.add_argument("--max-join-dt-ms", type=float, default=60.0)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import DracoPy

    clip = args.clip
    sha12 = hashlib.sha256(clip.encode()).hexdigest()[:12]
    cart, zb = CartesianBEVSpec(), ZBand()

    pf = pq.ParquetFile(Path(LIDAR_CACHE) / f"{clip}.lidar_top_360fov.parquet")
    sp = pq.read_table(Path(LIDAR_CACHE) / f"{clip}.lidar_top_360fov.parquet",
                       columns=["spin_start_timestamp", "spin_end_timestamp"]).to_pydict()
    s_mid = ((np.asarray(sp["spin_start_timestamp"], dtype=np.int64)
              + np.asarray(sp["spin_end_timestamp"], dtype=np.int64)) // 2)
    ext = load_extrinsics(clip, "lidar_top_360fov")
    ego = load_ego(clip)

    jt, ja = [], []
    with lzma.open(JOIN, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("clip_id") == clip:
                jt.append(float(r["t_s"]))
                ja.append(r.get("agents", []))
    jt = np.asarray(jt)
    if jt.size == 0:
        raise SystemExit("clip has no rows in the B1 EVAL join")

    # ⭐ POOLED counters, not a mean of per-frame rates: a mean-of-ratios over frames
    # with 1-2 box cells is dominated by the sparsest frames. Pool the cells.
    tot = {k: {"hit": 0, "cells": 0} for k in list(ARMS) + ["rot90"]}
    marg_hit = marg_cells = 0
    per_frame, t0 = [], time.time()
    n_used = n_skipped_dt = n_skipped_nocells = 0

    for si in range(pf.metadata.num_row_groups):
        t_ref = int(s_mid[si])
        ji = int(np.abs(jt - t_ref * 1e-6).argmin())
        dt_ms = abs(jt[ji] - t_ref * 1e-6) * 1e3
        if dt_ms > args.max_join_dt_ms:
            n_skipped_dt += 1
            continue
        agents = ja[ji]
        blob = pf.read_row_group(si, columns=["draco_encoded_pointcloud"]).column(0)[0].as_py()
        pc = DracoPy.decode(blob)
        pts = np.asarray(pc.points)
        if pts.shape[0] == 0 or float(np.abs(pts).max()) == 0.0:
            raise SystemExit(f"spin {si}: EMPTY cloud - refusing to score")
        at = {a["unique_id"]: a["data"] for a in pc.attributes}
        v_ms, yr = ego_state_at(ego, t_ref)
        rig = deskew_rigid(lidar_to_rig(pts, ext),
                           at[DRACO_ATTR_POINT_TS_US][:, 0].astype(np.int64),
                           t_ref, v_ms, yr)
        C = rasterise_cartesian(rig, at[DRACO_ATTR_INTENSITY][:, 0].astype(np.uint8),
                                cart, zb)
        occ = C["occ"] > 0
        obs = C["observed"] > 0
        marg_hit += int(occ[obs].sum())
        marg_cells += int(obs.sum())

        row = {"spin": si, "t_s": round(t_ref * 1e-6, 4), "join_dt_ms": round(dt_ms, 1),
               "n_agents": len(agents)}
        any_cells = False
        for name, kw in ARMS.items():
            bc = box_cells(agents, cart, **kw) & obs
            n = int(bc.sum())
            tot[name]["cells"] += n
            tot[name]["hit"] += int(occ[bc].sum())
            row[name] = [int(occ[bc].sum()), n]
            any_cells |= (name == "real" and n > 0)
        bc = rot90_cells(agents, cart) & obs
        tot["rot90"]["cells"] += int(bc.sum())
        tot["rot90"]["hit"] += int(occ[bc].sum())
        row["rot90"] = [int(occ[bc].sum()), int(bc.sum())]
        if not any_cells:
            n_skipped_nocells += 1
        n_used += 1
        per_frame.append(row)

    base = marg_hit / marg_cells if marg_cells else float("nan")
    out = {
        "schema": "tanitad.lidar_bev_registration/1",
        "clip_sha12": sha12,
        "spins_scored": n_used,
        "spins_skipped_join_dt": n_skipped_dt,
        "frames_with_no_box_cell_in_grid": n_skipped_nocells,
        "max_join_dt_ms": args.max_join_dt_ms,
        "wall_s": round(time.time() - t0, 1),
        "marginal_occ_rate_observed": base,
        "marginal_cells": marg_cells,
        "arms": {k: {"hit_cells": v["hit"], "box_cells_observed": v["cells"],
                     "hit_rate": (v["hit"] / v["cells"]) if v["cells"] else None,
                     "lift_over_marginal": ((v["hit"] / v["cells"]) / base)
                     if (v["cells"] and base > 0) else None}
                 for k, v in tot.items()},
        "per_frame": per_frame,
        "note": "hit_rate is POOLED over cells, not a mean of per-frame rates. Cells are "
                "0.5 m; boxes are `obstacle.offline` footprints in the rig frame; "
                "occupancy is LiDAR-derived. Everything is a LABEL - inference is VISION-ONLY.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"spins scored {n_used}  marginal {base:.5f} over {marg_cells:,} observed cells")
    for k, v in out["arms"].items():
        hr = v["hit_rate"]
        print(f"  {k:10s} hit {v['hit_cells']:>6,} / {v['box_cells_observed']:>6,} cells  "
              f"= {hr if hr is None else round(hr, 4)}   lift "
              f"{v['lift_over_marginal'] if v['lift_over_marginal'] is None else round(v['lift_over_marginal'], 2)}x")
    print(f"[p5] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
