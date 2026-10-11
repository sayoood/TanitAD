"""CEILING of the head's output resolution: if a head predicted the label PERFECTLY at cell size c (block-mean SDF, bilinear
interpolation between cell centres), how often would the resulting field give NAVSIM's exact drivable-area verdict at the
footprint corners of (shifted) rank-0 targets?  A head cannot beat this, so it bounds what a 2 m output grid can do for
a scorer-side use of the predicted field.  Same 20 samples / same 13 lateral shifts as proto_label.py.

Run: PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe resolution_ceiling.py
Evidence class: MEASURED (this script, CPU, dev box)."""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import proto_label as PL  # noqa: E402  (sets the env roots, imports navsim_dac / navtrain_scenarios)
import navsim_dac as ND  # noqa: E402
import navtrain_scenarios as NS  # noqa: E402

CLIP = 6.0           # the head's SDF range (m)
SHIFTS = (-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def block_mean(sdf_m, b):
    nx, ny = sdf_m.shape
    return sdf_m.reshape(nx // b, b, ny // b, b).mean((1, 3))


def interp(coarse, cell, pts):
    """Bilinear interpolation of a coarse field defined at cell centres; pts [...,2] in the ego frame (m)."""
    nx, ny = coarse.shape
    fx = (pts[..., 0] - PL.X0) / cell - 0.5
    fy = (pts[..., 1] - PL.Y0) / cell - 0.5
    inb = (fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
    x0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
    wx, wy = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
    v = ((1 - wx) * (1 - wy) * coarse[x0, y0] + wx * (1 - wy) * coarse[x0 + 1, y0]
         + (1 - wx) * wy * coarse[x0, y0 + 1] + wx * wy * coarse[x0 + 1, y0 + 1])
    return v, inb


def main():
    import shapely
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    dbs = NS.index_dbs()
    rows = [json.loads(l) for l in open("E:/Projects/TanitAD/data/refe_navtrain/r0_shard0/targets_rank0.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["log_name"] in dbs]
    by = {}
    for r in rows:
        by.setdefault(r["log_name"], []).append(r)
    rng = random.Random(20261011)
    logs = sorted(by); rng.shuffle(logs)
    picks = [rng.choice(by[lg]) for lg in logs[:20]]            # IDENTICAL selection to proto_label.py
    cells = {1: 0.5, 2: 1.0, 4: 2.0, 8: 4.0}                     # block -> cell size (m)
    acc = {b: dict(pts=0, dis=0, n=0, vdis=0, nviol=0, oob=0) for b in cells}
    acc["mask2m_nearest"] = dict(pts=0, dis=0, n=0, vdis=0, nviol=0, oob=0)
    for r in picks:
        map_name, _ = NS._log_rows(dbs[r["log_name"]])
        api = get_maps_api(os.environ["NUPLAN_MAPS_ROOT"], NS.MAP_VERSION, map_name)
        pose = tuple(r["origin_world"])
        cx = pose[0] + ND.REAR_AXLE_TO_CENTER * math.cos(pose[2]); cy = pose[1] + ND.REAR_AXLE_TO_CENTER * math.sin(pose[2])
        polys = PL.polys_at(api, cx, cy)
        _, sdf_i8 = PL.build_label(polys, pose)
        sdf_m = np.clip(sdf_i8.astype(np.float64) * PL.SDF_UNIT_M, -CLIP, CLIP)
        fine_mask = sdf_i8 > 0
        for shift in SHIFTS:
            q = np.asarray(r["traj"], float).copy(); q[:, 1] += shift
            p8 = ND.to_navsim(q)
            ce = PL.corners_ego(p8)
            cg = ND.corners_global(ND.at_10hz(p8), *pose)
            flat = cg.reshape(-1, 2)
            ins = np.zeros(len(flat), bool)
            for pg in polys:
                ins |= shapely.contains_xy(pg, flat[:, 0], flat[:, 1])
            ins = ins.reshape(cg.shape[:-1])
            exact = bool(~(ins.sum(-1) == 4).all())
            for b, cell in cells.items():
                v, inb = interp(block_mean(sdf_m, b), cell, ce)
                pred = v > 0
                a = acc[b]; a["pts"] += ins.size; a["dis"] += int((pred != ins).sum()); a["oob"] += int((~inb).sum())
                a["n"] += 1; a["nviol"] += int(exact); a["vdis"] += int(bool(~(pred.sum(-1) == 4).all()) != exact)
            # alternative head output: block-mean MASK fraction at 2 m, nearest cell, >0.5
            frac = fine_mask.reshape(PL.NX // 4, 4, PL.NY // 4, 4).mean((1, 3))
            ix = np.clip(((ce[..., 0] - PL.X0) // 2.0).astype(int), 0, frac.shape[0] - 1)
            iy = np.clip(((ce[..., 1] - PL.Y0) // 2.0).astype(int), 0, frac.shape[1] - 1)
            pred = frac[ix, iy] > 0.5
            a = acc["mask2m_nearest"]; a["pts"] += ins.size; a["dis"] += int((pred != ins).sum()); a["n"] += 1
            a["nviol"] += int(exact); a["vdis"] += int(bool(~(pred.sum(-1) == 4).all()) != exact)
    out = {}
    for k, a in acc.items():
        out[str(k) if not isinstance(k, int) else f"sdf_cell_{cells[k]}m"] = dict(
            corner_agree=1 - a["dis"] / a["pts"], verdict_agree=1 - a["vdis"] / a["n"], n_verdicts=a["n"], n_violating=a["nviol"],
            corners_outside_extent=a["oob"])
    (HERE / "resolution_ceiling.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out.items():
        print(f"{k:<18} corner agree {v['corner_agree']*100:6.2f}%  verdict agree {v['verdict_agree']*100:6.2f}%  (n {v['n_verdicts']}, violating {v['n_violating']})")


if __name__ == "__main__":
    main()
