"""PROTOTYPE: ego-frame drivable-area label (binary mask + signed distance field) from nuPlan map polygons,
using EXACTLY the polygon set NAVSIM's drivable-area check queries (refe/navsim_dac.py `drivable_polygons`:
ROADBLOCK + INTERSECTION + CARPARK_AREA around the ego CENTRE, radius 100 m).

Two build paths are timed and compared (same samples, same raster):
  A  via a NuPlanScenario (what refe/onpolicy_label.py does): needs the navtrain log DB + the map.
  B  via the bank row's own `origin_world` pose + the map only (no scenario, no DB beyond the map name).
Fidelity: the raster's verdict at the footprint corners of (shifted) rank-0 target trajectories is compared with
NAVSIM's EXACT point-in-polygon verdict (`navsim_dac.violations`) on the same corners.

Run (driverl-eval has nuplan + shapely):
  PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe proto_label.py --n 20
Evidence class of every number it prints: MEASURED (this script, this dev box, CPU only).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("REFE_NUPLAN_DB_ROOT", "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
HERE = Path(__file__).resolve().parent
REFE = HERE.parents[1] / "refe"
sys.path.insert(0, str(REFE))
import navsim_dac as ND  # noqa: E402
import navtrain_scenarios as NS  # noqa: E402

# ---- the label spec under test (see SPEC_AUX_DRIVABLE_HEAD.md section 1) ---------------------------------
X0, X1, Y0, Y1, RES = -8.0, 72.0, -32.0, 32.0, 0.5       # rear-axle frame, x forward, y left
NX, NY = int(round((X1 - X0) / RES)), int(round((Y1 - Y0) / RES))          # 160 x 128
SDF_CLIP_M, SDF_UNIT_M = 12.7, 0.1                                          # int8: +-127 * 0.1 m
PAD_M = SDF_CLIP_M + 0.5                                                   # SDF is computed on a padded grid


def grid_centres(pad_m: float = 0.0):
    p = int(round(pad_m / RES))
    ix = np.arange(-p, NX + p)
    iy = np.arange(-p, NY + p)
    x = X0 + (ix + 0.5) * RES
    y = Y0 + (iy + 0.5) * RES
    return np.meshgrid(x, y, indexing="ij")                                  # [NX+2p, NY+2p]


def to_global(x, y, pose):
    x0, y0, h0 = pose
    c, s = math.cos(h0), math.sin(h0)
    return x0 + x * c - y * s, y0 + x * s + y * c


def polys_at(map_api, cx: float, cy: float, radius: float = ND.MAP_RADIUS):
    """`navsim_dac.drivable_polygons` verbatim, but from a bare centre point (no EgoState)."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    layers = [L.ROADBLOCK, L.ROADBLOCK_CONNECTOR, L.INTERSECTION, L.CARPARK_AREA]
    objs = map_api.get_proximal_map_objects(Point2D(cx, cy), radius, layers)
    return [o.polygon for layer in (L.ROADBLOCK, L.INTERSECTION, L.CARPARK_AREA) for o in objs[layer]]


def rasterize(polys, gx, gy) -> np.ndarray:
    """Point-in-union at every grid centre: the SAME test NAVSIM applies to each footprint corner."""
    import shapely
    inside = np.zeros(gx.shape, dtype=bool)
    fx, fy = gx.ravel(), gy.ravel()
    box = shapely.box(fx.min(), fy.min(), fx.max(), fy.max())
    flat = np.zeros(fx.shape, dtype=bool)
    for p in polys:
        if not shapely.intersects(box, p):
            continue
        flat |= shapely.contains_xy(p, fx, fy)
    return flat.reshape(gx.shape) | inside


def signed_distance(inside_pad: np.ndarray) -> np.ndarray:
    """Metres, + inside / - outside, boundary at 0 (half-cell corrected), on the padded raster."""
    from scipy.ndimage import distance_transform_edt as edt
    d_in = edt(inside_pad)            # inside cells: distance (cells) to the nearest outside cell
    d_out = edt(~inside_pad)
    sdf = np.where(inside_pad, d_in - 0.5, -(d_out - 0.5)) * RES
    return sdf


def build_label(polys, pose):
    """-> (mask [NX,NY] bool, sdf_int8 [NX,NY]) in the rear-axle frame, from map polygons + the ego pose."""
    px, py = grid_centres(PAD_M)
    gx, gy = to_global(px, py, pose)
    inside_pad = rasterize(polys, gx, gy)
    sdf = signed_distance(inside_pad)
    p = int(round(PAD_M / RES))
    sdf_c = np.clip(sdf[p:p + NX, p:p + NY], -SDF_CLIP_M, SDF_CLIP_M)
    return inside_pad[p:p + NX, p:p + NY].copy(), np.round(sdf_c / SDF_UNIT_M).astype(np.int8)


# ---- fidelity: raster verdict vs NAVSIM's exact verdict on the same corners --------------------------------
def corners_ego(poses8):
    """Footprint corners in the EGO (rear-axle) frame at 0.1 s over 4 s: [41, 4, 2] (same maths as corners_global)."""
    return ND.corners_global(ND.at_10hz(poses8), 0.0, 0.0, 0.0)


def raster_inside(mask, sdf_i8, pts):
    """Per-point verdict from the raster: nearest-cell mask and bilinear SDF > 0. pts [...,2] ego frame."""
    fx = (pts[..., 0] - X0) / RES - 0.5
    fy = (pts[..., 1] - Y0) / RES - 0.5
    in_ext = (fx >= 0) & (fx <= NX - 1) & (fy >= 0) & (fy <= NY - 1)
    ix = np.clip(np.rint(fx).astype(int), 0, NX - 1)
    iy = np.clip(np.rint(fy).astype(int), 0, NY - 1)
    nearest = mask[ix, iy]
    x0 = np.clip(np.floor(fx).astype(int), 0, NX - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, NY - 2)
    wx, wy = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
    s = sdf_i8.astype(np.float64) * SDF_UNIT_M
    bil = ((1 - wx) * (1 - wy) * s[x0, y0] + wx * (1 - wy) * s[x0 + 1, y0]
           + (1 - wx) * wy * s[x0, y0 + 1] + wx * wy * s[x0 + 1, y0 + 1])
    return nearest, bil > 0, in_ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--targets", default="E:/Projects/TanitAD/data/refe_navtrain/r0_shard0/targets_rank0.jsonl")
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()

    dbs = NS.index_dbs()
    rows = []
    with open(a.targets, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            if r["log_name"] in dbs:
                rows.append(r)
    by_log: dict = {}
    for r in rows:
        by_log.setdefault(r["log_name"], []).append(r)
    rng = random.Random(a.seed)
    logs = sorted(by_log)
    rng.shuffle(logs)
    picks = [rng.choice(by_log[lg]) for lg in logs[:a.n]]
    print(f"rows with a local DB: {len(rows)} in {len(by_log)} logs; sampling {len(picks)} (1 per log, seed {a.seed})")
    print(f"raster: x[{X0},{X1}) y[{Y0},{Y1}) res {RES} m -> {NX}x{NY} = {NX*NY} cells; SDF clip +-{SDF_CLIP_M} m, "
          f"int8 unit {SDF_UNIT_M} m; padded SDF grid {grid_centres(PAD_M)[0].shape}")

    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    res, masks, sdfs = [], [], []
    n_corner = {"pts": 0, "out_ext": 0, "nearest_dis": 0, "bil_dis": 0}
    ver = {"n": 0, "viol": 0, "nearest_dis": 0, "bil_dis": 0}
    bnd_dist_dis = []
    for k, r in enumerate(picks):
        lg, tok, step = r["log_name"], r["token"], int(r["step"])
        row = {"i": k, "log": lg, "token": tok}
        # ---------------- path B: bank row pose + map only
        t = time.perf_counter()
        map_name, _rows = NS._log_rows(dbs[lg])
        row["t_logrows_s"] = time.perf_counter() - t           # DB read for the map name (path B pays this once per LOG)
        t = time.perf_counter()
        map_api = get_maps_api(os.environ["NUPLAN_MAPS_ROOT"], NS.MAP_VERSION, map_name)
        row["t_mapapi_s"] = time.perf_counter() - t            # first call per location is the cold map load
        pose = tuple(r["origin_world"])                         # x, y, heading of the REAR AXLE (checked below)
        cx = pose[0] + ND.REAR_AXLE_TO_CENTER * math.cos(pose[2])
        cy = pose[1] + ND.REAR_AXLE_TO_CENTER * math.sin(pose[2])
        t = time.perf_counter()
        polys = polys_at(map_api, cx, cy)
        row["t_query_s"] = time.perf_counter() - t
        t = time.perf_counter()
        px, py = grid_centres(PAD_M)
        gx, gy = to_global(px, py, pose)
        inside_pad = rasterize(polys, gx, gy)
        row["t_raster_s"] = time.perf_counter() - t
        t = time.perf_counter()
        sdf = signed_distance(inside_pad)
        row["t_sdf_s"] = time.perf_counter() - t
        p = int(round(PAD_M / RES))
        mask = inside_pad[p:p + NX, p:p + NY].copy()
        sdf_i8 = np.round(np.clip(sdf[p:p + NX, p:p + NY], -SDF_CLIP_M, SDF_CLIP_M) / SDF_UNIT_M).astype(np.int8)
        row["t_B_warm_s"] = row["t_query_s"] + row["t_raster_s"] + row["t_sdf_s"]
        row.update(map=map_name, n_polys=len(polys), n_vertices=int(sum(len(p_.exterior.coords) for p_ in polys if hasattr(p_, "exterior"))),
                   drivable_frac=float(mask.mean()), pose=list(pose))
        masks.append(mask); sdfs.append(sdf_i8)

        # ---------------- path A: scenario (the labeller's route) -> must give the SAME mask
        t = time.perf_counter()
        sc = next(iter(NS.build_scenarios_for_log(dbs[lg], [tok])))
        row["t_A_scenario_s"] = time.perf_counter() - t
        ego = sc.get_ego_state_at_iteration(step)
        ra = ego.rear_axle
        row["origin_vs_rear_axle_m"] = float(math.hypot(ra.x - pose[0], ra.y - pose[1]))
        row["origin_vs_heading_rad"] = float(abs(((ra.heading - pose[2]) + math.pi) % (2 * math.pi) - math.pi))
        t = time.perf_counter()
        polys_a = ND.drivable_polygons(sc.map_api, ego)
        row["t_A_query_s"] = time.perf_counter() - t
        mask_a, _ = build_label(polys_a, (ra.x, ra.y, ra.heading))
        row["A_equals_B_cells_differing"] = int((mask_a != mask).sum())
        row["A_polys_equal_B_polys"] = bool(len(polys_a) == len(polys))

        # ---------------- fidelity vs NAVSIM's exact verdict on shifted copies of the rank-0 target
        poses20 = np.concatenate([np.asarray(r["traj"], float)], 0)
        for shift in (-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            q = poses20.copy(); q[:, 1] += shift
            p8 = ND.to_navsim(q)
            ce = corners_ego(p8)                                            # [41,4,2] ego frame
            cg = ND.corners_global(ND.at_10hz(p8), *pose)                   # global, same maths as the scorer's raw mode
            exact = ND.violations(polys, cg[None])[0]                       # NAVSIM's rule on these corners
            nearest, bil, in_ext = raster_inside(mask, sdf_i8, ce)
            # exact per-corner verdict (union of polygons)
            import shapely
            ins = np.zeros(cg.reshape(-1, 2).shape[0], bool)
            for pg in polys:
                ins |= shapely.contains_xy(pg, cg.reshape(-1, 2)[:, 0], cg.reshape(-1, 2)[:, 1])
            ins = ins.reshape(cg.shape[:-1])
            n_corner["pts"] += ins.size; n_corner["out_ext"] += int((~in_ext).sum())
            dn = nearest != ins; db = bil != ins
            n_corner["nearest_dis"] += int(dn.sum()); n_corner["bil_dis"] += int(db.sum())
            if db.any():                                                    # distance (m) of a bilinear-disagreeing corner to the true boundary
                bnd = shapely.union_all(polys).boundary
                for pt in cg[db][:50]:
                    bnd_dist_dis.append(float(shapely.distance(bnd, shapely.Point(*pt))))
            v_near = ~(nearest.sum(-1) == 4).all(-1) if True else None
            v_bil = ~(bil.sum(-1) == 4).all(-1)
            ver["n"] += 1; ver["viol"] += int(exact)
            ver["nearest_dis"] += int(bool(v_near) != bool(exact)); ver["bil_dis"] += int(bool(v_bil) != bool(exact))
        res.append(row)
        print(f"[{k:02d}] {map_name:<24} polys {row['n_polys']:3d} drv {row['drivable_frac']:.2f} | B: query {row['t_query_s']*1e3:6.1f} ms "
              f"raster {row['t_raster_s']*1e3:6.1f} ms sdf {row['t_sdf_s']*1e3:5.1f} ms (map api {row['t_mapapi_s']*1e3:7.1f} ms) | "
              f"A: scenario {row['t_A_scenario_s']*1e3:7.1f} ms | A==B cells differing {row['A_equals_B_cells_differing']}"
              f" pose-dev {row['origin_vs_rear_axle_m']:.4f} m", flush=True)

    # ------------------------------------------------ summary
    def stat(key):
        v = np.array([x[key] for x in res]) * 1e3
        return {"median_ms": float(np.median(v)), "p90_ms": float(np.percentile(v, 90)), "max_ms": float(v.max()), "mean_ms": float(v.mean())}
    # warm = skip the first sample per location for map api; use per-sample query+raster+sdf
    summ = {
        "n": len(res), "raster": dict(x0=X0, x1=X1, y0=Y0, y1=Y1, res_m=RES, nx=NX, ny=NY, sdf_clip_m=SDF_CLIP_M, sdf_unit_m=SDF_UNIT_M),
        "timing_ms": {k: stat(k) for k in ("t_query_s", "t_raster_s", "t_sdf_s", "t_B_warm_s", "t_mapapi_s", "t_logrows_s", "t_A_scenario_s", "t_A_query_s")},
        "locations": sorted({x["map"] for x in res}),
        "A_equals_B_all": bool(all(x["A_equals_B_cells_differing"] == 0 for x in res)),
        "max_origin_vs_rear_axle_m": max(x["origin_vs_rear_axle_m"] for x in res),
        "max_origin_vs_heading_rad": max(x["origin_vs_heading_rad"] for x in res),
        "corner_fidelity": {**n_corner, "nearest_agree_frac": 1 - n_corner["nearest_dis"] / n_corner["pts"],
                            "bilinear_sdf_agree_frac": 1 - n_corner["bil_dis"] / n_corner["pts"],
                            "bilinear_disagree_boundary_dist_m_p50_p90_max": (
                                [float(np.percentile(bnd_dist_dis, q)) for q in (50, 90, 100)] if bnd_dist_dis else None)},
        "verdict_fidelity": {**ver, "nearest_agree_frac": 1 - ver["nearest_dis"] / ver["n"], "bil_agree_frac": 1 - ver["bil_dis"] / ver["n"]},
        "sizes_bytes_per_sample": {},
    }
    outd = Path(a.out)
    packed = [np.packbits(m, axis=None) for m in masks]
    np.savez_compressed(outd / "labels_proto.npz", sdf_int8=np.stack(sdfs), mask_packed=np.stack(packed),
                        logs=np.array([x["log"] for x in res]), tokens=np.array([x["token"] for x in res]))
    sz = (outd / "labels_proto.npz").stat().st_size
    import zlib
    summ["sizes_bytes_per_sample"] = {
        "sdf_int8_raw": int(sdfs[0].nbytes), "mask_packed_raw": int(packed[0].nbytes),
        "sdf_int8_zlib_mean": float(np.mean([len(zlib.compress(s.tobytes(), 6)) for s in sdfs])),
        "npz_total_per_sample": sz / len(res)}
    (outd / "proto_label_results.json").write_text(json.dumps({"summary": summ, "rows": res}, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, indent=1, default=str))

    # contact sheet (6 samples): mask + SDF + GT trajectory
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 6, figsize=(18, 6))
        for j in range(min(6, len(res))):
            axs[0, j].imshow(masks[j].T[::-1], extent=[X0, X1, Y0, Y1], cmap="gray", vmin=0, vmax=1)
            axs[1, j].imshow(sdfs[j].T[::-1] * SDF_UNIT_M, extent=[X0, X1, Y0, Y1], cmap="RdBu", vmin=-6, vmax=6)
            tr = np.asarray(picks[j]["traj"], float)
            for ax in axs[:, j]:
                ax.plot(tr[:, 0], tr[:, 1], "g.-", ms=2, lw=0.8); ax.plot([0], [0], "r^", ms=5)
            axs[0, j].set_title(f"{res[j]['map'][:12]} polys {res[j]['n_polys']}", fontsize=8)
        fig.suptitle("drivable mask (top) and SDF m (bottom), rear-axle frame, x fwd; green = rank-0 target")
        fig.savefig(outd / "labels_proto_contact.png", dpi=70, bbox_inches="tight")
    except Exception as e:  # plotting must never fail the measurement
        print("contact sheet skipped:", type(e).__name__, e)


if __name__ == "__main__":
    main()
