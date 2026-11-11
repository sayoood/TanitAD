#!/usr/bin/env python3
"""TF-2 (PREREG_TF2.md, sha256 65189b98...): margin shaping (L2b), DAC-certified fallback (L3) and the 20-pose
submission probe (P-S20), on the 1,018 held-out rank-0 NAVTRAIN teacher targets of D1 / P5 / P5-S / E2 / TF-1.
Not navtest. CPU only.

One NAVSIM PDMScorer call per (log, token) (vendored navsim @ 3e8291b) on
  [PDM-Closed, T, Tc20s(banked), Tc20s_re, M, Mw, H (8 poses @0.5 s, LQR), T_sub20 (20 poses @0.2 s, LQR),
   T_transparent, M_transparent (as executed, E2's builder)].
Per arm: the PDMS sub-scores (TF-1's formula) + anatomy from the scorer's own state: min signed corner margin to the
union of the polygons the scorer's DAC reads, first violating time, worst time/corner, worst-step centre in an
INTERSECTION polygon, collision and TTC time.
Usage: python tf2_score.py --w <scratch dir holding refe/ eval/ p5s/> --e2score <dir of e2_score.py> --out <jsonl>
       --shard i/N [--limit n] [--only log:token,...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("REFE_SIM_HZ", "10")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

DB_ROOT = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
M_STAR, SLOPE, CAP, ITER_SHAPE, MIN_DOT, MIN_SPAN = 0.60, 0.10, 1.0, 3, 0.5, 0.2      # PREREG_TF2 s4, frozen
TRACKED = ("T_tracked", "Tc20s_tracked", "Tc20s_re_tracked", "M_tracked", "Mw_tracked", "H_tracked", "T_sub20_tracked")
TRANSP = ("T_transparent", "M_transparent")
LEFT = (0, 1)        # BBCoordsIndex FRONT_LEFT, REAR_LEFT; RIGHT = REAR_RIGHT 2, FRONT_RIGHT 3


def dac_union(dam):
    import shapely
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    idx = dam.get_indices_of_map_type([L.ROADBLOCK, L.INTERSECTION, L.DRIVABLE_AREA, L.CARPARK_AREA])
    inter = dam.get_indices_of_map_type([L.INTERSECTION])
    U = shapely.union_all([dam._geometries[i] for i in idx])
    I = shapely.union_all([dam._geometries[i] for i in inter]) if len(inter) else None
    return U, I


def signed(U, pts):
    """[N, 2] map-frame points -> signed distance to the boundary of U (> 0 inside)."""
    import shapely
    g = shapely.points(pts[:, 0], pts[:, 1])
    d = shapely.distance(U.boundary, g)
    return np.where(shapely.contains(U, g), d, -d)


def dilate(p):
    """PREREG s4: triangle dilation (slope SLOPE per 0.2 s index) of a non-negative profile p[0..20], then the anchor
    cap SLOPE * i (the origin at t = 0 cannot move)."""
    i = np.arange(len(p))
    out = np.max(p[None, :] - SLOPE * np.abs(i[:, None] - i[None, :]), axis=1)
    return np.minimum(np.maximum(out, 0.0), SLOPE * i)


def apply_shift(t20, a):
    """Shift the 20 label poses laterally by a[1..20] along each ORIGINAL pose's left normal; heading += the slope of
    the shift on the original arc length (central, one-sided at i = 20; unchanged where the span < MIN_SPAN)."""
    q = t20.copy()
    h = t20[:, 2]
    n = np.stack([-np.sin(h), np.cos(h)], -1)
    q[:, :2] = t20[:, :2] + a[1:, None] * n
    xy = np.vstack([np.zeros((1, 2)), t20[:, :2]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    for i in range(1, 21):
        lo, hi = i - 1, min(i + 1, 20)
        span = s[hi] - s[lo]
        if span >= MIN_SPAN:
            q[i - 1, 2] = h[i - 1] + math.atan2(a[hi] - a[lo], span)
    return q


def shape(ND, t20, ego, U, sign=1.0):
    """PREREG_TF2 s4 margin shaping. Returns (shaped 20 poses, final shift profile a[0..20])."""
    from shapely.ops import nearest_points
    import shapely
    ra = ego.rear_axle
    a = np.zeros(21)
    B = U.boundary
    for _ in range(ITER_SHAPE):
        q = apply_shift(t20, a)
        cr = ND.corners_global(q, ra.x, ra.y, ra.heading)                      # [20, 4, 2]
        d = signed(U, cr.reshape(-1, 2)).reshape(20, 4)
        prof = np.zeros(21)
        for i in range(20):
            k = int(np.argmin(d[i]))
            delta = max(0.0, M_STAR - float(d[i, k]))
            if delta <= 0.0:
                continue
            c = cr[i, k]
            npnt = nearest_points(B, shapely.Point(c[0], c[1]))[0]
            v = np.array([npnt.x - c[0], npnt.y - c[1]])
            nv = float(np.linalg.norm(v))
            hq = q[i, 2]
            nrm = np.array([-math.sin(hq), math.cos(hq)])
            if nv < 1e-9:
                sg, dot = (-1.0 if k in LEFT else 1.0), 1.0
            else:
                g = (-v / nv) if d[i, k] >= 0 else (v / nv)                       # toward the interior, map frame
                ch, sh = math.cos(-ra.heading), math.sin(-ra.heading)
                ge = np.array([g[0] * ch - g[1] * sh, g[0] * sh + g[1] * ch])    # ego frame
                dot = float(ge @ nrm)
                sg = 1.0 if dot > 1e-6 else (-1.0 if dot < -1e-6 else (-1.0 if k in LEFT else 1.0))
            prof[i + 1] = sg * min(delta / max(abs(dot), MIN_DOT), CAP)
        a = a + dilate(np.maximum(prof, 0.0)) - dilate(np.maximum(-prof, 0.0))
    a = np.clip(a, -CAP, CAP)
    if sign < 0:
        a = -a
    return apply_shift(t20, a), a


def score(parts, plans, executed, U, I):
    """TF-1's scorer call, generalised to (poses, sampling) plans, plus anatomy from the scorer's own state."""
    import navsim_pdm_targets as PT
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (MultiMetricIndex,
                                                                                 WeightedMetricIndex)
    import shapely
    S = PT._setup()
    pdm_traj, ego, obs, centerline, lanes, dam = parts
    inp = [get_trajectory_as_array(pdm_traj, S["samp"], ego.time_point)]
    for poses, samp in plans:
        it = transform_trajectory(Trajectory(np.asarray(poses, np.float32), samp), ego)
        inp.append(get_trajectory_as_array(it, S["samp"], ego.time_point))
    sim = S["sim"].simulate_proposals(np.stack(inp, 0), ego)
    if executed:
        sim = np.concatenate([sim, np.stack(executed, 0)], 0)
    sc = S["scorer"]
    sc.score_proposals(sim, obs, centerline, lanes, dam)
    mult = sc._multi_metrics.prod(axis=0)
    raw = sc._progress_raw * mult
    ttc = np.asarray(sc._weighted_metrics[WeightedMetricIndex.TTC])
    com = np.asarray(sc._weighted_metrics[WeightedMetricIndex.COMFORTABLE])
    ddc = np.asarray(sc._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION])
    coords = np.asarray(sc._ego_coords)                                         # [P, 41, 5, 2]
    out = []
    for j in range(1, sim.shape[0]):
        m = max(raw[0], raw[j])
        ep = raw[j] / m if m > 5.0 else (1.0 if mult[j] != 0 else 0.0)
        pdms = mult[j] * (5 * ep + 5 * ttc[j] + 2 * com[j]) / 12.0
        cr = coords[j, :, :4]                                                   # [41, 4, 2]
        d = signed(U, cr.reshape(-1, 2)).reshape(41, 4)
        dmin = d.min(1)
        viol = np.where(dmin < 0)[0]
        w = int(np.argmin(dmin))
        ctr = coords[j, w, 4]
        out.append({"nc": float(sc._multi_metrics[MultiMetricIndex.NO_COLLISION][j]),
                    "dac": float(sc._multi_metrics[MultiMetricIndex.DRIVABLE_AREA][j]),
                    "ddc": float(ddc[j]), "mult": float(mult[j]), "ep": float(ep), "ttc": float(ttc[j]),
                    "c": float(com[j]), "pdms": float(pdms),
                    "an": {"min_margin": float(dmin.min()), "first_viol_t": (round(0.1 * int(viol[0]), 1) if len(viol) else None),
                           "worst_t": round(0.1 * w, 1), "worst_corner": int(np.argmin(d[w])),
                           "worst_in_intersection": (bool(shapely.contains_xy(I, ctr[0], ctr[1])) if I is not None else False),
                           "coll_t": (None if not np.isfinite(sc._collision_time_idcs[j]) else round(0.1 * float(sc._collision_time_idcs[j]), 1)),
                           "ttc_t": (None if not np.isfinite(sc._ttc_time_idcs[j]) else round(0.1 * float(sc._ttc_time_idcs[j]), 1))}})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", required=True)
    ap.add_argument("--e2score", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--resume", action="store_true", help="skip rows already in --out (added after shard 2 was "
                                                           "stopped at 69 rows to keep commit memory above 10 GB)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.w, "refe"))
    sys.path.insert(0, os.path.join(a.w, "eval"))
    sys.path.insert(0, a.e2score)
    import navsim_dac as ND
    import navtrain_scenarios as NS
    import navsim_pdm_targets as PT
    from navsim_pdm.pdm_array_representation import ego_state_to_state_array
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from e2_score import transparent_states, human_rel                          # E2's builders, unchanged
    from p5_target_labels import iterate20                                      # P5-S's derivation, unchanged
    zc = np.load(os.path.join(a.w, "p5s", "p5s_cubic.npz"))
    lab = {}
    for i, (lg, tok) in enumerate(zip(zc["logs"].tolist(), zc["tokens"].tolist())):
        lab[(lg, tok)] = (zc["labels20_t20_tc20_tw"][i, 0], zc["labels20_t20_tc20_tw"][i, 1])
    keys = sorted(lab)
    if a.only:
        want = {tuple(x.split(":")) for x in a.only.split(",")}
        keys = [k for k in keys if k in want]
    si, sn = (int(x) for x in a.shard.split("/"))
    done = set()
    if a.resume and os.path.exists(a.out):                  # rows are deterministic: skip (log, token) already written
        for ln in open(a.out, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                done.add((r["log"], r["token"]))
    by: dict = {}
    for lg, tok in keys[si::sn]:
        if (lg, tok) not in done:
            by.setdefault(lg, []).append(tok)
    dbs = NS.index_dbs(db_root=DB_ROOT)
    s8, s20 = TrajectorySampling(num_poses=8, interval_length=0.5), TrajectorySampling(num_poses=20, interval_length=0.2)
    n_done, t0 = 0, time.time()
    with open(a.out, "a", encoding="utf-8") as fo:
        for lg in sorted(by):
            toks = by[lg]
            if not os.path.exists(dbs.get(lg, "")):
                for t in toks:
                    fo.write(json.dumps({"log": lg, "token": t, "error": "no local DB"}) + "\n")
                continue
            for sc in NS.build_scenarios_for_log(dbs[lg], toks):
                tok = sc._initial_lidar_token
                rec = {"log": lg, "token": tok}
                try:
                    t20, tc20s = (np.asarray(x, np.float64) for x in lab[(lg, tok)])
                    parts = PT.metric_cache_parts(sc)
                    ego = parts[1]
                    s0 = ego_state_to_state_array(ego)
                    U, I = dac_union(parts[5])
                    ctl = {"knot_exact": True, "knot_checks": 0}
                    tc_re = iterate20(ND, t20, ND.to_navsim(t20), ego, ctl, None, "cubic")[0]
                    qM, aM = shape(ND, t20, ego, U, +1.0)
                    qW, aW = shape(ND, t20, ego, U, -1.0)
                    lM = iterate20(ND, qM, ND.to_navsim(qM), ego, ctl, None, "cubic")[0]
                    lW = iterate20(ND, qW, ND.to_navsim(qW), ego, ctl, None, "cubic")[0]
                    h8 = human_rel(sc, ego)[4::5][:8]
                    plans = [(ND.to_navsim(t20), s8), (ND.to_navsim(tc20s), s8), (ND.to_navsim(tc_re), s8),
                             (ND.to_navsim(lM), s8), (ND.to_navsim(lW), s8), (h8, s8), (t20, s20)]
                    res = score(parts, plans, [transparent_states(t20, ego, s0), transparent_states(qM, ego, s0)], U, I)
                    for nm, r in zip(TRACKED + TRANSP, res):
                        rec[nm] = r
                    rec["info"] = {"v0": float(ego.dynamic_car_state.speed), "turn_deg": float(abs(math.degrees(t20[-1, 2]))),
                                   "path_m": float(np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), t20[:, :2]]), axis=0), axis=1).sum()),
                                   "c2_label_maxdev_m": float(np.abs(tc_re - tc20s).max()),
                                   "c3_unshifted": bool(not np.any(aM)), "c3_label_eq_re": bool(np.array_equal(lM, tc_re)),
                                   "shift_max_m": float(np.abs(aM).max()), "shift_n_poses": int(np.count_nonzero(aM[1:])),
                                   "knot_exact": bool(ctl["knot_exact"]), "knot_maxdev": float(ctl.get("knot_maxdev", 0.0))}
                except Exception as e:                                                # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                n_done += 1
                if a.limit and n_done >= a.limit:
                    print("ZZTF2_SHARD_DONE", a.shard, n_done, round(time.time() - t0, 1), flush=True)
                    return 0
    print("ZZTF2_SHARD_DONE", a.shard, n_done, round(time.time() - t0, 1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
