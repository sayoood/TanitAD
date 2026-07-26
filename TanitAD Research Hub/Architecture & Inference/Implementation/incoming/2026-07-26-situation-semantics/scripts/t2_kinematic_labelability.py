"""T2 — what can lane change / roundabout / intersection be labelled FROM, on PhysicalAI-AV?

CPU only, dev box, read-only over `C:/Users/Admin/tanitad-data/physicalai/`. No pod touched.
No new download. Parity untouched — nothing here re-selects training episodes.

The question this answers is NOT "build me a labeller". It is the narrower, decisive one:

  1. What is the ACTUAL feature coverage of the corpus? (`obstacle.offline` = 97.44 % is
     INHERITED everywhere in the program and has never been re-derived. It is here.)
  2. The junction stratum in use (`corridor.JUNCTION_DEG = 10 deg` of net heading change over
     2 s) is a KINEMATIC SIGNATURE, and `corridor.py` explicitly forbids renaming it
     "intersection". **How wrong would that rename be?** That is measurable: a junction turn and
     a motorway bend produce the same heading change over very different arc lengths, so the
     TURN RADIUS `R = ds / |dpsi|` separates them. This script measures the conflation rate.
  3. Do lane change and roundabout have separable kinematic signatures at all, and at what rate
     do candidate windows occur? (Rates, not validated labels — stated as such.)

Two independent curvature estimates are computed on purpose (the "absence found at ONE location
is not absence" rule applied to a positive claim): one from the quaternion heading series and one
from `egomotion`'s own native `curvature` column. If they disagree the finding is not admissible.

usage:  python t2_kinematic_labelability.py --root <physicalai root> --out <artifacts dir>
"""
from __future__ import annotations

import argparse
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HZ = 10.0                 # analysis grid; egomotion is stored at ~200 Hz
WIN_2S = int(2 * HZ)
WIN_4S = int(4 * HZ)
JUNCTION_DEG = 10.0       # taniteval.corridor.JUNCTION_DEG — imported value, not re-chosen
LANE_M = 2.5              # candidate lateral displacement for a lane change (< one lane width)


def yaw_from_quat(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def unwrap_deg(psi_rad):
    return np.degrees(np.unwrap(psi_rad))


CLIP_S = 20.0             # the PhysicalAI-AV clip length


def per_clip(df):
    """Resample one clip's egomotion to 10 Hz and derive the kinematic quantities.

    🔴 **THE TRAP THIS FUNCTION EXISTS TO AVOID** (MEASURED 2026-07-26, 6 clips of chunk_0036):
    `egomotion` and `obstacle.offline` do NOT cover the same interval and do NOT start at the
    same instant. On the clips measured, `egomotion` spans **~140 s** and starts at a slightly
    NEGATIVE timestamp (~-0.196 s), while `obstacle.offline` spans **~20 s** starting at ~+0.05 s
    — the clip. Both are microseconds in the SAME clip-relative frame, so the correct handling is
    to use the RAW timestamps for both and never to re-zero each series on its own first sample.
    Re-zeroing independently introduces a ~0.2 s offset (up to 3.7 s on one clip measured), and
    taking the egomotion span as "the clip" analyses ~7x more driving than the corpus's clip and
    silently scores 6/7 of it as "no agents present".
    """
    t = df["timestamp"].to_numpy().astype(np.float64) / 1e6          # us -> s, RAW origin kept
    hi = min(CLIP_S, float(t.max()))
    if hi <= 0:
        return None
    n = int(hi * HZ) + 1
    if n < WIN_4S + 2:
        return None
    tg = np.arange(n) / HZ                                            # 0 .. 20 s, clip frame
    g = lambda c: np.interp(tg, t, df[c].to_numpy().astype(np.float64))    # noqa: E731
    x, y = g("x"), g("y")
    psi = unwrap_deg(np.interp(tg, t, np.unwrap(yaw_from_quat(
        df["qx"].to_numpy(), df["qy"].to_numpy(),
        df["qz"].to_numpy(), df["qw"].to_numpy()))))
    v = np.sqrt(g("vx") ** 2 + g("vy") ** 2 + g("vz") ** 2)
    kappa_native = g("curvature")
    ds = np.concatenate([[0.0], np.hypot(np.diff(x), np.diff(y))])
    s = np.cumsum(ds)
    return dict(t=tg, x=x, y=y, psi=psi, v=v, s=s, kappa_native=kappa_native, n=n)


def windows(c):
    """Per-frame window features. A window starts at i and looks forward 2 s / 4 s."""
    n = c["n"]
    hi2 = np.minimum(np.arange(n) + WIN_2S, n - 1)
    hi4 = np.minimum(np.arange(n) + WIN_4S, n - 1)
    ok = (np.arange(n) + WIN_4S) < n
    hd2 = c["psi"][hi2] - c["psi"]                       # net heading change over 2 s, deg
    hd4 = c["psi"][hi4] - c["psi"]
    ds2 = c["s"][hi2] - c["s"]                           # arc length over 2 s, m
    ds4 = c["s"][hi4] - c["s"]
    # turn radius from the 2 s window: R = ds / |dpsi_rad|. Large R = a road bend.
    with np.errstate(divide="ignore", invalid="ignore"):
        R2 = ds2 / np.maximum(np.abs(np.radians(hd2)), 1e-9)
        kappa_geom = np.abs(np.radians(hd2)) / np.maximum(ds2, 1e-6)      # 1/m
    # LANE-CHANGE proxy: lateral offset from the straight continuation of the heading at i,
    # over 4 s, WITH the net heading change small (a lane change returns to the road direction).
    dx, dy = c["x"][hi4] - c["x"], c["y"][hi4] - c["y"]
    ps = np.radians(c["psi"])
    lat4 = -np.sin(ps) * dx + np.cos(ps) * dy            # + = left of the initial heading
    return dict(ok=ok, hd2=hd2, hd4=hd4, ds2=ds2, ds4=ds4, R2=R2, kappa_geom=kappa_geom,
                lat4=lat4, v=c["v"], kappa_native=np.abs(c["kappa_native"]))


VEH = {"automobile", "heavy_truck", "bus", "other_vehicle", "trailer"}


def cross_traffic(zf_obs, nm, c):
    """MAP-FREE INTERSECTION SIGNAL: is a VEHICLE ahead of us oriented ACROSS our path?

    `obstacle.offline` boxes are `reference_frame = "rig"` on 100 % of rows (MEASURED), so the
    agent's yaw in the ego frame is read straight off its quaternion — no extrinsics, no map, no
    lane graph. A vehicle whose heading is 50-130 deg from ours, ahead of us and within 40 m, is
    the classic crossing-conflict geometry of an intersection.

    ⚠️ It is a SIGNAL, not a label: a car parked perpendicular in a bay, a driveway, and a
    car-park aisle all enter this set. That is stated, not tuned away.
    """
    df = pq.read_table(io.BytesIO(zf_obs.read(nm))).to_pandas()
    if not len(df):
        return None
    t = df["timestamp_us"].to_numpy().astype(np.float64) / 1e6   # RAW origin — see per_clip()
    qz, qw = df["orientation_z"].to_numpy(), df["orientation_w"].to_numpy()
    qx, qy = df["orientation_x"].to_numpy(), df["orientation_y"].to_numpy()
    yaw = np.abs(np.degrees(yaw_from_quat(qx, qy, qz, qw)))
    yaw = np.minimum(yaw, 180.0 - yaw)                      # fold to [0, 90]: axis, not direction
    rng = np.hypot(df["center_x"].to_numpy(), df["center_y"].to_numpy())
    isveh = df["label_class"].isin(VEH).to_numpy()
    hit = isveh & (yaw >= 50.0) & (rng <= 40.0) & (df["center_x"].to_numpy() > 0)
    # COVERAGE mask: only grid frames inside the obstacle track's own time span are DECIDED.
    # Outside it the answer is UNKNOWN, never "no agents" — that conflation is the trap.
    grid = np.arange(c["n"]) / HZ
    cov = (grid >= t.min()) & (grid <= t.max())
    m = np.zeros(c["n"], bool)
    if hit.any():
        ht = np.unique(np.round(t[hit] * HZ).astype(int))
        ht = ht[(ht >= 0) & (ht < c["n"])]
        m[ht] = True
    out = np.where(cov, m.astype(float), np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chunks", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---------------------------------------------------------- 1. FEATURE COVERAGE (the
    # INHERITED 97.44 % figure, re-derived from the corpus' own manifest)
    fp = pd.read_parquet(os.path.join(args.root, "metadata", "feature_presence.parquet"))
    cov = {c: {"frac": float(fp[c].mean()), "n": int(fp[c].sum())} for c in fp.columns}
    ing = ["camera_front_wide_120fov", "egomotion", "camera_intrinsics", "sensor_extrinsics"]
    coverage = {
        "source": "metadata/feature_presence.parquet (the corpus' OWN manifest)",
        "n_clips_total": int(len(fp)), "n_features": int(fp.shape[1]),
        "per_feature": cov,
        "read_by_our_ingest": {k: cov[k] for k in ing},
        "identical_coverage_groups": {},
    }
    byfrac = {}
    for k, vv in cov.items():
        byfrac.setdefault(round(vv["frac"], 6), []).append(k)
    coverage["identical_coverage_groups"] = {str(k): sorted(v) for k, v in
                                             sorted(byfrac.items(), reverse=True)}
    json.dump(coverage, open(os.path.join(args.out, "t2_feature_coverage.json"), "w"), indent=2)
    print(f"[cov] {len(fp)} clips x {fp.shape[1]} features; "
          f"obstacle.offline = {cov['obstacle.offline']['frac']*100:.4f} % "
          f"({cov['obstacle.offline']['n']})", flush=True)

    # ---------------------------------------------------------- 2. KINEMATICS
    obs_dir = os.path.join(args.root, "labels", "obstacle.offline")
    ego_dir = os.path.join(args.root, "labels", "egomotion")
    chunks = sorted(f.split(".")[-2] for f in os.listdir(obs_dir) if f.endswith(".zip"))
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    acc = {k: [] for k in ("hd2", "hd4", "ds2", "R2", "kappa_geom", "kappa_native",
                           "lat4", "v", "xtraf")}
    n_clips = 0
    n_clips_with_obs = 0
    per_clip_roundabout = []
    for ci, ch in enumerate(chunks):
        zp = os.path.join(ego_dir, f"egomotion.{ch}.zip")
        op = os.path.join(obs_dir, f"obstacle.offline.{ch}.zip")
        if not os.path.exists(zp):
            continue
        zf_obs = zipfile.ZipFile(op) if os.path.exists(op) else None
        obs_names = set(zf_obs.namelist()) if zf_obs else set()
        with zipfile.ZipFile(zp) as zf:
            for nm in zf.namelist():
                if not nm.endswith(".parquet"):
                    continue
                df = pq.read_table(io.BytesIO(zf.read(nm))).to_pandas()
                c = per_clip(df)
                if c is None:
                    continue
                w = windows(c)
                m = w["ok"]
                uuid = nm.split(".")[0]
                onm = f"{uuid}.obstacle.offline.parquet"
                xt = None
                if zf_obs is not None and onm in obs_names:
                    xt = cross_traffic(zf_obs, onm, c)
                    if xt is not None:
                        n_clips_with_obs += 1
                w["xtraf"] = xt if xt is not None else np.full(c["n"], np.nan)
                for k in acc:
                    acc[k].append(w[k][m] if w[k].shape == m.shape else w[k][:len(m)][m])
                # ROUNDABOUT candidate: sustained same-sign heading sweep with a small radius.
                # Measured per CLIP, because a roundabout is a clip-level event.
                psi = c["psi"]
                sweep = 0.0
                best = 0.0
                for L in (int(6 * HZ), int(10 * HZ), int(15 * HZ)):
                    if L >= c["n"]:
                        continue
                    d = psi[L:] - psi[:-L]
                    arc = c["s"][L:] - c["s"][:-L]
                    r = arc / np.maximum(np.abs(np.radians(d)), 1e-9)
                    cand = np.abs(d) * (r < 30.0)
                    if cand.size:
                        best = max(best, float(cand.max()))
                per_clip_roundabout.append(best)
                n_clips += 1
        if zf_obs is not None:
            zf_obs.close()
        print(f"[kin] chunk {ch} ({ci+1}/{len(chunks)}) clips so far {n_clips} "
              f"(with obstacle.offline {n_clips_with_obs})", flush=True)
    A = {k: np.concatenate(v) for k, v in acc.items()}
    N = len(A["hd2"])

    junc = np.abs(A["hd2"]) >= JUNCTION_DEG
    R = A["R2"][junc]
    v_j = A["v"][junc]
    bands = {"R_lt_25m (junction-scale turn)": float((R < 25).mean()),
             "R_25_100m (ambiguous)": float(((R >= 25) & (R < 100)).mean()),
             "R_100_300m (road curve)": float(((R >= 100) & (R < 300)).mean()),
             "R_ge_300m (near-straight / noise)": float((R >= 300).mean())}

    # cross-check the two independent curvature estimates on the SAME windows
    kg, kn = A["kappa_geom"], A["kappa_native"]
    fin = np.isfinite(kg) & np.isfinite(kn) & (A["ds2"] > 1.0)
    corr = float(np.corrcoef(kg[fin], kn[fin])[0, 1]) if fin.sum() > 100 else float("nan")

    lane_cand = (np.abs(A["lat4"]) >= LANE_M) & (np.abs(A["hd4"]) < JUNCTION_DEG) & (A["v"] >= 5.0)
    rab = np.asarray(per_clip_roundabout)

    # ---- lane-change SENSITIVITY: how much of the candidate set is just road curvature?
    lane_sens = {}
    for hdmax in (10.0, 5.0, 2.0, 1.0):
        for latmin in (2.5, 3.5):
            k = f"|hd4|<{hdmax}deg & |lat4|>={latmin}m"
            mm = ((np.abs(A["lat4"]) >= latmin) & (np.abs(A["hd4"]) < hdmax) & (A["v"] >= 5.0))
            lane_sens[k] = {"rate": float(mm.mean()), "n": int(mm.sum())}

    # ---- MAP-FREE INTERSECTION SIGNAL and its agreement with the kinematic stratum
    xt = A["xtraf"]
    have = np.isfinite(xt)
    xtb = xt[have] > 0.5
    xtraffic = {
        "definition": "at least one VEHICLE-class obstacle.offline box AHEAD (center_x>0), within "
                      "40 m, whose heading AXIS is >= 50 deg from the ego's (rig frame)",
        "n_windows_with_obstacle_offline": int(have.sum()),
        "frac_of_windows_covered": float(have.mean()),
        "rate": float(xtb.mean()) if have.any() else None,
        "AGREEMENT_with_kinematic_junction_stratum": None,
        "status": "SIGNAL, not a label — a perpendicular parked car, a driveway and a car-park "
                  "aisle all enter this set. It is INDEPENDENT of the ego's own trajectory, "
                  "which is what makes the agreement below worth reading.",
    }
    if have.any():
        jj = junc[have]
        p_j_given_x = float(jj[xtb].mean()) if xtb.any() else None
        p_j_given_nx = float(jj[~xtb].mean()) if (~xtb).any() else None
        xtraffic["AGREEMENT_with_kinematic_junction_stratum"] = {
            "P(kinematic_junction | cross_traffic)": p_j_given_x,
            "P(kinematic_junction | no_cross_traffic)": p_j_given_nx,
            "lift": (p_j_given_x / p_j_given_nx) if (p_j_given_x and p_j_given_nx) else None,
            "P(cross_traffic | kinematic_junction)": float(xtb[jj].mean()) if jj.any() else None,
            "P(cross_traffic | not kinematic_junction)":
                float(xtb[~jj].mean()) if (~jj).any() else None,
        }

    res = {
        "n_chunks": len(chunks), "n_clips": n_clips,
        "n_clips_with_obstacle_offline": n_clips_with_obs, "n_windows_10hz": int(N),
        "grid_hz": HZ, "junction_deg_imported_from": "taniteval.corridor.JUNCTION_DEG",
        "junction_deg": JUNCTION_DEG,
        "JUNCTION_STRATUM": {
            "definition": "|net heading change over the FIRST 2 s| >= 10 deg (corridor.junction_mask)",
            "rate": float(junc.mean()), "n_positive": int(junc.sum()),
            "speed_mean_mps_inside": float(v_j.mean()) if junc.any() else None,
            "speed_mean_mps_outside": float(A["v"][~junc].mean()),
            "turn_radius_bands_within_the_stratum": bands,
            "CONFLATION_RATE_road_curve_not_junction": float((R >= 100).mean()),
            "R_quantiles_m": {q: float(np.quantile(R, float(q)))
                              for q in ("0.05", "0.25", "0.5", "0.75", "0.95")} if junc.any()
            else None,
        },
        "CURVATURE_CROSS_CHECK": {
            "why": "two independent estimates of the same quantity — quaternion-heading geometry "
                   "vs egomotion's OWN `curvature` column — so a curvature-based claim does not "
                   "rest on one derivation",
            "pearson_r": corr, "n": int(fin.sum()),
            "kappa_geom_median": float(np.median(kg[fin])),
            "kappa_native_median": float(np.median(kn[fin])),
        },
        "LANE_CHANGE_CANDIDATE": {
            "definition": f"|lateral offset from the straight continuation over 4 s| >= {LANE_M} m "
                          f"AND |net heading change over 4 s| < {JUNCTION_DEG} deg AND v >= 5 m/s",
            "rate": float(lane_cand.mean()), "n_positive": int(lane_cand.sum()),
            "lat4_abs_quantiles_m": {q: float(np.quantile(np.abs(A["lat4"]), float(q)))
                                     for q in ("0.5", "0.9", "0.99", "0.999")},
            "status": "CANDIDATE RATE, NOT A VALIDATED LABEL — there is no lane graph to "
                      "confirm a boundary was crossed; a wide bend and a within-lane swerve "
                      "both enter this set",
            "sensitivity_how_much_is_road_curvature": lane_sens,
        },
        "CROSS_TRAFFIC_MAP_FREE_INTERSECTION_SIGNAL": xtraffic,
        "ROUNDABOUT_CANDIDATE": {
            "definition": "max sustained same-sign heading sweep over 6/10/15 s with turn "
                          "radius < 30 m, per CLIP",
            "clip_rate_ge_180deg": float((rab >= 180).mean()),
            "clip_rate_ge_270deg": float((rab >= 270).mean()),
            "clip_rate_ge_360deg": float((rab >= 360).mean()),
            "sweep_quantiles_deg": {q: float(np.quantile(rab, float(q)))
                                    for q in ("0.5", "0.9", "0.99", "1.0")},
            "status": "CANDIDATE RATE, NOT A VALIDATED LABEL — a U-turn, a tight corner and a "
                      "car park loop all enter this set; nothing in the corpus can tell them "
                      "apart from a roundabout",
        },
        "speed_quantiles_mps": {q: float(np.quantile(A["v"], float(q)))
                                for q in ("0.05", "0.25", "0.5", "0.75", "0.95")},
    }
    json.dump(res, open(os.path.join(args.out, "t2_kinematics.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "speed_quantiles_mps"},
                     indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()
