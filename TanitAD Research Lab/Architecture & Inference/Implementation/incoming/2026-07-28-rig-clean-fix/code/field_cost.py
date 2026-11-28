"""WHAT THE RIG-CLEAN FRAME COSTS IN FIELD — measured on real agent positions.

Shrinking the vertical field to what BOTH rigs observe throws away real pixels
for the majority rig. This prices that, using the corpus' own `obstacle.offline`
3-D agent tracks projected into the CYLINDRICAL output frame with each clip's
OWN intrinsics + extrinsics (mandatory: the rig split makes a global (cx, cy)
~215 px wrong for rig B).

Two prices, because they answer different questions:

  P1 AGENTS. For every agent sample, is its cuboid visible in the parent frame
     (256x640, 120 deg) but NOT in the candidate? Stratified by range, class, and
     rig. An agent counts as visible if ANY of its 8 cuboid corners lands inside.
  P2 GROUND. The nearest ground distance the frame still shows, straight ahead —
     exact, per clip, from the real extrinsics (so the mount pitch and the
     per-clip camera height are in it, not a constant; class C28).

Cylindrical inverse projection, from `calib.cylindrical_rays`: a camera-frame ray
(x, y, z) lands at col = (W-1)/2 + f*atan2(x, z) and row = (H-1)/2 + f*y/hypot(x, z).

🔒 No clip UUID is written to the artifact.
"""
from __future__ import annotations
import argparse
import glob
import io
import json
import math
import os
import sys
import time
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)
from crux import (CI, SE, calib_clips, clip_rig, in_frame, poly_r,  # noqa: E402
                  project, q2R)

DR = os.environ.get("TANITAD_PAI_ROOT", r"C:\Users\Admin\tanitad-data\physicalai")
FW = "camera_front_wide_120fov"
PARENT_H, PARENT_W, F_REF = 256, 640, 305.5774907364391
RIG_BOUNDARY_CY = 650.872          # measured, from the cy bimodality (n=3,000)


def cyl_rowcol(Pc, h=PARENT_H, w=PARENT_W, f=F_REF):
    """camera-frame points [N,3] -> (row, col) in an h x w cylindrical frame."""
    x, y, z = Pc[:, 0], Pc[:, 1], Pc[:, 2]
    rho = np.hypot(x, z)
    phi = np.arctan2(x, z)
    with np.errstate(divide="ignore", invalid="ignore"):
        yn = np.where(rho > 1e-9, y / np.maximum(rho, 1e-9), np.nan)
    return (h - 1) / 2.0 + f * yn, (w - 1) / 2.0 + f * phi, z


def corners(df):
    """8 cuboid corners in RIG coords for every row -> [N, 8, 3]."""
    c = df[["center_x", "center_y", "center_z"]].to_numpy(float)
    s = df[["size_x", "size_y", "size_z"]].to_numpy(float) / 2.0
    q = df[["orientation_x", "orientation_y", "orientation_z",
            "orientation_w"]].to_numpy(float)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                      for sz in (-1, 1)], float)              # [8,3]
    out = np.empty((len(df), 8, 3))
    for i in range(len(df)):
        R = q2R(*q[i])
        out[i] = c[i] + (signs * s[i]) @ R.T
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="176x624,160x592,128x576")
    ap.add_argument("--max-clips", type=int, default=400)
    ap.add_argument("--per-chunk", type=int, default=20,
                    help="clips per chunk — spread the sample over "
                         "chunks so BOTH rigs are represented")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cands = []
    for t in a.cands.split(","):
        h, w = t.split("x")
        cands.append((int(h), int(w)))

    zips = sorted(glob.glob(os.path.join(DR, "labels", "obstacle.offline",
                                         "*.zip")))
    out = {"measured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "parent": f"{PARENT_H}x{PARENT_W}f{F_REF:.4f}cyl",
           "candidates": [f"{h}x{w}" for h, w in cands],
           "n_chunks": len(zips), "clips": 0,
           "class_counts": {}, "errors": Counter()}

    RNG_BINS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 1e9)]
    tally = {}          # (cand, stratum) -> [n_visible_parent, n_lost]
    ground = {"A": [], "B": []}
    camh = {"A": [], "B": []}
    n_samples = 0
    rig_clip = Counter()

    def bump(cand, stratum, vis_par, lost):
        k = (cand, stratum)
        t = tally.setdefault(k, [0, 0])
        t[0] += int(vis_par.sum())
        t[1] += int((vis_par & lost).sum())

    t0 = time.time()
    for zp in zips:
        if out["clips"] >= a.max_clips:
            break
        z = zipfile.ZipFile(zp)
        used = 0
        for n in [x for x in z.namelist() if x.endswith(".parquet")]:
            if out["clips"] >= a.max_clips or used >= a.per_chunk:
                break
            clip = n.split("/")[-1].split(".")[0]
            if clip not in calib_clips:
                continue
            K = clip_rig(clip)
            if K is None:
                continue
            try:
                df = pd.read_parquet(io.BytesIO(z.read(n)))
            except Exception:                                 # noqa: BLE001
                out["errors"]["parquet"] += 1
                continue
            if not len(df):
                continue
            k = K[FW]
            rig = "B" if k["cy"] >= RIG_BOUNDARY_CY else "A"
            rig_clip[rig] += 1
            out["clips"] += 1
            used += 1
            for c, v in df.label_class.value_counts().items():
                out["class_counts"][c] = out["class_counts"].get(c, 0) + int(v)

            # ---- P2: nearest visible ground point straight ahead ----------
            camh[rig].append(float(k["t"][2]))
            d = np.linspace(0.2, 40.0, 3980)
            Pg = np.stack([d, np.zeros_like(d), np.zeros_like(d)], 1)
            Pc = (Pg - k["t"]) @ k["Rt"].T
            ug, vg, thg = project(Pg, k)
            obs_g = in_frame(ug, vg, thg, k)   # ⭐ the sensor really has it
            for (h, w) in [(PARENT_H, PARENT_W)] + cands:
                rr, cc, zz = cyl_rowcol(Pc, h, w)
                ok = (rr >= 0) & (rr <= h - 1) & (zz > 0)
                ground.setdefault(f"{h}x{w}", {}).setdefault(rig, []).append(
                    float(d[ok][0]) if ok.any() else float("nan"))
                oko = ok & obs_g
                ground.setdefault(f"{h}x{w}_obs", {}).setdefault(
                    rig, []).append(float(d[oko][0]) if oko.any()
                                    else float("nan"))

            # ---- P1: agents -----------------------------------------------
            C = corners(df)                                    # [N,8,3]
            flat = C.reshape(-1, 3)
            Pc = (flat - k["t"]) @ k["Rt"].T
            # ⭐ OBSERVED, not merely in-bounds: a corner that lands on a MASKED
            # output pixel is black for that rig and was never visible. Rig B's
            # bottom rows are exactly that, so counting bounds alone would bill
            # rig B for content it never had.
            u_n, v_n, th_n = project(flat, k)
            observed = in_frame(u_n, v_n, th_n, k)
            rng = np.linalg.norm(
                df[["center_x", "center_y"]].to_numpy(float), axis=1)
            cls = df.label_class.to_numpy()
            n_samples += len(df)
            vis, visobs = {}, {}
            for (h, w) in [(PARENT_H, PARENT_W)] + cands:
                rr, cc, zz = cyl_rowcol(Pc, h, w)
                inside = ((rr >= 0) & (rr <= h - 1) & (cc >= 0) &
                          (cc <= w - 1) & (zz > 0))
                vis[(h, w)] = inside.reshape(-1, 8).any(1)
                visobs[(h, w)] = (inside & observed).reshape(-1, 8).any(1)
            vp = vis[(PARENT_H, PARENT_W)]
            vpo = visobs[(PARENT_H, PARENT_W)]
            # how much of what the parent BOUNDS show is already masked away
            bump("_masked_away", "ALL", vp, vp & ~vpo)
            bump("_masked_away", f"rig{rig}", vp, vp & ~vpo)
            for (h, w) in cands:
                cand = f"{h}x{w}"
                lost = vp & ~vis[(h, w)]
                losto = vpo & ~visobs[(h, w)]
                bump(cand, "ALL", vp, lost)
                bump(cand, f"rig{rig}", vp, lost)
                bump(cand + "_obs", "ALL", vpo, losto)
                bump(cand + "_obs", f"rig{rig}", vpo, losto)
                for lo, hi in RNG_BINS:
                    m = (rng >= lo) & (rng < hi)
                    bump(cand + "_obs",
                         f"rng_{lo}-{hi if hi < 1e8 else 'inf'}", vpo & m, losto)
                for lo, hi in RNG_BINS:
                    m = (rng >= lo) & (rng < hi)
                    bump(cand, f"rng_{lo}-{hi if hi < 1e8 else 'inf'}",
                         vp & m, lost)
                for cname in ("automobile", "heavy_truck", "pedestrian",
                              "bicycle", "motorcycle", "protruding_object"):
                    m = cls == cname
                    if m.any():
                        bump(cand, f"cls_{cname}", vp & m, lost)

    out["seconds"] = round(time.time() - t0, 1)
    out["n_agent_samples"] = int(n_samples)
    out["rig_clips"] = dict(rig_clip)
    out["errors"] = dict(out["errors"])
    res = {}
    for (cand, stratum), (npar, nlost) in sorted(tally.items()):
        res.setdefault(cand, {})[stratum] = {
            "n_visible_parent": npar, "n_lost": nlost,
            "frac_lost": (nlost / npar) if npar else None}
    out["agent_loss"] = res
    gsum = {}
    for frame, per in ground.items():
        if not isinstance(per, dict):
            continue
        gsum[frame] = {}
        for rg, v in per.items():
            v = [x for x in v if not math.isnan(x)]
            if v:
                gsum[frame][rg] = {
                    "n": len(v), "mean_m": round(float(np.mean(v)), 4),
                    "median_m": round(float(np.median(v)), 4),
                    "p90_m": round(float(np.quantile(v, 0.9)), 4),
                    "min_m": round(float(np.min(v)), 4),
                    "max_m": round(float(np.max(v)), 4)}
    out["nearest_visible_ground_m"] = gsum
    out["camera_height_m"] = {
        rg: {"n": len(v), "min": round(float(np.min(v)), 4),
             "median": round(float(np.median(v)), 4),
             "max": round(float(np.max(v)), 4)}
        for rg, v in camh.items() if v}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("FIELD_COST_DONE " + json.dumps({
        "clips": out["clips"], "samples": out["n_agent_samples"],
        "rig_clips": out["rig_clips"],
        "loss_ALL": {c: res[c]["ALL"]["frac_lost"] for c in res},
        "ground": {f: {rg: gsum[f][rg]["median_m"] for rg in gsum[f]}
                   for f in gsum},
        "seconds": out["seconds"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
