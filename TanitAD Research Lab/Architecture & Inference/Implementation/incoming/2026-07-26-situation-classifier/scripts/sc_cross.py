"""Situation classifier — CROSS TRAFFIC and the PER-CAMERA NEED (dev box, CPU, read-only).

Two things this file produces, both on the EPISODE frame index so they line up with the features:

  1. **`cross[t]`** — the discriminator of `PRE_REGISTRATION.md` Sec 2.3: at least one moving agent
     that is roughly PERPENDICULAR to the ego and whose constant-velocity path CROSSES the ego's
     realised path within 40 m ahead. This is what separates a junction from a road curve, and it is
     what makes the intersection label more than a heading-change proxy.
  2. **`need[X][t]`** — for each additional camera `X`, whether an agent projects INTO `X` but NOT
     into the canonical 51.4 deg encoder crop. This is H2's `T_off` machinery re-used verbatim
     (`crux.project` / `in_frame` / `in_model_crop`, per-clip `(cx,cy)` + per-clip 6-DoF extrinsics —
     mandatory on this two-rig corpus).

### C-ALIGN — the clock map, fitted on POSITION and reported in METRES

`build_episode` evaluates the ego at `t_query = linspace(t_frames[0], t_frames[-1], n)` on the
camera clock and then drops the first `n_stack-1 = 2` samples. Obstacles live on the
egomotion/obstacle clock. The two endpoints are recovered per clip by minimising the mean POSITION
error between the stored `poses[:, :2]` and egomotion interpolated at the candidate linspace —
a 2-parameter coarse-to-fine search whose residual is a **distance in metres**, not a correlation.
A clip is admitted at median residual <= 0.50 m; the distribution is published.

usage:  python sc_cross.py <poses_dir> <bundle_dir> <out_dir> [n_chunks]
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

# `crux.py` — the validated E0/E1 projection machinery (per-clip (cx,cy) + per-clip 6-DoF
# extrinsics, mandatory on this two-rig corpus). It lives in the session scratchpad rather than the
# repo; point SC_CRUX_DIR at whatever directory holds it.
SCRATCH = os.environ.get("SC_CRUX_DIR") or os.path.join(
    os.environ.get("LOCALAPPDATA", ""), "Temp", "claude",
    "G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD")
if not os.path.isfile(os.path.join(SCRATCH, "crux.py")):
    hits = [os.path.join(dp, "crux.py") for dp, _, fn in os.walk(SCRATCH) if "crux.py" in fn]
    if hits:
        SCRATCH = os.path.dirname(hits[0])
sys.path.insert(0, SCRATCH)
from crux import (CAMS, calib_clips, clip_rig, in_frame,  # noqa: E402
                  in_model_crop, project)

DR = r"C:\Users\Admin\tanitad-data\physicalai"
FW = "camera_front_wide_120fov"
EXTRA_CAMS = [c for c in CAMS if c != FW]

HZ = 10.0
PERP_COS = 0.643          # |cos(dpsi)| <= 0.643  <=>  50-130 deg  (PERPENDICULAR)
AGENT_V_MIN = 2.0         # m/s   actually moving
CROSS_AHEAD_M = 40.0      # m     ego arc length within which the crossing must occur
CROSS_H_S = 4.0           # s     agent CV horizon (and backward reach)
ALIGN_MAX_RES_M = 0.50    # m     C-ALIGN admission floor
DYNAMIC_EXCLUDE = {"trafficlight", "trafficsign", "trailer_hitch"}


# ------------------------------------------------------------------------- C-ALIGN: the clock map
def fit_clock(P, ego):
    """-> (t_query [T], median position residual in metres).

    ⭐ MEASURED, and it corrected a wrong assumption in the first version of this file: an EPISODE
    is a ~19.9 s slice of a clip whose egomotion spans **36-140 s**, so searching for the linspace
    endpoints anywhere near the egomotion endpoints is hopeless (median residual 24 m, 85 % of clips
    refused). The map is recovered instead by DIRECT INVERSION:

      1. nearest-neighbour match each stored pose (x, y) to an egomotion sample -> t_j,
      2. robust linear fit t_j = alpha*j + beta (`t_query` IS a linspace, so the map is exactly
         affine; two MAD-trimmed re-fits remove the ambiguous matches where the path self-crosses),
      3. residual = median | poses(x,y) - egomotion(x,y) interpolated at the FITTED times |, metres.

    On the first four clips this gives median residuals of **0.005-0.021 m** and a fitted step of
    0.0971-0.1007 s, i.e. the 10 Hz grid, recovered to centimetres.
    """
    t = ego["t_us"].to_numpy(np.float64)
    ex, ey = ego["x"].to_numpy(float), ego["y"].to_numpy(float)
    T = len(P)
    px, py = P[:, 0].astype(float), P[:, 1].astype(float)
    d2 = (ex[None, :] - px[:, None]) ** 2 + (ey[None, :] - py[:, None]) ** 2
    tj = t[np.argmin(d2, 1)]
    j = np.arange(T, dtype=np.float64)
    keep = np.ones(T, bool)
    A = np.polyfit(j, tj, 1)
    for _ in range(2):
        r = np.abs(tj - np.polyval(A, j))
        mad = max(float(np.median(r)), 1.0)
        keep = r <= 5.0 * mad
        if keep.sum() < max(20, T // 4):
            break
        A = np.polyfit(j[keep], tj[keep], 1)
    q = np.polyval(A, j)
    resid = float(np.median(np.hypot(np.interp(q, t, ex) - px, np.interp(q, t, ey) - py)))
    return q, resid


# ------------------------------------------------------------------------- geometry: the crossing
def seg_cross(ax, ay, bx, by, cx, cy, dx, dy):
    """Do segments AB and CD intersect?  Vectorised over A/B (agents) vs one C/D pair."""
    def o(px, py, qx, qy, rx, ry):
        return np.sign((qy - py) * (rx - qx) - (qx - px) * (ry - qy))
    o1 = o(ax, ay, bx, by, cx, cy)
    o2 = o(ax, ay, bx, by, dx, dy)
    o3 = o(cx, cy, dx, dy, ax, ay)
    o4 = o(cx, cy, dx, dy, bx, by)
    return (o1 != o2) & (o3 != o4)


def run_clip(P, ego, obst, K):
    """-> dict of per-frame boolean arrays on the episode index."""
    T = len(P)
    tq, resid = fit_clock(P, ego)
    x, y, psi = P[:, 0].astype(float), P[:, 1].astype(float), np.unwrap(P[:, 2].astype(float))
    ot = obst.timestamp_us.to_numpy(np.int64)
    # ⭐ THE `obstacle.offline` CLOCK JOIN, PROVEN not assumed (the trap that moved a sibling
    # stream's headline rate by x5.7): `egomotion` spans 36-140 s while `obstacle.offline` spans
    # ~20 s. The episode's t_query is recovered here from POSITION ALONE — no timestamp is read —
    # and it must then land INSIDE the obstacle window. The overlap fraction is published per clip.
    overlap = (min(tq[-1], float(ot.max())) - max(tq[0], float(ot.min()))) / max(tq[-1] - tq[0], 1.0)
    out = {"resid_m": resid, "obst_overlap": float(overlap), "cross": np.zeros(T, bool),
           "perp_present": np.zeros(T, bool),
           **{f"need_{c}": np.zeros(T, bool) for c in EXTRA_CAMS},
           "any_off_front": np.zeros(T, bool), "n_agents": np.zeros(T, np.int32)}
    if resid > ALIGN_MAX_RES_M:
        out["admitted"] = False
        return out
    out["admitted"] = True

    grid = tq.astype(np.int64)
    R = _resample(obst, grid)
    if R is None or not len(R):
        return out
    gi = {int(t): i for i, t in enumerate(grid)}
    R = R[R.t_us.isin(gi)].copy()
    if not len(R):
        return out
    ki = R.t_us.map(gi).to_numpy()
    P3 = R[["X", "Y", "Z"]].to_numpy(float)

    # ---- per-camera frustum membership (per-clip cx/cy + per-clip extrinsics) ----
    u, v, th = project(P3, K[FW])
    front_full = in_frame(u, v, th, K[FW])
    in_crop = in_model_crop(u, v, K[FW]) & front_full
    vis = {}
    for cam in EXTRA_CAMS:
        uu, vv, tt = project(P3, K[cam])
        vis[cam] = in_frame(uu, vv, tt, K[cam])
    for cam in EXTRA_CAMS:
        m = vis[cam] & ~in_crop
        np.logical_or.at(out[f"need_{cam}"], ki[m], True)
    m = (~in_crop) & (front_full | np.any(np.stack([vis[c] for c in EXTRA_CAMS]), 0))
    np.logical_or.at(out["any_off_front"], ki[m], True)
    np.add.at(out["n_agents"], ki, 1)

    # ---- world-frame agent state ----
    c_, s_ = np.cos(psi[ki]), np.sin(psi[ki])
    WX = x[ki] + c_ * R.X.to_numpy() - s_ * R.Y.to_numpy()
    WY = y[ki] + s_ * R.X.to_numpy() + c_ * R.Y.to_numpy()
    WYAW = psi[ki] + R.YAWR.to_numpy()
    uniq, inv = np.unique(R.track_id.to_numpy(), return_inverse=True)
    nt = len(uniq)
    AX = np.full((nt, T), np.nan)
    AY = np.full((nt, T), np.nan)
    AP = np.full((nt, T), np.nan)
    AX[inv, ki] = WX
    AY[inv, ki] = WY
    AP[inv, ki] = WYAW
    dyn = np.ones(nt, bool)
    cls = pd.Series(R.cls.to_numpy()).groupby(inv).first().reindex(range(nt)).to_numpy()
    for i, c in enumerate(cls):
        if isinstance(c, str) and c.lower() in DYNAMIC_EXCLUDE:
            dyn[i] = False
    w = 3
    AVX = np.full((nt, T), np.nan)
    AVY = np.full((nt, T), np.nan)
    AVX[:, w:T - w] = (AX[:, 2 * w:] - AX[:, :T - 2 * w]) / (2 * w / HZ)
    AVY[:, w:T - w] = (AY[:, 2 * w:] - AY[:, :T - 2 * w]) / (2 * w / HZ)

    # ---- the CROSS predicate, per frame ----
    H = int(CROSS_H_S * HZ)
    seg = np.hypot(np.diff(x), np.diff(y))
    S = np.concatenate([[0.0], np.cumsum(seg)])
    for t in range(T):
        j = min(t + H, T - 1)
        far = np.searchsorted(S, S[t] + CROSS_AHEAD_M)
        j = min(j, max(int(far), t + 1))
        if j <= t:
            continue
        spd = np.hypot(AVX[:, t], AVY[:, t])
        ok = dyn & np.isfinite(spd) & (spd >= AGENT_V_MIN)
        if not ok.any():
            continue
        head = np.arctan2(AVY[ok, t], AVX[ok, t])
        ok_i = np.nonzero(ok)[0][np.abs(np.cos(head - psi[t])) <= PERP_COS]
        if not len(ok_i):
            continue
        # `perp_present` = a moving PERPENDICULAR agent exists at all. Published beside `cross`
        # because it is the sibling stream's statistic (31.86 % of covered windows, INHERITED) and
        # reproducing it is the cheapest possible check that this join is not silently empty.
        out["perp_present"][t] = True
        # agent CV segment over [t - CROSS_H_S, t + CROSS_H_S]
        a0x = AX[ok_i, t] - AVX[ok_i, t] * CROSS_H_S
        a0y = AY[ok_i, t] - AVY[ok_i, t] * CROSS_H_S
        a1x = AX[ok_i, t] + AVX[ok_i, t] * CROSS_H_S
        a1y = AY[ok_i, t] + AVY[ok_i, t] * CROSS_H_S
        hit = False
        for q in range(t, j):
            if seg_cross(a0x, a0y, a1x, a1y, x[q], y[q], x[q + 1], y[q + 1]).any():
                hit = True
                break
        out["cross"][t] = hit
    return out


def _resample(df, grid_us, max_gap_us=500_000):
    recs = []
    for tid, g in df.groupby("track_id", sort=False):
        g = g.sort_values("timestamp_us")
        ts = g.timestamp_us.to_numpy(np.int64)
        if len(ts) < 2:
            continue
        ok = (grid_us >= ts[0] - max_gap_us) & (grid_us <= ts[-1] + max_gap_us)
        gg = grid_us[ok]
        if not len(gg):
            continue
        idx = np.clip(np.searchsorted(ts, gg), 1, len(ts) - 1)
        gg = gg[np.minimum(np.abs(gg - ts[idx - 1]), np.abs(ts[idx] - gg)) <= max_gap_us]
        if len(gg) < 7:
            continue
        qz, qw = g.orientation_z.to_numpy(float), g.orientation_w.to_numpy(float)
        qx, qy = g.orientation_x.to_numpy(float), g.orientation_y.to_numpy(float)
        yaw = np.unwrap(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2)))
        recs.append(pd.DataFrame(dict(
            t_us=gg, track_id=tid,
            X=np.interp(gg, ts, g.center_x.to_numpy(float)),
            Y=np.interp(gg, ts, g.center_y.to_numpy(float)),
            Z=np.interp(gg, ts, g.center_z.to_numpy(float)),
            YAWR=np.interp(gg, ts, yaw), cls=g.label_class.iloc[0])))
    return pd.concat(recs) if recs else None


def ego_frame(ch, clip, cache={}):
    if ch not in cache:
        z = zipfile.ZipFile(DR + rf"\labels\egomotion\egomotion.chunk_{ch}.zip")
        cache[ch] = (z, {n.split("/")[-1].split(".")[0]: n
                         for n in z.namelist() if n.endswith(".parquet")})
    z, d = cache[ch]
    if clip not in d:
        return None
    e = pd.read_parquet(io.BytesIO(z.read(d[clip])))
    e["t_us"] = e.timestamp.astype(np.int64)
    return e.sort_values("t_us")


def main():
    poses_dir, bundle, out = sys.argv[1:4]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    os.makedirs(out, exist_ok=True)
    Z = np.load(os.path.join(poses_dir, "poses.npz"))
    pmeta = json.load(open(os.path.join(poses_dir, "poses_meta.json")))
    meta = json.load(open(os.path.join(bundle, "sc_meta.json")))
    k2clip = json.load(open(os.path.join(bundle, "_LOCAL_ONLY_k2clip.json")))
    file2i = {m["file"]: m["i"] for m in pmeta if m["cache"] == "train"}
    have = {os.path.basename(p).split(".")[2].replace("chunk_", "")
            for p in os.listdir(DR + r"\labels\obstacle.offline")}
    packs, rec, t0 = {}, [], time.time()
    zcache = {}
    for m in meta:
        ch = m["chunk"]
        clip = k2clip[str(m["k"])]
        if ch not in have or clip not in calib_clips:
            continue
        if ch not in zcache:
            zcache[ch] = zipfile.ZipFile(
                DR + rf"\labels\obstacle.offline\obstacle.offline.chunk_{ch}.zip")
        z = zcache[ch]
        nm = [n for n in z.namelist() if n.endswith(".parquet") and clip in n]
        if not nm:
            continue
        K = clip_rig(clip)
        ego = ego_frame(ch, clip)
        if K is None or ego is None or len(ego) < 100:
            continue
        try:
            obst = pd.read_parquet(io.BytesIO(z.read(nm[0])))
        except Exception:                                   # noqa: BLE001
            continue
        if not len(obst):
            continue
        P = Z[f"p{file2i[m['file']]}"]
        try:
            r = run_clip(P, ego, obst, K)
        except Exception as exc:                            # noqa: BLE001
            print(f"  [k={m['k']}] FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        packs[f"c{m['k']}_cross"] = r["cross"]
        packs[f"c{m['k']}_perp_present"] = r["perp_present"]
        packs[f"c{m['k']}_any_off_front"] = r["any_off_front"]
        packs[f"c{m['k']}_n_agents"] = r["n_agents"]
        for c in EXTRA_CAMS:
            packs[f"c{m['k']}_need_{c}"] = r[f"need_{c}"]
        rec.append(dict(k=m["k"], chunk=ch, side=m["side"], T=int(len(P)),
                        resid_m=round(float(r["resid_m"]), 4), admitted=bool(r["admitted"]),
                        obst_overlap=round(float(r["obst_overlap"]), 4),
                        n_agent_frames=int((r["n_agents"] > 0).sum()),
                        cross_frac=round(float(r["cross"].mean()), 5),
                        perp_frac=round(float(r["perp_present"].mean()), 5)))
        if len(rec) % 50 == 0:
            print(f"[cross] {len(rec)} clips ({time.time()-t0:.0f}s)", flush=True)
        if limit and len(rec) >= limit:
            break
    D = pd.DataFrame(rec)
    np.savez_compressed(os.path.join(out, "sc_cross.npz"), **packs)
    D.to_parquet(os.path.join(out, "sc_cross_index.parquet"))
    res = D.resid_m.to_numpy() if len(D) else np.zeros(0)
    summary = {"n_clips": len(D), "admitted": int(D.admitted.sum()) if len(D) else 0,
               "C_ALIGN": {"statistic": "median per-frame POSITION residual, metres",
                           "admission_floor_m": ALIGN_MAX_RES_M,
                           "quantiles_m": {q: round(float(np.quantile(res, float(q))), 5)
                                           for q in ("0.0", "0.5", "0.9", "0.99", "1.0")}
                           if len(res) else {},
                           "fail_frac": round(float((res > ALIGN_MAX_RES_M).mean()), 5)
                           if len(res) else None},
               "OBSTACLE_CLOCK_JOIN_PROOF": {
                   "why": ("egomotion spans 36-140 s; obstacle.offline spans ~20 s. The episode's "
                           "t_query is recovered from POSITION ALONE and must land inside the "
                           "obstacle window -- an independent confirmation that the two share an "
                           "origin, never an assumption that they do."),
                   "overlap_quantiles": {q: round(float(np.quantile(D.obst_overlap, float(q))), 5)
                                         for q in ("0.0", "0.01", "0.5", "1.0")} if len(D) else {},
                   "clips_with_agent_frames": int((D.n_agent_frames > 0).sum()) if len(D) else 0},
               "cross_frame_rate": round(float(D.cross_frac.mean()), 5) if len(D) else None,
               "perp_present_frame_rate": round(float(D.perp_frac.mean()), 5) if len(D) else None}
    json.dump(summary, open(os.path.join(out, "cross_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
