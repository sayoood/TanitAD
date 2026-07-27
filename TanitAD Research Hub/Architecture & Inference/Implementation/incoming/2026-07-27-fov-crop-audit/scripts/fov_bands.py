"""FOV crop audit — PART 1: where does decision-relevant content sit, per situation?

Every `obstacle.offline` 3-D agent sample is projected into the front-wide camera with the clip's
OWN intrinsics + extrinsics (mandatory: the rig split makes a global (cx, cy) ~215 px wrong for
rig B) and assigned to exactly one band:

    IN_CROP       inside the 256x256 canonical crop the encoder actually receives
                  (square of half-side r(25.697 deg) about the per-clip (cx, cy))
    CROPPED_AWAY  inside the native front-wide frame but OUTSIDE that crop
                  -> already in the front camera's own pixels; a wider crop recovers it
    OFF_FRONT     outside the native front-wide frame
                  -> only cross_left / cross_right / rear can see it; a true sensor request

Three nested populations, all reported:

    P_ALL     every agent sample
    P_NEAR    range <= 40 m
    P_CROSS   the intersection discriminator itself: moving >= 2 m/s, roughly perpendicular,
              constant-velocity path CROSSING the ego's realised path within 40 m ahead.
              This is the decision-relevant population.

⭐ The single number that makes any crop width answerable later: `req_half_px`, the square
half-side (in native px) a crop centred on (cx, cy) would need in order to contain this agent
sample. Band membership at ANY candidate half-angle theta is then `in_front & (req_half_px <=
r(theta))` — so the whole recovery curve falls out of one stored scalar per sample, with no
re-projection.

The clock map onto the episode index is `sc_cross.fit_clock`'s: direct inversion of the stored
poses against egomotion, residual reported in METRES, admitted at <= 0.50 m.

🔒 No clip UUID is written to any artifact — clips carry the integer episode index `k` only.

usage:  python fov_bands.py <poses_dir> <bundle_dir> <out_dir> [n_clips]
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
# extrinsics). It lives in the session scratchpad rather than the repo; point SC_CRUX_DIR at
# whatever directory holds it. Same resolution order as the situation-classifier stream.
SCRATCH = os.environ.get("SC_CRUX_DIR") or os.path.join(
    os.environ.get("LOCALAPPDATA", ""), "Temp", "claude",
    "G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD")
if not os.path.isfile(os.path.join(SCRATCH, "crux.py")):
    hits = [os.path.join(dp, "crux.py") for dp, _, fn in os.walk(SCRATCH) if "crux.py" in fn]
    if hits:
        SCRATCH = os.path.dirname(hits[0])
sys.path.insert(0, SCRATCH)
from crux import CAMS, calib_clips, clip_rig, in_frame, poly_r, project   # noqa: E402

DR = os.environ.get("TANITAD_PAI_ROOT", r"C:\Users\Admin\tanitad-data\physicalai")
FW = "camera_front_wide_120fov"
EXTRA_CAMS = [c for c in CAMS if c != FW]

HZ = 10.0
F_REF, SIZE = 266.0, 256
CANON_HALF_DEG = math.degrees(math.atan((SIZE / 2) / F_REF))     # 25.697
PERP_COS = 0.643          # |cos(dpsi)| <= 0.643  <=>  50-130 deg
AGENT_V_MIN = 2.0         # m/s
CROSS_AHEAD_M = 40.0
CROSS_H_S = 4.0
NEAR_M = 40.0
ALIGN_MAX_RES_M = 0.50
DYNAMIC_EXCLUDE = {"trafficlight", "trafficsign", "trailer_hitch"}

# candidate half-angles for the recovery curve (deg). 25.697 is the status quo; 60.25 is native.
HALF_GRID_DEG = [CANON_HALF_DEG, 28.0, 30.0, 32.5, 35.0, 37.5, 40.0, 45.0, 50.0, 55.0, 60.25]


# --------------------------------------------------------------------- clock map (from sc_cross)
def fit_clock(P, ego):
    """-> (t_query [T], median position residual in METRES). Direct inversion; see sc_cross.py."""
    t = ego["t_us"].to_numpy(np.float64)
    ex, ey = ego["x"].to_numpy(float), ego["y"].to_numpy(float)
    T = len(P)
    px, py = P[:, 0].astype(float), P[:, 1].astype(float)
    d2 = (ex[None, :] - px[:, None]) ** 2 + (ey[None, :] - py[:, None]) ** 2
    tj = t[np.argmin(d2, 1)]
    j = np.arange(T, dtype=np.float64)
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


def seg_cross(ax, ay, bx, by, cx, cy, dx, dy):
    def o(px, py, qx, qy, rx, ry):
        return np.sign((qy - py) * (rx - qx) - (qx - px) * (ry - qy))
    return ((o(ax, ay, bx, by, cx, cy) != o(ax, ay, bx, by, dx, dy))
            & (o(cx, cy, dx, dy, ax, ay) != o(cx, cy, dx, dy, bx, by)))


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
        recs.append(pd.DataFrame(dict(
            t_us=gg, track_id=tid,
            X=np.interp(gg, ts, g.center_x.to_numpy(float)),
            Y=np.interp(gg, ts, g.center_y.to_numpy(float)),
            Z=np.interp(gg, ts, g.center_z.to_numpy(float)),
            cls=g.label_class.iloc[0])))
    return pd.concat(recs) if recs else None


# --------------------------------------------------------------------------------- the per-clip
def run_clip(P, ego, obst, K):
    """-> (per-sample record dict, per-frame dict, diagnostics)."""
    T = len(P)
    tq, resid = fit_clock(P, ego)
    diag = {"resid_m": resid, "admitted": bool(resid <= ALIGN_MAX_RES_M), "T": T}
    if resid > ALIGN_MAX_RES_M:
        return None, None, diag
    grid = tq.astype(np.int64)
    R = _resample(obst, grid)
    if R is None or not len(R):
        return None, None, diag
    gi = {int(t): i for i, t in enumerate(grid)}
    R = R[R.t_us.isin(gi)].copy()
    if not len(R):
        return None, None, diag
    ki = R.t_us.map(gi).to_numpy()
    P3 = R[["X", "Y", "Z"]].to_numpy(float)

    # ---- front-wide projection, per-clip (cx, cy) + per-clip extrinsics ----
    u, v, th = project(P3, K[FW])
    front_full = in_frame(u, v, th, K[FW])
    # the square half-side a crop centred on (cx, cy) would need to contain this sample
    req_half = np.maximum(np.abs(u - K["camera_front_wide_120fov"]["cx"]),
                          np.abs(v - K[FW]["cy"]))
    in_crop = (req_half <= K[FW]["c_half"]) & front_full

    # ---- visibility in any OTHER camera (so OFF_FRONT is "someone else could see it") ----
    vis_other = np.zeros(len(P3), bool)
    for cam in EXTRA_CAMS:
        uu, vv, tt = project(P3, K[cam])
        vis_other |= in_frame(uu, vv, tt, K[cam])

    rng = np.linalg.norm(P3[:, :2], axis=1)
    az = np.degrees(np.arctan2(P3[:, 1], P3[:, 0]))

    # ---- P_CROSS: the intersection discriminator, per (agent, frame) ----
    x, y, psi = P[:, 0].astype(float), P[:, 1].astype(float), np.unwrap(P[:, 2].astype(float))
    c_, s_ = np.cos(psi[ki]), np.sin(psi[ki])
    WX = x[ki] + c_ * R.X.to_numpy() - s_ * R.Y.to_numpy()
    WY = y[ki] + s_ * R.X.to_numpy() + c_ * R.Y.to_numpy()
    uniq, inv = np.unique(R.track_id.to_numpy(), return_inverse=True)
    nt = len(uniq)
    AX = np.full((nt, T), np.nan)
    AY = np.full((nt, T), np.nan)
    AX[inv, ki] = WX
    AY[inv, ki] = WY
    cls = pd.Series(R.cls.to_numpy()).groupby(inv).first().reindex(range(nt)).to_numpy()
    dyn = np.array([not (isinstance(c, str) and c.lower() in DYNAMIC_EXCLUDE) for c in cls])
    w = 3
    AVX = np.full((nt, T), np.nan)
    AVY = np.full((nt, T), np.nan)
    AVX[:, w:T - w] = (AX[:, 2 * w:] - AX[:, :T - 2 * w]) / (2 * w / HZ)
    AVY[:, w:T - w] = (AY[:, 2 * w:] - AY[:, :T - 2 * w]) / (2 * w / HZ)

    is_cross = np.zeros((nt, T), bool)
    H = int(CROSS_H_S * HZ)
    S = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
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
        a0x = AX[ok_i, t] - AVX[ok_i, t] * CROSS_H_S
        a0y = AY[ok_i, t] - AVY[ok_i, t] * CROSS_H_S
        a1x = AX[ok_i, t] + AVX[ok_i, t] * CROSS_H_S
        a1y = AY[ok_i, t] + AVY[ok_i, t] * CROSS_H_S
        hit = np.zeros(len(ok_i), bool)
        for q in range(t, j):
            hit |= seg_cross(a0x, a0y, a1x, a1y, x[q], y[q], x[q + 1], y[q + 1])
        is_cross[ok_i[hit], t] = True
    smp_cross = is_cross[inv, ki]

    rec = dict(t=ki.astype(np.int16), req_half=req_half.astype(np.float32),
               in_front=front_full, in_crop=in_crop, vis_other=vis_other,
               rng=rng.astype(np.float32), az=az.astype(np.float32), cross=smp_cross)
    return rec, None, diag


# ------------------------------------------------------------------------------------ per-frame
def frame_flags(rec, T, half_px):
    """Per-frame presence booleans for each (population, band) at crop half-side `half_px`."""
    out = {}
    pops = {"ALL": np.ones(len(rec["t"]), bool),
            "NEAR": rec["rng"] <= NEAR_M,
            "CROSS": rec["cross"]}
    in_crop = rec["in_front"] & (rec["req_half"] <= half_px)
    bands = {"IN_CROP": in_crop,
             "CROPPED_AWAY": rec["in_front"] & ~in_crop,
             "OFF_FRONT": (~rec["in_front"]) & rec["vis_other"]}
    for pn, pm in pops.items():
        for bn, bm in bands.items():
            a = np.zeros(T, bool)
            m = pm & bm
            if m.any():
                np.logical_or.at(a, rec["t"][m], True)
            out[f"{pn}_{bn}"] = a
    return out


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

    packs, rows, zcache, t0 = {}, [], {}, time.time()
    for m in meta:
        ch, clip = m["chunk"], k2clip[str(m["k"])]
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
        except Exception:                                          # noqa: BLE001
            continue
        if not len(obst):
            continue
        P = Z[f"p{file2i[m['file']]}"]
        try:
            rec, _, diag = run_clip(P, ego, obst, K)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [k={m['k']}] FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        if rec is None:
            rows.append(dict(k=m["k"], chunk=ch, side=m["side"], T=diag["T"],
                             resid_m=round(diag["resid_m"], 4), admitted=diag["admitted"],
                             n_samples=0, c_half=float(K[FW]["c_half"]),
                             cy=float(K[FW]["cy"]), rig="B" if K[FW]["cy"] >= 650 else "A"))
            continue
        kk = m["k"]
        for name, arr in rec.items():
            packs[f"c{kk}_{name}"] = arr
        rows.append(dict(k=kk, chunk=ch, side=m["side"], T=diag["T"],
                         resid_m=round(diag["resid_m"], 4), admitted=True,
                         n_samples=int(len(rec["t"])), c_half=float(K[FW]["c_half"]),
                         cy=float(K[FW]["cy"]), rig="B" if K[FW]["cy"] >= 650 else "A",
                         n_cross=int(rec["cross"].sum())))
        if len(rows) % 50 == 0:
            print(f"[bands] {len(rows)} clips ({time.time()-t0:.0f}s)", flush=True)
        if limit and len(rows) >= limit:
            break

    D = pd.DataFrame(rows)
    np.savez_compressed(os.path.join(out, "fov_bands.npz"), **packs)
    D.to_parquet(os.path.join(out, "fov_bands_index.parquet"))
    poly = (0.0, 927.5032, 23.1353, -58.5012, 16.5067)   # corpus-median, for the GRID ONLY
    summary = {
        "n_clips_attempted": len(D), "n_admitted": int(D.admitted.sum()),
        "canonical_half_deg": round(CANON_HALF_DEG, 4),
        "half_grid_deg": HALF_GRID_DEG,
        "half_grid_r_px_median_intr": [round(float(poly_r(poly, math.radians(h))), 2)
                                       for h in HALF_GRID_DEG],
        "rig_split": D.rig.value_counts().to_dict() if len(D) else {},
        "C_ALIGN_resid_m": {q: round(float(np.quantile(D.resid_m, float(q))), 5)
                            for q in ("0.0", "0.5", "0.9", "0.99", "1.0")} if len(D) else {},
        "note": "per-clip c_half is used for every band assignment; the grid r(theta) above uses "
                "the corpus-median poly and is reported for orientation only",
        "wallclock_s": round(time.time() - t0, 1),
    }
    json.dump(summary, open(os.path.join(out, "bands_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
