"""H2 E1/E0 — build the HELD-OUT label table.

Extends the substrate audit's `crux3.py` with exactly TWO additions and no changes:
  * `in_front_full` — is the agent inside the FULL 120.5 deg front_wide frame (E0's question),
    alongside the existing `in_crop` (= the 51.4 deg encoder crop);
  * per-frame `junction` / `lane_change` stratum masks, computed with `situ_full.py`'s detector
    thresholds verbatim, on the SAME 10 Hz grid as the labels (so indices align by construction).

Everything else -- f-theta projection, per-clip (cx,cy) + per-clip 6-DoF extrinsics (two-rig
corpus), the 0.5 s max-gap track resampler, the constant-speed constant-heading counterfactual ego
continuation, the 4 s response window -- is IMPORTED from `crux.py` / re-implemented identically
from `crux3.py`. The conflict-distance loop is vectorised (same arithmetic, ~30x faster).

CPU only. Read-only. No pod touched.

usage:  python h2e_build.py <out.parquet> [chunk ...]
"""
import glob, io, math, os, sys, time, zipfile

import numpy as np
import pandas as pd

S = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad"
sys.path.insert(0, S)
from crux import (CAMS, CI, SE, calib_clips, clip_rig, in_frame, in_model_crop,  # noqa: E402
                  project, resample_tracks)

DR = r"C:\Users\Admin\tanitad-data\physicalai"
FW = "camera_front_wide_120fov"
CL = "camera_cross_left_120fov"
CR = "camera_cross_right_120fov"
TAU = 4.0
H = int(TAU * 10)                  # 4.0 s at 10 Hz
TRACK_BLOCK = 24                   # tracks per vectorised block (bounds peak RAM)

_EGOZ: dict = {}


def ego_frame(ch, clip):
    if ch not in _EGOZ:
        z = zipfile.ZipFile(DR + rf"\labels\egomotion\egomotion.chunk_{ch}.zip")
        _EGOZ[ch] = (z, {n.split("/")[-1].split(".")[0]: n
                         for n in z.namelist() if n.endswith(".parquet")})
    z, d = _EGOZ[ch]
    if clip not in d:
        return None
    e = pd.read_parquet(io.BytesIO(z.read(d[clip])))
    e["t_us"] = e.timestamp.astype(np.int64)
    e["v"] = np.hypot(e.vx, e.vy)
    return e.sort_values("t_us")


def situation_masks(x, y, yaw_unwrapped, v, t_s):
    """situ_full.py detectors, thresholds verbatim, expressed as PER-FRAME masks.

    A window that fires marks every frame it spans. STRIDE=5 as in situ_full.py.
    """
    n = len(t_s)
    junc = np.zeros(n, bool)
    lane = np.zeros(n, bool)
    if n < 110:
        return junc, lane
    yaw = yaw_unwrapped
    for i in range(0, n - 100, 5):
        j6, j8 = i + 60, i + 80
        d6 = math.degrees(yaw[j6] - yaw[i])
        d8 = math.degrees(yaw[j8] - yaw[i])
        a6 = float(np.trapezoid(v[i:j6 + 1], t_s[i:j6 + 1]))
        R6 = abs(a6 / math.radians(abs(d6))) if abs(d6) > 1e-3 else 1e9
        if abs(d6) >= 45 and R6 <= 30:                       # intersection / junction
            junc[i:j6 + 1] = True
        if v[i] >= 8.0 and abs(d6) <= 3.0:                   # lane change
            s8 = np.degrees(np.diff(yaw[i:j8 + 1]))
            sp = s8[:60]
            pos, neg = sp[sp > 0].sum(), -sp[sp < 0].sum()
            if min(pos, neg) >= 2.0:
                c, s = math.cos(yaw[i]), math.sin(yaw[i])
                lat = -s * (x[i:j6 + 1] - x[i]) + c * (y[i:j6 + 1] - y[i])
                if 2.5 <= abs(lat[-1]) <= 5.0 and abs(lat[-1]) >= 0.8 * np.abs(lat).max():
                    lane[i:j6 + 1] = True
    return junc, lane


def run_clip(clip, ch, obst_df, K):
    e = ego_frame(ch, clip)
    if e is None or len(e) < 100:
        return None
    t0 = max(int(obst_df.timestamp_us.min()), int(e.t_us.min()))
    t1 = min(int(obst_df.timestamp_us.max()), int(e.t_us.max()))
    grid = np.arange(t0, t1, 100_000, dtype=np.int64)
    if len(grid) < 60:
        return None
    R = resample_tracks(obst_df, grid)
    if R is None or not len(R):
        return None

    et = e.t_us.to_numpy(np.int64)
    EX = np.interp(grid, et, e.x.to_numpy(float))
    EY = np.interp(grid, et, e.y.to_numpy(float))
    EV = np.interp(grid, et, e.v.to_numpy(float))
    qs = np.stack([np.interp(grid, et, e[c].to_numpy(float)) for c in ("qx", "qy", "qz", "qw")], 1)
    yaw = np.arctan2(2 * (qs[:, 3] * qs[:, 2] + qs[:, 0] * qs[:, 1]),
                     1 - 2 * (qs[:, 1] ** 2 + qs[:, 2] ** 2))
    ng = len(grid)

    gi = {t: i for i, t in enumerate(grid)}
    R = R[R.t_us.isin(gi)].copy()
    if not len(R):
        return None
    ii = R.t_us.map(gi).to_numpy()
    R["gi"] = ii
    cy_, sy_ = np.cos(yaw[ii]), np.sin(yaw[ii])
    R["WX"] = EX[ii] + cy_ * R.X.to_numpy() - sy_ * R.Y.to_numpy()
    R["WY"] = EY[ii] + sy_ * R.X.to_numpy() + cy_ * R.Y.to_numpy()

    # ---- projection: per-clip (cx,cy) + per-clip 6-DoF extrinsics (two-rig safe) ----
    P = R[["X", "Y", "Z"]].to_numpy(float)
    u, v_, th = project(P, K[FW])
    front_frame = in_frame(u, v_, th, K[FW])                 # FULL 120.5 deg field  <-- E0
    R["in_front_full"] = front_frame
    R["in_crop"] = in_model_crop(u, v_, K[FW]) & front_frame  # 51.4 deg encoder crop
    for cam in (CL, CR):
        uu, vv, tt = project(P, K[cam])
        R["v_" + cam] = in_frame(uu, vv, tt, K[cam])
    R["rng"] = np.hypot(R.X, R.Y)
    R["az"] = np.degrees(np.arctan2(R.Y, R.X))

    # ---- counterfactual (constant speed + constant heading) conflict distance ----
    nk = ng - H
    if nk <= 0:
        return None
    hs = np.arange(1, H + 1) * 0.1                            # [H]
    ks = np.arange(nk)
    cfx = EX[:nk, None] + np.cos(yaw[:nk])[:, None] * EV[:nk, None] * hs[None, :]   # [nk,H]
    cfy = EY[:nk, None] + np.sin(yaw[:nk])[:, None] * EV[:nk, None] * hs[None, :]
    idx = ks[:, None] + np.arange(1, H + 1)[None, :]          # [nk,H] future frame indices
    rex = EX[idx]
    rey = EY[idx]

    tids = R.track_id.to_numpy()
    uniq, inv = np.unique(tids, return_inverse=True)
    POSX = np.full((len(uniq), ng), np.nan)
    POSY = np.full((len(uniq), ng), np.nan)
    POSX[inv, R.gi.to_numpy()] = R.WX.to_numpy()
    POSY[inv, R.gi.to_numpy()] = R.WY.to_numpy()

    dcf = np.full((len(uniq), ng), np.nan)
    dre = np.full((len(uniq), ng), np.nan)
    for a in range(0, len(uniq), TRACK_BLOCK):
        b = slice(a, a + TRACK_BLOCK)
        AX = POSX[b][:, idx]                                  # [nb,nk,H]
        AY = POSY[b][:, idx]
        with np.errstate(invalid="ignore"):
            d1 = np.hypot(AX - cfx[None], AY - cfy[None])
            d2 = np.hypot(AX - rex[None], AY - rey[None])
        allnan = np.all(np.isnan(d1), axis=2)
        m1 = np.where(allnan, np.nan, np.nanmin(np.where(np.isnan(d1), np.inf, d1), axis=2))
        m2 = np.where(allnan, np.nan, np.nanmin(np.where(np.isnan(d2), np.inf, d2), axis=2))
        dcf[b, :nk] = m1
        dre[b, :nk] = m2
    # a frame with no future track sample at all keeps NaN (crux3 left dmin=1e9 unreachable ->
    # dropped by the same dropna downstream); guard the inf sentinel too
    dcf[np.isinf(dcf)] = np.nan
    dre[np.isinf(dre)] = np.nan

    R["dmin_cf"] = dcf[inv, R.gi.to_numpy()]
    R["dmin_real"] = dre[inv, R.gi.to_numpy()]

    dv = np.full(ng, np.nan)
    dv[:nk] = EV[H:] - EV[:nk]
    R["ego_dv"] = dv[R.gi.to_numpy()]
    R["ego_v"] = EV[R.gi.to_numpy()]

    # ---- strata on the SAME grid ----
    yaw_u = np.unwrap(yaw)
    junc, lane = situation_masks(EX, EY, yaw_u, EV, grid / 1e6)
    R["junction"] = junc[R.gi.to_numpy()]
    R["lane_change"] = lane[R.gi.to_numpy()]

    R["clip_id"] = clip
    R["chunk"] = ch
    return R.drop(columns=["WX", "WY", "t_us", "X", "Y", "Z", "L"])


def main(out_path, chunks):
    keep = ["gi", "track_id", "cls", "in_front_full", "in_crop",
            "v_" + CL, "v_" + CR, "rng", "az", "dmin_cf", "dmin_real",
            "ego_dv", "ego_v", "junction", "lane_change", "clip_id", "chunk"]
    parts, t_start = [], time.time()
    for ch in chunks:
        zp = DR + rf"\labels\obstacle.offline\obstacle.offline.chunk_{ch}.zip"
        if not os.path.exists(zp):
            print(f"[{ch}] no obstacle zip -- skip", flush=True)
            continue
        if not os.path.exists(DR + rf"\labels\egomotion\egomotion.chunk_{ch}.zip"):
            print(f"[{ch}] no egomotion -- skip", flush=True)
            continue
        z = zipfile.ZipFile(zp)
        n_ok = n_try = 0
        for n in [q for q in z.namelist() if q.endswith(".parquet")]:
            clip = n.split("/")[-1].split(".")[0]
            if clip not in calib_clips:
                continue
            n_try += 1
            K = clip_rig(clip)
            if K is None:
                continue
            try:
                df = pd.read_parquet(io.BytesIO(z.read(n)))
            except Exception:                                  # noqa: BLE001
                continue
            if not len(df):
                continue
            r = run_clip(clip, ch, df, K)
            if r is None or not len(r):
                continue
            parts.append(r[keep])
            n_ok += 1
        print(f"[chunk {ch}] clips ok {n_ok}/{n_try}   t={time.time()-t_start:.0f}s", flush=True)
    A = pd.concat(parts, ignore_index=True)
    A.to_parquet(out_path)
    print(f"\nWROTE {out_path}: rows {len(A):,}  clips {A.clip_id.nunique()}  "
          f"chunks {A.chunk.nunique()}  t={time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
