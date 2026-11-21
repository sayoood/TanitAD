"""H2 L2 — build the BEHAVIOURAL sensor-need label table (per FRAME, 10 Hz).

Implements `PRE_REGISTRATION_L2.md` Sec 1 exactly. CPU only, read-only, no pod touched.

What is REUSED verbatim from the validated E0/E1 machinery (`crux.py`, reproduced the substrate
audit to three digits in `fidelity_check.json`):
    clip_rig, project, in_frame, in_model_crop      -- f-theta projection with PER-CLIP (cx, cy)
                                                       and PER-CLIP 6-DoF extrinsics (two-rig safe)
What is NEW (and is the whole point):
    * agents extrapolated at CONSTANT VELOCITY, not along their realised track            (M1)
    * the ego follows its REALISED PATH with its speed FROZEN, and brakes on it           (M2)
    * the trigger axis is REQUIRED DECELERATION over real oriented footprints, not
      centre-to-centre Euclidean distance                                                 (M3)
    * the response is a rare DECELERATION ONSET FROM FREE FLOW, not a 4 s speed difference

Outputs one row per (clip, frame) with the aggregated a_req fields for every ego/agent variant,
so DEV and CONFIRM analyses are pure table reads.

usage:  python l2_build.py <out_dir> [chunk ...]
"""
import io
import math
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

SCRATCH = (r"C:\Users\Admin\AppData\Local\Temp\claude"
           r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
           r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, SCRATCH)
from crux import (CI, calib_clips, clip_rig, in_frame, in_model_crop,  # noqa: E402
                  project)

DR = r"C:\Users\Admin\tanitad-data\physicalai"
FW = "camera_front_wide_120fov"
CL = "camera_cross_left_120fov"
CR = "camera_cross_right_120fov"

TAU_H = 4.0                        # conflict / response horizon, seconds
H = int(TAU_H * 10)                # 40 steps at 10 Hz
HS = np.arange(1, H + 1) * 0.1     # (0, 4.0]
A_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0])
VEL_HALFWIN = 3                    # +-0.3 s central difference for agent velocity
MIN_TRACK_SAMPLES = 7
PAIR_BLOCK = 400_000

# ego footprint -- MEASURED, calibration/calibration/vehicle_dimensions (identical on all 100 rows)
EGO_L, EGO_W = 4.872, 2.121
R_MAX_PAD = 8.0                    # generous footprint-sum bound for the cheap pre-filter

_EGOZ: dict = {}


# --------------------------------------------------------------------------- geometry primitives
def rect_radius(theta, L, W):
    """Exact radial extent of a centred rectangle (half-length L/2, half-width W/2) along `theta`
    measured in the rectangle's own frame. min(a/|cos|, b/|sin|)."""
    c = np.abs(np.cos(theta))
    s = np.abs(np.sin(theta))
    with np.errstate(divide="ignore", invalid="ignore"):
        ra = np.where(c > 1e-9, (L * 0.5) / np.maximum(c, 1e-9), np.inf)
        rb = np.where(s > 1e-9, (W * 0.5) / np.maximum(s, 1e-9), np.inf)
    return np.minimum(ra, rb)


def gap_field(ex, ey, epsi, ax, ay, ayaw, La, Wa):
    """gap = ||dp|| - rho_ego - rho_agent, all args broadcastable to [P, H]."""
    dx = ax - ex
    dy = ay - ey
    d = np.hypot(dx, dy)
    bw = np.arctan2(dy, dx)                       # world bearing ego -> agent
    rho_e = rect_radius(bw - epsi, EGO_L, EGO_W)
    rho_a = rect_radius(bw + math.pi - ayaw, La, Wa)
    return d - rho_e - rho_a


# --------------------------------------------------------------------------- per-clip inputs
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


def _movavg(x, n):
    """centred moving average, edge-padded."""
    if n < 2:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def resample_tracks_full(df, grid_us, max_gap_us=500_000):
    """crux.resample_tracks + size_y + yaw (rig frame). Same 0.5 s max-gap guard."""
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
        gap = np.minimum(np.abs(gg - ts[idx - 1]), np.abs(ts[idx] - gg))
        gg = gg[gap <= max_gap_us]
        if len(gg) < MIN_TRACK_SAMPLES:
            continue
        qz = g.orientation_z.to_numpy(float)
        qw = g.orientation_w.to_numpy(float)
        qx = g.orientation_x.to_numpy(float)
        qy = g.orientation_y.to_numpy(float)
        yaw = np.unwrap(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2)))
        recs.append(pd.DataFrame(dict(
            t_us=gg, track_id=tid,
            X=np.interp(gg, ts, g.center_x.to_numpy(float)),
            Y=np.interp(gg, ts, g.center_y.to_numpy(float)),
            Z=np.interp(gg, ts, g.center_z.to_numpy(float)),
            SX=np.interp(gg, ts, g.size_x.to_numpy(float)),
            SY=np.interp(gg, ts, g.size_y.to_numpy(float)),
            YAWR=np.interp(gg, ts, yaw),
            cls=g.label_class.iloc[0])))
    return pd.concat(recs) if recs else None


def situation_masks(x, y, yaw_u, v, t_s):
    """situ_full.py detectors, thresholds VERBATIM (copied from h2e_build.py), as per-frame masks."""
    n = len(t_s)
    junc = np.zeros(n, bool)
    lane = np.zeros(n, bool)
    if n < 110:
        return junc, lane
    yaw = yaw_u
    for i in range(0, n - 100, 5):
        j6, j8 = i + 60, i + 80
        d6 = math.degrees(yaw[j6] - yaw[i])
        a6 = float(np.trapezoid(v[i:j6 + 1], t_s[i:j6 + 1]))
        R6 = abs(a6 / math.radians(abs(d6))) if abs(d6) > 1e-3 else 1e9
        if abs(d6) >= 45 and R6 <= 30:
            junc[i:j6 + 1] = True
        if v[i] >= 8.0 and abs(d6) <= 3.0:
            s8 = np.degrees(np.diff(yaw[i:j8 + 1]))
            sp = s8[:60]
            pos, neg = sp[sp > 0].sum(), -sp[sp < 0].sum()
            if min(pos, neg) >= 2.0:
                c, s = math.cos(yaw[i]), math.sin(yaw[i])
                lat = -s * (x[i:j6 + 1] - x[i]) + c * (y[i:j6 + 1] - y[i])
                if 2.5 <= abs(lat[-1]) <= 5.0 and abs(lat[-1]) >= 0.8 * np.abs(lat).max():
                    lane[i:j6 + 1] = True
    return junc, lane


# --------------------------------------------------------------------------- the ego counterfactual
def ego_traj_pathspeed(EX, EY, PSI, EV, nk):
    """[nA, nk, H] ego positions + headings: realised PATH, speed frozen at v(t), braking at A.

    Beyond the end of the realised path the route is continued STRAIGHT along the final heading --
    which is exactly the case the label must cover (the ego stopped, so its realised path ends).
    """
    seg = np.hypot(np.diff(EX), np.diff(EY))
    S = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], np.diff(S) > 1e-3])
    Sd, Xd, Yd, Pd = S[keep], EX[keep], EY[keep], np.unwrap(PSI)[keep]
    tail_s = np.arange(1.0, 401.0)
    Sd = np.concatenate([Sd, Sd[-1] + tail_s])
    Xd = np.concatenate([Xd, Xd[-1] + np.cos(Pd[-1]) * tail_s])
    Yd = np.concatenate([Yd, Yd[-1] + np.sin(Pd[-1]) * tail_s])
    Pd = np.concatenate([Pd, np.full(len(tail_s), Pd[-1])])

    v0 = EV[:nk][:, None]                                        # [nk,1]
    out_x = np.empty((len(A_GRID), nk, H))
    out_y = np.empty((len(A_GRID), nk, H))
    out_p = np.empty((len(A_GRID), nk, H))
    for ai, A in enumerate(A_GRID):
        if A <= 0:
            s = v0 * HS[None, :]
        else:
            s = np.minimum(v0 * HS[None, :] - 0.5 * A * HS[None, :] ** 2, v0 ** 2 / (2 * A))
        tgt = S[:nk][:, None] + np.maximum(s, 0.0)
        out_x[ai] = np.interp(tgt.ravel(), Sd, Xd).reshape(nk, H)
        out_y[ai] = np.interp(tgt.ravel(), Sd, Yd).reshape(nk, H)
        out_p[ai] = np.interp(tgt.ravel(), Sd, Pd).reshape(nk, H)
    return out_x, out_y, out_p, S


def ego_traj_cv(EX, EY, PSI, EV, nk):
    """[nA, nk, H] ego positions: CONSTANT HEADING straight line (the `L1_gate` model), braking at A."""
    c, s_ = np.cos(PSI[:nk])[:, None], np.sin(PSI[:nk])[:, None]
    v0 = EV[:nk][:, None]
    out_x = np.empty((len(A_GRID), nk, H))
    out_y = np.empty((len(A_GRID), nk, H))
    out_p = np.empty((len(A_GRID), nk, H))
    for ai, A in enumerate(A_GRID):
        if A <= 0:
            d = v0 * HS[None, :]
        else:
            d = np.minimum(v0 * HS[None, :] - 0.5 * A * HS[None, :] ** 2, v0 ** 2 / (2 * A))
        d = np.maximum(d, 0.0)
        out_x[ai] = EX[:nk][:, None] + c * d
        out_y[ai] = EY[:nk][:, None] + s_ * d
        out_p[ai] = np.repeat(PSI[:nk][:, None], H, axis=1)
    return out_x, out_y, out_p


# --------------------------------------------------------------------------- a_req over flagged pairs
def solve_areq(EPX, EPY, EPP, kk, AXf, AYf, AYAWf, La, Wa):
    """Smallest A in A_GRID that keeps min_h gap > 0, for each flagged pair, PLUS the A = 0 minimum
    separation (MSD). Vectorised, A ascending with a shrinking unresolved set (A = 0 evaluates all;
    higher A only the survivors). Blocked over pairs so peak RAM is bounded.

    `areq == A_GRID[-1]` (8.0) is the sentinel for NOT AVOIDABLE BY BRAKING within the grid.
    """
    P = len(kk)
    areq = np.full(P, np.nan)
    msd = np.full(P, np.nan)
    for b0 in range(0, P, PAIR_BLOCK):
        sl = np.arange(b0, min(b0 + PAIR_BLOCK, P))
        unres = sl
        for ai in range(len(A_GRID)):
            if not len(unres):
                break
            g = gap_field(EPX[ai][kk[unres]], EPY[ai][kk[unres]], EPP[ai][kk[unres]],
                          AXf[unres], AYf[unres], AYAWf[unres],
                          La[unres][:, None], Wa[unres][:, None])
            with np.errstate(invalid="ignore"):
                gmin = np.nanmin(np.where(np.isnan(g), np.inf, g), axis=1)
            if ai == 0:
                msd[unres] = np.where(np.isinf(gmin), np.nan, gmin)
            clear = gmin > 0.0
            areq[unres[clear]] = A_GRID[ai]
            unres = unres[~clear]
    areq[np.isnan(areq)] = A_GRID[-1]
    return areq, msd


# --------------------------------------------------------------------------- one clip
def run_clip(clip, ch, obst_df, K):
    e = ego_frame(ch, clip)
    if e is None or len(e) < 100:
        return None
    t0 = max(int(obst_df.timestamp_us.min()), int(e.t_us.min()))
    t1 = min(int(obst_df.timestamp_us.max()), int(e.t_us.max()))
    grid = np.arange(t0, t1, 100_000, dtype=np.int64)
    ng = len(grid)
    nk = ng - H
    if nk < 40:
        return None
    R = resample_tracks_full(obst_df, grid)
    if R is None or not len(R):
        return None

    et = e.t_us.to_numpy(np.int64)
    EX = np.interp(grid, et, e.x.to_numpy(float))
    EY = np.interp(grid, et, e.y.to_numpy(float))
    EV = np.interp(grid, et, e.v.to_numpy(float))
    qs = np.stack([np.interp(grid, et, e[c].to_numpy(float)) for c in ("qx", "qy", "qz", "qw")], 1)
    PSI = np.arctan2(2 * (qs[:, 3] * qs[:, 2] + qs[:, 0] * qs[:, 1]),
                     1 - 2 * (qs[:, 1] ** 2 + qs[:, 2] ** 2))

    # ---- ego longitudinal acceleration from the native-rate SPEED series (the `ax` column's frame
    #      is not documented in features.csv, so it is not used) ----
    ts = et / 1e6
    vs = e.v.to_numpy(float)
    dt = float(np.median(np.diff(ts))) if len(ts) > 2 else 0.01
    n_half = max(2, int(round(0.5 / max(dt, 1e-4))))
    alon_raw = np.gradient(vs, ts)
    alon_ma = _movavg(alon_raw, n_half)                 # centred 0.5 s moving average
    alon_pre_series = np.convolve(np.pad(alon_raw, (n_half - 1, 0), mode="edge"),
                                  np.ones(n_half) / n_half, mode="valid")   # trailing 0.5 s mean
    ALON_MA = np.interp(grid, et, alon_ma)
    ALON_PRE = np.interp(grid, et, alon_pre_series)
    fut_idx = np.arange(nk)[:, None] + np.arange(1, H + 1)[None, :]
    ALON_FUT_MIN = ALON_MA[fut_idx].min(axis=1)          # min over (t, t+4 s]

    # ---- agent world state on the grid ----
    gi = {t: i for i, t in enumerate(grid)}
    R = R[R.t_us.isin(gi)].copy()
    if not len(R):
        return None
    ki = R.t_us.map(gi).to_numpy()
    cy_, sy_ = np.cos(PSI[ki]), np.sin(PSI[ki])
    WX = EX[ki] + cy_ * R.X.to_numpy() - sy_ * R.Y.to_numpy()
    WY = EY[ki] + sy_ * R.X.to_numpy() + cy_ * R.Y.to_numpy()
    WYAW = PSI[ki] + R.YAWR.to_numpy()

    # ---- projection: PER-CLIP (cx, cy) + PER-CLIP 6-DoF extrinsics (two-rig corpus) ----
    P3 = R[["X", "Y", "Z"]].to_numpy(float)
    u, vv, th = project(P3, K[FW])
    front_full = in_frame(u, vv, th, K[FW])
    in_crop = in_model_crop(u, vv, K[FW]) & front_full
    uu, vv2, tt = project(P3, K[CL])
    vis_L = in_frame(uu, vv2, tt, K[CL])
    uu, vv2, tt = project(P3, K[CR])
    vis_R = in_frame(uu, vv2, tt, K[CR])

    uniq, inv = np.unique(R.track_id.to_numpy(), return_inverse=True)
    nt = len(uniq)
    AX = np.full((nt, ng), np.nan)
    AY = np.full((nt, ng), np.nan)
    AYAW = np.full((nt, ng), np.nan)
    AX[inv, ki] = WX
    AY[inv, ki] = WY
    AYAW[inv, ki] = WYAW
    LA = np.zeros(nt)
    WA = np.zeros(nt)
    np.maximum.at(LA, inv, R.SX.to_numpy())
    np.maximum.at(WA, inv, R.SY.to_numpy())
    LA = np.clip(LA, 0.4, 20.0)
    WA = np.clip(WA, 0.4, 4.0)

    # agent velocity: central finite difference of the WORLD position over +-0.3 s
    AVX = np.full((nt, ng), np.nan)
    AVY = np.full((nt, ng), np.nan)
    w = VEL_HALFWIN
    AVX[:, w:ng - w] = (AX[:, 2 * w:] - AX[:, :ng - 2 * w]) / (2 * w * 0.1)
    AVY[:, w:ng - w] = (AY[:, 2 * w:] - AY[:, :ng - 2 * w]) / (2 * w * 0.1)

    # ---- cheap pre-filter: no conflict is reachable in 4 s ----
    kk_all = np.arange(nk)
    d0 = np.hypot(AX[:, :nk] - EX[:nk][None, :], AY[:, :nk] - EY[:nk][None, :])
    aspd = np.hypot(AVX[:, :nk], AVY[:, :nk])
    reach = TAU_H * (EV[:nk][None, :] + np.nan_to_num(aspd, nan=0.0)) + R_MAX_PAD
    cand = np.isfinite(d0) & np.isfinite(AVX[:, :nk]) & (d0 <= reach)
    ii, kk = np.nonzero(cand)
    if not len(ii):
        ii = np.zeros(0, int)
        kk = np.zeros(0, int)

    # ---- agent futures for the flagged pairs ----
    def agent_cv():
        return (AX[ii, kk][:, None] + AVX[ii, kk][:, None] * HS[None, :],
                AY[ii, kk][:, None] + AVY[ii, kk][:, None] * HS[None, :],
                np.repeat(AYAW[ii, kk][:, None], H, axis=1))

    def agent_real():
        fi = kk[:, None] + np.arange(1, H + 1)[None, :]
        return AX[ii[:, None], fi], AY[ii[:, None], fi], AYAW[ii[:, None], fi]

    EPX_ps, EPY_ps, EPP_ps, S_arc = ego_traj_pathspeed(EX, EY, PSI, EV, nk)
    EPX_cv, EPY_cv, EPP_cv = ego_traj_cv(EX, EY, PSI, EV, nk)

    variants = {}
    if len(ii):
        acx, acy, acyaw = agent_cv()
        arx, ary, aryaw = agent_real()
        La_, Wa_ = LA[ii], WA[ii]
        variants["ps_cv"], variants["msd"] = solve_areq(
            EPX_ps, EPY_ps, EPP_ps, kk, acx, acy, acyaw, La_, Wa_)
        variants["cv_cv"], _ = solve_areq(EPX_cv, EPY_cv, EPP_cv, kk, acx, acy, acyaw, La_, Wa_)
        variants["ps_real"], _ = solve_areq(EPX_ps, EPY_ps, EPP_ps, kk, arx, ary, aryaw, La_, Wa_)
        # L1_gate replication: ego CV (A = 0), agent REALISED, CENTRE-to-centre distance
        dl = np.full(len(kk), np.nan)
        for b0 in range(0, len(kk), PAIR_BLOCK):
            sl = slice(b0, min(b0 + PAIR_BLOCK, len(kk)))
            d_l1 = np.hypot(arx[sl] - EPX_cv[0][kk[sl]], ary[sl] - EPY_cv[0][kk[sl]])
            with np.errstate(invalid="ignore"):
                dl[sl] = np.nanmin(np.where(np.isnan(d_l1), np.inf, d_l1), axis=1)
        dl[np.isinf(dl)] = np.nan
        variants["dmin_l1"] = dl
    else:
        for kname in ("ps_cv", "cv_cv", "ps_real", "dmin_l1", "msd"):
            variants[kname] = np.zeros(0)

    # ---- per-(agent, frame) visibility for the flagged pairs ----
    VIS_CROP = np.zeros((nt, ng), bool)
    VIS_FULL = np.zeros((nt, ng), bool)
    VIS_L = np.zeros((nt, ng), bool)
    VIS_R = np.zeros((nt, ng), bool)
    VIS_CROP[inv, ki] = in_crop
    VIS_FULL[inv, ki] = front_full
    VIS_L[inv, ki] = vis_L
    VIS_R[inv, ki] = vis_R
    CLS = pd.Series(R.cls.to_numpy()).groupby(inv).first().reindex(range(nt)).to_numpy()

    p_crop = VIS_CROP[ii, kk]
    p_full = VIS_FULL[ii, kk]
    p_L = VIS_L[ii, kk] & ~p_crop
    p_R = VIS_R[ii, kk] & ~p_crop
    p_Lr = VIS_L[ii, kk] & ~p_full        # genuine off-front residual (E0's 36.4 %)
    p_Rr = VIS_R[ii, kk] & ~p_full

    # ---- aggregate to per-FRAME maxima ----
    def agg_max(vals, mask):
        out = np.zeros(nk)
        if mask.any():
            np.maximum.at(out, kk[mask], vals[mask])
        return out

    def agg_min(vals, mask):
        out = np.full(nk, np.inf)
        if mask.any():
            np.minimum.at(out, kk[mask], vals[mask])
        return out

    def agg_sum(vals, mask):
        out = np.zeros(nk)
        if mask.any():
            np.add.at(out, kk[mask], vals[mask])
        return out

    def agg_argcls(vals, mask):
        out = np.array([""] * nk, dtype=object)
        best = np.full(nk, -1.0)
        if mask.any():
            order = np.argsort(vals[mask])
            idx = np.nonzero(mask)[0][order]
            best[kk[idx]] = vals[idx]
            out[kk[idx]] = CLS[ii[idx]]
        return out, best

    A = variants["ps_cv"]
    # RESOLVABLE-only variant: max over agents whose conflict braking can actually resolve
    # (a_req < 8 sentinel). Needed because `max` over all agents lets one unresolvable close-pass
    # geometry mask a real, brakeable conflict on the same frame.
    A_RES = np.where(A < A_GRID[-1], A, 0.0)
    unres_flag = (A >= A_GRID[-1]).astype(float)
    cols = dict(
        gi=kk_all,
        areq_off_L=agg_max(A, p_L), areq_off_R=agg_max(A, p_R),
        areq_off_Lr=agg_max(A, p_Lr), areq_off_Rr=agg_max(A, p_Rr),
        areq_seen=agg_max(A, p_crop),
        areq_off_L_res=agg_max(A_RES, p_L), areq_off_R_res=agg_max(A_RES, p_R),
        areq_off_Lr_res=agg_max(A_RES, p_Lr), areq_off_Rr_res=agg_max(A_RES, p_Rr),
        areq_seen_res=agg_max(A_RES, p_crop),
        n_unres_off=agg_sum(unres_flag, p_L | p_R),
        areq_cv_off_L=agg_max(variants["cv_cv"], p_L),
        areq_cv_off_R=agg_max(variants["cv_cv"], p_R),
        areq_cv_seen=agg_max(variants["cv_cv"], p_crop),
        areq_re_off_L=agg_max(variants["ps_real"], p_L),
        areq_re_off_R=agg_max(variants["ps_real"], p_R),
        areq_re_seen=agg_max(variants["ps_real"], p_crop),
    )
    dl1 = np.nan_to_num(variants["dmin_l1"], nan=1e9)
    cols["dmin_l1_off"] = agg_min(dl1, p_L | p_R)
    cols["dmin_l1_seen"] = agg_min(dl1, p_crop)
    msd = np.nan_to_num(variants["msd"], nan=1e9)
    cols["msd_off"] = agg_min(msd, p_L | p_R)
    cols["msd_seen"] = agg_min(msd, p_crop)
    clsL, _ = agg_argcls(A_RES, p_L)
    clsR, _ = agg_argcls(A_RES, p_R)
    cols["cls_L"] = clsL
    cols["cls_R"] = clsR

    present = np.zeros(nk)
    kv, kc = np.unique(ki[ki < nk], return_counts=True)
    present[kv] = kc

    yaw_u = np.unwrap(PSI)
    junc, lane = situation_masks(EX, EY, yaw_u, EV, grid / 1e6)

    out = pd.DataFrame(cols)
    out["ego_v"] = EV[:nk]
    out["alon_pre"] = ALON_PRE[:nk]
    out["alon_fut_min"] = ALON_FUT_MIN
    out["ego_dv4"] = EV[H:H + nk] - EV[:nk]
    out["n_agents"] = present
    out["junction"] = junc[:nk]
    out["lane_change"] = lane[:nk]
    out["clip_id"] = clip
    out["chunk"] = ch
    return out[out.n_agents > 0]        # same denominator convention as the E0/E1 tables


def main(out_dir, chunks):
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()
    for ch in chunks:
        dst = os.path.join(out_dir, f"l2_{ch}.parquet")
        if os.path.exists(dst):
            print(f"[{ch}] exists -- skip", flush=True)
            continue
        zp = DR + rf"\labels\obstacle.offline\obstacle.offline.chunk_{ch}.zip"
        if not os.path.exists(zp) or not os.path.exists(
                DR + rf"\labels\egomotion\egomotion.chunk_{ch}.zip"):
            print(f"[{ch}] missing artifact -- skip", flush=True)
            continue
        z = zipfile.ZipFile(zp)
        parts, n_ok, n_try = [], 0, 0
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
            except Exception:                                    # noqa: BLE001
                continue
            if not len(df):
                continue
            try:
                r = run_clip(clip, ch, df, K)
            except Exception as exc:                             # noqa: BLE001
                print(f"   [{clip}] FAILED: {type(exc).__name__}: {exc}", flush=True)
                continue
            if r is None or not len(r):
                continue
            parts.append(r)
            n_ok += 1
        if not parts:
            print(f"[chunk {ch}] no clips", flush=True)
            continue
        Adf = pd.concat(parts, ignore_index=True)
        Adf.to_parquet(dst)
        print(f"[chunk {ch}] clips {n_ok}/{n_try}  frames {len(Adf):,}  "
              f"t={time.time() - t_start:.0f}s", flush=True)
    print(f"DONE  t={time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
