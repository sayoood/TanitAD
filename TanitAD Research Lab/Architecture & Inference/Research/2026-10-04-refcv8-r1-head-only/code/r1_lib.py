"""refcv8 R1 (head-only, frozen trunk) -- pure numpy helpers shared by the capture, the arms and the scoring.

No torch at import, no model, no launch-tree import. Every threshold is a LITERAL written here before any R1 number
existed (2026-10-04); the GT-turn rule is the route package's SPEC.md sec. 3 rule (copied from its
`code/route_metrics.py`, md5 in LANDING_READY_R1.txt), so R1 classes windows exactly as RESULT.md / A5 / A6 do.

Frame: ego at NOW, x forward, y LEFT, metres; paths are [..., 8, 2] at the slots (5,10,15,20,30,40,50,60) ticks of
0.1 s (0.5 ... 6.0 s). Prints / stores sha12 only (sha256(clip_id)[:12]); a raw clip id is never written.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
SLOT_T_S = np.array([h * 0.1 for h in HORIZONS])
STALL_M = 0.05
# SPEC.md sec. 3 (route package) -- the GT-turn rule every route number uses
TURN_DEG, STRAIGHT_DEG, HEAD_AGREE_DEG = 30.0, 10.0, 15.0
MIN_LEN_M = 5.0
TAU_C = 0.18063741505146028            # the run's own --nav-compliance-tau-rad (10.35 deg): candidate direction class

# ---- the v7 tactical vocabulary (vocab_v7.py:320-331 at fec3a0d; asserted against the tree at run time) ----------- #
LAT_V7 = ("LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC", "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R")
LON_V7 = ("FOLLOW", "CRUISE", "YIELD_MERGE", "BRAKE_TO", "CREEP", "HOLD", "ADAPT_SPEED_FOR_CURVE", "ACCELERATE")
IGNORE_ID = -100

# ---- R1 dense labels (H2) -- LITERALS, fixed 2026-10-04 before any R1 arm was scored ------------------------------ #
#   lateral, 3-way, from the ego's OWN next 6 s (the 8 GT slots):
#     segment headings th_k = atan2(dy, dx) of the 8 consecutive segments origin->slot5->...->slot60 (ego frame);
#     TURN_L if max_k th_k >= +30 deg, TURN_R if min_k th_k <= -30 deg (both: the larger |th|); else LANE_KEEP.
#     IGNORE when slot 60 is not valid; LANE_KEEP when the 6-s path is shorter than 5 m (no turn is identifiable).
#   longitudinal, one declared window [NOW, NOW + 6.0 s] on the 0.1-s speed channel (pose v):
#     HOLD if max v < 0.5 m/s; CREEP if max v <= 2.0 m/s; ACCELERATE if v(6 s) - v(0) >= +1.5 m/s;
#     BRAKE_TO if v(6 s) - v(0) <= -1.5 m/s; else CRUISE.  IGNORE when the 6-s future is not valid.
#   (+1.5 m/s is the DOCUMENTED bar; D2 measured the builder applying +1.0 -- K11.)
DENSE_TURN_DEG = 30.0
DENSE_MIN_LEN_M = 5.0
DENSE_STOP_MS = 0.5
DENSE_CRAWL_MS = 2.0
DENSE_DV_MS = 1.5
DENSE_LON_WINDOW_STEPS = 60            # 6.0 s at 0.1 s


def sha12(clip_id: str) -> str:
    return hashlib.sha256(str(clip_id).encode()).hexdigest()[:12]


def wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------------------------------------- #
# geometry (route_metrics.py, SPEC.md sec. 3)                                                         #
# ------------------------------------------------------------------------------------------------- #
def terminal_heading(P):
    """theta of the slot-50 -> slot-60 segment; 0.0 when it is shorter than 0.05 m (compliance_target)."""
    P = np.asarray(P, np.float64)
    d = P[..., -1, :] - P[..., -2, :]
    th = np.arctan2(d[..., 1], d[..., 0])
    return np.where(np.hypot(d[..., 0], d[..., 1]) < STALL_M, 0.0, th)


def dir_class(theta, tau=TAU_C):
    th = np.asarray(theta, np.float64)
    return np.where(th >= tau, 1, np.where(th <= -tau, -1, 0)).astype(np.int8)


def path_length(P):
    P = np.asarray(P, np.float64)
    Q = np.concatenate([np.zeros(P.shape[:-2] + (1, 2)), P], axis=-2)
    return np.linalg.norm(np.diff(Q, axis=-2), axis=-1).sum(-1)


def gt_class(gt, gt_valid):
    """-> (cls, theta): 'turnL','turnR','straight','gentle','unclassified' per the route package SPEC sec. 3."""
    th = terminal_heading(gt)
    ln = path_length(gt)
    ok = np.asarray(gt_valid)[..., -1] & (ln >= MIN_LEN_M)
    deg = np.degrees(np.abs(th))
    cls = np.full(th.shape, "unclassified", dtype=object)
    cls[ok & (deg >= TURN_DEG) & (th > 0)] = "turnL"
    cls[ok & (deg >= TURN_DEG) & (th < 0)] = "turnR"
    cls[ok & (deg < STRAIGHT_DEG)] = "straight"
    cls[ok & (deg >= STRAIGHT_DEG) & (deg < TURN_DEG)] = "gentle"
    return cls, th


def ade(P, gt, gt_valid):
    P = np.asarray(P, np.float64)
    gt = np.asarray(gt, np.float64)
    v = np.asarray(gt_valid, bool)
    if P.ndim == gt.ndim + 1:
        gt = gt[:, None]
        v = v[:, None]
    d = np.linalg.norm(P - gt, axis=-1)
    vv = np.broadcast_to(v, d.shape)
    n = vv.sum(-1)
    return np.where(n > 0, (d * vv).sum(-1) / np.maximum(n, 1), np.nan)


def ego_waypoints(pose_last, fut, horizons=HORIZONS):
    """numpy twin of refb_labels.waypoint_targets (ego_frame of xy[t+k] - xy[t] by -yaw[t]); pose [.., 4] = x, y, yaw, v.
    Used only for WINDOW SELECTION (CPU); every scored GT is read from the capture, which uses the tree's own function."""
    pl = np.asarray(pose_last, np.float64)
    f = np.asarray(fut, np.float64)
    yaw = pl[..., 2]
    c, s = np.cos(-yaw), np.sin(-yaw)
    out = []
    for k in horizons:
        d = f[..., k - 1, :2] - pl[..., :2]
        out.append(np.stack([d[..., 0] * c - d[..., 1] * s, d[..., 0] * s + d[..., 1] * c], -1))
    return np.stack(out, -2)


def episode_window_gt(poses, window, max_h=60):
    """All windows t of one episode (the V3Dataset contract, refc_v3_train.py:3808-3816: NOW row = t + w - 1,
    future = rows t + w ... t + w + 59 clamped, valid = row <= T - 1) -> (gt [n, 8, 2], gt_valid [n, 8]).
    t runs over 0 .. T - w (the base index is filtered by the dataset; callers intersect with e_ds.index)."""
    P = np.asarray(poses, np.float64)
    T = P.shape[0]
    n = T - window + 1
    ts = np.arange(n)
    now = ts + window - 1
    idx = now[:, None] + 1 + np.arange(max_h)[None]
    valid = idx <= T - 1
    fut = P[np.clip(idx, 0, T - 1)]
    gt = ego_waypoints(P[now], fut)
    gv = np.stack([valid[:, h - 1] for h in HORIZONS], 1)
    return ts, gt, gv


# ------------------------------------------------------------------------------------------------- #
# H2 dense labels                                                                                     #
# ------------------------------------------------------------------------------------------------- #
def dense_lat(gt, gt_valid):
    """[N, 8, 2] GT slots -> lat id in the v7 vocabulary (LANE_KEEP 0 / TURN_L 6 / TURN_R 7) or IGNORE_ID."""
    gt = np.asarray(gt, np.float64)
    gv = np.asarray(gt_valid, bool)
    Q = np.concatenate([np.zeros(gt.shape[:-2] + (1, 2)), gt], axis=-2)
    d = np.diff(Q, axis=-2)
    seglen = np.hypot(d[..., 0], d[..., 1])
    th = np.where(seglen < STALL_M, 0.0, np.arctan2(d[..., 1], d[..., 0]))
    mx, mn = th.max(-1), th.min(-1)
    t = math.radians(DENSE_TURN_DEG)
    left = mx >= t
    right = mn <= -t
    out = np.full(gt.shape[:-2], LAT_V7.index("LANE_KEEP"), np.int64)
    both = left & right
    out[left & ~both] = LAT_V7.index("TURN_L")
    out[right & ~both] = LAT_V7.index("TURN_R")
    out[both & (mx >= -mn)] = LAT_V7.index("TURN_L")
    out[both & (mx < -mn)] = LAT_V7.index("TURN_R")
    short = path_length(gt) < DENSE_MIN_LEN_M
    out[short] = LAT_V7.index("LANE_KEEP")
    out[~gv[..., -1]] = IGNORE_ID
    return out


def dense_lon(v_now, v_future, fut_valid):
    """v_now [N], v_future [N, 60] (0.1 s), fut_valid [N, 60] -> lon id (HOLD/CREEP/ACCELERATE/BRAKE_TO/CRUISE) or IGNORE."""
    v0 = np.asarray(v_now, np.float64)
    vf = np.asarray(v_future, np.float64)[:, :DENSE_LON_WINDOW_STEPS]
    ok = np.asarray(fut_valid, bool)[:, DENSE_LON_WINDOW_STEPS - 1]
    vmax = np.maximum(vf.max(-1), v0)
    dv = vf[:, -1] - v0
    out = np.full(v0.shape, LON_V7.index("CRUISE"), np.int64)
    out[dv >= DENSE_DV_MS] = LON_V7.index("ACCELERATE")
    out[dv <= -DENSE_DV_MS] = LON_V7.index("BRAKE_TO")
    out[vmax <= DENSE_CRAWL_MS] = LON_V7.index("CREEP")
    out[vmax < DENSE_STOP_MS] = LON_V7.index("HOLD")
    out[~ok] = IGNORE_ID
    return out


def shuffle_within_clip(labels, clip, seed=0):
    """H2s: each clip's label sequence permuted among that clip's own windows (seeded); a clip with one window keeps
    it (reported). Returns the shuffled array and the fraction of windows whose label CHANGED."""
    labels = np.asarray(labels)
    clip = np.asarray(clip).astype(str)
    out = labels.copy()
    rng = np.random.default_rng(seed)
    for c in sorted(set(clip.tolist())):
        ix = np.nonzero(clip == c)[0]
        if len(ix) > 1:
            out[ix] = labels[ix[rng.permutation(len(ix))]]
    return out, float((out != labels).mean())


# ------------------------------------------------------------------------------------------------- #
# H3 nav: A5's nav_tl rule restricted to ANNOUNCED entries (SPEC_ADDENDUM_A6)                          #
# ------------------------------------------------------------------------------------------------- #
ANCHOR_S = 8.0
NAV_H_S = 6.0
TOKEN_SIDE = {"NAV_TURN_L": 1, "NAV_TURN_R": -1}
NAV_INDEX = {"follow": 0, "left": 1, "right": 2, "straight": 3}     # refc.NAV_COMMANDS (asserted at run time)
SIDE_TO_NAV = {0: 0, 1: 1, -1: 2}


def nav_ann_state(entries, flags, t_rel, H=NAV_H_S):
    """A5 nav_tl_state with a non-announced entry SKIPPED (A6): the first not-finished ANNOUNCED turn entry; side if
    its start is <= H ahead (or under way), else follow. A FOLLOW_ROAD entry ends the search with follow, as in A5."""
    for e, ok in zip(entries, flags):
        if e["t_end_s"] > t_rel:
            side = TOKEN_SIDE.get(e["token"], 0)
            if side == 0:
                return 0, "follow_token"
            if not ok:
                continue
            lead = e["t_start_s"] - t_rel
            if lead <= H:
                return side, ("under_way" if lead <= 0 else "ahead_within_H")
            return 0, "beyond_H"
    return 0, "no_entry_left"


# ------------------------------------------------------------------------------------------------- #
# storage codec for the BEV map (int8 affine, per window x channel)                                    #
# ------------------------------------------------------------------------------------------------- #
def quant_u8(x):
    """x [N, C, H, W] float -> (q uint8, lo [N, C] f32, scale [N, C] f32); dequant = lo + q * scale."""
    import torch
    x = x.float()
    lo = x.amin(dim=(2, 3))
    hi = x.amax(dim=(2, 3))
    scale = ((hi - lo) / 255.0).clamp_min(1e-12)
    q = torch.round((x - lo[..., None, None]) / scale[..., None, None]).clamp(0, 255).to(torch.uint8)
    return q, lo, scale


def quant_u8_row(x):
    """x [N, C, X, Y] float -> (q uint8, lo [N, C, X] f32, scale [N, C, X] f32): int8 affine per (channel, X row).
    The R1 production BEV codec since 2026-10-04 (codec study `raw/codec_study_gpu.json`: fan p99 0.028 m, 0 / 192
    differing picks; the per-channel codec read 0.059 m and FAILED I-S's 0.05 m bar)."""
    x = x.float()
    lo = x.amin(dim=3)
    hi = x.amax(dim=3)
    scale = ((hi - lo) / 255.0).clamp_min(1e-12)
    q = torch_round_clamp((x - lo[..., None]) / scale[..., None])
    return q, lo, scale


def torch_round_clamp(z):
    import torch
    return torch.round(z).clamp(0, 255).to(torch.uint8)


def dequant_u8(q, lo, scale):
    """per-channel (lo [N, C]) or per-(channel, row) (lo [N, C, X]) affine int8 -> float."""
    if lo.dim() == 3:
        return lo[..., None] + q.float() * scale[..., None]
    return lo[..., None, None] + q.float() * scale[..., None, None]
