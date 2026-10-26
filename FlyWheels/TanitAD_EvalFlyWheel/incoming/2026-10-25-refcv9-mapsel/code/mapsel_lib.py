"""mapsel_lib.py -- refcv9 MAP-SELECT (PREREG_MAPSEL.md): the pure functions. CPU torch/numpy only; no model, no GPU, no file IO.

FRAME / GRID (the 10 cm map head's, `semantic_map_gt_fine` / `refcv8_conditioning._drv_cells`): ego frame at t0, x forward, y LEFT; raster [H=1000 rows along x from 0..100 m,
W=600 cols along y from -30 m (RIGHT) .. +30 m]; cell (i, j) <-> x in [0.1 i, 0.1 (i+1)), y in [-30 + 0.1 j, -30 + 0.1 (j+1)).  GT codes: 0 nocls 1 drivable 2 lane 3 crosswalk
4 arrow 5 edge 6 hatched 7 sidewalk, 255 = not seen.  DRIVABLE = (1, 2, 3, 4, 6) (refcv8_conditioning.DRIVABLE_CODES: markings lie ON the road surface); AMENDMENT 1 (PREREG_MAPSEL_AMENDMENT1.md): the T1 VIOLATION is a check point on a SEEN SIDEWALK
cell (code 7) only -- nocls (0) / not seen (255) are UNKNOWN (reported as COVERAGE); the own-map mirror is P^(sidewalk) >= 128/255.

PATH: the plan is 8 REAR-AXLE slots at t = 0.5, 1, 1.5, 2, 3, 4, 5, 6 s.  The path is the polyline origin -> slot 0 -> ... -> slot 7, sampled CONTINUOUSLY every 0.25 m of path length
from the 0.5 s slot (the t >= 0.5 s rule) to the 6 s slot (the last slot is always sampled).  Heading = the segment's direction (a segment < 5 cm keeps the previous heading,
the first falls back to 0 = straight ahead; `refcv8_conditioning.plan_headings`).  FOOTPRINT = nuPlan's 5.176 x 2.297 m box, centre 1.461 m ahead of the rear axle; the check points
are its 4 corners + the centre (FP5), exactly `drivable_feature`'s set.  A point outside the raster is UNCHECKED (unknown passes), as in D6-P4.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

CELL_M, H, W, Y_HALF_M = 0.1, 1000, 600, 30.0
DRV_CODES, LANE_CODE, EDGE_CODE, NOT_SEEN = (1, 2, 3, 4, 6), 2, 5, 255
SIDEWALK_CODE, NOCLS_CODE = 7, 0
BAD_CODES = (7,)            # AMENDMENT 1: T1 off-drivable = a check point on a SEEN SIDEWALK cell (the known non-drivable class); nocls (0) and not-seen (255) are UNKNOWN (never a violation)
EGO_L, EGO_W, REAR_TO_CTR = 5.176, 2.297, 1.461            # pdm_proxy.PROXY (pinned by a unit test against the module)
STEP_M, NS_MAX = 0.25, 800
EDGE_DIL, LANE_DIL, PDRV_MIN_U8, PSW_MIN_U8 = 3, 2, 128, 128   # edge margin 0.3 m, lane band +-0.2 m; PDRV_MIN_U8 only for the K_LIVE liveness; OWN violation: P^(sidewalk) >= 128/255 (~0.502)
STALL_M = 0.05
SLOT_T = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
HUMAN_STEPS = (4, 9, 14, 19, 29, 39, 49, 59)                 # 10 Hz future frame index of each slot time (verified against the banked S6 `g`)


def path_samples(plans):
    """plans [N, 8, 2] (torch float32, ego frame) -> (pts [N, S, 2], heading [N, S], valid [N, S]); S = NS_MAX + 1 (the last column = the final slot, always valid)."""
    plans = torch.as_tensor(plans, dtype=torch.float32)
    n = plans.shape[0]
    P = torch.cat([plans.new_zeros(n, 1, 2), plans], 1)                      # [N, 9, 2]
    seg = P[:, 1:] - P[:, :-1]                                               # [N, 8, 2]
    ln = torch.linalg.vector_norm(seg, dim=-1)                              # [N, 8]
    cum = torch.cat([ln.new_zeros(n, 1), ln.cumsum(1)], 1)                   # [N, 9]
    th = torch.atan2(seg[..., 1], seg[..., 0])
    ok = ln >= STALL_M
    prev, hs = th.new_zeros(n), []
    for j in range(8):
        prev = torch.where(ok[:, j], th[:, j], prev)
        hs.append(prev)
    hseg = torch.stack(hs, 1)                                                # [N, 8]
    s0, s1 = cum[:, 1], cum[:, 8]
    k = torch.arange(NS_MAX, dtype=torch.float32)
    s = torch.cat([s0[:, None] + STEP_M * k[None], s1[:, None]], 1)         # [N, S]
    valid = torch.cat([s[:, :NS_MAX] <= s1[:, None] + 1e-6, torch.ones(n, 1, dtype=torch.bool)], 1)
    idx = torch.searchsorted(cum[:, 1:].contiguous(), s.contiguous()).clamp(0, 7)   # segment j: cum[j] < s <= cum[j+1]
    ar = torch.arange(n)[:, None]
    frac = (s - cum[ar, idx]) / ln[ar, idx].clamp_min(1e-9)
    frac = torch.where(ln[ar, idx] > 1e-9, frac.clamp(0.0, 1.0), torch.zeros_like(frac))
    pts = P[ar, idx] + frac[..., None] * seg[ar, idx]
    return pts, hseg[ar, idx], valid


def footprint5(pts, hd):
    """pts [N, S, 2] rear-axle points, hd [N, S] -> [N, S, 5, 2]: front-left, rear-left, rear-right, front-right corners, then the box centre (index 4)."""
    c, s = torch.cos(hd), torch.sin(hd)
    cx, cy = pts[..., 0] + REAR_TO_CTR * c, pts[..., 1] + REAR_TO_CTR * s
    hl, hw = EGO_L / 2.0, EGO_W / 2.0
    sx = torch.tensor([hl, -hl, -hl, hl, 0.0])
    sy = torch.tensor([hw, hw, -hw, -hw, 0.0])
    x = cx[..., None] + sx * c[..., None] - sy * s[..., None]
    y = cy[..., None] + sx * s[..., None] + sy * c[..., None]
    return torch.stack([x, y], -1)


def to_cells(xy):
    """[..., 2] metres -> (ix, iy clamped long, inside bool)."""
    ix = torch.floor(xy[..., 0] / CELL_M).long()
    iy = torch.floor((xy[..., 1] + Y_HALF_M) / CELL_M).long()
    inside = (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W)
    return ix.clamp(0, H - 1), iy.clamp(0, W - 1), inside


def dilate(mask, r):
    """bool [H, W] -> bool [H, W], square structuring element of half-width r cells (r = 0: unchanged). Separable shifted-OR (about 40x faster than max_pool2d on CPU)."""
    m = torch.as_tensor(mask).bool()
    if r <= 0:
        return m
    a = m.clone()
    for s in range(1, r + 1):
        a[s:] |= m[:-s]
        a[:-s] |= m[s:]
    b = a.clone()
    for s in range(1, r + 1):
        b[:, s:] |= a[:, :-s]
        b[:, :-s] |= a[:, s:]
    return b


def masks_gt(codes):
    """GT codes uint8 [H, W] -> dict of bool [H, W]: bad (T1 violation: seen cell in BAD_CODES = sidewalk), edge (dilated EDGE_DIL), lane (dilated LANE_DIL), nocls, unseen (coverage)."""
    c = torch.as_tensor(np.asarray(codes))
    seen = c != NOT_SEEN
    bad = torch.zeros_like(seen)
    for k in BAD_CODES:
        bad |= c == k
    return {"bad": bad & seen, "edge": dilate((c == EDGE_CODE) & seen, EDGE_DIL), "lane": dilate((c == LANE_CODE) & seen, LANE_DIL), "nocls": c == NOCLS_CODE, "unseen": ~seen}


def masks_own(psw_u8, code_u8):
    """OWN planes (P^(sidewalk) as uint8, decided class code uint8 under the declared prior-corrected rule) -> the same dict (violation: P^(sidewalk) >= 128/255; edge / lane from the decided code)."""
    p, c = torch.as_tensor(np.asarray(psw_u8)), torch.as_tensor(np.asarray(code_u8))
    return {"bad": p >= PSW_MIN_U8, "edge": dilate(c == EDGE_CODE, EDGE_DIL), "lane": dilate(c == LANE_CODE, LANE_DIL), "nocls": c == NOCLS_CODE, "unseen": torch.zeros_like(c, dtype=torch.bool)}


class Geom:
    """The candidate geometry of ONE window, computed once and reused for every raster (GT / OWN / DER / human)."""

    def __init__(self, plans):
        pts, hd, valid = path_samples(plans)
        fp = footprint5(pts, hd)                                             # [N, S, 5, 2]
        self.ix, self.iy, ins = to_cells(fp)
        self.inside = ins & valid[..., None]                                 # [N, S, 5]
        self.n = plans.shape[0]
        self.n_samples = valid.sum(1)

    def flags(self, m):
        """per-candidate bool [N] x3 (off_drv, edge_contact, lane_cross) under the mask dict m; lane = the CENTRE point (centre line) only."""
        ix, iy, ins = self.ix, self.iy, self.inside
        off = (ins & m["bad"][ix, iy]).flatten(1).any(1)
        edge = (ins & m["edge"][ix, iy]).flatten(1).any(1)
        lane = (ins[..., 4] & m["lane"][ix[..., 4], iy[..., 4]]).flatten(1).any(1)
        return off, edge, lane

    def own_cont(self, psw_u8, pedge_u8):
        """continuous OWN features per candidate [N]: max P^(sidewalk), mean P^(sidewalk), max P^(edge) over the in-range FP5 points (0, 0, 0 with none in range)."""
        ix, iy, ins = self.ix, self.iy, self.inside
        ps = torch.as_tensor(np.asarray(psw_u8))[ix, iy].float() / 255.0
        pe = torch.as_tensor(np.asarray(pedge_u8))[ix, iy].float() / 255.0
        cnt = ins.flatten(1).sum(1).float()
        pmax = torch.where(ins, ps, torch.zeros_like(ps)).flatten(1).amax(1)
        pmean = torch.where(cnt > 0, (ps * ins).flatten(1).sum(1) / cnt.clamp_min(1.0), torch.zeros_like(cnt))
        pemax = torch.where(ins, pe, torch.zeros_like(pe)).flatten(1).amax(1)
        return pmax, pmean, pemax

    def coverage(self, m):
        """COVERAGE per candidate [N]: the share of the IN-RANGE FP5 samples that land on nocls cells and on not-seen cells (GT masks); (0, 0) with none in range."""
        ix, iy, ins = self.ix, self.iy, self.inside
        cnt = ins.flatten(1).sum(1).float().clamp_min(1.0)
        return (ins & m["nocls"][ix, iy]).flatten(1).sum(1).float() / cnt, (ins & m["unseen"][ix, iy]).flatten(1).sum(1).float() / cnt


def tier_pick(score, elig, off, edge, lane, tier, lc_window=False):
    """The PI's three tiers as lexicographic filters with fallbacks over the E9-ELIGIBLE set.  score [N] float, elig/off/edge/lane [N] bool (numpy), tier in {0, 1, 2, 3}
    (0 = the native pick).  -> (pick, [fallback_t1, fallback_t2, fallback_t3], n_kept).  A tier whose filter would empty the set is SKIPPED (fallback = 1); T3 is skipped
    outright when the window is a lane-change window (not counted as a fallback)."""
    S = np.asarray(elig, bool).copy()
    fb = [0, 0, 0]
    for t, bad in ((1, off), (2, edge), (3, lane)):
        if tier < t:
            break
        if t == 3 and lc_window:
            continue
        S2 = S & ~np.asarray(bad, bool)
        if S2.any():
            S = S2
        else:
            fb[t - 1] = 1
    sc = np.where(S, np.asarray(score, np.float64), -np.inf)
    return int(np.argmax(sc)), fb, int(S.sum())


def eligible(score, reach, ceil, emit):
    """EXACT numpy replica of refc_v3.e9_rank's finite mask. score [n,N]; masks [n,N] int8 with -1 = ABSENT (None semantics)."""
    fin = np.isfinite(score)

    def has(m):
        return m is not None and not (np.asarray(m) == -1).all()
    f0 = fin & (reach.astype(bool) if has(reach) else True)
    f1 = f0 & ceil.astype(bool) if has(ceil) else f0
    dead = ~f1.any(1)
    f1 = np.where(dead[:, None], f0, f1)
    if has(emit):
        eb = emit.astype(bool)
        f2 = f1 & eb
        dead2 = ~f2.any(1)
        return np.where(dead2[:, None], fin & eb, f2)
    return f1
