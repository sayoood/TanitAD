"""channels_d2.py -- PREREG_CHANNELS_D2 (sealed 2026-10-26 08:00:56): curve geometry (CURVE) and a crosswalk ahead (XWALK-OWN / XWALK-GT) as decel
information channels.  DIAGNOSTIC on logged eval139 labels and the planner's own logged outputs; no planner is run, no bar for any deliverable.
Data, target and estimator are IDENTICAL to leadinfo_d.py (imported, not copied): S6 compliant windows, l2_common.lon_indicators populations,
y = 1[GT decel], A = softmax(6 LON6 logits) + v0, L2 logistic C=1, 5-fold episode-grouped cross-fit (fold seed 0), AUC of the out-of-fold P(decel),
episode-RESAMPLING bootstrap B 2000 seed 0 with the out-of-fold predictions fixed.  Each channel = A + its block, with its own shuffled control (the block
permuted across EPISODES, seed 0); a channel's windows are those it can cover and ITS A is re-fit on the same windows.

Order of work (each stage asserts before the next):
  0. both prereg hashes verified (D2 and LEADINFO_D); 1. synthetic controls through the PRODUCTION functions (analytic curvature targets; a crosswalk patch at
  x=20 m, y=0 found at distance 20, built with the SOURCE's own mapsel_lib.to_cells); 2. identity with LEADINFO_D (A's out-of-fold predictions reproduced
  bit-for-bit on LEADINFO's windows); 3. features; 4. real convention controls on the rasters (the human path lands on drivable cells only under the
  declared convention, not mirrored / transposed); 5. the three channels and the words.

Run:  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 python -X utf8 channels_d2.py   (PYTHONPATH as in the brief).   No clip id is printed or written.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path("C:/Users/Admin/refcv8_pf/refcv9_leadinfo")
sys.path.insert(0, str(HERE / "code"))
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_mapsel/code")
import leadinfo_d as LD  # noqa: E402  (the LEADINFO_D estimator, imported not copied)

RES, SCR = HERE / "result", LD.SCR
PREREG, PREREG_SHA = HERE / "PREREG_CHANNELS_D2.md", HERE / "PREREG_CHANNELS_D2.sha256"
RAST = Path("F:/refcv9_mapsel/rast_s0")
TRACE = Path("F:/refcv9_mapsel/trace_s0.npz")
log, pr, ci = LD.log, LD.pr, LD.ci

# ---- the raster convention (SOURCE: refcv9_mapsel/code/mapsel_lib.py:3-4 header, :19 CELL_M/H/W/Y_HALF_M, :70 to_cells; semantic_map_gt_fine.py:295 EXTENT_REFCV7 = 100 x 30) ----
CELL_M, H, W, Y_HALF_M = 0.1, 1000, 600, 30.0
# ---- the corridor (prereg s2): |y| <= 1.8 m, 4 <= x <= 40 m, tested on CELL CENTRES: rows 40..399 (x in [4.0, 40.0)), cols 282..317 (centre y in [-1.75, +1.75]) ----
X_LO_M, X_HI_M, Y_ABS_M = 4.0, 40.0, 1.8
R0, R1 = int(round(X_LO_M / CELL_M)), int(round(X_HI_M / CELL_M))
ROWS = np.arange(H)
COLS = np.arange(W)
_cy = -Y_HALF_M + CELL_M * (COLS + 0.5)
C0, C1 = int(COLS[np.abs(_cy) <= Y_ABS_M + 1e-9][0]), int(COLS[np.abs(_cy) <= Y_ABS_M + 1e-9][-1]) + 1
# ---- class ids FROM SOURCE (asserted at run time against the tree's own CLASS_KEYS) ----
DRV_CODES, NOT_SEEN = (1, 2, 3, 4, 6), 255
KAPPA_FLOOR, MERGE_M, A_LAT_COMF = 1e-3, 0.5, 2.0
DELTA_BAR, CTRL_MAX = 0.03, 0.01


def class_ids_from_source() -> dict:
    """crosswalk / edge ids read from tanitad.models.map_head_hires.CLASS_KEYS (map_head_hires.py:998-999) and cross-checked against mapsel_lib.EDGE_CODE."""
    from tanitad.models import map_head_hires as MH
    import mapsel_lib as ML
    ck = tuple(MH.CLASS_KEYS)
    ids = {"crosswalk": ck.index("crosswalk"), "edge": ck.index("edge"), "sidewalk": ck.index("sidewalk"), "drivable": ck.index("drivable")}
    assert ids["crosswalk"] == 3 and ids["edge"] == 5 and ids["edge"] == ML.EDGE_CODE and tuple(ML.DRV_CODES) == DRV_CODES and ML.NOT_SEEN == NOT_SEEN
    assert (ML.CELL_M, ML.H, ML.W, ML.Y_HALF_M) == (CELL_M, H, W, Y_HALF_M)
    return {"class_keys": list(ck), **ids, "sources": ["stack/tanitad/models/map_head_hires.py:998-999 (CLASS_KEYS, code order 0..7)",
                                                     "stack/tanitad/data/semantic_map_gt_fine.py:116-117 (N_FINE_CLASSES = 8, NOT_SEEN_CODE = 255)",
                                                     "refcv9_mapsel/code/mapsel_lib.py:19-21 (CELL_M, H, W, Y_HALF_M; DRV_CODES, EDGE_CODE = 5)",
                                                     "refcv9_mapsel/code/mapsel_roll.py:33-35 (DRV_CODES, EDGE_CODE, SIDEWALK_CODE)"]}


def check_seal() -> dict:
    want = PREREG_SHA.read_text(encoding="utf-8").split()[0].strip().lower()
    got = LD.sha256_file(PREREG)
    if got != want or not want.startswith("35fb8d02"):
        raise SystemExit(f"STOP: PREREG_CHANNELS_D2 sha256 {got} != sealed {want}")
    return {"prereg_sha256": got, "sealed_equal": True, "leadinfo_d_seal": LD.check_seal()}


# =====================================================================================================================
# CURVE
# =====================================================================================================================
CURVE_COLS = ("kappa_max", "alat_demand", "v_curve", "req_decel", "heading_change_abs", "dist_to_kmax")


def merge_short(P: np.ndarray, min_m: float = MERGE_M) -> np.ndarray:
    """Walk the polyline keeping a point only if it is >= min_m from the LAST KEPT point (segments < 0.5 m merged)."""
    keep = [P[0]]
    for p in P[1:]:
        if np.hypot(*(p - keep[-1])) >= min_m:
            keep.append(p)
    return np.asarray(keep)


def curve_block(plan8: np.ndarray, v0: float, signed_heading: bool = False) -> dict:
    """plan8 [8, 2] rear-axle slots (t = 0.5 .. 6 s) in the ego frame; the polyline is origin -> slot 0 -> ... -> slot 7, short segments merged.
    kappa_max : max over INTERIOR vertices of the three-point (Menger) curvature 2|cross| / (|a||b||c|); floored at 1e-3 (R = 1 km) so the derived quantities stay finite;
    alat_demand = v0^2 kappa_max;  v_curve = sqrt(2.0 / kappa_max);  req_decel = max(0, v0 - v_curve);
    heading change = heading of the last merged segment minus that of the first (wrapped; |.| unless signed_heading);  dist_to_kmax = path length origin -> the max-kappa vertex
    (no vertex with kappa above the floor -> the total path length: the bend is not inside the plan)."""
    P = np.vstack([np.zeros((1, 2)), np.asarray(plan8, np.float64)])
    Q = merge_short(P)
    seg = np.diff(Q, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    total = float(L.sum())
    kmax, imax = 0.0, -1
    for i in range(1, len(Q) - 1):
        a, b, c = L[i - 1], L[i], float(np.hypot(*(Q[i + 1] - Q[i - 1])))
        cr = abs(seg[i - 1, 0] * seg[i, 1] - seg[i - 1, 1] * seg[i, 0])
        k = 2.0 * cr / max(a * b * c, 1e-12)
        if k > kmax:
            kmax, imax = k, i
    dpsi = 0.0
    if len(seg) >= 2:
        h0, h1 = np.arctan2(seg[0, 1], seg[0, 0]), np.arctan2(seg[-1, 1], seg[-1, 0])
        dpsi = float((h1 - h0 + np.pi) % (2 * np.pi) - np.pi)
    kf = max(kmax, KAPPA_FLOOR)
    v_curve = float(np.sqrt(A_LAT_COMF / kf))
    dist = float(L[:imax].sum()) if (imax >= 0 and kmax > KAPPA_FLOOR) else total
    return {"kappa_max": kf, "alat_demand": float(v0) ** 2 * kf, "v_curve": v_curve, "req_decel": max(0.0, float(v0) - v_curve),
            "heading_change_abs": (dpsi if signed_heading else abs(dpsi)), "dist_to_kmax": dist, "n_pts": len(Q), "kappa_raw": kmax}


def curve_synthetic() -> dict:
    """ANALYTIC targets, written as LITERALS (not as expressions over the code under test)."""
    res, ok = {}, True

    def chk(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        res[name] = {"pass": bool(cond), "detail": detail}

    # (1) a straight line at 10 m/s: kappa 0 -> floor, v_curve 44.72, no required decel, heading change 0, dist = total path length 60
    b = curve_block(np.array([[5.0 * (k + 1), 0.0] for k in range(8)]), 10.0)
    chk("straight_kappa0", b["kappa_raw"] == 0.0 and b["kappa_max"] == 1e-3 and b["req_decel"] == 0.0 and b["heading_change_abs"] == 0.0
        and abs(b["dist_to_kmax"] - 40.0) < 1e-9, {k: b[k] for k in ("kappa_raw", "kappa_max", "req_decel", "heading_change_abs", "dist_to_kmax")})
    # (2) points on a circle of R = 50 m (origin on the circle, heading +x): the three-point curvature of ANY three points on a circle is exactly 1/R = 0.02
    R = 50.0
    th = np.linspace(0, np.pi / 2, 9)[1:]                                      # origin = angle 0; 8 slots up to a quarter circle (arc 78.5 m)
    circ = np.column_stack([R * np.sin(th), R * (1 - np.cos(th))])
    b = curve_block(circ, 15.0)
    chk("circle_R50_kappa_0.02", abs(b["kappa_raw"] - 0.02) < 1e-9 and abs(b["v_curve"] - 10.0) < 1e-9 and abs(b["alat_demand"] - 4.5) < 1e-9
        and abs(b["req_decel"] - 5.0) < 1e-9, {k: b[k] for k in ("kappa_raw", "v_curve", "alat_demand", "req_decel")})
    # heading change of 9 points (8 segments) on a quarter circle: first segment heading = step/2, last = pi/2 - step/2, step = pi/16 -> pi/2 - pi/16 = 1.37444678594552
    chk("circle_heading_change_literal", abs(b["heading_change_abs"] - 1.3744467859455345) < 1e-9, {"heading_change_abs": b["heading_change_abs"]})
    # signed variant: a RIGHT turn (mirror y) has the opposite sign
    bm = curve_block(circ * np.array([1.0, -1.0]), 15.0, signed_heading=True)
    chk("signed_heading_mirror", abs(bm["heading_change_abs"] + 1.3744467859455345) < 1e-9, {"signed": bm["heading_change_abs"]})
    # (3) short-segment merging: a standing stop = 3 trailing points within 0.2 m of the last circle point must not create a spurious bend
    tail = np.vstack([circ[:5], circ[4] + [0.1, 0.0], circ[4] + [0.15, 0.15], circ[4] + [0.2, -0.1]])
    b = curve_block(tail, 15.0)
    chk("merge_short_segments", abs(b["kappa_raw"] - 0.02) < 1e-9 and b["n_pts"] == 6, {"kappa_raw": b["kappa_raw"], "n_pts_after_merge": b["n_pts"]})
    # (4) a single right-angle corner: (0,0),(10,0),(20,0),(20,10),(20,20),...: only the vertex (20,0) bends; Menger kappa = 2*|10*10| / (10 * 10 * sqrt(200)) = sqrt(2)/10 = 0.1414213562373095,
    #     20.0 m along the path (all other vertices are collinear)
    corner = np.array([[10, 0], [20, 0], [20, 10], [20, 20], [20, 30], [20, 40], [20, 50], [20, 60]], float)
    b = curve_block(corner, 20.0)
    chk("corner_kappa_and_distance", abs(b["kappa_raw"] - 0.1414213562373095) < 1e-12 and abs(b["dist_to_kmax"] - 20.0) < 1e-9, {k: b[k] for k in ("kappa_raw", "dist_to_kmax")})
    # (5) required decel only when v0 exceeds the curve-limited speed: R=50 -> 10 m/s; v0 = 8 -> 0
    chk("no_decel_below_curve_speed", curve_block(circ, 8.0)["req_decel"] == 0.0, {})
    return {"all_pass": bool(ok), "n": len(res), "cases": res}


# =====================================================================================================================
# XWALK
# =====================================================================================================================
XW_COLS = ("share", "nearest_dist_m")


def corridor_feats(plane: np.ndarray, code: int) -> tuple[float, float]:
    """share of `code` cells inside the corridor (cell centres |y| <= 1.8, rows 40..399) over ALL corridor cells (unseen counts as not-`code`), and the
    NEAR EDGE (0.1 * row) of the nearest such cell; none -> share 0, distance 40.0 (the corridor end)."""
    sub = plane[R0:R1, C0:C1] == code
    n = int(sub.sum())
    if n == 0:
        return 0.0, X_HI_M
    row_min = int(np.flatnonzero(sub.any(axis=1))[0])
    return n / sub.size, (R0 + row_min) * CELL_M


def xwalk_synthetic(ids: dict) -> dict:
    """A crosswalk patch at x = 20 m, y = 0 built with the SOURCE's mapping function (mapsel_lib.to_cells) must be found at distance 20.0; negatives; the corridor
    cell count; and the near-edge / mirrored-y symmetry."""
    import torch
    import mapsel_lib as ML
    res, ok = {}, True

    def chk(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        res[name] = {"pass": bool(cond), "detail": detail}

    def patch(x0, x1, y0, y1, code, base=0):
        pl = np.full((H, W), base, np.uint8)
        xs, ys = np.meshgrid(np.arange(x0 + 0.025, x1, 0.05), np.arange(y0 + 0.025, y1, 0.05), indexing="ij")
        ix, iy, ins = ML.to_cells(torch.as_tensor(np.stack([xs, ys], -1), dtype=torch.float32))
        ix, iy, ins = ix.numpy(), iy.numpy(), ins.numpy()
        pl[ix[ins], iy[ins]] = code
        return pl

    cw, ed = ids["crosswalk"], ids["edge"]
    chk("corridor_cells", (R1 - R0, C1 - C0) == (360, 36), {"rows": R1 - R0, "cols": C1 - C0, "R0": R0, "C0": C0, "C1": C1})
    chk("mapping_x20_y0_is_row200_col300", tuple(int(v[0, 0]) for v in ML.to_cells(torch.tensor([[[20.0, 0.0]]], dtype=torch.float32))[:2]) == (200, 300), {})
    p = patch(20.0, 21.0, -1.0, 1.0, cw)
    s, d = corridor_feats(p, cw)
    chk("patch_x20_y0_found_at_20", abs(d - 20.0) < 1e-9 and s > 0, {"nearest_dist_m": d, "share": s, "patch_cells": int((p == cw).sum())})
    chk("patch_share_literal", abs(s - 200 / (360 * 36)) < 1e-12, {"share": s, "expect": 200 / (360 * 36)})
    s_e, d_e = corridor_feats(p, ed)
    chk("other_class_not_found", s_e == 0.0 and d_e == 40.0, {"edge_share": s_e, "edge_dist": d_e})
    for nm, (x0, x1, y0, y1), want in (("x3_inside_4m_zone_excluded", (3.0, 3.9, -1, 1), False), ("x45_beyond_40m", (45.0, 46.0, -1, 1), False), ("y3_outside_corridor", (20, 21, 3.0, 4.0), False),
                                      ("y_minus3_outside_corridor", (20, 21, -4.0, -3.0), False), ("y_plus1.7_inside", (20, 21, 1.5, 1.7), True), ("y_minus1.7_inside", (20, 21, -1.7, -1.5), True),
                                      ("x39.95_just_inside", (39.9, 39.99, -0.5, 0.5), True), ("x40.5_just_outside", (40.0, 40.9, -0.5, 0.5), False)):
        pp = patch(x0, x1, y0, y1, cw)
        s2, d2 = corridor_feats(pp, cw)
        chk(nm, (s2 > 0) == want, {"share": s2, "dist": d2, "want_found": want})
    # nearest of two patches wins; moving the patch by +5 m moves the distance by +5
    p2 = np.maximum(patch(30.0, 31.0, -1, 1, cw), patch(12.0, 13.0, 0.0, 1.0, cw))
    chk("nearest_of_two", abs(corridor_feats(p2, cw)[1] - 12.0) < 1e-9, {})
    chk("shift_plus_5m", abs(corridor_feats(patch(25.0, 26.0, -1, 1, cw), cw)[1] - 25.0) < 1e-9, {})
    # a patch drawn in the TRANSPOSED convention (rows <-> cols: row = (y+30)/0.1, col = x/0.1) is, under the declared convention, a patch at x ~ 29-31 m, y = -10..-9: outside the corridor -> not found
    pt = np.zeros((H, W), np.uint8)
    pt[290:310, 200:210] = cw
    st, dt_ = corridor_feats(pt, cw)
    chk("transposed_patch_not_found", st == 0.0 and dt_ == 40.0, {"share": st, "dist": dt_})
    # a patch drawn with REVERSED rows (row = 999 - x/0.1) is a patch at x ~ 79 m: not found
    pr_ = np.zeros((H, W), np.uint8)
    pr_[H - 1 - 209:H - 1 - 199, 290:310] = cw
    chk("row_reversed_patch_not_found", corridor_feats(pr_, cw)[0] == 0.0, {})
    return {"all_pass": bool(ok), "n": len(res), "cases": res}


# =====================================================================================================================
# raster pass (features + convention controls); cached on F:
# =====================================================================================================================
def raster_pass(ids, ui, ep_w, ws_w, G):
    cache = SCR / "d2_rast_feats.pkl"
    key = LD.hashlib.sha256(np.stack([ep_w[ui], ws_w[ui]]).tobytes()).hexdigest()[:16]
    if cache.exists():
        c = pickle.load(open(cache, "rb"))
        if c.get("key") == key:
            log("[3] raster feature cache reused")
            return c
    cw, ed = ids["crosswalk"], ids["edge"]
    n = len(ui)
    out = {"key": key, "has_rast": np.zeros(n, bool), "has_gt": np.zeros(n, bool), "own": np.full((n, 4), np.nan), "gt": np.full((n, 4), np.nan),
           "gt_seen_share": np.full(n, np.nan), "own_classes_in_corridor": np.zeros((n, 8), np.int32)}
    convs = ("decl", "mirror_y", "transposed", "rows_reversed")
    conv = {(pl, cv, sub): np.zeros((n, 2), np.int64) for pl in ("own", "gt") for cv in convs for sub in ("all", "lat2")}
    t0 = time.time()
    for r, i in enumerate(ui):
        fp = RAST / f"e{int(ep_w[i]):03d}_w{int(ws_w[i]):04d}.npz"
        if not fp.exists():
            continue
        z = np.load(fp)
        out["has_rast"][r] = True
        code = z["code"]
        assert code.shape == (H, W) and code.dtype == np.uint8
        out["own"][r] = [*corridor_feats(code, cw), *corridor_feats(code, ed)]
        out["own_classes_in_corridor"][r] = np.bincount(code[R0:R1, C0:C1].ravel(), minlength=8)[:8]
        planes = {"own": code}
        if "gt" in z.files:
            gt = z["gt"]
            assert gt.shape == (H, W) and gt.dtype == np.uint8
            out["has_gt"][r] = True
            out["gt"][r] = [*corridor_feats(gt, cw), *corridor_feats(gt, ed)]
            out["gt_seen_share"][r] = float((gt[R0:R1, C0:C1] != NOT_SEEN).mean())
            planes["gt"] = gt
        # convention control: the human GT path (6 points, 1..6 s, ego frame at t0) must land on DRIVABLE cells under the declared convention only
        x, y = G[i, :, 0].astype(np.float64), G[i, :, 1].astype(np.float64)
        for cv in convs:
            if cv == "decl":
                rr, cc = np.floor(x / CELL_M).astype(int), np.floor((y + Y_HALF_M) / CELL_M).astype(int)
            elif cv == "mirror_y":
                rr, cc = np.floor(x / CELL_M).astype(int), np.floor((-y + Y_HALF_M) / CELL_M).astype(int)
            elif cv == "rows_reversed":
                rr, cc = H - 1 - np.floor(x / CELL_M).astype(int), np.floor((y + Y_HALF_M) / CELL_M).astype(int)
            else:
                rr, cc = np.floor((y + Y_HALF_M) / CELL_M).astype(int), np.floor(x / CELL_M).astype(int)
            ins = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            for pl, plane in planes.items():
                v = np.full(len(x), NOT_SEEN, np.uint8)
                v[ins] = plane[rr[ins], cc[ins]]
                seen = v != NOT_SEEN
                drv = np.isin(v, DRV_CODES)
                for sub, m in (("all", np.ones(len(x), bool)), ("lat2", np.abs(y) >= 2.0)):
                    conv[(pl, cv, sub)][r] = [int((seen & m).sum()), int((drv & seen & m).sum())]
        if r % 500 == 0:
            log(f"   raster pass {r}/{n} {time.time() - t0:.0f}s")
    out["conv"] = conv
    pickle.dump(out, open(cache, "wb"))
    return out


def conv_summary(rp) -> dict:
    s = {}
    for (pl, cv, sub), a in rp["conv"].items():
        tot = a.sum(0)
        s.setdefault(pl, {}).setdefault(cv, {})[sub] = {"n_path_points_seen": int(tot[0]), "drivable_share": pr(tot[1] / max(tot[0], 1))}
    return s


# =====================================================================================================================
def run_channel(name, block, covered, ui, ep_w, ws_w, y_all, A_all, v0_w, boot_B=LD.BOOT_B, hgb=True) -> dict:
    """block [len(ui), k] aligned with ui; covered bool [len(ui)].  Returns the channel result dict (+ the OOF arrays under '_oof')."""
    sel = ui[covered]
    blk = block[covered]
    ep_s, ws_s, y_s = ep_w[sel], ws_w[sel], y_all[sel]
    A_s = A_all[sel]
    B_s = np.column_stack([A_s, blk])
    pi = LD.cross_episode_permutation(ep_s, LD.SEED)
    assert (ep_s[pi] != ep_s).all()
    C_s = np.column_stack([A_s, blk[pi]])
    folds, _ = LD.make_folds(ep_s)
    ue = np.unique(ep_s)
    ep_pos = np.searchsorted(ue, ep_s)
    oof = {"A": LD.cross_fit(A_s, y_s, ep_s, folds), "B": LD.cross_fit(B_s, y_s, ep_s, folds), "C": LD.cross_fit(C_s, y_s, ep_s, folds)}
    pt = {k: LD.auc_np(y_s, v) for k, v in oof.items()}
    fns = {k: LD.WAuc(y_s, v) for k, v in oof.items()}
    for k, f in fns.items():
        assert abs(f(np.ones(len(y_s))) - pt[k]) < 1e-9
    bt = LD.boot_cluster(ep_pos, len(ue), fns, B=boot_B)
    dBA, dCA, dBC = pt["B"] - pt["A"], pt["C"] - pt["A"], pt["B"] - pt["C"]
    ciBA, ciCA, ciBC = ci(bt["B"] - bt["A"]), ci(bt["C"] - bt["A"]), ci(bt["B"] - bt["C"])
    c1 = bool(dBA >= DELTA_BAR and ciBA[0] > 0.0)
    c2 = bool(dCA <= CTRL_MAX)
    res = {"channel": name, "block_columns": None, "n_windows": int(len(y_s)), "n_episodes": int(len(ue)), "n_decel": int(y_s.sum()), "n_cruise": int((y_s == 0).sum()),
           "decel_base_rate": pr(y_s.mean()),
           "auc": {k: {"auc": pr(pt[k]), "ci95": [pr(x) for x in ci(bt[k])]} for k in ("A", "B", "C")},
           "delta_B_minus_A": {"point": pr(dBA), "ci95": [pr(x) for x in ciBA]}, "delta_C_minus_A": {"point": pr(dCA), "ci95": [pr(x) for x in ciCA]},
           "delta_B_minus_C": {"point": pr(dBC), "ci95": [pr(x) for x in ciBC]},
           "clauses": {"1_dAUC_BA_ge_+0.03_and_ci_lo_gt_0": c1, "2_shuffled_control_dAUC_point_le_+0.01": c2},
           "word": "PRESENT" if (c1 and c2) else "ABSENT", "_ids": (ep_s, ws_s)}
    if hgb:
        oh = {"A": LD.cross_fit_hgb(A_s, y_s, folds), "B": LD.cross_fit_hgb(B_s, y_s, folds), "C": LD.cross_fit_hgb(C_s, y_s, folds)}
        fh = {k: LD.WAuc(y_s, v) for k, v in oh.items()}
        bh = LD.boot_cluster(ep_pos, len(ue), fh, B=boot_B)
        one = np.ones(len(y_s))
        res["DESCRIPTIVE_nonlinear_probe_hgb"] = {
            "auc": {k: pr(fh[k](one)) for k in fh},
            "delta_B_minus_A": {"point": pr(fh["B"](one) - fh["A"](one)), "ci95": [pr(x) for x in ci(bh["B"] - bh["A"])]},
            "delta_C_minus_A": {"point": pr(fh["C"](one) - fh["A"](one)), "ci95": [pr(x) for x in ci(bh["C"] - bh["A"])]},
            "config": "one fixed untuned HistGradientBoostingClassifier (leadinfo_d.cross_fit_hgb), same folds; cannot change the word"}
    return res


def delta_only(block, covered, ui, ep_w, ws_w, y_all, A_all, boot_B=LD.BOOT_B) -> dict:
    """A compact B-vs-A / C-vs-A read for a variant block (descriptive)."""
    r = run_channel("variant", block, covered, ui, ep_w, ws_w, y_all, A_all, None, boot_B=boot_B, hgb=False)
    return {"n_windows": r["n_windows"], "auc": {k: r["auc"][k]["auc"] for k in ("A", "B", "C")}, "delta_B_minus_A": r["delta_B_minus_A"],
            "delta_C_minus_A": r["delta_C_minus_A"], "word_if_it_were_the_channel": r["word"]}


def block_alone(block, covered, ui, ep_w, y_all, v0_w, boot_B=LD.BOOT_B) -> dict:
    """DESCRIPTIVE: does the block carry label information ON ITS OWN (beyond v0), whether or not the lon head already has it?  AUC(v0 + block) vs AUC(v0) on the channel's windows,
    same folds / bootstrap as the channel (logistic L2, out-of-fold)."""
    sel = ui[covered]
    ep_s, y_s, v_s = ep_w[sel], y_all[sel], v0_w[sel]
    folds, _ = LD.make_folds(ep_s)
    ue = np.unique(ep_s)
    ep_pos = np.searchsorted(ue, ep_s)
    o0 = LD.cross_fit(v_s.reshape(-1, 1), y_s, ep_s, folds)
    o1 = LD.cross_fit(np.column_stack([v_s, block[covered]]), y_s, ep_s, folds)
    f = {"v0": LD.WAuc(y_s, o0), "v0+block": LD.WAuc(y_s, o1)}
    bt = LD.boot_cluster(ep_pos, len(ue), f, B=boot_B)
    one = np.ones(len(y_s))
    return {"auc_v0_only": LD.pr(f["v0"](one)), "auc_v0_plus_block": LD.pr(f["v0+block"](one)),
            "ci95_v0_plus_block": [LD.pr(x) for x in LD.ci(bt["v0+block"])],
            "delta_block_minus_v0": {"point": LD.pr(f["v0+block"](one) - f["v0"](one)), "ci95": [LD.pr(x) for x in LD.ci(bt["v0+block"] - bt["v0"])]}}


def main():
    t_start = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    out = {"prereg": "PREREG_CHANNELS_D2.md", "tier": "DIAGNOSTIC on logged eval139 labels and the planner's own logged outputs; GT channels are UPPER BOUNDS, never inference inputs",
           "estimator": ("IDENTICAL to LEADINFO_D (imported from leadinfo_d.py): 5-fold episode-grouped cross-fit (fold seed 0) of L2 logistic regression (C=1, standardised on the fit fold); "
                         "score = AUC of the out-of-fold P(decel); CIs = episode-RESAMPLING bootstrap (B=2000, seed 0, OOF predictions fixed, 2.5/97.5). It answers 'another draw of "
                         "EPISODES'; it is blind to training-run variance (convex models: nil) and to INFERENCE-run variance of the stochastic planner (CURVE reads a sampled plan; "
                         "see the plan-draw variant)."),
           "deviations": []}
    out["seal"] = check_seal()
    log("[0] seals OK", out["seal"]["prereg_sha256"][:8], out["seal"]["leadinfo_d_seal"]["prereg_sha256"][:8])
    ids = class_ids_from_source()
    out["class_ids_from_source"] = ids
    log("[0] class ids", {k: ids[k] for k in ("crosswalk", "edge", "sidewalk", "drivable")})

    # ---- 1. synthetic controls ---------------------------------------------------------------------------------
    cs, xs = curve_synthetic(), xwalk_synthetic(ids)
    out["synthetic_controls"] = {"curve": cs, "xwalk": xs}
    log(f"[1] synthetic: curve {cs['n']} cases pass={cs['all_pass']}; xwalk {xs['n']} cases pass={xs['all_pass']}")
    if not (cs["all_pass"] and xs["all_pass"]):
        bad = {k: v for sec in (cs, xs) for k, v in sec["cases"].items() if not v["pass"]}
        (RES / "CHANNELS_D2.json").write_text(json.dumps({"STOP": "synthetic control failed", "failed": bad}, indent=1, default=str), encoding="utf-8")
        raise SystemExit(f"STOP: synthetic control failed: {bad}")

    # ---- 2. windows + identity with LEADINFO_D ------------------------------------------------------------------
    C, E, comp, gt, pop_cruise, pop_decel = LD.load_windows()
    union = pop_cruise | pop_decel
    ui = np.flatnonzero(union)
    ep_w, ws_w, v0_w = np.asarray(E["clip"]), np.asarray(E["ws"]), np.asarray(E["v0"], np.float64)
    y_all = pop_decel.astype(np.int64)
    P6 = LD.softmax(C.lon6(E["logits"]))
    A_all = np.column_stack([P6, v0_w])
    out["population"] = {"s6_windows": int(len(ws_w)), "union_n": int(len(ui)), "gt_decel": int(pop_decel.sum()), "gt_cruise": int(pop_cruise.sum()),
                         "decel_base_rate": pr(pop_decel[ui].mean())}
    z = np.load(RES / "leadinfo_windows.npz")
    key = {(int(e), int(w)): i for i, (e, w) in enumerate(zip(ep_w, ws_w))}
    li = np.array([key[(int(e), int(w))] for e, w in zip(z["ep"], z["ws"])])
    assert np.array_equal(y_all[li], z["y"]) and np.allclose(v0_w[li], z["v0"])
    folds0, _ = LD.make_folds(ep_w[li])
    assert np.array_equal(folds0, z["fold"])
    oofA = LD.cross_fit(A_all[li], y_all[li], ep_w[li], folds0)
    d_id = float(np.abs(oofA - z["oof_A"]).max())
    out["identity_with_leadinfo_d"] = {"n_windows": int(len(li)), "max_abs_diff_oofA": d_id, "folds_equal": True, "y_equal": True,
                                       "reading": "A's out-of-fold predictions on LEADINFO_D's 2348 windows are reproduced by this script's imported estimator"}
    log("[2] identity with LEADINFO_D: max |oofA - saved| =", d_id)
    assert d_id < 1e-12, "estimator identity with LEADINFO_D broken"

    # ---- 3. features -----------------------------------------------------------------------------------------
    # CURVE from the battery-scored native plan (the one whose lon logits feed A): E['plan8'] = DEPLOY8_POS S6_plan8, and a second DRAW from the MAPSEL roll
    assert float(E["checks"]["plan8_slots_equal_os"]) == 0.0, "S6_plan8 slots != the battery os arm"
    plan8 = np.asarray(E["plan8"], np.float64)
    assert plan8.shape == (len(ws_w), 8, 2) and np.isfinite(plan8).all()
    tr = np.load(TRACE, allow_pickle=True)
    tkey = {(int(e), int(w)): i for i, (e, w) in enumerate(zip(tr["e_i"], tr["ws"]))}
    ti = np.array([tkey[(int(e), int(w))] for e, w in zip(ep_w, ws_w)])
    plan_tr = np.asarray(tr["plan8"], np.float64)[ti]
    assert np.allclose(tr["v0_in"][ti], v0_w, atol=0.0), "MAPSEL trace v0 != battery v0 (join broken)"
    dpl = np.abs(plan_tr - plan8)
    out["plan_join"] = {
        "primary_source": "E['plan8'] = DEPLOY8_POS S6_plan8 of the banked 48,852 per-frame battery (seed 0); its slots [1,3,4,5,6,7] equal the battery 'os' arm exactly (asserted, max diff 0.0); "
                           "the SAME roll that produced the lon logits of feature set A; join by (clip, ws) asserted through l2_common.load_eval_pf's row checks",
        "mapsel_trace_join": "F:/refcv9_mapsel/trace_s0.npz plan8, joined on (e_i, ws): all S6 windows found, v0_in == battery v0 exactly",
        "mapsel_trace_vs_battery_plan8": {"max_abs_diff_m": pr(dpl.max(), 3), "median_over_windows_of_max_abs_diff_m": pr(np.median(dpl.max(axis=(1, 2))), 3),
                                          "frac_windows_differ_gt_1e-4": pr((dpl.max(axis=(1, 2)) > 1e-4).mean()),
                                          "reading": "two different DRAWS of the stochastic (ddim) planner at inference seed 0 -- the RNG is consumed differently by a 13-arm battery roll and an os-only roll; "
                                                     "the CURVE feature is a function of the draw, so the second draw is an inference-variance replicate (variant below)"}}
    log("[3] plan join", out["plan_join"]["mapsel_trace_vs_battery_plan8"])

    def curve_matrix(plans, signed=False):
        rows = [curve_block(plans[i], v0_w[i], signed_heading=signed) for i in ui]
        return np.array([[r[c] for c in CURVE_COLS] for r in rows]), rows

    CURVE, curve_rows = curve_matrix(plan8)
    CURVE_TR, _ = curve_matrix(plan_tr)
    CURVE_SGN, _ = curve_matrix(plan8, signed=True)
    cstat = {c: [pr(q, 3) for q in np.percentile(CURVE[:, j], [5, 50, 95])] for j, c in enumerate(CURVE_COLS)}
    out["curve_features"] = {"columns": list(CURVE_COLS), "q05_q50_q95": cstat,
                             "share_windows_kappa_raw_gt_floor": pr(np.mean([r["kappa_raw"] > KAPPA_FLOOR for r in curve_rows])),
                             "share_windows_req_decel_gt_0": pr(np.mean(CURVE[:, 3] > 0)), "share_windows_req_decel_gt_1ms": pr(np.mean(CURVE[:, 3] > 1.0)),
                             "n_pts_after_merge_counts": {int(k): int(v) for k, v in zip(*np.unique([r["n_pts"] for r in curve_rows], return_counts=True))}}
    rp = raster_pass(ids, ui, ep_w, ws_w, np.asarray(E["g"], np.float64))
    OWN, GTF = rp["own"], rp["gt"]

    # ---- 4. real convention controls --------------------------------------------------------------------------
    cv = conv_summary(rp)
    out["real_convention_controls"] = {
        "what": "the human GT path (E['g'], 6 points at 1..6 s, ego frame at t0) mapped to raster cells under three conventions; the share of SEEN path cells that are DRIVABLE "
                "(codes 1,2,3,4,6) must be high only under the declared one (row = x/0.1, col = (y+30)/0.1). 'lat2' = path points with |y| >= 2 m (where a mirror matters).",
        "shares": cv}
    for pl in ("gt", "own"):
        a, m, t, rv = cv[pl]["decl"], cv[pl]["mirror_y"], cv[pl]["transposed"], cv[pl]["rows_reversed"]
        out["real_convention_controls"][f"assert_{pl}"] = {
            "declared_minus_rows_reversed_all_ge_0.10": bool(a["all"]["drivable_share"] - rv["all"]["drivable_share"] >= 0.10),
            "declared_all_ge_0.80": bool(a["all"]["drivable_share"] >= 0.80),
            "declared_minus_mirror_lat2_ge_0.10": bool(a["lat2"]["drivable_share"] - m["lat2"]["drivable_share"] >= 0.10),
            "declared_minus_transposed_all_ge_0.10": bool(a["all"]["drivable_share"] - t["all"]["drivable_share"] >= 0.10)}
    log("[4] convention controls", json.dumps(cv))
    for pl in ("gt", "own"):
        assert all(out["real_convention_controls"][f"assert_{pl}"].values()), f"raster convention control failed for {pl}: {cv[pl]}"
    # own vs GT agreement on 'crosswalk ahead' (descriptive)
    both = rp["has_gt"]
    ow, gw = OWN[both, 0] > 0, GTF[both, 0] > 0
    out["real_convention_controls"]["own_vs_gt_crosswalk_in_corridor"] = {
        "n_windows_with_both": int(both.sum()), "own_any": pr(ow.mean()), "gt_any": pr(gw.mean()), "agree_share": pr((ow == gw).mean()),
        "own_any_given_gt_any": pr(ow[gw].mean()) if gw.any() else None, "gt_any_given_own_any": pr(gw[ow].mean()) if ow.any() else None,
        "own_edge_any": pr((OWN[both, 2] > 0).mean()), "gt_edge_any": pr((GTF[both, 2] > 0).mean())}

    # ---- coverage -------------------------------------------------------------------------------------------
    allc = np.ones(len(ui), bool)
    cov = {"union_windows": int(len(ui)),
           "CURVE": {"covered": int(allc.sum()), "rule": "every window has a finite native plan8 (asserted)"},
           "XWALK-OWN": {"covered": int(rp["has_rast"].sum()), "rule": "a MAPSEL raster file (e, ws) exists in F:/refcv9_mapsel/rast_s0 (the `code` plane is always present)"},
           "XWALK-GT": {"covered": int(rp["has_gt"].sum()), "rule": "the raster file carries a `gt` plane (clips with no SAM3 GT file have none; unseen cells inside the corridor count as not-crosswalk / not-edge)"},
           "gt_corridor_seen_share_quantiles_q0_q05_q25_q50": [pr(q, 3) for q in np.nanpercentile(rp["gt_seen_share"][rp["has_gt"]], [0, 5, 25, 50])]}
    for nm, m in (("CURVE", allc), ("XWALK-OWN", rp["has_rast"]), ("XWALK-GT", rp["has_gt"])):
        cov[nm]["fraction"] = pr(m.mean())
        cov[nm]["decel_covered"] = int((m & pop_decel[ui]).sum())
        cov[nm]["cruise_covered"] = int((m & pop_cruise[ui]).sum())
        cov[nm]["uncovered_by_episode_index"] = {int(e): int(((ep_w[ui] == e) & ~m).sum()) for e in np.unique(ep_w[ui][~m])}
    out["coverage"] = cov
    log("[4] coverage", {k: (v["covered"], v["fraction"]) for k, v in cov.items() if isinstance(v, dict) and "covered" in v})

    # ---- 5. channels ----------------------------------------------------------------------------------------
    chans = {}
    chans["CURVE"] = run_channel("CURVE", CURVE, allc, ui, ep_w, ws_w, y_all, A_all, v0_w)
    chans["CURVE"]["block_columns"] = list(CURVE_COLS)
    own_blk = np.nan_to_num(OWN, nan=0.0)
    gt_blk = np.nan_to_num(GTF, nan=0.0)
    chans["XWALK-OWN"] = run_channel("XWALK-OWN", own_blk, rp["has_rast"], ui, ep_w, ws_w, y_all, A_all, v0_w)
    chans["XWALK-OWN"]["block_columns"] = ["xwalk_share", "xwalk_nearest_dist_m", "edge_share", "edge_nearest_dist_m"]
    chans["XWALK-GT"] = run_channel("XWALK-GT", gt_blk, rp["has_gt"], ui, ep_w, ws_w, y_all, A_all, v0_w)
    chans["XWALK-GT"]["block_columns"] = ["xwalk_share", "xwalk_nearest_dist_m", "edge_share", "edge_nearest_dist_m"]
    # descriptive variants (cannot change a word)
    var = {}
    var["CURVE_second_planner_draw_MAPSEL_trace_plan8"] = delta_only(CURVE_TR, allc, ui, ep_w, ws_w, y_all, A_all)
    var["CURVE_signed_heading_change"] = delta_only(CURVE_SGN, allc, ui, ep_w, ws_w, y_all, A_all)
    var["XWALK-OWN_crosswalk_columns_only"] = delta_only(own_blk[:, :2], rp["has_rast"], ui, ep_w, ws_w, y_all, A_all)
    var["XWALK-GT_crosswalk_columns_only"] = delta_only(gt_blk[:, :2], rp["has_gt"], ui, ep_w, ws_w, y_all, A_all)
    var["XWALK-GT_edge_columns_only"] = delta_only(gt_blk[:, 2:], rp["has_gt"], ui, ep_w, ws_w, y_all, A_all)
    # A on its channel windows and the XWALK-OWN block on the SAME windows as XWALK-GT (the GT-vs-own comparison on identical windows)
    same = rp["has_gt"] & rp["has_rast"]
    var["XWALK-OWN_on_the_XWALK-GT_windows"] = delta_only(own_blk, same, ui, ep_w, ws_w, y_all, A_all)
    # feature prevalence by class (descriptive)
    y_u = y_all[ui]
    prev = {}
    for nm, blk, m in (("XWALK-OWN", OWN, rp["has_rast"]), ("XWALK-GT", GTF, rp["has_gt"])):
        for cn, j in (("crosswalk", 0), ("edge", 2)):
            has = blk[:, j] > 0
            prev[f"{nm}_{cn}_in_corridor_decel_vs_cruise"] = {"decel": pr(has[m & (y_u == 1)].mean()), "cruise": pr(has[m & (y_u == 0)].mean()), "overall": pr(has[m].mean())}
    prev["CURVE_req_decel_gt_0_decel_vs_cruise"] = {"decel": pr((CURVE[y_u == 1, 3] > 0).mean()), "cruise": pr((CURVE[y_u == 0, 3] > 0).mean())}
    out["block_alone_DESCRIPTIVE"] = {
        "CURVE": block_alone(CURVE, allc, ui, ep_w, y_all, v0_w),
        "XWALK-OWN": block_alone(own_blk, rp["has_rast"], ui, ep_w, y_all, v0_w),
        "XWALK-GT": block_alone(gt_blk, rp["has_gt"], ui, ep_w, y_all, v0_w)}
    out["variants_DESCRIPTIVE"] = var
    out["prevalence_DESCRIPTIVE"] = prev
    for k in chans:
        chans[k].pop("_ids", None)
    out["channels"] = chans
    words = {k: v["word"] for k, v in chans.items()}
    out["words"] = words
    nxt = []
    if words["CURVE"] == "PRESENT":
        nxt.append("CURVE PRESENT -> a deployable curve feature into the fresh-split gate (HSGATE-v2 / SELCOR-v2), pre-registered first, 0 GPU.")
    if words["XWALK-GT"] == "PRESENT" and words["XWALK-OWN"] != "PRESENT":
        nxt.append("XWALK-GT PRESENT but XWALK-OWN not -> the map fine-tune (PREREG_MAPFT) gains a second consumer; read XWALK-OWN again on MAPFT's map.")
    if all(w == "ABSENT" for w in words.values()):
        nxt.append("All ABSENT -> the window-level decel decision is not recoverable from these explicit channels: the refcv9 lever is TRAINING the lon decision "
                   "(a clean-split lon head with richer supervision), named in PLAN_REFCV9 with this evidence.")
    if not nxt:
        nxt.append("Mixed outcome not covered by a single prereg branch: " + json.dumps(words) + " (XWALK-OWN PRESENT or XWALK-GT PRESENT with OWN PRESENT) -- the branches CURVE / GT-not-OWN / all-ABSENT do not "
                   "fire as written; reported without choosing a branch.")
    out["next_per_prereg_s3"] = nxt
    out["deviations"] += [
        "D1 plan source: the prereg says 'the planner's own native S6 plan P_os8'. Used E['plan8'] (DEPLOY8_POS S6_plan8 of the banked battery roll -- the roll that produced feature set A's lon logits; its S6 slots equal the battery os arm exactly). "
        "The MAPSEL roll's plan8 is a DIFFERENT DRAW of the stochastic planner (median max-diff %.2f m), so it is reported as an inference-variance variant, not as the channel." % out["plan_join"]["mapsel_trace_vs_battery_plan8"]["median_over_windows_of_max_abs_diff_m"],
        "D2 details the prereg leaves open (fixed in code before any AUC): polyline = origin + 8 rear-axle slots; short segments merged by keeping a point only if >= 0.5 m from the last kept point; kappa_max floored at 1e-3 (R = 1 km) "
        "so v_curve <= 44.7 m/s and the derived columns stay finite; the 'curve-limited speed sqrt(2.0 / kappa_max)' uses a_lat = 2.0 m/s^2 as written; heading change = |last merged segment heading - first| "
        "(absolute value -- the sign is left/right, not decel-relevant; the SIGNED form is a descriptive variant); dist_to_kmax = total path length when no vertex exceeds the floor.",
        "D3 corridor on CELL CENTRES: rows 40..399 (x in [4, 40) m, near edge 0.1*row), cols 282..317 (centre y in [-1.75, +1.75]) = 360 x 36 cells; share = class cells / ALL corridor cells (unseen counts as not-class); "
        "nearest distance = near edge of the nearest class cell, none -> 40.0 m; XWALK blocks = [xwalk share, xwalk nearest dist, edge share, edge nearest dist] (4 columns).",
        "D4 coverage rules: CURVE all windows; XWALK-OWN = windows with a MAPSEL raster file; XWALK-GT = windows whose raster carries a `gt` plane. Each channel re-fits A on its own windows (prereg s1).",
        "D5 estimator note (same as LEADINFO_D D5): episode-RESAMPLING bootstrap of the AUC with OOF predictions fixed; fold assignment is re-drawn (seed 0) on each channel's own episodes, so channel folds differ from LEADINFO_D's.",
    ]
    out["inputs_sha256"] = {"prereg_d2": out["seal"]["prereg_sha256"], "prereg_leadinfo_d": out["seal"]["leadinfo_d_seal"]["prereg_sha256"],
                            "trace_s0_npz": LD.sha256_file(TRACE), "leadinfo_windows_npz": LD.sha256_file(RES / "leadinfo_windows.npz"),
                            "extras_npz_dump_s0": LD.sha256_file(LD.ARCH / "dump_s0" / "refcv6_extras.npz"), "code_channels_d2.py": LD.sha256_file(__file__),
                            "code_leadinfo_d.py": LD.sha256_file(LD.HERE / "code" / "leadinfo_d.py"), "rast_dir": str(RAST), "n_rast_files": len(list(RAST.glob("e*.npz")))}
    out["elapsed_s"] = round(time.time() - t_start, 1)
    np.savez_compressed(RES / "channels_d2_features.npz", ui=ui, ep=ep_w[ui], ws=ws_w[ui], y=y_u, v0=v0_w[ui], curve=CURVE, curve_cols=np.array(CURVE_COLS), own=OWN, gt=GTF,
                        has_rast=rp["has_rast"], has_gt=rp["has_gt"], gt_seen_share=rp["gt_seen_share"])
    (RES / "CHANNELS_D2.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    write_md(out)
    log("RESULT", json.dumps(words), "elapsed", out["elapsed_s"])


def fmt(d):
    return f"{d['point']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"


def write_md(o):
    L = [f"# CHANNELS_D2 -- " + ", ".join(f"{k} {v}" for k, v in o["words"].items()) + "\n"]
    L.append(f"PREREG_CHANNELS_D2.md (sha256 `{o['seal']['prereg_sha256']}`); DIAGNOSTIC on logged eval139 labels and the planner's own logged outputs (S6, compliant, seed-0 battery draw `step48852ema_pf`); "
             "GT channels are UPPER BOUNDS, never inference inputs.  Rule (s3): PRESENT iff dAUC(channel - A) >= +0.03 with CI lo > 0 AND the shuffled control's dAUC point estimate <= +0.01.\n")
    L.append("## Words\n\n| channel | word | n windows / episodes | dAUC(channel - A) [CI] | shuffled dAUC(C - A) [CI] | clause 1 | clause 2 |\n|---|---|---|---|---|---|---|")
    for k, c in o["channels"].items():
        cl = list(c["clauses"].values())
        L.append(f"| {k} | **{c['word']}** | {c['n_windows']} / {c['n_episodes']} | {fmt(c['delta_B_minus_A'])} | {fmt(c['delta_C_minus_A'])} | {cl[0]} | {cl[1]} |")
    L.append("\nNext per prereg s3: " + " ".join(o["next_per_prereg_s3"]) + "\n")
    L.append("## Estimator\n" + o["estimator"] + "\n")
    L.append("## AUCs (out-of-fold P(decel); A re-fit on each channel's own windows)\n\n| channel | n (decel / cruise), base rate | A | A + block | A + shuffled block | B - C [CI] |\n|---|---|---|---|---|---|")
    for k, c in o["channels"].items():
        a = c["auc"]
        L.append(f"| {k} | {c['n_windows']} ({c['n_decel']} / {c['n_cruise']}), {c['decel_base_rate']} | {a['A']['auc']:.4f} [{a['A']['ci95'][0]:.4f}, {a['A']['ci95'][1]:.4f}] | "
                 f"{a['B']['auc']:.4f} [{a['B']['ci95'][0]:.4f}, {a['B']['ci95'][1]:.4f}] | {a['C']['auc']:.4f} [{a['C']['ci95'][0]:.4f}, {a['C']['ci95'][1]:.4f}] | {fmt(c['delta_B_minus_C'])} |")
    cov = o["coverage"]
    L.append("\n## Coverage (union population %d windows = %d GT-decel + %d GT-cruise, compliant S6)\n" % (cov["union_windows"], o["population"]["gt_decel"], o["population"]["gt_cruise"]))
    for k in ("CURVE", "XWALK-OWN", "XWALK-GT"):
        L.append(f"* **{k}**: {cov[k]['covered']}/{cov['union_windows']} ({cov[k]['fraction']}); decel {cov[k]['decel_covered']}, cruise {cov[k]['cruise_covered']}; rule: {cov[k]['rule']}; "
                 f"uncovered windows by episode index: {cov[k]['uncovered_by_episode_index']}")
    L.append(f"* GT corridor seen-share quantiles (q0, q5, q25, q50) over windows with a gt plane: {cov['gt_corridor_seen_share_quantiles_q0_q05_q25_q50']}\n")
    L.append("## Convention controls\n")
    ids = o["class_ids_from_source"]
    L.append(f"* class ids FROM SOURCE: crosswalk = {ids['crosswalk']}, drivable edge = {ids['edge']} (CLASS_KEYS {ids['class_keys']}); " + "; ".join(ids["sources"]) + ".")
    L.append("* raster convention (mapsel_lib.py:3-4, :19, :70): ego frame at t0, x forward, y left; [H=1000 rows along x 0..100 m, W=600 cols along y -30 (right) .. +30 (left)], 0.1 m cells; "
             "cell (i, j) = x in [0.1 i, 0.1 (i+1)), y in [-30 + 0.1 j, ...); corridor = rows 40..399 x cols 282..317 (360 x 36 cells).")
    xs, cs = o["synthetic_controls"]["xwalk"], o["synthetic_controls"]["curve"]
    pd_ = xs["cases"]["patch_x20_y0_found_at_20"]["detail"]
    L.append(f"* synthetic crosswalk (patch built with the SOURCE's `mapsel_lib.to_cells`): **a patch at x = 20 m, y = 0 is found at distance {pd_['nearest_dist_m']:.1f} m** "
             f"(share {pd_['share']:.5f} = 200 / 12960 cells); {xs['n']} cases all pass = {xs['all_pass']} (x < 4 m excluded, x > 40 m and |y| > 1.8 m not found, +/-1.7 m inside, class mismatch not found, nearest wins, shift +5 m -> +5 m).")
    L.append(f"* synthetic curve (analytic literals): {cs['n']} cases all pass = {cs['all_pass']} (circle R = 50 m -> kappa 0.02, v_curve 10.0, alat 4.5, required decel 5.0 at v0 15; quarter-circle heading change 1.37445; right-angle corner kappa 0.141421 at 20.0 m; short-segment merge).")
    rc = o["real_convention_controls"]
    for pl in ("gt", "own"):
        s = rc["shares"][pl]
        L.append(f"* real, human path on drivable cells ({pl} plane; seen cells; drivable share, all / |y|>=2 m): declared {s['decl']['all']['drivable_share']} / {s['decl']['lat2']['drivable_share']} "
                 f"(n {s['decl']['all']['n_path_points_seen']} / {s['decl']['lat2']['n_path_points_seen']}); y-mirrored {s['mirror_y']['all']['drivable_share']} / {s['mirror_y']['lat2']['drivable_share']}; "
                 f"transposed {s['transposed']['all']['drivable_share']} / {s['transposed']['lat2']['drivable_share']}; rows-reversed {s['rows_reversed']['all']['drivable_share']} / {s['rows_reversed']['lat2']['drivable_share']} -> asserts {rc['assert_' + pl]}")
    a = rc["own_vs_gt_crosswalk_in_corridor"]
    L.append(f"* own vs GT map, crosswalk in the corridor (n {a['n_windows_with_both']}): own any {a['own_any']}, GT any {a['gt_any']}, agree {a['agree_share']}, own-any given GT-any {a['own_any_given_gt_any']}, "
             f"GT-any given own-any {a['gt_any_given_own_any']}; drivable edge in corridor: own {a['own_edge_any']}, GT {a['gt_edge_any']}.")
    pj = o["plan_join"]
    L.append(f"* plan join: {pj['primary_source']}.  MAPSEL trace plan8 vs the battery plan8: max |diff| {pj['mapsel_trace_vs_battery_plan8']['max_abs_diff_m']} m, median over windows of the per-window max "
             f"{pj['mapsel_trace_vs_battery_plan8']['median_over_windows_of_max_abs_diff_m']} m, {pj['mapsel_trace_vs_battery_plan8']['frac_windows_differ_gt_1e-4']} of windows differ -- a second DRAW of the stochastic planner.")
    idn = o["identity_with_leadinfo_d"]
    L.append(f"* identity with LEADINFO_D: A's out-of-fold predictions on LEADINFO_D's {idn['n_windows']} windows reproduced, max |diff| {idn['max_abs_diff_oofA']:.1e}.\n")
    cf = o["curve_features"]
    L.append("## Feature census (descriptive)\n")
    L.append(f"* CURVE (q05/q50/q95): " + "; ".join(f"{c} {cf['q05_q50_q95'][c]}" for c in cf["columns"]) + f"; windows with kappa above the floor {cf['share_windows_kappa_raw_gt_floor']}, required decel > 0: {cf['share_windows_req_decel_gt_0']}, > 1 m/s: {cf['share_windows_req_decel_gt_1ms']}.")
    for k, v in o["prevalence_DESCRIPTIVE"].items():
        L.append(f"* {k}: {v}")
    L.append("\n## Descriptive extras (cannot change a word)\n")
    for k, v in o["variants_DESCRIPTIVE"].items():
        L.append(f"* {k} (n {v['n_windows']}): AUC A {v['auc']['A']}, A+block {v['auc']['B']}, A+shuffled {v['auc']['C']}; dAUC(block - A) {fmt(v['delta_B_minus_A'])}; shuffled {fmt(v['delta_C_minus_A'])} (would-be word {v['word_if_it_were_the_channel']}).")
    for k, c in o["channels"].items():
        h = c["DESCRIPTIVE_nonlinear_probe_hgb"]
        L.append(f"* {k}, nonlinear probe ({h['config']}): AUC A {h['auc']['A']}, B {h['auc']['B']}, C {h['auc']['C']}; dAUC(B - A) {fmt(h['delta_B_minus_A'])}; dAUC(C - A) {fmt(h['delta_C_minus_A'])}.")
    L.append("* block alone (does the block carry label information beyond v0 by itself, whether or not the lon head already has it?): " + "; ".join(
        f"{k}: AUC v0 {v['auc_v0_only']}, v0 + block {v['auc_v0_plus_block']} [{v['ci95_v0_plus_block'][0]}, {v['ci95_v0_plus_block'][1]}], delta {fmt(v['delta_block_minus_v0'])}"
        for k, v in o["block_alone_DESCRIPTIVE"].items()) + ".")
    L.append("\n## Deviations\n" + "\n".join(f"* {d}" for d in o["deviations"]) + "\n")
    L.append("Tier: DIAGNOSTIC (logged labels and the planner's own logged outputs, no closed loop) -- never driving performance. The CI answers 'another draw of EPISODES'.\n")
    (RES / "CHANNELS_D2.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
