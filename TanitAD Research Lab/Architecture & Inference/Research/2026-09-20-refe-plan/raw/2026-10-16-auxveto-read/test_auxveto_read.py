#!/usr/bin/env python3
"""test_auxveto_read.py -- validation of auxveto_read.py BEFORE any arm's Delta is computed: every §9 validity check as code, analytic known-answer worlds,
and deliberate-regression mutants that MUST turn a check RED.

    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I test_auxveto_read.py     # prints  ZZTEST_PASS <n_checks> <n_mutants>
    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q test_auxveto_read.py

How the red arms work (as `test_geodac.py`): `checks()` runs the whole battery against the module AS IMPORTED and returns {check_name: bool}.  A mutant patches ONE
function / constant of `auxveto_read` (a dropped clause, a swapped weight, an SDF channel read instead of the mask, ...); with it applied, the named detector checks MUST
fail.  A battery that stays green under a mutant is blind to that bug.  No check here computes a raster arm's Delta on the real data (the real-data checks are the
§9 gates: B reproduces the pick, V-PRED's mask reproduces the IoU, the zero control, and the mask-derived signed distance reproducing the label store's).

THE SYNTHETIC WORLD (analytic).  Road = the band -2 < y < 6 m (all x), identical on the 0.5 m GT-like grid and the 2 m predicted-like grid (the edges are on cell edges of both).
Six straight proposals at lateral offsets o = 0, 3, 4.7, 5, 6, 9 m (heading 0, 10 m/s): their footprint's worst corner is at y = o + 1.1485, i.e. a margin of 4.8515 - o to the upper
edge: +0.15, -0.15, -1.15, -4.15 for o = 4.7, 5, 6, 9.  (Every proposal also carries the t = 0 footprint at the origin, whose worst corner is 0.8515 m inside the lower edge.)  So at margin m the vetoed set is
exactly as tabulated in `EXPECT_VETO`.
"""
from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import auxveto_read as av  # noqa: E402
import aux_label_store as A  # noqa: E402  (av puts refe/ on the path)
import geodac as G  # noqa: E402

LIT_WZ, LIT_WN = 529 / 12146, 11617 / 12146        # re-derived literally here, not imported
OFF = np.array([0.0, 3.0, 4.7, 5.0, 6.0, 9.0])
ROAD = (-2.0, 6.0)
# which of the 6 proposals are vetoed at each margin (derived by hand from the footprint arithmetic above)
EXPECT_VETO = {0.0: [3, 4, 5], 0.25: [4, 5], 0.5: [4, 5], 1.0: [4, 5], 2.0: [5]}


def road_masks(lo=ROAD[0], hi=ROAD[1]):
    yc05 = -32 + (np.arange(128) + 0.5) * 0.5
    yc2 = -32 + (np.arange(32) + 0.5) * 2.0
    m_gt = np.broadcast_to(((yc05 > lo) & (yc05 < hi))[None, :], (160, 128)).copy()
    m_pr = np.broadcast_to(((yc2 > lo) & (yc2 < hi))[None, :], (40, 32)).copy()
    return m_gt, m_pr


def straight(o, v=10.0):
    t = np.arange(1, 9) * 0.5
    return np.stack([v * t, np.full(8, float(o)), np.zeros(8)], -1)


# type -> (offsets, sel, dac, nc, ep, expected arm pick at m = 0.25)
TYPES = {
    "A": (OFF, [.5, .3, .2, .1, .4, .9], [1, 1, 1, 1, 0, 0], [1] * 6, [.9, .7, .5, .3, .9, .9], 0),      # Z: B picks the far-off-road proposal; recovered by the veto
    "B": (OFF, [.9, .5, .4, .3, .2, .1], [1] * 6, [1] * 6, [.5] * 6, 0),                                  # N: B's pick on the road, untouched
    "C": (OFF, [.2, .8, .3, .1, .95, .05], [1, 1, 1, 1, 1, 0], [1] * 6, [.2, .6, .4, .3, .9, .9], 1),     # N: B's pick (o = 6) vetoed although NAVSIM says DAC = 1 (false veto)
    "D": ([6, 7, 8, 9, 10, 11], [.1, .2, .3, .9, .4, .5], [0] * 6, [1] * 6, [.5] * 6, 3),                 # Z: every proposal vetoed => B's pick (index 3, not 0)
    "E": (OFF, [.9, .5, .4, .3, .2, .1], [1] * 6, [0, 1, 1, 1, 1, 1], [.5] * 6, 0),                        # Z: NC = 0 at B's pick (DAC = 1): the veto cannot recover it
}
Z_TYPES = "A" * 6 + "D" * 2 + "E" * 2
N_TYPES = "B" * 5 + "C" * 3


def syn_world(order=None):
    kinds = list(Z_TYPES + N_TYPES)
    n = len(kinds)
    props, sel, sub, exp = [], [], [], []
    for k in kinds:
        off, s, dac, nc, ep, ex = TYPES[k]
        props.append(np.stack([straight(o) for o in off]))
        sel.append(s)
        sub.append(np.stack([np.array(nc, float), np.array(dac, float), np.array(ep, float), np.ones(6), np.ones(6), np.ones(6)], -1))
        exp.append(ex)
    props, sel, sub = np.array(props), np.array(sel, float), np.array(sub)
    pdms = sub[..., 0] * sub[..., 1] * (5 * sub[..., 2] + 5 * sub[..., 3] + 2 * sub[..., 4]) / 12.0
    tokens = [f"{(i * 7919 + 13) % 65521:016x}" for i in range(n)]
    logs = [f"log{i % 6}" for i in range(n)]
    is_z = np.array([i < len(Z_TYPES) for i in range(n)])
    w = av.World(token=tokens, log=logs, is_z=is_z, proposals=props, sel=sel, sub=sub, pdms=pdms)
    return w, np.array(exp), kinds


def syn_masks(w, geo):
    m_gt, m_pr = road_masks()
    return {t: (m_gt if geo == "gt" else m_pr) for t in w.token}


def syn_margins(w, geo):
    m = syn_masks(w, geo)
    return av.margins_from_masks(w, m, 0.5 if geo == "gt" else 2.0, G.GT_GEOM if geo == "gt" else G.PRED_GEOM)


def syn_counts(w):
    ix, names = av.log_index(w.log)
    return ix, av.boot_counts(len(names), B=400)


def stratified_literal(x, is_z):
    return LIT_WZ * x[is_z].mean() + LIT_WN * x[~is_z].mean()


# ------------------------------------------------------------------------------------------------------------------ mutant implementations
def mut_no_allveto(sel, veto, b_pick):
    return int(np.argmax(np.where(np.asarray(veto, bool), -np.inf, sel)))


def mut_pick_among_vetoed(sel, veto, b_pick):
    veto = np.asarray(veto, bool)
    if veto.all() or not veto.any():
        return int(b_pick)
    return int(np.argmax(np.where(veto, sel, -np.inf)))


def mut_pred_mask_sdf(logit, sdf):
    return np.asarray(sdf) > 0


def mut_vetoed_wrong_sign(margin, m):
    return np.asarray(margin) < m


def mut_vetoed_nonstrict(margin, m):
    return np.asarray(margin) <= -m


def mut_sdf_inverted(mask, cell_m):
    return -_orig_sdf(mask, cell_m)


def mut_sdf_no_halfcell(mask, cell_m):
    from scipy.ndimage import distance_transform_edt as edt
    m = np.asarray(mask, bool)
    if not m.any():
        return np.full(m.shape, -av.BIG_M)
    if m.all():
        return np.full(m.shape, av.BIG_M)
    return np.where(m, edt(m), -edt(~m)) * cell_m


def mut_margins_last_pose(field, geom, poses):
    return G.geodac(field, geom, np.asarray(poses)[:, -1:, :], horizon="as_given", mode="bilinear", unknown="drivable").min_margin_m


def mut_margins_positional(w, masks, cell_m, geom, *, transform=None):
    vals = list(masks.values())
    out = np.empty(w.sel.shape)
    for i in range(w.n):
        out[i] = av.min_margins(av.sdf_from_mask(vals[i], cell_m), geom, w.proposals[i])
    return out


def mut_assemble_sorted(unit_files, table_tokens):
    toks = sorted(table_tokens)
    n = len(toks)
    logit = np.full((n, 40, 32), np.nan, np.float32)
    sdf = np.full((n, 40, 32), np.nan, np.float32)
    for f in unit_files:
        z = np.load(f, allow_pickle=False)
        logit[z["held_idx"]], sdf[z["held_idx"]] = z["logit"], z["sdf"]
    return {toks[i]: (logit[i], sdf[i]) for i in range(n)}


def mut_strat_boot_unpaired(X, is_z, log_ix, counts):
    """Z and N use DIFFERENT resamples (N's is the previous row): not paired."""
    X = np.atleast_2d(np.asarray(X, float))
    L = counts.shape[1]
    zf, nf = is_z.astype(float), (~is_z).astype(float)
    cn = np.roll(counts, 1, axis=0)
    out = {}
    nz, nn = np.bincount(log_ix, weights=zf, minlength=L), np.bincount(log_ix, weights=nf, minlength=L)
    sz = np.stack([np.bincount(log_ix, weights=x * zf, minlength=L) for x in X])
    sn = np.stack([np.bincount(log_ix, weights=x * nf, minlength=L) for x in X])
    bs = av.W_Z * ((counts @ sz.T) / (counts @ nz)[:, None]) + av.W_N * ((cn @ sn.T) / (cn @ nn)[:, None])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    pt = av.W_Z * X[:, is_z].mean(1) + av.W_N * X[:, ~is_z].mean(1)
    out.update(point=pt, lo=lo, hi=hi, z=X[:, is_z].mean(1), z_lo=lo, z_hi=hi, n=X[:, ~is_z].mean(1), n_lo=lo, n_hi=hi)
    return out


def mut_strat_boot_resampled_weights(X, is_z, log_ix, counts):
    """The weights are recomputed from each resample's own Z share (the prereg says they stay FIXED)."""
    X = np.atleast_2d(np.asarray(X, float))
    L = counts.shape[1]
    zf, nf = is_z.astype(float), (~is_z).astype(float)
    nz, nn = np.bincount(log_ix, weights=zf, minlength=L), np.bincount(log_ix, weights=nf, minlength=L)
    sz = np.stack([np.bincount(log_ix, weights=x * zf, minlength=L) for x in X])
    sn = np.stack([np.bincount(log_ix, weights=x * nf, minlength=L) for x in X])
    dz, dn = counts @ nz, counts @ nn
    wz = (dz / (dz + dn))[:, None]
    bs = wz * ((counts @ sz.T) / dz[:, None]) + (1 - wz) * ((counts @ sn.T) / dn[:, None])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    pt = av.W_Z * X[:, is_z].mean(1) + av.W_N * X[:, ~is_z].mean(1)
    return dict(point=pt, lo=lo, hi=hi, z=X[:, is_z].mean(1), z_lo=lo, z_hi=hi, n=X[:, ~is_z].mean(1), n_lo=lo, n_hi=hi)


def mut_iou_wrong_window(pred, truth, lo=0, hi=40):
    p, t = pred[:, lo:hi], truth[:, lo:hi]
    return float((p & t).sum() / (p | t).sum())


def mut_b_picks_argmin(sel):
    return np.asarray([int(np.argmin(s)) for s in sel])


def mut_odac_inverted(w):
    return w.sub[:, :, 1] == 1


def mut_decide_gt(vpred, vgt):
    """AV-GO / AV-PROMISING boundary at a strict > 0.30 instead of >= 0.30."""
    lo, hi, pt = vpred["lo"], vpred["hi"], vpred["point"]
    if lo > 0 and pt > 0.30:
        return {"row": "AV-GO", "gt_branch": None}
    if lo > 0:
        return {"row": "AV-PROMISING", "gt_branch": None}
    return _orig_decide(vpred, vgt)


def mut_margins_ignore_transform(w, masks, cell_m, geom, *, transform=None):
    return _orig_margins_from_masks(w, masks, cell_m, geom, transform=None)


def mut_pick_outcomes_dac_only(w, picks):
    o = _orig_pick_outcomes(w, picks)
    o["rec_ncdac"] = o["rec_dac"]
    return o


def mut_deltas_reversed(w, picks):
    d = _orig_deltas(w, picks)
    return {k: -v for k, v in d.items()}


_orig_sdf = av.sdf_from_mask
_orig_decide = av.decide
_orig_pick_outcomes = av.pick_outcomes
_orig_deltas = av.deltas
_orig_margins_from_masks = av.margins_from_masks
_orig_prereg_path = av.PREREG_PATH


# ------------------------------------------------------------------------------------------------------------------ real-data cache (labels / table do not depend on any mutant)
_REAL = {}


def real():
    if not _REAL:
        w, banked = av.load_world()
        _REAL["w"], _REAL["banked"] = w, banked
        _REAL["gt"] = av.load_gt_masks(w)
        _REAL["gt2"] = np.stack([av.gt_mask_to_2m(_REAL["gt"][t]) for t in w.token])
    return _REAL


def real_subset_world(n=200):
    w = real()["w"]
    return av.World(token=w.token[:n], log=w.log[:n], is_z=w.is_z[:n], proposals=w.proposals[:n], sel=w.sel[:n], sub=w.sub[:n], pdms=w.pdms[:n])


# ------------------------------------------------------------------------------------------------------------------ the battery
def checks(include_real: bool = True) -> dict:
    out = {}

    def ok(name, cond):
        out[name] = bool(cond)

    # ---- sdf_from_mask --------------------------------------------------------------------------------------------------
    m_gt, m_pr = road_masks()
    s_gt, s_pr = av.sdf_from_mask(m_gt, 0.5), av.sdf_from_mask(m_pr, 2.0)
    yc05 = -32 + (np.arange(128) + 0.5) * 0.5
    yc2 = -32 + (np.arange(32) + 0.5) * 2.0
    exp05 = np.minimum(yc05 - ROAD[0], ROAD[1] - yc05)
    exp2 = np.minimum(yc2 - ROAD[0], ROAD[1] - yc2)
    # a cell centre is `|y - nearest edge|` from the boundary: the half-cell-corrected Euclidean distance is exactly that, everywhere on these two grids
    ok("sdf_band_analytic", np.allclose(s_gt, exp05[None, :], atol=1e-9) and np.allclose(s_pr, exp2[None, :], atol=1e-9))
    ok("sdf_sign_positive_inside", (s_gt[m_gt] > 0).all() and (s_gt[~m_gt] < 0).all() and (s_pr[m_pr] > 0).all() and (s_pr[~m_pr] < 0).all())
    rng = np.random.default_rng(3)
    eq = True
    for _ in range(5):
        mk = rng.random((160, 128)) > 0.6
        eq &= bool(np.allclose(av.sdf_from_mask(mk, 0.5), A.signed_distance(mk), atol=1e-12))
    ok("sdf_equals_label_store_function", eq)
    ok("sdf_degenerate_branches", (av.sdf_from_mask(np.ones((40, 32), bool), 2.0) == av.BIG_M).all() and (av.sdf_from_mask(np.zeros((40, 32), bool), 2.0) == -av.BIG_M).all())

    # ---- pred_mask: the logit, not the SDF channel -----------------------------------------------------------------------------
    lg, sd = np.array([[1.0, -1.0, 0.5]]), np.array([[-1.0, 1.0, 0.5]])           # the two channels disagree on the first two cells
    ok("pred_mask_is_logit_sign", (av.pred_mask(lg, sd) == (lg > 0)).all() and not (av.pred_mask(lg, sd) == (sd > 0)).all())

    # ---- veto statistic, the strict threshold, the extent ----------------------------------------------------------------------
    ok("vetoed_strict_threshold", (av.vetoed(np.array([-0.25, -0.2500001, -0.2499999, 0.1]), 0.25) == [False, True, False, False]).all())
    w, exp, kinds = syn_world()
    mg_gt, mg_pr = syn_margins(w, "gt"), syn_margins(w, "pred")
    expm = np.minimum(0.8515, np.minimum(OFF + 0.8515, 4.8515 - OFF))      # the t = 0 footprint sits at the origin (y = 0): its right corner is 0.8515 m inside the lower edge
    atype = [i for i, k in enumerate(kinds) if k == "A"][0]
    ok("margins_analytic_gt_grid", np.allclose(mg_gt[atype], expm, atol=1e-9))
    ok("margins_analytic_pred_grid", np.allclose(mg_pr[atype], expm, atol=1e-9))
    sets_ok = True
    for m, vs in EXPECT_VETO.items():
        for mg in (mg_gt, mg_pr):
            v = av.vetoed(mg[atype], m)
            sets_ok &= sorted(np.where(v)[0].tolist()) == vs
    ok("veto_sets_at_margins", sets_ok)
    exc = np.stack([[5.0 * t, y, 0.0] for t, y in zip(np.arange(1, 9) * 0.5, [0, 1, 7, 7, 1, 0, 0, 0])])[None]         # out and back: only mid-horizon steps are off the road
    mgx = av.min_margins(s_gt, G.GT_GEOM, exc)
    ok("excursion_vetoed", av.vetoed(mgx, 0.25).all() and not av.vetoed(av.min_margins(s_gt, G.GT_GEOM, straight(0.0)[None]), 0.25).any())
    far = np.array([[[100.0 * t, 0.0, 0.0] for t in np.arange(1, 9) * 0.5]])                                              # entirely beyond x = 72
    ok("unknown_outside_extent_not_vetoed", not av.vetoed(av.min_margins(s_gt, G.GT_GEOM, far), 2.0).any())

    # ---- the pick rule -------------------------------------------------------------------------------------------------------
    sel = np.array([.5, .3, .9, .7, .2])
    cases = [(np.array([0, 0, 1, 0, 0], bool), 2, 3), (np.zeros(5, bool), 2, 2), (np.ones(5, bool), 2, 2), (np.array([1, 1, 1, 1, 0], bool), 2, 4),
             (np.array([1, 0, 0, 0, 0], bool), 0, 2)]
    ok("masked_pick_cases", all(av.masked_pick(sel, v, b) == e for v, b, e in cases) and av.masked_pick(np.array([.5, .5, .5]), np.array([1, 0, 0], bool), 0) == 1)
    veto25 = av.vetoed(mg_gt, 0.25)
    picks = av.all_picks(w, veto25)
    ok("world_arm_picks_synthetic", (picks == exp).all() and (w.b_pick == np.array([5] * 6 + [3] * 2 + [0] * 2 + [0] * 5 + [4] * 3)).all())
    ok("pred_grid_picks_equal_gt_grid_picks", (av.all_picks(w, av.vetoed(mg_pr, 0.25)) == picks).all())

    # ---- outcomes, the estimator and the §8 counts on the synthetic world ----------------------------------------------------------------
    ix, counts = syn_counts(w)
    d = av.deltas(w, picks)
    exp_d = 100 * (w.pdms[np.arange(w.n), exp] - w.pdms[np.arange(w.n), w.b_pick])
    ok("deltas_synthetic_analytic", np.allclose(d["pdms"], exp_d) and abs(exp_d[:6].mean() - 100 * (w.pdms[0, 0])) < 1e-9 and (exp_d[6:15] == 0).all() and (exp_d[15:] < 0).all()
       and d["pdms"][0] > 90.0)
    res = av.strat_boot(d["pdms"], w.is_z, ix, counts)
    ok("stratified_point_uses_prereg_weights", abs(res["point"][0] - stratified_literal(exp_d, w.is_z)) < 1e-9)
    ok("weights_constants", av.N_Z == 529 and av.N_N == 11617 and abs(av.W_Z - LIT_WZ) < 1e-15 and abs(av.W_N - LIT_WN) < 1e-15 and abs(av.W_Z + av.W_N - 1) < 1e-15)
    blk = av.arm_block(w, veto25, picks, ix, counts)
    ok("z_recovery_counts", blk["z_recovery"]["ncdac_count"] == 6 and blk["z_recovery"]["dac_only_count"] == 8 and blk["z_recovery"]["of"] == 10)
    ok("false_veto_counts", blk["false_veto_N"]["b_pick_vetoed_count"] == 3 and blk["false_veto_N"]["of"] == 8 and blk["false_veto_N"]["b_pick_vetoed_in_Z_count"] == 6 + 2)
    ok("all_vetoed_counts", blk["all_vetoed_tokens"] == {"Z": 2, "N": 0} and blk["changed_picks"] == {"Z": 6, "N": 3})

    # ---- the bootstrap ---------------------------------------------------------------------------------------------------------------------
    L = len(set(w.log))
    rng = np.random.default_rng(20260927)
    first = np.bincount(rng.integers(0, 123, 123), minlength=123)
    c10k = av.boot_counts(123)
    ok("bootstrap_first_row_matches_spec", av.B_BOOT == 10000 and av.SEED == 20260927 and c10k.shape == (10000, 123) and (c10k[0] == first).all() and (c10k.sum(1) == 123).all())
    members = [np.where(ix == l)[0] for l in range(L)]
    X = np.random.default_rng(5).normal(size=(2, w.n))
    ref = []
    rng = np.random.default_rng(20260927)
    for b in range(400):
        idx = rng.integers(0, L, L)
        toks = np.concatenate([members[l] for l in idx])
        zt, nt = toks[w.is_z[toks]], toks[~w.is_z[toks]]
        ref.append([LIT_WZ * X[k][zt].mean() + LIT_WN * X[k][nt].mean() for k in range(2)])
    lo_hi = np.percentile(np.array(ref), [2.5, 97.5], axis=0)
    r2 = av.strat_boot(X, w.is_z, ix, counts)
    ok("bootstrap_matches_bruteforce", np.allclose(r2["lo"], lo_hi[0], atol=1e-9) and np.allclose(r2["hi"], lo_hi[1], atol=1e-9))
    same = av.strat_boot(np.zeros(w.n), w.is_z, ix, counts)
    ok("bootstrap_identical_arms_ci_zero", same["point"][0] == 0.0 and same["lo"][0] == 0.0 and same["hi"][0] == 0.0)

    # ---- §9 validity checks as code (logic) ----------------------------------------------------------------------------------------------------
    b = np.arange(10)
    ok("check1_logic", av.check_b_reproduces(b, b)[0] and not av.check_b_reproduces(np.r_[b[:-1], 99], b)[0])
    ok("check2_logic", av.check_iou_identity(0.675218)[0] and av.check_iou_identity(0.675218 + 9e-7)[0] and not av.check_iou_identity(0.675218 + 2e-6)[0] and not av.check_iou_identity(0.6188)[0])
    ok("check3_logic", av.check_zero_control(0, 0.0, True)[0] and not av.check_zero_control(1, 0.0, True)[0] and not av.check_zero_control(0, 1e-12, True)[0] and not av.check_zero_control(0, 0.0, False)[0])
    ok("check4_logic", av.check_odac_bounds(10, {"a": 10, "b": 4})[0] and not av.check_odac_bounds(10, {"a": 11})[0])
    ok("check5_logic", av.check_yflip_below(1.0, 0.5)[0] and not av.check_yflip_below(1.0, 1.0)[0] and not av.check_yflip_below(1.0, 1.2)[0])
    # zero control through the whole pipeline, both geometries
    zc = True
    for geo, shape, cell, geom in (("pred", (40, 32), 2.0, G.PRED_GEOM), ("gt", (160, 128), 0.5, G.GT_GEOM)):
        mg = av.margins_from_masks(w, {t: np.ones(shape, bool) for t in w.token}, cell, geom)
        v = av.vetoed(mg, av.MARGIN)
        pk = av.all_picks(w, v)
        r0 = av.strat_boot(av.deltas(w, pk)["pdms"], w.is_z, ix, counts)
        zc &= av.check_zero_control(int(v.sum()), float(r0["point"][0]), bool((pk == w.b_pick).all()))[0]
    ok("zero_control_synthetic", zc)
    # O-DAC bounds the raster arms (check 4 on the synthetic world; all-N tokens have NC x DAC > 0 so only Z tokens count)
    recs = {}
    for nm, mg in (("gt", mg_gt), ("pred", mg_pr)):
        recs[nm] = float(av.pick_outcomes(w, av.all_picks(w, av.vetoed(mg, av.MARGIN)))["rec_ncdac"][w.is_z].sum())
    rec_o = float(av.pick_outcomes(w, av.all_picks(w, av.odac_veto(w)))["rec_ncdac"][w.is_z].sum())
    ok("odac_bounds_synthetic", av.check_odac_bounds(rec_o, recs)[0] and rec_o == 6.0)
    ok("odac_veto_is_dac_zero", (av.odac_veto(w) == (w.sub[:, :, 1] == 0)).all())
    # check 5 on the synthetic world: the y-flipped asymmetric road reads lower than the road
    mg_flip = av.margins_from_masks(w, syn_masks(w, "gt"), 0.5, G.GT_GEOM, transform=lambda m: m[:, ::-1])
    d_gt = av.strat_boot(av.deltas(w, av.all_picks(w, av.vetoed(mg_gt, av.MARGIN)))["pdms"], w.is_z, ix, counts)["point"][0]
    d_fl = av.strat_boot(av.deltas(w, av.all_picks(w, av.vetoed(mg_flip, av.MARGIN)))["pdms"], w.is_z, ix, counts)["point"][0]
    ok("yflip_synthetic", av.check_yflip_below(d_gt, d_fl)[0])

    # ---- token-string alignment ---------------------------------------------------------------------------------------------------------------------------
    sel2 = np.array([[.2, .9, .3, .1, .4, .5]] * 2)                    # wide road: {4,5} vetoed -> pick 1 ; narrow road (-2, 2): {1..5} vetoed -> pick 0
    w2 = av.World(token=["aaaa", "bbbb"], log=["l0", "l1"], is_z=np.array([True, False]), proposals=np.stack([np.stack([straight(o) for o in OFF])] * 2), sel=sel2, sub=w.sub[:2], pdms=w.pdms[:2])
    wide, narrow = road_masks(-2, 6)[0], road_masks(-2, 2)[0]
    mk = {"aaaa": wide, "bbbb": narrow}
    base = av.all_picks(w2, av.vetoed(av.margins_from_masks(w2, mk, 0.5, G.GT_GEOM), 0.25))
    wp = av.World(token=["bbbb", "aaaa"], log=["l1", "l0"], is_z=np.array([False, True]), proposals=w2.proposals[::-1].copy(), sel=w2.sel[::-1].copy(), sub=w2.sub[::-1].copy(), pdms=w2.pdms[::-1].copy())
    perm = av.all_picks(wp, av.vetoed(av.margins_from_masks(wp, mk, 0.5, G.GT_GEOM), 0.25))
    ok("margins_by_token_string", base[0] != base[1] and perm[0] == base[1] and perm[1] == base[0])
    with tempfile.TemporaryDirectory(dir="C:/Users/Admin/tmpavr") as td:
        toks = ["t3", "t0", "t2", "t1"]                                   # table order differs from sorted order
        files = []
        for k, rows in enumerate(([0, 2], [1, 3])):
            lg = np.stack([np.full((40, 32), float(int(toks[r][1]) + 1), np.float32) for r in rows])
            f = Path(td) / f"V_s0_f{k}.npz"
            np.savez(f, held_idx=np.array(rows), logit=lg, sdf=lg)
            files.append(f)
        u = av.assemble_units(files, toks)
        ok("assemble_units_by_table_order", all(float(u[t][0][0, 0]) == float(int(t[1]) + 1) for t in toks))

    # ---- the decision list, the prereg guard ------------------------------------------------------------------------------------------------------------------------------------------
    D = av.decide
    ok("decide_rows", D({"lo": .1, "hi": 1, "point": .30}, {"lo": 0})["row"] == "AV-GO" and D({"lo": .1, "hi": 1, "point": .2999999}, {"lo": 0})["row"] == "AV-PROMISING"
       and D({"lo": -1, "hi": -.1, "point": -.5}, {"lo": .2})["row"] == "AV-HARM" and "S1c" in D({"lo": -1, "hi": -.1, "point": -.5}, {"lo": .2})["gt_branch"]
       and D({"lo": -1, "hi": -.1, "point": -.5}, {"lo": -.2})["gt_branch"].startswith("GT ceiling not positive")
       and D({"lo": -.2, "hi": .3, "point": .1}, {"lo": .05})["row"] == "AV-CEILING-ONLY" and D({"lo": -.2, "hi": .3, "point": .1}, {"lo": 0.0})["row"] == "AV-NULL"
       and D({"lo": 0.0, "hi": .3, "point": .2}, {"lo": 0.0})["row"] == "AV-NULL" and D({"lo": -.3, "hi": 0.0, "point": -.1}, {"lo": .1})["row"] == "AV-CEILING-ONLY")
    ok("prereg_next_arm_verbatim", av.prereg_next_arm("AV-GO").startswith('**"separated, single head seed, not yet a lever claim"**') and "seed-1 head replicate" in av.prereg_next_arm("AV-GO")
       and av.prereg_next_arm("AV-NULL").startswith("the TRACKED variant"))
    guard = True
    try:
        av.assert_prereg()
    except SystemExit:
        guard = False
    with tempfile.TemporaryDirectory(dir="C:/Users/Admin/tmpavr") as td:
        bad = Path(td) / "PREREG_AUXVETO.md"
        bad.write_bytes(av.PREREG_PATH.read_bytes() + b" ")
        with mock.patch.object(av, "PREREG_PATH", bad):
            try:
                av.assert_prereg()
                guard = False
            except SystemExit:
                pass
    ok("prereg_guard", guard)

    # ---- REAL DATA: the §9 gates ------------------------------------------------------------------------------------------------------------------------------------------------------------
    if include_real:
        R = real()
        wf, banked = av.load_world()                                     # a fresh world: a mutated `b_picks` shows up here
        ok("real_B_reproduces", av.check_b_reproduces(wf.b_pick, banked)[0] and wf.n == 1058)
        ok("real_population", int(R["w"].is_z.sum()) == 529 and (R["w"].sub[np.arange(1058), R["w"].b_pick, 0] * R["w"].sub[np.arange(1058), R["w"].b_pick, 1])[R["w"].is_z].max() == 0.0)
        pm, _ = av.load_pred_masks(R["w"], av.UNIT_DIRS["V-PRED"])
        iou = av.iou_pooled(av.stack_masks(R["w"], pm), R["gt2"])
        ok("real_iou_identity", av.check_iou_identity(iou)[0])
        banked_geo = json.load(open(av.PRED_GEOMETRY))
        i2, _ = av.load_pred_masks(R["w"], av.UNIT_DIRS["V-PRED-2k"])
        i4, _ = av.load_pred_masks(R["w"], av.UNIT_DIRS["V-PRED-400"])
        ok("real_iou_2k_400_vs_banked", abs(av.iou_pooled(av.stack_masks(R["w"], i2), R["gt2"]) - banked_geo["PROBE_2000_updates_V_s0"]["iou_x0_40m_identity"]) < 1e-6
           and abs(av.iou_pooled(av.stack_masks(R["w"], i4), R["gt2"]) - banked_geo["A3R_400_updates"]["iou_x0_40m_identity"]) < 1e-6)
        ws = real_subset_world(200)
        ixs = av.log_index(ws.log)[0]
        cs = av.boot_counts(len(set(ws.log)), B=200)
        zc = True
        for shape, cell, geom in (((40, 32), 2.0, G.PRED_GEOM), ((160, 128), 0.5, G.GT_GEOM)):
            mg = av.margins_from_masks(ws, {t: np.ones(shape, bool) for t in ws.token}, cell, geom)
            v = av.vetoed(mg, av.MARGIN)
            pk = av.all_picks(ws, v)
            r0 = av.strat_boot(av.deltas(ws, pk)["pdms"], ws.is_z, ixs, cs)
            zc &= av.check_zero_control(int(v.sum()), float(r0["point"][0]), bool((pk == ws.b_pick).all()))[0]
        ok("real_zero_control_subset", zc)
        worst, n_cmp = 0.0, 0
        for t, lg in list(zip(R["w"].token, R["w"].log))[:40]:
            st = A.read_label(av.STORE, lg, t).astype(np.float64) * 0.1
            mine = av.sdf_from_mask(R["gt"][t], 0.5)
            ixg, iyg = np.meshgrid(np.arange(160), np.arange(128), indexing="ij")
            edge = np.minimum.reduce([ixg + 0.5, 160 - ixg - 0.5, iyg + 0.5, 128 - iyg - 0.5]) * 0.5
            valid = (np.abs(st) < 12.6) & (edge >= np.abs(st) + 0.5)
            worst, n_cmp = max(worst, float(np.abs(mine - st)[valid].max())), n_cmp + int(valid.sum())
        ok("real_sdf_matches_stored_int8", worst <= 0.0501 and n_cmp > 100000)
    return out


# ------------------------------------------------------------------------------------------------------------------ mutants
@contextlib.contextmanager
def patched(spec):
    with contextlib.ExitStack() as st:
        for name, new in spec.items():
            st.enter_context(mock.patch.object(av, name, new))
        yield


MUTANTS = {
    "no_allveto_clause": ({"masked_pick": mut_no_allveto}, ["masked_pick_cases", "world_arm_picks_synthetic"]),
    "sdf_channel_instead_of_mask": ({"pred_mask": mut_pred_mask_sdf}, ["pred_mask_is_logit_sign", "real_iou_identity"]),
    "swap_stratum_weights": ({"W_Z": av.W_N, "W_N": av.W_Z}, ["stratified_point_uses_prereg_weights", "weights_constants", "bootstrap_matches_bruteforce"]),
    "veto_wrong_sign": ({"vetoed": mut_vetoed_wrong_sign}, ["veto_sets_at_margins", "zero_control_synthetic", "vetoed_strict_threshold"]),
    "veto_nonstrict": ({"vetoed": mut_vetoed_nonstrict}, ["vetoed_strict_threshold"]),
    "pick_among_vetoed": ({"masked_pick": mut_pick_among_vetoed}, ["masked_pick_cases", "world_arm_picks_synthetic"]),
    "sdf_sign_inverted": ({"sdf_from_mask": mut_sdf_inverted}, ["sdf_band_analytic", "sdf_sign_positive_inside", "zero_control_synthetic", "real_sdf_matches_stored_int8"]),
    "sdf_no_halfcell_correction": ({"sdf_from_mask": mut_sdf_no_halfcell}, ["sdf_equals_label_store_function", "veto_sets_at_margins", "real_sdf_matches_stored_int8"]),
    "margins_last_pose_only": ({"min_margins": mut_margins_last_pose}, ["excursion_vetoed"]),
    "margins_by_row_position": ({"margins_from_masks": mut_margins_positional}, ["margins_by_token_string"]),
    "yflip_transform_ignored": ({"margins_from_masks": mut_margins_ignore_transform}, ["yflip_synthetic"]),
    "assemble_units_sorted_tokens": ({"assemble_units": mut_assemble_sorted}, ["assemble_units_by_table_order", "real_iou_identity"]),
    "bootstrap_seed_changed": ({"SEED": 20260928}, ["bootstrap_first_row_matches_spec", "bootstrap_matches_bruteforce"]),
    "bootstrap_unpaired_strata": ({"strat_boot": mut_strat_boot_unpaired}, ["bootstrap_matches_bruteforce"]),
    "bootstrap_weights_from_resample": ({"strat_boot": mut_strat_boot_resampled_weights}, ["bootstrap_matches_bruteforce"]),
    "iou_window_wrong": ({"iou_pooled": mut_iou_wrong_window}, ["real_iou_identity", "real_iou_2k_400_vs_banked"]),
    "b_picks_argmin": ({"b_picks": mut_b_picks_argmin}, ["real_B_reproduces", "world_arm_picks_synthetic"]),
    "odac_veto_inverted": ({"odac_veto": mut_odac_inverted}, ["odac_bounds_synthetic", "odac_veto_is_dac_zero"]),
    "decide_strict_gt_030": ({"decide": mut_decide_gt}, ["decide_rows"]),
    "recovery_dac_only": ({"pick_outcomes": mut_pick_outcomes_dac_only}, ["z_recovery_counts", "odac_bounds_synthetic"]),
    "deltas_reversed": ({"deltas": mut_deltas_reversed}, ["deltas_synthetic_analytic", "stratified_point_uses_prereg_weights"]),
    "prereg_guard_off": ({"PREREG_PREFIX": ""}, ["prereg_guard"]),
}


def run_mutants() -> dict:
    real()                                                              # load the real-data cache UNPATCHED, before any mutant is applied
    res = {}
    for name, (spec, detectors) in MUTANTS.items():
        with patched(spec):
            try:
                out = checks(include_real=any(d.startswith("real_") for d in detectors))
            except SystemExit as e:                                        # a mutant that makes the module refuse is RED as well
                out = {c: False for c in detectors}
                out["__refused__"] = repr(e)[:100]
            except Exception as e:                                         # noqa: BLE001 -- a crash is RED (noted)
                out = {c: False for c in detectors}
                out["__crashed__"] = repr(e)[:120]
        red = sorted(k for k, v in out.items() if v is False)
        res[name] = {"detectors": detectors, "detected": any(out.get(c) is False for c in detectors), "detectors_red": [c for c in detectors if out.get(c) is False],
                     "n_red": len(red), "red": red[:10]}
    return res


def test_battery_is_green():
    r = checks()
    bad = [k for k, v in r.items() if not v]
    assert not bad, bad


def test_every_mutant_goes_red():
    res = run_mutants()
    miss = {k: v for k, v in res.items() if not v["detected"]}
    assert not miss, miss


def test_baseline_is_restored_after_mutants():
    run_mutants()
    assert all(checks().values())


if __name__ == "__main__":
    base = checks()
    bad = [k for k, v in base.items() if not v]
    mut = run_mutants()
    for k, v in mut.items():
        print(f"mutant {k:34s} detected={'OK ' if v['detected'] else 'FAIL'} n_red={v['n_red']:2d}  detectors_red={v['detectors_red']}")
    after = [k for k, v in checks().items() if not v]
    miss = [k for k, v in mut.items() if not v["detected"]]
    print(f"baseline: {len(base) - len(bad)}/{len(base)} green; failing={bad}; after-mutants failing={after}; mutants not red={miss}")
    if bad or after or miss:
        print("ZZTEST_FAIL")
        sys.exit(1)
    print(f"ZZTEST_PASS {len(base)} {len(mut)}")
