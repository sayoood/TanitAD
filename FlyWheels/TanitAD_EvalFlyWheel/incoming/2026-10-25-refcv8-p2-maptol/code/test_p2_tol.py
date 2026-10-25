"""test_p2_tol.py -- the P2 instrument's controls with KNOWN answers (analytic / independent algorithm / mutation).

Run:  PYTHONPATH=<tree>/taniteval:<tree>/stack python -X utf8 test_p2_tol.py     (prints ZZ-tokens, exit code != 0 on any failure)

Expectations are LITERALS or closed forms derived on paper (strip geometry), never an expression over the code under test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2_tol_lib as T  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if (detail and not cond) else ""))
    if not cond:
        FAILS.append(name)


def strip_scene(w, d, H=400, W=100, x0=30, cls=2):
    gt = np.zeros((H, W), np.uint8)
    pr = np.zeros((H, W), np.uint8)
    gt[:, x0:x0 + w] = cls
    pr[:, x0 + d:x0 + d + w] = cls
    return pr, gt


def test_strips_literal():
    # hand-derived literals: (w, d) -> {k: tpp_k per ROW}; inter per row = max(0, w-d); pred = gt = w per row
    # w=1,d=0 : exact; w=1,d=1: k=1 hits; w=2,d=1: k=1 covers both; w=2,d=3: k=1 none, k=2 one of two, k=5 both
    lit = {
        (1, 0): {"inter": 1, 1: 1, 2: 1, 5: 1, 10: 1},
        (1, 1): {"inter": 0, 1: 1, 2: 1, 5: 1, 10: 1},
        (1, 2): {"inter": 0, 1: 0, 2: 1, 5: 1, 10: 1},
        (2, 1): {"inter": 1, 1: 2, 2: 2, 5: 2, 10: 2},
        (2, 3): {"inter": 0, 1: 0, 2: 1, 5: 2, 10: 2},
        (3, 6): {"inter": 0, 1: 0, 2: 0, 5: 2, 10: 3},
    }
    for (w, d), exp in lit.items():
        pr, gt = strip_scene(w, d)
        st = T.tol_stats(pr, gt)
        rows = 200
        ok = True
        for b in (0, 1):
            ok &= st["pred"][2, b] == w * rows and st["gt"][2, b] == w * rows
            ok &= st["inter"][2, b] == exp["inter"] * rows
            for k in T.KS:
                ok &= st[f"tpp{k}"][2, b] == exp[k] * rows and st[f"tpg{k}"][2, b] == exp[k] * rows
        check(f"strip literal w={w} d={d}", bool(ok))


def test_strips_closed_form():
    bad = []
    for w in (1, 2, 3, 5):
        for d in (0, 1, 2, 3, 5, 7, 12):
            pr, gt = strip_scene(w, d, W=120)
            st = T.tol_stats(pr, gt)
            for k in T.KS:
                exp = min(w, max(0, k + w - d)) * 200          # derived on paper: pred col x0+d+j is within k of the GT strip iff j <= k+w-1-d
                if st[f"tpp{k}"][2, 0] != exp or st[f"tpg{k}"][2, 0] != exp:
                    bad.append((w, d, k))
            iou0 = st["inter"][2, 0] / st["union"][2, 0]
            if abs(iou0 - T.analytic_iou0_strip(w, d)) > 1e-12:
                bad.append(("iou0", w, d))
    check("strip closed form (28 geometries x 4 k x 2 sides + exact IoU)", not bad, str(bad[:4]))


def random_scene(rng, H=400, W=120):
    # blobby multi-class map: smoothed noise quantised to codes, plus a not-seen margin
    from scipy.ndimage import gaussian_filter
    z = gaussian_filter(rng.standard_normal((H, W)), 6)
    g = np.digitize(z, np.quantile(z, [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])).astype(np.uint8)
    g[:5] = 255
    g[:, -7:] = 255
    return g


def test_gt_as_prediction_and_all_one_class():
    rng = np.random.default_rng(1)
    g = random_scene(rng)
    st = T.tol_stats(np.where(g == 255, 0, g).astype(np.uint8), g)   # a prediction is a class everywhere; unseen cells are not scored
    ok, nclass = True, 0
    for c in range(8):
        if st["gt"][c].sum() == 0:
            continue
        nclass += 1
        for b in range(st["gt"].shape[1]):
            n = st["gt"][c, b]
            ok &= st["pred"][c, b] == n and st["inter"][c, b] == n
            for k in T.KS:
                ok &= st[f"tpp{k}"][c, b] == n and st[f"tpg{k}"][c, b] == n
    P, R, F, I = T.prf_from_counts(st["pred"], st["gt"], st["tpp5"], st["tpg5"])
    msk = st["gt"] > 0
    ok &= bool(np.all(I[msk] == 1.0))
    check(f"GT-as-prediction: P=R=F=IoU=1.0 at every k ({nclass} classes)", bool(ok))
    one = np.full_like(g, 1)
    st = T.tol_stats(one, g)
    ok = True
    for c in range(8):
        if c == 1 or st["gt"][c].sum() == 0:
            continue
        for k in T.KS:
            ok &= int(st[f"tpg{k}"][c].sum()) == 0 and int(st[f"tpp{k}"][c].sum()) == 0 and int(st["inter"][c].sum()) == 0
        _, R, F, I = T.prf_from_counts(st["pred"][c], st["gt"][c], st["tpp10"][c], st["tpg10"][c])
        ok &= bool(np.all(np.nan_to_num(I[st["gt"][c] > 0], nan=-1.0) == 0.0))
    check("all-one-class prediction: IoU_k = 0 for every other class at every k", bool(ok))


def test_diagonal_literal():
    # one GT cell and one pred cell at diagonal offset d: Chebyshev distance d (a taxicab metric would say 2d)
    for d, hit in ((1, (1, 1, 1, 1)), (2, (0, 1, 1, 1)), (3, (0, 0, 1, 1)), (6, (0, 0, 0, 1))):
        gt = np.zeros((400, 100), np.uint8)
        pr = np.zeros((400, 100), np.uint8)
        gt[100, 40] = 2
        pr[100 + d, 40 + d] = 2
        st = T.tol_stats(pr, gt)
        got = tuple(int(st[f"tpp{k}"][2, 0]) for k in T.KS)
        check(f"diagonal d={d}: tpp at k=(1,2,5,10) == {hit}", got == hit, str(got))
    # MUTATION: the taxicab metric must be DETECTED by that same literal (the test is not vacuous)
    d, hit = 3, (0, 0, 1, 1)
    gt = np.zeros((400, 100), np.uint8)
    pr = np.zeros((400, 100), np.uint8)
    gt[100, 40] = 2
    pr[100 + d, 40 + d] = 2
    st = T.tol_stats(pr, gt, _metric="taxicab")
    got = tuple(int(st[f"tpp{k}"][2, 0]) for k in T.KS)
    check("mutation (taxicab metric) is caught by the diagonal literal", got != hit, str(got))


def test_vs_independent_instruments():
    import torch
    import diag_metrics as dm
    from taniteval import map_hires_metrics as MH
    rng = np.random.default_rng(7)
    bad, n = [], 0
    for trial in range(6):
        g = random_scene(rng)
        # prediction = GT with a random lateral jitter per row-block + label noise
        p = np.where(g == 255, 0, g).astype(np.uint8)
        for r0 in range(0, p.shape[0], 40):
            sh = int(rng.integers(-4, 5))
            p[r0:r0 + 40] = np.roll(p[r0:r0 + 40], sh, axis=1)
        flip = rng.random(p.shape) < 0.03
        p[flip] = rng.integers(0, 8, flip.sum()).astype(np.uint8)
        st = T.tol_stats(p, g)
        # (a) max-pool dilation instrument (torch) -- a different algorithm
        gt_t = torch.from_numpy(g.astype(np.int64))[None]
        sup = gt_t != 255
        G = dm.onehot_masks(gt_t, sup)
        P = dm.onehot_masks(torch.from_numpy(p.astype(np.int64))[None], sup)
        M = dm.band_matrix(g.shape[0], "cpu")
        tc = dm.tolerant_counts(P, G, M)
        for nm_mine, nm_dm in [("pred", "pred"), ("gt", "gt"), ("inter", "inter")] + \
                [(f"tpp{k}", f"tpp{k}") for k in T.KS] + [(f"tpg{k}", f"tpg{k}") for k in T.KS]:
            a = st[nm_mine]
            b = tc[nm_dm][0].numpy().astype(np.int64)
            n += a.size
            if not np.array_equal(a, b):
                bad.append((trial, nm_mine))
        # (b) the programme's Euclidean-EDT stats (map_hires_metrics.window_stats): k=1 chessboard == EDT < 0.2 m
        ws = MH.window_stats(p, g)
        for c in (2, 3, 5):
            n += 2
            if not np.array_equal(ws["tp_pred"][c], st["tpp1"][c]) or not np.array_equal(ws["tp_gt"][c], st["tpg1"][c]):
                bad.append((trial, "EDT<0.2 vs cheb k=1", c))
        for nm in ("inter", "union"):
            if not np.array_equal(ws[nm], st[nm]):
                bad.append((trial, nm, "MH"))
        if not np.array_equal(ws["n_gt"], st["gt"]) or not np.array_equal(ws["n_pred"], st["pred"]):
            bad.append((trial, "n_gt/n_pred", "MH"))
    check(f"vs torch max-pool dilate + MH.window_stats EDT (6 random scenes, {n} count cells, exact)", not bad, str(bad[:5]))


def test_prf_conventions():
    P, R, F, I = T.prf_from_counts(np.array([0, 10, 0, 10]), np.array([0, 0, 10, 10]),
                                   np.array([0, 0, 0, 5]), np.array([0, 0, 0, 5]))
    check("prf: neither side -> NaN", bool(np.isnan(F[0]) and np.isnan(I[0])))
    check("prf: pred only -> F=0, IoU=0", bool(F[1] == 0 and I[1] == 0))
    check("prf: gt only -> F=0, IoU=0", bool(F[2] == 0 and I[2] == 0))
    check("prf: P=R=0.5 -> F=0.5, IoU=1/3", abs(F[3] - 0.5) < 1e-15 and abs(I[3] - 1.0 / 3.0) < 1e-15)


def test_thickness_census():
    g = np.zeros((400, 100), np.uint8)
    g[10:160, 20:23] = 2            # a long line, lateral width 3 cells, 150 rows long: thickness 3
    g[50:52, 40:80] = 3             # a wide stripe 2 rows x 40 cols: thickness 2 (lateral 40, longitudinal 2)
    g[250:255, 60] = 4              # a 1-cell-wide, 5-long mark in band 1: thickness 1
    h = T.thickness_hist(g)
    check("census: lane line thickness 3 x 450 cells in band 0", h["thick"][0, 0, 3] == 450 and h["thick"][0].sum() == 450)
    check("census: crosswalk stripe thickness 2 x 80 cells", h["thick"][1, 0, 2] == 80 and h["thick"][1].sum() == 80)
    check("census: lateral runs of the stripe are 40 wide", h["lat"][1, 0, 40] == 80)
    check("census: arrow mark thickness 1 x 5 cells in band 1", h["thick"][2, 1, 1] == 5 and h["thick"][2].sum() == 5)
    s = T.thickness_summary(h["thick"][0, 0])
    check("census summary: median 3 cells = 0.3 m, share_le_2 = 0", s["median_cells"] == 3 and abs(s["median_m"] - 0.3) < 1e-12 and s["share_le_2"] == 0.0)


if __name__ == "__main__":
    test_strips_literal()
    test_strips_closed_form()
    test_gt_as_prediction_and_all_one_class()
    test_diagonal_literal()
    test_vs_independent_instruments()
    test_prf_conventions()
    test_thickness_census()
    print(f"ZZP2TESTS fails={len(FAILS)} ZZ")
    sys.exit(1 if FAILS else 0)
