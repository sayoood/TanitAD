"""p2_tol_lib.py -- P2: boundary-tolerant map statistics, per window x class x 20 m band.  Pure numpy/scipy, no torch.

For one window (``pred``/``gt`` = [H, W] uint8 class codes 0..7, gt 255 = not seen; the scored cells are ``gt != 255``,
EXACTLY ``taniteval.map_hires_metrics._scored_mask``) it returns, per class c and band b (200 rows = 20 m):

    pred, gt, inter, union          -- exact (k = 0) counts, identical in definition to ``MH.window_stats``
    tpp{k}                          -- #pred cells of c with a GT cell of c within Chebyshev distance <= k cells
    tpg{k}                          -- #GT cells of c with a pred cell of c within Chebyshev distance <= k cells

Chebyshev = chessboard metric = the separable (2k+1)^2 square dilation of ``diag_metrics.dilate`` (cells outside the grid are
never True).  Computed here by ONE chessboard distance transform per (class, direction) -- an independent algorithm from the
max-pool dilation, which is what makes ``test_p2_tol.py`` a real cross-check.

Pooled: P_k = tpp_k / pred, R_k = tpg_k / gt, F_k = 2 P_k R_k / (P_k + R_k), IoU_k = F_k / (2 - F_k).
At k = 0 tpp = tpg = inter, F_0 = 2 inter / (pred + gt), IoU_0 = inter / union (the programme's exact IoU).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

N_CLS = 8
BAND_ROWS = 200
KS = (1, 2, 5, 10)
NOT_SEEN = 255
CELL_M = 0.1
CENSUS_CLASSES = (2, 3, 4, 5, 6)            # lane, crosswalk, arrow, edge, hatched
CAP = 64                                    # thickness histogram cap (bin CAP = ">= CAP cells")
CLASS_NAMES = ("seen, no map class", "drivable", "lane / road line", "crosswalk", "arrow / text",
               "non-drivable edge", "hatched area", "sidewalk / verge")


def band_edges(n_rows: int):
    return [(a, min(a + BAND_ROWS, int(n_rows))) for a in range(0, int(n_rows), BAND_ROWS)]


def _band_sum(m: np.ndarray, edges) -> np.ndarray:
    return np.array([int(m[a:b].sum()) for a, b in edges], dtype=np.int64)


def stat_names(ks=KS):
    return ["pred", "gt", "inter", "union"] + [f"tpp{k}" for k in ks] + [f"tpg{k}" for k in ks]


def cheb_dist_to(mask: np.ndarray, metric: str = "chessboard"):
    """Distance (cells, int) from every cell to the nearest True cell of ``mask``; None when ``mask`` is empty."""
    if not mask.any():
        return None
    return ndimage.distance_transform_cdt(~mask, metric=metric)


def tol_stats(pred, gt, ks=KS, _metric: str = "chessboard") -> dict:
    """{stat: [8, n_bands] int64} for one window.  ``_metric`` exists ONLY so the mutation test can swap the metric."""
    p = np.asarray(pred)
    g = np.asarray(gt)
    if p.ndim != 2 or p.shape != g.shape:
        raise ValueError(f"pred/gt must be [H, W] on one grid, got {p.shape} / {g.shape}")
    if p.size and (p > N_CLS - 1).any():
        raise ValueError(f"pred carries code {int(p.max())}: not a class 0..{N_CLS - 1}")
    s = g != NOT_SEEN
    edges = band_edges(g.shape[0])
    nb = len(edges)
    out = {n: np.zeros((N_CLS, nb), np.int64) for n in stat_names(ks)}
    for c in range(N_CLS):
        gk = (g == c) & s
        pk = (p == c) & s
        out["pred"][c] = _band_sum(pk, edges)
        out["gt"][c] = _band_sum(gk, edges)
        out["inter"][c] = _band_sum(gk & pk, edges)
        out["union"][c] = _band_sum(gk | pk, edges)
        if pk.any() and gk.any():
            d_to_g = cheb_dist_to(gk, _metric)
            d_to_p = cheb_dist_to(pk, _metric)
            for k in ks:
                out[f"tpp{k}"][c] = _band_sum(pk & (d_to_g <= k), edges)
                out[f"tpg{k}"][c] = _band_sum(gk & (d_to_p <= k), edges)
    return out


# ------------------------------------------------------------------------------------------------------------------ #
# pooled metrics from COUNTS (any leading batch shape): used by the analysis, the controls and the bootstrap            #
# ------------------------------------------------------------------------------------------------------------------ #
def prf_from_counts(pred, gt, tpp, tpg):
    """Vectorised P, R, F, IoU (float64, NaN where undefined) from pooled counts.  Conventions = ``MH._f1``:
    neither side present -> NaN; exactly one side empty -> F = 0 (P or R is NaN / 0); P + R == 0 -> F = 0."""
    pred = np.asarray(pred, np.float64)
    gt = np.asarray(gt, np.float64)
    tpp = np.asarray(tpp, np.float64)
    tpg = np.asarray(tpg, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(pred > 0, tpp / pred, np.nan)
        R = np.where(gt > 0, tpg / gt, np.nan)
        den = P + R
        F = np.where((pred > 0) & (gt > 0), np.where(den > 0, 2.0 * P * R / den, 0.0), np.nan)
        F = np.where((pred > 0) ^ (gt > 0), 0.0, F)
        IoU = F / (2.0 - F)
    return P, R, F, IoU


# ------------------------------------------------------------------------------------------------------------------ #
# GT line-thickness census                                                                                            #
# ------------------------------------------------------------------------------------------------------------------ #
def _run_len_dense(mask: np.ndarray) -> np.ndarray:
    """[H, W] bool -> [H, W] int32: for every True cell, the length of its maximal run along axis 1 (0 elsewhere)."""
    H, W = mask.shape
    m = np.zeros((H, W + 2), np.int8)
    m[:, 1:-1] = mask
    d = np.diff(m, axis=1)
    sr, sc = np.nonzero(d == 1)
    er, ec = np.nonzero(d == -1)
    L = (ec - sc).astype(np.int32)                       # row-major order => pairs start/end in order
    rr, cc = np.nonzero(mask)                            # row-major True cells, runs contiguous in this order
    out = np.zeros((H, W), np.int32)
    out[rr, cc] = np.repeat(L, L)
    return out


def thickness_hist(gt, classes=CENSUS_CLASSES, cap: int = CAP) -> dict:
    """Per class x band histograms of the GT line thickness, in CELLS (0.1 m).
    thickness(cell) = min(maximal lateral run through it, maximal longitudinal run through it)  (orientation-robust);
    also the lateral-run and longitudinal-run histograms alone.  -> {'thick'|'lat'|'lon': [len(classes), nb, cap+1] int64}."""
    g = np.asarray(gt)
    s = g != NOT_SEEN
    H, W = g.shape
    edges = band_edges(H)
    nb = len(edges)
    band_of_row = np.minimum(np.arange(H) // BAND_ROWS, nb - 1)
    out = {k: np.zeros((len(classes), nb, cap + 1), np.int64) for k in ("thick", "lat", "lon")}
    for ci_, c in enumerate(classes):
        m = (g == c) & s
        if not m.any():
            continue
        lat = _run_len_dense(m)
        lon = _run_len_dense(m.T).T
        rr, cc = np.nonzero(m)
        bnd = band_of_row[rr]
        for key, arr in (("lat", lat[rr, cc]), ("lon", lon[rr, cc]),
                         ("thick", np.minimum(lat[rr, cc], lon[rr, cc]))):
            v = np.minimum(arr, cap)
            out[key][ci_] += np.bincount(bnd * (cap + 1) + v, minlength=nb * (cap + 1)).reshape(nb, cap + 1)
    return out


def thickness_summary(hist_row: np.ndarray, cap: int = CAP) -> dict:
    """hist_row [cap+1] counts of CELLS by thickness -> n, median, mean, p10/p90 (cells and m), shares <= 1/2/3/5 cells."""
    h = np.asarray(hist_row, np.float64)
    n = h.sum()
    if n == 0:
        return {"n_cells": 0}
    v = np.arange(h.size)
    cdf = np.cumsum(h) / n
    q = lambda p: int(v[np.searchsorted(cdf, p)])            # noqa: E731
    out = {"n_cells": int(n), "median_cells": q(0.5), "p10_cells": q(0.1), "p90_cells": q(0.9),
           "mean_cells": float((h * v).sum() / n), "capped_share": float(h[-1] / n)}
    out["median_m"] = out["median_cells"] * CELL_M
    for w in (1, 2, 3, 5):
        out[f"share_le_{w}"] = float(h[:w + 1].sum() / n)
    return out


def analytic_iou0_strip(w: int, d: int) -> float:
    """Exact IoU of two parallel strips of width w cells, laterally offset by d cells (same length): (w-d)+/(w+d)."""
    return max(0, w - d) / (w + d) if (w + d) > 0 else math.nan
