"""refcv7 map/box diagnostics -- the NEW metrics (SPEC.md sec. 2-3), pure functions, no trainer import.

Every metric the trainer already has (per-class inter/union under a rule, `decide`, `per_class_signal`,
`detection_metrics.*`) is IMPORTED by the driver, never re-implemented here. What lives here is only what the
trainer does not compute:

* per-EPISODE pooled counts for an arbitrary list of decisions (partition rules and one-vs-rest thresholds),
  so the pooled IoU gets an episode-cluster bootstrap;
* boundary-tolerant precision / recall / F / IoU_k (Chebyshev k cells), IoU_k = F_k / (2 - F_k);
* score histograms (logit of the class posterior) on GT-positive / GT-negative cells -> AUROC and the exact
  threshold sweep at bin edges;
* the GT line-width and the edge-to-drivable-boundary registration census (M-f);
* box presence recalibration (shift / temperature / Platt fits, the analytic focal inversion, ECE).

Tests: ``test_diag_metrics.py`` (controls C3, C4, C6 of SPEC.md sec. 4).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

N_CLS = 8
CLASS_KEYS = ("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched", "sidewalk")
THIN = (2, 3, 4, 5, 6)
BAND_ROWS = 200                       # 20 m of 0.1 m rows (semantic_map_gt_fine.BAND_ROWS_FINE)
TOL_K = (1, 2, 5, 10)                 # k = 0 is the exact intersection; k = 5, 10 are POST-HOC (RESULT.md)
HIST_LO, HIST_HI, HIST_NB = -20.0, 20.0, 4000
REG_RADII = (0, 1, 2, 5, 10, 20)      # registration buckets: 0,1,2,3-5,6-10,11-20,>20
REG_BUCKETS = ("0", "1", "2", "3-5", "6-10", "11-20", ">20")
RUN_CAP = 400                         # line-width histogram cap (cells)


# ----------------------------------------------------------------------------------------------- #
# band helpers                                                                                     #
# ----------------------------------------------------------------------------------------------- #
def band_matrix(n_rows: int, device) -> torch.Tensor:
    """[H, nb] one-hot row -> 20 m band (the last band partial), = map_head_hires.per_class_signal's M."""
    nb = (int(n_rows) + BAND_ROWS - 1) // BAND_ROWS
    rb = torch.arange(int(n_rows), device=device) // BAND_ROWS
    return F.one_hot(rb, nb).to(torch.float32)


def per_window_bands(mask: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] (bool/float) -> [B, C, nb] float64 counts per 20 m band."""
    return (mask.to(torch.float32).sum(dim=3) @ M).double()


# ----------------------------------------------------------------------------------------------- #
# decisions                                                                                        #
# ----------------------------------------------------------------------------------------------- #
def log_w(class_weight: torch.Tensor, device) -> torch.Tensor:
    w = class_weight.detach().to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(w).all()) or bool((w <= 0).any()):
        raise ValueError("class weights must be finite and > 0 here (all 8 are, in the run's file)")
    return torch.log(w)


def onehot_masks(codes_hw: torch.Tensor, sup: torch.Tensor) -> torch.Tensor:
    """[B, H, W] long codes in 0..7 -> [B, 8, H, W] bool, restricted to ``sup``."""
    return F.one_hot(codes_hw.clamp(0, N_CLS - 1), N_CLS).permute(0, 3, 1, 2).bool() & sup[:, None]


def decision_masks(z: torch.Tensor, sup: torch.Tensor, lw: torch.Tensor, decisions: dict) -> dict:
    """``decisions``: name -> spec. Specs:
    ``("argmax", offset[8])`` -- partition ``argmax_c (z_c + offset_c)``;
    ``("thr_phat", tau_logit[8])`` -- one-vs-rest ``logit(p_hat_c) >= tau_c``, p_hat = softmax(z - ln w);
    ``("thr_q", tau_logit[8])`` -- one-vs-rest ``logit(softmax(z)_c) >= tau_c``.
    Returns name -> [B, 8, H, W] bool restricted to ``sup``."""
    out = {}
    cache = {}
    for name, (kind, vec) in decisions.items():
        v = torch.as_tensor(vec, dtype=torch.float32, device=z.device).view(1, -1, 1, 1)
        if kind == "argmax":
            out[name] = onehot_masks((z + v).argmax(dim=1), sup)
        elif kind in ("thr_phat", "thr_q"):
            if kind not in cache:
                cache[kind] = class_logits(z - lw.view(1, -1, 1, 1) if kind == "thr_phat" else z)
            out[name] = (cache[kind] >= v) & sup[:, None]
        else:
            raise ValueError(f"decision kind {kind!r}")
    return out


def class_logits(z: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> logit(softmax(z)_c) = z_c - logsumexp_{j != c} z_j (exact, stable)."""
    outs = []
    for c in range(z.shape[1]):
        zo = z.clone()
        zo[:, c] = float("-inf")
        outs.append(z[:, c] - torch.logsumexp(zo, dim=1))
    return torch.stack(outs, dim=1)


# ----------------------------------------------------------------------------------------------- #
# tolerant counts                                                                                  #
# ----------------------------------------------------------------------------------------------- #
def dilate(mask: torch.Tensor, k: int) -> torch.Tensor:
    """[B, C, H, W] bool -> Chebyshev dilation by k cells (square (2k+1)^2), separable max-pool."""
    if k == 0:
        return mask
    x = mask.to(torch.float16) if mask.is_cuda else mask.to(torch.float32)
    B, C, H, W = x.shape
    x = x.reshape(B * C, 1, H, W)
    x = F.max_pool2d(x, kernel_size=(2 * k + 1, 1), stride=1, padding=(k, 0))
    x = F.max_pool2d(x, kernel_size=(1, 2 * k + 1), stride=1, padding=(0, k))
    return x.reshape(B, C, H, W) > 0.5


def tolerant_counts(pred: torch.Tensor, gt: torch.Tensor, M: torch.Tensor, ks=TOL_K) -> dict:
    """Per window x class x band: pred, gt, inter (k=0) and TPp_k / TPg_k for k in ``ks``.
    ``pred``/``gt`` [B, 8, H, W] bool (both already restricted to supervised cells)."""
    out = {"pred": per_window_bands(pred, M), "gt": per_window_bands(gt, M),
           "inter": per_window_bands(pred & gt, M)}
    for k in ks:
        out[f"tpp{k}"] = per_window_bands(pred & dilate(gt, k), M)
        out[f"tpg{k}"] = per_window_bands(gt & dilate(pred, k), M)
    return out


def tolerant_summary(c: dict, ks=(0,) + TOL_K) -> dict:
    """Pooled counts (any leading shape summed already) -> P_k, R_k, F_k, IoU_k (None when undefined)."""
    out = {}
    P, G = float(c["pred"]), float(c["gt"])
    ks = [k for k in ks if k == 0 or f"tpp{k}" in c]          # a pass banked with fewer k carries fewer keys
    for k in ks:
        tpp = float(c["inter"] if k == 0 else c[f"tpp{k}"])
        tpg = float(c["inter"] if k == 0 else c[f"tpg{k}"])
        prec = tpp / P if P > 0 else None
        rec = tpg / G if G > 0 else None
        if prec is None or rec is None:
            f = None if (P == 0 and G == 0) else (0.0 if (P == 0 or G == 0) else None)
        else:
            f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out[f"P{k}"], out[f"R{k}"], out[f"F{k}"] = prec, rec, f
        out[f"IoU{k}"] = None if f is None else f / (2.0 - f)
    return out


def iou(inter, pred, gt):
    u = float(pred) + float(gt) - float(inter)
    return float(inter) / u if u > 0 else None


# ----------------------------------------------------------------------------------------------- #
# histograms                                                                                       #
# ----------------------------------------------------------------------------------------------- #
def hist_bins(s: torch.Tensor) -> torch.Tensor:
    w = (HIST_HI - HIST_LO) / HIST_NB
    return ((s.clamp(HIST_LO, HIST_HI - 1e-6) - HIST_LO) / w).floor().long().clamp(0, HIST_NB - 1)


def score_histograms(s: torch.Tensor, gt: torch.Tensor, sup: torch.Tensor, n_bands: int) -> torch.Tensor:
    """``s`` [B, 8, H, W] scores (logit of a class posterior), ``gt`` [B, 8, H, W] bool one-hot, ``sup`` [B, H, W].
    -> [8, nb, 2, NB] int64: (class, band, gt-negative/positive, bin)."""
    B, C, H, W = s.shape
    rb = (torch.arange(H, device=s.device) // BAND_ROWS).view(1, H, 1).expand(B, H, W)
    out = torch.zeros(C, n_bands, 2, HIST_NB, dtype=torch.int64, device=s.device)
    rbs = rb[sup]
    for c in range(C):
        bins = hist_bins(s[:, c][sup])
        y = gt[:, c][sup].long()
        idx = (rbs * 2 + y) * HIST_NB + bins
        out[c] += torch.bincount(idx, minlength=n_bands * 2 * HIST_NB).view(n_bands, 2, HIST_NB)
    return out


def hist_edges() -> np.ndarray:
    """Lower bin edges in logit space (len NB)."""
    return HIST_LO + (HIST_HI - HIST_LO) / HIST_NB * np.arange(HIST_NB)


def auroc_from_hist(neg: np.ndarray, pos: np.ndarray):
    """Binned Mann-Whitney AUROC, ties within a bin counted half. None when a side is empty."""
    neg = np.asarray(neg, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return None
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (P * N))


def sweep_from_hist(neg: np.ndarray, pos: np.ndarray):
    """For every lower bin edge i: mask = {bin >= i}; -> (pred[i], inter[i]) arrays, G."""
    neg = np.asarray(neg, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    inter = np.cumsum(pos[::-1])[::-1]
    pred = inter + np.cumsum(neg[::-1])[::-1]
    return pred, inter, float(pos.sum())


def best_threshold(neg, pos):
    """-> (best lower-edge index, best IoU) over the sweep (first index on ties)."""
    pred, inter, G = sweep_from_hist(neg, pos)
    u = pred + G - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        j = np.where(u > 0, inter / u, 0.0)
    i = int(np.argmax(j))
    return i, float(j[i])


def iou_at_index(neg, pos, i: int):
    pred, inter, G = sweep_from_hist(neg, pos)
    return iou(inter[i], pred[i], G), float(pred[i]), float(inter[i]), G


# ----------------------------------------------------------------------------------------------- #
# M-f: line widths and registration                                                                #
# ----------------------------------------------------------------------------------------------- #
def run_length_hist(codes: np.ndarray, sup: np.ndarray, c: int, n_bands: int) -> np.ndarray:
    """Lateral (along the last axis) maximal runs of class c, per band. -> [nb, RUN_CAP+1] counts
    (bin RUN_CAP = longer)."""
    m = ((codes == c) & sup).astype(np.int8)                    # [B, H, W]
    B, H, W = m.shape
    pad = np.zeros((B, H, W + 2), np.int8)
    pad[:, :, 1:-1] = m
    d = np.diff(pad, axis=2)
    sb, sh, sw = np.nonzero(d == 1)
    eb, eh, ew = np.nonzero(d == -1)
    out = np.zeros((n_bands, RUN_CAP + 1), np.int64)
    if sb.size == 0:
        return out
    # np.nonzero is row-major, so starts and ends pair up in order
    L = np.minimum(ew - sw, RUN_CAP)
    band = np.minimum(sh // BAND_ROWS, n_bands - 1)
    np.add.at(out, (band, L), 1)
    return out


def boundary(mask: torch.Tensor, sup: torch.Tensor) -> torch.Tensor:
    """[B, H, W] bool drivable -> cells of ``mask`` with a 4-neighbour supervised non-mask cell."""
    other = (sup & ~mask).float()[:, None]
    k = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=other.dtype, device=other.device).view(1, 1, 3, 3)
    nb = F.conv2d(other, k, padding=1)[:, 0] > 0
    return mask & nb


def registration_hist(edge: torch.Tensor, bnd: torch.Tensor, n_bands: int) -> torch.Tensor:
    """Chebyshev distance bucket of each ``edge`` cell to the nearest ``bnd`` cell. -> [nb, 7] int64."""
    B, H, W = edge.shape
    rb = (torch.arange(H, device=edge.device) // BAND_ROWS).view(1, H, 1).expand(B, H, W)
    assigned = torch.zeros_like(edge)
    out = torch.zeros(n_bands, len(REG_BUCKETS), dtype=torch.int64, device=edge.device)
    for j, r in enumerate(REG_RADII):
        hit = dilate(bnd[:, None], r)[:, 0] & edge & ~assigned
        out[:, j] = torch.bincount(rb[hit], minlength=n_bands)[:n_bands]
        assigned |= hit
    rest = edge & ~assigned
    out[:, -1] = torch.bincount(rb[rest], minlength=n_bands)[:n_bands]
    return out


# ----------------------------------------------------------------------------------------------- #
# bootstrap                                                                                        #
# ----------------------------------------------------------------------------------------------- #
def cluster_bootstrap_ratio(num: np.ndarray, den: np.ndarray, B: int = 1000, seed: int = 0,
                            num2=None, den2=None):
    """Episode-cluster bootstrap of a pooled ratio sum(num)/sum(den) over episodes (axis 0).
    With num2/den2 (same episodes) -> the PAIRED difference r - r2. Returns (lo, hi) 95 % percentile."""
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    n = num.shape[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        d = den[idx].sum()
        r = num[idx].sum() / d if d > 0 else np.nan
        if num2 is not None:
            d2 = np.asarray(den2, np.float64)[idx].sum()
            r = r - (np.asarray(num2, np.float64)[idx].sum() / d2 if d2 > 0 else np.nan)
        vals.append(r)
    v = np.asarray(vals)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def bootstrap_iou(inter, pred, gt, B=1000, seed=0, other=None):
    """IoU = inter / (pred + gt - inter) per episode arrays; ``other`` = (inter2, pred2, gt2) for the PAIRED
    difference IoU - IoU2 (same episodes)."""
    inter, pred, gt = (np.asarray(a, np.float64) for a in (inter, pred, gt))
    union = pred + gt - inter
    if other is None:
        return cluster_bootstrap_ratio(inter, union, B, seed)
    i2, p2, g2 = (np.asarray(a, np.float64) for a in other)
    return cluster_bootstrap_ratio(inter, union, B, seed, num2=i2, den2=p2 + g2 - i2)


def bootstrap_iou_unpaired(a, b, B=1000, seed=0):
    """IoU(a) - IoU(b) for two DIFFERENT episode sets (independent resamples). a, b = (inter, pred, gt)."""
    rng = np.random.default_rng(seed)
    A = [np.asarray(x, np.float64) for x in a]
    Bb = [np.asarray(x, np.float64) for x in b]
    vals = []
    for _ in range(B):
        ia = rng.integers(0, A[0].shape[0], A[0].shape[0])
        ib = rng.integers(0, Bb[0].shape[0], Bb[0].shape[0])
        ua = A[1][ia].sum() + A[2][ia].sum() - A[0][ia].sum()
        ub = Bb[1][ib].sum() + Bb[2][ib].sum() - Bb[0][ib].sum()
        if ua > 0 and ub > 0:
            vals.append(A[0][ia].sum() / ua - Bb[0][ib].sum() / ub)
    if not vals:
        return None
    v = np.asarray(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ----------------------------------------------------------------------------------------------- #
# box presence recalibration                                                                       #
# ----------------------------------------------------------------------------------------------- #
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def logit(p, eps=1e-12):
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def _bce(z, y):
    z = np.asarray(z, np.float64)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def fit_affine(z, y, fit_a: bool, fit_b: bool, iters: int = 100):
    """argmin_{a, b} BCE(sigmoid(a z + b), y) by Newton on the convex objective; returns (a, b).
    fit_a False -> a = 1 (pure shift); fit_b False -> b = 0 (pure temperature, T = 1/a)."""
    z = np.asarray(z, np.float64)
    y = np.asarray(y, np.float64)
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = sigmoid(a * z + b)
        r = p - y
        s = p * (1 - p) + 1e-12
        g, Hm, idx = [], [], []
        if fit_a:
            idx.append("a")
        if fit_b:
            idx.append("b")
        X = np.stack([z if k == "a" else np.ones_like(z) for k in idx], 1)
        g = X.T @ r / len(z)
        Hm = (X * s[:, None]).T @ X / len(z) + 1e-9 * np.eye(len(idx))
        step = np.linalg.solve(Hm, g)
        # damped Newton with backtracking on the BCE
        cur = _bce(a * z + b, y)
        t = 1.0
        while True:
            na, nb = a, b
            for j, k in enumerate(idx):
                if k == "a":
                    na = a - t * step[j]
                else:
                    nb = b - t * step[j]
            if _bce(na * z + nb, y) <= cur + 1e-15 or t < 1e-6:
                break
            t *= 0.5
        if abs(na - a) < 1e-10 and abs(nb - b) < 1e-10:
            a, b = na, nb
            break
        a, b = na, nb
    return float(a), float(b)


def ece(p, y, n_bins: int = 15, min_p: float | None = None):
    """Expected calibration error with equal-width bins; -> (ece, table[bins])."""
    p = np.asarray(p, np.float64)
    y = np.asarray(y, np.float64)
    if min_p is not None:
        keep = p >= float(min_p)
        p, y = p[keep], y[keep]
    if p.size == 0:
        return None, []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    tot, table = 0.0, []
    for i in range(n_bins):
        m = b == i
        n = int(m.sum())
        if n:
            conf, acc = float(p[m].mean()), float(y[m].mean())
            tot += n / p.size * abs(acc - conf)
            table.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": n, "conf": conf, "acc": acc})
        else:
            table.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": 0, "conf": None, "acc": None})
    return float(tot), table


def focal_inversion_table(presence_optimum, n: int = 2001):
    """(p*, pi) pairs: p* = presence_optimum(pi, 'focal') on a pi grid -> monotone map p -> pi for np.interp."""
    pis = np.concatenate([np.logspace(-8, -1, n // 2, endpoint=False), np.linspace(0.1, 1 - 1e-8, n - n // 2)])
    ps = np.array([presence_optimum(float(pi), "focal", tol=1e-10) for pi in pis])
    order = np.argsort(ps)
    return ps[order], pis[order]
