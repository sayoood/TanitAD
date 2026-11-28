"""E0j / C11 -- the Gram-matrix form of the gradient share (REFCV9_DESIGN 2.2, "How the weight is chosen").

Pure numpy/torch, NO repo imports, so every formula here is unit-testable on a CPU with analytic answers
(`test_e0j.py`). The GPU job `pgrad_gram.py` produces the block matrix K; `e0j_read.py` turns K into the reading.

DEFINITIONS (all gradients are of the UNWEIGHTED term L_i, i.e. at w = 1; summed over the batches of the draw)
    g_i      gradient of term i over one parameter group
    Gm[i][j] = <g_i, g_j>                         (n_terms x n_terms, symmetric PSD)
    G(w)     = sum_j w_j g_j                       (the update direction at weights w)
    s_i(w)   = w_i <g_i, G(w)> / ||G(w)||^2        = w_i (Gm w)_i / (w^T Gm w)     EXACT for any w, from Gm alone
  sum_i s_i(w) = 1 identically (control S). s_i is a PROJECTED share: it can be negative (the term opposes the update) and
  is not monotone in w_i in general -- the selector therefore scans, it does not assume.

BLOCK MATRIX K (per group): the vectors are v[(b, t)] for batch b and t in TERMS + ("direct",), where "direct" is the gradient
of the weighted total sum_j w2_j L_j taken by ONE backward at the second weight w2 (the linearity control).
    K[(b,s),(c,t)] = <v[(b,s)], v[(c,t)]>.
  The pooled Gram (the gradient SUMMED over all batches = one 64-window batch) is  sum_{b,c} K[(b,i),(c,j)];  a single batch's
  Gram is the diagonal block; leave-one-batch-out and a batch bootstrap are exact from K, for ANY w.
"""
from __future__ import annotations

import numpy as np

TERMS = ("map", "box3d", "agent")
DIRECT = "direct"


# ------------------------------------------------------------------------------------------------------ the share
def share_from_gram(gm, w) -> np.ndarray | None:
    """s_i(w) = w_i (Gm w)_i / (w^T Gm w). None when the update direction has zero norm (no share is defined)."""
    gm = np.asarray(gm, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    den = float(w @ gm @ w)
    if not den > 0.0:
        return None
    return w * (gm @ w) / den


def gram_from_vectors(vecs: dict, terms=TERMS) -> np.ndarray:
    """<g_i, g_j> in float64 from a {term: 1-D tensor/array} dict."""
    import torch
    out = np.zeros((len(terms), len(terms)), dtype=np.float64)
    for i, a in enumerate(terms):
        for j in range(i, len(terms)):
            v = float(torch.dot(torch.as_tensor(vecs[a]).double().reshape(-1), torch.as_tensor(vecs[terms[j]]).double().reshape(-1)))
            out[i, j] = out[j, i] = v
    return out


def check_gram(gm, rel_tol: float = 1e-9) -> dict:
    """Symmetric (relative asymmetry), PSD (smallest eigenvalue >= -rel_tol * largest), Cauchy-Schwarz on every pair."""
    gm = np.asarray(gm, dtype=np.float64)
    asym = float(np.max(np.abs(gm - gm.T)) / max(1e-300, np.max(np.abs(gm))))
    ev = np.linalg.eigvalsh((gm + gm.T) / 2.0)
    d = np.sqrt(np.clip(np.diag(gm), 0.0, None))
    cs = max((abs(gm[i, j]) / (d[i] * d[j]) if d[i] * d[j] > 0 else 0.0) for i in range(len(gm)) for j in range(len(gm)))
    return {"max_rel_asymmetry": asym, "min_eig": float(ev[0]), "max_eig": float(ev[-1]),
            "psd": bool(ev[0] >= -rel_tol * max(1e-300, abs(ev[-1]))), "max_abs_cos": float(cs), "cs_ok": bool(cs <= 1.0 + 1e-9),
            "symmetric": bool(asym <= 1e-12)}


# --------------------------------------------------------------------------------------- pooled Gram from the block matrix
def block_index(n_batches: int, kinds=TERMS + (DIRECT,)) -> dict:
    """(b, kind) -> row of K. Rows are batch-major: b * len(kinds) + position of the kind."""
    return {(b, k): b * len(kinds) + t for b in range(n_batches) for t, k in enumerate(kinds)}


def pooled(K, n_batches: int, batches=None, weights=None, kinds=TERMS + (DIRECT,)):
    """-> (Gm over TERMS, dots <g_i, G2> per term, ||G2||^2) of the SUMMED gradient over `batches` (default all), each batch
    b counted `weights[b]` times (integer multiplicities = a bootstrap draw)."""
    K = np.asarray(K, dtype=np.float64)
    nk = len(kinds)
    bs = list(range(n_batches)) if batches is None else list(batches)
    m = np.ones(n_batches) if weights is None else np.asarray(weights, dtype=np.float64)
    rows = {k: [b * nk + kinds.index(k) for b in bs] for k in kinds}
    cnt = {k: np.array([m[b] for b in bs]) for k in kinds}

    def blk(a, b_):
        return float(cnt[a] @ K[np.ix_(rows[a], rows[b_])] @ cnt[b_])
    gm = np.array([[blk(a, b_) for b_ in TERMS] for a in TERMS])
    dots = np.array([blk(a, DIRECT) for a in TERMS])
    return gm, dots, blk(DIRECT, DIRECT)


def linearity_report(K, n_batches: int, w2) -> dict:
    """The linearity control from K alone. For the pooled sum AND for every batch:
       rel_grad_err = ||sum_j w2_j g_j - G2|| / ||G2||            (the P-GRAD control, <= 1e-4)
       share_diff   = |s_i(w2) from the Gram  -  w2_i <g_i, G2> / ||G2||^2 (direct)|   max over terms   (the E0j control, <= 1e-4)
    """
    w2 = np.asarray(w2, dtype=np.float64)
    out = {"pooled": None, "per_batch": []}
    for tag, batches in [("pooled", None)] + [(b, [b]) for b in range(n_batches)]:
        gm, dots, g2g2 = pooled(K, n_batches, batches)
        num = float(w2 @ gm @ w2 - 2.0 * (w2 @ dots) + g2g2)
        rel = float(np.sqrt(max(num, 0.0) / g2g2)) if g2g2 > 0 else None
        s_gram = share_from_gram(gm, w2)
        s_dir = (w2 * dots / g2g2) if g2g2 > 0 else None
        sd = None if (s_gram is None or s_dir is None) else float(np.max(np.abs(s_gram - s_dir)))
        rec = {"rel_grad_err": rel, "share_diff": sd}
        if tag == "pooled":
            out["pooled"] = rec
        else:
            out["per_batch"].append(rec)
    vals = [r["rel_grad_err"] for r in out["per_batch"] + [out["pooled"]] if r["rel_grad_err"] is not None]
    sds = [r["share_diff"] for r in out["per_batch"] + [out["pooled"]] if r["share_diff"] is not None]
    out["max_rel_grad_err"] = max(vals) if vals else None
    out["max_share_diff"] = max(sds) if sds else None
    return out


# -------------------------------------------------------------------------------------------------------------- selector
def w_grid(lo: float = 1e-3, hi: float = 1e4, n: int = 4001) -> np.ndarray:
    return np.exp(np.linspace(np.log(lo), np.log(hi), n))


def _bisect(pred, a: float, b: float, iters: int = 80) -> float:
    """pred(a) False, pred(b) True -> the boundary (returns the True side)."""
    for _ in range(iters):
        m = float(np.sqrt(a * b))
        a, b = (a, m) if pred(m) else (m, b)
    return b


def _shares_grid(gm, ws, w_box, w_agent):
    """Vectorised s(w) over a grid of w_map values: [n, 3] (NaN where the update direction has zero norm)."""
    gm = np.asarray(gm, dtype=np.float64)
    W = np.stack([ws, np.full_like(ws, w_box), np.full_like(ws, w_agent)], axis=1)
    gw = W @ gm
    den = np.einsum("ni,ni->n", gw, W)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = W * gw / den[:, None]
    s[~(den > 0.0)] = np.nan
    return s


def select_w_map(gms: dict, target: float, *, w_box: float = 1.0, w_agent: float = 1.0, group: str = "layer2",
                 floors: dict | None = None, grid=None) -> dict:
    """The SMALLEST w_map whose map share in `group` is >= target, with the box3d floors reported at that w_map.

    gms: {"layer2": 3x3, "trunk": 3x3, "lift": 3x3} (rows/cols = TERMS). floors: {group: min box3d share}.
    Returns {w_star, found, at_grid_floor, shares_at_w_star{group: {term: s}}, floors_ok{...}, floors_all_ok, w_floor_limit,
             map_share_monotone_on_grid}. `w_floor_limit` = the LARGEST w_map on the grid at which every floor holds (box3d
    shares fall as w_map rises); floors_all_ok False at w_star means the target and the floors cannot both hold."""
    floors = {"layer2": 0.20, "trunk": 0.20, "lift": 0.25} if floors is None else floors
    ws = w_grid() if grid is None else np.asarray(grid, dtype=np.float64)
    idx = {t: i for i, t in enumerate(TERMS)}

    def shares(g, w):
        s = share_from_gram(gms[g], [w, w_box, w_agent])
        return None if s is None else {t: float(s[idx[t]]) for t in TERMS}

    def meets(w):
        s = shares(group, w)
        return s is not None and s["map"] >= target
    sg = _shares_grid(gms[group], ws, w_box, w_agent)[:, idx["map"]]
    hit = np.nan_to_num(sg, nan=-np.inf) >= target
    res = {"target": float(target), "group": group, "w_box3d": w_box, "w_agent": w_agent, "floors": floors}
    fin = sg[np.isfinite(sg)]
    res["map_share_monotone_on_grid"] = bool(np.all(np.diff(fin) >= -1e-12)) if len(fin) > 1 else True
    ok = np.ones(len(ws), dtype=bool)
    for g, fl in floors.items():
        ok &= np.nan_to_num(_shares_grid(gms[g], ws, w_box, w_agent)[:, idx["box3d"]], nan=-np.inf) >= fl
    res["w_floor_limit"] = float(ws[np.nonzero(ok)[0].max()]) if ok.any() else None
    if not hit.any():
        return {**res, "found": False, "w_star": None, "reason": f"map share in {group} never reaches {target} on [{ws[0]:.3g}, {ws[-1]:.3g}]"}
    k = int(np.nonzero(hit)[0].min())
    w_star = float(ws[0]) if k == 0 else _bisect(meets, float(ws[k - 1]), float(ws[k]))
    res.update({"found": True, "w_star": w_star, "at_grid_floor": bool(k == 0),
                "shares_at_w_star": {g: shares(g, w_star) for g in gms}})
    res["floors_ok"] = {g: bool(shares(g, w_star)["box3d"] >= fl) for g, fl in floors.items()}
    res["floors_all_ok"] = bool(all(res["floors_ok"].values()))
    return res


def sum_gram_check(gms: dict, ws=(0.25, 1.0, 4.0, 16.0)) -> float:
    """Control S: max |sum_i s_i(w) - 1| over groups and a few weights (identically 0 up to round-off)."""
    m = 0.0
    for gm in gms.values():
        for w in ws:
            s = share_from_gram(gm, [w, 1.0, 1.0])
            if s is not None:
                m = max(m, abs(float(s.sum()) - 1.0))
    return m


# ------------------------------------------------------------------------------------------- the block matrix (GPU job side)
def block_matrix(vlist, device: str = "cpu", chunk: int = 1 << 21) -> np.ndarray:
    """vlist: list over batches of [n_kinds, n] tensors (batch-major rows: b * n_kinds + kind). -> K = V V^T in float64,
    accumulated over column chunks (the trunk group is ~4.3e7 columns; float64 on the whole matrix would not fit)."""
    import torch
    M = torch.cat([torch.as_tensor(v) for v in vlist], dim=0)
    K = torch.zeros(M.shape[0], M.shape[0], dtype=torch.float64, device=device)
    for lo in range(0, M.shape[1], chunk):
        c = M[:, lo:lo + chunk].to(device).double()
        K += c @ c.T
    return K.cpu().numpy()
