"""PRIVGEOM-CLEAN admissibility check (A-ADMIT): "could the DAC verdict be RECOMPUTED from this block?", answered by a SEARCH with PLANTED controls and a RED arm, never by inspecting column definitions.

WHY (R54, `RETRACTION_LOG.md`): removing a SIGN is not removing the INFORMATION.  PRIVGEOM-SAFE's `DAC-nosign` block kept `|c_min|` beside `|c_t|`; `c_min <= c_t` at every key step, so
`|c_t| < |c_min|` PROVES `c_min < 0` (DAC = 0).  The pair re-derived the verdict through the ORDER of its unsigned columns.  Admissibility is a property of what a model can COMPUTE from the columns.

THE RECOMPUTER is a search over the families a verdict-proof can take (all one-sided: a rule that PROVES DAC-UNSAFE; "every key clearance is large, so the proposal is DAC-safe" is what proximity MEANS and is
reported, not penalised):
  U    unary        C_a < theta | C_a > theta                         (VALUE leaks)                  } the SH5 A-TEST families, 792 rules (re-implemented here; SH5's
  P    pair         C_a + m < C_b, every ordered pair, m in MARGINS    (ORDER leaks; the R54 case)    }  `sh5_admit.py` sha256 is recorded in the prereg as the ancestor)
  V    V-dip        C_m + m < min(C_a, C_b), a < m < b                 (a crossing's signature)
  A    aggregate    max_k C_k < theta | min_k C_k > theta
  T    tree         a depth-limited histogram tree on [values, all pairwise differences, equally spaced second differences, min/max/mean]   (nonlinear COMBINATIONS)
  LIN  linear       an L2 logistic regression on the same features; flagged iff the top of its ranking is a proof                              (LINEAR COMBINATIONS)
A block is NOT ADMISSIBLE iff some rule / node / prefix PROVES DAC-UNSAFE: precision >= PREC_BAR with recall >= RECALL_MIN of the UNSAFE class, over ALL proposals handed in.

CONTROLS (a check is trusted only if every one reads right; run on planted data BEFORE the real block is judged):
  POS (MUST be flagged): POS-REGISTERED (the 10-col DAC-nosign block, R54's case) | POS-EQUAL (the label itself as a column) | POS-VALUE (label*5 + noise) | POS-ORDER (a column equal to C_ref -+ 0.05 by the label:
       no unary rule separates it, only its ORDER against C_ref does) | POS-LIN (a column equal to a fixed linear combination of three columns +- 0.05 by the label: only a COMBINATION proves it)
  NEG (MUST NOT be flagged): NEG-PERM (every column permuted within each set) | NEG-NOISE (candidate + a pure-noise column) | NEG-PROX (candidate + a noisy PROXIMITY column C_ref + N(0, 1): informative about
       the label, not a function of the verdict)
RED ARM: `AdmitConfig(disabled=True)` makes the search return "nothing proven".  Under it the POS plants GET THROUGH, so `controls_ok` is False -- the planted leak is what turns the check RED.  A mutation table
(`mutations()`) disables / blunts each part of the check and requires the controls to break.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

PREC_BAR, RECALL_MIN = 0.999, 0.005
THETAS = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0)
MARGINS = (0.0, 0.25, 0.5, 1.0, 2.0)
FAMILIES = ("U", "P", "V", "A", "T", "LIN")
N_BINS, TREE_DEPTH, SUBSAMPLE = 32, 5, 300_000
REF_COL = 6                                  # key step t30: the column the POS-ORDER / NEG-PROX plants are built against
LIN_COMBO = ((1, 0.7), (4, 0.4), (7, -0.9))  # POS-LIN: 0.7 C1 + 0.4 C4 - 0.9 C7


@dataclass
class AdmitConfig:
    prec_bar: float = PREC_BAR
    recall_min: float = RECALL_MIN
    families: tuple = FAMILIES
    disabled: bool = False                    # the RED arm
    invert: bool = False                      # a RED mutation: flag when nothing is proven
    seed: int = 20261019


def _eval(mask: np.ndarray, unsafe: np.ndarray, n_u: int) -> tuple:
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0.0
    tp = int((mask & unsafe).sum())
    return tp / n, tp / max(n_u, 1)           # precision_unsafe, recall_unsafe


def _rules(C: np.ndarray, fam: tuple):
    M, K = C.shape
    if "U" in fam:
        for a in range(K):
            for th in THETAS:
                yield "U", f"C{a} < {th}", C[:, a] < th
                yield "U", f"C{a} > {th}", C[:, a] > th
    if "P" in fam:
        for a, b in itertools.permutations(range(K), 2):
            for mg in MARGINS:
                yield "P", f"C{a} + {mg} < C{b}", C[:, a] + mg < C[:, b]
    if "V" in fam:
        for a, m, b in itertools.combinations(range(K), 3):
            for mg in MARGINS[:3]:
                yield "V", f"V(C{m} + {mg} < min(C{a}, C{b}))", C[:, m] + mg < np.minimum(C[:, a], C[:, b])
    if "A" in fam:
        mx, mn = C.max(1), C.min(1)
        for th in THETAS:
            yield "A", f"max C < {th}", mx < th
            yield "A", f"min C > {th}", mn > th


def features(C: np.ndarray) -> tuple:
    """[M, F] float32 features of the T / LIN families and their names."""
    M, K = C.shape
    cols, names = [C[:, a] for a in range(K)], [f"C{a}" for a in range(K)]
    for a, b in itertools.combinations(range(K), 2):
        cols.append(C[:, b] - C[:, a])
        names.append(f"(C{b}-C{a})")
    K9 = min(K, 9)
    for a, m, b in itertools.combinations(range(K9), 3):
        if m - a == b - m:
            cols.append(C[:, a] + C[:, b] - 2 * C[:, m])
            names.append(f"(C{a}+C{b}-2C{m})")
    cols += [C.min(1), C.max(1), C.mean(1)]
    names += ["min", "max", "mean"]
    return np.stack(cols, 1).astype(np.float32), names


def _tree_hits(X: np.ndarray, names: list, unsafe: np.ndarray, n_u_total: int, cfg: AdmitConfig) -> tuple:
    """A greedy depth-limited histogram tree; EVERY node (a conjunction of <= depth splits) is evaluated as a rule.  Returns (hits, best (precision, recall, rule) at recall >= recall_min)."""
    M, F = X.shape
    edges, Xb = [], np.empty((M, F), np.uint8)
    qs = np.linspace(0, 1, N_BINS + 1)[1:-1]
    for f in range(F):
        e = np.unique(np.quantile(X[:, f], qs))
        edges.append(e)
        Xb[:, f] = np.searchsorted(e, X[:, f], side="right").astype(np.uint8)
    y = unsafe.astype(np.int64)
    min_leaf = max(50, int(cfg.recall_min * n_u_total))
    hits, best = [], (0.0, 0.0, "")
    stack = [(np.arange(M), 0, ())]
    while stack:
        idx, d, path = stack.pop()
        n, nu = len(idx), int(y[idx].sum())
        prec, rec = (nu / n if n else 0.0), nu / max(n_u_total, 1)
        if path and rec >= cfg.recall_min:
            if prec > best[0]:
                best = (prec, rec, " & ".join(path))
            if prec >= cfg.prec_bar:
                hits.append({"rule": " & ".join(path), "family": "T", "precision": prec, "recall": rec})
        if d >= TREE_DEPTH or n < 2 * min_leaf or nu == 0 or nu == n:
            continue
        bestg, bf, bt = 0.0, -1, -1
        p0 = nu / n
        g0 = 2 * p0 * (1 - p0)
        for f in range(F):
            h = np.bincount(Xb[idx, f].astype(np.int64) * 2 + y[idx], minlength=2 * N_BINS).reshape(N_BINS, 2)
            cs = np.cumsum(h, 0)[:-1]
            nl = cs.sum(1)
            nr = n - nl
            ok = (nl >= min_leaf) & (nr >= min_leaf)
            if not ok.any():
                continue
            pl = np.where(nl > 0, cs[:, 1] / np.maximum(nl, 1), 0.0)
            pr = np.where(nr > 0, (nu - cs[:, 1]) / np.maximum(nr, 1), 0.0)
            g = g0 - (nl / n) * 2 * pl * (1 - pl) - (nr / n) * 2 * pr * (1 - pr)
            g = np.where(ok, g, -1.0)
            t = int(np.argmax(g))
            if g[t] > bestg:
                bestg, bf, bt = float(g[t]), f, t
        if bf < 0:
            continue
        thr = edges[bf][bt] if bt < len(edges[bf]) else np.inf
        left = Xb[idx, bf] <= bt
        stack.append((idx[left], d + 1, path + (f"{names[bf]} <= {thr:.4g}",)))
        stack.append((idx[~left], d + 1, path + (f"{names[bf]} > {thr:.4g}",)))
    return hits, best


def _lin_hits(X: np.ndarray, unsafe: np.ndarray, n_u_total: int, cfg: AdmitConfig) -> tuple:
    """An L2 logistic regression (LBFGS, fixed init) on the standardised features; flagged iff a prefix of its UNSAFE-score ranking is a proof."""
    import torch
    mu, sd = X.mean(0), X.std(0)
    Z = torch.as_tensor(((X - mu) / np.where(sd > 1e-8, sd, 1.0)).astype(np.float32))
    y = torch.as_tensor(unsafe.astype(np.float32))
    w = torch.zeros(Z.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, y) + 1e-5 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        s = (Z @ w + b).numpy()
    order = np.argsort(-s, kind="stable")
    cum = np.cumsum(unsafe[order])
    k = np.arange(1, len(order) + 1)
    prec = cum / k
    rec = cum / max(n_u_total, 1)
    ok = (prec >= cfg.prec_bar) & (rec >= cfg.recall_min)
    best = (0.0, 0.0, "")
    sel = rec >= cfg.recall_min
    if sel.any():
        j = int(np.argmax(np.where(sel, prec, -1.0)))
        best = (float(prec[j]), float(rec[j]), "LIN top-prefix")
    hits = []
    if ok.any():
        j = int(np.nonzero(ok)[0].max())
        hits.append({"rule": "LIN top-prefix", "family": "LIN", "precision": float(prec[j]), "recall": float(rec[j])})
    return hits, best


def search(C: np.ndarray, label_dac: np.ndarray, cfg: AdmitConfig | None = None) -> dict:
    """C [..., K] block (proposals on the leading axes, UNSIGNED columns), label_dac [...] (0 = unsafe, 1 = safe; 0.5 excluded)."""
    cfg = cfg or AdmitConfig()
    K = C.shape[-1]
    C2 = np.asarray(C, np.float64).reshape(-1, K)
    lab = np.asarray(label_dac).reshape(-1)
    ok = (lab == 0.0) | (lab == 1.0)
    C2, unsafe = C2[ok], (lab[ok] == 0.0)
    n_u = int(unsafe.sum())
    out = {"flagged": False, "n_proposals": int(len(unsafe)), "n_unsafe": n_u, "prec_bar": cfg.prec_bar, "recall_min": cfg.recall_min, "families": list(cfg.families), "disabled": cfg.disabled,
           "hits_by_family": {f: 0 for f in FAMILIES}, "best_unsafe_proof_by_family": {}, "top": [], "n_rules": 0, "safe_class_proofs_descriptive": 0}
    if cfg.disabled or n_u == 0:
        out["flagged"] = bool(cfg.invert and not cfg.disabled and n_u == 0)
        return out
    best = {f: (0.0, 0.0, "") for f in FAMILIES}
    hits = []
    n_safe = int(len(unsafe) - n_u)
    for fam, name, mask in _rules(C2, tuple(cfg.families)):
        out["n_rules"] += 1
        pu, ru = _eval(mask, unsafe, n_u)
        if ru >= cfg.recall_min and pu > best[fam][0]:
            best[fam] = (pu, ru, name)
        if pu >= cfg.prec_bar and ru >= cfg.recall_min:
            hits.append({"rule": name, "family": fam, "precision": pu, "recall": ru})
        n = int(mask.sum())
        if n and (n - int((mask & unsafe).sum())) / n >= cfg.prec_bar and (n - int((mask & unsafe).sum())) / max(n_safe, 1) >= cfg.recall_min:
            out["safe_class_proofs_descriptive"] += 1
    if "T" in cfg.families or "LIN" in cfg.families:
        rng = np.random.default_rng(cfg.seed)
        if len(unsafe) > SUBSAMPLE:
            sel = np.sort(rng.choice(len(unsafe), SUBSAMPLE, replace=False))
        else:
            sel = np.arange(len(unsafe))
        X, names = features(C2[sel])
        us = unsafe[sel]
        n_us = int(us.sum())
        if n_us > 0:
            if "T" in cfg.families:
                h, b = _tree_hits(X, names, us, n_us, cfg)
                hits += h
                best["T"] = b
            if "LIN" in cfg.families:
                h, b = _lin_hits(X, us, n_us, cfg)
                hits += h
                best["LIN"] = b
    for h in hits:
        out["hits_by_family"][h["family"]] += 1
    hits.sort(key=lambda h: -h["recall"])
    out["top"] = hits[:5]
    out["best_unsafe_proof_by_family"] = {f: {"precision": v[0], "recall": v[1], "rule": v[2]} for f, v in best.items() if f in cfg.families}
    out["flagged"] = bool(hits)
    if cfg.invert:
        out["flagged"] = not out["flagged"]
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# planted controls
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def permute_columns_within_set(C: np.ndarray, seed: int) -> np.ndarray:
    """every column independently permuted across the 64 proposals of each set (marginals kept, every proposal-wise relation destroyed)"""
    rng = np.random.default_rng(seed)
    N, P, K = C.shape
    out = np.empty_like(C)
    for k in range(K):
        perm = np.argsort(rng.random((N, P)), axis=1)
        out[:, :, k] = np.take_along_axis(C[:, :, k], perm, axis=1)
    return out


def build_controls(block: np.ndarray, label_dac: np.ndarray, leaky10: np.ndarray | None, seed: int = 20261019) -> dict:
    """{name: (array [N,64,K'], must_be_flagged)}.  `block` is the candidate [N,64,9]; `leaky10` the registered DAC-nosign block [N,64,10] (|c_min| + 9 |c_t|) or None."""
    rng = np.random.default_rng(seed)
    lab = np.asarray(label_dac, np.float64)
    unsafe = (lab == 0.0)
    sgn = np.where(unsafe, -1.0, 1.0)
    B = np.asarray(block, np.float64)
    ctl = {}
    if leaky10 is not None:
        ctl["POS-REGISTERED"] = (np.asarray(leaky10, np.float64), True)
    ctl["POS-EQUAL"] = (np.concatenate([B, lab[..., None]], -1), True)
    ctl["POS-VALUE"] = (np.concatenate([B, (np.where(unsafe, 0.0, 5.0) + rng.normal(0.0, 0.5, lab.shape))[..., None]], -1), True)
    ctl["POS-ORDER"] = (np.concatenate([B, (B[..., REF_COL] + 0.05 * sgn)[..., None]], -1), True)
    combo = sum(w * B[..., c] for c, w in LIN_COMBO)
    ctl["POS-LIN"] = (np.concatenate([B, (combo + 0.05 * sgn)[..., None]], -1), True)
    ctl["NEG-PERM"] = (permute_columns_within_set(B, seed), False)
    ctl["NEG-NOISE"] = (np.concatenate([B, rng.normal(0.0, 1.0, lab.shape)[..., None]], -1), False)
    ctl["NEG-PROX"] = (np.concatenate([B, (B[..., REF_COL] + rng.normal(0.0, 1.0, lab.shape))[..., None]], -1), False)
    return ctl


def a_test(block: np.ndarray, label_dac: np.ndarray, leaky10: np.ndarray | None = None, cfg: AdmitConfig | None = None, seed: int = 20261019) -> dict:
    """The full admissibility test of the candidate `block` [N,64,9] against the labels [N,64]: the planted controls, then the candidate.
    -> {"admissible": bool, "controls_ok": bool, "controls": {name: {"flagged", "expected", "read_right", ...}}, "candidate": {...}}"""
    cfg = cfg or AdmitConfig()
    res = {"controls": {}}
    for name, (arr, must) in build_controls(block, label_dac, leaky10, seed).items():
        r = search(arr, label_dac, cfg)
        res["controls"][name] = {"flagged": r["flagged"], "expected_flagged": must, "read_right": bool(r["flagged"] == must), "hits_by_family": r["hits_by_family"],
                                 "best_unsafe_proof_by_family": r["best_unsafe_proof_by_family"], "top": r["top"][:2]}
    res["candidate"] = search(block, label_dac, cfg)
    res["controls_ok"] = bool(all(v["read_right"] for v in res["controls"].values()))
    res["admissible"] = bool(res["controls_ok"] and not res["candidate"]["flagged"])
    return res


def mutations() -> dict:
    """RED mutations of the check itself: each must make `controls_ok` False (the planted controls then read wrong and the gate would FAIL)."""
    return {"disabled (the RED arm)": AdmitConfig(disabled=True),
            "precision bar above 1 (nothing can be proven)": AdmitConfig(prec_bar=1.01),
            "recall floor above 1 (nothing can be proven)": AdmitConfig(recall_min=2.0),
            "sense inverted (flag when nothing is proven)": AdmitConfig(invert=True),
            "no families": AdmitConfig(families=())}


def family_ablation(block: np.ndarray, label_dac: np.ndarray, leaky10: np.ndarray | None, seed: int = 20261019) -> dict:
    """For each family alone: which POS plants it flags and whether a NEG plant is (wrongly) flagged.  DESCRIPTIVE: shows which plant needs which family (defence in depth means several families may catch one)."""
    out = {}
    ctl = build_controls(block, label_dac, leaky10, seed)
    for fam in FAMILIES:
        cfg = AdmitConfig(families=(fam,))
        out[fam] = {name: bool(search(arr, label_dac, cfg)["flagged"]) for name, (arr, _) in ctl.items()}
    return out
