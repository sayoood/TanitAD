"""PREREG_V8F_CAPLADDER library (T0, no model training). Dual (kernel) ridge on standardised features, refit-under-permutation reference,
episode-cluster bootstrap, the READABLE / LOSS two-part rules, the calibration and dimension controls. Pure numpy (+ torch for the Gram).
Spec: IMPL_CHOICES_CAPLADDER.md (sha256 in IMPL_CHOICES_CAPLADDER.sha256). Everything here is shared by the real run and by test_capladder_controls.py."""
import hashlib
import sys

import numpy as np

sys.dont_write_bytecode = True

NF = 5                                                   # outer episode folds (card); change here only
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)      # card: lambda = c * tr(K_fit) / n_fit
R_REF, N_CAL = 199, 20
REF_SEEDS = tuple(range(1, R_REF + 1))
CAL_SEEDS = tuple(range(1001, 1001 + N_CAL))
B_BOOT = 2000


# ------------------------------------------------------------------------------------------------ folds, bootstrap, derangement
def _h(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def outer_fold(eids, nf=NF):
    """Episode-level fold: sha256(episode)[:8] % nf (IMAG1B's hash rule, modulus 5 instead of 2; see IMPL_CHOICES section 1)."""
    return np.array([_h(str(e)) % nf for e in eids])


def inner_fold(eids, nf=5):
    """Inner CV fold: sha256(episode + '|inner')[:8] % 5 (IMAG1B's inner salt)."""
    return np.array([_h(str(e) + "|inner") % nf for e in eids])


def boot_weights_local(uniq, n_boot=B_BOOT, seed=0):
    """Same rng calls as wmact_lib.boot_weights / taniteval.ci._draws. uniq = sorted unique episode ids (strings)."""
    rng = np.random.default_rng(seed)
    n = len(uniq)
    W = np.zeros((n_boot, n))
    for b in range(n_boot):
        pick = rng.choice(uniq, size=n, replace=True)
        W[b] = np.bincount(np.searchsorted(uniq, pick), minlength=n)
    return W


def perm_labels(y, ep, rng):
    """imag1b_lib.py:133-141, verbatim: the fit-fold episodes are put in a random order, each takes the previous episode's label sequence cycled to its length."""
    out = np.empty_like(y)
    order = np.unique(ep)[rng.permutation(len(np.unique(ep)))]
    for e, e2 in zip(order, np.roll(order, 1)):
        src = y[ep == e2]
        n = int((ep == e).sum())
        out[ep == e] = src[np.arange(n) % len(src)]
    return out


def build_perm_fit(y, ep, fold, seeds, nf=NF):
    """Per outer fold k the deranged fit-fold labels [n_F_k, len(seeds)]; draw s uses default_rng(s), consumed over folds 0..nf-1 in order."""
    out = [np.empty((int((fold != k).sum()), len(seeds))) for k in range(nf)]
    for j, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        for k in range(nf):
            F = np.flatnonzero(fold != k)
            out[k][:, j] = perm_labels(y[F], ep[F], rng)
    return out


# ------------------------------------------------------------------------------------------------ the Gram (kernel) of standardised features
def gram_folds(chunks, folds_by_set, device="cpu"):
    """chunks: iterable of float arrays [n_all, dc] (rows of every set stacked as the caller defines). folds_by_set: {name: (row_index_array, fold_array)}.
    For each set and outer fold k: standardise (mean, population sd + 1e-8 of the FIT-FOLD rows of the set), K^k += X~ X~^T over the chunk.
    Returns {name: float32 array [NF, n_set, n_set]}. float32 accumulation, TF32 off."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Ks, meta = {}, {}
    for name, (rows, fold) in folds_by_set.items():
        n = len(rows)
        Ks[name] = torch.zeros(NF, n, n, dtype=torch.float32, device=device)
        meta[name] = (torch.as_tensor(np.asarray(rows), device=device), [torch.as_tensor(fold != k, device=device) for k in range(NF)])
    for X in chunks:
        X = torch.as_tensor(np.ascontiguousarray(X) if isinstance(X, np.ndarray) else X).to(device=device, dtype=torch.float32)
        for name, (rows, fm) in meta.items():
            Xs_all = X[rows]
            for k in range(NF):
                Xf = Xs_all[fm[k]]
                mu = Xf.mean(0)
                sd = Xf.std(0, unbiased=False) + 1e-8
                Z = (Xs_all - mu) / sd
                Ks[name][k] += Z @ Z.T
                del Z, Xf
            del Xs_all
        del X
    out = {}
    for name, K in Ks.items():
        out[name] = ((K + K.transpose(1, 2)) / 2).cpu().numpy()
    return out


def gram_from_matrix(X, folds_by_set, device="cpu", chunk=16384):
    X = np.asarray(X)
    return gram_folds((X[:, i:i + chunk] for i in range(0, X.shape[1], chunk)), folds_by_set, device)


# ------------------------------------------------------------------------------------------------ dual ridge with an intercept
class Split:
    """One (fit rows F, score rows S) split: eigendecomposition of the kernel centred on F, once; predictions for any c and any label matrix."""

    def __init__(self, K, F, S, Yfit):
        KFF = K[np.ix_(F, F)].astype(np.float64)
        KSF = K[np.ix_(S, F)].astype(np.float64)
        r = KFF.mean(0)
        m = r.mean()
        KFFc = KFF - r[None, :] - r[:, None] + m
        self.KSFc = KSF - KSF.mean(1, keepdims=True) - r[None, :] + m
        s, U = np.linalg.eigh((KFFc + KFFc.T) / 2)
        self.s = np.clip(s, 0.0, None)
        self.U = U
        self.nF = len(F)
        self.tr = float(self.s.sum())
        self.ym = Yfit.mean(0)
        self.Z = U.T @ (Yfit - self.ym[None, :])
        self.V = self.KSFc @ U

    def predict(self, c):
        lam = c * self.tr / self.nF
        return self.V @ (self.Z / (self.s[:, None] + lam)) + self.ym[None, :]


def oof_fold(K, rows, F, S, inner_F, Yfit, keep=False):
    """K [n_all, n_all] kernel of this outer fold; rows = indices of the set's rows in K (local row i is K index rows[i]); F, S local index arrays of the fit and
    scoring rows; inner_F = inner fold of the fit rows; Yfit [n_F, nd] labels of the fit rows (one column per draw / target).
    Chooses c per column by inner 5-fold episode CV (pooled SSE, first on ties), refits on all of F, returns (pred [n_S, nd], cstar [nd], Split or None)."""
    nd = Yfit.shape[1]
    sse = np.zeros((len(C_GRID), nd))
    for j in range(5):
        tr, te = inner_F != j, inner_F == j
        if not te.any() or not tr.any():
            continue
        sp = Split(K, rows[F[tr]], rows[F[te]], Yfit[tr])
        Ys = Yfit[te]
        for ci, c in enumerate(C_GRID):
            sse[ci] += ((Ys - sp.predict(c)) ** 2).sum(0)
    cstar = sse.argmin(0)
    sp = Split(K, rows[F], rows[S], Yfit)
    pred = np.empty((len(S), nd))
    for ci, c in enumerate(C_GRID):
        m = cstar == ci
        if m.any():
            pred[:, m] = sp.predict(c)[:, m]
    return pred, cstar, (sp if keep else None)


def run_set(K5, rows, Ytrue, Yfit_by_fold, fold, inner, pos, n_ep, keep_fold_splits=False):
    """One rung on one set. K5 [NF, n_all, n_all]; rows = the set's row indices in K; Ytrue [n, nd] the TRUE labels each column is scored against;
    Yfit_by_fold[k] [n_F_k, nd] the labels the probe of column j is FITTED on (real labels, or deranged); fold, inner, pos [n] per set row.
    Returns dict: pred [n, nd] (each row scored by the probe of its own fold), cstar [NF, nd], sse_e [nd, n_ep], d_e [nd, n_ep], splits (optional)."""
    n, nd = Ytrue.shape
    pred = np.empty((n, nd))
    cstar = np.zeros((NF, nd), int)
    fmean = np.empty((n, nd))
    splits = {}
    for k in range(NF):
        F, S = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        if len(S) == 0:
            continue
        p, cs, sp = oof_fold(K5[k], rows, F, S, inner[F], Yfit_by_fold[k], keep=keep_fold_splits)
        pred[S] = p
        cstar[k] = cs
        fmean[S] = Ytrue[F].mean(0)                      # the REAL fit-fold mean: the same denominator in every draw
        if keep_fold_splits:
            splits[k] = (F, S, sp)
    err = (Ytrue - pred) ** 2
    den = (Ytrue - fmean) ** 2
    sse_e = np.zeros((n_ep, nd))
    d_e = np.zeros((n_ep, nd))
    np.add.at(sse_e, pos, err)
    np.add.at(d_e, pos, den)
    return {"pred": pred, "cstar": cstar, "sse_e": sse_e.T.copy(), "d_e": d_e.T.copy(), "splits": splits}


# ------------------------------------------------------------------------------------------------ statistics on per-episode sums
def nmse_stats(sse_e, d_e, W):
    """sse_e, d_e [nd, n_ep]; W [B, n_ep]. Returns point [nd], boot [B, nd] (episode-cluster bootstrap of the pooled ratio)."""
    point = sse_e.sum(1) / d_e.sum(1)
    boot = (W @ sse_e.T) / (W @ d_e.T)
    return point, boot


def readable(point, boot_col, ref_pts):
    """READABLE: bootstrap 97.5th percentile < 1 AND point < 2.5th percentile of the reference (numpy linear interpolation)."""
    ub = float(np.percentile(boot_col, 97.5))
    thr = float(np.percentile(ref_pts, 2.5))
    return bool(ub < 1.0 and point < thr), {"ub": ub, "ref_p2_5": thr}


def loss(d_point, d_boot_col, d_ref):
    """LOSS: paired bootstrap 2.5th percentile > 0 AND point > 97.5th percentile of the reference Deltas."""
    lb = float(np.percentile(d_boot_col, 2.5))
    thr = float(np.percentile(d_ref, 97.5))
    return bool(lb > 0.0 and d_point > thr), {"lb": lb, "ref_p97_5": thr}


class Rung:
    """Per-episode sums of one rung on the lead_gap set: column 0 real, 1..R reference, R+1..R+NC calibration, then any extra real columns."""

    def __init__(self, sse_e, d_e, W, R=R_REF, NC=N_CAL):
        self.R, self.NC = R, NC
        self.point, self.boot = nmse_stats(sse_e, d_e, W)
        self.sse_e, self.d_e = sse_e, d_e

    @property
    def ref(self):
        return self.point[1:1 + self.R]

    def cal_cols(self):
        return range(1 + self.R, 1 + self.R + self.NC)


def readable_row(rg, col=0):
    """READABLE of column `col` (0 = real; a calibration column is scored as if real) against the rung's own reference."""
    return readable(rg.point[col], rg.boot[:, col], rg.ref)


def loss_row(r0, r1, col=0):
    """LOSS(r0 -> r1) of column `col` against the reference Deltas of the same pair (both terms from the same deranged labels)."""
    dp = r1.point[col] - r0.point[col]
    db = r1.boot[:, col] - r0.boot[:, col]
    dref = r1.ref - r0.ref
    sep, info = loss(dp, db, dref)
    info["delta"] = float(dp)
    return sep, info


def calibration_share(rungs, pairs):
    """Control 1: the share of the 20 calibration draws' rows that come out separated, per row kind. rungs {name: Rung}; pairs [(a, b)] LOSS rows a -> b."""
    out = {"READABLE": [0, 0], "LOSS": [0, 0]}
    for name, rg in rungs.items():
        for c in rg.cal_cols():
            out["READABLE"][1] += 1
            out["READABLE"][0] += int(readable_row(rg, c)[0])
    for a, b in pairs:
        for c in rungs[a].cal_cols():
            out["LOSS"][1] += 1
            out["LOSS"][0] += int(loss_row(rungs[a], rungs[b], c)[0])
    return {k: {"separated": v[0], "rows": v[1], "share": (v[0] / v[1] if v[1] else float("nan"))} for k, v in out.items()}


# ------------------------------------------------------------------------------------------------ a whole rung from a feature matrix (used by the test and, through the Gram files, by the run)
def rung_from_K(K5, rows, y, ep, fold, inner, pos, n_ep, W, R=R_REF, NC=N_CAL, extra_true=None, extra_fit=None):
    """Real + R reference + NC calibration columns for target y on a set (+ optional extra real columns: extra_true [n, m], extra_fit list per fold of [n_F, m])."""
    seeds = REF_SEEDS[:R] + CAL_SEEDS[:NC]
    perm = build_perm_fit(y, ep, fold, seeds)
    Yfit = []
    for k in range(NF):
        F = np.flatnonzero(fold != k)
        cols = [y[F][:, None], perm[k]]
        if extra_fit is not None:
            cols.append(extra_fit[k])
        Yfit.append(np.concatenate(cols, 1))
    Yt = [np.repeat(y[:, None], 1 + len(seeds), 1)]
    if extra_true is not None:
        Yt.append(extra_true)
    res = run_set(K5, rows, np.concatenate(Yt, 1), Yfit, fold, inner, pos, n_ep)
    return Rung(res["sse_e"], res["d_e"], W, R, NC), res


# ------------------------------------------------------------------------------------------------ control 4 helpers (R0 identity)
def pool_field(f, gh, gw, GH=16, GW=40):
    """pull_encode_pooled.py:254-260: [T, 640, D] -> [T, gh*gw, D], non-overlapping average pooling, cells row-major."""
    T, N, D = f.shape
    assert N == GH * GW and GH % gh == 0 and GW % gw == 0
    return f.reshape(T, gh, GH // gh, gw, GW // gw, D).mean(dim=(2, 4)).reshape(T, gh * gw, D)


def rel_l2(a, b):
    """Per-frame relative L2 error ||a - b|| / ||b|| over the flattened cells: a, b [T, cells, D] float."""
    a = a.reshape(a.shape[0], -1).double()
    b = b.reshape(b.shape[0], -1).double()
    return ((a - b).norm(dim=1) / b.norm(dim=1)).numpy()


def control4_pass(err8, err4, tol=0.08, share=0.99):
    return bool((err8 <= tol).mean() >= share and (err4 <= tol).mean() >= share)
