"""tp_lib.py -- pure numpy / sklearn layer of the trigger probe (PREREG_FROZEN.md).  No file I/O except memmap reads handed in by the caller.  Unit-tested in test_trigger_probe.py.

EVIDENCE CLASS of everything computed with it: MEASURED (artifacts under raw/).  ESTIMATOR: episode(clip)-cluster bootstrap, B = 2000, seed 0 -- it answers "another draw of EPISODES",
never "another training run" (H-ESTIM-SEED-1)."""
from __future__ import annotations

import hashlib

import numpy as np

DT = 0.2
LAUNCH_V0_MAX, LAUNCH_DV = 6.0, 1.0
N_BOOT, SEED = 2000, 0
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
ALPHA_GRID = (1e-2, 1.0, 10.0)


# ------------------------------------------------------------------------------------------------ labels
def go_label(v0, g, knot_end):
    """GO = (v0 < 6) and (|g[:, knot_end] - g[:, knot_end - 1]| / 0.2 >= v0 + 1).  knot_end = 29 is the REGISTERED launch mask (v8f_mini.launch_mask, mlaunch.launch_subset: the chord over the last
    0.2 s of the 6 s horizon); knot_end = 11 is the secondary GO24 (the chord over knots 11 -> 12, the end of the executed 2.4 s horizon).  ``g`` [N, 30, 2] the GT in the ego frame at NOW."""
    v0 = np.asarray(v0, np.float64)
    g = np.asarray(g, np.float64)
    vg = np.linalg.norm(g[:, knot_end] - g[:, knot_end - 1], axis=-1) / DT
    with np.errstate(invalid="ignore"):
        return (v0 < LAUNCH_V0_MAX) & (vg >= v0 + LAUNCH_DV)


# ------------------------------------------------------------------------------------------------ folds
def sha12_int(sha12: str) -> int:
    return int(sha12, 16)


def outer_fold(sha12s, k=5):
    return np.array([sha12_int(s) % k for s in sha12s], np.int64)


def inner_fold(sha12s, k=3, outer_k=5):
    return np.array([(sha12_int(s) // outer_k) % k for s in sha12s], np.int64)


# ------------------------------------------------------------------------------------------------ AUC (tie-aware, weighted)
class WAuc:
    """Mann-Whitney AUC with window weights (multiplicities of an episode bootstrap resample).  Precomputes the tie groups of ``score`` once; ``auc(w)`` is then O(N)."""

    def __init__(self, score, y):
        score = np.asarray(score, np.float64)
        self.y = np.asarray(y).astype(bool)
        _, self.inv = np.unique(score, return_inverse=True)
        self.G = int(self.inv.max()) + 1

    def auc(self, w=None):
        w = np.ones(len(self.y)) if w is None else np.asarray(w, np.float64)
        gp = np.bincount(self.inv, weights=w * self.y, minlength=self.G)
        gn = np.bincount(self.inv, weights=w * (~self.y), minlength=self.G)
        sp, sn = gp.sum(), gn.sum()
        if sp <= 0 or sn <= 0:
            return float("nan")
        cumneg = np.cumsum(gn) - gn
        return float((gp * (cumneg + 0.5 * gn)).sum() / (sp * sn))


def boot_counts(clip_idx, n_boot=N_BOOT, seed=SEED):
    """[n_boot, K] multiplicities of a clip-cluster bootstrap over the K distinct clips of ``clip_idx`` (ints); also the dense clip map."""
    u, inv = np.unique(np.asarray(clip_idx), return_inverse=True)
    rng = np.random.default_rng(seed)
    K = len(u)
    cnt = np.stack([np.bincount(rng.integers(0, K, K), minlength=K) for _ in range(n_boot)])
    return cnt, inv


def auc_ci(score, y, clip_idx, n_boot=N_BOOT, seed=SEED, counts=None):
    """{auc, lo, hi, n, n_pos, n_clips}: the pooled AUC and the 95 % percentile CI of an episode-cluster bootstrap."""
    wa = WAuc(score, y)
    cnt, inv = counts if counts is not None else boot_counts(clip_idx, n_boot, seed)
    b = np.array([wa.auc(c[inv]) for c in cnt])
    b = b[np.isfinite(b)]
    return {"auc": wa.auc(), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)), "n": int(len(y)), "n_pos": int(np.sum(y)), "n_clips": int(len(np.unique(clip_idx)))}


def dauc_ci(score_a, score_b, y, clip_idx, n_boot=N_BOOT, seed=SEED, counts=None):
    """Paired dAUC = AUC(a) - AUC(b) on the SAME windows with the SAME bootstrap resample."""
    wa, wb = WAuc(score_a, y), WAuc(score_b, y)
    cnt, inv = counts if counts is not None else boot_counts(clip_idx, n_boot, seed)
    d = np.array([wa.auc(c[inv]) - wb.auc(c[inv]) for c in cnt])
    d = d[np.isfinite(d)]
    return {"delta": wa.auc() - wb.auc(), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "auc_a": wa.auc(), "auc_b": wb.auc(),
            "excludes_0": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0), "n": int(len(y))}


# ------------------------------------------------------------------------------------------------ streaming randomised PCA
class StreamPCA:
    """PCA-k of an [N, D] matrix that is only ever read in chunks (``read(rows) -> float32 [len(rows), D]``).  Centred, NOT scaled.  Randomised SVD (Halko et al.): oversample 32,
    ``n_iter`` power iterations, seed ``seed``; the passes over the data are 1 (mean) + 1 + 2 n_iter + 1."""

    def __init__(self, k=64, oversample=32, n_iter=2, seed=0, chunk=500):
        self.k, self.p, self.q, self.seed, self.chunk = k, oversample, n_iter, seed, chunk

    def _chunks(self, rows):
        for a in range(0, len(rows), self.chunk):
            yield rows[a:a + self.chunk]

    def fit(self, read, rows, D):
        rows = np.asarray(rows)
        N = len(rows)
        kk = min(self.k + self.p, N, D)
        mu = np.zeros(D, np.float64)
        for r in self._chunks(rows):
            mu += read(r).astype(np.float64).sum(0)
        mu = (mu / N).astype(np.float32)
        rng = np.random.default_rng(self.seed)
        Om = rng.standard_normal((D, kk)).astype(np.float32)
        Y = np.concatenate([(read(r) - mu) @ Om for r in self._chunks(rows)], 0)           # [N, kk]
        for _ in range(self.q):
            Q, _ = np.linalg.qr(Y)
            Zt = np.zeros((D, kk), np.float32)
            a = 0
            for r in self._chunks(rows):
                Zt += (read(r) - mu).T @ Q[a:a + len(r)]
                a += len(r)
            Y = np.concatenate([(read(r) - mu) @ Zt for r in self._chunks(rows)], 0)
        Q, _ = np.linalg.qr(Y)
        Bt = np.zeros((D, kk), np.float32)
        a = 0
        for r in self._chunks(rows):
            Bt += (read(r) - mu).T @ Q[a:a + len(r)]
            a += len(r)
        _, S, Vt = np.linalg.svd(Bt.T.astype(np.float64), full_matrices=False)               # B = Q^T X_c : [kk, D]
        self.mu, self.comp, self.S = mu, Vt[:self.k].astype(np.float32), S[:self.k] / np.sqrt(max(N - 1, 1))
        return self

    def transform(self, read, rows):
        rows = np.asarray(rows)
        out = [(read(r) - self.mu) @ self.comp.T for r in self._chunks(rows)]
        return np.concatenate(out, 0).astype(np.float32) if out else np.zeros((0, self.k), np.float32)


# ------------------------------------------------------------------------------------------------ probes
def _make(family, hp, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    if family == "linear":
        return LogisticRegression(C=hp, solver="lbfgs", max_iter=2000)
    return MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam", alpha=hp, learning_rate_init=1e-3, early_stopping=True, validation_fraction=0.15,
                         n_iter_no_change=10, max_iter=300, random_state=seed)


def _score(model, X):
    return model.decision_function(X) if hasattr(model, "decision_function") else model.predict_proba(X)[:, 1]


def fit_probe(Xtr, ytr, inner_grp, family, seed=0):
    """Standardise on the training rows, pick C / alpha by inner 3-fold (grouped) mean AUC, refit on all training rows.  Returns (predict_fn, hp, cv_table)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xs = sc.transform(Xtr).astype(np.float64)
    grid = C_GRID if family == "linear" else ALPHA_GRID
    table = {}
    ytr = np.asarray(ytr).astype(int)
    for hp in grid:
        aucs = []
        for f in range(3):
            te = inner_grp == f
            tr = ~te
            if te.sum() < 5 or tr.sum() < 5 or len(np.unique(ytr[te])) < 2 or len(np.unique(ytr[tr])) < 2:
                continue
            m = _make(family, hp, seed).fit(Xs[tr], ytr[tr])
            aucs.append(roc_auc_score(ytr[te], _score(m, Xs[te])))
        table[hp] = float(np.mean(aucs)) if aucs else float("nan")
    ok = {k: v for k, v in table.items() if np.isfinite(v)}
    best = max(ok, key=lambda k: (ok[k], -abs(np.log10(k) - 0.0))) if ok else grid[len(grid) // 2]
    m = _make(family, best, seed).fit(Xs, ytr)

    def predict(X):
        return _score(m, sc.transform(X).astype(np.float64))
    return predict, best, table


def cv_oof(feat_fn, y, outer, inner, family, seed=0, folds=range(5)):
    """Pooled out-of-fold scores.  ``feat_fn(f)`` returns the feature matrix [N, d] for fold ``f`` (f = 0..4: PCA / scaler fitted on that fold's training rows; the matrix holds ALL N rows).
    Returns (oof [N], hps)."""
    N = len(y)
    oof = np.full(N, np.nan)
    hps = {}
    for f in folds:
        te = outer == f
        X = feat_fn(f)
        pred, hp, tab = fit_probe(X[~te], y[~te], inner[~te], family, seed)
        oof[te] = pred(X[te])
        hps[int(f)] = {"hp": hp, "inner_cv_auc": {str(k): v for k, v in tab.items()}}
    return oof, hps


def shuffle_labels(y, seed):
    return np.random.default_rng(seed).permutation(np.asarray(y))


def donor_labels(y, clip_idx, seed):
    """C2: each window's label is drawn from a window of ANOTHER clip (a random derangement of the distinct clips, then a random window of the donor)."""
    clip_idx = np.asarray(clip_idx)
    u = np.unique(clip_idx)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(u))
    while len(u) > 1 and np.any(perm == np.arange(len(u))):
        perm = rng.permutation(len(u))
    donor_of = dict(zip(u.tolist(), u[perm].tolist()))
    by = {int(c): np.flatnonzero(clip_idx == c) for c in u}
    out = np.empty(len(y), y.dtype)
    for i in range(len(y)):
        out[i] = y[rng.choice(by[donor_of[int(clip_idx[i])]])]
    return out


# ------------------------------------------------------------------------------------------------ POST-FREEZE sensitivity metric: the mean of per-fold AUCs
class FoldWAuc:
    """POST-FREEZE (found by control C2, see RESULTS.md): the pooled out-of-fold AUC is biased BELOW 0.5 on clip-clustered null labels, because each fold's model carries an intercept fitted to
    the OTHER folds' prevalence, so a fold with more positives is scored lower.  The mean over outer folds of the within-fold AUC does not carry that intercept.  Same bootstrap weights."""

    def __init__(self, score, y, fold):
        self.parts = []
        fold = np.asarray(fold)
        for f in np.unique(fold):
            ix = np.flatnonzero(fold == f)
            if len(np.unique(np.asarray(y)[ix])) == 2:
                self.parts.append((ix, WAuc(np.asarray(score)[ix], np.asarray(y)[ix])))

    def auc(self, w=None):
        v = [wa.auc(None if w is None else np.asarray(w)[ix]) for ix, wa in self.parts]
        v = [x for x in v if np.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")


def auc_ci_fold(score, y, clip_idx, fold, n_boot=N_BOOT, seed=SEED, counts=None):
    fa = FoldWAuc(score, y, fold)
    cnt, inv = counts if counts is not None else boot_counts(clip_idx, n_boot, seed)
    b = np.array([fa.auc(c[inv]) for c in cnt])
    b = b[np.isfinite(b)]
    return {"auc": fa.auc(), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)), "n": int(len(y)), "n_pos": int(np.sum(y))}


def dauc_ci_fold(score_a, score_b, y, clip_idx, fold, n_boot=N_BOOT, seed=SEED, counts=None):
    fa, fb = FoldWAuc(score_a, y, fold), FoldWAuc(score_b, y, fold)
    cnt, inv = counts if counts is not None else boot_counts(clip_idx, n_boot, seed)
    d = np.array([fa.auc(c[inv]) - fb.auc(c[inv]) for c in cnt])
    d = d[np.isfinite(d)]
    return {"delta": fa.auc() - fb.auc(), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "auc_a": fa.auc(), "auc_b": fb.auc(),
            "excludes_0": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0), "n": int(len(y))}
