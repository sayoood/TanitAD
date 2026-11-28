"""K-FOLD FIT + PER-CLIP SCORE — the same statistic, 13x cheaper.

⛔ THE MISPRICING THIS FIXES. I moved the ego panels from 20 to 129 clips to cure an
underpowering problem and quoted the cost as "6.5x". **Leave-one-CLIP-out is
O(n^2)** — n fits, each on n-1 clips of data — so 129 vs 20 is **42x**, not 6.5x.
Three jobs burned 53k / 21k / 19k CPU-seconds and produced no readable output.

⭐ THE FIX KEEPS THE STATISTIC IDENTICAL. The t-test consumes PER-CLIP scores, and
there is no reason the FIT must be per-clip too:

    fit on K folds (K=10)  ->  score EVERY held-out clip individually
    => still 129 clip-level scores, but 10 ridge fits instead of 129.

Clip-disjointness is preserved (a scored clip is never in its own fit), and
`rff_fold`'s internal clip-disjoint lambda selection is untouched.

⚠️ THE ESTIMATOR CHANGES SLIGHTLY — each fit sees 90 % of the clips instead of
99.2 % — so the NULL MUST BE RE-MEASURED UNDER THIS SCHEME. `--null` does exactly
that by replacing the input with Gaussian noise of the same shape, and the two are
run from the SAME code path so they cannot drift apart.
"""
from __future__ import annotations

import numpy as np


def kfold_clip_scores(X_clips, Y_clips, rff_fold, within_clip_r, k_folds=10, seed=0):
    """Return one score per clip. ``X_clips``/``Y_clips`` are per-clip arrays.

    ⛔ A clip is NEVER in the fit that scores it.
    """
    n = len(X_clips)
    if n < k_folds:
        k_folds = max(2, n // 2)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, k_folds)
    scores = np.full(n, np.nan)
    for te in folds:
        tr = [q for q in range(n) if q not in set(te.tolist())]
        Xtr = [X_clips[q] for q in tr]
        ytr = [Y_clips[q] for q in tr]
        # ⭐ ONE fit, then score each held-out clip separately: the predictions are
        # concatenated in the order we stacked them, so split at the boundaries.
        Xte = np.vstack([X_clips[q] for q in te])
        pred, _ = rff_fold(Xtr, ytr, Xte)
        off = 0
        for q in te:
            m = len(X_clips[q])
            scores[q] = within_clip_r(pred[off:off + m], Y_clips[q].ravel())
            off += m
    if not np.isfinite(scores).all():
        raise RuntimeError("a clip went unscored — fold partition is wrong")
    return scores


def t_vs_shuffled(X_clips, Y_clips, rff_fold, within_clip_r, k_folds=10, seed=0):
    """(t, r_true, r_shuf) for one cell, with the time-shuffled control computed
    through the IDENTICAL path so the two cannot diverge."""
    rng = np.random.default_rng(seed + 1)
    Ysh = [y.ravel()[rng.permutation(len(y))][:, None] for y in Y_clips]
    tr = kfold_clip_scores(X_clips, Y_clips, rff_fold, within_clip_r, k_folds, seed)
    sh = kfold_clip_scores(X_clips, Ysh, rff_fold, within_clip_r, k_folds, seed)
    d = tr - sh
    t = float(d.mean()) / max(float(d.std(ddof=1) / np.sqrt(len(d))), 1e-12)
    return t, float(tr.mean()), float(sh.mean())
