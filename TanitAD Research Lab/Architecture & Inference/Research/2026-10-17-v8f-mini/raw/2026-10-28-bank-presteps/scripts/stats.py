"""Clip-cluster bootstrap in the construction of taniteval/ci.py (_draws): resample CLUSTERS with replacement, B = 2000, seed 0, 95 % percentile.

Works on per-cluster sufficient statistics: for a ratio of sums num/den (a mean over records, a share) resampling clusters and recomputing
sum(num[pick]) / sum(den[pick]) is exactly "resample the clusters' records and recompute the statistic".
`tests/test_stats.py` pins the equivalence to `ci.episode_cluster_bootstrap` (same seed, string-sorted cluster order).
"""
from __future__ import annotations

import numpy as np

B_DEFAULT = 2000
SEED_DEFAULT = 0


def picks(n_clusters: int, B: int = B_DEFAULT, seed: int = SEED_DEFAULT, chunk: int = 200):
    """Yield [<=chunk, n_clusters] int arrays of resampled cluster indices (indices into the string-sorted unique cluster ids)."""
    rng = np.random.default_rng(seed)
    done = 0
    while done < B:
        m = min(chunk, B - done)
        yield rng.choice(n_clusters, size=(m, n_clusters), replace=True)
        done += m


def ratio_boot(num, den, B: int = B_DEFAULT, seed: int = SEED_DEFAULT, alpha: float = 0.05):
    """num, den [n_clusters] sufficient statistics (cluster order = string-sorted ids). Returns dict(point, lo, hi, se, n_clusters, boots)."""
    num = np.asarray(num, np.float64)
    den = np.asarray(den, np.float64)
    point = float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")
    boots = []
    for idx in picks(len(num), B, seed):
        d = den[idx].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            boots.append(np.where(d > 0, num[idx].sum(1) / d, np.nan))
    boots = np.concatenate(boots)
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi), "se": float(np.nanstd(boots, ddof=1)), "n_clusters": int(len(num)),
            "n_boot": int(B), "seed": int(seed), "n_nan_boot": int(np.isnan(boots).sum()), "boots": boots}


def cluster_sums(cluster_ids, *values, n_clusters=None):
    """Sum each value array within clusters. cluster_ids int [N] (0..n_clusters-1). Returns arrays [n_clusters]."""
    cid = np.asarray(cluster_ids, np.int64)
    n = int(n_clusters if n_clusters is not None else cid.max() + 1)
    return [np.bincount(cid, weights=np.asarray(v, np.float64), minlength=n) for v in values]


def joint_boot(clusters: dict, fn, B: int = B_DEFAULT, seed: int = SEED_DEFAULT, alpha: float = 0.05):
    """Several per-cluster sufficient statistics resampled TOGETHER; fn(**sums over the resampled clusters) -> scalar (vectorised over a leading axis)."""
    keys = list(clusters)
    n = len(clusters[keys[0]])
    point = float(fn(**{k: np.asarray(clusters[k], np.float64).sum() for k in keys}))
    boots = []
    for idx in picks(n, B, seed):
        boots.append(np.asarray(fn(**{k: np.asarray(clusters[k], np.float64)[idx].sum(1) for k in keys}), np.float64))
    boots = np.concatenate(boots)
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_clusters": int(n), "n_boot": int(B), "seed": int(seed),
            "n_nan_boot": int(np.isnan(boots).sum())}


def two_sample_ratio_boot(num_a, den_a, num_b, den_b, B: int = B_DEFAULT, seed_a: int = 0, seed_b: int = 1, alpha: float = 0.05):
    """r = (num_a/den_a) / (num_b/den_b) with INDEPENDENT cluster resampling of the two sides (disjoint clip sets), paired by draw index."""
    a = ratio_boot(num_a, den_a, B, seed_a, alpha)
    b = ratio_boot(num_b, den_b, B, seed_b, alpha)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = a["boots"] / b["boots"]
    point = a["point"] / b["point"] if b["point"] != 0 else float("nan")
    lo, hi = np.nanpercentile(r, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n_boot": int(B), "a": {k: v for k, v in a.items() if k != "boots"},
            "b": {k: v for k, v in b.items() if k != "boots"}, "frac_boot_den_nonpositive": float((b["boots"] <= 0).mean())}
