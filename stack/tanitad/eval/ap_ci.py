"""Average precision with the program's BINDING interval estimator.

Why this module exists
----------------------
``taniteval.ci.episode_cluster_bootstrap`` resamples episodes and applies a
*reducer* to per-window values. That covers every mean-like metric (ADE, MAE,
miss-rate) but it **cannot express average precision**: AP is a property of the
joint ranking of (label, score) over the whole set, not a reduction of a
per-window number. Passing AP through a mean reducer is not merely imprecise —
it is a different statistic.

The situation-classifier stream reports AP and therefore had **no admissible
interval at all**: every sitclf number in the hub is a bare point estimate.
Per CLAUDE.md ("never quote an interval without its estimator", and "a missing
metric is a work item, not an excuse") this module supplies one.

The resampling is **not reimplemented**. :func:`_draws` below delegates to
``taniteval.ci._draws`` whenever ``taniteval`` is importable, so the episode
draws here are bit-identical to the ones behind every ADE interval in the
program; the local fallback exists only so this module imports on a pod that
has ``stack/`` but not ``taniteval/``, and ``tests/test_ap_ci.py`` asserts the
two agree draw-for-draw. Only the *statistic* changes, never the resampling.

Clusters, not windows, are the unit — consecutive frames of one clip are
nowhere near independent, so a frame-level interval would be far too narrow.
"""

from __future__ import annotations

import numpy as np

DEFAULT_N_BOOT = 2000
ESTIMATOR = "ap_episode_cluster_bootstrap"
PAIRED_ESTIMATOR = "paired_ap_episode_cluster_bootstrap"


# --------------------------------------------------------------------------- #
# the statistic                                                               #
# --------------------------------------------------------------------------- #
def average_precision(y, s) -> float:
    """Average precision of scores ``s`` against binary labels ``y``.

    Identical formulation to the situation-classifier trainer
    (``sc_train.py:231``) so the numbers stay comparable to the banked arms:
    sort by descending score, ``AP = sum(P@k * y_k) / n_pos``. ``mergesort`` is
    stable, so ties keep input order rather than resolving in the scorer's
    favour.

    Returns ``nan`` when there is no positive — a degenerate AP must not be
    silently reported as 0.0, which would read like a failing arm.
    """
    y = np.asarray(y).astype(np.float64).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError(f"y/s length mismatch: {y.shape} vs {s.shape}")
    if y.size == 0:
        raise ValueError("average_precision needs a non-empty set")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("average_precision needs binary 0/1 labels")
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    yt = y[order]
    prec = np.cumsum(yt) / np.arange(1, yt.size + 1)
    return float((prec * yt).sum() / n_pos)


def ap_lift(y, s) -> float:
    """AP divided by the base rate — "how many times better than chance".

    The base rate *is* the expected AP of a random scorer, so this is the
    scale-free form that survives a change of corpus. Comparing raw AP across
    caches with different base rates is the error the sitclf stream has already
    made once (four caches in play, base rate 0.028 -> 0.098).
    """
    y = np.asarray(y).astype(np.float64).ravel()
    base = y.mean()
    if base <= 0:
        return float("nan")
    return average_precision(y, s) / base


# --------------------------------------------------------------------------- #
# resampling — delegated to taniteval so the draws are the program's           #
# --------------------------------------------------------------------------- #
def _episode_index(eid):
    e = np.asarray([str(x) for x in eid])
    if e.size == 0:
        raise ValueError("ap episode-cluster bootstrap needs a non-empty eid array")
    uniq = np.unique(e)
    return uniq, {u: np.flatnonzero(e == u) for u in uniq}


def _draws(uniq, idx_by_ep, n_boot, seed):
    """Yield ``n_boot`` index selections, resampling CLUSTERS with replacement.

    Delegates to ``taniteval.ci._draws`` when available so this module cannot
    drift away from the estimator the rest of the program uses.
    """
    try:                                                    # noqa: SIM105
        from taniteval.ci import _draws as _tv_draws        # noqa: PLC0415
    except Exception:                                       # pragma: no cover
        rng = np.random.default_rng(seed)
        n_ep = len(uniq)
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=n_ep, replace=True)
            yield np.concatenate([idx_by_ep[p] for p in pick])
        return
    yield from _tv_draws(uniq, idx_by_ep, n_boot, seed)


def _bounds(vals, alpha):
    """Percentile bounds, or ``(nan, nan)`` when EVERY draw was degenerate.

    A situation can be so rare that most cluster resamples contain no positive
    at all; at the limit none does and ``vals`` is empty. ``np.percentile`` then
    raises an ``IndexError`` from deep inside numpy, which reads as a library
    bug rather than "this arm is unpowered here". Returning nan bounds keeps
    ``n_boot_valid: 0`` visible in the row, which is the honest report.
    """
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _pack(point, boots, alpha, extra):
    lo, hi = _bounds(boots, alpha)
    out = {"lo": round(float(lo), 5), "hi": round(float(hi), 5),
           "ci95": round(float((hi - lo) / 2.0), 5),
           "se": (round(float(boots.std(ddof=1)), 5) if boots.size > 1
                  else float("nan"))}
    out.update(extra)
    out["_lo_raw"], out["_hi_raw"] = float(lo), float(hi)
    return {**{"point": round(float(point), 5)}, **out}


def ap_episode_cluster_bootstrap(y, s, eid, *, n_boot=DEFAULT_N_BOOT, seed=0,
                                 alpha=0.05, lift=False) -> dict:
    """Percentile CI on AP (or AP-lift), resampling CLUSTERS with replacement.

    The point estimate is the **full-set** AP — the bootstrap supplies only the
    interval and never moves the centre. (That is the `overlapping_holdout_se`
    lesson: a mean-of-split-means both widens *and* biases.)

    Draws whose resample contains no positive yield ``nan`` AP; they are dropped
    from the percentile with ``n_boot_valid`` recorded, rather than silently
    propagating a nan bound.
    """
    y = np.asarray(y).astype(np.float64).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    stat = ap_lift if lift else average_precision
    uniq, idx_by_ep = _episode_index(eid)
    if len(eid) != y.size:
        raise ValueError(f"eid/label length mismatch: {len(eid)} vs {y.size}")
    point = stat(y, s)
    boots = np.array([stat(y[sel], s[sel])
                      for sel in _draws(uniq, idx_by_ep, n_boot, seed)])
    ok = np.isfinite(boots)
    return _pack(point, boots[ok], alpha, {
        "statistic": "ap_lift" if lift else "ap",
        "base_rate": round(float(y.mean()), 6),
        "n_pos": int(y.sum()), "n_windows": int(y.size),
        "n_episodes": int(len(uniq)), "n_boot": int(n_boot),
        "n_boot_valid": int(ok.sum()), "estimator": ESTIMATOR})


def stat_episode_cluster_bootstrap(fn, eid, *, n_boot=DEFAULT_N_BOOT, seed=0,
                                   alpha=0.05, name="stat") -> dict:
    """CI on ANY set-level statistic ``fn(row_indices) -> float``.

    AP is not the only metric that resists a per-window reduction: balanced
    accuracy, a confusion-matrix recall and R² are all set-level too, and the
    TACTICAL family is built from exactly those. Without this they would have to
    be quoted bare, which the program forbids.
    """
    uniq, idx_by_ep = _episode_index(eid)
    point = float(fn(np.arange(len(eid))))
    boots = np.array([float(fn(sel))
                      for sel in _draws(uniq, idx_by_ep, n_boot, seed)])
    ok = np.isfinite(boots)
    return _pack(point, boots[ok], alpha, {
        "statistic": name, "n_windows": int(len(eid)),
        "n_episodes": int(len(uniq)), "n_boot": int(n_boot),
        "n_boot_valid": int(ok.sum()), "estimator": ESTIMATOR})


def paired_stat_episode_cluster_bootstrap(fn_a, fn_b, eid, *,
                                          n_boot=DEFAULT_N_BOOT, seed=0,
                                          alpha=0.05, name="stat") -> dict:
    """CI on ``fn_a(sel) - fn_b(sel)`` over the SAME resampled clusters.

    Both arms must be evaluated on the same rows; the shared per-cluster
    difficulty then cancels inside each draw, which is the only valid way to
    compare two arms on one substrate.
    """
    uniq, idx_by_ep = _episode_index(eid)
    allidx = np.arange(len(eid))
    point = float(fn_a(allidx)) - float(fn_b(allidx))
    d = np.array([float(fn_a(sel)) - float(fn_b(sel))
                  for sel in _draws(uniq, idx_by_ep, n_boot, seed)])
    ok = np.isfinite(d)
    dv = d[ok]
    lo, hi = _bounds(dv, alpha)
    return {"delta": round(float(point), 5), "lo": round(float(lo), 5),
            "hi": round(float(hi), 5), "ci95": round(float((hi - lo) / 2.0), 5),
            "p_delta_gt0": (round(float((dv > 0).mean()), 4) if dv.size
                            else float("nan")),
            "separated": bool(lo > 0 or hi < 0), "statistic": name,
            "n_windows": int(len(eid)), "n_episodes": int(len(uniq)),
            "n_boot": int(n_boot), "n_boot_valid": int(ok.sum()),
            "estimator": PAIRED_ESTIMATOR}


def paired_ap_episode_cluster_bootstrap(y, s_a, s_b, eid, *,
                                        n_boot=DEFAULT_N_BOOT, seed=0,
                                        alpha=0.05, lift=False) -> dict:
    """CI on ``AP(a) - AP(b)`` with the SAME resampled clusters in each draw.

    Both arms are scored on the same rows and the same labels, so the shared
    per-cluster difficulty cancels *inside* each draw. This is the only valid
    two-arm comparison here: combining two single-arm AP intervals in quadrature
    would be wrong because the arms are strongly positively correlated.

    ``separated`` (the CI excludes zero) is the decision predicate and is
    evaluated on the UNROUNDED bounds.
    """
    y = np.asarray(y).astype(np.float64).ravel()
    a = np.asarray(s_a, dtype=np.float64).ravel()
    b = np.asarray(s_b, dtype=np.float64).ravel()
    if a.shape != b.shape or a.shape != y.shape:
        raise ValueError(f"paired AP needs aligned arms: y{y.shape} a{a.shape} b{b.shape}")
    if len(eid) != y.size:
        raise ValueError(f"eid/label length mismatch: {len(eid)} vs {y.size}")
    stat = ap_lift if lift else average_precision
    uniq, idx_by_ep = _episode_index(eid)
    point = stat(y, a) - stat(y, b)
    d = np.array([stat(y[sel], a[sel]) - stat(y[sel], b[sel])
                  for sel in _draws(uniq, idx_by_ep, n_boot, seed)])
    ok = np.isfinite(d)
    dv = d[ok]
    lo, hi = _bounds(dv, alpha)
    separated = bool(lo > 0 or hi < 0)
    return {"delta": round(float(point), 5),
            "lo": round(float(lo), 5), "hi": round(float(hi), 5),
            "ci95": round(float((hi - lo) / 2.0), 5),
            "p_delta_gt0": (round(float((dv > 0).mean()), 4) if dv.size
                            else float("nan")),
            "separated": separated,
            "statistic": "ap_lift" if lift else "ap",
            "ap_a": round(float(stat(y, a)), 5),
            "ap_b": round(float(stat(y, b)), 5),
            "base_rate": round(float(y.mean()), 6),
            "n_pos": int(y.sum()), "n_windows": int(y.size),
            "n_episodes": int(len(uniq)), "n_boot": int(n_boot),
            "n_boot_valid": int(ok.sum()),
            "estimator": PAIRED_ESTIMATOR}
