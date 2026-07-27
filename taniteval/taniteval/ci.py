"""TanitEval interval estimators — the program's ``CI-separated`` predicate.

WHY THIS MODULE EXISTS
----------------------
Until 2026-07-20 every single-arm interval in the registry came from
``bench._agg``: draw ``n_splits=8`` *independent random* 20 % episode holdouts,
score each, and report ``1.96 * std(v) / sqrt(8)``. That construction has three
defects, all confirmed at line level in the 360° review (W1):

1. **It is not a jackknife.** A jackknife is a *systematic* delete-1/delete-d
   resampling with a variance-inflation factor. Drawing 8 random 20 % holdouts
   is **Monte-Carlo cross-validation / repeated random subsampling**. The label
   "8-split episode-disjoint jackknife" was wrong everywhere it appeared.
2. **The sqrt(n) divisor assumes the splits are independent.** They are drawn
   from the *same* 40 episodes and overlap pairwise, so ``std/sqrt(8)`` is the
   standard error of the mean of 8 *correlated* estimates.
3. **It estimates the wrong quantity.** As ``n_splits -> inf`` the statistic
   converges to the full-set mean and its SE -> 0. It therefore describes
   **split-selection noise**, not uncertainty about the arm's performance on
   driving. The relevant sample size is the **40 val EPISODES**, not 881
   correlated windows and not 8 overlapping splits.

The replacement is an **episode-cluster bootstrap**: resample the episodes with
replacement, recompute the metric on the resampled window set, and report
percentile bounds. Windows inside one episode (stride 8 over a 199-frame clip)
are strongly dependent, so the episode is the independent unit and must be the
resampling unit.

WHAT IS PRESERVED
-----------------
``overlapping_holdout_se`` keeps the OLD estimator available under an honest
name so every historically published interval stays exactly reproducible. It is
deprecated for new claims, not deleted.

ONLY numpy. No torch, no pod paths — so the estimators are unit-testable
anywhere, which is what W8 asked for.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_N_BOOT",
    "overlapping_holdout_se",
    "episode_index",
    "episode_cluster_bootstrap",
    "paired_episode_cluster_bootstrap",
    "bootstrap_metrics",
    "REDUCERS",
    "resolve_reducer",
    "DISPLAY_DP",
    "MAX_DISPLAY_DP",
]

DEFAULT_N_BOOT = 2000
_OLD_ESTIMATOR = "overlapping_holdout_se"
_NEW_ESTIMATOR = "episode_cluster_bootstrap"

#: The program's publication precision for an interval. Everything readable is
#: rendered here; only :func:`_render_bounds` may go past it, and only when 4 dp
#: would make the printed interval CONTRADICT the printed verdict.
DISPLAY_DP = 4
#: The hard stop for that escalation. Past ~1e-12 a percentile of 2 000 float64
#: bootstrap draws is not carrying information, it is carrying arithmetic noise,
#: so the honest output there is a MARKER rather than more digits.
MAX_DISPLAY_DP = 12


def _render_bounds(lo: float, hi: float, separated: bool) -> tuple[int, bool]:
    """-> ``(decimals, degenerate)`` — a rendering that cannot contradict ``separated``.

    ⛔ **THE DEFECT THIS FIXES (2026-07-27).** ``separated`` is computed on the
    UNROUNDED bounds (``lo > 0 or hi < 0``) while ``delta``/``lo``/``hi`` were
    rounded to 4 dp for publication. When the true bounds are ~1e-9 of one sign —
    which happens whenever two arms are bit-identical or near-identical, e.g. a
    from-scratch model where the learned gate is exactly 0.0 — the emitted record
    reads::

        {"delta": 0.0, "lo": 0.0, "hi": 0.0, "separated": true}

    MEASURED in ``…/incoming/2026-07-27-vtband-decision/raw/legA_v5config_structural.json``.
    ⚠️ **Every stream in this program reads this output**, and an automated gate
    keying on ``separated`` sees a verdict beside a null effect.

    ⛔ **The statistics are NOT changed** — the unrounded test is the correct one.
    A rounded-bounds test would silently redefine ``separated`` as "separated by
    at least 5e-5", which is a *threshold nobody registered* and would flip real
    verdicts. **The display is what lies, so the display is what is fixed.**

    THE CHOSEN FIX, and why it beats the alternatives
    --------------------------------------------------
    **Adaptive precision**: keep 4 dp — so no existing number changes — unless 4
    dp would put the printed interval on the wrong side of zero, and in that case
    show exactly as many digits as it takes to agree with the verdict.

    * *"always more significant figures"* was rejected: it changes **every**
      published number to fix a degenerate case, and 12-dp intervals invite
      reading bootstrap noise as signal.
    * *"scientific notation below a threshold"* was rejected: it needs a
      threshold, and the value is only ever compared against **zero**, so the
      question is a SIGN question, not a magnitude one. A threshold would have to
      be tuned and would itself become quotable.
    * *"a marker alone"* was rejected as the sole fix: a reader who ignores the
      marker still sees ``0.0 [0.0, 0.0] separated``. The numbers must be right
      first; the marker is what remains when they cannot be.

    The marker is therefore the FALLBACK, not the mechanism: when even
    :data:`MAX_DISPLAY_DP` cannot reproduce the verdict — i.e. a bound is smaller
    than 1e-12, or is exactly 0.0 — ``degenerate`` goes True and the caller emits
    an explicit field rather than a number that lies more quietly.
    """
    for dp in range(DISPLAY_DP, MAX_DISPLAY_DP + 1):
        if bool(round(lo, dp) > 0 or round(hi, dp) < 0) == separated:
            return dp, False
    return MAX_DISPLAY_DP, True


# --------------------------------------------------------------------------- #
# The DEPRECATED estimator, under its honest name                              #
# --------------------------------------------------------------------------- #
def overlapping_holdout_se(values) -> float:
    """SE of the mean of ``n`` OVERLAPPING random-holdout estimates.

    **DEPRECATED — anti-conservative. Reproduction only.**

    This is the exact arithmetic of the pre-2026-07-20 ``bench._agg`` ci95
    (``1.96 * nanstd(v) / sqrt(len(v))``), preserved verbatim so every published
    interval remains reproducible. It is NOT a jackknife and it is NOT a
    standard error for the arm's performance: the splits share episodes, so the
    dispersion it measures is split-selection noise and it shrinks toward zero
    as ``n_splits`` grows. Use :func:`episode_cluster_bootstrap` for any new
    claim.

    Returns the 95 % half-width (i.e. what the registry printed as ``± ci95``).
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan")
    return float(1.96 * np.nanstd(v) / max(1, v.size) ** 0.5)


# --------------------------------------------------------------------------- #
# Episode clustering                                                           #
# --------------------------------------------------------------------------- #
def episode_index(eid):
    """``(unique_episodes, {episode: window_indices})`` for a per-window eid list.

    Fails loud on an empty eid — a silent zero-episode bootstrap would report a
    NaN interval that reads like a passing gate.
    """
    e = np.asarray([str(x) for x in eid])
    if e.size == 0:
        raise ValueError("episode_cluster bootstrap needs a non-empty eid array")
    uniq = np.unique(e)
    return uniq, {u: np.flatnonzero(e == u) for u in uniq}


def _draws(uniq, idx_by_ep, n_boot, seed):
    """Yield ``n_boot`` window-index selections, resampling EPISODES w/ replacement."""
    rng = np.random.default_rng(seed)
    n_ep = len(uniq)
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=n_ep, replace=True)
        yield np.concatenate([idx_by_ep[p] for p in pick])


# --------------------------------------------------------------------------- #
# Reducers — how a per-window component becomes a scalar metric                #
# --------------------------------------------------------------------------- #
def _red_mean(v):
    return float(np.nanmean(v))


def _red_rms(v):
    """sqrt(mean(v)) where v holds per-window SQUARED errors (rmse_xy)."""
    return float(np.sqrt(np.nanmean(v)))


def _red_median(v):
    return float(np.nanmedian(v))


def _quantile_reducer(q):
    def _r(v):
        return float(np.nanquantile(v, q))
    _r.__name__ = f"p{int(round(q * 100))}"
    return _r


# ``mean``/``rms`` are the historical two. ``median``/``p90``/``p10`` exist
# because R5 of the TanitEval v2 metric suite is a MEASURED finding, not a
# style preference: heading MAE@2s has a bootstrap CI of [2.34, 12.02] around
# a mean of 6.61 deg — a handful of windows dominate, so the mean is not the
# right reducer for heading, curvature or jerk. Reported as median +
# exceedance rate instead (suite §0 R5, escalation E3/P6).
REDUCERS = {"mean": _red_mean, "rms": _red_rms, "median": _red_median,
            "p90": _quantile_reducer(0.90), "p10": _quantile_reducer(0.10)}


def resolve_reducer(reduce):
    """``REDUCERS`` name **or** any callable ``values -> float``.

    The callable path is what makes kappa, macro-F1 and AUC bootstrappable
    with the same estimator as a mean (suite E3/P6); without it those
    statistics would have had to invent their own interval, which is how the
    deprecated one survived for so long.
    """
    if callable(reduce):
        return reduce
    if reduce not in REDUCERS:
        raise KeyError(f"unknown reducer {reduce!r}; known: "
                       f"{sorted(REDUCERS)} (or pass a callable)")
    return REDUCERS[reduce]


def _reducer_name(reduce):
    if callable(reduce):
        return getattr(reduce, "__name__", "callable")
    return reduce


# --------------------------------------------------------------------------- #
# The estimators                                                               #
# --------------------------------------------------------------------------- #
def episode_cluster_bootstrap(per_window, eid, reduce="mean",
                              n_boot=DEFAULT_N_BOOT, seed=0, alpha=0.05) -> dict:
    """Percentile CI on a per-window metric, resampling EPISODES with replacement.

    ``per_window`` [N] are the per-window components (displacements, 0/1 miss
    flags, squared errors for ``reduce="rms"``); ``eid`` [N] their episode ids.
    ``reduce`` is a :data:`REDUCERS` name or any callable ``values -> float``.

    The point estimate is the **full-set** value — the bootstrap supplies the
    interval, it does not move the mean. Returns a dict carrying its own
    provenance (``estimator``, ``n_episodes``, ``n_boot``) so a number can never
    be quoted without the construction that produced it.
    """
    v = np.asarray(per_window, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(f"per_window must be 1-D per-window values, got {v.shape}")
    if len(eid) != v.size:
        raise ValueError(f"eid/per_window length mismatch: {len(eid)} vs {v.size}")
    red = resolve_reducer(reduce)
    uniq, idx_by_ep = episode_index(eid)
    point = red(v)
    boots = np.array([red(v[sel]) for sel in _draws(uniq, idx_by_ep, n_boot, seed)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": round(point, 4),
            "lo": round(float(lo), 4),
            "hi": round(float(hi), 4),
            # symmetric half-width, for drop-in comparison with the old ± ci95
            "ci95": round(float((hi - lo) / 2.0), 4),
            "se": round(float(boots.std(ddof=1)), 4),
            "reducer": _reducer_name(reduce),
            "n_windows": int(v.size),
            "n_episodes": int(len(uniq)),
            "n_boot": int(n_boot),
            "estimator": _NEW_ESTIMATOR}


def paired_episode_cluster_bootstrap(a, b, eid, n_boot=DEFAULT_N_BOOT, seed=0,
                                     alpha=0.05, reduce="mean") -> dict:
    """CI on ``reduce(a) - reduce(b)`` with the SAME resampled episodes each draw.

    Both arms are scored on the *same* windows, so the shared per-window
    difficulty cancels inside each draw. This is strictly more powerful than
    combining two single-arm intervals in quadrature — and unlike the quadrature
    combination it is valid, because the two arm estimates are not independent.

    ``separated`` is the decision predicate: the CI excludes zero.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired test needs aligned arms: {a.shape} vs {b.shape}")
    if len(eid) != a.size:
        raise ValueError(f"eid/arm length mismatch: {len(eid)} vs {a.size}")
    uniq, idx_by_ep = episode_index(eid)
    red = resolve_reducer(reduce)
    point = float(red(a) - red(b))
    d = np.array([float(red(a[sel]) - red(b[sel]))
                  for sel in _draws(uniq, idx_by_ep, n_boot, seed)])
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    lo, hi = float(lo), float(hi)
    # ⛔ separated is decided on the UNROUNDED bounds — that is the correct test
    # and it is NOT touched here. Only the RENDERING adapts, so the printed
    # interval can never contradict the printed verdict. See _render_bounds.
    separated = bool(lo > 0 or hi < 0)
    dp, degenerate = _render_bounds(lo, hi, separated)
    out = {"delta": round(point, dp),
           "lo": round(lo, dp),
           "hi": round(hi, dp),
           "ci95": round((hi - lo) / 2.0, dp),
           "p_delta_gt0": round(float((d > 0).mean()), 4),
           "separated": separated,
           "reducer": _reducer_name(reduce),
           "n_windows": int(a.size),
           "n_episodes": int(len(uniq)),
           "n_boot": int(n_boot),
           "estimator": "paired_" + _NEW_ESTIMATOR}
    if dp != DISPLAY_DP:
        # the record says WHY it is not at the program's 4 dp, so a reader never
        # has to wonder whether the extra digits mean extra precision.
        out["display_dp"] = dp
        out["display_note"] = (
            f"rendered at {dp} dp, not the usual {DISPLAY_DP}: at {DISPLAY_DP} dp "
            f"the printed interval would have contradicted separated="
            f"{separated}. The extra digits are a RENDERING fix, not extra "
            f"precision — the estimator and the separation test are unchanged.")
    if degenerate:
        out["degenerate"] = True
        out["degenerate_note"] = (
            f"⛔ |bound| < 1e-{MAX_DISPLAY_DP}: this interval is at float64 "
            f"resolution, so separated={separated} is arithmetic, not evidence. "
            f"The arms are effectively IDENTICAL on these windows. Do not quote "
            f"this as a result and do not let an automated gate act on it.")
    return out


def bootstrap_metrics(components, eid, n_boot=DEFAULT_N_BOOT, seed=0,
                      alpha=0.05) -> dict:
    """Episode-cluster bootstrap over a whole metric suite in ONE resampling.

    ``components`` maps ``metric_name -> (per_window_values[N], reducer_name)``.
    Every metric is recomputed on the SAME resampled episode draw, so the
    reported intervals are mutually consistent (ade@1s and ade@2s move together
    exactly as they do in reality).
    """
    names = list(components)
    if not names:
        return {}
    vals = {k: np.asarray(components[k][0], dtype=np.float64) for k in names}
    reds = {k: resolve_reducer(components[k][1]) for k in names}
    n = vals[names[0]].size
    for k in names:
        if vals[k].size != n:
            raise ValueError(f"component {k!r} has {vals[k].size} windows, expected {n}")
    uniq, idx_by_ep = episode_index(eid)
    boots = {k: np.empty(n_boot, dtype=np.float64) for k in names}
    for i, sel in enumerate(_draws(uniq, idx_by_ep, n_boot, seed)):
        for k in names:
            boots[k][i] = reds[k](vals[k][sel])
    out = {}
    for k in names:
        lo, hi = np.percentile(boots[k], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[k] = {"mean": round(reds[k](vals[k]), 4),
                  "lo": round(float(lo), 4),
                  "hi": round(float(hi), 4),
                  "ci95": round(float((hi - lo) / 2.0), 4),
                  "se": round(float(boots[k].std(ddof=1)), 4),
                  "reducer": _reducer_name(components[k][1]),
                  "n_windows": int(n),
                  "n_episodes": int(len(uniq)),
                  "n_boot": int(n_boot),
                  "estimator": _NEW_ESTIMATOR}
    return out
