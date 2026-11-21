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
    "align_windows",
    "paired_stat_cluster_bootstrap",
    "stat_cluster_bootstrap",
    "e1_floor_read",
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
                              n_boot=DEFAULT_N_BOOT, seed=0, alpha=0.05,
                              dp=DISPLAY_DP) -> dict:
    """Percentile CI on a per-window metric, resampling EPISODES with replacement.

    ``per_window`` [N] are the per-window components (displacements, 0/1 miss
    flags, squared errors for ``reduce="rms"``); ``eid`` [N] their episode ids.
    ``reduce`` is a :data:`REDUCERS` name or any callable ``values -> float``.

    The point estimate is the **full-set** value — the bootstrap supplies the
    interval, it does not move the mean. Returns a dict carrying its own
    provenance (``estimator``, ``n_episodes``, ``n_boot``) so a number can never
    be quoted without the construction that produced it.

    ⚠️ ``dp`` (2026-08-23, ADDITIVE — the default is unchanged, so no published
    number moves) is the publication rounding. It exists because the programme's
    4 dp is a **display** convention chosen for metres and m/s, and it silently
    destroys a component whose natural magnitude is smaller: ``curvature_bias_1pm``
    runs at ~1e-4 1/m, so at 4 dp its interval renders ``[0.0, 0.0]`` — a
    degenerate-looking record around a real effect. Same defect class as
    :func:`_render_bounds`, one level up: **the display lied, so the display is
    what is parameterised.** The estimator is untouched.
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
    dp = int(dp)
    out = {"mean": round(point, dp),
           "lo": round(float(lo), dp),
           "hi": round(float(hi), dp),
           # symmetric half-width, for drop-in comparison with the old ± ci95
           "ci95": round(float((hi - lo) / 2.0), dp),
           "se": round(float(boots.std(ddof=1)), dp),
           "reducer": _reducer_name(reduce),
           "n_windows": int(v.size),
           "n_episodes": int(len(uniq)),
           "n_boot": int(n_boot),
           "estimator": _NEW_ESTIMATOR}
    if dp != DISPLAY_DP:
        out["display_dp"] = dp
    return out


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


def align_windows(keys_a, items_a, keys_b, items_b, eid_of=None):
    """Pair two arms' per-window items on their window KEYS (never on position).

    Returns ``(aligned_a, aligned_b, eid, report)`` over the keys both arms hold, in sorted key order. A duplicate key in
    either arm is REFUSED (a silent first-wins would pair the wrong windows). ``report`` carries the counts dropped from
    each side so a mismatch of the two arms' window sets is on the record, not silent. ``eid_of(key) -> episode id``
    (default: ``key[0]``, the clip id of a ``(clip, window_start)`` key)."""
    keys_a, keys_b = list(keys_a), list(keys_b)
    if len(keys_a) != len(items_a) or len(keys_b) != len(items_b):
        raise ValueError("keys / items length mismatch")
    for name, ks in (("a", keys_a), ("b", keys_b)):
        if len(set(ks)) != len(ks):
            raise ValueError("arm %s carries duplicate window keys: pairing on them would be arbitrary" % name)
    ia, ib = dict(zip(keys_a, items_a)), dict(zip(keys_b, items_b))
    common = sorted(set(ia) & set(ib), key=lambda k: (str(k),))
    if not common:
        raise ValueError("the two arms share no window key: nothing to pair")
    eo = eid_of if eid_of is not None else (lambda k: k[0] if isinstance(k, (tuple, list)) else k)
    return ([ia[k] for k in common], [ib[k] for k in common], [str(eo(k)) for k in common],
            {"n_common": len(common), "n_only_a": len(ia) - len(common), "n_only_b": len(ib) - len(common)})


def paired_stat_cluster_bootstrap(items_a, items_b, eid, stat, n_boot=DEFAULT_N_BOOT, seed=0, alpha=0.05) -> dict:
    """Paired episode-cluster bootstrap of ``stat(arm A windows) - stat(arm B windows)`` for a statistic that is NOT a
    mean of per-window scalars (AP over pooled detections, a ratio of pooled counts, ...).

    ``items_a`` / ``items_b`` [N] are the two arms' per-window objects, ALIGNED window-for-window on the same windows (see
    :func:`align_windows`); ``stat`` maps a list of window objects to a float and is evaluated on the SAME resampled
    episodes for both arms each draw, so shared window difficulty cancels. Point estimates are the FULL-set values.
    ``separated`` = the interval excludes 0 (necessary, not sufficient, for a LEVER claim: this answers *another draw of
    EPISODES*, never *another training run* -- read the E1 floor, :func:`e1_floor_read`)."""
    if len(items_a) != len(items_b) or len(eid) != len(items_a):
        raise ValueError("paired_stat_cluster_bootstrap needs aligned arms: %d / %d / eid %d"
                         % (len(items_a), len(items_b), len(eid)))
    uniq, idx_by_ep = episode_index(eid)
    as_d = lambda v: v if isinstance(v, dict) else {"_": v}       # noqa: E731  (a stat may return one float or a dict of them)
    scalar = not isinstance(stat(list(items_a)), dict)
    pa, pb = as_d(stat(list(items_a))), as_d(stat(list(items_b)))
    keys = list(pa)
    if set(pb) != set(keys):
        raise ValueError("the statistic returned different keys for the two arms")
    d = {k: np.empty(int(n_boot), dtype=np.float64) for k in keys}
    for i, sel in enumerate(_draws(uniq, idx_by_ep, n_boot, seed)):
        ra, rb = as_d(stat([items_a[j] for j in sel])), as_d(stat([items_b[j] for j in sel]))      # the SAME draw for both arms
        for k in keys:
            d[k][i] = float(ra[k]) - float(rb[k])
    out = {}
    for k in keys:
        fin = d[k][np.isfinite(d[k])]
        if fin.size == 0:        # undefined on these windows (e.g. no positive in a band): reported, never read as 0
            out[k] = {"a": float(pa[k]), "b": float(pb[k]), "delta": float("nan"), "separated": False, "undefined": True,
                      "n_windows": int(len(items_a)), "n_episodes": int(len(uniq)), "n_boot": int(n_boot)}
            continue
        lo, hi = (float(x) for x in np.percentile(fin, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
        separated = bool(lo > 0 or hi < 0)
        dp, degenerate = _render_bounds(lo, hi, separated)
        r = {"a": round(float(pa[k]), dp), "b": round(float(pb[k]), dp), "delta": round(float(pa[k]) - float(pb[k]), dp),
             "lo": round(lo, dp), "hi": round(hi, dp), "p_delta_gt0": round(float((fin > 0).mean()), 4),
             "separated": separated, "n_windows": int(len(items_a)), "n_episodes": int(len(uniq)), "n_boot": int(n_boot),
             "n_nonfinite_draws": int(d[k].size - fin.size), "estimator": "paired_" + _NEW_ESTIMATOR + "_stat"}
        if degenerate:
            r["degenerate"] = True
        out[k] = r
    return out["_"] if scalar else out


def stat_cluster_bootstrap(items, eid, stat, n_boot=DEFAULT_N_BOOT, seed=0, alpha=0.05) -> dict:
    """Single-arm sibling of :func:`paired_stat_cluster_bootstrap`: percentile CI of ``stat(items)`` (a float or a dict of
    floats) resampling EPISODES; the point estimate is the full-set value."""
    uniq, idx_by_ep = episode_index(eid)
    p = stat(list(items))
    scalar = not isinstance(p, dict)
    p = {"_": p} if scalar else p
    d = {k: np.empty(int(n_boot)) for k in p}
    for i, sel in enumerate(_draws(uniq, idx_by_ep, n_boot, seed)):
        r = stat([items[j] for j in sel])
        r = {"_": r} if scalar else r
        for k in p:
            d[k][i] = float(r[k])
    out = {}
    for k in p:
        fin = d[k][np.isfinite(d[k])]
        rec = {"point": float(p[k]), "n_episodes": int(len(uniq)), "n_boot": int(n_boot), "estimator": _NEW_ESTIMATOR + "_stat"}
        if fin.size:
            rec["lo"], rec["hi"] = (float(x) for x in np.percentile(fin, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
        else:
            rec["undefined"] = True
        out[k] = rec
    return out["_"] if scalar else out


def e1_floor_read(delta, separated, c0, c1, higher_is_better=True) -> dict:
    """REFCV9_DESIGN 4.3 keep-rule, ONE metric: a lever's effect is read against the rig's own run-to-run floor
    ``|C1 - C0|`` (same flags, training seeds 1 vs 0). Kept on this metric iff the paired interval is separated in the GOOD
    direction AND ``|delta| > 2 * |C1 - C0|``. A floor of exactly 0 is NOT a floor (two runs that agree to the last digit
    on one seed pair): it is reported ``floor_degenerate`` and the keep decision then rests on separation alone."""
    floor = abs(float(c1) - float(c0))
    good = (float(delta) > 0) if higher_is_better else (float(delta) < 0)
    margin = (abs(float(delta)) / floor) if floor > 0 else float("inf")
    return {"delta": float(delta), "floor_abs_c1_minus_c0": floor, "margin_over_floor": margin,
            "floor_degenerate": bool(floor == 0.0), "separated_good_direction": bool(separated and good),
            "separated_bad_direction": bool(separated and not good),
            "exceeds_2x_floor": bool(abs(float(delta)) > 2.0 * floor),
            "keep_on_this_metric": bool(separated and good and abs(float(delta)) > 2.0 * floor)}


def bootstrap_metrics(components, eid, n_boot=DEFAULT_N_BOOT, seed=0,
                      alpha=0.05) -> dict:
    """Episode-cluster bootstrap over a whole metric suite in ONE resampling.

    ``components`` maps ``metric_name -> (per_window_values[N], reducer_name)``
    — or, since 2026-08-23, ``(per_window_values[N], reducer_name, dp)`` so one
    suite can mix magnitudes (metres at 4 dp beside 1/m curvature at 6). The
    2-tuple form is unchanged and still rounds at :data:`DISPLAY_DP`, so no
    existing caller's numbers move.

    Every metric is recomputed on the SAME resampled episode draw, so the
    reported intervals are mutually consistent (ade@1s and ade@2s move together
    exactly as they do in reality).
    """
    names = list(components)
    if not names:
        return {}
    vals = {k: np.asarray(components[k][0], dtype=np.float64) for k in names}
    reds = {k: resolve_reducer(components[k][1]) for k in names}
    dps = {k: int(components[k][2]) if len(components[k]) > 2 else DISPLAY_DP
           for k in names}
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
        dp = dps[k]
        lo, hi = np.percentile(boots[k], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[k] = {"mean": round(reds[k](vals[k]), dp),
                  "lo": round(float(lo), dp),
                  "hi": round(float(hi), dp),
                  "ci95": round(float((hi - lo) / 2.0), dp),
                  "se": round(float(boots[k].std(ddof=1)), dp),
                  "reducer": _reducer_name(components[k][1]),
                  "n_windows": int(n),
                  "n_episodes": int(len(uniq)),
                  "n_boot": int(n_boot),
                  "estimator": _NEW_ESTIMATOR}
        if dp != DISPLAY_DP:
            out[k]["display_dp"] = dp
    return out
