"""CEIL-OBEY -- does the EMITTED plan obey the fed speed ceiling, and on how many
windows was the argmax filter structurally unable to make it?

SPEC_REFCV8_DRAFT section 8.3 (X3) (a)(2), registered text:

    CEIL-OBEY = share of S-ROUTE windows whose EMITTED plan's max speed exceeds
    the fed finite ceiling by > 0.5 m/s.        Bar: CEIL-OBEY <= 0.01.

This module also reports the second quantity the bar cannot be read without:

    EXEMPT-ROW rate = share of (finite-ceiling) windows on which NO candidate in
    the fan complies with the ceiling, so ``SpeedCeilingFilter`` could not act.

WHY BOTH (the filter's own docstring, ``refcv6_selection.SpeedCeilingFilter``):
the ceiling is applied as an ARGMAX FILTER, not a clamp, and "a row with no
survivor keeps its whole fan". Those rows are the only STRUCTURAL reason the
obedience rate can fall short of 1.0. So a CEIL-OBEY miss splits into two causes
that need different fixes, and a single share hides which one it is:

    violate AND exempt      the vocabulary offered nothing compliant -- a fan /
                            anchor-bank defect; no selection change can fix it.
    violate AND NOT exempt  a compliant candidate EXISTED and the emitted plan
                            still exceeded the ceiling -- the filter did not
                            reach the emitted plan (the refcv7 E9 re-rank defect,
                            SPEC_REFCV7 A1/A2), a wiring defect.

STATISTIC. The planned max speed is the filter's own: speed on interval i is
``|p_i - p_{i-1}| / dt_i`` with ``dt_i = (h_i - h_{i-1}) * tick_s`` DERIVED from
the waypoint tick grid ``horizons`` (never typed -- a typed 0.1 s reports every
speed 5x too high and fails an obedient model), the first interval measured from
the ego origin, and the maximum taken over intervals. ``planned_max_speed_np`` is
a numpy re-derivation of ``refcv6_selection.planned_max_speed``; the test
``test_matches_the_models_own_filter`` pins the two against each other on random
fans (skipped, loudly, when ``tanitad`` / ``torch`` are not importable).

TWO GRIDS. The emitted plan (``out["traj"]``) and the filter's candidates
(``anchor_horizons``) are separate objects. They coincide when the decoder's
horizons equal the emitted horizons (the ladder asserts that), but this module
takes ``horizons`` for the emitted plan and ``fan_horizons`` for the fan so a
mismatch is a parameter and not a silent mis-pairing.

TWO TOLERANCES, NEVER MIXED. ``margin_ms`` (0.5 m/s, SPEC) is what the
REPORTED violation uses. ``filter_tol_ms`` (0.0, ``SpeedCeilingFilter.tol_ms``
default) is what decides whether a candidate "complies", i.e. who is exempt.
The ladder's frozen ``ceil_violation`` is the STRICT form (> 1e-6, margin 0); it
is reported beside the SPEC form as ``strict_*`` so the two cannot be confused.

ESTIMATOR. With ``eid`` the shares carry ``taniteval.ci`` episode-cluster
bootstrap intervals. That interval answers "would another draw of EPISODES say
this?" -- it is blind to training variance (H-ESTIM-SEED-1) and, for a sampling
planner, to inference variance. The ladder keeps its own paired draws: use
``ceil_obey_windows`` for the per-window arrays and bootstrap them there.

FAIL LOUD. No window with a finite ceiling => the share is ``None`` with a
reason, never 0.0 (an empty denominator is not obedience). A non-finite emitted
waypoint on a finite-ceiling window, a negative ceiling, or a shape mismatch
raises.

Zero GPU, numpy only.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "SPEC_MARGIN_MS", "FILTER_TOL_MS", "BAR_CEIL_OBEY", "DEFAULT_TICK_S",
    "step_durations_s", "planned_max_speed_np", "ceil_obey_windows", "ceil_obey",
]

#: SPEC_REFCV8_DRAFT section 8.3 (a)(2): "exceeds the fed finite ceiling by > 0.5 m/s".
SPEC_MARGIN_MS: float = 0.5
#: ``refcv6_selection.SpeedCeilingFilter.__init__`` default ``tol_ms``.
FILTER_TOL_MS: float = 0.0
#: SPEC_REFCV8_DRAFT section 8.3 (c)(2): "CEIL-OBEY <= 0.01".
BAR_CEIL_OBEY: float = 0.01
#: the corpus tick (``RefCConfig`` ``anchor_dt``): waypoint horizons are in ticks.
DEFAULT_TICK_S: float = 0.1
#: the ladder's frozen strict form (``ladder_score.window_metrics``: ``> vl + 1e-6``).
_STRICT_EPS: float = 1e-6


# --------------------------------------------------------------------------- #
# the statistic                                                                #
# --------------------------------------------------------------------------- #
def step_durations_s(horizons: Sequence[int], tick_s: float = DEFAULT_TICK_S) -> tuple:
    """Interval durations in seconds, derived from the waypoint tick grid.

    ``horizons`` are strictly increasing positive TICKS; interval ``i`` spans
    ``(h_i - h_{i-1}) * tick_s`` with ``h_{-1} = 0`` (the ego origin)."""
    h = [int(x) for x in horizons]
    if not h:
        raise ValueError("horizons must be non-empty (in ticks)")
    if h[0] <= 0 or any(b <= a for a, b in zip(h, h[1:])):
        raise ValueError(f"horizons must be strictly increasing positive ticks, got {h}")
    prev, out = 0, []
    for k in h:
        out.append((k - prev) * float(tick_s))
        prev = k
    return tuple(out)


def planned_max_speed_np(plans, horizons: Sequence[int],
                         tick_s: float = DEFAULT_TICK_S) -> np.ndarray:
    """Max speed each plan reaches, m/s. ``plans`` ``[..., S, 2]`` metres, ego frame,
    origin implicit at (0, 0); returns ``[...]``. float64."""
    p = np.asarray(plans, dtype=np.float64)
    if p.ndim < 2 or p.shape[-1] != 2:
        raise ValueError(f"plans must be [..., S, 2], got {p.shape}")
    dts = np.asarray(step_durations_s(horizons, tick_s), dtype=np.float64)
    if p.shape[-2] != dts.size:
        raise ValueError(
            f"{dts.size} horizons but the plan has {p.shape[-2]} waypoints; a mismatched "
            f"grid pairs waypoint i with interval j and reports a speed that belongs to neither")
    origin = np.zeros(p.shape[:-2] + (1, 2), dtype=np.float64)
    seq = np.concatenate([origin, p], axis=-2)
    step = np.linalg.norm(np.diff(seq, axis=-2), axis=-1)            # [..., S]
    return (step / dts).max(axis=-1)


# --------------------------------------------------------------------------- #
# per-window                                                                   #
# --------------------------------------------------------------------------- #
def _fan_min_vmax(fan, fan_vmax, fan_min_vmax, fan_valid, fan_horizons, tick_s, W):
    """-> ``[W]`` min over the eligible candidates of the planned max speed, or None."""
    given = [x is not None for x in (fan, fan_vmax, fan_min_vmax)]
    if sum(given) > 1:
        raise ValueError("pass at most one of fan / fan_vmax / fan_min_vmax")
    if not any(given):
        return None
    if fan_min_vmax is not None:
        if fan_valid is not None:
            raise ValueError("fan_valid needs the per-candidate fan or fan_vmax, "
                             "not the pre-reduced fan_min_vmax")
        m = np.asarray(fan_min_vmax, dtype=np.float64)
        if m.shape != (W,):
            raise ValueError(f"fan_min_vmax must be [{W}], got {m.shape}")
        return m
    if fan is not None:
        vm = planned_max_speed_np(fan, fan_horizons, tick_s)         # [W, N]
    else:
        vm = np.asarray(fan_vmax, dtype=np.float64)
    if vm.ndim != 2 or vm.shape[0] != W:
        raise ValueError(f"per-candidate speeds must be [{W}, N], got {vm.shape}")
    if fan_valid is not None:
        ok = np.asarray(fan_valid, dtype=bool)
        if ok.shape != vm.shape:
            raise ValueError(f"fan_valid {ok.shape} != candidate speeds {vm.shape}")
        vm = np.where(ok, vm, np.inf)            # an ineligible candidate never complies
    return vm.min(axis=1)


def ceil_obey_windows(emitted=None, v_lim=None, *, horizons: Sequence[int],
                      emitted_vmax=None, tick_s: float = DEFAULT_TICK_S,
                      fan=None, fan_vmax=None, fan_min_vmax=None, fan_valid=None,
                      fan_horizons: Sequence[int] | None = None,
                      margin_ms: float = SPEC_MARGIN_MS,
                      filter_tol_ms: float = FILTER_TOL_MS) -> dict:
    """Per-window CEIL-OBEY arrays (all length W).

    ``emitted`` ``[W, S, 2]`` metres ego frame (or ``emitted_vmax`` ``[W]``);
    ``v_lim`` ``[W]`` the FED ceiling in m/s -- ``inf`` or NaN means no finite
    ceiling was fed (the window is outside every denominator). The fan, for the
    exempt-row determination, is ONE of ``fan`` ``[W, N, S, 2]``, ``fan_vmax``
    ``[W, N]`` or ``fan_min_vmax`` ``[W]`` (a hook on ``SpeedCeilingFilter`` can
    reduce to this in-flight, so the ladder never has to dump the whole fan);
    ``fan_valid`` ``[W, N]`` restricts the candidates that may comply (default:
    all, exactly what ``SpeedCeilingFilter`` sees).

    Returns ``finite, vmax, excess, violate, strict_violate, exempt`` (``exempt``
    is ``None`` when no fan was supplied) and ``fan_min_vmax``.
    """
    if v_lim is None:
        raise ValueError("v_lim is required")
    if (emitted is None) == (emitted_vmax is None):
        raise ValueError("pass exactly one of emitted / emitted_vmax")
    lim = np.asarray(v_lim, dtype=np.float64).reshape(-1)
    W = lim.size
    if emitted is not None:
        e = np.asarray(emitted, dtype=np.float64)
        if e.ndim != 3 or e.shape[0] != W:
            raise ValueError(f"emitted must be [{W}, S, 2], got {e.shape}")
        vmax = planned_max_speed_np(e, horizons, tick_s)
    else:
        vmax = np.asarray(emitted_vmax, dtype=np.float64).reshape(-1)
        if vmax.size != W:
            raise ValueError(f"emitted_vmax must be [{W}], got {vmax.shape}")
    finite = np.isfinite(lim)
    if (lim[finite] < 0).any():
        raise ValueError("a negative speed ceiling was fed; the ceiling is a speed in m/s")
    if not np.isfinite(vmax[finite]).all():
        raise ValueError(
            f"{int((~np.isfinite(vmax[finite])).sum())} finite-ceiling window(s) carry a "
            f"non-finite emitted plan; a NaN plan cannot be said to obey -- refusing to score")
    excess = np.where(finite, vmax - np.where(finite, lim, 0.0), np.nan)
    with np.errstate(invalid="ignore"):
        violate = finite & (excess > float(margin_ms))
        strict_violate = finite & (excess > _STRICT_EPS)
    fmin = _fan_min_vmax(fan, fan_vmax, fan_min_vmax, fan_valid,
                         fan_horizons if fan_horizons is not None else horizons, tick_s, W)
    if fmin is None:
        exempt = None
    else:
        with np.errstate(invalid="ignore"):
            # SpeedCeilingFilter: keep = vmax <= lim + tol; empty row <=> every candidate fails.
            exempt = finite & ~(fmin <= np.where(finite, lim, 0.0) + float(filter_tol_ms))
    return {"finite": finite, "vmax": vmax, "excess": excess, "violate": violate,
            "strict_violate": strict_violate, "exempt": exempt, "fan_min_vmax": fmin}


# --------------------------------------------------------------------------- #
# summary                                                                      #
# --------------------------------------------------------------------------- #
def _ci():
    try:
        from . import ci as _c            # type: ignore
        return _c
    except ImportError:
        from taniteval import ci as _c    # type: ignore
        return _c


def _share(flag, finite, eid, n_boot, seed):
    n = int(finite.sum())
    if n == 0:
        return {"share": None, "n": 0, "status": "UNDEFINED",
                "reason": "no window carried a finite ceiling -- an empty denominator is not obedience"}
    blk = {"share": float(flag[finite].mean()), "n": n}
    if eid is not None:
        c = _ci().episode_cluster_bootstrap(flag[finite].astype(np.float64),
                                            list(np.asarray(eid)[finite]),
                                            n_boot=n_boot, seed=seed)
        blk["ci"] = {k: c[k] for k in ("lo", "hi", "n_episodes", "n_boot", "estimator")}
    return blk


def ceil_obey(emitted=None, v_lim=None, *, horizons: Sequence[int], eid=None,
              n_boot: int = 2000, seed: int = 0, **kw) -> dict:
    """The CEIL-OBEY block. ``kw`` is everything ``ceil_obey_windows`` takes.

    ``ceil_obey`` is the SPEC share (violation share, LOWER is better, bar <=
    0.01); ``obey_rate`` is its complement. The bar is evaluated on the point
    estimate as the SPEC states it; ``bar_met`` is None while the share is
    undefined."""
    w = ceil_obey_windows(emitted, v_lim, horizons=horizons, **kw)
    finite, viol, strict, ex = w["finite"], w["violate"], w["strict_violate"], w["exempt"]
    n_all, n_fin = int(finite.size), int(finite.sum())
    margin = float(kw.get("margin_ms", SPEC_MARGIN_MS))
    tol = float(kw.get("filter_tol_ms", FILTER_TOL_MS))
    main = _share(viol, finite, eid, n_boot, seed)
    out: dict = {
        "metric": "CEIL-OBEY", "spec": "SPEC_REFCV8_DRAFT section 8.3 X3 (a)(2)",
        "margin_ms": margin, "filter_tol_ms": tol,
        "horizons_ticks": [int(h) for h in horizons],
        "tick_s": float(kw.get("tick_s", DEFAULT_TICK_S)),
        "n_windows": n_all, "n_finite_ceiling": n_fin,
        "n_no_finite_ceiling": n_all - n_fin,
        "n_violate": int(viol.sum()),
        "ceil_obey": main["share"],
        "obey_rate": (None if main["share"] is None else 1.0 - main["share"]),
        "denominator": "windows with a finite fed ceiling",
        "share_of_all_windows": (float(viol.sum()) / n_all if (n_all and n_fin) else None),
        "strict": {"n_violate": int(strict.sum()),
                   "share": (_share(strict, finite, None, n_boot, seed)["share"]),
                   "rule": f"excess > {_STRICT_EPS:g} m/s (the ladder's frozen ceil_violation)"},
        "bar": {"le": BAR_CEIL_OBEY,
                "bar_met": (None if main["share"] is None else bool(main["share"] <= BAR_CEIL_OBEY))},
        "estimator": ("episode_cluster_bootstrap (taniteval.ci) -- answers another draw of EPISODES "
                      "only; blind to training variance (H-ESTIM-SEED-1) and inference variance"
                      if eid is not None else "none (point estimate; pass eid for an interval)"),
    }
    if "ci" in main:
        out["ci"] = main["ci"]
    if main["share"] is None:
        out["status"] = main["status"]
        out["reason"] = main["reason"]
    if ex is None:
        out["exempt"] = {"status": "UNAVAILABLE",
                         "reason": "no fan supplied (fan / fan_vmax / fan_min_vmax): the exempt-row "
                                   "rate cannot be computed post hoc from the emitted plan alone"}
        return out
    exs = _share(ex, finite, eid, n_boot, seed)
    n_ve, n_vn = int((viol & ex).sum()), int((viol & ~ex & finite).sum())
    out["exempt"] = {
        "n": int(ex.sum()), "rate": exs["share"], "denominator": "windows with a finite fed ceiling",
        "rule": ("no fan candidate has planned max speed <= ceiling + filter_tol_ms, so "
                 "SpeedCeilingFilter keeps the whole fan (refcv6_selection.SpeedCeilingFilter)"),
        "n_violate_and_exempt": n_ve,
        "n_violate_and_not_exempt": n_vn,
        "share_violate_and_exempt": (n_ve / n_fin if n_fin else None),
        "share_violate_and_not_exempt": (n_vn / n_fin if n_fin else None),
        "n_strict_violate_and_not_exempt": int((strict & ~ex & finite).sum()),
        "_reads": ("violate AND NOT exempt = a compliant candidate existed and the emitted plan "
                   "still exceeded the ceiling (the filter did not reach the emitted plan); "
                   "violate AND exempt = the fan offered nothing compliant (vocabulary defect)"),
    }
    if "ci" in exs:
        out["exempt"]["ci"] = exs["ci"]
    return out
