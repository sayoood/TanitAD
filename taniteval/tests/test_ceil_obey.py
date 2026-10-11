"""Tests for ``taniteval.ceil_obey`` (SPEC_REFCV8_DRAFT section 8.3 X3 CEIL-OBEY and the
EXEMPT-ROW rate).

Every expected value below is a LITERAL derived by hand from a speed profile whose
waypoints are built here by cumulative sum -- never an expression over the code under
test. Speeds are dyadic (10.25, 10.5, 10.75 ...) and the intervals are 0.5 s / 1.0 s,
so every position, every speed and every comparison against the 0.5 m/s margin is
exact in binary floating point.

The deliberate-regression arms (``test_mutation_*``) re-introduce, one at a time, a
defect this instrument exists to avoid and require the SAME literal check to go RED:
  * a TYPED dt (0.1 s) instead of the derived grid     -> every speed 5x too high
  * a dropped ego-origin interval                      -> a fast first segment vanishes
  * ``>=`` instead of ``>`` at the margin              -> the exactly-0.5 window flips
  * exempt rows never counted                          -> the structural floor vanishes
  * the denominator widened to ALL windows             -> an empty-ceiling window reads as obedient

pytest is not installed on every pod; runs standalone too:
  python taniteval/tests/test_ceil_obey.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)              # <repo>/taniteval
if _PKG_PARENT not in sys.path:
    sys.path.append(_PKG_PARENT)
_STACK = os.path.join(os.path.dirname(_PKG_PARENT), "stack")
if os.path.isdir(_STACK) and _STACK not in sys.path:
    sys.path.append(_STACK)

from taniteval import ceil_obey as co   # noqa: E402

H8 = (5, 10, 15, 20, 30, 40, 50, 60)          # the ladder's HORIZONS (ticks of 0.1 s)
DT8 = (0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0)  # hand-written, NOT read from the module
INF = float("inf")
NAN = float("nan")


def _plan(speeds, heading=(1.0, 0.0), dts=DT8):
    """Waypoints of a straight plan with the given per-interval speeds (cumulative sum)."""
    d = np.cumsum(np.asarray(speeds, dtype=np.float64) * np.asarray(dts))
    return np.stack([d * heading[0], d * heading[1]], axis=-1)


def _const(v, n=3):
    return np.stack([_plan([v] * 8) for _ in range(n)])


# window table (see the module docstring of the literals) ---------------------
#            emitted max   fed ceiling   fan candidate max speeds
#   w0          10.0          10.0         5, 10, 12       compliant exists; excess 0
#   w1          10.5          10.0         9.5, 10.5, 12   compliant exists; excess EXACTLY 0.5
#   w2          10.75         10.0         10.25, 11, 12   nothing compliant -> exempt; excess 0.75
#   w3          20.0          inf          20, 21, 22      no finite ceiling
#   w4          12.0          nan          12, 13, 14      no finite ceiling
#   w5           9.0           8.0         9, 10, 11       nothing compliant -> exempt; excess 1.0
EMITTED_V = [10.0, 10.5, 10.75, 20.0, 12.0, 9.0]
#: a scalar is a constant-speed profile; a list is a per-interval speed profile. w5's emitted plan
#: and w2's best candidate reach their maximum on the FIRST interval only (so a statistic that drops
#: the ego-origin interval reads them far too low).
EMITTED_P = [10.0, 10.5, 10.75, 20.0, 12.0, [9.0, 5, 5, 5, 5, 5, 5, 5]]
LIM = [10.0, 10.0, 10.0, INF, NAN, 8.0]
FAN_V = [[5.0, 10.0, 12.0], [9.5, 10.5, 12.0], [10.25, 11.0, 12.0],
         [20.0, 21.0, 22.0], [12.0, 13.0, 14.0], [9.0, 10.0, 11.0]]
FAN_P = [[5.0, 10.0, 12.0], [9.5, 10.5, 12.0], [[10.25, 3, 3, 3, 3, 3, 3, 3], 11.0, 12.0],
         [20.0, 21.0, 22.0], [12.0, 13.0, 14.0], [9.0, 10.0, 11.0]]


def _prof(x):
    return [x] * 8 if np.isscalar(x) else list(x)


def _tables():
    emitted = np.stack([_plan(_prof(v)) for v in EMITTED_P])                 # [6, 8, 2]
    fan = np.stack([np.stack([_plan(_prof(v)) for v in row]) for row in FAN_P])   # [6, 3, 8, 2]
    return emitted, np.asarray(LIM), fan


def _check_literals():
    """The registered literals. Called on the real module (must pass) and under every
    mutation (must raise AssertionError)."""
    emitted, lim, fan = _tables()
    r = co.ceil_obey(emitted, lim, horizons=H8, fan=fan)
    assert r["n_windows"] == 6, r
    assert r["n_finite_ceiling"] == 4 and r["n_no_finite_ceiling"] == 2, r
    # SPEC form: excess > 0.5.   w2 (0.75) and w5 (1.0) violate; w1 is EXACTLY 0.5 -> not.
    assert r["n_violate"] == 2, r
    assert r["ceil_obey"] == 0.5 and r["obey_rate"] == 0.5, r
    assert abs(r["share_of_all_windows"] - 2 / 6) < 1e-12, r
    # the ladder's strict form: excess > 1e-6.  w1, w2, w5.
    assert r["strict"]["n_violate"] == 3 and r["strict"]["share"] == 0.75, r
    # exempt rows (filter tol 0): w2 (min fan 10.25 > 10) and w5 (min 9 > 8).
    assert r["exempt"]["n"] == 2 and r["exempt"]["rate"] == 0.5, r
    assert r["exempt"]["n_violate_and_exempt"] == 2, r
    assert r["exempt"]["n_violate_and_not_exempt"] == 0, r
    # w1 is a STRICT violator with a compliant candidate present: the filter did not reach it.
    assert r["exempt"]["n_strict_violate_and_not_exempt"] == 1, r
    assert r["bar"] == {"le": 0.01, "bar_met": False}, r
    return r


# --------------------------------------------------------------------------- #
# the statistic                                                                #
# --------------------------------------------------------------------------- #
def test_step_durations_are_derived_from_the_grid():
    assert co.step_durations_s(H8, 0.1) == DT8
    assert co.step_durations_s((5, 10, 15, 20), 0.1) == (0.5, 0.5, 0.5, 0.5)
    assert co.step_durations_s((1, 2, 4), 0.25) == (0.25, 0.25, 0.5)
    for bad in ((), (0, 5), (5, 5), (10, 5)):
        try:
            co.step_durations_s(bad)
        except ValueError:
            continue
        raise AssertionError(f"horizons {bad} must be refused")


def test_planned_max_speed_literals():
    # constant 10 m/s
    assert co.planned_max_speed_np(_plan([10.0] * 8), H8) == 10.0
    # a 13 m/s burst on the sixth interval only
    assert co.planned_max_speed_np(_plan([10, 10, 10, 10, 10, 13, 10, 10]), H8) == 13.0
    # the first interval is measured from the ego ORIGIN: 12 m in 0.5 s = 24 m/s, then crawl
    fast_start = _plan([24.0, 1, 1, 1, 1, 1, 1, 1])
    assert fast_start[0, 0] == 12.0
    assert co.planned_max_speed_np(fast_start, H8) == 24.0
    # speed is a NORM: a 3-4-5 rotated plan at 10 m/s reads 10 (an x-difference would read 6)
    assert co.planned_max_speed_np(_plan([10.0] * 8, heading=(0.6, 0.8)), H8) == 10.0
    # leading batch axes are preserved
    out = co.planned_max_speed_np(np.stack([_plan([10.0] * 8), _plan([13.0] * 8)]), H8)
    assert out.shape == (2,) and out[0] == 10.0 and out[1] == 13.0
    # a mismatched grid is refused, not silently re-paired
    try:
        co.planned_max_speed_np(_plan([10.0] * 8), (5, 10, 15, 20))
    except ValueError:
        pass
    else:
        raise AssertionError("8 waypoints against 4 horizons must be refused")


# --------------------------------------------------------------------------- #
# the block                                                                    #
# --------------------------------------------------------------------------- #
def test_ceil_obey_literals():
    r = _check_literals()
    assert r["metric"] == "CEIL-OBEY" and r["margin_ms"] == 0.5 and r["filter_tol_ms"] == 0.0
    assert r["estimator"].startswith("none")           # no eid -> no interval claimed
    assert "ci" not in r


def test_per_window_arrays_literals():
    emitted, lim, fan = _tables()
    w = co.ceil_obey_windows(emitted, lim, horizons=H8, fan=fan)
    assert w["finite"].tolist() == [True, True, True, False, False, True]
    assert w["vmax"].tolist() == EMITTED_V
    assert w["excess"][[0, 1, 2, 5]].tolist() == [0.0, 0.5, 0.75, 1.0]
    assert np.isnan(w["excess"][[3, 4]]).all()
    assert w["violate"].tolist() == [False, False, True, False, False, True]
    assert w["strict_violate"].tolist() == [False, True, True, False, False, True]
    assert w["exempt"].tolist() == [False, False, True, False, False, True]
    assert w["fan_min_vmax"].tolist() == [5.0, 9.5, 10.25, 20.0, 12.0, 9.0]


def test_filter_tolerance_decides_who_is_exempt_and_margin_decides_who_violates():
    emitted, lim, fan = _tables()
    # filter_tol 0.5: w2's best candidate (10.25) now complies with 10.0 + 0.5 -> not exempt;
    # w5's best (9.0) still exceeds 8.0 + 0.5 -> exempt.  The REPORTED violation count is unmoved.
    r = co.ceil_obey(emitted, lim, horizons=H8, fan=fan, filter_tol_ms=0.5)
    assert r["exempt"]["n"] == 1 and r["exempt"]["rate"] == 0.25, r
    assert r["exempt"]["n_violate_and_exempt"] == 1 and r["exempt"]["n_violate_and_not_exempt"] == 1, r
    assert r["n_violate"] == 2
    # margin 0.0: the SPEC margin switched off reads the strict-like count
    r0 = co.ceil_obey(emitted, lim, horizons=H8, fan=fan, margin_ms=0.0)
    assert r0["n_violate"] == 3 and r0["ceil_obey"] == 0.75, r0
    # margin 1.0: only excess > 1.0 -> none (w5 is exactly 1.0)
    r1 = co.ceil_obey(emitted, lim, horizons=H8, fan=fan, margin_ms=1.0)
    assert r1["n_violate"] == 0 and r1["ceil_obey"] == 0.0 and r1["bar"]["bar_met"] is True, r1


def test_fan_input_forms_agree():
    emitted, lim, fan = _tables()
    a = co.ceil_obey_windows(emitted, lim, horizons=H8, fan=fan)["exempt"]
    cand_v = np.asarray(FAN_V)
    b = co.ceil_obey_windows(emitted, lim, horizons=H8, fan_vmax=cand_v)["exempt"]
    c = co.ceil_obey_windows(emitted, lim, horizons=H8, fan_min_vmax=cand_v.min(1))["exempt"]
    d = co.ceil_obey_windows(emitted_vmax=np.asarray(EMITTED_V), v_lim=lim, horizons=H8,
                             fan_min_vmax=cand_v.min(1))["exempt"]
    want = [False, False, True, False, False, True]
    assert a.tolist() == b.tolist() == c.tolist() == d.tolist() == want


def test_fan_on_a_different_grid_than_the_emitted_plan():
    # fan on the 4-slot anchor grid (0.5 s each): 10.25 m/s candidates; emitted on the 8-slot grid.
    lim = np.asarray([10.0])
    emitted = _plan([10.0] * 8)[None]
    fan4 = _plan([10.25] * 4, dts=(0.5,) * 4)[None, None]               # [1, 1, 4, 2]
    w = co.ceil_obey_windows(emitted, lim, horizons=H8, fan=fan4, fan_horizons=(5, 10, 15, 20))
    assert w["fan_min_vmax"].tolist() == [10.25] and w["exempt"].tolist() == [True]
    # without fan_horizons the 4-slot fan is mis-paired with the 8-slot grid: REFUSED
    try:
        co.ceil_obey_windows(emitted, lim, horizons=H8, fan=fan4)
    except ValueError:
        pass
    else:
        raise AssertionError("a fan on another grid must be refused unless fan_horizons is given")


def test_fan_valid_restricts_who_may_comply():
    emitted, lim, fan = _tables()
    valid = np.ones((6, 3), dtype=bool)
    valid[0] = [False, False, True]          # w0: only the 12 m/s candidate is eligible -> exempt
    valid[1] = [False, True, True]           # w1: the 9.5 candidate is ineligible -> min 10.5 > 10 -> exempt
    w = co.ceil_obey_windows(emitted, lim, horizons=H8, fan=fan, fan_valid=valid)
    assert w["exempt"].tolist() == [True, True, True, False, False, True]
    # every candidate ineligible: nothing can comply -> exempt
    none_ok = np.zeros((6, 3), dtype=bool)
    assert co.ceil_obey_windows(emitted, lim, horizons=H8, fan=fan,
                                fan_valid=none_ok)["exempt"].tolist() == [True, True, True, False, False, True]


def test_no_fan_means_exempt_is_unavailable_not_zero():
    emitted, lim, _ = _tables()
    r = co.ceil_obey(emitted, lim, horizons=H8)
    assert r["exempt"]["status"] == "UNAVAILABLE" and "rate" not in r["exempt"], r
    assert r["ceil_obey"] == 0.5                       # the share itself is still defined


def test_no_finite_ceiling_is_undefined_not_zero():
    emitted = np.stack([_plan([30.0] * 8)] * 2)
    r = co.ceil_obey(emitted, [INF, NAN], horizons=H8)
    assert r["ceil_obey"] is None and r["obey_rate"] is None, r
    assert r["status"] == "UNDEFINED" and "denominator" in r["reason"], r
    assert r["bar"]["bar_met"] is None, r


def test_refusals():
    emitted, lim, fan = _tables()
    cases = {
        "negative ceiling": lambda: co.ceil_obey_windows(emitted, [-1.0] * 6, horizons=H8),
        "both emitted forms": lambda: co.ceil_obey_windows(emitted, lim, horizons=H8,
                                                           emitted_vmax=np.zeros(6)),
        "no emitted form": lambda: co.ceil_obey_windows(None, lim, horizons=H8),
        "two fan forms": lambda: co.ceil_obey_windows(emitted, lim, horizons=H8, fan=fan,
                                                      fan_vmax=np.zeros((6, 3))),
        "fan_valid with pre-reduced fan": lambda: co.ceil_obey_windows(
            emitted, lim, horizons=H8, fan_min_vmax=np.zeros(6), fan_valid=np.ones((6, 3), bool)),
        "window count mismatch": lambda: co.ceil_obey_windows(emitted[:5], lim, horizons=H8),
        "fan window mismatch": lambda: co.ceil_obey_windows(emitted, lim, horizons=H8,
                                                            fan_vmax=np.zeros((5, 3))),
    }
    for name, fn in cases.items():
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError(f"{name} must raise ValueError")
    # a NaN emitted plan on a FINITE-ceiling window raises ...
    bad = emitted.copy()
    bad[0, 3, 0] = np.nan
    try:
        co.ceil_obey_windows(bad, lim, horizons=H8)
    except ValueError as e:
        assert "non-finite emitted plan" in str(e)
    else:
        raise AssertionError("a NaN plan on a finite-ceiling window must raise")
    # ... but a NaN plan on a window with NO finite ceiling is outside every denominator
    ok = emitted.copy()
    ok[3, 2, 1] = np.nan                                  # w3 has lim = inf
    assert co.ceil_obey_windows(ok, lim, horizons=H8)["finite"].tolist() == \
        [True, True, True, False, False, True]


def test_interval_with_eid_carries_its_estimator():
    emitted, lim, fan = _tables()
    eid = ["a", "a", "b", "b", "c", "c"]
    r = co.ceil_obey(emitted, lim, horizons=H8, fan=fan, eid=eid, n_boot=200, seed=0)
    assert r["ceil_obey"] == 0.5                          # point estimate is the full-set share
    assert "episode_cluster" in r["estimator"]
    # finite windows sit in episodes a (w0, w1), b (w2) and c (w5)
    assert r["ci"]["n_episodes"] == 3 and r["ci"]["n_boot"] == 200, r["ci"]
    assert 0.0 <= r["ci"]["lo"] <= 0.5 <= r["ci"]["hi"] <= 1.0, r["ci"]
    assert r["exempt"]["ci"]["n_episodes"] == 3, r["exempt"]


def test_matches_the_models_own_filter():
    """The numpy statistic must equal ``refcv6_selection.planned_max_speed`` and the exempt
    count must equal the filter's own ``speed_rows_empty`` telemetry (independent code)."""
    try:
        import torch
        from tanitad.refs import refcv6_selection as v6sel
    except Exception as e:                                  # pragma: no cover - env without the stack
        msg = f"test_matches_the_models_own_filter needs torch + tanitad: {type(e).__name__}: {e}"
        try:
            import pytest
        except ImportError:
            print("SKIP", msg)                              # standalone runner: say so, loudly
            return
        pytest.skip(msg)                                     # a COUNTED skip, never a silent pass
    rng = np.random.default_rng(0)
    W, N = 64, 17
    speeds = rng.uniform(2.0, 30.0, (W, N, 8))
    fan = np.cumsum(speeds * np.asarray(DT8), axis=-1)
    fan = np.stack([fan, rng.normal(0, 3.0, fan.shape)], axis=-1)       # a lateral wiggle too
    lim = rng.uniform(8.0, 28.0, W)
    t = torch.as_tensor(fan, dtype=torch.float32)
    theirs = v6sel.planned_max_speed(t, horizons=H8, tick_s=0.1).double().numpy()
    ours = co.planned_max_speed_np(fan.astype(np.float32).astype(np.float64), H8, 0.1)
    assert np.allclose(ours, theirs, rtol=1e-5, atol=1e-4), np.abs(ours - theirs).max()
    keep, tele = v6sel.SpeedCeilingFilter(horizons=H8, tick_s=0.1)(
        t, torch.as_tensor(lim, dtype=torch.float32))
    w = co.ceil_obey_windows(fan[:, 0], lim, horizons=H8, fan=fan.astype(np.float32).astype(np.float64))
    # float32 vs float64 can disagree only on a candidate within 1e-4 of the ceiling; none at this seed
    assert int(w["exempt"].sum()) == int(tele["speed_rows_empty"]), (int(w["exempt"].sum()), tele)
    # and the filter's per-row survivors: a non-exempt row keeps exactly its compliant candidates
    comp = (theirs <= lim[:, None]).any(axis=1)
    assert (~w["exempt"]).tolist() == comp.tolist()


# --------------------------------------------------------------------------- #
# deliberate regressions: each re-introduced defect must turn the literals RED   #
# --------------------------------------------------------------------------- #
def _expect_red(patch):
    """Apply ``patch(co) -> restore`` and require ``_check_literals`` to FAIL."""
    restore = patch(co)
    try:
        try:
            _check_literals()
        except AssertionError:
            return
        raise AssertionError("the deliberate regression stayed GREEN -- the literals cannot see it")
    finally:
        restore()


def test_mutation_typed_dt_goes_red():
    def patch(m):
        orig = m.step_durations_s
        m.step_durations_s = lambda horizons, tick_s=0.1: tuple(float(tick_s) for _ in horizons)
        return lambda: setattr(m, "step_durations_s", orig)
    _expect_red(patch)


def test_mutation_dropped_origin_interval_goes_red():
    def patch(m):
        orig = m.planned_max_speed_np

        def no_origin(plans, horizons, tick_s=0.1):
            p = np.asarray(plans, dtype=np.float64)
            dts = np.asarray(orig.__globals__["step_durations_s"](horizons, tick_s))
            step = np.linalg.norm(np.diff(p, axis=-2), axis=-1)            # no origin row
            return (step / dts[1:]).max(axis=-1)
        m.planned_max_speed_np = no_origin
        return lambda: setattr(m, "planned_max_speed_np", orig)
    _expect_red(patch)


def test_mutation_ge_at_the_margin_goes_red():
    def patch(m):
        orig = m.ceil_obey_windows

        def ge(*a, **k):
            w = orig(*a, **k)
            w["violate"] = w["finite"] & (w["excess"] >= float(k.get("margin_ms", 0.5)))
            return w
        m.ceil_obey_windows = ge
        return lambda: setattr(m, "ceil_obey_windows", orig)
    _expect_red(patch)


def test_mutation_exempt_rows_never_counted_goes_red():
    def patch(m):
        orig = m.ceil_obey_windows

        def blind(*a, **k):
            w = orig(*a, **k)
            if w["exempt"] is not None:
                w["exempt"] = np.zeros_like(w["exempt"])
            return w
        m.ceil_obey_windows = blind
        return lambda: setattr(m, "ceil_obey_windows", orig)
    _expect_red(patch)


def test_mutation_denominator_over_all_windows_goes_red():
    def patch(m):
        orig = m.ceil_obey_windows

        def wide(*a, **k):
            w = orig(*a, **k)
            w["finite"] = np.ones_like(w["finite"])
            return w
        m.ceil_obey_windows = wide
        return lambda: setattr(m, "ceil_obey_windows", orig)
    _expect_red(patch)


if __name__ == "__main__":                                  # pragma: no cover
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for k, v in fns:
        try:
            v()
            print("PASS", k)
        except Exception as e:
            bad += 1
            print("FAIL", k, type(e).__name__, e)
    sys.exit(1 if bad else 0)
