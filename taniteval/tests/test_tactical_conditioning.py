"""R8-4 metric suite (`taniteval.tactical_conditioning`) -- literal known-value tests, mutation tests, independence.

RULES THIS FILE FOLLOWS (CLAUDE.md "a check that shares the defect it checks is green forever")
* Every expected value is a LITERAL (or closed-form mathematics typed here: a circle's chord heading is the mean of its
  end tangents, a constant deceleration stops after v0^2 / 2a); it is never an expression over the code under test.
* The tracks are ANALYTIC and built in this file by closed forms that do not touch the module.
* Every metric has (a) a control that must read a known value and (b) a MUTATION test: a deliberately broken variant,
  monkeypatched in, that the SAME known-value check must reject (`pytest.raises(AssertionError)`), and that passes again
  once the patch is undone.
* The path classifier is held against the route package's `route_metrics` on the banked refcv7 capture (load by path,
  TEST-ONLY -- the module itself imports nothing from it) and must agree EXACTLY on the direction class.

Where the module under test comes from: the test puts the sibling `taniteval/` directory (the one holding
`tactical_conditioning.py`) first on `taniteval.__path__` and ASSERTS the loaded `__file__` is that file, so a stale copy
elsewhere on PYTHONPATH can never be tested by accident.  Run from anywhere with:

  PYTHONIOENCODING=utf-8 PYTHONPATH="<fix>;<tip>/stack;<tip>/taniteval" python -m pytest -q -rs -s test_tactical_conditioning.py
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
import pathlib
import sys

import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve()
_INNER = _HERE.parents[1] / "taniteval"                      # <root>/taniteval : the directory holding the module
_MODULE_FILE = _INNER / "tactical_conditioning.py"


def _load_tc():
    import taniteval
    if str(_INNER) not in [str(p) for p in taniteval.__path__]:
        taniteval.__path__.insert(0, str(_INNER))
    sys.modules.pop("taniteval.tactical_conditioning", None)
    import taniteval.tactical_conditioning as m
    return m


tc = _load_tc()
print(f"\n[test] taniteval.tactical_conditioning imported from: {tc.__file__}")

# the banked inputs (read-only) -- the cross-check section skips, with a stated reason, if they are absent
ROUTE_FILE = pathlib.Path(r"D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/"
                          r"2026-10-04-refcv7-route-following/code/route_metrics.py")
BIN = pathlib.Path(r"D:/refcv7_route_bin/2026-10-04")
MD5 = {"eval_s0g": "1c48a53e84e3a6296004efc0a81a0a64", "eval_s1": "5f783c842497b9dff0d878c98b23f883"}

#: the slot times, re-typed in the test (NOT read from the module)
T = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
#: stop-distance tolerance, derived from the slot spacing and STATED: the flagged segment's own arc is at most
#: v_stop * dt_max = 0.5 m/s * 1.0 s (the last four slots are 1.0 s apart; the first four 0.5 s) = 0.5 m, and the
#: reported point lies inside that segment.
V_STOP, DT_MAX = 0.5, 1.0
TOL_M = V_STOP * DT_MAX
assert float(np.max(np.diff(np.concatenate([[0.0], T])))) == DT_MAX and TOL_M == 0.5


# --------------------------------------------------------------------------------------------------------------- #
# analytic track builders (closed forms; nothing here calls the module)                                              #
# --------------------------------------------------------------------------------------------------------------- #
def arc_then_straight(R, v, total_deg, side, t=T):
    """Constant speed v on a circle of radius R for ``total_deg`` of turning, then straight on the new heading.
    side +1 = left (y > 0), -1 = right."""
    a = math.radians(total_deg)
    s = v * t
    s_arc = np.minimum(s, R * a)
    th = s_arc / R
    rest = s - s_arc
    x = R * np.sin(th) + rest * math.cos(a)
    y = side * (R * (1.0 - np.cos(th)) + rest * math.sin(a))
    return np.stack([x, y], axis=-1)


def full_circle(R, v, side, t=T):
    th = v * t / R
    return np.stack([R * np.sin(th), side * R * (1.0 - np.cos(th))], axis=-1)


def straight(v, t=T):
    return np.stack([v * t, np.zeros_like(t)], axis=-1)


def line_at(deg, v=10.0, t=T):
    a = math.radians(deg)
    return np.stack([v * t * math.cos(a), v * t * math.sin(a)], axis=-1)


def decel_straight(a, v0, t=T):
    """Constant deceleration a (m/s^2) from v0 along x, resting after v0 / a."""
    tt = np.minimum(t, v0 / a)
    s = v0 * tt - 0.5 * a * tt ** 2
    return np.stack([s, np.zeros_like(s)], axis=-1)


def decel_on_arc(a, v0, R, t=T):
    tt = np.minimum(t, v0 / a)
    s = v0 * tt - 0.5 * a * tt ** 2
    th = s / R
    return np.stack([R * np.sin(th), R * (1.0 - np.cos(th))], axis=-1)


def accel_straight(a, v0, t=T):
    return np.stack([v0 * t + 0.5 * a * t ** 2, np.zeros_like(t)], axis=-1)


def from_segment_speeds(v, t=T):
    dt = np.diff(np.concatenate([[0.0], t]))
    return np.stack([np.cumsum(np.asarray(v, float) * dt), np.zeros(len(dt))], axis=-1)


LEFT, RIGHT = arc_then_straight(15.0, 5.0, 90.0, +1), arc_then_straight(15.0, 5.0, 90.0, -1)
STRAIGHT10 = straight(10.0)
STATIONARY = np.zeros((8, 2))


# --------------------------------------------------------------------------------------------------------------- #
# the module under test: where it came from, what it may import, what state it may hold                              #
# --------------------------------------------------------------------------------------------------------------- #
def test_module_is_imported_from_the_fix_tree():
    got = pathlib.Path(tc.__file__).resolve()
    print(f"[test] tactical_conditioning.__file__ = {got}")
    assert got == _MODULE_FILE.resolve()


def test_independence_imports_only_numpy_math_and_the_ci_bootstrap():
    """The module is the CROSS-CHECK of the model's tagger: no tanitad, no refcv8_conditioning, no route package."""
    tree = ast.parse(_MODULE_FILE.read_text(encoding="utf-8"))
    mods = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods += [(a.name, ()) for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            mods.append((n.module or "", tuple(a.name for a in n.names)))
        elif isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) in ("__import__", "import_module"):
            pytest.fail("dynamic import in the metric module")
    assert mods, "found no imports at all -- the AST walk is broken"
    for name, names in mods:
        assert name.split(".")[0] in ("__future__", "math", "numpy", "taniteval"), name
        if name.split(".")[0] == "taniteval":
            assert name == "taniteval" and names == ("ci",), (name, names)       # the bootstrap, nothing else
    flat = " ".join(f"{m}:{','.join(n)}" for m, n in mods)
    for bad in ("tanitad", "refcv8", "route_metrics", "refs"):
        assert bad not in flat, flat


def test_no_module_level_mutable_state():
    for k, v in vars(tc).items():
        if k.startswith("__"):
            continue
        assert not isinstance(v, (list, dict, set, np.ndarray)), f"module-level mutable {k}: {type(v)}"


def test_constants_are_the_retyped_literals():
    assert tc.SLOT_T_S == (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert tc.TAU_DIR_RAD == 0.18063741505146028 and tc.STALL_M == 0.05
    assert (tc.TURN_DEG, tc.STRAIGHT_DEG, tc.HEAD_AGREE_DEG, tc.MIN_LEN_M) == (30.0, 10.0, 15.0, 5.0)
    assert (tc.V_STOP_MS, tc.V_CREEP_MS, tc.DV_MS) == (0.5, 2.0, 1.5)
    assert (tc.ROUTE_DIR_BAR, tc.ROUTE_HEAD15_BAR, tc.CONTROLLABILITY_BAR) == (0.95, 0.70, 0.95)
    assert tc.LON_NAMES == ("HOLD", "CREEP", "STOP", "FOLLOW", "DECELERATE", "ACCELERATE", "KEEP")


# --------------------------------------------------------------------------------------------------------------- #
# path geometry on analytic tracks                                                                                   #
# --------------------------------------------------------------------------------------------------------------- #
def _check_arc_headings(m):
    """R = 15 m arcs at 5 m/s turning 90 deg: the last 1-s segment is straight on the new heading, so the terminal
    heading is +-pi/2 exactly (mathematics, not code)."""
    assert float(m.terminal_heading(LEFT)) == pytest.approx(math.pi / 2, abs=1e-9)
    assert float(m.terminal_heading(RIGHT)) == pytest.approx(-math.pi / 2, abs=1e-9)
    assert int(m.dir_class(m.terminal_heading(LEFT))) == 1 and int(m.dir_class(m.terminal_heading(RIGHT))) == -1
    assert int(m.lat3_class(LEFT)) == 1 and int(m.lat3_class(RIGHT)) == 2
    assert float(m.heading_excursion(LEFT)) == pytest.approx(math.pi / 2, abs=1e-9)
    assert float(m.heading_excursion(RIGHT)) == pytest.approx(-math.pi / 2, abs=1e-9)


def test_circle_arcs_left_and_right():
    _check_arc_headings(tc)


def test_full_six_second_circle_terminal_heading_is_the_mean_end_tangent():
    """R = 15 m, v = 5 m/s: tangent heading at t is v t / R, so the 5 -> 6 s chord heading is the mean of 25/15 and
    30/15 rad = 11/6 rad (a circular chord's heading is the mean of its two end tangents)."""
    left = full_circle(15.0, 5.0, +1)
    assert float(tc.terminal_heading(left)) == pytest.approx(11.0 / 6.0, abs=1e-9)
    assert float(tc.terminal_heading(full_circle(15.0, 5.0, -1))) == pytest.approx(-11.0 / 6.0, abs=1e-9)
    assert int(tc.lat3_class(left)) == 1                                  # 105 deg excursion, 30 m of path


def test_straight_track_reads_lane_keep_keep_and_zero_heading():
    assert float(tc.terminal_heading(STRAIGHT10)) == 0.0
    assert int(tc.dir_class(tc.terminal_heading(STRAIGHT10))) == 0
    assert int(tc.lat3_class(STRAIGHT10)) == 0 and float(tc.heading_excursion(STRAIGHT10)) == 0.0
    assert tc.LON_NAMES[int(tc.lon_class(STRAIGHT10, 10.0))] == "KEEP"
    assert np.allclose(tc.segment_speeds(STRAIGHT10), 10.0, atol=1e-12)
    assert float(tc.path_length(STRAIGHT10)) == pytest.approx(60.0, abs=1e-12)


def test_stationary_track_is_hold_with_zero_headings():
    assert float(tc.terminal_heading(STATIONARY)) == 0.0                    # shorter than 0.05 m -> exactly 0.0
    assert np.all(tc.segment_headings(STATIONARY) == 0.0)
    assert int(tc.lat3_class(STATIONARY)) == 0
    assert tc.LON_NAMES[int(tc.lon_class(STATIONARY, 0.0))] == "HOLD"
    assert float(tc.stop_distance(STATIONARY)) == 0.0                       # at rest from the first segment


def test_stalled_segments_inherit_the_previous_heading_but_terminal_heading_reads_zero():
    P = np.array([[1, 0], [2, 0], [3, 1], [3, 2], [3, 2], [3, 2], [3, 2], [3, 2]], dtype=float)
    expect = [0.0, 0.0, math.pi / 4, math.pi / 2, math.pi / 2, math.pi / 2, math.pi / 2, math.pi / 2]
    assert np.allclose(tc.segment_headings(P), expect, atol=1e-12)
    assert float(tc.terminal_heading(P)) == 0.0                              # route rule: a stalled last segment reads 0.0
    assert float(tc.heading_excursion(P)) == pytest.approx(math.pi / 2, abs=1e-12)


def test_dir_class_thresholds_are_the_route_tau():
    th = np.array([0.2, 0.18063741505146028, 0.1806, -0.2, -0.18063741505146028, -0.1806, 0.0, np.nan])
    assert list(tc.dir_class(th)) == [1, 1, 0, -1, -1, 0, 0, -100]


def test_lat3_threshold_30_degrees_and_side_by_sign():
    assert [int(tc.lat3_class(line_at(a))) for a in (31.0, 29.0, -31.0, -29.0, 0.0)] == [1, 0, 2, 0, 0]


def test_lat3_short_path_is_lane_keep_whatever_its_heading():
    """R = 1 m at 0.5 m/s: 3 m of path turning 172 deg.  Under 5 m -> LANE_KEEP; with the rule off it is a TURN_L."""
    tiny = full_circle(1.0, 0.5, +1)
    assert float(tc.path_length(tiny)) == pytest.approx(3.0, abs=0.05) and int(tc.lat3_class(tiny)) == 0
    assert int(tc.lat3_class(tiny, min_len_m=0.0)) == 1


def test_non_finite_paths_never_become_a_class():
    bad = LEFT.copy()
    bad[3, 0] = np.nan                                                          # mid-plan: the last segment is fine
    assert int(tc.lat3_class(bad)) == -100 and int(tc.lon_class(bad, 5.0)) == -100
    assert np.isnan(float(tc.stop_distance(bad))) and np.isnan(float(tc.heading_excursion(bad)))
    assert np.isfinite(float(tc.terminal_heading(bad)))                         # ... so the route-style terminal heading is still defined
    worse = LEFT.copy()
    worse[7, 1] = np.inf                                                        # the last slot
    assert np.isnan(float(tc.terminal_heading(worse))) and int(tc.dir_class(tc.terminal_heading(worse))) == -100


def _check_dir_not_nan_straight(m):
    assert int(m.dir_class(np.nan)) == -100            # a NaN heading must not be read as "straight"


def test_dir_class_nan_is_ignored_not_straight():
    _check_dir_not_nan_straight(tc)


def test_segment_speeds_literals_and_slot_t_and_v0_guard():
    assert np.allclose(tc.segment_speeds(accel_straight(1.0, 5.0)), [5.25, 5.75, 6.25, 6.75, 7.5, 8.5, 9.5, 10.5], atol=1e-12)
    assert np.allclose(tc.segment_speeds(decel_straight(2.0, 10.0)), [9.5, 8.5, 7.5, 6.5, 5.0, 3.0, 1.0, 0.0], atol=1e-12)
    t2 = np.arange(1.0, 9.0)
    assert np.allclose(tc.segment_speeds(straight(7.0, t2), slot_t=t2), 7.0, atol=1e-12)
    with pytest.raises(ValueError):
        tc.segment_speeds(np.zeros((4, 3, 8, 2)), v0=np.zeros(4))            # [W] against [W, N]: the axis mix-up
    tc.segment_speeds(np.zeros((4, 3, 8, 2)), v0=np.zeros((4, 1)))            # [W, 1] is fine
    with pytest.raises(ValueError):
        tc.segment_speeds(np.zeros((8, 2)), slot_t=(1.0, 2.0))               # wrong number of slots


# --------------------------------------------------------------------------------------------------------------- #
# stop distance, longitudinal class                                                                                  #
# --------------------------------------------------------------------------------------------------------------- #
def _check_stop_distance_curved(m):
    """-2 m/s^2 from 10 m/s on an R = 15 m arc: 25.0 m of ARC (chord from the origin would read 22.2 m)."""
    d = float(m.stop_distance(decel_on_arc(2.0, 10.0, 15.0)))
    assert abs(d - 25.0) <= TOL_M, d


def test_stop_distance_constant_deceleration():
    """v0^2 / 2a = 100 / 4 = 25.0 m; tolerance TOL_M = 0.5 m = v_stop x the 1.0 s slot gap (STATED above)."""
    assert abs(float(tc.stop_distance(decel_straight(2.0, 10.0))) - 25.0) <= TOL_M
    assert abs(float(tc.stop_distance(decel_straight(3.0, 10.0))) - 100.0 / 6.0) <= TOL_M       # 16.667 m, off the slot grid
    assert abs(float(tc.stop_distance(decel_straight(2.1, 10.0))) - 100.0 / 4.2) <= TOL_M         # 23.81 m


def test_stop_distance_interpolation_convention_is_pinned_inside_the_flagged_segment():
    """The tolerance test above admits ANY point of the flagged segment; this pins the documented convention.  Segment
    speeds 10 x6, 0.9, 0.4: arc at the start of the last segment = 5+5+5+5+10+10+0.9 = 40.9 m; segment 7 (0.9 m/s) is
    not at rest, segment 8 (0.4 m/s <= 0.5) is; phi = (0.9 - 0.5) / (0.9 - 0.4) = 0.8 and the segment's arc is 0.4 m, so
    d = 40.9 + 0.8 x 0.4 = 41.22 m (end-of-segment would read 41.3, start-of-segment 40.9)."""
    P = from_segment_speeds([10, 10, 10, 10, 10, 10, 0.9, 0.4])
    assert float(tc.stop_distance(P)) == pytest.approx(41.22, abs=1e-9)


def test_stop_distance_is_arc_length_on_a_curved_stop():
    _check_stop_distance_curved(tc)


def test_stop_distance_nan_when_never_at_rest_and_zero_when_already_at_rest():
    assert np.isnan(float(tc.stop_distance(STRAIGHT10))) and np.isnan(float(tc.stop_distance(accel_straight(1.0, 5.0))))
    assert float(tc.stop_distance(STATIONARY)) == 0.0
    batch = np.stack([decel_straight(2.0, 10.0), STRAIGHT10])
    d = tc.stop_distance(batch)
    assert d.shape == (2,) and abs(d[0] - 25.0) <= TOL_M and np.isnan(d[1])


def test_lon_class_analytic_tracks():
    names = tc.LON_NAMES
    assert names[int(tc.lon_class(STRAIGHT10, 10.0))] == "KEEP"
    assert names[int(tc.lon_class(decel_straight(2.0, 10.0), 10.0))] == "STOP"
    assert names[int(tc.lon_class(accel_straight(1.0, 5.0), 5.0))] == "ACCELERATE"
    assert names[int(tc.lon_class(STATIONARY, 0.0))] == "HOLD"
    assert names[int(tc.lon_class(straight(1.5), 1.5))] == "CREEP"
    assert names[int(tc.lon_class(decel_straight(1.0, 10.0), 10.0))] == "DECELERATE"        # 10 -> 4 m/s, never at rest


def test_lon_class_precedence_literals():
    names = tc.LON_NAMES
    f = lambda v, v0: names[int(tc.lon_class(from_segment_speeds(v), v0))]            # noqa: E731
    assert f([5, 5, 7, 7, 0.2, 0.2, 0.2, 0.2], 5.0) == "ACCELERATE"        # an ACCEL event (7 >= 6.5) comes before the stop
    assert f([5, 5, 3, 0.2, 0.2, 0.2, 0.2, 0.2], 5.0) == "STOP"
    assert f([5, 5, 3, 3, 6.6, 6.6, 6.6, 6.6], 5.0) == "DECELERATE"        # first event is the decel (3 <= 3.5)
    assert f([1.8, 1.5, 1.0, 0.3, 0.3, 0.3, 0.3, 0.3], 1.8) == "CREEP"      # CREEP outranks STOP
    assert f([0.3] * 8, 0.4) == "HOLD"
    assert f([0.3] * 7 + [1.0], 0.4) == "CREEP"                               # v0 <= 0.5 but the plan moves off
    assert f([2.5, 1.5, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3], 3.0) == "STOP"         # v0 > 0.5, max segment speed 2.5 > 2.0
    # documented edge: CREEP is judged on the SEGMENT speeds only (v9 text "max <= 2.0"; v0 enters HOLD and the events),
    # so a brake from 3 m/s that is already under 2 m/s on average over the first 0.5 s reads CREEP, not STOP
    assert f([0.3] * 8, 3.0) == "CREEP"
    assert f([10, 10, 10, 10, 10, 10, 10, 8.4], 10.0) == "DECELERATE"          # 8.4 <= 10 - 1.5
    assert f([10, 10, 10, 10, 10, 10, 10, 8.6], 10.0) == "KEEP"                # 8.6 > 8.5: not an event
    assert f([10, 10, 10, 10, 10, 10, 10, 11.4], 10.0) == "KEEP" and f([10, 10, 10, 10, 10, 10, 10, 11.6], 10.0) == "ACCELERATE"


def test_lon_class_never_emits_follow_and_batches_broadcast():
    stack = np.stack([STRAIGHT10, decel_straight(2.0, 10.0), accel_straight(1.0, 5.0)])
    c = tc.lon_class(stack, np.array([10.0, 10.0, 5.0]))
    assert list(c) == [6, 2, 5] and 3 not in c


def test_observable_in_plan():
    assert list(tc.observable_in_plan(np.array([-1.0, 0.0, 2.0, 5.0, 5.01, np.nan, np.inf]))) == \
        [True, True, True, True, False, False, False]
    assert bool(tc.observable_in_plan(6.0, max_start_s=7.0))


def test_plan_lon_to_v7_allowed_literals():
    A = tc.plan_lon_to_v7_allowed(np.array([2, 6, 0, 3, -100]))              # STOP, KEEP, HOLD, FOLLOW, ignore
    assert A.shape == (5, 8)
    assert [list(np.nonzero(r)[0]) for r in A] == [[3], [0, 1], [5], [0], []]


# --------------------------------------------------------------------------------------------------------------- #
# (i) class_report                                                                                                   #
# --------------------------------------------------------------------------------------------------------------- #
NAMES3 = ("A", "B", "C")


def test_class_report_perfect_predictor():
    tgt = np.array([0, 1, 2, 0, 1, 2, 0, 0])
    r = tc.class_report(tgt, tgt, 3, NAMES3)
    assert r["accuracy"] == 1.0 and r["macro_f1"] == 1.0
    assert [r["per_class"][n]["recall"] for n in NAMES3] == [1.0, 1.0, 1.0]
    assert r["n_scored"] == 8 and r["n_ignored"] == 0 and r["majority_control"] == {"class": "A", "accuracy": 0.5}


def test_class_report_constant_predictor_equals_the_majority_control_exactly():
    tgt = np.array([0] * 6 + [1] * 3 + [2])                                 # shares 0.6 / 0.3 / 0.1
    r = tc.class_report(np.zeros(10, dtype=int), tgt, 3, NAMES3)
    assert r["accuracy"] == r["majority_control"]["accuracy"] == pytest.approx(0.6, abs=1e-12)
    assert r["majority_control"]["class"] == "A"
    # class A: precision 0.6, recall 1.0 -> F1 = 0.75; B and C never predicted -> F1 0; macro = 0.25
    assert r["per_class"]["A"]["f1"] == pytest.approx(0.75, abs=1e-12) and r["macro_f1"] == pytest.approx(0.25, abs=1e-12)
    assert r["per_class"]["B"]["precision"] is None and r["per_class"]["B"]["f1"] == 0.0


def test_class_report_ignore_index_and_empty_class_conventions():
    tgt = np.array([0, 0, -100, 1, -100, 0])
    pred = np.array([0, 1, 2, 1, 0, 0])
    r = tc.class_report(pred, tgt, 3, NAMES3)
    assert r["n_scored"] == 4 and r["n_ignored"] == 2 and r["accuracy"] == pytest.approx(0.75, abs=1e-12)
    assert r["per_class"]["C"]["n"] == 0 and r["per_class"]["C"]["recall"] is None and r["per_class"]["C"]["f1"] is None
    assert r["macro_f1"] is not None                                         # over classes with n > 0 only
    with pytest.raises(ValueError):
        tc.class_report(np.array([5, 0]), np.array([0, 0]), 3, NAMES3)       # out-of-range pred is an error, not "wrong"


def _check_partial(m):
    """Six windows, classes A,B,C = LANE_KEEP, LC_L, LC_R.  Hand-scored:
       w0 target A allowed{A}    pred A  correct        w3 target - allowed{A,B} pred C  wrong
       w1 target A allowed{A}    pred B  wrong          w4 target C allowed{C}   pred C  correct
       w2 target - allowed{A,B}  pred B  correct (partial credit)        w5 target - allowed{}     ignored
       -> 5 scored, 3 correct: accuracy 0.6   (ignoring `allowed` the scored set is {w0,w1,w4}: 2/3)."""
    tgt = np.array([0, 0, -100, -100, 2, -100])
    pred = np.array([0, 1, 1, 2, 2, 0])
    A = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
    r = m.class_report(pred, tgt, 3, NAMES3, allowed=A, train_marginal=[0.1, 0.2, 0.7])
    assert r["n_scored"] == 5 and r["n_ignored"] == 1 and r["n_partial_scored"] == 2
    assert r["accuracy"] == pytest.approx(0.6, abs=1e-12)
    assert r["per_class"]["A"]["n"] == 2 and r["per_class"]["A"]["recall"] == pytest.approx(0.5, abs=1e-12)
    assert r["per_class"]["B"]["n"] == 0 and r["per_class"]["B"]["precision"] == pytest.approx(0.5, abs=1e-12)
    assert r["per_class"]["C"]["recall"] == 1.0
    # best constant: A is allowed on w0..w3 (4/5), B on w2,w3 (2/5), C on w4 (1/5)
    assert r["majority_control"] == {"class": "A", "accuracy": pytest.approx(0.8, abs=1e-12)}
    assert r["train_marginal_control"] == {"class": "C", "accuracy": pytest.approx(0.2, abs=1e-12)}


def test_class_report_partial_labels():
    _check_partial(tc)


def test_class_report_train_marginal_control_without_partial_labels():
    tgt = np.array([0, 0, 1, 2])
    r = tc.class_report(np.array([0, 1, 1, 2]), tgt, 3, NAMES3, train_marginal=[0.2, 0.5, 0.3])
    assert r["train_marginal_control"] == {"class": "B", "accuracy": pytest.approx(0.25, abs=1e-12)}


def _check_left_right_recall(m):
    """L and R recall are 1.0 for the true labels and 0.0 for the sign-flipped prediction."""
    tgt = np.array([1, 1, 2, 2, 0, 0])
    good = m.class_report(tgt, tgt, 3, m.LAT3_NAMES)
    flipped = m.class_report(np.array([2, 2, 1, 1, 0, 0]), tgt, 3, m.LAT3_NAMES)
    assert good["per_class"]["TURN_L"]["recall"] == 1.0 and good["per_class"]["TURN_R"]["recall"] == 1.0
    assert flipped["per_class"]["TURN_L"]["recall"] == 0.0 and flipped["per_class"]["TURN_R"]["recall"] == 0.0
    assert flipped["per_class"]["LANE_KEEP"]["recall"] == 1.0


def test_sign_flipped_lat_swaps_left_right_recall():
    _check_left_right_recall(tc)


# --------------------------------------------------------------------------------------------------------------- #
# goal AP, constraint MAE                                                                                            #
# --------------------------------------------------------------------------------------------------------------- #
def _check_ap_constant_is_prevalence(m):
    """A constant scorer's AP equals the prevalence EXACTLY (ties are one threshold); literals 1/3 and 1/2."""
    y = np.array([[0], [0], [0], [0], [1], [1]])
    r = m.goal_ap(np.full((6, 1), 0.3), y, np.ones((6, 1)), ["tok"])["tok"]
    assert r["ap"] == r["prevalence"] and r["ap"] == pytest.approx(1.0 / 3.0, abs=1e-15)
    y2 = np.array([[1], [0], [1], [0]])
    r2 = m.goal_ap(np.full((4, 1), -2.0), y2, np.ones((4, 1)), ["tok"])["tok"]
    assert r2["ap"] == r2["prevalence"] == 0.5 and r2["lift"] == 1.0


def test_goal_ap_constant_scorer_equals_prevalence_exactly():
    _check_ap_constant_is_prevalence(tc)


def test_goal_ap_ignore_cells_do_not_count():
    S = np.full((8, 1), 0.7)
    y = np.array([[1], [1], [1], [1], [0], [0], [0], [0]])
    w = np.array([[1], [1], [0], [0], [1], [1], [1], [1]])               # IGNORE two positives
    r = tc.goal_ap(S, y, w, ["tok"])["tok"]
    assert (r["n"], r["n_pos"], r["n_neg"]) == (6, 2, 4) and r["ap"] == r["prevalence"] == pytest.approx(1.0 / 3.0, abs=1e-15)


def test_goal_ap_known_rankings():
    w1 = np.ones((4, 1))
    y = np.array([[1], [1], [0], [0]])
    perfect = tc.goal_ap(np.array([[0.9], [0.8], [0.2], [0.1]]), y, w1, ["t"])["t"]
    assert perfect["ap"] == pytest.approx(1.0, abs=1e-15)
    reversed_ = tc.goal_ap(np.array([[0.1], [0.2], [0.3], [0.4]]), y, w1, ["t"])["t"]
    assert reversed_["ap"] == pytest.approx(5.0 / 12.0, abs=1e-12)           # precisions 1/3 and 2/4 at the two positives
    tie = tc.goal_ap(np.array([[1.0], [1.0], [0.0], [0.0]]), np.array([[1], [0], [1], [0]]), w1, ["t"])["t"]
    assert tie["ap"] == pytest.approx(0.5, abs=1e-15)                        # two tied blocks of (1 pos, 1 neg)


def test_goal_ap_token_without_positives_is_none_not_zero():
    r = tc.goal_ap(np.array([[0.1], [0.2]]), np.array([[0], [0]]), np.ones((2, 1)), ["t"])["t"]
    assert r["ap"] is None and r["n_pos"] == 0 and r["prevalence"] == 0.0


def test_constraint_mae_literals():
    tgt = np.array([10.0, 20.0, 30.0, np.nan])
    pred = np.array([11.0, 18.0, 33.0, 5.0])
    r = tc.constraint_mae({"d": pred}, {"d": tgt}, {"d": np.ones(4, bool)}, {"d": 20.0}, prior={"d": np.array([10.0, 25.0, 20.0, 0.0])})["d"]
    assert r["n"] == 3 and r["mae"] == pytest.approx(2.0, abs=1e-12)
    assert r["mae_const"] == pytest.approx(20.0 / 3.0, abs=1e-12) and r["ratio"] == pytest.approx(0.3, abs=1e-12)
    assert r["mae_prior"] == pytest.approx(5.0, abs=1e-12) and r["ratio_prior"] == pytest.approx(0.4, abs=1e-12)
    arr = tc.constraint_mae(pred, tgt, np.array([1, 1, 1, 0], bool), 20.0)["value"]
    assert arr["n"] == 3 and arr["mae"] == pytest.approx(2.0, abs=1e-12)
    bad = pred.copy()
    bad[1] = np.nan
    with pytest.raises(ValueError):
        tc.constraint_mae(bad, tgt, np.ones(4, bool), 20.0)                  # dropping it would flatter the head


# --------------------------------------------------------------------------------------------------------------- #
# (ii) controllability                                                                                               #
# --------------------------------------------------------------------------------------------------------------- #
def _fan(tracks, W=2):
    return np.broadcast_to(np.stack(tracks), (W, len(tracks), 8, 2)).copy()


LEFT_ARCS = [arc_then_straight(R, 5.0, 90.0, +1) for R in (15.0, 20.0, 25.0, 30.0)] + [full_circle(15.0, 5.0, +1)]


def test_controllability_pure_left_fan_reads_one_and_zero():
    fan = _fan(LEFT_ARCS)
    pick = np.zeros(2, dtype=int)
    v0 = np.array([5.0, 5.0])
    L = tc.controllability(fan, "TURN_L", None, pick, v0)
    R = tc.controllability(fan, "TURN_R", None, pick, v0)
    K = tc.controllability(fan, "LANE_KEEP", None, pick, v0)
    assert L["mean_share_cond"] == 1.0 and L["mean_pick_ok"] == 1.0 and L["mean_share_cond_alt"] == 1.0
    assert R["mean_share_cond"] == 0.0 and R["mean_pick_ok"] == 0.0 and K["mean_share_cond"] == 0.0
    assert L["alt_reading"] == "lat3_class" and L["n_windows"] == 2 and L["n_candidates"] == 5


def test_controllability_condition_ignored_reads_the_base_rate():
    """A model that ignores the condition returns the same fan whatever is forced: 3 left + 3 straight -> 0.5."""
    fan = _fan([LEFT_ARCS[0], LEFT_ARCS[1], LEFT_ARCS[2], straight(6.0), straight(8.0), straight(10.0)])
    r = tc.controllability(fan, "TURN_L", None, np.zeros(2, dtype=int), np.full(2, 5.0))
    assert r["mean_share_cond"] == 0.5 and r["mean_share_all"] == 0.5
    assert tc.controllability(fan, "LANE_KEEP", None, np.zeros(2, dtype=int), np.full(2, 5.0))["mean_share_cond"] == 0.5


def _check_cond_mask(m):
    """3 conditioned left arcs + 3 unconditioned straights: the share is read over the CONDITIONED candidates only (1.0);
    the unconditioned base rate of the same fan is 0.5."""
    fan = _fan([LEFT_ARCS[0], LEFT_ARCS[1], LEFT_ARCS[2], straight(6.0), straight(8.0), straight(10.0)])
    cond = np.broadcast_to(np.array([1, 1, 1, 0, 0, 0], bool), (2, 6))
    r = m.controllability(fan, "TURN_L", cond, np.zeros(2, dtype=int), np.full(2, 5.0))
    assert r["mean_share_cond"] == 1.0 and r["mean_share_all"] == 0.5


def test_controllability_reads_the_conditioned_candidates_only():
    _check_cond_mask(tc)


def test_controllability_window_without_conditioned_candidates_is_nan_not_zero():
    fan = _fan(LEFT_ARCS[:3])
    cond = np.array([[1, 1, 1], [0, 0, 0]], bool)
    r = tc.controllability(fan, "TURN_L", cond, np.zeros(2, dtype=int), np.full(2, 5.0))
    assert r["n_windows_no_cond"] == 1 and np.isnan(r["share_cond"][1]) and r["mean_share_cond"] == 1.0


def test_controllability_stop_distance_share_pick_and_alt_reading():
    """Stop at d = 25 m, tol = max(2, 2.5) = 2.5: A (-2.0, 25.0 m) and B (-2.1, 23.8 m) obey, C (-3, 16.7 m) and D (never
    stops) do not -> 0.5.  The class reading (lon_class == STOP) also counts C: 3/4."""
    A, B, C = decel_straight(2.0, 10.0), decel_straight(2.1, 10.0), decel_straight(3.0, 10.0)
    fan = _fan([A, B, C, STRAIGHT10])
    pick = np.array([0, 3])                                                  # window 0 picks A (obeys), window 1 picks D (not)
    r = tc.controllability(fan, "STOP", None, pick, np.full(2, 10.0), stop_d=np.full(2, 25.0))
    assert r["mean_share_cond"] == 0.5 and r["mean_pick_ok"] == 0.5 and r["mean_share_cond_alt"] == 0.75
    assert r["alt_reading"] == "lon_class==STOP"
    tight = tc.controllability(fan, "STOP", None, pick, np.full(2, 10.0), stop_d=np.full(2, 25.0), stop_tol=lambda d: 0.1)
    assert tight["mean_share_cond"] == 0.25                                  # only A: a scalar-only tolerance callable works
    with pytest.raises(ValueError):
        tc.controllability(fan, "STOP", None, pick, np.full(2, 10.0))        # STOP needs a distance
    with pytest.raises(ValueError):
        tc.controllability(fan, "REVERSE", None, pick, np.full(2, 10.0))


def _arm(pick_ok):
    return {"pick_ok": np.asarray(pick_ok, bool)}


def test_controllability_vs_shuffled_verdicts():
    eid = np.repeat([f"e{i}" for i in range(20)], 5)                          # 20 episodes x 5 windows
    forced = np.ones(100, bool)
    rng = np.random.default_rng(0)
    shuf = rng.random(100) < 0.4
    v = tc.controllability_vs_shuffled(_arm(forced), _arm(shuf), eid, n_boot=500)
    assert v["verdict"]["pass"] is True and v["verdict"]["forced_pick_rate"] == 1.0
    assert v["verdict"]["shuffled_pick_rate"] == pytest.approx(float(shuf.mean()), abs=1e-12) and v["paired_delta"]["separated"]
    # forced arm below the bar -> fail
    low = forced.copy()
    low[::10] = False                                                         # 0.90
    assert tc.controllability_vs_shuffled(_arm(low), _arm(shuf), eid, n_boot=500)["verdict"]["pass"] is False
    # shuffled arm equal to the forced arm (the condition did nothing) -> fail on the shuffled side
    same = tc.controllability_vs_shuffled(_arm(forced), _arm(forced), eid, n_boot=500)["verdict"]
    assert same["pass"] is False and same["shuffled_below_bar"] is False and same["forced_above_shuffled_separated"] is False
    with pytest.raises(ValueError):
        tc.controllability_vs_shuffled(_arm(forced), _arm(shuf[:50]), eid, n_boot=10)


# --------------------------------------------------------------------------------------------------------------- #
# (iii) consistency                                                                                                  #
# --------------------------------------------------------------------------------------------------------------- #
CONS_TRACKS = ([arc_then_straight(R, 5.0, 90.0, +1) for R in (15.0, 20.0, 25.0, 30.0)]
               + [straight(v) for v in (6.0, 8.0, 10.0, 12.0)]
               + [arc_then_straight(R, 5.0, 90.0, -1) for R in (15.0, 20.0, 25.0, 30.0)])
CONS_LABELS = np.array([1] * 4 + [0] * 4 + [2] * 4)                           # lat3 ids of the 12 candidates (by construction)


def _check_consistency_literals(m):
    """12-candidate fan, 4 left / 4 straight / 4 right.  Own labels as tags -> 1.0.  Over ALL 12 cyclic shifts of the tag
    vector each ordered pair of same-class candidates matches exactly once, so the mean share is sum(n_c^2)/12^2 = 48/144
    = 1/3 (the class base rate); a shift by one class block (4) matches nothing -> 0.0."""
    fan = _fan(CONS_TRACKS)
    pick = np.zeros(2, dtype=int)
    own = np.broadcast_to(CONS_LABELS, (2, 12))
    r = m.consistency(fan, own, pick)
    assert r["summary"]["all"]["cand_share"] == 1.0 and r["summary"]["all"]["pick_own_tag"] == 1.0
    shifts = [m.consistency(fan, np.roll(own, s, axis=1), pick)["summary"]["all"]["cand_share"] for s in range(12)]
    assert float(np.mean(shifts)) == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert shifts[4] == 0.0 and shifts[8] == 0.0 and shifts[0] == 1.0


def test_consistency_own_labels_one_and_permuted_tags_fall_to_the_class_base_rate():
    _check_consistency_literals(tc)


def test_consistency_untagged_pick_and_tactical_argmax_masks():
    fan = _fan(CONS_TRACKS, W=3)
    tags = np.broadcast_to(CONS_LABELS, (3, 12)).copy()
    tags[:, :2] = -1                                                           # two untagged left arcs per window
    tags[2, 4:8] = 2                                                           # window 2: straights tagged TURN_R (wrong)
    pick = np.array([2, 4, 4])                                                  # a tagged left arc / a straight / a straight
    tac = np.array([1, 2, 0])
    r = tc.consistency(fan, tags, pick, tac, masks={"all": np.ones(3, bool), "w2": np.array([0, 0, 1], bool)})
    # every window has 12 - 2 = 10 tagged candidates; in window 2 the four straights (idx 4..7) carry TURN_R -> 6 / 10
    assert r["cand_share"].tolist() == [1.0, 1.0, pytest.approx(0.6, abs=1e-12)]
    assert r["pick_own_tag"].tolist() == [1.0, 1.0, 0.0]                        # window 2: the pick is a straight tagged TURN_R
    assert r["pick_vs_tac"].tolist() == [1.0, 0.0, 1.0]                        # pick classes (1, 0, 0) against tac (1, 2, 0)
    assert r["summary"]["w2"]["n_windows"] == 1 and r["summary"]["w2"]["pooled_cand_share"] == pytest.approx(0.6, abs=1e-12)
    assert r["summary"]["all"]["n_tagged_candidates"] == 30 and r["summary"]["all"]["n_windows"] == 3
    # a pick that is itself untagged -> NaN, never 0
    assert np.isnan(tc.consistency(fan, tags, np.array([0, 4, 4]))["pick_own_tag"][0])
    assert np.isnan(tc.consistency(fan, tags, pick)["pick_vs_tac"]).all()      # no tactical argmax given


def test_consistency_two_readings_disagree_on_a_gentle_candidate():
    """A straight line at 20 deg: lat3 (30-deg excursion) calls it LANE_KEEP, the route direction class calls it LEFT."""
    fan = _fan([line_at(20.0)], W=1)
    tags = np.array([[1]])                                                      # tagged TURN_L
    lat3 = tc.consistency(fan, tags, np.zeros(1, dtype=int), reading="lat3")
    side = tc.consistency(fan, tags, np.zeros(1, dtype=int), reading="dir")
    assert lat3["summary"]["all"]["cand_share"] == 0.0 and side["summary"]["all"]["cand_share"] == 1.0
    with pytest.raises(ValueError):
        tc.consistency(fan, tags, np.zeros(1, dtype=int), reading="chord")
    with pytest.raises(ValueError):
        tc.consistency(fan, np.array([[3]]), np.zeros(1, dtype=int))


# --------------------------------------------------------------------------------------------------------------- #
# (iv) route following                                                                                               #
# --------------------------------------------------------------------------------------------------------------- #
def test_route_following_hand_scored_windows():
    gt = np.stack([LEFT, RIGHT, LEFT, STRAIGHT10, line_at(20.0), LEFT, STATIONARY])
    gv = np.ones((7, 8), bool)
    gv[5, -1] = False                                                           # slot 60 not observed -> unclassified
    pick = np.stack([LEFT, STRAIGHT10, line_at(70.0), STRAIGHT10, STRAIGHT10, STRAIGHT10, STRAIGHT10])
    r = tc.route_following(pick, gt, gv)
    assert list(r["gt_cls"]) == ["turnL", "turnR", "turnL", "straight", "gentle", "unclassified", "unclassified"]
    s = r["summary"]
    # turn windows w0,w1,w2: w0 ok/ok, w1 wrong/wrong, w2 right side (+) but 20 deg off -> dir ok, head15 not
    assert s["turn"]["n"] == 3 and s["turn"]["n_dir_correct"] == 2 and s["turn"]["n_head15"] == 1
    assert s["turn"]["dir_correct"] == pytest.approx(2.0 / 3.0, abs=1e-12) and s["turn"]["head15"] == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert s["turnL"]["n"] == 2 and s["turnR"]["n"] == 1 and s["turnR"]["dir_correct"] == 0.0
    assert s["straight"]["dir_correct"] == 1.0 and s["straight"]["head15"] == 1.0
    assert s["gentle"]["dir_correct"] == 0.0 and s["gentle"]["head15"] == 0.0
    assert s["all_classified"]["n"] == 5 and s["all_classified"]["dir_correct"] == pytest.approx(0.6, abs=1e-12)
    assert s["all_classified"]["head15"] == pytest.approx(0.4, abs=1e-12)
    assert np.isnan(r["dir_ok"][5]) and np.isnan(r["head15"][6])
    assert r["bars"]["dir_correct_turn"] == {"value": pytest.approx(2.0 / 3.0), "bar": 0.95, "meets": False}
    assert r["bars"]["head15_turn"]["bar"] == 0.70 and r["bars"]["head15_turn"]["meets"] is False
    assert (tc.ROUTE_DIR_BAR, tc.ROUTE_HEAD15_BAR) == (0.95, 0.70)


def test_route_following_a_perfect_pick_meets_both_bars_and_a_nan_pick_counts_wrong():
    gt = np.stack([LEFT, RIGHT, STRAIGHT10])
    gv = np.ones((3, 8), bool)
    r = tc.route_following(gt.copy(), gt, gv)
    assert r["bars"]["dir_correct_turn"]["meets"] and r["bars"]["head15_turn"]["meets"] and r["n_pick_nonfinite"] == 0
    bad = gt.copy()
    bad[2, 7, 0] = np.nan                                                       # NaN on the straight window: must NOT read as "straight, correct"
    rb = tc.route_following(bad, gt, gv)
    assert rb["n_pick_nonfinite"] == 1 and rb["summary"]["straight"]["dir_correct"] == 0.0 and rb["summary"]["straight"]["head15"] == 0.0


def test_route_following_bootstrap_interval_is_carried_with_its_estimator():
    eid = np.array(["a", "a", "b", "b", "c", "c"])
    gt = np.stack([LEFT, LEFT, LEFT, LEFT, LEFT, LEFT])
    pick = np.stack([LEFT, STRAIGHT10, LEFT, LEFT, STRAIGHT10, STRAIGHT10])
    r = tc.route_following(pick, gt, np.ones((6, 8), bool), eid=eid, n_boot=200)
    ci = r["summary"]["turn"]["dir_correct_ci"]
    assert ci["estimator"] == "episode_cluster_bootstrap" and ci["n_episodes"] == 3 and ci["mean"] == pytest.approx(0.5, abs=1e-4)


# --------------------------------------------------------------------------------------------------------------- #
# MUTATIONS -- each must be caught by the SAME known-value check, and the check must be green again once undone       #
# --------------------------------------------------------------------------------------------------------------- #
def _mutate_and_catch(monkeypatch, name, mutant, check):
    check(tc)                                                                   # control: the real function is green
    with monkeypatch.context() as mp:
        mp.setattr(tc, name, mutant)
        with pytest.raises(AssertionError):
            check(tc)                                                           # the mutant is RED
    check(tc)                                                                   # restored: green again


def test_m1a_mirrored_y_inside_segment_headings_turns_TURN_L_into_TURN_R(monkeypatch):
    orig = tc.segment_headings
    _mutate_and_catch(monkeypatch, "segment_headings", lambda p: orig(np.asarray(p, float) * np.array([1.0, -1.0])), _check_arc_headings)


def test_m1b_mirrored_y_inside_terminal_heading_flips_direction_and_controllability(monkeypatch):
    orig = tc.terminal_heading

    def check(m):
        _check_arc_headings(m)
        fan = _fan(LEFT_ARCS)
        r = m.controllability(fan, "TURN_L", None, np.zeros(2, dtype=int), np.full(2, 5.0))
        assert r["mean_share_cond"] == 1.0                                       # DESIGN §4: mirrored y => TURN_L reads 0
    _mutate_and_catch(monkeypatch, "terminal_heading", lambda p: orig(np.asarray(p, float) * np.array([1.0, -1.0])), check)
    with monkeypatch.context() as mp:
        mp.setattr(tc, "terminal_heading", lambda p: orig(np.asarray(p, float) * np.array([1.0, -1.0])))
        fan = _fan(LEFT_ARCS)
        assert tc.controllability(fan, "TURN_L", None, np.zeros(2, dtype=int), np.full(2, 5.0))["mean_share_cond"] == 0.0


def test_m2_chord_instead_of_arc_breaks_the_curved_stop_distance(monkeypatch):
    """On the R = 15 m stopping track the chord from the origin is 22.2 m for 25.0 m of arc: error 2.8 m >> 0.5 m."""
    chord = lambda paths: np.linalg.norm(np.asarray(paths, float), axis=-1)       # noqa: E731 (|P_j|, not the arc sum)
    _mutate_and_catch(monkeypatch, "_cum_arc", chord, _check_stop_distance_curved)
    with monkeypatch.context() as mp:
        mp.setattr(tc, "_cum_arc", chord)
        assert abs(float(tc.stop_distance(decel_on_arc(2.0, 10.0, 15.0))) - 25.0) > 2.0        # the size of the miss, stated


def test_m3_class_report_ignoring_allowed_is_caught(monkeypatch):
    orig = tc.class_report

    def mutant(pred, target, n_classes, names, allowed=None, **kw):
        return orig(pred, target, n_classes, names, allowed=None, **kw)
    _mutate_and_catch(monkeypatch, "class_report", mutant, _check_partial)
    with monkeypatch.context() as mp:
        mp.setattr(tc, "class_report", mutant)
        tgt = np.array([0, 0, -100, -100, 2, -100])
        A = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]], bool)
        assert tc.class_report(np.array([0, 1, 1, 2, 2, 0]), tgt, 3, NAMES3, allowed=A)["accuracy"] == pytest.approx(2.0 / 3.0)   # 0.6667, not 0.6


def test_m4_goal_ap_with_ties_broken_by_index_is_caught(monkeypatch):
    def by_index(scores, y):
        s = np.asarray(scores, float)
        yy = np.asarray(y, float)[np.argsort(-s, kind="stable")]               # a tied block is ordered by index
        return float((np.cumsum(yy) / np.arange(1, len(yy) + 1) * yy).sum() / yy.sum())
    _mutate_and_catch(monkeypatch, "_average_precision", by_index, _check_ap_constant_is_prevalence)
    with monkeypatch.context() as mp:
        mp.setattr(tc, "_average_precision", by_index)
        r = tc.goal_ap(np.full((6, 1), 0.3), np.array([[0], [0], [0], [0], [1], [1]]), np.ones((6, 1)), ["t"])["t"]
        assert r["ap"] == pytest.approx((1 / 5 + 2 / 6) / 2, abs=1e-12) and r["ap"] != r["prevalence"]       # 0.2667 vs 1/3


def test_m5_controllability_reading_the_unconditioned_fan_is_caught(monkeypatch):
    orig = tc.controllability

    def mutant(paths, forced, cond_mask, pick_idx, v0, **kw):
        return orig(paths, forced, None, pick_idx, v0, **kw)                    # ignores the conditioning mask
    _mutate_and_catch(monkeypatch, "controllability", mutant, _check_cond_mask)
    with monkeypatch.context() as mp:
        mp.setattr(tc, "controllability", mutant)
        fan = _fan([LEFT_ARCS[0], LEFT_ARCS[1], LEFT_ARCS[2], straight(6.0), straight(8.0), straight(10.0)])
        cond = np.broadcast_to(np.array([1, 1, 1, 0, 0, 0], bool), (2, 6))
        assert tc.controllability(fan, "TURN_L", cond, np.zeros(2, dtype=int), np.full(2, 5.0))["mean_share_cond"] == 0.5


def test_m6_dir_class_without_the_finite_guard_reads_nan_as_straight(monkeypatch):
    """The historical defect this module guards: np.where(th >= tau, ...) calls NaN 'straight'."""
    unguarded = lambda theta, tau=0.18063741505146028: np.where(  # noqa: E731
        np.asarray(theta) >= tau, 1, np.where(np.asarray(theta) <= -tau, -1, 0)).astype(np.int8)
    _mutate_and_catch(monkeypatch, "dir_class", unguarded, _check_dir_not_nan_straight)


def test_m7_label_swap_mutation_is_visible_as_swapped_left_right_recall(monkeypatch):
    swap = lambda c: np.where(np.asarray(c) == 1, 2, np.where(np.asarray(c) == 2, 1, c))          # noqa: E731
    orig = tc.class_report

    def mutant(pred, target, n_classes, names, **kw):
        return orig(swap(pred), target, n_classes, names, **kw)
    _mutate_and_catch(monkeypatch, "class_report", mutant, _check_left_right_recall)


def test_m8_consistency_with_an_off_by_one_tag_alphabet_is_caught(monkeypatch):
    """A tag alphabet where left/right are swapped (1<->2) must break the own-label control (1.0 -> 0.33)."""
    orig = tc._cand_class

    def mutant(paths, reading):
        c = orig(paths, reading)
        return np.where(c == 1, 2, np.where(c == 2, 1, c))
    _mutate_and_catch(monkeypatch, "_cand_class", mutant, _check_consistency_literals)


# --------------------------------------------------------------------------------------------------------------- #
# cross-check against the route package on the banked refcv7 capture (TEST-ONLY; the module never imports it)         #
# --------------------------------------------------------------------------------------------------------------- #
def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@pytest.fixture(scope="module")
def route():
    if not ROUTE_FILE.is_file():
        pytest.skip(f"route package not present: {ROUTE_FILE}")
    spec = importlib.util.spec_from_file_location("route_metrics_under_cross_check", ROUTE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _capture(tag):
    p = BIN / f"{tag}.npz"
    if not p.is_file():
        pytest.skip(f"banked capture not present: {p}")
    assert _md5(p) == MD5[tag], f"{p} md5 changed"
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in ("fan", "traj", "sel_idx", "gt", "gt_valid", "win_sha12")}


@pytest.fixture(scope="module")
def cap_s0():
    return _capture("eval_s0g")


@pytest.fixture(scope="module")
def cap_s1():
    return _capture("eval_s1")


def test_cross_check_terminal_heading_and_direction_class_are_exactly_the_routes(route, cap_s0):
    assert route.TAU_C == tc.TAU_DIR_RAD and route.STALL_M == tc.STALL_M          # the re-typed constants agree
    for name in ("fan", "traj", "gt"):
        P = cap_s0[name]
        mine, theirs = tc.terminal_heading(P), route.terminal_heading(P)
        assert mine.shape == theirs.shape and np.isfinite(mine).all()
        assert np.array_equal(mine, theirs), f"{name}: terminal heading differs from the route package"
        assert np.array_equal(tc.dir_class(mine), route.dir_class(theirs)), f"{name}: direction class differs"
    assert np.allclose(tc.path_length(cap_s0["gt"]), route.path_length(cap_s0["gt"]), atol=1e-9)


def test_cross_check_has_power_a_perturbed_tau_disagrees_with_the_route(route, cap_s0):
    th = tc.terminal_heading(cap_s0["fan"])
    other = tc.dir_class(th, tau=0.25)
    assert (other != route.dir_class(route.terminal_heading(cap_s0["fan"]))).sum() > 1000      # the comparison can fail


@pytest.mark.parametrize("which,n_dir,n_head", [("cap_s0", 90, 55), ("cap_s1", 91, 53)])
def test_cross_check_gt_classes_and_pick_counts_reproduce_the_route_result(route, request, which, n_dir, n_head):
    """Literals: route package RESULT.md §1.1 / raw/route_analysis.json (EVAL seed 0: 107 turns = 40 L + 67 R, 588 straight,
    105 gentle, 312 unclassified; E9 pick turn dir-correct 0.8411 = 90/107, heading-within-15 0.514 = 55/107;
    seed 1: 0.8505 = 91/107 and 0.4953 = 53/107)."""
    cap = request.getfixturevalue(which)
    cls_mine, th_mine = tc.gt_route_class(cap["gt"], cap["gt_valid"])
    cls_route, th_route = route.gt_class(cap["gt"], cap["gt_valid"])
    assert np.array_equal(cls_mine.astype(object), cls_route) and np.array_equal(th_mine, th_route)
    u, n = np.unique(cls_mine, return_counts=True)
    assert dict(zip(u.tolist(), n.tolist())) == {"turnL": 40, "turnR": 67, "straight": 588, "gentle": 105, "unclassified": 312}
    ar = np.arange(len(cap["sel_idx"]))
    pick = cap["fan"][ar, cap["sel_idx"]]
    assert np.array_equal(pick, cap["traj"])                                      # the emitted plan IS the selected candidate
    r = tc.route_following(pick, cap["gt"], cap["gt_valid"])
    t = r["summary"]["turn"]
    assert (t["n"], t["n_dir_correct"], t["n_head15"]) == (107, n_dir, n_head)
    assert r["bars"]["dir_correct_turn"]["meets"] is False and r["bars"]["head15_turn"]["meets"] is False
    # and the independent implementation agrees window by window
    dir_route = route.dir_class(route.terminal_heading(pick)) == route.dir_class(th_route)
    classified = cls_route != "unclassified"
    assert np.array_equal(np.nan_to_num(r["dir_ok"], nan=-1.0)[classified], dir_route[classified].astype(float))
