"""A2 behaviour-term unit tests on SYNTHETIC scenes (no map, no nuPlan DB): a straight route, constant-velocity objects, hand-built speed profiles.
Needs numpy + scipy + shapely + the nuplan/navsim import path (the driverl-eval venv):   python tests/run_tests.py
Each term has (i) a case that must read < 1, (ii) a control that must read EXACTLY 1.0 -- the control proves the probe can read 1 at all."""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
A2 = os.path.dirname(HERE)
TIP = os.environ.get("A2_TIP_REFE", "F:/a2_scratch/tip_refe")
NAVSIM = os.environ.get("A2_NAVSIM_ROOT", "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896")
for p in (NAVSIM, TIP, os.path.join(A2, "overlay", "refe")):
    if p not in sys.path:
        sys.path.insert(0, p)
shapely = pytest.importorskip("shapely")
pytest.importorskip("scipy")
pytest.importorskip("nuplan")
from shapely.geometry import LineString, Polygon          # noqa: E402
from shapely.ops import substring                          # noqa: E402
from shapely.strtree import STRtree                        # noqa: E402

import behaviour_terms as BT                               # noqa: E402

DT = 0.1
T = 41


class FakePath:
    def __init__(self, length=400.0):
        self.linestring = LineString([(0.0, 0.0), (length, 0.0)])
        self.length = length

    def project(self, pts):
        return self.linestring.project(pts)

    def substring(self, a, b):
        return substring(self.linestring, float(a), float(b))


class FakeMap:
    def __init__(self, geoms):
        self._tree = STRtree(geoms) if len(geoms) else None

    def query(self, g, predicate=None):
        return self._tree.query(g, predicate=predicate)


def box(cx, cy, l=4.5, w=2.0):
    return Polygon([(cx + l / 2, cy + w / 2), (cx - l / 2, cy + w / 2), (cx - l / 2, cy - w / 2), (cx + l / 2, cy - w / 2)])


def make_scene(objects, red_x=None):
    """objects: list of (token, x0, y0, vx, vy).  Constant velocity forecast at 0.2 s resolution as navsim's PDMObservation.  red_x: a red-light pseudo-object at x."""
    toks = [o[0] for o in objects]
    vel = {o[0]: ((o[3], o[4]) if (o[3] or o[4]) else None) for o in objects}
    if red_x is not None:
        toks.append("red_light_1")
    frames = []
    for i in range(52):
        t = (i // 2) * 2 * DT
        gs = [box(o[1] + o[3] * t, o[2] + o[4] * t) for o in objects]
        if red_x is not None:
            gs.append(box(red_x, 0.0, l=1.0, w=3.0))
        arr = np.empty(len(gs), dtype=object)
        for k, g in enumerate(gs):
            arr[k] = g
        frames.append((toks, arr, FakeMap(arr) if len(gs) else None))

    def occ(i):
        t, g, m = frames[i]
        if m is None:
            class _E:
                def query(self, *a, **k):
                    return np.array([[], []], dtype=int)
            return t, g, _E()
        return t, g, m
    return BT.Scene(FakePath(), occ, vel, "red_light", (), n_occ=52)


def states_from_speed(v):
    """v [M, T] -> sim-like states [M, T, 11] driving straight along +x from x=0 (rear axle)."""
    v = np.asarray(v, float)
    M = v.shape[0]
    s = np.zeros((M, T, 11))
    s[..., BT.X] = np.concatenate([np.zeros((M, 1)), np.cumsum(v[:, :-1] * DT, 1)], 1)
    s[..., BT.VX] = v
    s[..., BT.AX] = np.gradient(v, DT, axis=1)
    return s


class VP:   # vehicle parameters stub (Pacifica-like)
    half_length, half_width, rear_axle_to_center = 2.5, 1.1, 1.46


def run(v_rows, scene, ref=None, **kw):
    v_rows = np.asarray(v_rows, float)
    ref = np.asarray(ref if ref is not None else v_rows[0], float)[None]
    sim = states_from_speed(np.concatenate([ref, v_rows], 0))
    return BT.behaviour_terms(sim, None, scene=scene, vehicle_params=VP(), **kw)


const = lambda v: np.full(T, float(v))


# ---------------------------------------------------------------------------------------------- q_free
def test_q_free_penalises_phantom_braking_and_control_reads_one():
    sc = make_scene([])
    r = run([const(10.0), const(3.0), const(12.0), const(9.5)], sc, ref=const(10.0))
    assert r["act_free"].all()
    assert r["q_free"][0] == 1.0                           # control: as fast as the reference
    assert r["q_free"][2] == 1.0 and r["q_free"][3] == 1.0   # faster / within 10 %
    assert r["q_free"][1] == pytest.approx(0.3)            # 3 < 0.5 * 10 -> floor
    v5 = run([const(7.0)], sc, ref=const(10.0))["q_free"][0]
    assert 0.3 < v5 < 1.0 and v5 == pytest.approx(1 - 0.7 * (9.0 - 7.0) / 4.0)


def test_q_free_not_applicable_with_lead_red_light_or_stopped_reference():
    lead = make_scene([("car", 20.0, 0.0, 0.0, 0.0)])
    r = run([const(3.0)], lead, ref=const(10.0))
    assert not r["act_free"][0] and r["q_free"][0] == 1.0
    red = make_scene([], red_x=18.0)
    r = run([const(3.0)], red, ref=const(10.0))
    assert not r["act_free"][0]
    r = run([const(0.0)], make_scene([]), ref=const(0.0))
    assert not r["act_free"][0]                              # a stopped reference: nothing to be "phantom" about


def test_q_free_conflict_mover_entering_corridor_blocks_free_road():
    cross = make_scene([("ped", 20.0, -9.0, 0.0, 2.5)])      # enters |y|<1.5 after ~2.6 s at x=20: inside the 20 m corridor ahead
    r = run([const(3.0)], cross, ref=const(10.0))
    assert r["n_free_steps"][0] < 5 and not r["act_free"][0]


# ---------------------------------------------------------------------------------------------- q_gap
def test_q_gap_time_gap_bands():
    # lead moving at the ego's speed, ego at 10 m/s: time gap = bumper gap / 10
    def q_for(gap):
        # lead rear bumper `gap` m ahead of the ego front bumper (ego front = x + 3.96 rear-axle-to-front is VP-dependent; use the measured gap output instead)
        sc = make_scene([("lead", 4.0 + 1.0 + gap + 2.25, 0.0, 10.0, 0.0)])
        r = run([const(10.0)], sc, ref=const(10.0))
        return r["q_gap"][0], r["min_time_gap_s"][0], r["act_gap"][0]
    q20, tg20, a = q_for(20.0)
    assert a and 1.5 <= tg20 <= 2.5 and q20 == 1.0           # in-band control reads exactly 1
    q8, tg8, _ = q_for(7.0)
    assert tg8 < 1.0 and q8 < 0.5
    q_far, tg_far, _ = q_for(34.0)                           # lead beyond max(2.5 v, 20 m)=25 m -> NOT a lead at all
    assert not np.isfinite(tg_far) or tg_far > 3.0


def test_q_gap_piecewise_function_values():
    f = BT.q_gap_of
    assert f(np.array([20.0]), np.array([10.0]))[0] == 1.0       # tg 2.0
    assert f(np.array([15.0]), np.array([10.0]))[0] == 1.0       # tg 1.5 boundary
    assert f(np.array([10.0]), np.array([10.0]))[0] == pytest.approx(0.5)   # tg 1.0
    assert f(np.array([7.0]), np.array([10.0]))[0] == pytest.approx(0.2)    # tg 0.7
    assert f(np.array([35.0]), np.array([10.0]))[0] == pytest.approx(0.5)   # tg 3.5
    assert f(np.array([50.0]), np.array([10.0]))[0] == pytest.approx(0.5)   # beyond
    assert f(np.array([0.5]), np.array([0.2]))[0] == 0.2         # v < 1: gap < 1 m
    assert f(np.array([3.0]), np.array([0.2]))[0] == 1.0


# ---------------------------------------------------------------------------------------------- q_osc
def test_q_osc_fires_on_oscillation_not_on_smooth_or_gentle():
    tt = np.arange(T) * DT
    osc = 8.0 + 1.5 * np.sin(2 * np.pi * tt / 1.6)           # a amplitude 1.5*2pi/1.6 = 5.9 m/s^2 ... large; flips ~5
    gentle = 8.0 + 0.1 * np.sin(2 * np.pi * tt / 1.6)        # a amplitude 0.39 < 0.5 deadband
    ramp = 4.0 + 1.0 * tt                                     # one-signed accel
    brake = 10.0 - 1.0 * tt                                   # one-signed decel
    r = run([osc, gentle, ramp, brake], make_scene([]), ref=const(8.0))
    assert r["q_osc"][0] == 0.8 and r["osc_flips"][0] > 2
    assert list(r["q_osc"][1:]) == [1.0, 1.0, 1.0]            # controls read exactly 1


# ---------------------------------------------------------------------------------------------- q_stall
def test_q_stall_fires_only_when_clear():
    stopped = const(0.0)
    clear = run([stopped], make_scene([]), ref=const(8.0))
    assert clear["q_stall"][0] == 0.5 and clear["stall_run_s"][0] == pytest.approx(4.1)
    moving = run([const(5.0)], make_scene([]), ref=const(8.0))
    assert moving["q_stall"][0] == 1.0                                   # control
    red = run([stopped], make_scene([], red_x=12.0), ref=const(8.0))
    assert red["q_stall"][0] == 1.0                                      # red light in the corridor: a stop is correct
    lead = run([stopped], make_scene([("car", 12.0, 0.0, 0.0, 0.0)]), ref=const(8.0))
    assert lead["q_stall"][0] == 1.0                                     # stopped behind a stopped lead
    cross = run([stopped], make_scene([("ped", 8.0, -4.0, 0.0, 2.0)]), ref=const(8.0))
    assert cross["q_stall"][0] == 1.0                                    # a mover is about to enter the corridor: not clear
    short = const(0.0).copy()
    short[20:] = 5.0                                                     # stopped for 2.0 s only (20 samples, not > 2 s)
    assert run([short], make_scene([]), ref=const(8.0))["q_stall"][0] == 1.0


# ---------------------------------------------------------------------------------------------- q_comf
def test_q_comf_matches_navsim_c_and_pilot_noise_artefact():
    smooth = const(10.0)
    harsh = np.where(np.arange(T) < 20, 10.0, 10.0 - 7.0 * (np.arange(T) - 19) * DT).clip(0.0)     # ~ -7 m/s^2 braking: lon_acc violation
    r = run([smooth, harsh], make_scene([]), ref=smooth)
    assert r["q_comf"][0] == 1.0 and not r["comf_viol"][0].any()
    assert r["comf_viol"][1][0] and r["q_comf"][1] <= 0.7 + 1e-12
    # the artefact the pilot's detector has: 10 Hz tracker noise on the acceleration channel
    sim = states_from_speed(np.stack([smooth, smooth]))
    rng = np.random.default_rng(0)
    sim[..., BT.AX] += rng.normal(0.0, 0.5, sim[..., BT.AX].shape)
    navsim_v = BT.comfort_violations(sim[1:])
    pilot_v = BT.pilot_style_comfort(sim[1:])
    assert not navsim_v.any()                                            # NAVSIM's filtered metric: comfortable
    assert pilot_v[0, 3]                                                  # the pilot's raw-difference lon_jerk channel fires on pure noise


# ---------------------------------------------------------------------------------------------- structural properties
def test_rows_are_independent_chunk_invariant_and_deterministic():
    sc = make_scene([("car", 60.0, 0.0, 6.0, 0.0)])
    rows = [const(10.0), const(6.0), const(0.0), 8 + np.sin(np.arange(T) * 0.4), 4.0 + 0.8 * np.arange(T) * DT]
    full = run(rows, sc, ref=const(9.0))
    again = run(rows, sc, ref=const(9.0))
    for k in BT.ORDER:
        assert np.array_equal(full[k], again[k])                          # deterministic
    for sl in (slice(0, 2), slice(2, 5), [4, 0, 3]):
        sub = run([rows[i] for i in (range(5)[sl] if isinstance(sl, slice) else sl)], sc, ref=const(9.0))
        idx = list(range(5)[sl]) if isinstance(sl, slice) else sl
        for k in BT.ORDER:
            assert np.array_equal(full[k][idx], sub[k]), k                # chunking / permuting proposals changes nothing
    q, a = BT.pack(full)
    assert q.shape == (5, 5) and a.shape == (5, 5) and q.dtype == np.float32 and a.dtype == bool
    assert np.all((q > 0) & (q <= 1.0))


def test_route_area_filter_ignores_objects_whose_centre_is_off_the_route_lane():
    lane = box(100.0, 0.0, l=400.0, w=3.0)                               # the route lane: |y| <= 1.5
    parked = make_scene([("parked", 14.0, 1.2, 0.0, 0.0)])                # its box (y 0.2..2.2) grazes the 1.5 m corridor; its CENTRE (y=1.2) is inside |y|<=1.5 -> still a lead
    parked.route_area = lane
    assert run([const(8.0)], parked, ref=const(8.0))["act_gap"][0]
    beside = make_scene([("beside", 14.0, 1.9, 0.0, 0.0)])                # centre y=1.9 > 1.5: off the lane, box (y 0.9..2.9) still touches the corridor
    assert run([const(8.0)], beside, ref=const(8.0))["act_gap"][0]        # unfiltered: read as a lead
    beside.route_area = lane
    r = run([const(8.0)], beside, ref=const(8.0))
    assert not r["act_gap"][0] and r["q_gap"][0] == 1.0                   # filtered: ignored
    lead = make_scene([("lead", 30.0, 0.0, 0.0, 0.0)])
    lead.route_area = lane
    assert run([const(8.0)], lead, ref=const(8.0))["act_gap"][0]          # a car IN the lane is still a lead with the filter on (control)
