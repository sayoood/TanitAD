"""refcv6's NavSim inputs, each against a LITERAL or ANALYTIC target, each with an arm that must
go RED. TANITAD VENV:  pytest -q tests/test_inputs6.py

Expectations are written as literals, never as an expression over the code under test
(CLAUDE.md: "a check that shares the defect it checks for is green forever").
"""
from __future__ import annotations

import copy
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import refcv6_bridge as R6  # noqa: E402
import torch  # noqa: E402
from tanitad.models.ego_history import ego_channels_from_poses  # noqa: E402

TIMES = [-1.5, -1.0, -0.5, 0.0]


def _es(x, y, h, vx, vy=0.0, ax=0.0, ay=0.0, cmd=(0, 1, 0, 0)):
    return {"ego_pose": [x, y, h], "ego_velocity": [vx, vy], "ego_acceleration": [ax, ay],
            "driving_command": list(cmd)}


def _const_accel_statuses(a=2.0, v_t0=10.0):
    """A straight constant-acceleration history: v(t) = v_t0 + a t, x(t) = v_t0 t + a t^2 / 2."""
    out = []
    for t in TIMES:
        out.append(_es(v_t0 * t + 0.5 * a * t * t, 0.0, 0.0, v_t0 + a * t))
    return out


def _channels(hist):
    return ego_channels_from_poses(torch.from_numpy(hist)[None], 8, dt=0.1)[0].numpy()


# --------------------------------------------------------------------------- #
# ego history (10 Hz from 2 Hz)                                                #
# --------------------------------------------------------------------------- #
def test_hist_constant_speed_straight():
    st = [_es(12.0 * t, 0.0, 0.0, 12.0) for t in TIMES]
    h = R6.ego_history_poses(R6.declare6(st, "R6_A1"), TIMES)
    assert h.shape == (8, 4)
    np.testing.assert_allclose(h[:, 3], 12.0, atol=1e-6)
    ch = _channels(h)
    np.testing.assert_allclose(ch[1:, 1], 0.0, atol=1e-4)      # dv/dt
    np.testing.assert_allclose(ch[1:, 2], 0.0, atol=1e-6)      # dyaw/dt
    np.testing.assert_allclose(h[:, 0], [-8.4, -7.2, -6.0, -4.8, -3.6, -2.4, -1.2, 0.0], atol=1e-5)


def test_hist_constant_acceleration_reads_2_mps2():
    h = R6.ego_history_poses(R6.declare6(_const_accel_statuses(2.0, 10.0), "R6_A1"), TIMES)
    np.testing.assert_allclose(h[:, 3], [8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0], atol=1e-5)
    ch = _channels(h)
    np.testing.assert_allclose(ch[1:, 1], 2.0, atol=1e-3)       # LITERAL: the history's accel


def test_hist_constant_yaw_rate():
    st = [_es(0.0, 0.0, -0.2 * t, 8.0) for t in TIMES]           # yaw(t) = -0.2 t -> +0.2 at -1 s
    h = R6.ego_history_poses(R6.declare6(st, "R6_A1"), TIMES)
    ch = _channels(h)
    np.testing.assert_allclose(ch[1:, 2], -0.2, atol=1e-4)


def test_hist_yaw_unwrap_across_pi():
    """A heading crossing +-pi must not read as a ~60 rad/s yaw rate."""
    yaws = [3.10, 3.13, -3.13, -3.10]                               # +0.06 rad per 0.5 s
    st = [_es(0.0, 0.0, y, 8.0) for y in yaws]
    ch = _channels(R6.ego_history_poses(R6.declare6(st, "R6_A1"), TIMES))
    assert np.all(np.abs(ch[1:, 2]) < 0.2), ch[:, 2]


def test_hist_refuses_non_2hz_times():
    st = _const_accel_statuses()
    with pytest.raises(R6.RefusedInput):
        R6.ego_history_poses(R6.declare6(st, "R6_A1"), [-0.3, -0.2, -0.1, 0.0])


@pytest.mark.parametrize("times", [[-2.0, -1.5, -0.5, 0.0], [-2.0, -1.5, -1.0, 0.0],
                                   [-2.0, -1.0, -0.5, 0.0]])
def test_hist_dropped_frame_is_interpolated_at_actual_times(times):
    """SPEC amendment A1: a log that dropped a 2 Hz frame (navhard 2 / navtest 15 scenes) — the
    linear history is still read exactly (linear interpolation of a linear v(t) is exact at any
    spacing), so dv/dt stays the literal 2.0 m/s^2."""
    st = [_es(10.0 * t + t * t, 0.0, 0.0, 10.0 + 2.0 * t) for t in times]
    h = R6.ego_history_poses(R6.declare6(st, "R6_A1"), times)
    np.testing.assert_allclose(h[:, 3], [8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0], atol=1e-5)
    np.testing.assert_allclose(_channels(h)[1:, 1], 2.0, atol=1e-3)


@pytest.mark.parametrize("times", [[-3.0, -2.5, -1.2, 0.0],      # a 1.2 s gap
                                   [-1.0, -0.6, -0.3, 0.0],      # oldest used state after -0.7
                                   [-1.5, -1.0, -0.5, 0.1]])     # t0 != 0
def test_hist_refuses_gaps_extrapolation_and_bad_t0(times):
    st = _const_accel_statuses()
    with pytest.raises(R6.RefusedInput):
        R6.ego_history_poses(R6.declare6(st, "R6_A1"), times)


def test_REGRESSION_hist_on_the_wrong_grid_goes_red():
    """The defect this guards against: treating the 2 Hz samples as consecutive 10 Hz samples
    (placing t0-1.0 / -0.5 / 0 at the last three 0.1 s slots). The channel would read 10 m/s^2
    for a 2 m/s^2 history — this arm computes exactly that and must NOT pass the literal."""
    decl = R6.declare6(_const_accel_statuses(2.0, 10.0), "R6_A1")
    v = [R6.speed_of(decl[f"ego_velocity[{i}]"]) for i in (1, 2, 3)]
    wrong = np.zeros((8, 4), np.float32)
    wrong[:, 3] = v[0]
    wrong[-3:, 3] = v                                     # 9, 10 ... at 0.1 s spacing
    ch = _channels(wrong)
    assert not np.allclose(ch[-2:, 1], 2.0, atol=1e-3), "the regression arm failed to go red"


# --------------------------------------------------------------------------- #
# the declared-input seam (gate navsim.ego_enforcement)                        #
# --------------------------------------------------------------------------- #
def _inputs(st, arm):
    d = R6.declare6(st, arm)
    return (R6.ego_history_poses(d, TIMES).tobytes(), R6.v0_of(d), R6.nav_input(d, arm)["nav_index"])


@pytest.mark.parametrize("arm", ["R6_A1", "R6_NAVOFF", "R6_VMAXOFF", "R6_BLIND"])
def test_undeclared_fields_are_invisible(arm):
    rng = np.random.default_rng(0)
    base = _const_accel_statuses()
    ref = _inputs(base, arm)
    decl = set(R6.ARMS6[arm]["declared"])
    for _ in range(20):
        st = copy.deepcopy(base)
        for i in range(4):
            for f in R6.FIELDS:
                if f"{f}[{i}]" in decl:
                    continue
                n = len(st[i][f])
                st[i][f] = (list(np.eye(4)[rng.integers(4)]) if f == "driving_command"
                            else list(rng.normal(size=n) * 50))
        assert _inputs(st, arm) == ref


def test_REGRESSION_declared_field_mutation_changes_inputs():
    base = _const_accel_statuses()
    st = copy.deepcopy(base)
    st[3]["ego_velocity"] = [15.0, 0.0]
    assert _inputs(st, "R6_A1") != _inputs(base, "R6_A1")
    st2 = copy.deepcopy(base)
    st2[3]["driving_command"] = [1, 0, 0, 0]
    assert _inputs(st2, "R6_A1") != _inputs(base, "R6_A1")      # nav IS read by A1 ...
    assert _inputs(st2, "R6_NAVOFF") == _inputs(base, "R6_NAVOFF")  # ... and NOT by NAVOFF


# --------------------------------------------------------------------------- #
# nav: NavSim one-hot -> the v7 token (refb.NAV_COMMANDS = follow, left, right, straight)     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cmd,name,idx", [((1, 0, 0, 0), "left", 1), ((0, 1, 0, 0), "follow", 0),
                                          ((0, 0, 1, 0), "right", 2), ((0, 0, 0, 1), "follow", 0)])
def test_nav_map_literal(cmd, name, idx):
    st = _const_accel_statuses()
    st[3]["driving_command"] = list(cmd)
    nav = R6.nav_input(R6.declare6(st, "R6_A1"), "R6_A1")
    assert (nav["name"], nav["nav_index"]) == (name, idx)


def test_nav_refuses_non_one_hot():
    st = _const_accel_statuses()
    st[3]["driving_command"] = [0.5, 0.5, 0, 0]
    with pytest.raises(R6.RefusedInput):
        R6.nav_input(R6.declare6(st, "R6_A1"), "R6_A1")


def test_nav_indices_match_the_trainer_vocabulary():
    """The v7 token the model trained on lands on refb.NAV_COMMANDS rows via
    NAV_TOKEN_TO_LEGACY; the NavSim map must hit the same rows (literal)."""
    from tanitad.refs import refb
    assert tuple(refb.NAV_COMMANDS)[:3] == ("follow", "left", "right")


# --------------------------------------------------------------------------- #
# max speed: posted limit -> the model's containing-window one-hot (30/50/100/120 km/h)       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mps,bin_kmh", [(11.17568, 50),     # 25 mph (the PIT warmup value)
                                         (15.64579, 100),    # 35 mph
                                         (6.70561, 30),      # 15 mph
                                         (20.11689, 100),    # 45 mph
                                         (13.88, 50)])       # just under 50 km/h
def test_vmax_bins_literal(mps, bin_kmh):
    r = R6.max_speed_input({"status": "limit", "speed_limit_mps": mps}, "R6_A1")
    assert r["v_max_valid"] == 1.0 and r["bin_kmh"] == bin_kmh and r["v_max_ms"] == mps


@pytest.mark.parametrize("rec", [None, {"status": "no_limit", "speed_limit_mps": None},
                                 {"status": "none", "speed_limit_mps": None}])
def test_vmax_unknown_is_valid_zero(rec):
    r = R6.max_speed_input(rec, "R6_A1")
    assert (r["v_max_ms"], r["v_max_valid"]) == (0.0, 0.0)


def test_vmax_withheld_arm_ignores_the_map():
    r = R6.max_speed_input({"status": "limit", "speed_limit_mps": 11.17568}, "R6_VMAXOFF")
    assert (r["v_max_ms"], r["v_max_valid"]) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# SPEC §12 (A4): the max-speed ORACLE — the channel's TRAINING definition on the 2 Hz log      #
# --------------------------------------------------------------------------- #

def _log(speeds_by_offset, t0_us=1_000_000_000):
    """Synthetic log frames: {offset_s: speed} -> frames with ego_dynamic_state [v, 0, 0, 0]."""
    return [{"timestamp": t0_us + int(round(o * 1e6)), "ego_dynamic_state": [v, 0.0, 0.0, 0.0]}
            for o, v in sorted(speeds_by_offset.items())]


#: in-window speeds 5..9..5 (max 9.0 at t0+4.0 s) and a 30 m/s SPIKE just OUTSIDE on each side
_ORACLE_LOG = {**{-1.0: 3.0, -0.5: 3.0, 0.0: 3.0, 0.5: 3.0, 1.0: 3.0, 1.5: 30.0},
               **dict(zip([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                          [5.0, 6.0, 7.0, 8.0, 9.0, 8.0, 7.0, 6.0, 5.0])),
               6.5: 30.0, 7.0: 3.0}


def test_oracle_is_the_max_over_t0_plus_2_to_6_literal():
    import vmax_oracle as VO
    r = VO.oracle_from_frames(1_000_000_000, _log(_ORACLE_LOG))
    assert r["status"] == "limit" and r["speed_limit_mps"] == 9.0 and r["n_samples"] == 9


def test_REGRESSION_oracle_with_the_wrong_window_goes_red(monkeypatch):
    """The deliberate regression: a window that starts at t0 (not t0+2 s) reads the 1.5 s spike —
    the literal above (9.0) must then FAIL."""
    import vmax_oracle as VO
    monkeypatch.setattr(VO, "WIN_LO_S", 0.0)
    r = VO.oracle_from_frames(1_000_000_000, _log(_ORACLE_LOG))
    assert r["speed_limit_mps"] != 9.0 and r["speed_limit_mps"] == 30.0


@pytest.mark.parametrize("drop", [[5.5, 6.0], [2.0], [3.0, 3.5, 4.0]])
def test_oracle_refuses_an_uncovered_window(drop):
    """end of log (no t0+6 s), a missing first sample, or a gap > 1.0 s -> no_future (valid 0)."""
    import vmax_oracle as VO
    lg = {k: v for k, v in _ORACLE_LOG.items() if k not in drop and k not in (6.5, 7.0)}
    if drop == [5.5, 6.0]:
        lg = {k: v for k, v in lg.items() if k <= 5.0}
    r = VO.oracle_from_frames(1_000_000_000, _log(lg))
    assert r["status"] == "no_future"
    assert R6.max_speed_input(r, "R6_VMAXORACLE")["v_max_valid"] == 0.0


@pytest.mark.parametrize("mps,bin_kmh", [(12.0, 50), (0.0, 30), (27.0, 100), (8.3, 30)])
def test_oracle_arm_bins_literal(mps, bin_kmh):
    """12 m/s = 43.2 km/h -> (30, 50]; a human who never moved (0.0) -> the lowest bin, VALID
    (training saw exactly that, never 'unknown'); 27 m/s = 97.2 km/h -> 100; 8.3 m/s = 29.9 -> 30."""
    rec = {"status": "limit", "speed_limit_mps": mps, "why": "ORACLE: test"}
    r = R6.max_speed_input(rec, "R6_VMAXORACLE")
    assert r["v_max_valid"] == 1.0 and r["bin_kmh"] == bin_kmh and r["v_max_ms"] == mps


def test_oracle_arm_refuses_a_map_record():
    """The privileged arm must be fed the ORACLE file, never the map's (a silent swap would make
    ORACLE - A1 read exactly 0 and 'price' the mismatch at nothing)."""
    with pytest.raises(R6.RefusedInput):
        R6.max_speed_input({"status": "limit", "speed_limit_mps": 11.17568}, "R6_VMAXORACLE")


def test_oracle_arm_shares_A1s_seed_and_declaration():
    """Common random numbers: ORACLE - A1 must be a pure input difference."""
    a, o = R6.ARMS6["R6_A1"], R6.ARMS6["R6_VMAXORACLE"]
    assert (o["seed"], o["frames"], o["nav"], o["declared"]) == (a["seed"], a["frames"], a["nav"],
                                                                 a["declared"])
    assert o["vmax"] == "oracle" and a["vmax"] == "map"


def test_vmax_unknown_is_an_all_zero_one_hot_in_the_model_encoder():
    """The model side (ONE ladder): valid=0 -> all-zero row, never bin 0 = 30 km/h."""
    from tanitad.refs.refcv6_max_speed import MaxSpeedOneHotEncoder
    enc = MaxSpeedOneHotEncoder()
    oh, _ = enc(torch.tensor([11.17568, 0.0]), torch.tensor([1.0, 0.0]))
    np.testing.assert_array_equal(oh.numpy(), [[0, 1, 0, 0], [0, 0, 0, 0]])


# --------------------------------------------------------------------------- #
# the 6 s -> 4 s @ 0.5 s conversion (E2's knots_to_navsim, imported): analytic targets         #
# --------------------------------------------------------------------------- #
def test_conversion_constant_velocity_line():
    v = 9.0
    knots = np.array([[v * t, 0.0] for t in R6.KNOT_T_S])
    p = R6.knots_to_navsim(knots)
    np.testing.assert_allclose(p[:, 0], [v * t for t in R6.NAVSIM_T_S], atol=1e-9)
    np.testing.assert_allclose(p[:, 1:], 0.0, atol=1e-9)


def test_conversion_circle_arc_heading():
    """A left arc of radius R at speed v: position on the circle at the knots (exact) and the
    interpolated 2.5 / 3.5 s within 5 cm; heading = v t / R (the tangent) within 0.01 rad."""
    R, v = 40.0, 8.0

    def pt(t):
        th = v * t / R
        return [R * math.sin(th), R * (1 - math.cos(th))]
    p = R6.knots_to_navsim(np.array([pt(t) for t in R6.KNOT_T_S]))
    want = np.array([pt(t) for t in R6.NAVSIM_T_S])
    np.testing.assert_allclose(p[:, :2], want, atol=0.05)
    np.testing.assert_allclose(p[:, 2], [v * t / R for t in R6.NAVSIM_T_S], atol=0.01)


def test_REGRESSION_conversion_with_wrong_knot_times_goes_red():
    """If the knots were read as UNIFORM 0.5 s samples (index-based, the 6 s horizon squeezed
    into 4 s), the constant-velocity line would come out at the wrong positions."""
    v = 9.0
    knots = np.array([[v * t, 0.0] for t in R6.KNOT_T_S])
    wrong = R6.B2.knots_to_navsim(knots, knot_t=R6.NAVSIM_T_S)
    assert not np.allclose(wrong[:, 0], [v * t for t in R6.NAVSIM_T_S], atol=1e-3)


def test_knot_times_are_refcv6s_horizons():
    assert R6.KNOT_T_S == (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert R6.NAVSIM_T_S == (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    assert R6.HIST_T_S == pytest.approx((-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0))
