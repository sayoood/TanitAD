"""The S1 collision-gate harness core (``taniteval/tools/s1_gate.py``), against ANALYTIC targets.

PREREG_S1 AMENDMENT + ERRATA (2026-09-19): every arm is a selection rule over ONE fan, gated by
the IMPORTED item-19 checker (``tanitad.rl.pdm_proxy``) and SCORED against the RECORDED
future. Each expectation here is a literal or a closed form, never the code under test
re-run; every guard was also shown RED under mutation (``stack/scripts/mutate_s1_gate.py``).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "taniteval" / "tools"))

import s1_gate as S                                       # noqa: E402
from tanitad.rl import pdm_proxy as P                    # noqa: E402

H = (5, 10, 15, 20, 30, 40, 50, 60)       # A8's stamped horizons (config.json), ticks @ 10 Hz
DT = P.PROXY.dt
N_T = P.PROXY.n_ticks


def _slots(fn):
    """Waypoints at the horizons from a closed-form path fn(t) -> (x, y)."""
    return np.array([fn(h * DT) for h in H], dtype=np.float64)[None]


# ------------------------------------------------------------------ the spline (S1A.2)
def test_spline_STRAIGHT_constant_velocity_is_EXACT():
    """x = v0·t, y = 0 meets every knot, the clamped start slope and the natural end, so the
    spline IS the line: exact, not approximate."""
    v0 = 10.0
    x, y, yaw, v = S.expand_waypoints(_slots(lambda t: (v0 * t, 0.0)), H, v0, N_T, DT)
    t = np.arange(N_T + 1) * DT
    np.testing.assert_allclose(x[0], v0 * t, atol=1e-9)
    np.testing.assert_allclose(y[0], 0.0, atol=1e-9)
    np.testing.assert_allclose(yaw[0], 0.0, atol=1e-9)
    np.testing.assert_allclose(v[0], v0, atol=1e-9)


def test_spline_CIRCLE_yaw_rate_is_v_over_R():
    """A circle of radius R at speed v turns at exactly v/R rad/s. Interior accuracy only; the
    natural END condition is not a circle's, so 1 % is the stated tolerance."""
    v, R = 10.0, 50.0
    w = v / R
    circ = lambda t: (R * math.sin(w * t), R * (1 - math.cos(w * t)))   # noqa: E731
    x, y, yaw, sp = S.expand_waypoints(_slots(circ), H, v, N_T, DT)
    k = 20                                                               # 2 s
    assert yaw[0, k] == pytest.approx(w * k * DT, rel=0.01)
    assert sp[0, k] == pytest.approx(v, rel=0.01)


def test_spline_REFUSES_to_extrapolate_past_its_last_knot():
    with pytest.raises(ValueError, match="extrapolating"):
        S.expand_waypoints(np.zeros((1, 4, 2)), (5, 10, 15, 20), 5.0, N_T, DT)


def test_candidate_states_equal_the_HUMAN_constructor_on_the_same_path():
    """S1A.2's promise: candidate and human share ONE construction. A straight world path
    scored both ways must agree, including the tick-0 ego state (0, 0, 0, v0)."""
    x0, y0, h0, v0 = 100.0, 50.0, 0.7, 10.0
    pl = torch.tensor([x0, y0, h0, v0])
    st = S.candidate_states(_slots(lambda t: (v0 * t, 0.0)), H, pl)
    t = torch.arange(1, N_T + 1, dtype=torch.float64) * DT
    fut = torch.stack([x0 + v0 * t * math.cos(h0), y0 + v0 * t * math.sin(h0),
                       torch.full_like(t, h0), torch.full_like(t, v0)], dim=-1).float()
    human = P.ego_states_from_poses(pl[None], fut[None])[0]
    torch.testing.assert_close(st[0], human, atol=2e-4, rtol=0)
    torch.testing.assert_close(st[0, 0], torch.tensor([0.0, 0.0, 0.0, v0]), atol=1e-5, rtol=0)


def test_HUMAN_ROUND_TRIP_on_smooth_paths():
    """The synthetic twin of the S1A.6 control: a smooth human path, cut to the 8 slots and
    re-expanded, must stay within 5 cm over the 4 s the proxy scores."""
    for fn, v0 in ((lambda t: (8 * t + 0.5 * t * t, 0.0), 8.0),
                   (lambda t: (12 * math.sin(0.1 * t) / 0.1, 12 * (1 - math.cos(0.1 * t)) / 0.1), 12.0)):
        x, y, _, _ = S.expand_waypoints(_slots(fn), H, v0, N_T, DT)
        t = np.arange(N_T + 1) * DT
        ref = np.array([fn(tt) for tt in t])
        err = np.hypot(x[0] - ref[:, 0], y[0] - ref[:, 1]).max()
        assert err < 0.05, err


# ------------------------------------------------------------------ tracks (S1A.4/.5)
POSE = torch.tensor([10.0, -4.0, 0.3, 12.0])


def test_from_frames_is_the_IDENTITY_under_a_constant_ego_pose():
    a = {"track_id": 1, "cx": 20.0, "cy": 3.0, "yaw": 0.2, "l": 4.5, "w": 1.9,
         "cls": "vehicle", "vx": 5.0, "vy": -1.0}
    tr = S._tracks_from_t0_agents([a], POSE, 10)
    k = torch.arange(10, dtype=torch.float32)
    torch.testing.assert_close(tr.xy[:, 0, 0], 20.0 + 5.0 * k * DT, atol=1e-4, rtol=0)
    torch.testing.assert_close(tr.xy[:, 0, 1], 3.0 - 1.0 * k * DT, atol=1e-4, rtol=0)
    assert tr.speed[3, 0].item() == pytest.approx(math.hypot(5.0, 1.0), rel=1e-4)
    assert not bool(tr.static[0])
    st = S._tracks_from_t0_agents([dict(a, cls=P.PROXY.static_classes[0])], POSE, 10)
    assert bool(st.static[0]), "static is decided by CLASS (pdm_proxy.py:246)"


def test_EMPTY_tracks_have_no_valid_agent():
    tr = S.empty_tracks(POSE)
    assert tr.xy.shape[0] == S.AGENT_TICKS and not bool(tr.valid.any())


def test_ORACLE_CV_reproduces_a_constant_velocity_recorded_track_EXACTLY():
    """Known value: an agent moving at constant WORLD velocity is also constant-velocity in the
    fixed t0 frame, so ORACLE-CV must give back the recorded track, even though the recorded
    frames were written from a MOVING and TURNING ego."""
    T = 12
    ego = torch.zeros(T, 4)
    for k in range(T):                                   # ego drives and turns
        ego[k] = torch.tensor([3.0 * k * DT * 10, 0.5 * k * DT, 0.02 * k, 30.0])
    aw = lambda k: (40.0 + 7.0 * k * DT, 5.0 - 2.0 * k * DT)   # noqa: E731  world agent
    frames = []
    for k in range(T):                                   # world -> ego@k, as the join writes
        wx, wy = aw(k)
        dx, dy = wx - float(ego[k, 0]), wy - float(ego[k, 1])
        c, s = math.cos(-float(ego[k, 2])), math.sin(-float(ego[k, 2]))
        frames.append([{"track_id": "a", "cx": dx * c - dy * s, "cy": dx * s + dy * c,
                        "yaw": 0.1 - float(ego[k, 2]), "l": 4.0, "w": 2.0, "cls": "vehicle"}])
    rec = P.AgentTracks.from_frames(frames, ego)
    cv = S.cv_tracks_from_recorded(rec, ego[0])
    torch.testing.assert_close(cv.xy, rec.xy, atol=2e-3, rtol=0)
    assert bool(cv.static[0]) == bool(rec.static[0])


def test_PRED_agents_threshold_classes_and_add_v0():
    dec = {"presence_logit": torch.tensor([3.0, -3.0]),
           "cls_logits": torch.tensor([[0.0, 5.0], [5.0, 0.0]]),
           "box": torch.tensor([[25.0, 1.0, 4.0, 2.0], [9.0, 9.0, 1.0, 1.0]]),
           "yaw": torch.tensor([0.1, 0.0]),
           "rates": torch.tensor([[-4.0, 0.5, 0.0], [0.0, 0.0, 0.0]])}
    ag = S.pred_agents_from_decoded(dec, v0=12.0, classes=("vehicle", "pedestrian"))
    assert len(ag) == 1, "sigmoid(-3) = 0.047 < 0.5 must be dropped"
    assert ag[0]["cls"] == "pedestrian"
    assert ag[0]["vx"] == pytest.approx(8.0) and ag[0]["vy"] == pytest.approx(0.5)


# ------------------------------------------------------------------ the gate (S1A.2/.6)
def _scene():
    """Two candidates, both at 10 m/s. A drives straight into a STOPPED box 30 m ahead (at
    fault); B swerves 6 m left of it. The model prefers A."""
    pl = torch.tensor([0.0, 0.0, 0.0, 10.0])
    straight = _slots(lambda t: (10 * t, 0.0))[0]
    left = np.array([(10 * h * DT, 6.0 * min(h * DT / 2.0, 1.0) ** 2) for h in H])
    states = S.candidate_states(np.stack([straight, left]), H, pl)
    box = S._tracks_from_t0_agents([{"track_id": 0, "cx": 30.0, "cy": 0.0, "yaw": 0.0,
                                     "l": 4.0, "w": 2.0, "cls": "vehicle",
                                     "vx": 0.0, "vy": 0.0}], pl)
    rank = torch.tensor([2.0, 1.0])
    human = states[1]
    route = human[:, :2]
    return pl, states, box, rank, human, route


def test_the_checker_itself_reads_the_collision():
    pl, states, box, *_ = _scene()
    nc = P.no_at_fault_collision(states, box)
    assert float(nc[0]) == 0.0 and float(nc[1]) == 1.0


def test_GATE_takes_the_free_candidate_and_CONST_changes_NOTHING():
    pl, states, box, rank, human, route = _scene()
    m = S.evaluate_window(states, rank, None, human, route, box,
                          {"GATE_ORACLE": box, "GATE_CONST": S.empty_tracks(pl)})
    assert m["BASE"]["idx"] == 0 and m["BASE"]["collided"] == 1.0
    assert m["GATE_ORACLE"]["idx"] == 1 and m["GATE_ORACLE"]["collided"] == 0.0, (
        "the gate did not re-order under REAL tracks: CONST's zero would be a dead gate")
    assert m["GATE_CONST"]["idx"] == m["BASE"]["idx"], "CONST must recover EXACTLY nothing"
    assert m["_fan"]["collision_free_share"] == 0.5 and m["_fan"]["any_free"]


def test_every_pick_is_SCORED_against_the_RECORDED_future():
    """PRED's tracks miss the box, so PRED keeps A. It must still be scored as COLLIDED,
    because scoring reads the recorded future, never the gate's own tracks."""
    pl, states, box, rank, human, route = _scene()
    m = S.evaluate_window(states, rank, None, human, route, box,
                          {"GATE_PRED": S.empty_tracks(pl)})
    assert m["GATE_PRED"]["idx"] == 0 and m["GATE_PRED"]["collided"] == 1.0


def test_all_blocked_falls_back_to_BASE():
    idx, fb = S.select(torch.tensor([1.0, 3.0, 2.0]), None, torch.tensor([False] * 3))
    assert (idx, fb) == (1, True)


def test_the_MODEL_mask_is_honoured_before_the_gate():
    rank = torch.tensor([9.0, 1.0, 5.0])
    keep = torch.tensor([False, True, True])             # the model's own reach mask
    assert S.select(rank, keep, None) == (2, False)
    assert S.select(rank, keep, torch.tensor([True, True, False])) == (1, False)


def test_FAMILIES_read_known_values():
    """Along/cross are read in the HUMAN's frame. A candidate 1 m to the human's LEFT and 2 m
    AHEAD, both heading along +y (yaw pi/2), reads along = 2, cross = 1 exactly."""
    T = N_T + 1
    h = torch.zeros(T, 4)
    h[:, 1] = torch.arange(T) * DT * 10.0           # the human drives along +y
    h[:, 2] = math.pi / 2
    h[:, 3] = 10.0
    c = h.clone()
    c[:, 0] -= 1.0                                   # +y heading => LEFT is -x
    c[:, 1] += 2.0
    c[:, 3] = 12.0
    f = S.pick_families(c, h)
    assert f["along_2s"] == pytest.approx(2.0, abs=1e-5)
    assert f["cross_2s"] == pytest.approx(1.0, abs=1e-5)
    assert f["speed_err_4s"] == pytest.approx(2.0, abs=1e-5)
    assert f["heading_err_4s"] == pytest.approx(0.0, abs=1e-6)
    assert f["curv_err_2s"] is None, "a straight human path is MASKED, not scored as 0"
    g = S.pick_families(h, h)
    assert g["along_4s"] == 0.0 and g["cross_4s"] == 0.0 and g["speed_err_4s"] == 0.0


@pytest.mark.parametrize("n", [5, 128])
def test_RANDOM_is_the_EXACT_fan_mean_over_N_read_from_the_fan(n):
    g = torch.Generator().manual_seed(n)
    true = {k: torch.rand(n, generator=g) for k in ("dac", "ep", "ttc", "comfort", "pdms")}
    true["nc"] = (torch.rand(n, generator=g) > 0.3).float()
    m = S.arm_metrics({"BASE": (0, False)}, true)
    assert m["RANDOM"]["n_candidates"] == n
    assert m["RANDOM"]["pdms"] == pytest.approx(float(true["pdms"].double().mean()), abs=1e-12)
    assert m["RANDOM"]["collided"] == pytest.approx(float((true["nc"] != 1).double().mean()))
