"""Tests for `taniteval.lead_source` — the obstacle.offline -> win["lead"] wiring.

Every case is hand-computable. The registration and window-grid tests are the load-bearing ones:
a silently mis-registered window puts the lead in the wrong place and the metric still returns a
plausible number, which is the failure mode this module exists to make impossible.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from taniteval.lead_source import (LEAD, NO_LABEL, NO_LEAD, RegistrationError,
                                   lead_block, lead_track_in_window,
                                   register_poses_to_time, select_lead_causal,
                                   window_last_indices)

REPO = Path(__file__).resolve().parents[2]
VAL40_MANIFEST = (REPO / "TanitAD Research Hub" / "Architecture & Inference" / "Implementation"
                  / "incoming" / "2026-07-26-s3-decision-grade" / "artifacts"
                  / "manifest_EVALPOD_val40.json")


# --------------------------------------------------------------------------- #
# 1. the window grid                                                           #
# --------------------------------------------------------------------------- #
def test_window_last_indices_matches_rollout_collect_formula():
    # collect: starts = range(0, T - window - K_MAX, stride); origin = start + window - 1
    t = 199
    want = np.arange(0, t - 8 - 20, 8) + 7
    assert np.array_equal(window_last_indices(t), want)
    assert window_last_indices(t)[0] == 7          # first origin is the 8th pose
    assert len(window_last_indices(t)) == 22


def test_window_grid_reproduces_the_canonical_881_windows():
    """⭐ THE registration proof. The published open-loop statistic is 881 stride-8 windows over
    the 40 canonical val episodes. If this module's grid did not match `rollout.collect`'s, a
    lead block built from it would be misaligned with every banked pred/gt row."""
    if not VAL40_MANIFEST.exists():                     # gated-corpus artifact
        pytest.skip("val40 manifest not present")
    man = json.loads(VAL40_MANIFEST.read_text())
    total = sum(len(window_last_indices(e["T"])) for e in man["episodes"])
    assert total == 881


def test_window_grid_is_empty_for_a_too_short_episode():
    assert len(window_last_indices(28)) == 0
    assert len(window_last_indices(0)) == 0


# --------------------------------------------------------------------------- #
# 2. registration                                                              #
# --------------------------------------------------------------------------- #
def _straight_ego(t_end=20.0, hz=100.0, v=10.0):
    t = np.arange(0.0, t_end, 1.0 / hz)
    return t, v * t, np.zeros_like(t), np.zeros_like(t), np.full_like(t, v)


def test_register_recovers_a_known_affine_grid():
    t, x, y, _yaw, _v = _straight_ego()
    a, b, tt = 0.35, 0.1, 150                       # episode grid: t = 0.35 + 0.1*i
    ti = a + b * np.arange(tt)
    poses = np.column_stack([np.interp(ti, t, x), np.interp(ti, t, y)])
    r = register_poses_to_time(poses, t, x, y)
    assert r["residual_m"]["median"] < 0.05
    assert abs(r["b"] - b) < 1e-3
    assert abs(r["a"] - a) < 1e-2
    assert np.allclose(r["t_s"], ti, atol=2e-2)


def test_register_refuses_the_wrong_clip():
    t, x, y, _, _ = _straight_ego()
    poses = np.column_stack([np.full(150, 5000.0), np.full(150, -5000.0)])
    with pytest.raises(RegistrationError):
        register_poses_to_time(poses, t, x, y)


def test_register_works_on_a_curving_track():
    t = np.arange(0.0, 20.0, 0.01)
    x, y = 30.0 * np.sin(t / 6.0), 30.0 * (1 - np.cos(t / 6.0))
    ti = 1.0 + 0.1 * np.arange(150)
    poses = np.column_stack([np.interp(ti, t, x), np.interp(ti, t, y)])
    r = register_poses_to_time(poses, t, x, y)
    assert r["residual_m"]["median"] < 0.05
    assert abs(r["b"] - 0.1) < 1e-3


# --------------------------------------------------------------------------- #
# 3. causal lead selection                                                     #
# --------------------------------------------------------------------------- #
def _obs(rows):
    """rows = (t, track, cx, cy, size_x, is_vehicle)."""
    a = list(zip(*rows))
    return (np.array(a[0], float), np.array(a[1], object), np.array(a[2], float),
            np.array(a[3], float), np.array(a[4], float), np.array(a[5], bool))


def test_lead_selection_takes_the_nearest_in_corridor_vehicle():
    t, trk, cx, cy, sx, veh = _obs([
        (5.0, "far", 40.0, 0.0, 4.0, True),
        (5.0, "near", 20.0, 0.5, 4.0, True),
        (5.0, "sidecar", 10.0, 3.0, 4.0, True),      # outside the corridor
        (5.0, "walker", 8.0, 0.0, 0.7, False),       # not a vehicle
    ])
    got, gap, size = select_lead_causal(t, trk, cx, cy, sx, veh, 5.0)
    assert got == "near"
    assert gap == pytest.approx(20.0 - 2.0)
    assert size == pytest.approx(4.0)


def test_lead_selection_is_strictly_causal():
    t, trk, cx, cy, sx, veh = _obs([(6.0, "future", 15.0, 0.0, 4.0, True)])
    assert select_lead_causal(t, trk, cx, cy, sx, veh, 5.0)[0] is None


def test_lead_selection_rejects_a_stale_sample():
    t, trk, cx, cy, sx, veh = _obs([(3.0, "old", 15.0, 0.0, 4.0, True)])
    assert select_lead_causal(t, trk, cx, cy, sx, veh, 5.0)[0] is None       # 2 s stale
    assert select_lead_causal(t, trk, cx, cy, sx, veh, 3.4)[0] == "old"      # 0.4 s stale


def test_lead_selection_rejects_a_vehicle_behind_or_beyond_the_max_gap():
    t, trk, cx, cy, sx, veh = _obs([(5.0, "behind", -10.0, 0.0, 4.0, True),
                                    (5.0, "miles", 200.0, 0.0, 4.0, True)])
    assert select_lead_causal(t, trk, cx, cy, sx, veh, 5.0)[0] is None


# --------------------------------------------------------------------------- #
# 4. the rig -> world -> t0 composition                                        #
# --------------------------------------------------------------------------- #
def test_a_world_static_lead_stays_put_in_the_window_frame():
    """The whole reason the frame chain exists. A PARKED car 30 m ahead stays at x = 30 in the
    window-origin frame for the whole horizon — it is the EGO's own advance along its predicted
    path that closes the gap, and `per_step_gap` subtracts the path. Its rig-frame coordinate,
    by contrast, marches down to 10 m."""
    et, ex, ey, eyaw, _ = _straight_ego(v=10.0)
    ts = np.arange(0.0, 20.0, 0.1)
    cx = 30.0 - 10.0 * ts               # the parked car, as the rig sees it at each sample
    got = lead_track_in_window(ts, np.array(["p"] * ts.size, object), cx, np.zeros_like(cx),
                               "p", 0.0, np.array([0.5, 1.0, 1.5, 2.0]), et, ex, ey, eyaw)
    assert np.allclose(got[:, 0], 30.0, atol=1e-6)
    assert np.allclose(got[:, 1], 0.0, atol=1e-9)


def test_raw_rig_coordinates_would_understate_the_gap_by_the_distance_travelled():
    """Negative control. Reading the rig coordinate directly puts the lead 20 m closer at +2 s —
    exactly the distance the ego covers — i.e. it would invent tailgating everywhere."""
    et, ex, ey, eyaw, _ = _straight_ego(v=10.0)
    ts = np.arange(0.0, 20.0, 0.1)
    cx = 30.0 - 10.0 * ts
    got = lead_track_in_window(ts, np.array(["p"] * ts.size, object), cx, np.zeros_like(cx),
                               "p", 0.0, np.array([2.0]), et, ex, ey, eyaw)
    raw = float(np.interp(2.0, ts, cx))
    assert raw == pytest.approx(10.0)                    # the naive read
    assert float(got[0, 0]) == pytest.approx(30.0)       # the composed read
    assert float(got[0, 0]) - raw == pytest.approx(20.0)  # == v * horizon
    # a lead that is genuinely closing on the ego lands between the two
    cx2 = 30.0 - 12.0 * ts
    got2 = lead_track_in_window(ts, np.array(["q"] * ts.size, object), cx2, np.zeros_like(cx2),
                                "q", 0.0, np.array([2.0]), et, ex, ey, eyaw)
    assert float(got2[0, 0]) == pytest.approx(30.0 - 12.0 * 2.0 + 10.0 * 2.0)


def test_lead_track_is_nan_where_the_track_is_stale():
    et, ex, ey, eyaw, _ = _straight_ego()
    ts = np.array([0.0, 0.1, 0.2])
    got = lead_track_in_window(ts, np.array(["p"] * 3, object), np.array([30.0, 29.0, 28.0]),
                               np.zeros(3), "p", 0.0, np.array([0.5, 5.0]), et, ex, ey, eyaw)
    assert np.isfinite(got[0]).all()                  # 0.5 s: 0.3 s stale, inside MAX_STALE_S
    assert np.isnan(got[1]).all()                     # 5.0 s: far past the track's end


# --------------------------------------------------------------------------- #
# 5. the three window states                                                   #
# --------------------------------------------------------------------------- #
def _ego_dict():
    et, ex, ey, eyaw, ev = _straight_ego()
    return {"t": et, "x": ex, "y": ey, "yaw": eyaw, "v": ev}


def _obs_dict(ts, cx, cy=None, track="p", size_x=4.0, vehicle=True):
    n = np.asarray(ts).size
    return {"t": np.asarray(ts, float), "track": np.array([track] * n, object),
            "center_x": np.asarray(cx, float),
            "center_y": (np.zeros(n) if cy is None else np.asarray(cy, float)),
            "size_x": np.full(n, size_x), "is_vehicle": np.full(n, vehicle)}


def test_no_obstacle_feature_is_NO_LABEL_never_free_flow():
    """⛔ The bias this instrument exists to avoid. 2.44 % of the corpus has no obstacle.offline;
    calling those windows 'no lead agent' would manufacture empty road."""
    b = lead_block([1.0, 2.0], [0.5, 1.0, 1.5, 2.0], None, _ego_dict())
    assert list(b["state"]) == [NO_LABEL, NO_LABEL]
    assert b["counts"][NO_LABEL] == 2
    assert b["counts"][NO_LEAD] == 0
    assert not b["has_lead"].any()
    assert np.isnan(b["leads"]).all()


def test_a_window_whose_horizon_leaves_the_labelled_span_is_NO_LABEL():
    """⛔ egomotion runs 20-140 s while obstacle.offline stops at ~20 s, so MOST of a long clip
    has no labels. A horizon that leaves the span looks exactly like an empty road."""
    ts = np.arange(0.0, 20.0, 0.1)
    obs = _obs_dict(ts, np.full(ts.size, 30.0))       # a car keeping pace, 30 m ahead throughout
    b = lead_block([5.0, 19.5], [0.5, 1.0, 1.5, 2.0], obs, _ego_dict())
    assert b["state"][0] == LEAD
    assert b["state"][1] == NO_LABEL          # 19.5 + 2.0 = 21.5 s > span end + guard


def test_labels_present_but_road_clear_is_NO_LEAD():
    ts = np.arange(0.0, 20.0, 0.1)
    obs = _obs_dict(ts, np.full(ts.size, -20.0))        # the only vehicle is BEHIND
    b = lead_block([5.0], [0.5, 1.0, 1.5, 2.0], obs, _ego_dict())
    assert b["state"][0] == NO_LEAD
    assert b["counts"][NO_LABEL] == 0


def test_lead_block_speeds_come_from_egomotion_at_t0():
    ts = np.arange(0.0, 20.0, 0.1)
    b = lead_block([5.0, 9.0], [1.0], _obs_dict(ts, 30.0 - 10.0 * ts), _ego_dict())
    assert np.allclose(b["speeds"], 10.0)


def test_lead_block_feeds_distance_keeping_end_to_end():
    from taniteval.lead_metrics import distance_keeping
    ts = np.arange(0.0, 20.0, 0.1)
    obs = _obs_dict(ts, 30.0 - 10.0 * ts)               # parked car 30 m ahead at t=0
    rel = np.array([0.5, 1.0, 1.5, 2.0])
    b = lead_block([0.0], rel, obs, _ego_dict())
    paths = np.stack([np.column_stack([10.0 * rel, np.zeros(4)])])   # ego holds 10 m/s
    dk = distance_keeping(paths, b["leads"], b["lead_lens"], b["speeds"], dt=0.5)
    assert dk["status"] == "OK"
    # at t=2 s: lead at 10 m in the window frame, ego at 20 m -> the gap has gone NEGATIVE, so
    # the last admissible step is the tightest non-negative one.
    assert dk["n"] == 1
    assert np.isfinite(dk["_per_window"]["headway_min_m"][0]
                       if "_per_window" in dk else dk["headway_min_m"][0])
