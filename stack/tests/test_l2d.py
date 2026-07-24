"""Unit tests for the L2D adapter (data/l2d.py) — pure functions, no network/video.

Covers the trap-mitigations that matter: drive-level de-dup + drive-disjoint split
(overlap trap), the km/h speed unit, the instruction/speed-limit parsers, and the
native vocab-slot minting staying inside the FROZEN vocabulary.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from tanitad.data import l2d
from tanitad.lake import vocab as V


# --------------------------------------------------------------------------- #
# parsers                                                                       #
# --------------------------------------------------------------------------- #
def test_parse_distance_m():
    assert l2d.parse_distance_m("go straight for 0.7 km") == 700.0
    assert l2d.parse_distance_m("In 150 m turn left") == 150.0
    assert l2d.parse_distance_m("go straight for 450 m") == 450.0
    assert l2d.parse_distance_m("Exit the roundabout using the first exit") is None


def test_parse_route_token():
    assert l2d.parse_route_token("In 100 m exit the roundabout using the first exit") == "roundabout"
    assert l2d.parse_route_token("In 150 m turn left at the intersection") == "turn_left"
    assert l2d.parse_route_token("Turn right onto the primary road") == "turn_right"
    assert l2d.parse_route_token("go straight on the secondary road for 450 m") == "straight"
    assert l2d.parse_route_token("In 200 m take the highway exit on the slight right") == "exit_right"
    assert l2d.parse_route_token("merge onto the motorway") == "merge"
    assert l2d.parse_route_token("observe the speed limit of 50 km/h") is None


def test_parse_max_speed_and_policy():
    assert l2d.parse_max_speed_kmh("50.0") == 50.0
    assert l2d.parse_max_speed_kmh("0.0") is None          # no posted limit
    assert l2d.parse_max_speed_kmh("NA") is None
    assert l2d.parse_max_speed_kmh(None) is None
    assert l2d.speedpolicy_token(30.0) == "cap_low"
    assert l2d.speedpolicy_token(50.0) == "cap_med"
    assert l2d.speedpolicy_token(100.0) == "cap_high"
    assert l2d.speedpolicy_token(None) is None


# --------------------------------------------------------------------------- #
# native vocab-slot minting stays inside the FROZEN vocabulary                  #
# --------------------------------------------------------------------------- #
def test_native_slots_respect_freeze():
    out = l2d.l2d_native_slots(
        instruction="go straight on the secondary road for 0.6 km, "
                    "observe the speed limit of 50 km/h",
        max_speed_kmh=50.0, turn_signal=1)
    g = out["goal_slots"]
    assert g["SIGNAL"] == {"token": "indicator_left", "prov": "human"}
    assert g["VSOURCE"]["token"] == "sign_limit" and g["VSOURCE"]["prov"] == "map"
    assert g["VTARGET"]["token"] == V.vtarget_band(50.0 / 3.6)
    assert g["SPEEDPOLICY"]["token"] == "cap_med"
    assert g["ROUTE"]["token"] == "straight"
    # ROUTEDIST is the v1.1 CANDIDATE — must NOT be a frozen goal slot
    assert "ROUTEDIST" not in V.GOAL_SLOTS
    assert out["routedist"]["token"] == V.routedist_band(600.0, observed_arc_m=V.ROUTEDIST_LOOKED_ENOUGH_M)
    # overlaying the native slots onto an empty goal must keep the freeze valid
    goal = V.empty_goal(); goal.update(g)
    V.validate_goal(goal)                                  # raises if the freeze broke
    assert set(goal) == set(V.GOAL_SLOTS)


def test_turn_signal_mapping():
    assert l2d.l2d_native_slots(instruction="", max_speed_kmh=None,
                                turn_signal=0)["goal_slots"]["SIGNAL"]["token"] == "none"
    assert l2d.l2d_native_slots(instruction="", max_speed_kmh=None,
                                turn_signal=2)["goal_slots"]["SIGNAL"]["token"] == "indicator_right"


# --------------------------------------------------------------------------- #
# drive reconstruction + de-dup + drive-disjoint split (the overlap trap)       #
# --------------------------------------------------------------------------- #
def _ep(idx, session, t0_s, dur_s=30, data_file=0, front_file=0):
    """A synthetic episode-meta dict at a given absolute start (seconds)."""
    return {"episode_index": idx, "session_id": session,
            "canonical_name": f"veh/{session}", "length": dur_s * 10,
            "ts_min": int(t0_s * 1e9), "ts_max": int((t0_s + dur_s) * 1e9),
            "data_file": data_file, "front_file": front_file, "tasks": ["go straight"]}


def test_drive_dedup_and_split():
    # session A: 5 sliding 30 s windows at 15 s stride (each overlaps its neighbour)
    A = [_ep(i, "A", 15 * i) for i in range(5)]
    # session B: 2 non-overlapping windows
    B = [_ep(100 + i, "B", 60 * i) for i in range(2)]
    index = A + B

    drives = l2d.reconstruct_drives(index)
    assert set(drives) == {"A", "B"} and len(drives["A"]) == 5

    keptA = l2d.select_nonoverlapping(A)
    # windows at 0,15,30,45,60 -> greedy keeps 0,30,60 (15 & 45 overlap a kept one)
    assert [e["episode_index"] for e in keptA] == [0, 2, 4]

    dd = l2d.dedup_index(index)
    assert len(dd) == 3 + 2                                # 3 kept from A, 2 from B

    split = l2d.split_by_drive(dd, val_frac=0.5, seed=0)
    tr = {e["session_id"] for e in split["train"]}
    va = {e["session_id"] for e in split["val"]}
    assert not (tr & va)                                   # drive-disjoint (I3)


def test_episode_id_stable_and_distinct():
    e1 = _ep(0, "A", 0); e2 = _ep(1, "A", 15)
    assert l2d.episode_id_of(e1) == l2d.episode_id_of(_ep(0, "A", 0))   # stable
    assert l2d.episode_id_of(e1) != l2d.episode_id_of(e2)              # distinct
    assert 0 < l2d.episode_id_of(e1) < (1 << 63)                       # int64-safe


# --------------------------------------------------------------------------- #
# state -> poses/actions (km/h unit + kinematic derivation)                     #
# --------------------------------------------------------------------------- #
def test_poses_actions_from_state():
    T = 30
    veh = np.zeros((T, 8), dtype=np.float64)
    # drive north at a constant 36 km/h (= 10 m/s); lat increases, lon fixed
    v_kmh = 36.0
    veh[:, l2d.VEH["speed"]] = v_kmh
    lat0, lon0 = 52.0, 8.0
    dlat_per_step = (10.0 * 0.1) / l2d.EARTH_M_PER_DEG_LAT            # 1 m/step north
    veh[:, l2d.VEH["lat"]] = lat0 + dlat_per_step * np.arange(T)
    veh[:, l2d.VEH["lon"]] = lon0
    unix_ns = (np.arange(T) * 1e8).astype(np.int64)                  # 10 Hz

    poses, actions = l2d.poses_actions_from_state(veh, unix_ns)
    assert poses.shape == (T, 4) and actions.shape == (T, 2)
    # v (m/s) == km/h / 3.6
    assert abs(float(poses[5, 3]) - v_kmh / 3.6) < 1e-3
    # heading north -> yaw ~ +pi/2 (atan2(+dy, 0))
    assert abs(float(poses[5, 2]) - math.pi / 2) < 1e-2
    # x (east) ~ 0, y (north) grows
    assert abs(float(poses[0, 0])) < 1e-6 and poses[-1, 1] > poses[0, 1]
    # constant speed -> ~zero accel, straight -> ~zero steer
    assert abs(float(actions[5, 1])) < 1e-2
    assert abs(float(actions[5, 0])) < 1e-2


def test_stack_frames_shape():
    vid = torch.zeros((10, 3, 32, 32), dtype=torch.uint8)
    st = l2d.stack_frames(vid, n_stack=3)
    assert st.shape == (8, 9, 32, 32)
