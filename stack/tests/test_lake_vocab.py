"""CPU tests — the frozen v3 vocabulary (``lake.vocab``) + the KINEMATIC goal
minter (``lake.goal_labels``, Stage 4). Synthetic unicycle trajectories only; no
GPU, no VLM, no data.
"""

import math

import pytest
import torch

from tanitad.lake import goal_labels as GL
from tanitad.lake import vocab as V


def _roll(yaw_rate, speed, T, dt=0.1):
    """Unicycle rollout -> contract poses [T,4] = (x, y, yaw, v)."""
    rows, x, y, yaw = [], 0.0, 0.0, 0.0
    for t in range(T):
        v = float(speed[t])
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += float(yaw_rate[t]) * dt
    return torch.tensor(rows, dtype=torch.float32)


def _const(v, T):
    return _roll(torch.zeros(T), torch.full((T,), float(v)), T)


# --------------------------------------------------------------------------- #
# VOCAB — frozen tokens + banding                                             #
# --------------------------------------------------------------------------- #
def test_vocab_frozen_counts():
    # the enumerated per-slot rows are authoritative (doc header totals are stale)
    assert len(V.GOAL_SLOTS) == 18
    assert sum(len(t) for t in V.STRATEGIC_TOKENS.values()) == 34
    assert sum(len(t) for t in V.TACTICAL_TOKENS.values()) == 80
    assert len(V.VTARGET_TOKENS) == 23
    assert V.VTARGET_TOKENS[0] == "v_stop"
    assert V.VTARGET_TOKENS[-1] == "v(37.5-40]"


@pytest.mark.parametrize("speed,expect", [
    (0.0, "v_stop"), (0.3, "v_stop"), (0.9, "v(0-1]"), (5.2, "v(5-6]"),
    (9.99, "v(9-10]"), (10.0, "v(9-10]"), (10.1, "v(10-12.5]"),
    (12.5, "v(10-12.5]"), (30.0, "v(27.5-30]"), (33.0, "v(32.5-35]"),
    (41.0, "v(37.5-40]"),                       # clamp above 40
])
def test_vtarget_band(speed, expect):
    assert V.vtarget_band(speed) == expect


def test_headway_band_snaps_to_nearest_nominal():
    assert V.headway_band(0.5) == "hw_0.8s"
    assert V.headway_band(1.5) == "hw_1.45s"
    assert V.headway_band(3.0) == "hw_2.5s+"
    assert V.headway_band(float("nan")) == V.UNKNOWN


def test_empty_goal_is_all_unknown_and_valid():
    g = V.empty_goal()
    assert set(g) == set(V.GOAL_SLOTS)
    assert all(v == {"token": "unknown", "prov": "unknown"} for v in g.values())
    V.validate_goal(g)                          # must not raise
    assert V.goal_provenance_summary(g)["unknown"] == 18


def test_validate_goal_rejects_bad_token_and_provenance():
    g = V.empty_goal()
    g["LONMODE"] = {"token": "not_a_mode", "prov": "kinematic"}
    with pytest.raises(ValueError, match="not in vocab"):
        V.validate_goal(g)
    g2 = V.empty_goal()
    with pytest.raises(ValueError, match="bad provenance"):
        V.slot("free_cruise", "telepathy")
    # an 'unknown' token with a concrete kinematic provenance is dishonest -> reject
    g2["ROUTE"] = {"token": "unknown", "prov": "kinematic"}
    with pytest.raises(ValueError):
        V.validate_goal(g2)


def test_slot_helpers():
    s = V.slot("follow", "kinematic")
    assert s == {"token": "follow", "prov": "kinematic"}
    assert V.is_pending(V.empty_goal(), "ROUTE") is True


# --------------------------------------------------------------------------- #
# GOAL MINTING (Stage 4) — kinematic slots, honest gaps                        #
# --------------------------------------------------------------------------- #
def test_cruise_mints_free_cruise_follow_and_speed_band():
    p = _const(30.0, 300)
    g = GL.mint_kinematic_goal(p, 0)
    assert g["LONMODE"]["token"] == "free_cruise"
    assert g["ROUTE"]["token"] == "follow"
    assert g["VTARGET"]["token"] == "v(27.5-30]"
    assert g["DYN"]["token"] == "gentle"
    # every minted slot is provenance kinematic; VLM/map slots stay unknown
    for s in ("VTARGET", "VSOURCE", "LONMODE", "LATMANEUVER", "DYN", "ROUTE"):
        assert g[s]["prov"] == "kinematic"
    for s in ("HEADWAY", "INTERACT", "TACPOINT", "LIGHTSTATE", "MISSION"):
        assert g[s] == {"token": "unknown", "prov": "unknown"}


def test_brake_to_stop_then_hold():
    # brake to a full stop within the 2 s label horizon, then hold
    sp = torch.cat([torch.full((40,), 15.0), torch.linspace(15, 0.0, 20),
                    torch.zeros(60)])
    p = _roll(torch.zeros(len(sp)), sp, len(sp))
    assert GL.lonmode_at(p, 45)["token"] == "stop_at_point"    # braking to a stop
    assert GL.lonmode_at(p, 90)["token"] == "hold_stop"        # stopped, held


def test_launch_from_standstill():
    # stopped, then an immediate launch that is under way inside the label horizon
    sp = torch.cat([torch.zeros(30), torch.linspace(0.0, 12.0, 20),
                    torch.full((70,), 12.0)])
    p = _roll(torch.zeros(len(sp)), sp, len(sp))
    assert GL.lonmode_at(p, 30)["token"] == "launch"


def test_junction_turn_vs_road_curve_not_conflated():
    # tight junction left (R=15 m) -> a genuine route turn
    T = 320
    yr = torch.zeros(T)
    yr[120:160] = 8.0 / 15.0
    junc = _roll(yr, torch.full((T,), 8.0), T)
    assert GL.route_at(junc, 60)["token"] == "turn_left"
    # gentle highway curve (R=300 m) at speed -> lane-keeping, NOT a route turn
    curve = _roll(torch.full((T,), 30.0 / 300.0), torch.full((T,), 30.0), T)
    assert GL.route_at(curve, 10)["token"] == "follow"


def test_short_or_ambiguous_future_is_unknown_route():
    p = _const(20.0, 80)                         # < NAV_MIN_STEPS future -> unknown
    assert GL.route_at(p, 0)["token"] == "unknown"
    assert GL.route_at(p, 0)["prov"] == "unknown"


def test_signal_only_minted_from_can_blinker():
    p = _const(20.0, 200)
    g0 = GL.mint_kinematic_goal(p, 0)
    assert g0["SIGNAL"] == {"token": "unknown", "prov": "unknown"}
    g1 = GL.mint_kinematic_goal(p, 0, has_can=True, blinker="indicator_left")
    assert g1["SIGNAL"] == {"token": "indicator_left", "prov": "human"}


def test_mint_validates_and_summary_takes_mode():
    p = _const(25.0, 300)
    g = GL.mint_kinematic_goal(p, 0)
    V.validate_goal(g)                          # a minted goal is always valid
    summ = GL.episode_goal_summary(p)
    assert summ["LONMODE"]["token"] == "free_cruise"
    assert summ["ROUTE"]["token"] == "follow"
    assert summ["VTARGET"]["prov"] == "kinematic"
