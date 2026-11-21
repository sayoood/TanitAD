"""Pin the three corrections the PI made on 2026-08-27.

1. **Bands.** Tactical is **2–6 s**, not 0–6 s; strategic is **8–30 s**, so
   nothing before 8 s may carry a strategic token. The reported defect was
   `TURN_RIGHT_FOLLOW_ROUTE` with `by_time_s: 6.0`.
2. **The 6–8 s gap is real** and must stay visible rather than being absorbed.
3. **A road curve is not a turn.** A long bend taken at speed emits
   `ADAPT_SPEED_FOR_CURVE`, never `TURN_L`/`TURN_R`.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

_P = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "s2_geom_emit_v7.py"
_spec = importlib.util.spec_from_file_location("emit_v7", _P)
EM7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EM7)

HZ = 10.0


def drive(legs):
    """Build poses from (speed, turn_radius_or_None, duration_s) legs."""
    x = y = yaw = 0.0
    out = []
    for v, R, dur in legs:
        for _ in range(int(dur * HZ)):
            om = 0.0 if R is None else v / R
            yaw += om / HZ
            x += v * np.cos(yaw) / HZ
            y += v * np.sin(yaw) / HZ
            out.append((x, y, yaw, v))
    return np.array(out, dtype=np.float64)


# --- 1. the bands ----------------------------------------------------------

def test_no_strategic_token_before_8_seconds():
    """⛔ THE REPORTED DEFECT: `TURN_RIGHT_FOLLOW_ROUTE  by_time_s: 6.0`."""
    # a slow, tight turn beginning at t+6.0 s — inside the 6-8 s GAP
    poses = drive([(5.0, None, 6.0), (5.0, 12.0, 4.0), (5.0, None, 22.0)])
    seq = EM7.manoeuvre_sequence(poses, 0)
    g, a = EM7.strategic(poses, 0, seq)
    by = (g.get("args") or {}).get("by_time_s")
    assert by is None or by >= EM7.STRATEGIC_S[0], (
        f"strategic token at t={by}s, before the band opens at "
        f"{EM7.STRATEGIC_S[0]}s: {g}")


def test_a_turn_inside_the_strategic_band_is_strategic():
    poses = drive([(5.0, None, 10.0), (5.0, 12.0, 4.0), (5.0, None, 18.0)])
    seq = EM7.manoeuvre_sequence(poses, 0)
    g, _ = EM7.strategic(poses, 0, seq)
    assert g["token"].endswith("_FOLLOW_ROUTE") and g["token"] != "FOLLOW_ROUTE"
    assert g["args"]["by_time_s"] >= EM7.STRATEGIC_S[0]


def test_the_band_constants_are_what_the_PI_specified():
    assert EM7.OPERATIVE_S == (0.0, 2.0)
    assert EM7.TACTICAL_S == (2.0, 6.0), "tactical is 2-6 s, NOT 0-6 s"
    assert EM7.STRATEGIC_S == (8.0, 30.0)


def test_the_6_to_8_second_gap_is_reported_not_absorbed():
    """A manoeuvre owned by no layer must surface, not vanish into one."""
    poses = drive([(5.0, None, 6.2), (5.0, 12.0, 1.6), (5.0, None, 24.0)])
    seq = EM7.manoeuvre_sequence(poses, 0)
    tac, strat, gap = EM7.split_by_band(poses, 0, seq)
    assert gap, "a manoeuvre wholly inside 6-8 s must be reported as unassigned"
    for g in gap:
        assert EM7.GAP_S[0] <= g[0] < EM7.GAP_S[1]


# --- 2. turn vs road curve -------------------------------------------------

def test_a_long_bend_at_speed_is_not_a_turn():
    """⛔ `43bbcbf9`: ~37 deg over ~80 m at 12-14 m/s, labelled TURN_L.

    ⚠️ R = 120 m at 13 m/s is chosen so the bend IS detected: yaw rate
    13/120 = 6.2 deg/s clears the 6 deg/s segment threshold, and the arc radius
    (120 m) is inside `TURN_MAX_ARC_R_M`. So geometry alone would call it a
    turn — only the SPEED gate refuses it, which is precisely the case the PI
    reported. A gentler bend would prove nothing here: it never reaches the
    detector at all.
    """
    poses = drive([(13.0, None, 2.0), (13.0, 120.0, 5.0), (13.0, None, 25.0)])
    seq = EM7.manoeuvre_sequence(poses, 0)
    assert seq, "the bend should still be DETECTED as a manoeuvre"
    assert seq[0][3] <= EM7.TURN_MAX_ARC_R_M, \
        "radius alone would admit this — the speed gate must be what refuses it"
    assert not any(EM7.is_turn(s) for s in seq), \
        f"a 120 m-radius bend at 13 m/s is a CURVE, not a turn: {seq}"


def test_a_tight_slow_junction_turn_is_a_turn():
    poses = drive([(5.0, None, 2.0), (5.0, 10.0, 3.5), (5.0, None, 24.0)])
    seq = EM7.manoeuvre_sequence(poses, 0)
    assert any(EM7.is_turn(s) for s in seq), f"this IS a junction turn: {seq}"


def test_the_radius_is_the_ARC_radius_not_peak_curvature():
    """A constant-radius arc must report ~that radius.

    Peak instantaneous curvature reported 40 m for a 144 m arc on `43bbcbf9`,
    which is what made a motorway bend look like a junction.
    """
    R_true = 100.0
    poses = drive([(12.0, None, 1.0), (12.0, R_true, 5.0), (12.0, None, 10.0)])
    seq = EM7.manoeuvre_sequence(poses, 0)
    assert seq
    R = seq[0][3]
    assert 0.75 * R_true <= R <= 1.25 * R_true, \
        f"arc radius {R} m is not within 25 % of the true {R_true} m"


def test_speed_is_what_the_gate_adds():
    """The same SWEEP, driven fast, stops being a turn.

    MEASURED on the corpus: the speed gate removes 70 of 159 false positives
    (44 %), lifting precision 58.4 % -> 70.6 %. This is that effect in one case.

    ⚠️ The radii differ (40 m vs 128 m) SO THAT the yaw rate and total heading
    change come out IDENTICAL — v/R is 0.125 rad/s in both, and both sweep
    28.6 deg over 4 s. Holding the radius fixed instead would change the yaw
    rate and confound the comparison with the detector's own segment threshold.
    The only free variable here is speed.
    """
    slow = drive([(5.0, None, 2.0), (5.0, 40.0, 4.0), (5.0, None, 24.0)])
    fast = drive([(16.0, None, 2.0), (16.0, 128.0, 4.0), (16.0, None, 24.0)])
    s_slow = EM7.manoeuvre_sequence(slow, 0)
    s_fast = EM7.manoeuvre_sequence(fast, 0)
    assert s_slow and s_fast, "both must be DETECTED for the comparison to mean anything"
    assert abs(abs(s_slow[0][2]) - abs(s_fast[0][2])) < 4.0, \
        f"the two sweeps must match: {s_slow[0][2]} vs {s_fast[0][2]}"
    assert any(EM7.is_turn(s) for s in s_slow), s_slow
    assert not any(EM7.is_turn(s) for s in s_fast), \
        f"the same sweep at 16 m/s is a curve, not a turn: {s_fast}"


# --- 3. the goal and the action never disagree -----------------------------

@pytest.mark.parametrize("legs", [
    [(5.0, None, 2.0), (5.0, 10.0, 3.5), (5.0, None, 24.0)],     # a turn
    [(13.0, None, 2.0), (13.0, 140.0, 5.0), (13.0, None, 25.0)],  # a curve
    [(12.0, None, 30.0)],                                         # straight
])
def test_turn_goal_and_turn_action_agree_in_both_directions(legs):
    """One fact, one detector — checked BOTH ways.

    A one-sided check certified 70 broken clips once already, and gating only
    the goal side left 329 clips with `lat=TURN` and no turn goal.
    """
    poses = drive(legs)
    seq = EM7.manoeuvre_sequence(poses, 0)
    act = EM7.tactical_actions(poses, 0, seq)
    goals, _anchor, _viol = EM7.tactical_goals(poses, 0, seq, None,
                                               lat_action=act["lat"])
    has_goal = any(k.startswith(("TURN_", "YIELD_FOR_TURN_")) for k in goals)
    has_act = act["lat"].startswith("TURN_")
    assert has_goal == has_act, (
        f"goal says turn={has_goal} but action says turn={has_act}: "
        f"{list(goals)} / {act['lat']}")
