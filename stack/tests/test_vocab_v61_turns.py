"""v6.1 turn vocabulary: the tokens, the tensor contract, and the window.

⛔ WHAT THESE PIN, all found by the PI on ONE clip (`0e56dae2`, 2026-08-23):

 1. `g_str = TURN_LEFT` with `a_str = HOLD_CORRIDOR` — the strategic ACTION
    contradicted its own GOAL, because v6.0 had no token for committing to a
    junction turn.
 2. `a_tac.lat = LANE_KEEP` while traversing an intersection — there are no
    lanes to keep in a junction, and v6.0 could not say anything else.
 3. The turn begins at **t+2.5 s and reaches +120 deg**, but `a_tac` was scored
    on a **0-2 s** window, so it could not see the manoeuvre at all.

⭐ AND THE TENSOR CONTRACT, which is why these are APPENDED tuples behind a
version switch rather than edits: both vocabularies SIZE LIVE EMBEDDING TABLES,
and a resume is tensor-strict. Indices must keep their meaning.
"""
from __future__ import annotations

import numpy as np
import pytest

from tanitad.models import v6


# -- the tensor contract ----------------------------------------------------
def test_v61_appends_and_never_reorders():
    """Indices 0..n-1 must keep their meaning so checkpoints stay loadable."""
    for base, ext in ((v6.TACTICAL_LAT_ACTIONS, v6.TACTICAL_LAT_ACTIONS_V61),
                      (v6.STRATEGIC_ACTION_TOKENS,
                       v6.STRATEGIC_ACTION_TOKENS_V61)):
        assert ext[:len(base)] == base, "v6.1 reordered or inserted"
        assert len(ext) == len(base) + 2
        assert len(set(ext)) == len(ext), "duplicate token"


def test_the_default_is_still_v60_so_importing_changes_nothing():
    assert v6.tactical_lat_actions() == v6.TACTICAL_LAT_ACTIONS
    assert v6.strategic_action_tokens() == v6.STRATEGIC_ACTION_TOKENS


def test_v61_carries_the_turn_tokens_on_both_layers():
    assert {"TURN_L", "TURN_R"} <= set(v6.tactical_lat_actions("v6.1"))
    assert {"PREPARE_TURN_L", "PREPARE_TURN_R"} <= set(
        v6.strategic_action_tokens("v6.1"))


def test_an_unknown_version_raises_rather_than_guessing():
    for fn in (v6.tactical_lat_actions, v6.strategic_action_tokens):
        with pytest.raises(ValueError):
            fn("v9.9")


def test_the_two_layers_must_be_selected_together():
    """A strategic 'prepare to turn' with a tactical vocab that cannot turn
    reproduces the same contradiction one level down."""
    lat61 = set(v6.tactical_lat_actions("v6.1"))
    str61 = set(v6.strategic_action_tokens("v6.1"))
    assert ("TURN_L" in lat61) == ("PREPARE_TURN_L" in str61)


# -- the emitter's behaviour ------------------------------------------------
HZ = 10.0


def _drive(segments, hz=HZ):
    import math
    xs, ys, yaws, vs = [], [], [], []
    x = y = yaw = 0.0
    for v, radius, secs in segments:
        for _ in range(int(round(secs * hz))):
            if radius:
                yaw += (v / hz) / radius
            x += v / hz * math.cos(yaw)
            y += v / hz * math.sin(yaw)
            xs.append(x); ys.append(y); yaws.append(yaw); vs.append(v)
    return np.stack([xs, ys, yaws, vs], axis=1)


def _emit_actions(poses, key=0, **kw):
    import importlib.util
    from pathlib import Path
    p = (Path(__file__).resolve().parents[1] / "scripts" / "s2_geom_emit.py")
    spec = importlib.util.spec_from_file_location("s2_geom_emit", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._tactical_actions(poses, key, **kw)


def test_a_junction_turn_is_named_a_turn_not_a_lane_keep():
    poses = _drive([(6.0, None, 1.0), (6.0, 8.0, 4.0), (6.0, None, 4.0)])
    got = _emit_actions(poses)
    assert got["lat"] == "TURN_L", got
    assert got["vocab"] == "v6.1"


def test_a_turn_starting_after_2s_is_still_seen():
    """The exact shape of 0e56dae2: onset t+2.5 s, +120 deg."""
    poses = _drive([(8.0, None, 2.5), (6.0, 7.0, 3.0), (6.0, None, 3.0)])
    assert _emit_actions(poses)["lat"].startswith("TURN_"), "2 s window blindness"


def test_the_action_window_is_the_full_plan_rollout():
    got = _emit_actions(_drive([(8.0, None, 8.0)]))
    assert got["window_s"] == [0.0, 6.0], got["window_s"]


def test_a_straight_road_is_still_a_lane_keep():
    assert _emit_actions(_drive([(10.0, None, 8.0)]))["lat"] == "LANE_KEEP"


def test_v60_abstains_rather_than_emitting_a_false_lane_keep():
    """Under v6.0 a junction cannot be named; saying LANE_KEEP would be false."""
    poses = _drive([(6.0, None, 1.0), (6.0, 8.0, 4.0), (6.0, None, 4.0)])
    assert _emit_actions(poses, vocab_version="v6.0")["lat"] == "ABSTAIN"


# -- band separation: which LAYER owns a manoeuvre --------------------------
def _emit_one(clip_poses_fn):
    """Load the emitter module once for band tests."""
    import importlib.util
    from pathlib import Path
    p = (Path(__file__).resolve().parents[1] / "scripts" / "s2_geom_emit.py")
    spec = importlib.util.spec_from_file_location("s2_geom_emit", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_turn_inside_the_plan_horizon_is_TACTICAL_not_strategic():
    """PI 2026-08-23: "the turning is happening within the 6 s horizon, so it
    is no strategic action any more, it's tactical."

    The exact shape of `0e56dae2`: turn at t+2.5 s, then the road continues.
    The strategic band (8-30 s) must see FOLLOW_MAIN_ROAD, and the tactical
    action must see the turn.
    """
    from tanitad.data import ego_manoeuvre as EM
    from tanitad.data import tactical_goals as TG
    mod = _emit_one(None)
    poses = _drive([(8.0, None, 2.5), (6.0, 7.0, 3.5), (8.0, None, 26.0)])
    key = 0
    # tactical action over the 0-6 s plan sees the turn
    assert mod._tactical_actions(poses, key)["lat"].startswith("TURN_")
    # strategic band 8-30 s sees the road, not the turn
    lo = key + int(round(mod.STRAT_MIN_S * mod.HZ))
    man = EM.analyse(poses, key=lo, hz=mod.HZ)
    tok, _ = TG.strategic_from_geometry(man)
    assert tok == "FOLLOW_MAIN_ROAD", (tok, man.lateral_class)


def test_a_turn_still_ahead_in_the_band_IS_strategic():
    """The mirror case: a turn the ego has NOT started is a strategic goal."""
    from tanitad.data import ego_manoeuvre as EM
    from tanitad.data import tactical_goals as TG
    mod = _emit_one(None)
    poses = _drive([(8.0, None, 12.0), (6.0, 8.0, 4.0), (8.0, None, 16.0)])
    lo = int(round(mod.STRAT_MIN_S * mod.HZ))
    man = EM.analyse(poses, key=lo, hz=mod.HZ)
    tok, _ = TG.strategic_from_geometry(man)
    assert tok in ("TURN_LEFT", "TURN_RIGHT"), (tok, man.lateral_class)
    # ...and the plan horizon must NOT see it
    assert mod._tactical_actions(poses, 0)["lat"] == "LANE_KEEP"


def test_the_strategic_band_never_starts_before_8s():
    mod = _emit_one(None)
    assert mod.STRAT_MIN_S == 8.0
    assert mod.TAC_ACTION_S == 6.0
    assert mod.TAC_ACTION_S <= mod.STRAT_MIN_S, \
        "the plan horizon must not overlap the strategic band"
