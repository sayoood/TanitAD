"""Tests for ``taniteval.nav_compliance`` — every control must read its KNOWN
value on a synthetic corpus, and the deliberate-regression arm (a token ECHO
that never reaches the path) must be INVISIBLE to the behavioural readout.

pytest is not installed on every pod; runs standalone too:
  python taniteval/tests/test_nav_compliance.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)              # <repo>/taniteval
if _PKG_PARENT not in sys.path:
    sys.path.append(_PKG_PARENT)                  # eager: a REGULAR package beats
                                                  # a namespace portion found later
_STACK = os.path.join(os.path.dirname(_PKG_PARENT), "stack")
if os.path.isdir(_STACK) and _STACK not in sys.path:
    sys.path.append(_STACK)

from taniteval import nav_compliance as nc   # noqa: E402

N_BOOT = 200          # fast; the estimator itself is pinned in test_ci.py


# --------------------------------------------------------------------------- #
# the synthetic corpus                                                         #
# --------------------------------------------------------------------------- #
def _corpus(n_ep=40, n_win=12, seed=0):
    """40 episodes x 12 windows (t_now 0.5 s apart). Even episodes carry a
    LEFT/RIGHT command with a turn executed at [4, 8] s; odd ones are FOLLOW.
    GT heading over the plan horizon is ±0.5 rad while the turn overlaps the
    6 s plan, ~0 otherwise (with small noise)."""
    rng = np.random.default_rng(seed)
    nav, valid, eid, t_now, ts, te = [], [], [], [], [], []
    gt = []
    for e in range(n_ep):
        turn = e % 2 == 0
        side = 1 if (e // 2) % 2 == 0 else -1
        for w in range(n_win):
            tn = 0.5 * w
            nav.append((nc.NAV_LEFT if side > 0 else nc.NAV_RIGHT) if turn else nc.NAV_FOLLOW)
            valid.append(True)
            eid.append(f"ep{e:03d}")
            t_now.append(tn)
            ts.append(4.0 if turn else np.nan)
            te.append(8.0 if turn else np.nan)
            executing = turn and (min(8.0, tn + 6.0) - max(4.0, tn) >= 1.0)
            gt.append(side * 0.5 + rng.normal(0, 0.03) if executing
                      else rng.normal(0, 0.03))
    return {k: np.asarray(v) for k, v in dict(
        nav=nav, valid=valid, eid=eid, t_now=t_now, ts=ts, te=te, gt=gt).items()}


def _shuffle(nav, seed=0):
    rng = np.random.default_rng(seed)
    return nav[rng.permutation(nav.size)]


def _report(model_signal_fn, C, extra_controls=None, **kw):
    """``model_signal_fn(nav_fed) -> signal[N]``; nav_zero feeds FOLLOW."""
    shuf = _shuffle(C["nav"])
    fed = {"nav_true": C["nav"], "nav_shuffled": shuf,
           "nav_zero": np.zeros_like(C["nav"])}
    sig = {c: model_signal_fn(fed[c]) for c in fed}
    controls = {"ha0": np.zeros(C["nav"].size),
                "ha0_ext": np.zeros(C["nav"].size)}       # a straight-line floor
    if extra_controls:
        controls.update(extra_controls)
    return nc.nav_compliance_report(
        readouts={"plan": sig, "gstr": sig, "anchor": sig},
        nav_true=C["nav"], nav_valid=C["valid"], nav_fed_by_cond=fed, eid=C["eid"],
        t_now_s=C["t_now"], t_start_s=C["ts"], t_end_s=C["te"],
        gt_signal={"plan": C["gt"], "gstr": C["gt"]},
        controls=controls, n_boot=N_BOOT, seed=0, **kw)


# --------------------------------------------------------------------------- #
# conventions                                                                  #
# --------------------------------------------------------------------------- #
def test_nav_indices_pinned_against_the_stack():
    try:
        from tanitad.refs import refc
        from tanitad.refs import refb
        import refb_labels  # noqa: F401  (stack/scripts) — optional
    except Exception:
        refc = refb = None
    if refc is not None:
        assert tuple(refc.NAV_COMMANDS)[:3] == ("follow", "left", "right")
        assert refc.NAV_COMMANDS.index("left") == nc.NAV_LEFT
        assert refc.NAV_COMMANDS.index("right") == nc.NAV_RIGHT
        assert refc.NAV_COMMANDS.index("follow") == nc.NAV_FOLLOW


def test_left_is_positive_heading_and_positive_lateral():
    # a CCW arc: y grows with x — the repo's +y = left convention
    t = np.linspace(0, 1, 9)
    left = np.stack([np.sin(t) * 10, (1 - np.cos(t)) * 10], -1)[None]
    right = left.copy()
    right[..., 1] *= -1
    assert nc.terminal_heading(left)[0] > 0.5 and nc.terminal_heading(right)[0] < -0.5
    assert nc.end_bearing(left)[0] > 0 and nc.end_bearing(right)[0] < 0
    assert nc.lateral_end(left)[0] > 0
    assert nc.commanded_side([nc.NAV_LEFT, nc.NAV_RIGHT, nc.NAV_FOLLOW]).tolist() == [1, -1, 0]


def test_stalled_last_segment_reads_zero_not_noise():
    p = np.zeros((1, 8, 2))
    p[0, :, 0] = np.linspace(0, 1, 8)
    p[0, -1] = p[0, -2] + np.array([0.001, 0.001])      # 1.4 mm, "turning" 45 deg
    assert nc.terminal_heading(p)[0] == 0.0


def test_ego_frame_matches_a_hand_rotation():
    pose_last = np.array([[10.0, -3.0, math.pi / 2, 5.0]])     # facing +y (world)
    fut = np.array([[[10.0, 7.0], [8.0, 7.0]]])                 # 10 m ahead; then 2 m LEFT of it
    e = nc.ego_frame(fut, pose_last)
    assert np.allclose(e[0, 0], [10.0, 0.0], atol=1e-9)
    assert np.allclose(e[0, 1], [10.0, 2.0], atol=1e-9)         # left = +y


def test_kinematic_heading_twin_matches_echo_gate_arc_length():
    try:
        from tanitad.eval import echo_gate as eg
    except Exception:
        return                                       # stack not on path: skip
    v0, a0, k0 = 8.0, -0.5, 0.02
    xy = eg._np_extrapolate(v0, a0, k0, 60)          # 6 s at 0.1 s
    s = math.asin(xy[-1, 0] * k0) / k0               # x = sin(k s)/k
    assert abs(nc.kinematic_heading([v0], [a0], [k0])[0] - k0 * s) < 1e-9


# --------------------------------------------------------------------------- #
# the informative windows and the tolerance                                    #
# --------------------------------------------------------------------------- #
def test_informative_mask_boundaries():
    nav = np.array([nc.NAV_LEFT] * 4 + [nc.NAV_FOLLOW])
    valid = np.ones(5, bool)
    t_now = np.array([0.0, 2.9, 3.0, 8.0, 3.0])
    ts = np.array([4.0] * 5)
    te = np.array([8.0] * 5)
    m = nc.informative_mask(nav, valid, t_now, ts, te, horizon_s=6.0, min_overlap_s=1.0)
    # overlap: [0,6]∩[4,8]=2 ✓ ; [2.9,8.9]∩ = 4 ✓ ; [3,9] = 4 ✓ ; [8,14]∩[4,8] = 0 ✗ ; follow ✗
    assert m.tolist() == [True, True, True, False, False]


def test_tolerance_lands_between_the_two_populations():
    d = nc.derive_tolerance(pos=[0.5, 0.45, 0.6, -0.5], neg=[0.02, -0.03, 0.05, 0.0])
    assert d["status"] == "OK" and 0.05 < d["tau"] <= 0.45
    assert d["youden_j"] == 1.0


def test_complies_requires_sign_and_magnitude():
    side = np.array([1, 1, -1, 0])
    sig = np.array([0.3, -0.3, -0.3, 0.3])
    assert nc.complies(sig, side, 0.2).tolist() == [1.0, 0.0, 1.0, 0.0]
    assert nc.complies(sig, side, 0.4).tolist() == [0.0, 0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# the panel — every control reads its known value                              #
# --------------------------------------------------------------------------- #
def test_follower_reads_the_ceiling_exactly_and_is_FOLLOWS_NAV():
    C = _corpus()
    rep = _report(lambda fed: nc.commanded_side(fed) * 0.5, C)
    plan = rep["readouts"]["plan"]
    assert plan["powered"]
    assert plan["conditionings"]["nav_true"]["compliance_with_TRUE_command"]["mean"] == 1.0
    ceil = plan["always_commanded_ceiling"]
    assert plan["paired_true_minus_shuffled"]["delta"] == ceil["delta_true_minus_shuffled"]
    assert plan["paired_true_minus_shuffled"]["separated"]
    assert plan["paired_true_minus_zero"]["delta"] == 1.0
    cs = plan["changed_subset"]
    assert cs["follows_FED_command"]["mean"] == 1.0 and cs["follows_TRUE_command"]["mean"] == 0.0
    assert rep["verdicts"]["plan"]["verdict"] == "FOLLOWS_NAV"
    assert rep["seam"]["reading"] == "GOAL_AND_PATH_FOLLOW"


def test_scene_driven_model_reads_zero_delta_and_is_NAV_BLIND_COINCIDENT():
    C = _corpus()
    rep = _report(lambda fed: C["gt"], C)              # ignores the token entirely
    plan = rep["readouts"]["plan"]
    assert plan["conditionings"]["nav_true"]["compliance_with_TRUE_command"]["mean"] > 0.95
    d = plan["paired_true_minus_shuffled"]
    # identical arms: the paired estimator reads EXACTLY 0.0 [0.0, 0.0] and is
    # not separated — the known value a nav-blind arm must produce
    assert d["delta"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0
    assert d["separated"] is False
    assert rep["verdicts"]["plan"]["verdict"] == "NAV_BLIND_COINCIDENT"
    assert rep["seam"]["reading"] == "NEITHER_FOLLOWS"


def test_DELIBERATE_REGRESSION_a_token_echo_is_invisible_to_the_behavioural_readout():
    """The old metric scored 1.0000 on an arm whose ROUTE HEAD copied the nav
    token. Give such an arm to this metric: its PATH is scene-driven and its
    head is an echo. The plan readout must read NO nav effect — the echo has
    nowhere to land, because a path is not a token."""
    C = _corpus()
    rep = _report(lambda fed: C["gt"], C)              # the path: pure scene
    echo_head = {c: nc.commanded_side(fed) for c, fed in
                 (("nav_true", C["nav"]), ("nav_shuffled", _shuffle(C["nav"])))}
    # the head's own 'accuracy against a label that IS the token' would be 1.0:
    assert np.mean(echo_head["nav_true"] == nc.commanded_side(C["nav"])) == 1.0
    # ...and the behavioural metric sees nothing of it:
    assert rep["readouts"]["plan"]["paired_true_minus_shuffled"]["delta"] == 0.0
    assert rep["verdicts"]["plan"]["verdict"] != "FOLLOWS_NAV"


def test_straight_line_control_reads_exactly_zero_and_panel_is_ok():
    C = _corpus()
    rep = _report(lambda fed: nc.commanded_side(fed) * 0.5, C)
    ha0 = rep["controls"]["ha0"]
    assert ha0["compliance_with_TRUE_command"]["mean"] == 0.0
    assert ha0["reads_known_value"] is True and rep["panel_ok"] is True
    assert ha0["delta_true_minus_shuffled"] == 0.0


def test_gt_control_reads_near_one_on_the_informative_windows():
    C = _corpus()
    rep = _report(lambda fed: nc.commanded_side(fed) * 0.5, C)
    assert rep["controls"]["gt"]["compliance_with_TRUE_command"]["mean"] >= 0.95
    tol = rep["tolerance"]["derived"]["plan"]
    assert tol["status"] == "OK" and 0.1 < tol["tau"] < 0.45


def test_unpowered_when_too_few_informative_windows():
    C = _corpus(n_ep=6, n_win=4)
    rep = _report(lambda fed: nc.commanded_side(fed) * 0.5, C)
    assert rep["readouts"]["plan"]["powered"] is False
    assert rep["verdicts"]["plan"]["verdict"] == "UNPOWERED"


def test_missing_shuffle_is_INADMISSIBLE_not_a_number():
    C = _corpus()
    fed = {"nav_true": C["nav"]}
    rep = nc.nav_compliance_report(
        readouts={"plan": {"nav_true": nc.commanded_side(C["nav"]) * 0.5}},
        nav_true=C["nav"], nav_valid=C["valid"], nav_fed_by_cond=fed, eid=C["eid"],
        t_now_s=C["t_now"], t_start_s=C["ts"], t_end_s=C["te"],
        gt_signal={"plan": C["gt"]}, n_boot=N_BOOT)
    assert rep["readouts"]["plan"]["paired_true_minus_shuffled"]["status"] == "UNAVAILABLE"
    assert rep["verdicts"]["plan"]["verdict"] == "INADMISSIBLE"


def test_seam_failure_is_localised_right_goal_wrong_path():
    C = _corpus()
    shuf = _shuffle(C["nav"])
    fed = {"nav_true": C["nav"], "nav_shuffled": shuf, "nav_zero": np.zeros_like(C["nav"])}
    follower = {c: nc.commanded_side(f) * 0.5 for c, f in fed.items()}
    scene = {c: C["gt"] for c in fed}
    rep = nc.nav_compliance_report(
        readouts={"plan": scene, "gstr": follower, "anchor": scene},
        nav_true=C["nav"], nav_valid=C["valid"], nav_fed_by_cond=fed, eid=C["eid"],
        t_now_s=C["t_now"], t_start_s=C["ts"], t_end_s=C["te"],
        gt_signal={"plan": C["gt"], "gstr": C["gt"]},
        controls={"ha0": np.zeros(C["nav"].size), "ha0_ext": np.zeros(C["nav"].size)},
        n_boot=N_BOOT)
    assert rep["verdicts"]["gstr"]["verdict"] == "FOLLOWS_NAV"
    assert rep["seam"]["reading"] == "SEAM_FAILURE_RIGHT_GOAL_WRONG_PATH"
    rep2 = nc.nav_compliance_report(
        readouts={"plan": follower, "gstr": scene, "anchor": scene},
        nav_true=C["nav"], nav_valid=C["valid"], nav_fed_by_cond=fed, eid=C["eid"],
        t_now_s=C["t_now"], t_start_s=C["ts"], t_end_s=C["te"],
        gt_signal={"plan": C["gt"], "gstr": C["gt"]},
        controls={"ha0": np.zeros(C["nav"].size), "ha0_ext": np.zeros(C["nav"].size)},
        n_boot=N_BOOT)
    assert rep2["seam"]["reading"] == "PATH_FOLLOWS_WITHOUT_GOAL"
    assert "selection_note" in rep2["seam"]


def test_anti_compliant_is_named():
    C = _corpus()
    rep = _report(lambda fed: -nc.commanded_side(fed) * 0.5, C)   # turns the WRONG way
    # under the TRUE token it complies on 0; under the shuffle it complies where
    # the fed side is the opposite of the true side — a NEGATIVE delta
    assert rep["verdicts"]["plan"]["verdict"] == "ANTI_COMPLIANT"


def test_fan_coverage_reports_when_the_vocabulary_cannot_comply():
    C = _corpus()
    n = C["nav"].size
    fan = {"nav_true": {"heading": np.zeros((n, 5)), "keep": np.ones((n, 5), bool)}}
    rep = _report(lambda fed: nc.commanded_side(fed) * 0.5, C, fan=fan)
    assert rep["fan_coverage"]["nav_true"]["fan_has_compliant_candidate"]["mean"] == 0.0


# --------------------------------------------------------------------------- #
# the label join                                                               #
# --------------------------------------------------------------------------- #
def test_time_base_control_picks_relative_when_the_turn_sits_at_t0_plus_rel():
    yaw = np.zeros(300)
    yaw[120:160] = np.linspace(0, math.radians(60), 40)    # turn at 12-16 s absolute
    yaw[160:] = math.radians(60)
    labels = {"c": {"nav_token": "NAV_TURN_L", "side": 1, "t0_s": 8.0,
                    "turns": [(4.0, 8.0, 60.0, True)],
                    "nav_time_rel_s": 4.0, "nav_distance_m": 30.0}}
    tb = nc.time_base_control({"c": yaw}, labels)
    assert tb["status"] == "OK" and tb["chosen"] == "relative"
    yaw2 = np.zeros(300)
    yaw2[40:80] = np.linspace(0, math.radians(60), 40)     # turn at 4-8 s absolute
    yaw2[80:] = math.radians(60)
    tb2 = nc.time_base_control({"c": yaw2}, labels)
    assert tb2["chosen"] == "absolute"
    tb3 = nc.time_base_control({"c": np.zeros(300)}, labels)
    assert tb3["status"] == "BROKEN" and tb3["chosen"] is None


def test_join_uses_the_commanded_side_turn_only():
    labels = {"c": {"nav_token": "NAV_TURN_R", "side": -1, "t0_s": 8.0,
                    "turns": [(2.0, 3.0, +20.0, True), (5.0, 9.0, -45.0, True)],
                    "nav_time_rel_s": 5.0, "nav_distance_m": 40.0},
              "f": {"nav_token": "NAV_FOLLOW_ROAD", "side": 0, "t0_s": 8.0, "turns": [],
                    "nav_time_rel_s": None, "nav_distance_m": None}}
    j = nc.join_windows_to_labels(["c", "f", "zz"], [1.0, 1.0, 1.0], labels)
    assert j["t_start_s"][0] == 13.0 and j["t_end_s"][0] == 17.0 and j["dyaw_deg"][0] == -45.0
    assert np.isnan(j["t_start_s"][1]) and j["side"][1] == 0 and j["has_record"][1]
    assert not j["has_record"][2]


def test_falsifiability_statement_is_present():
    assert "bijection" in nc.FALSIFIABILITY and "shuffle" in nc.FALSIFIABILITY.lower()


# --------------------------------------------------------------------------- #
# harvested from the predecessor's draft: strata, flip, deferred consistency  #
# --------------------------------------------------------------------------- #
def test_flip_swaps_left_and_right_only():
    nav = np.array([nc.NAV_FOLLOW, nc.NAV_LEFT, nc.NAV_RIGHT, nc.NAV_STRAIGHT])
    assert nc.flip_nav(nav).tolist() == [nc.NAV_FOLLOW, nc.NAV_RIGHT, nc.NAV_LEFT, nc.NAV_STRAIGHT]


def test_strata_are_assigned_from_timing_and_gt():
    nav = np.array([nc.NAV_LEFT] * 6 + [nc.NAV_FOLLOW, nc.NAV_LEFT])
    valid = np.array([True] * 7 + [False])
    t_now = np.array([0.0, 2.0, 2.0, 2.0, 9.0, 0.0, 2.0, 2.0])
    ts = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 20.0, np.nan, 4.0])
    te = np.array([8.0, 8.0, 8.0, 8.0, 8.0, 24.0, np.nan, 8.0])
    gt = np.array([0.5, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.5])
    s = nc.assign_strata(nav, valid, t_now, ts, te, gt, tau=0.2)
    assert s.tolist() == ["imminent", "imminent", "conflict", "ambiguous", "stale",
                          "deferred", "follow", "unlabeled"]


def test_flip_arm_reads_the_follower_and_the_scene_model_apart():
    C = _corpus()
    fed = {"nav_true": C["nav"], "nav_shuffled": _shuffle(C["nav"]),
           "nav_zero": np.zeros_like(C["nav"]), "nav_flipped": nc.flip_nav(C["nav"])}
    follower = {c: nc.commanded_side(f) * 0.5 for c, f in fed.items()}
    rep = nc.nav_compliance_report(
        readouts={"plan": follower}, nav_true=C["nav"], nav_valid=C["valid"],
        nav_fed_by_cond=fed, eid=C["eid"], t_now_s=C["t_now"], t_start_s=C["ts"],
        t_end_s=C["te"], gt_signal={"plan": C["gt"]},
        controls={"ha0": np.zeros(C["nav"].size), "ha0_ext": np.zeros(C["nav"].size)},
        n_boot=N_BOOT)
    plan = rep["readouts"]["plan"]
    assert plan["paired_true_minus_flipped"]["delta"] == 1.0
    assert plan["flipped"]["follows_FED_command"]["mean"] == 1.0
    assert plan["flipped"]["follows_TRUE_command"]["mean"] == 0.0
    scene = {c: C["gt"] for c in fed}
    rep2 = nc.nav_compliance_report(
        readouts={"plan": scene}, nav_true=C["nav"], nav_valid=C["valid"],
        nav_fed_by_cond=fed, eid=C["eid"], t_now_s=C["t_now"], t_start_s=C["ts"],
        t_end_s=C["te"], gt_signal={"plan": C["gt"]},
        controls={"ha0": np.zeros(C["nav"].size), "ha0_ext": np.zeros(C["nav"].size)},
        n_boot=N_BOOT)
    assert rep2["readouts"]["plan"]["paired_true_minus_flipped"]["delta"] == 0.0


def test_deferred_consistency_reads_hold_and_point():
    """Deferred windows: the turn is beyond the 6 s plan. A consistent follower
    HOLDS the plan (|heading| < tau) while g_str points the commanded way."""
    n_ep, n_win = 40, 12
    rng = np.random.default_rng(1)
    nav, valid, eid, t_now, ts, te, gt, arc = [], [], [], [], [], [], [], []
    for e in range(n_ep):
        turn = e % 2 == 0
        side = 1 if (e // 2) % 2 == 0 else -1
        for w in range(n_win):
            tn = 0.5 * w
            nav.append((nc.NAV_LEFT if side > 0 else nc.NAV_RIGHT) if turn else nc.NAV_FOLLOW)
            valid.append(True); eid.append(f"ep{e:03d}"); t_now.append(tn)
            ts.append(14.0 if turn else np.nan); te.append(18.0 if turn else np.nan)
            gt.append(rng.normal(0, 0.02))                  # nobody turns within 6 s
            arc.append(8.0 * (14.0 - tn) if turn else np.nan)   # 8 m/s to the turn
    nav, valid, eid = np.array(nav), np.array(valid), np.array(eid)
    fed = {"nav_true": nav, "nav_shuffled": _shuffle(nav), "nav_zero": np.zeros_like(nav)}
    plan = {c: np.zeros(nav.size) for c in fed}                     # holds
    gstr = {c: nc.commanded_side(f) * 0.4 for c, f in fed.items()}   # points as fed
    rep = nc.nav_compliance_report(
        readouts={"plan": plan, "gstr": gstr}, nav_true=nav, nav_valid=valid,
        nav_fed_by_cond=fed, eid=eid, t_now_s=np.array(t_now), t_start_s=np.array(ts),
        t_end_s=np.array(te), arc_to_turn_m=np.array(arc),
        gt_signal={"plan": np.array(gt), "gstr": np.array(gt)},
        tolerance={"plan": 0.2, "gstr": 0.1},
        controls={"ha0": np.zeros(nav.size), "ha0_ext": np.zeros(nav.size)}, n_boot=N_BOOT)
    assert rep["strata"]["census"]["imminent"] == 0
    cons = rep["consistency_deferred"]
    assert cons["powered"] and cons["n_windows"] > 0
    assert cons["conditionings"]["nav_true"]["plan_holds"]["mean"] == 1.0
    assert cons["conditionings"]["nav_true"]["gstr_points_commanded_way"]["mean"] == 1.0
    assert cons["conditionings"]["nav_true"]["consistent_hold_and_point"]["mean"] == 1.0
    assert cons["gstr_paired_true_minus_zero"]["delta"] == 1.0
    # the plan readout itself is UNPOWERED here (no imminent window) — and says so
    assert rep["verdicts"]["plan"]["verdict"] == "UNPOWERED"


def test_kinematic_end_signals_agree_with_echo_gate_geometry_and_the_heading_twin():
    v0, a0, k0 = np.array([8.0, 5.0]), np.array([-0.5, 0.2]), np.array([0.02, -0.03])
    e = nc.kinematic_end_signals(v0, a0, k0)
    assert np.allclose(e["plan"], nc.kinematic_heading(v0, a0, k0))
    assert e["plan_lateral"][0] > 0 and e["plan_lateral"][1] < 0         # sign = kappa sign
    assert np.sign(e["plan_bearing"]).tolist() == [1.0, -1.0]
    try:
        from tanitad.eval import echo_gate as eg
    except Exception:
        return
    xy = eg._np_extrapolate(8.0, -0.5, 0.02, 60)
    assert abs(xy[-1, 1] - e["plan_lateral"][0]) < 1e-9
    assert abs(math.atan2(xy[-1, 1], xy[-1, 0]) - e["plan_bearing"][0]) < 1e-9


def test_controls_are_scored_per_readout_with_matching_signals():
    C = _corpus()
    n = C["nav"].size
    ext = {"plan": np.zeros(n), "plan_bearing": np.zeros(n), "plan_lateral": np.zeros(n)}
    rep = _report(lambda fed: nc.commanded_side(fed) * 0.5, C,
                  extra_controls={"ha0_ext": ext})
    # plan_bearing / plan_lateral readouts were not supplied here, so no per_readout rows
    assert rep["controls"]["ha0_ext"]["per_readout"] == {}
    assert rep["controls"]["ha0_ext"]["compliance_with_TRUE_command"]["mean"] == 0.0
    assert rep["verdicts"]["gstr"].get("_floor_note")          # gstr has no kinematic floor


def test_unavailable_block_sits_at_every_registered_key():
    b = nc.unavailable_block("no sidecar", 12)
    for r in ("plan", "gstr", "anchor"):
        assert b["readouts"][r]["paired_true_minus_shuffled"]["status"] == "UNAVAILABLE"
        assert b["readouts"][r]["paired_true_minus_zero"]["reason"] == "no sidecar"
        assert b["readouts"][r]["conditionings"]["nav_true"]["compliance_with_TRUE_command"]["n"] == 12


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as ex:      # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(ex).__name__}: {ex}")
    sys.exit(1 if fails else 0)
