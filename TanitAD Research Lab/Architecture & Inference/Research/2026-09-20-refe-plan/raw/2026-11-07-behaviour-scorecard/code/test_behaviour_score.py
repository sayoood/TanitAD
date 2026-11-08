"""Tests for behaviour_score.py / behaviour_suite.py on SYNTHETIC rollouts (no sim). CPU only.
Run:  python test_behaviour_score.py [--only name1,name2]   (also pytest-compatible: plain functions, no fixtures)
Every rule has a positive control that PASSES and a deliberate-regression arm that must FAIL (BEHAVIOUR_SCORECARD section 6, G-METRIC).
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.environ.get("BEHAV_MODULE_DIR"):          # mutation_check.py points this at a mutated COPY of behaviour_score.py
    sys.path.insert(0, os.environ["BEHAV_MODULE_DIR"])

import numpy as np  # noqa: E402

import behaviour_score as B  # noqa: E402
import behaviour_suite as SU  # noqa: E402
import synth_rollouts as S  # noqa: E402

T20 = S.tgrid(20.0)


# ------------------------------------------------------------------------------------------------ helpers
def score(roll, hum=None):
    return B.score_arrays(roll, roll if hum is None else hum)


def st(sc, r):
    return sc["rules"][r]["status"]


def raw(sc, r):
    return sc["rules"][r]["raw_status"]


def cruise(v=15.0, t=T20, **kw):
    return S.make(t, np.full_like(t, v) if np.isscalar(v) else v, **kw)


def lead_scn(t, ego_x0, human_gap=30.0, v=15.0):
    """steady lead at constant speed; human follows at `human_gap` bumper gap; ego starts ego_x0 ahead along the path"""
    xl0 = S.EGO_L / 2 + human_gap + 2.25
    ag = [S.const_agent(10, t, xl0, 0.0, v)]
    return S.make(t, np.full_like(t, v), x0=ego_x0, agents=ag, category="steady_lead"), S.make(t, np.full_like(t, v), x0=0.0, agents=ag, category="steady_lead")


# ------------------------------------------------------------------------------------------------ instrument integrity
def test_vendored_navsim_is_pinned_and_tamper_is_refused():
    m, SI = B._navsim()
    assert SI.size() == 11 and hasattr(m, "ego_is_comfortable")
    tmp = Path(tempfile.mkdtemp(prefix="bv_"))
    try:
        shutil.copytree(B._VENDOR_DIR / "navsim_pdm", tmp / "navsim_pdm")
        p = tmp / "navsim_pdm" / "pdm_comfort_metrics.py"
        p.write_text(p.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
        old = (B._VENDOR_DIR, B._NAVSIM)
        B._VENDOR_DIR, B._NAVSIM = tmp, None
        try:
            B._navsim()
            raise AssertionError("tampered vendored file was accepted")
        except RuntimeError as e:
            assert "pinned" in str(e)
        finally:
            B._VENDOR_DIR, B._NAVSIM = old
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_every_rule_has_exactly_one_mutation_marker():
    src = (Path(B.__file__)).read_text(encoding="utf-8")
    for r in B.RULES:
        assert src.count(f"# MUT:{r}\n") == 1, r


def test_schema_validation_rejects_bad_input():
    good = cruise()
    assert B.validate(good, "rollout") == []
    bad = dict(good); bad.pop("ego_v")
    assert any("ego_v" in p for p in B.validate(bad, "rollout"))
    bad = dict(good); bad["t"] = good["t"][::2]                      # 10 Hz instead of 20 Hz
    assert any("uniform" in p for p in B.validate(bad, "rollout"))
    bad = dict(good); bad["category"] = np.array("nonsense")
    assert any("category" in p for p in B.validate(bad, "rollout"))
    bad = dict(good); bad["ego_x"] = good["ego_x"].copy(); bad["ego_x"][5] = np.nan
    assert any("non-finite" in p for p in B.validate(bad, "rollout"))
    try:
        B.score_arrays(bad, good)
        raise AssertionError("expected SchemaError")
    except B.SchemaError:
        pass


def test_resample_to_10hz_is_exact_subsample_when_aligned():
    r = cruise(np.linspace(10, 20, len(T20)))
    R = B.build_run(r)
    assert R["n"] == 201 and np.allclose(R["v"], r["ego_v"][::2]) and np.allclose(R["x"], r["ego_x"][::2])


# ------------------------------------------------------------------------------------------------ lead derivation (hand-built geometry)
def _geom(agents):
    t = S.tgrid(5.0)
    r = S.make(t, np.full_like(t, 10.0), agents=agents)
    sc, ticks = B.score_arrays(r, r)
    return sc, ticks


def test_lead_derivation_known_gap_ttc_timegap():
    t = S.tgrid(5.0)
    lead = S.const_agent(7, t, 32.25, 0.0, 5.0)                       # rear bumper at 30.0; ego front bumper at 2.4 -> gap 27.6
    off = S.const_agent(3, t, 15.0, 3.0, 10.0, w=2.0)                 # y in [2,4]: outside the 1.5 m half-width corridor
    behind = S.const_agent(5, t, -20.0, 0.0, 10.0)                    # in lane but behind the ego front bumper
    far = S.const_agent(9, t, 60.0, 0.0, 10.0)                        # in corridor, farther than the lead
    ped = S.const_agent(11, t, 12.0, 0.0, 0.0, cls=1, l=0.6, w=0.6)   # pedestrian in the lane: not a "vehicle" lead (A-2)
    _, tk = _geom([lead, off, behind, far, ped])
    assert tk["lead_id"][0] == 7
    assert abs(tk["gap"][0] - 27.6) < 1e-6, tk["gap"][0]
    assert abs(tk["ttc"][0] - 27.6 / 5.0) < 1e-6                       # v_rel = 10 - 5
    assert abs(tk["time_gap"][0] - 2.76) < 1e-6                        # gap / v_ego
    # 0.1 s later the gap shrinks by v_rel * 0.1
    assert abs(tk["gap"][1] - (27.6 - 0.5)) < 1e-6
    # corridor edge: lateral box edge at 1.4 m (< 1.5) is IN, at 1.6 m is OUT
    side_in = S.const_agent(5, t, 20.0, 2.4, 10.0, w=2.0)             # d range [1.4, 3.4]
    side_out = S.const_agent(5, t, 20.0, 2.6, 10.0, w=2.0)            # d range [1.6, 3.6]
    _, tk = _geom([lead, side_in])
    assert tk["lead_id"][0] == 5 and abs(tk["gap"][0] - (20.0 - 2.25 - 2.4)) < 1e-6
    _, tk = _geom([lead, side_out])
    assert tk["lead_id"][0] == 7


def test_free_road_has_both_clauses_gap_and_ttc():
    t = S.tgrid(6.0)
    def free0(lead_x0, lead_v):
        r = S.make(t, np.full_like(t, 15.0), agents=[S.const_agent(1, t, lead_x0, 0.0, lead_v)])
        _, tk = B.score_arrays(r, r)
        return bool(tk["free"][0])
    assert free0(2.4 + 2.25 + 70.0, 15.0) is True       # gap 70 > max(4*15, 15)=60 and not closing
    assert free0(2.4 + 2.25 + 50.0, 15.0) is False      # gap 50 <= 60
    assert free0(2.4 + 2.25 + 70.0, 5.0) is False       # gap 70 > 60 but TTC = 70/10 = 7 s <= 8 s: still a cause for slowing
    assert free0(2.4 + 2.25 + 70.0, 12.0) is True       # TTC = 70/3 = 23 s


# ------------------------------------------------------------------------------------------------ human log against itself (Gate H)
def test_human_scores_itself_exactly_on_a1_a2_e1_e2():
    h = cruise(15.0)
    sc, tk = score(h)
    for r in ("a1", "a2", "e1", "e2"):
        assert st(sc, r) == "PASS", r
    assert sc["rules"]["a1"]["value"] == 1.0 and sc["rules"]["a2"]["value"] == 0
    assert np.max(np.abs(tk["cross_track"])) < 1e-9 and np.max(np.abs(tk["head_err"])) < 1e-9
    assert sc["rules"]["e1"]["value"] == 0 and sc["rules"]["e2"]["value"]["p95_heading_err"] < 1e-9
    hc = cruise(10.0, curvature=0.03)                                   # a curved human path too
    sc2, tk2 = score(hc)
    assert all(st(sc2, r) == "PASS" for r in ("a1", "a2", "e1", "e2")) and np.max(np.abs(tk2["cross_track"])) < 1e-9
    assert sc2["rules"]["e2"]["value"]["p95_kappa_err"] < 1e-9


def test_positive_control_perturbed_ego_passes_every_applicable_rule():
    h = cruise(15.0)
    e = cruise(15.0 * 1.04, y=0.2)
    sc, _ = score(e, h)
    for r in ("a1", "a2", "d1", "d2", "e1", "e2", "f1", "f2"):
        assert st(sc, r) == "PASS", (r, sc["rules"][r])
    for r in ("b1", "b2", "c1", "c2", "c3", "c4"):
        assert st(sc, r) == "NA", r


# ------------------------------------------------------------------------------------------------ (a) free-road speed and phantom braking
def _phantom():
    return cruise(v=S.kf([(0, 15), (2.5, 15), (7.5, 7.5), (20, 7.5)], T20))


def test_a_phantom_braking_fails_a1_and_a2():
    sc, _ = score(_phantom(), cruise(15.0))
    assert raw(sc, "a1") == "FAIL" and raw(sc, "a2") == "FAIL", (sc["rules"]["a1"], sc["rules"]["a2"])
    assert sc["rules"]["a2"]["value"] == 1                              # overlapping windows merge into ONE event
    assert st(sc, "a1") == "FAIL" and st(sc, "a2") == "FAIL"           # human passes both -> not INFORMATIONAL


def test_a_half_speed_policy_fails_a1_and_a2():                          # G-METRIC arm: 0.5 * v_h on free road (ego starts at human speed)
    sc, _ = score(cruise(v=S.kf([(0, 15), (1, 15), (6, 7.5), (20, 7.5)], T20)), cruise(15.0))
    assert st(sc, "a1") == "FAIL" and st(sc, "a2") == "FAIL"


def test_a_human_explained_slowdown_is_not_phantom():
    prof = S.kf([(0, 15), (2.5, 15), (7.5, 7.5), (20, 7.5)], T20)
    sc, _ = score(cruise(v=prof), cruise(v=prof))                       # the human slows identically: geometry explains it
    assert st(sc, "a2") == "PASS" and st(sc, "a1") == "PASS"


def test_a1_over_speed_fails_two_sided():
    sc, _ = score(cruise(v=15.0 * 1.2), cruise(15.0))
    assert st(sc, "a1") == "FAIL"


# ------------------------------------------------------------------------------------------------ (b) lead time gap
def test_b_tailgating_at_08s_fails_b1_and_b2():
    e, h = lead_scn(T20, ego_x0=18.0)                                   # bumper gap 12 m at 15 m/s = 0.8 s; human keeps 2.0 s
    sc, _ = score(e, h)
    assert st(sc, "b1") == "FAIL" and st(sc, "b2") == "FAIL", (sc["rules"]["b1"], sc["rules"]["b2"])
    assert abs(sc["rules"]["b1"]["value"] - 0.8) < 1e-6
    assert sc["rules"]["b2"]["human_status"] == "PASS" and sc["rules"]["b1"]["human_status"] == "PASS"
    assert st(sc, "f2") == "PASS" and st(sc, "f1") == "PASS"            # tailgating is not a TTC event while speeds match


def test_b_follow_at_2s_passes_b1_and_b2():
    e, h = lead_scn(T20, ego_x0=0.0)
    sc, _ = score(e, h)
    assert st(sc, "b1") == "PASS" and st(sc, "b2") == "PASS" and abs(sc["rules"]["b1"]["value"] - 2.0) < 1e-6


def test_b2_persistence_two_tick_dip_is_logged_not_scored():
    t = T20
    e, h = lead_scn(t, ego_x0=0.0)
    x = e["ego_x"].copy(); v = e["ego_v"].copy()
    x[100:104] += 18.0                                                   # ego jumps toward the lead for 4 ticks of 20 Hz = 2 ticks at 10 Hz
    e["ego_x"] = x
    sc, _ = score(e, h)
    assert st(sc, "b2") == "PASS" and sc["rules"]["b2"]["details"]["n_short_runs_lt3"] >= 1


# ------------------------------------------------------------------------------------------------ (c) junction / light rules (synthetic slots)
LINE = 150.0
GREEN = np.full(len(S.tgrid(16.0)), 3, np.int8)


def _junction_green(prof_kf=None):
    t = S.tgrid(16.0)
    v = np.full_like(t, 12.0) if prof_kf is None else S.kf(prof_kf, t)
    return S.make(t, v, category="junction", synthetic=True, signal=GREEN, stop_line_s=LINE)


def test_c1_stall_at_clear_junction_fails():                              # G-METRIC arm: min speed 0.4 m/s before a clear line
    h = _junction_green()
    e = _junction_green([(0, 12), (8.5, 12), (11.5, 0.4), (16, 0.4)])
    sc, _ = score(e, h)
    assert st(sc, "c1") == "FAIL" and abs(sc["rules"]["c1"]["value"] - 0.4) < 1e-6, sc["rules"]["c1"]
    assert sc["rules"]["c1"]["human_status"] == "PASS"


def test_c1_flow_through_clear_junction_passes():
    sc, _ = score(_junction_green(), _junction_green())
    assert st(sc, "c1") == "PASS" and sc["rules"]["c1"]["value"] > 11.9


def test_c1_window_ends_when_signal_stops_being_clear_and_literal_mode_differs():
    # green -> red at t=10 while the ego (inside 40 m since t=9) is already braking to stop at the line: A-5
    t = S.tgrid(20.0)
    sig = np.where(t < 10.0, 3, 1).astype(np.int8)
    def scn():
        return S.make(t, S.stop_profile(t, 12.0, 0.0, 145.6), category="traffic_light", synthetic=True, signal=sig, stop_line_s=LINE)
    sc, _ = score(scn(), scn())
    assert st(sc, "c1") == "PASS" and sc["rules"]["c1"]["value"] > 10.0, sc["rules"]["c1"]
    old = B.P["C1_MODE"]
    B.P["C1_MODE"] = "literal"                         # the scorecard's wording read literally: window runs to the crossing / end
    try:
        sc2, _ = score(scn(), scn())
    finally:
        B.P["C1_MODE"] = old
    assert sc2["rules"]["c1"]["raw_status"] == "FAIL" and sc2["rules"]["c1"]["value"] < 0.1


def _red(x_stop):
    t = S.tgrid(20.0)
    return S.make(t, S.stop_profile(t, 12.0, 0.0, x_stop), category="traffic_light", synthetic=True,
                  signal=np.full(len(t), 1, np.int8), stop_line_s=LINE)


def test_c2_stop_on_red_passes_when_bumper_stops_2m_before_line():
    sc, _ = score(_red(145.6), _red(145.6))                               # front bumper 148.0, line at 150
    assert st(sc, "c2") == "PASS" and abs(sc["rules"]["c2"]["details"]["stop_distances_before_line_m"][0] - 2.0) < 0.05


def test_c2_stop_3m_past_line_on_red_fails():                             # G-METRIC arm
    sc, _ = score(_red(150.6), _red(145.6))
    assert st(sc, "c2") == "FAIL" and sc["rules"]["c2"]["value"] >= 1


def test_c2_crawl_stop_far_short_of_the_line_fails_the_6m_cap():
    sc, _ = score(_red(137.6), _red(145.6))                               # stops 10 m before the line
    assert st(sc, "c2") == "FAIL" and sc["rules"]["c2"]["details"]["n_stops_beyond_6m_cap"] == 1


ZONE = S.box_zone(152, 162, -2, 2)


def _conflict(t_cross, ego_v):
    t = S.tgrid(20.0)
    yy = 6.0 * (t_cross - t)
    ag = S.agent(20, np.full_like(t, 157.0), yy, np.full_like(t, 6.0), yaw=-np.pi / 2)
    return S.make(t, np.full_like(t, ego_v), agents=[ag], category="junction", synthetic=True, stop_line_s=LINE, zone=ZONE, conf_ids=[20])


def test_c3_entering_zone_with_conflicting_agent_within_2s_fails():
    sc, _ = score(_conflict(14.8, 12.0), _conflict(14.8, 8.0))           # ego enters ~12.5 s, agent in zone from ~14.1 s; human arrives after
    assert st(sc, "c3") == "FAIL" and sc["rules"]["c3"]["value"] == 1 and sc["rules"]["c3"]["details"]["rule_f_holds"] is True, sc["rules"]["c3"]
    assert sc["rules"]["c3"]["human_status"] == "PASS"


def test_c3_yielding_until_the_agent_has_cleared_passes():
    sc, _ = score(_conflict(8.0, 12.0), _conflict(8.0, 8.0))
    assert st(sc, "c3") == "PASS" and sc["rules"]["c3"]["details"]["ego_ever_in_zone"] is True


def _launch(kfp):
    t = S.tgrid(20.0)
    return S.make(t, S.kf(kfp, t), category="traffic_light", synthetic=True, signal=np.where(t < 10.0, 1, 3).astype(np.int8))


HUMAN_LAUNCH = [(0, 10), (2, 10), (6, 0), (10.3, 0), (14.3, 6), (20, 6)]


def test_c4_launch_within_3s_of_green_passes():
    sc, _ = score(_launch([(0, 10), (2, 10), (6, 0), (10.8, 0), (14.8, 6), (20, 6)]), _launch(HUMAN_LAUNCH))
    assert st(sc, "c4") == "PASS" and sc["rules"]["c4"]["details"]["events"][0]["launch_s"] <= 1.6


def test_c4_creep_stall_after_green_fails():
    sc, _ = score(_launch([(0, 10), (2, 10), (6, 0), (6.5, 0.3), (15, 0.3), (19, 6), (20, 6)]), _launch(HUMAN_LAUNCH))
    assert st(sc, "c4") == "FAIL" and sc["rules"]["c4"]["details"]["events"][0]["launch_s"] > 5.0


# ------------------------------------------------------------------------------------------------ AMENDMENT_3: CH-1 hold, CH-2 c1 NA, CH-3 c2 crossing event, CH-4 slot, Gate H
def _tl(v, sig, line=LINE, governs=None, t=None, **kw):
    t = S.tgrid(30.0) if t is None else t
    return S.make(t, v(t) if callable(v) else v, category="traffic_light", signal=sig(t) if callable(sig) else sig, stop_line_s=line, governs=governs, **kw)


def _stop_then_go(t, t_go=22.3, x_stop=145.6):
    """cruise 12 m/s, brake to stand at x_stop (front bumper 2 m before a line at 150), stand until t_go, then launch at 2.5 m/s^2 to 8 m/s"""
    return np.where(t < t_go, S.stop_profile(t, 12.0, 0.0, x_stop), np.minimum(8.0, 2.5 * (t - t_go)))


def _red_until(t_green):
    return lambda t: np.where(t < t_green, 1, 3).astype(np.int8)


def test_ch1_stationary_at_red_before_the_line_has_no_free_tick_a1_na_a2_na():          # mutation site: MUT:hold
    t = S.tgrid(20.0)
    e = S.make(t, np.zeros_like(t), x0=0.0, category="traffic_light", signal=np.ones(len(t), np.int8), stop_line_s=10.0)
    sc, tk = score(e, e)
    assert not tk["free"].any(), "a held tick must not be free road"
    assert raw(sc, "a1") == "NA" and raw(sc, "a2") == "NA", (raw(sc, "a1"), raw(sc, "a2"))


def test_ch1_launch_ticks_between_green_and_v_gt_1_are_held_and_ticks_after_are_free():
    t = S.tgrid(20.0)
    v = S.kf([(0, 0), (13, 0), (14, 4), (20, 4)], t)                                 # stands at the red, green at 10.0 s, starts moving at 13 s
    e = S.make(t, v, category="traffic_light", signal=np.where(t < 10.0, 1, 3).astype(np.int8), stop_line_s=40.0)
    sc, tk = score(e, e)
    tt, free = tk["t"], tk["free"]
    i_launch = int(np.argmax((tt >= 13.0) & (tk["v"] > 1.0)))
    assert abs(tt[i_launch] - 13.3) < 0.11, tt[i_launch]                              # v = 4 * (t-13) > 1 from 13.25 s on the 10 Hz grid
    assert not free[:i_launch].any(), "red ticks and the green->launch ticks (incl. the stationary 10-13 s) are held"
    assert free[i_launch:].all(), "the launch tick and everything after it is free road again"


def test_ch1_control_past_the_line_when_it_turns_red_stays_free_and_a1_is_scored():
    t = S.tgrid(20.0)
    sig = np.where(t < 6.0, 3, 1).astype(np.int8)                                    # ego (12 m/s) passes a line at 30 m at ~2.3 s, red from 6 s
    e = S.make(t, np.full_like(t, 12.0), category="traffic_light", signal=sig, stop_line_s=30.0)
    sc, tk = score(e, e)
    assert tk["free"][int(np.argmax(tk["t"] >= 2.0)):].all()
    assert raw(sc, "a1") == "PASS", sc["rules"]["a1"]


def test_ch1_control_signal_that_does_not_govern_the_ego_does_not_hold():
    t = S.tgrid(20.0)
    kw = dict(category="traffic_light", signal=np.ones(len(t), np.int8), stop_line_s=10.0)
    e = S.make(t, np.zeros_like(t), governs=[False], **kw)
    sc, tk = score(e, e)
    assert tk["free"][int(np.argmax(tk["t"] >= 2.0)):].all()
    assert raw(sc, "a1") == "FAIL", sc["rules"]["a1"]                               # stationary vs the 3 m/s floor: scored exactly as before A3
    held = S.make(t, np.zeros_like(t), governs=[True], **kw)
    assert raw(score(held, held)[0], "a1") == "NA"


def test_ch1_hold_starts_only_within_40m_of_the_line():
    t = S.tgrid(20.0)
    e = S.make(t, np.zeros_like(t), category="traffic_light", signal=np.ones(len(t), np.int8), stop_line_s=60.0)    # front bumper 2.4 m, line 57.6 m ahead
    sc, tk = score(e, e)
    assert tk["free"][int(np.argmax(tk["t"] >= 2.0)):].all(), "a stop 57.6 m short of a red is not excused as a hold"
    near = S.make(t, np.zeros_like(t), category="traffic_light", signal=np.ones(len(t), np.int8), stop_line_s=42.0)   # 39.6 m ahead
    assert not score(near, near)[1]["free"].any()


def test_ch1_control_a_scenario_with_no_signal_scores_bit_identically_to_before_amendment_3():
    t = S.tgrid(20.0)
    e, hh = lead_scn(T20, 18.0)
    phantom = cruise(np.where(t < 8, 15.0, 7.5))
    cases = {"lead": (e, hh), "conflict": (_conflict(14.8, 12.0), _conflict(14.8, 8.0)), "phantom": (phantom, cruise()),
             "green_signal_no_red": (_junction_green([(0, 12), (8.5, 12), (11.5, 0.4), (16, 0.4)]), _junction_green())}
    golden = {                                                                       # sha256 of json.dumps(score, sort_keys) / of the `free` series, from the PRE-A3 scorer (CODE_PIN.before_a3.json)
        "lead": ("e9227f9f7e362869fa718f2877384d6319f14d5f8ee5c14b1540dd2e952b28c7", "ab3df617aaa3140f04dc53f65b5446f34a6b2bdbb1f7b78db8db4d067ba14db9"),
        "conflict": ("665caf539726947af28c02b48a2ad38d408199b26f490e55d0f001ade7d83d8d", "e14e6efc585c2da5ff6283e333fe34edcf83bfbf102c027a5b0e7c552186e0cc"),
        "phantom": ("387204f55d37da576e08a2851fcaf462f58acd6dba0200c4ee463cbbfec7af1f", "e14e6efc585c2da5ff6283e333fe34edcf83bfbf102c027a5b0e7c552186e0cc"),
        "green_signal_no_red": ("5d587707d436fa809d2d1651565dafd485ea0ef156dc709ad182875776ef6440", "9c6b45e205361e87e516ed272f11bebcdaafc2ed0474088a5d76eafdb609c773"),
    }
    for k, (a, b) in cases.items():
        sc, tk = score(a, b)
        got = (hashlib.sha256(json.dumps(sc, sort_keys=True, allow_nan=False).encode()).hexdigest(), hashlib.sha256(np.asarray(tk["free"]).tobytes()).hexdigest())
        assert got == golden[k], (k, got)


def test_ch2_start_from_stop_at_green_is_c1_na_with_the_reason():                         # mutation site: MUT:c1na
    e = _tl(lambda t: _stop_then_go(t), _red_until(22.0))
    sc, _ = score(e, e)
    r = sc["rules"]["c1"]
    assert r["raw_status"] == "NA" and "stopped at the first clear tick" in r["details"]["reason"], r


def test_ch2_control_rolling_approach_that_then_stalls_below_1_mps_while_clear_fails():
    t = S.tgrid(16.0)
    e = S.make(t, S.kf([(0, 12), (8.5, 12), (11.5, 0.4), (16, 0.4)], t), category="junction", signal=np.full(len(t), 3, np.int8), stop_line_s=LINE)
    sc, _ = score(e, e)
    assert raw(sc, "c1") == "FAIL" and abs(sc["rules"]["c1"]["value"] - 0.4) < 1e-6, sc["rules"]["c1"]


def test_ch2_control_rolling_approach_holding_above_1_mps_passes():
    t = S.tgrid(20.0)
    e = S.make(t, S.kf([(0, 12), (8, 12), (11, 2.0), (14, 2.0), (18, 8)], t), category="junction", signal=np.full(len(t), 3, np.int8), stop_line_s=LINE)
    sc, _ = score(e, e)
    assert raw(sc, "c1") == "PASS" and sc["rules"]["c1"]["value"] >= 2.0 - 1e-6, sc["rules"]["c1"]


def test_ch2_a_brief_stop_shorter_than_1s_at_the_first_clear_tick_is_not_na():
    t = S.tgrid(20.0)
    v = S.kf([(0, 12), (8.5, 12), (9.0, 0.5), (9.5, 0.5), (9.6, 5), (14, 5), (16, 12), (20, 12)], t)    # v <= 1.0 for ~0.7 s only
    e = S.make(t, v, category="junction", signal=np.full(len(t), 3, np.int8), stop_line_s=LINE)
    sc, _ = score(e, e)
    assert raw(sc, "c1") != "NA", sc["rules"]["c1"]


def test_ch3_green_pass_then_red_later_is_not_a_crossing():                               # mutation site: MUT:c2x
    t = S.tgrid(20.0)
    sig = np.where(t < 15.0, 3, 1).astype(np.int8)
    e = S.make(t, np.full_like(t, 12.0), category="traffic_light", signal=sig, stop_line_s=100.0)      # passes the line at ~8.1 s, red from 15 s
    sc, _ = score(e, e)
    assert raw(sc, "c2") == "PASS" and sc["rules"]["c2"]["value"] == 0, sc["rules"]["c2"]


def test_ch3_control_crossing_on_red_is_one_event_and_fails():
    sc, _ = score(_red(150.6), _red(145.6))
    assert st(sc, "c2") == "FAIL" and sc["rules"]["c2"]["value"] == 1, sc["rules"]["c2"]


def test_ch4_slot_key_is_optional_validated_and_the_suite_reports_a_descriptive_per_slot_row():
    plain = cruise()
    with_slot = dict(plain, slot=np.array("TL-G"))
    assert B.validate(with_slot, "rollout") == [] and B.validate(plain, "rollout") == []
    assert any("slot" in p for p in B.validate(dict(plain, slot=np.array("nonsense")), "rollout"))
    sc_plain, _ = score(plain)
    sc_slot, _ = score(with_slot)
    assert "slot" not in sc_plain and sc_slot["slot"] == "TL-G"
    sc_slot.pop("slot")
    assert json.dumps(sc_plain, sort_keys=True) == json.dumps(sc_slot, sort_keys=True)       # the slot key changes nothing else
    a = dict(sc_plain, scenario_id="a", log_id="a", category="traffic_light", slot="TL-G")
    b = dict(sc_plain, scenario_id="b", log_id="b", category="traffic_light", slot="TL-S")
    c = dict(sc_plain, scenario_id="c", log_id="c", category="traffic_light")
    blk = SU.aggregate([a, b, c])["blocks"]["nuplan_seeded"]
    assert set(blk["per_slot"]) == {"TL-G", "TL-S"} and blk["per_slot"]["TL-G"]["n_scenarios"] == 1 and blk["per_category"]["traffic_light"]["n_scenarios"] == 3
    assert "per_slot" not in SU.aggregate([c])["blocks"]["nuplan_seeded"]


def test_gate_h_stop_at_red_then_launch_on_green_reads_a1_na_or_pass_c1_na_c2_c4_pass_a2_not_fail():
    h = _tl(lambda t: _stop_then_go(t), _red_until(22.0))
    sc, tk = score(h, h)
    r = sc["rules"]
    assert r["a1"]["human_status"] in ("PASS", "NA"), r["a1"]
    assert r["c1"]["human_status"] == "NA" and "start-from-stop" in r["c1"]["details"]["reason"], r["c1"]
    assert r["c2"]["human_status"] == "PASS", r["c2"]
    assert r["c4"]["human_status"] == "PASS", r["c4"]
    assert r["a2"]["human_status"] != "FAIL", r["a2"]
    assert r["c2"]["details"]["stop_distances_before_line_m"][0] < 6.0


def _ref_case(t_valid=None):
    """ego = human until 10 s, then (rollout only) a 0.8 m lateral offset for 5 s, a 0.15 rad heading bias and half the speed; a green->stop->go trigger at 12 s"""
    t = S.tgrid(20.0)
    late = t >= 10.0
    v_h = np.full_like(t, 15.0)
    v_e = np.where(late, 7.5, 15.0)
    h = S.make(t, v_h, category="traffic_light")
    kw = {} if t_valid is None else dict(human_ref_valid_until_s=np.array(t_valid, float))
    e = S.make(t, v_e, y=np.where((t >= 10.0) & (t < 15.0), 0.8, 0.0), yaw=np.where(late, 0.15, 0.0), category="traffic_light")
    e.update(kw)
    return e, h


def test_a5_human_ref_valid_until_makes_a1_a2_e1_e2_na_after_it_and_reports_the_count():     # mutation site: MUT:refvalid
    e, h = _ref_case(9.0)
    sc, _ = score(e, h)
    r = sc["rules"]
    assert [r[k]["raw_status"] for k in ("a1", "a2", "e1", "e2")] == ["PASS", "PASS", "PASS", "PASS"], {k: r[k]["raw_status"] for k in ("a1", "a2", "e1", "e2")}
    assert r["e1"]["details"]["human_ref_valid_until_s"] == 9.0 and r["e1"]["details"]["n_ticks_na_after_human_ref"] == 110, r["e1"]["details"]   # 10 Hz ticks 9.1 .. 20.0 s
    assert sc["human_ref_valid_until_s"] == 9.0
    # c4 is also bounded: a trigger after the valid time is NA
    t = S.tgrid(20.0)
    mk = lambda **kw: dict(S.make(t, S.kf([(0, 10), (2, 10), (6, 0), (10.8, 0), (14.8, 6), (20, 6)], t), category="traffic_light", signal=np.where(t < 10.0, 1, 3).astype(np.int8)), **kw)
    assert raw(score(mk(), mk())[0], "c4") == "PASS"
    c4 = score(mk(human_ref_valid_until_s=np.array(8.0)), mk())[0]["rules"]["c4"]
    assert c4["raw_status"] == "NA" and c4["details"]["n_ticks_na_after_human_ref"] > 0, c4
    # a key at or beyond the end of the run changes no verdict
    e2, _ = _ref_case(25.0)
    assert [score(e2, h)[0]["rules"][k]["raw_status"] for k in ("a1", "a2", "e1", "e2")] == [score(_ref_case()[0], h)[0]["rules"][k]["raw_status"] for k in ("a1", "a2", "e1", "e2")]
    bad = dict(e, human_ref_valid_until_s=np.array(-1.0))
    assert any("human_ref_valid_until_s" in p for p in B.validate(bad, "rollout"))


def test_a5_control_without_the_key_the_same_rollout_fails_a1_a2_e1_e2_and_output_has_no_new_key():
    e, h = _ref_case()
    sc, _ = score(e, h)
    r = sc["rules"]
    assert [r[k]["raw_status"] for k in ("a1", "a2", "e1", "e2")] == ["FAIL", "FAIL", "FAIL", "FAIL"], {k: r[k]["raw_status"] for k in ("a1", "a2", "e1", "e2")}
    assert "human_ref_valid_until_s" not in sc and "n_ticks_na_after_human_ref" not in json.dumps(sc)


# ------------------------------------------------------------------------------------------------ (d) comfort and oscillation
def _windows_share(roll):
    """independent d1 re-computation: build the NAVSIM state array from the documented mapping and call ego_is_comfortable directly"""
    m, SI = B._navsim()
    t, v, a, yaw, x, y, stv = (roll[k][::2] for k in ("t", "ego_v", "ego_a", "ego_yaw", "ego_x", "ego_y", "ego_steer"))
    yr = np.gradient(np.unwrap(yaw), 0.1)
    st_ = np.zeros((len(t), 11))
    st_[:, 0], st_[:, 1], st_[:, 2], st_[:, 3] = x, y, np.unwrap(yaw), v
    st_[:, 5], st_[:, 6], st_[:, 7], st_[:, 8] = a, v * yr, stv, np.gradient(stv, 0.1)
    st_[:, 9], st_[:, 10] = yr, np.gradient(yr, 0.1)
    ok = [bool(m.ego_is_comfortable(st_[None, i:i + 41], 0.1 * np.arange(41)).all()) for i in range(20, len(t) - 40, 10)]
    return float(np.mean(ok)), len(ok)


def test_d_oscillating_speed_fails_d2():                                   # G-METRIC arm: +-1.5 m/s, 5 s period
    t = T20
    osc = cruise(v=20.0 + 1.5 * np.sin(2 * np.pi * t / 5.0))
    sc, _ = score(osc, cruise(20.0))
    assert st(sc, "d2") == "FAIL" and sc["rules"]["d2"]["value"] >= 1, sc["rules"]["d2"]
    exp, n = _windows_share(osc)
    assert abs(sc["rules"]["d1"]["value"] - exp) < 1e-12 and sc["rules"]["d1"]["n"] == n


def test_d1_harsh_braking_fails_and_matches_navsim_on_the_same_states():
    brake = cruise(v=S.kf([(0, 20), (8, 20), (10, 8), (20, 8)], T20))      # -6 m/s^2 for 2 s: one swing, not an oscillation
    sc, _ = score(brake, cruise(20.0))
    exp, n = _windows_share(brake)
    assert 0.0 < exp < 0.95 and abs(sc["rules"]["d1"]["value"] - exp) < 1e-12
    assert st(sc, "d1") == "FAIL"
    assert st(sc, "d2") == "PASS"          # "a single large slow-down is rule a2/c, not an oscillation"


def test_d1_threshold_relaxes_to_human_share_minus_005():
    t = T20
    harsh = cruise(v=20.0 + 3.0 * np.sin(2 * np.pi * t / 2.0))
    sc, _ = score(harsh, harsh)                                              # the human is equally uncomfortable
    assert sc["rules"]["d1"]["value"] < 0.2 and abs(sc["rules"]["d1"]["threshold"] - (sc["rules"]["d1"]["human_value"] - 0.05)) < 1e-12
    assert st(sc, "d1") == "PASS"


def test_d2_small_wiggle_and_monotone_accel_pass():
    sc, _ = score(cruise(v=20.0 + 0.4 * np.sin(2 * np.pi * T20 / 5.0)), cruise(20.0))
    assert st(sc, "d2") == "PASS"
    sc, _ = score(cruise(v=S.kf([(0, 15), (20, 17)], T20)), cruise(15.0))
    assert st(sc, "d2") == "PASS" and st(sc, "d1") == "PASS"


# ------------------------------------------------------------------------------------------------ (e) lane keeping
def _drift(amp):
    y = amp * (np.clip(T20 - 6, 0, 1) - np.clip(T20 - 9, 0, 1))
    return cruise(15.0, y=y)


def test_e1_lane_drift_08m_for_3s_fails():                                  # G-METRIC arm
    sc, _ = score(_drift(0.8), cruise(15.0))
    assert st(sc, "e1") == "FAIL" and sc["rules"]["e1"]["details"]["max_abs_cross_track"] > 0.79


def test_e1_offset_of_03m_passes_and_a_short_excursion_is_not_a_run():
    sc, _ = score(_drift(0.3), cruise(15.0))
    assert st(sc, "e1") == "PASS"
    y = np.where((T20 > 6) & (T20 < 7.2), 0.9, 0.0)                         # 1.2 s excursion: shorter than the 2.0 s horizon
    sc, _ = score(cruise(15.0, y=y), cruise(15.0))
    assert st(sc, "e1") == "PASS"


def test_e2_heading_bias_fails_and_curvature_ringing_fails_but_clean_passes():
    sc, _ = score(cruise(15.0, yaw=0.15), cruise(15.0))
    assert st(sc, "e2") == "FAIL" and not sc["rules"]["e2"]["details"]["heading_ok"] and st(sc, "e1") == "PASS"
    ring = 0.04 * np.sin(2 * np.pi * T20 / 1.0)
    sc, _ = score(cruise(10.0, yaw=ring), cruise(10.0))
    assert st(sc, "e2") == "FAIL" and sc["rules"]["e2"]["details"]["heading_ok"] and not sc["rules"]["e2"]["details"]["kappa_ok"]
    sc, _ = score(cruise(10.0, curvature=0.03), cruise(10.0, curvature=0.03))
    assert st(sc, "e2") == "PASS"


# ------------------------------------------------------------------------------------------------ (f) collisions and TTC
def _obstacle_scn(ego_stop=None, ego_v=10.0):
    t = S.tgrid(14.0)
    ob = S.const_agent(30, t, 100.0, 0.0, 0.0)
    v = np.full_like(t, ego_v) if ego_stop is None else S.stop_profile(t, ego_v, 0.0, ego_stop)
    return S.make(t, v, agents=[ob], category="lead_braking")


def test_f_overlap_with_a_stationary_agent_fails_f1_and_f2():
    h = _obstacle_scn(ego_stop=100.0 - 2.25 - 5.0 - 2.4)                      # human stands 5 m short
    sc, _ = score(_obstacle_scn(), h)
    assert st(sc, "f1") == "FAIL" and st(sc, "f2") == "FAIL", (sc["rules"]["f1"], sc["rules"]["f2"])
    assert sc["rules"]["f1"]["details"]["events"][0]["kind"] == "STOPPED_TRACK" and sc["rules"]["f1"]["value"] == 1
    assert sc["rules"]["f1"]["human_status"] == "PASS" and sc["rules"]["f2"]["human_status"] == "PASS"


def test_f_braking_in_time_passes_f1_and_f2():
    h = _obstacle_scn(ego_stop=100.0 - 2.25 - 5.0 - 2.4)
    sc, _ = score(h, h)
    assert st(sc, "f1") == "PASS" and st(sc, "f2") == "PASS"


def test_f_constant_speed_through_a_braking_lead_fails_f1_and_f2():         # G-METRIC arm
    t = S.tgrid(12.0)
    lv = S.kf([(0, 15), (2, 15), (8, 0), (12, 0)], t)
    lead = S.agent(31, S.integrate(lv, S.EGO_L / 2 + 30.0 + 2.25), np.zeros_like(t), lv)
    hum = S.make(t, lv, agents=[lead], category="lead_braking")             # human follows the lead's speed profile at 30 m
    ego = S.make(t, np.full_like(t, 15.0), agents=[lead], category="lead_braking")
    sc, _ = score(ego, hum)
    assert st(sc, "f1") == "FAIL" and st(sc, "f2") == "FAIL"
    assert sc["rules"]["f1"]["details"]["events"][0]["kind"] == "ACTIVE_FRONT"
    assert sc["rules"]["f1"]["human_status"] == "PASS"


def test_f1_rear_end_hit_on_the_ego_is_not_at_fault():
    t = S.tgrid(8.0)
    v = np.full_like(t, 5.0)
    hit = S.agent(40, S.integrate(np.full_like(t, 12.0), -30.0), np.zeros_like(t), np.full_like(t, 12.0))   # faster agent from behind
    sc, _ = score(S.make(t, v, agents=[hit]), S.make(t, v, agents=[hit]))
    ev = sc["rules"]["f1"]["details"]["events"]
    assert ev and ev[0]["kind"] == "ACTIVE_REAR" and sc["rules"]["f1"]["value"] == 0 and sc["rules"]["f1"]["details"]["n_overlap_events_total"] >= 1


def test_cut_in_agent_absent_then_entering_the_corridor_is_a_lead_only_from_then():
    t = S.tgrid(12.0)
    yy = np.clip(4.0 - 2.0 * (t - 4.0), 0.0, 4.0)                          # starts 4 m left of the path, moves in at 2 m/s from t=4 s, centred at t=6 s
    present = t >= 3.0                                                    # appears at t=3 s (NaN padded before)
    cut = S.agent(50, 40.0 + 15.0 * t, yy, np.full_like(t, 15.0), present=present)
    r = S.make(t, np.full_like(t, 15.0), agents=[cut], category="merge_cutin")
    sc, tk = B.score_arrays(r, r)
    first = int(np.argmax(tk["lead_id"] == 50))
    # box half-width 1.0 + corridor 1.5: the agent joins the corridor once its centre offset <= 2.5 m  ->  t = 4.75 s
    assert tk["lead_id"][0] == -1 and abs(tk["t"][first] - 4.8) < 0.11, tk["t"][first]
    assert np.all(tk["lead_id"][first:] == 50) and not np.any(np.isnan(tk["gap"][first:]))


def test_scale_many_agents_runs_fast_and_deterministically():
    import time
    rng = np.random.default_rng(7)
    t = S.tgrid(17.0)
    ags = [S.const_agent(i, t, float(rng.uniform(-50, 300)), float(rng.uniform(-12, 12)), float(rng.uniform(0, 15))) for i in range(100)]
    r = S.make(t, np.full_like(t, 12.0), agents=ags)
    t0 = time.time()
    a = json.dumps(B.score_arrays(r, r)[0], sort_keys=True)
    dt = time.time() - t0
    assert dt < 20.0, dt
    assert a == json.dumps(B.score_arrays(r, r)[0], sort_keys=True)


# ------------------------------------------------------------------------------------------------ Gate H, determinism, files
def test_gate_h_human_failed_rule_becomes_informational():
    e, _ = lead_scn(T20, ego_x0=0.0, human_gap=12.0)                           # human AND ego tailgate at 0.8 s
    sc, _ = score(e, e)
    assert raw(sc, "b1") == "FAIL" and st(sc, "b1") == "INFORMATIONAL" and st(sc, "b2") == "INFORMATIONAL"


def test_determinism_two_runs_bit_identical():
    e, h = lead_scn(T20, ego_x0=18.0)
    a = json.dumps(score(e, h)[0], sort_keys=True)
    b = json.dumps(score(e, h)[0], sort_keys=True)
    assert a == b and hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(b.encode()).hexdigest()


def test_file_roundtrip_and_cli_output_is_byte_identical_across_runs():
    tmp = Path(tempfile.mkdtemp(prefix="bv_"))
    try:
        e, h = lead_scn(T20, ego_x0=18.0)
        np.savez(tmp / "rollout.npz", **e)
        np.savez(tmp / "human.npz", **h)
        outs = []
        for k in range(2):
            B.main(["score", "--rollout", str(tmp / "rollout.npz"), "--human", str(tmp / "human.npz"), "--out", str(tmp / f"o{k}.json"), "--dump-ticks", str(tmp / f"t{k}.npz")])
            outs.append((tmp / f"o{k}.json").read_bytes())
        assert outs[0] == outs[1]
        j = json.loads(outs[0])
        assert j["rules"]["b2"]["status"] == "FAIL" and j["families"]["STRATEGIC"].startswith("PAUSED") and j["inputs"]["rollout_sha256"]
        assert np.array_equal(np.load(tmp / "t0.npz")["gap"], np.load(tmp / "t1.npz")["gap"], equal_nan=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_epdms_is_passed_through_and_identity_gate_reads_the_record():
    e = cruise(15.0, epdms=0.789)
    sc, _ = score(e)
    assert abs(sc["epdms"]["value"] - 0.789) < 1e-12 and sc["identity"]["complete"] is True and sc["category"] == "cruise"


def test_human_file_may_omit_agents_and_scenario_fields():
    e, h = lead_scn(T20, ego_x0=0.0)
    hm = {k: h[k] for k in B.EGO_KEYS}
    sc, _ = score(e, hm)
    assert st(sc, "b1") == "PASS"


# ------------------------------------------------------------------------------------------------ suite: aggregate + paired estimator
def fake(sid, rules, epdms=None, synthetic=False, human="PASS", cat="cruise", log=None):
    rr = {r: dict(status="NA", raw_status="NA", human_status="NA") for r in SU.RULES}
    for r, s in rules.items():
        rr[r] = dict(status=s, raw_status=s, human_status=human)
    return dict(scenario_id=sid, log_id=log or sid, category=cat, synthetic=synthetic, rules=rr, scorecard_sha256="x",
                epdms=dict(value=epdms), identity=dict(checkpoint_sha256="c", dtype="fp32", seed=1, suite_tokens_sha256="s", complete=True))


def test_aggregate_counts_blocks_and_families():
    scs = [fake("a", {"a1": "PASS", "e1": "PASS"}), fake("b", {"a1": "FAIL", "e1": "PASS"}), fake("c", {"a1": "PASS"}, cat="steady_lead"),
           fake("s1", {"c2": "FAIL"}, synthetic=True, cat="traffic_light")]
    agg = SU.aggregate(scs)
    nb, sb = agg["blocks"]["nuplan_seeded"], agg["blocks"]["synthetic_until_map_lifting"]
    assert nb["per_rule"]["a1"]["passes"] == 2 and nb["per_rule"]["a1"]["fails"] == 1 and abs(nb["per_rule"]["a1"]["score"] - 2 / 3) < 1e-12
    assert nb["per_rule"]["c2"]["n"] == 0 and sb["per_rule"]["c2"]["fails"] == 1        # synthetic never pooled into the nuPlan count
    assert nb["families"]["STRATEGIC"].startswith("PAUSED") and "N/A" in nb["families"]["TACTICAL"]["executed_decision_agreement"]
    assert nb["per_category"]["steady_lead"]["rules"]["a1"]["passes"] == 1 and "SYNTHETIC UNTIL MAP LIFTING" in sb["label"]


def test_gate_h_rule_definition_defect_over_30_percent_is_not_applied():
    scs = [fake(f"s{i}", {"b1": "FAIL"}, human=("FAIL" if i < 4 else "PASS")) for i in range(10)]       # human fails b1 on 40 %
    agg = SU.aggregate(scs)["blocks"]["nuplan_seeded"]
    assert agg["gate_h"]["b1"]["defect"] is True and agg["per_rule"]["b1"]["n"] == 0 and agg["per_rule"]["b1"]["informational"] == 10


def test_bootstrap_is_seeded_clustered_and_known():
    d = [1.0] * 10
    b = SU.paired_bootstrap(d, list(range(10)))
    assert b["mean"] == 1.0 and b["lo"] == 1.0 and b["hi"] == 1.0 and b["draws"] == 10000
    z = SU.paired_bootstrap([0.0] * 10, list(range(10)))
    assert z["lo"] == 0.0 and z["hi"] == 0.0                                  # no difference: the interval does not exclude 0
    d2 = [1, 1, 1, 1, 0, 0, 0, 0, -1, 0]
    r1, r2 = SU.paired_bootstrap(d2, list(range(10))), SU.paired_bootstrap(d2, list(range(10)))
    assert r1 == r2 and r1["lo"] < 0.3 < r1["hi"]
    # cluster = log: ten scenarios in two logs behave like two clusters (very wide)
    c = SU.paired_bootstrap([1.0] * 5 + [-1.0] * 5, ["x"] * 5 + ["y"] * 5)
    assert c["n_clusters"] == 2 and c["lo"] == -1.0 and c["hi"] == 1.0


def _suite(n, rule_status, epd=0.80):
    return [fake(f"s{i:02d}", {"a1": rule_status(i), "e1": "PASS", "f1": "PASS"}, epdms=epd) for i in range(n)]


def test_compare_rejects_a_separated_regression_above_the_floor():
    base = _suite(10, lambda i: "PASS")
    cand = _suite(10, lambda i: "FAIL" if i < 8 else "PASS")
    res = SU.compare(cand, base, rep=_suite(10, lambda i: "PASS"), tier=2)
    assert res["verdict"].startswith("REJECT") and "nuplan_seeded:a1" in res["cond1_no_rule_worse"]["worse"]
    assert res["next_arm"]["failing_rules"] == ["a1"] and "P-1" in res["next_arm"]["proxies_that_should_have_warned"]["a1"]


def test_compare_effect_inside_the_sim_floor_is_not_a_regression():
    base = _suite(10, lambda i: "PASS")
    rep = _suite(10, lambda i: "FAIL" if i < 9 else "PASS")                  # the baseline replicate itself differs by 0.9
    cand = _suite(10, lambda i: "FAIL" if i < 8 else "PASS")
    res = SU.compare(cand, base, rep=rep, tier=2)
    assert not res["verdict"].startswith("REJECT") and res["blocks"]["nuplan_seeded"]["rules"]["a1"]["floor"] == 0.9


def test_compare_tier1_can_nominate_never_adopt_and_tier2_adopts_one_seed():
    base = _suite(10, lambda i: "FAIL" if i < 8 else "PASS")
    cand = _suite(10, lambda i: "PASS")
    rep = _suite(10, lambda i: "FAIL" if i < 8 else "PASS")
    pd = dict(diff_x100=[0.1] * 50, log_id=list(range(50)))
    t1 = SU.compare(cand, base, rep=rep, tier=1, pdms=pd)
    t2 = SU.compare(cand, base, rep=rep, tier=2, pdms=pd)
    assert t1["verdict"].startswith("CANDIDATE") and t2["verdict"] == "ADOPTED" and t2["scope_label"].startswith("ONE-SEED")
    assert "nuplan_seeded:a1" in t2["cond2_a_failing_rule_fixed"]["fixed"]
    t2b = SU.compare(cand, base, rep=rep, tier=2, pdms=dict(diff_x100=[-2.0] * 50, log_id=list(range(50))))
    assert t2b["verdict"].startswith("NOT ADOPTED") and t2b["cond3_pdms"]["holds"] is False


def test_compare_epdms_guard_vetoes_and_missing_replicate_is_not_evaluable():
    base = _suite(10, lambda i: "FAIL" if i < 8 else "PASS", epd=0.80)
    cand = _suite(10, lambda i: "PASS", epd=0.70)
    pd = dict(diff_x100=[0.1] * 50, log_id=list(range(50)))
    res = SU.compare(cand, base, rep=base, tier=2, pdms=pd)
    assert res["epdms_guard"]["veto"] is True and res["verdict"] == "NOT ADOPTED (EPDMS guard veto)"
    assert SU.compare(cand, base, rep=None, tier=2, pdms=pd)["verdict"].startswith("NOT EVALUABLE")


def test_compare_does_not_pool_the_synthetic_block():
    base = [fake(f"n{i}", {"a1": "PASS"}) for i in range(10)] + [fake(f"x{i}", {"c2": "PASS"}, synthetic=True) for i in range(10)]
    cand = [fake(f"n{i}", {"a1": "PASS"}) for i in range(10)] + [fake(f"x{i}", {"c2": "FAIL"}, synthetic=True) for i in range(10)]
    res = SU.compare(cand, base, rep=base, tier=2)
    assert "synthetic_until_map_lifting:c2" in res["cond1_no_rule_worse"]["worse"] and "nuplan_seeded:c2" not in res["cond1_no_rule_worse"]["worse"]


# ------------------------------------------------------------------------------------------------ (f1) A-10: NAVSIM's own at-fault logic, hand-built geometry
# Ego: Pacifica box 5.176 x 2.297 m, rear axle 1.461 m behind the box centre, travelling +x from x = 0 at t = 0. Agents are 4.5 x 2.0 m boxes.
_NAV_SRC_UTILS = Path("F:/Archive/devbox-C/navsim/devkit/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer_utils.py")
_NUP_SRC_UTILS = Path("C:/Users/Admin/dz/DriveZero/DriveRL/nuplan-devkit/nuplan/planning/simulation/observation/idm/utils.py")
TH = np.pi / 2


def _f1(ego, hum=None):
    # assertions below read the rollout's OWN verdict (raw_status): with the ego as its own human reference a FAIL would be Gate H INFORMATIONAL
    sc, _ = score(ego, ego if hum is None else hum)
    return sc["rules"]["f1"], sc


def _front_scn(ego_v, ag_v, ag_x0, ag_yaw=0.0, t=None):
    """agent on the ego's path (y = 0) at x0 moving along its yaw at speed ag_v; the two meet head-on or in a rear approach"""
    t = S.tgrid(12.0) if t is None else t
    ag = S.agent(60, ag_x0 + ag_v * np.cos(ag_yaw) * t, np.zeros_like(t), np.full_like(t, ag_v), yaw=ag_yaw)
    return S.make(t, np.full_like(t, ego_v), agents=[ag])


def _chk_front_hit_on_a_moving_agent():
    f, _ = _f1(_front_scn(15.0, 5.0, 60.0))
    assert f["raw_status"] == "FAIL" and f["value"] == 1 and f["details"]["events"][0]["kind"] == "ACTIVE_FRONT", f
    assert f["details"]["events"][0]["navsim_no_at_fault_score"] == 0.0


def _chk_ego_stopped_threshold():            # NAVSIM: ego "stopped" is speed <= 5e-02 m/s (pdm_scorer_utils.py:12 default)
    f4, _ = _f1(_front_scn(0.04, 3.0, 30.0, ag_yaw=np.pi))                       # head-on into a (nearly) standing ego
    assert f4["raw_status"] == "PASS" and f4["details"]["events"][0]["kind"] == "STOPPED_EGO" and not f4["details"]["events"][0]["at_fault"], f4
    f6, _ = _f1(_front_scn(0.06, 3.0, 30.0, ag_yaw=np.pi))                       # 0.06 m/s is NOT stopped: the same hit is a front hit
    assert f6["raw_status"] == "FAIL" and f6["details"]["events"][0]["kind"] == "ACTIVE_FRONT", f6


def _chk_stopped_track_is_at_fault():        # NAVSIM: track "stopped" is speed <= 5e-02 m/s (is_track_stopped), tested before rear / front / lateral
    t = S.tgrid(12.0)
    for v, kind in ((0.0, "STOPPED_TRACK"), (0.04, "STOPPED_TRACK"), (0.06, "ACTIVE_FRONT")):
        f, _ = _f1(S.make(t, np.full_like(t, 10.0), agents=[S.agent(61, np.full_like(t, 80.0), np.zeros_like(t), np.full_like(t, v))]))
        assert f["raw_status"] == "FAIL" and f["details"]["events"][0]["kind"] == kind, (v, f)


def _side_scn(ct):
    """agent crosses from the ego's left at 3 m/s (yaw -90 deg) and meets the ego's left flank at t = 5 s, ego box centre abreast of the agent centre,
    ego driving `ct` m to the left of the human path -> ACTIVE_LATERAL (front bumper clears the agent, agent centre ~66 deg off the heading)"""
    t = S.tgrid(12.0)
    ag = S.agent(62, np.full_like(t, 50.0), (ct + 3.4065 + 15.0) - 3.0 * t, np.full_like(t, 3.0), yaw=-TH)
    return S.make(t, np.full_like(t, 10.0), y=ct, agents=[ag]), S.make(t, np.full_like(t, 10.0), agents=[ag])


def _chk_lateral_needs_the_lane_proxy():
    for ct, want in ((0.3, "PASS"), (0.8, "FAIL")):
        ego, hum = _side_scn(ct)
        f, sc = _f1(ego, hum)
        e = f["details"]["events"][0]
        assert e["kind"] == "ACTIVE_LATERAL" and e["lane_proxy_used"] and f["lane_proxy"] is True and f["raw_status"] == want, (ct, f)
        assert e["at_fault"] == (want == "FAIL")


def test_f1_front_bumper_hit_on_a_moving_agent_is_at_fault():
    _chk_front_hit_on_a_moving_agent()


def _runs_of(roll, a=0):
    Rc = B.build_run(roll)
    path = B.HumanPath(Rc)
    Dc = B.derive(Rc, path)
    pr = Rc["present"][a] & np.isfinite(Rc["ax"][a])
    ov = B._sat_overlap(B._corners(Rc["x"], Rc["y"], Rc["yaw"], Rc["L"], Rc["W"]),
                        B._corners(np.nan_to_num(Rc["ax"][a]), np.nan_to_num(Rc["ay"][a]), np.nan_to_num(Rc["ayaw"][a]), Rc["al"][a], Rc["aw"][a])) & pr
    return Rc, Dc, B._runs(ov)


def test_f1_rear_end_by_a_faster_agent_is_not_at_fault_and_the_exemption_holds_inside_the_run():
    t = S.tgrid(8.0)
    hit = S.agent(40, S.integrate(np.full_like(t, 12.0), -30.0), np.zeros_like(t), np.full_like(t, 12.0))     # 12 m/s from behind, ego 5 m/s
    ego = S.make(t, np.full_like(t, 5.0), agents=[hit])
    f, _ = _f1(ego)
    ev = f["details"]["events"]
    assert f["raw_status"] == "PASS" and len(ev) == 1 and ev[0]["kind"] == "ACTIVE_REAR" and not ev[0]["at_fault"] and not ev[0]["exempt"], f
    Rc, Dc, runs = _runs_of(ego)
    ct_last, fault_last = B._classify_tick(Rc, Dc, 0, runs[0][1])                  # the agent has passed THROUGH the ego: later ticks alone are a front hit
    assert fault_last and ct_last.name == "ACTIVE_FRONT_COLLISION", (ct_last, fault_last)   # ... so only the exemption keeps f1 at PASS


def test_f1_ego_stopped_is_speed_le_0p05_not_0p005():
    _chk_ego_stopped_threshold()


def test_f1_stopped_track_is_at_fault_and_its_threshold_is_0p05():
    _chk_stopped_track_is_at_fault()


def test_f1_lateral_at_fault_only_beyond_the_lane_proxy_cross_track_0p5():
    _chk_lateral_needs_the_lane_proxy()


def _relative_agent(d_pts, absent=None, aid=70):
    """agent on the ego's path at offset d(t) from the ego box centre (ego 10 m/s straight), agent speed 10 + d'(t)"""
    t = S.tgrid(12.0)
    d = S.kf(d_pts, t)
    present = np.ones_like(t, bool) if absent is None else ~((t >= absent[0]) & (t < absent[1]))
    ag = S.agent(aid, 10.0 * t + d, np.zeros_like(t), 10.0 + np.gradient(d, t), present=present)
    return S.make(t, np.full_like(t, 10.0), agents=[ag])


def test_f1_a_track_whose_first_collision_was_not_at_fault_is_exempt_afterwards():
    two = _relative_agent([(0, -12), (3, -3), (6, -12), (8, 12), (10, 3), (12, 3)], absent=(6.5, 8.0))   # rear contact, leaves, re-enters AHEAD and closes
    f, _ = _f1(two)
    ev = f["details"]["events"]
    assert f["raw_status"] == "PASS" and [e["kind"] for e in ev] == ["ACTIVE_REAR", "EXEMPT"] and ev[1]["exempt"] and f["details"]["n_overlap_events_total"] == 2, f
    Rc, Dc, runs = _runs_of(two)
    assert len(runs) == 2
    ct2, fault2 = B._classify_tick(Rc, Dc, 0, runs[1][0])                          # the second run on its own is a front hit ...
    assert fault2 and ct2.name == "ACTIVE_FRONT_COLLISION"
    only2 = _relative_agent([(0, 12), (8, 12), (10, 3), (12, 3)])                  # ... and WITHOUT the earlier rear contact it fails f1
    f2, _ = _f1(only2)
    assert f2["raw_status"] == "FAIL" and f2["details"]["events"][0]["kind"] == "ACTIVE_FRONT", f2


def test_f1_a_track_already_overlapping_at_the_first_tick_is_exempt():                # pdm_observation.py:209-220 -> collided_track_ids
    t = S.tgrid(12.0)
    ego = S.make(t, np.full_like(t, 10.0), agents=[S.agent(71, np.full_like(t, 0.0), np.zeros_like(t), np.zeros_like(t))])
    f, _ = _f1(ego)
    Rc, Dc, runs = _runs_of(ego)
    assert runs[0][0] == 0 and B._classify_tick(Rc, Dc, 0, 0)[1] is True              # in isolation: a stopped-track hit ...
    assert f["raw_status"] == "PASS" and f["details"]["events"][0]["kind"] == "EXEMPT", f  # ... but it was there before the rollout started


def test_f1_static_object_hit_is_at_fault_with_navsim_score_half():
    t = S.tgrid(12.0)
    ego = S.make(t, np.full_like(t, 10.0), agents=[S.agent(72, np.full_like(t, 80.0), np.zeros_like(t), np.zeros_like(t), cls=3)])
    f, _ = _f1(ego)
    e = f["details"]["events"][0]
    assert f["raw_status"] == "FAIL" and e["at_fault"] and e["navsim_no_at_fault_score"] == 0.5, f


def test_f1_behind_tolerance_is_150_deg_and_the_scorer_uses_the_rear_axle_pose():
    pc, sh = B._collision_mod()
    ego = sh.StateSE2(0.0, 0.0, 0.0)
    for deg, want in ((149.0, False), (151.0, True)):
        r = np.radians(deg)
        assert pc.is_agent_behind(ego, sh.StateSE2(5 * np.cos(r), 5 * np.sin(r), 0.0)) is want, deg
    assert abs(B.build_run(_front_scn(10.0, 5.0, 60.0))["RA"] - 1.461) < 1e-9       # nuPlan Pacifica rear axle -> box centre, unless the record says otherwise


def test_f1_vendored_source_constants_are_navsim_values():
    import inspect
    pc = B._collision_mod()[0]
    assert inspect.signature(pc.is_agent_behind).parameters["angle_tolerance"].default == 150
    assert inspect.signature(pc.is_track_stopped).parameters["stopped_speed_threshhold"].default == 5e-02
    assert inspect.signature(pc.get_collision_type).parameters["stopped_speed_threshold"].default == 5e-02
    assert "FRONT_DEG" not in B.P and "FRONT_DEG" not in inspect.getsource(B._classify_tick)


def test_f1_vendored_collision_files_are_pinned_and_tamper_is_refused():
    for rel in ("navsim_pdm/pdm_collision.py", "navsim_pdm/nuplan_shim.py"):
        assert rel in B._VENDOR_SHA
        tmp = Path(tempfile.mkdtemp(prefix="bv_"))
        try:
            shutil.copytree(B._VENDOR_DIR / "navsim_pdm", tmp / "navsim_pdm")
            p = tmp / rel
            p.write_text(p.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
            old = (B._VENDOR_DIR, B._NAVSIM)
            B._VENDOR_DIR, B._NAVSIM = tmp, None
            try:
                B._collision_mod()
                raise AssertionError(f"tampered {rel} was accepted")
            except RuntimeError as e:
                assert "pinned" in str(e) and rel in str(e)
            finally:
                B._VENDOR_DIR, B._NAVSIM = old
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _body(path, name):
    import ast
    txt = Path(path).read_bytes().decode("utf-8").replace("\r\n", "\n")
    lines = txt.split("\n")
    for n in ast.parse(txt).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return "\n".join(lines[min([n.lineno] + [d.lineno for d in n.decorator_list]) - 1:n.end_lineno])
    raise KeyError(name)


def test_f1_vendored_bodies_equal_sources():
    if not (_NAV_SRC_UTILS.exists() and _NUP_SRC_UTILS.exists()):
        print("    SKIPPED (sources not on this machine; the sha pin still protects the vendored file)")
        return
    v = B._VENDOR_DIR / "navsim_pdm" / "pdm_collision.py"
    for src, names in ((_NUP_SRC_UTILS, ("get_agent_relative_angle", "is_agent_behind", "is_track_stopped")), (_NAV_SRC_UTILS, ("get_collision_type",))):
        for n in names:
            assert _body(v, n) == _body(src, n), n                                   # byte-identical bodies (CRLF -> LF only)


def test_f1_adapter_matches_the_real_nuplan_and_navsim_source_on_fuzzed_geometry():
    if not _NAV_SRC_UTILS.exists():
        print("    SKIPPED (NAVSIM source not on this machine)")
        return
    import importlib.util
    import types
    try:
        from nuplan.common.actor_state.agent import Agent
        from nuplan.common.actor_state.oriented_box import OrientedBox
        from nuplan.common.actor_state.scene_object import SceneObjectMetadata
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
        from nuplan.common.actor_state.static_object import StaticObject
        from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    except Exception as e:                                                          # noqa: BLE001
        print("    SKIPPED (nuplan not importable:", type(e).__name__, ")")
        return
    from shapely import Polygon
    pc, sh = B._collision_mod()
    from navsim_pdm.pdm_enums import StateIndex
    names = ["navsim", "navsim.planning", "navsim.planning.simulation", "navsim.planning.simulation.planner", "navsim.planning.simulation.planner.pdm_planner",
             "navsim.planning.simulation.planner.pdm_planner.utils", "navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums"]
    saved = {k: sys.modules.get(k) for k in names}
    try:
        for k in names:
            sys.modules[k] = types.ModuleType(k)
        sys.modules[names[-1]].StateIndex = StateIndex
        spec = importlib.util.spec_from_file_location("navsim_real_pdm_scorer_utils", _NAV_SRC_UTILS)
        real = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(real)                                               # the REAL source, real nuplan types
    finally:
        for k, vv in saved.items():
            if vv is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = vv
    rng = np.random.default_rng(1)
    seen = set()

    def poly(x, y, h, l, w):
        c, s = np.cos(h), np.sin(h)
        return Polygon([(x + c * a * l / 2 - s * b * w / 2, y + s * a * l / 2 + c * b * w / 2) for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))])

    for _ in range(1500):
        ex, ey, eh = rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-np.pi, np.pi)
        ax, ay, ah = ex + rng.uniform(-5, 5), ey + rng.uniform(-5, 5), rng.uniform(-np.pi, np.pi)
        ev, av = rng.choice([0.0, 0.04, 0.06, 5.0]), rng.choice([0.0, 0.04, 0.06, 3.0])
        stc = np.zeros(11)
        stc[0], stc[1], stc[2], stc[3] = ex, ey, eh, ev
        box = OrientedBox(StateSE2(ax, ay, ah), 4.5, 2.0, 1.6)
        md = SceneObjectMetadata(0, "t", "1", "1")
        if rng.random() < 0.15:
            ro, vo = StaticObject(TrackedObjectType.GENERIC_OBJECT, box, md), sh.TrackedObject(ah)
        else:
            ro, vo = Agent(TrackedObjectType.VEHICLE, box, StateVector2D(av, 0.0), md), sh.Agent(ah, av)
        ep, ap = poly(ex, ey, eh, 5.176, 2.297), poly(ax, ay, ah, 4.5, 2.0)
        r, v = real.get_collision_type(stc, ep, ro, ap), pc.get_collision_type(stc, ep, vo, ap)
        assert int(r) == int(v), (r, v, stc, ax, ay, ah, av)
        seen.add(int(r))
    assert seen == {0, 1, 2, 3, 4}, seen                                            # all five collision types were exercised


def test_f1_old_bearing_rule_turns_the_a10_scenarios_red():                          # deliberate-regression arm: the superseded +-45 deg / 0.005 rule
    import superseded_a10_bearing_rule as OLD
    pc = B._collision_mod()[0]
    _chk_front_hit_on_a_moving_agent(); _chk_ego_stopped_threshold(); _chk_stopped_track_is_at_fault(); _chk_lateral_needs_the_lane_proxy()   # control: green on source
    keep = pc.get_collision_type
    red = {}
    try:
        pc.get_collision_type = OLD.old_collision_type
        for chk in (_chk_ego_stopped_threshold, _chk_stopped_track_is_at_fault):
            try:
                chk()
                red[chk.__name__] = False
            except AssertionError:
                red[chk.__name__] = True
    finally:
        pc.get_collision_type = keep
    assert red == {"_chk_ego_stopped_threshold": True, "_chk_stopped_track_is_at_fault": True}, red


def test_tier_stamp_and_agent_cls_defaulted_flag():
    sc, _ = score(_front_scn(15.0, 5.0, 60.0))
    assert sc["tier_stamp"] == "T2 (re-perception closed-loop sim; MM ruling 2026-11-07)" and sc["agent_cls_defaulted"] is False
    roll = _front_scn(15.0, 5.0, 60.0)
    roll.pop("agent_cls")
    sc2, _ = score(roll)
    assert sc2["agent_cls_defaulted"] is True and sc2["tier_stamp"] == sc["tier_stamp"]
    sc3, _ = score(cruise(15.0))
    assert sc3["agent_cls_defaulted"] is False                                       # no agents at all: nothing was defaulted


# ------------------------------------------------------------------------------------------------ runner
def _collect():
    return {k: v for k, v in globals().items() if k.startswith("test_") and callable(v)}


def main(argv):
    only = None
    if "--only" in argv:
        only = set(argv[argv.index("--only") + 1].split(","))
    tests = _collect()
    bad = []
    print("behaviour_score module:", B.__file__)
    for name, fn in tests.items():
        if only and name not in only:
            continue
        try:
            fn()
            print("PASS", name)
        except Exception:
            bad.append(name)
            print("FAIL", name)
            print("    " + traceback.format_exc().strip().replace("\n", "\n    ")[-1200:])
    n = len([k for k in tests if (not only or k in only)])
    print(f"{n - len(bad)}/{n} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
