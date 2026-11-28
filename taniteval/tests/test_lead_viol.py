"""LEAD-VIOL (SPEC_REFCV8_DRAFT.md section 8.7, X7) -- hand-computable cases only.

Registered definition under test: LEAD-VIOL = share of lead windows whose plan's minimum time gap
is < 1.0 s while the human's is >= 1.0 s.

Every expected value below is a LITERAL derived by hand from the fixture geometry, never an
expression over the code under test. The fixture (K = 4 steps, dt = 0.5 s, ego v0 = 10 m/s, lead
length 4 m so the rear face is 2 m behind its centre, lead stationary on the x axis):

    gap(step) = lead_x - 2 - plan_x(step)        time_gap = min_step(gap) / 10

    window  lead_x  plan x-track          human x-track        plan_tg  human_tg  lead window?
    w0      200     5,10,15,20            5,10,15,20            17.8     17.8     yes
    w1       60     15,30,45,56           5,10,15,20             0.2      3.8     yes  <- VIOLATION
    w2       60     10,20,30,40           5,10,15,20             1.8      3.8     yes
    w3       60     15,30,45,56           20,35,45,50            0.2      0.8     yes  (human already < 1 s)
    w4       60     5,10,15,20 at y=+5    5,10,15,20             NaN      3.8     yes  (plan left the lane)
    w5      none    5,10,15,20            5,10,15,20             NaN      NaN     no

    n_lead = 5, n_viol = 1  ->  LEAD-VIOL = 0.2
"""
from __future__ import annotations

import numpy as np
import pytest

from taniteval import lead_metrics as lm
from taniteval.lead_metrics import (
    LEAD_VIOL_TG_S, TTC_CAP_S, distance_keeping, lead_viol, lead_viol_vs_human,
    paired_lead_viol,
)

K, DT, V0 = 4, 0.5, 10.0
NAN = float("nan")


def _xs(vals, y=0.0):
    return np.stack([np.asarray(vals, dtype=float), np.full(len(vals), float(y))], axis=1)


def _lead(x):
    return np.stack([np.full(K, float(x)), np.zeros(K)], axis=1)


H_OK = [5, 10, 15, 20]
PLANS = np.stack([_xs(H_OK), _xs([15, 30, 45, 56]), _xs([10, 20, 30, 40]),
                  _xs([15, 30, 45, 56]), _xs(H_OK, y=5.0), _xs(H_OK)])
HUMANS = np.stack([_xs(H_OK), _xs(H_OK), _xs(H_OK), _xs([20, 35, 45, 50]), _xs(H_OK),
                   _xs(H_OK)])
LEADS = np.stack([_lead(200), _lead(60), _lead(60), _lead(60), _lead(60),
                  np.full((K, 2), NAN)])
LENS = np.array([4.0, 4.0, 4.0, 4.0, 4.0, NAN])
SPEEDS = np.full(6, V0)
LEAD_VALID = np.array([True, True, True, True, True, False])
EID = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)


def _score(plans=PLANS, humans=HUMANS, leads=LEADS, **kw):
    return lead_viol_vs_human(plans, humans, leads, LENS, SPEEDS, DT, eid=EID,
                              lead_valid=LEAD_VALID, n_boot=200, **kw)


# --------------------------------------------------------------------------- #
# the fixture itself: pin the hand arithmetic so the cases below are not circular
# --------------------------------------------------------------------------- #
def test_fixture_time_gaps_match_the_hand_table():
    plan = distance_keeping(PLANS, LEADS, LENS, SPEEDS, DT)["time_gap_min_s"]
    human = distance_keeping(HUMANS, LEADS, LENS, SPEEDS, DT)["time_gap_min_s"]
    np.testing.assert_allclose(plan, [17.8, 0.2, 1.8, 0.2, NAN, NAN], atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(human, [17.8, 3.8, 3.8, 0.8, 3.8, NAN], atol=1e-9, equal_nan=True)


def test_the_registered_threshold_is_one_second():
    assert LEAD_VIOL_TG_S == 1.0


# --------------------------------------------------------------------------- #
# the analytic case
# --------------------------------------------------------------------------- #
def _analytic_check(score=_score):
    """The literal assertions. Called green by the test, and called UNDER MUTATION below, where it
    must raise AssertionError -- that is what makes each mutation a deliberate regression."""
    r = score()
    assert r["status"] == "OK"
    assert r["n_lead_windows"] == 5 and r["n"] == 5
    assert r["n_viol"] == 1
    assert r["share"] == pytest.approx(0.2)
    np.testing.assert_array_equal(r["_per_window"]["viol"], [0, 1, 0, 0, 0, NAN])
    assert r["n_at_risk"] == 4                                   # w0 w1 w2 w4: human >= 1 s
    assert r["share_of_at_risk"] == pytest.approx(0.25)
    assert r["n_human_tg_below_thr"] == 1                        # w3
    assert r["n_human_tg_undefined"] == 0
    assert r["n_plan_lost_lead"] == 1                            # w4
    assert r["share_incl_lost_lead"] == pytest.approx(0.4)       # (1 + 1) / 5
    # boundary behaviour on literal time gaps: plan strictly < 1.0, human >= 1.0 inclusive
    flags = lm.lead_viol(
        _tg_dict([1.0, 0.999, 0.5, NAN, 0.5, 0.25]),
        _tg_dict([1.0, 1.0, 0.999, 1.5, NAN, 1.0]),
        lead_valid=[True] * 6)
    np.testing.assert_array_equal(flags["_per_window"]["viol"], [0, 1, 0, 0, 0, 1])
    return r


def _tg_dict(tg):
    tg = np.asarray(tg, dtype=float)
    return {"headway_min_m": np.where(np.isfinite(tg), 10.0, NAN),
            "time_gap_min_s": tg,
            "min_ttc_s": np.where(np.isfinite(tg), 5.0, NAN)}


def test_analytic_case_reads_one_violation_in_five_lead_windows():
    _analytic_check()


def test_the_interval_is_an_episode_cluster_bootstrap_with_its_provenance():
    r = _score()
    assert r["ci"]["estimator"] == "episode_cluster_bootstrap"
    assert r["ci"]["n_episodes"] == 3 and r["ci"]["n_windows"] == 5
    assert r["ci"]["mean"] == pytest.approx(0.2)
    assert "H-ESTIM-SEED-1" in r["ci"]["question_answered"]


def test_without_an_episode_id_the_interval_is_unavailable_with_a_reason_not_dropped():
    plan = distance_keeping(PLANS, LEADS, LENS, SPEEDS, DT)
    human = distance_keeping(HUMANS, LEADS, LENS, SPEEDS, DT)
    r = lead_viol(plan, human, None, lead_valid=LEAD_VALID)
    assert r["share"] == pytest.approx(0.2)
    assert r["ci"]["status"] == "UNAVAILABLE" and r["ci"]["n"] == 5 and r["ci"]["reason"]


def test_no_lead_window_is_not_applicable_with_n_never_a_pass():
    r = lead_viol(_tg_dict([0.1, 0.1]), _tg_dict([2.0, 2.0]), lead_valid=[False, False])
    assert r["status"] == "NOT-APPLICABLE" and r["n"] == 0 and r["reason"]
    assert "share" not in r


def test_denominator_falls_back_to_the_human_corridor_and_says_so():
    plan = distance_keeping(PLANS, LEADS, LENS, SPEEDS, DT)
    human = distance_keeping(HUMANS, LEADS, LENS, SPEEDS, DT)
    r = lead_viol(plan, human, EID)                              # no lead_valid
    assert r["denominator_source"].startswith("FALLBACK")
    assert r["n_lead_windows"] == 5 and r["share"] == pytest.approx(0.2)


def test_shape_contracts():
    with pytest.raises(ValueError):
        lead_viol(_tg_dict([1.0, 2.0]), _tg_dict([1.0]))
    with pytest.raises(ValueError):
        lead_viol(_tg_dict([1.0]), _tg_dict([1.0]), lead_valid=[True, True])
    with pytest.raises(KeyError):
        lead_viol({"headway_min_m": [1.0]}, _tg_dict([1.0]))


# --------------------------------------------------------------------------- #
# CONTROLS (SPEC X7 (e))
# --------------------------------------------------------------------------- #
def test_control_the_human_path_reads_exactly_zero():
    r = _score(plans=HUMANS)
    assert r["n_viol"] == 0 and r["share"] == 0.0


def test_control_human_as_plan_is_zero_for_any_time_gaps_by_construction():
    rng = np.random.default_rng(7)
    tg = rng.uniform(0.0, 3.0, 400)
    tg[::17] = NAN
    tg[::23] = 1.0
    r = lead_viol(_tg_dict(tg), _tg_dict(tg), lead_valid=np.ones(400, bool))
    assert r["n_viol"] == 0 and r["share"] == 0.0


def test_control_stop_plan_never_violates_and_is_ttc_censored_everywhere():
    """A STOP (all-zero) plan keeps the t0 gap: 58 m / 10 m/s = 5.8 s, never closes on a
    stationary lead -> every one of the 5 lead windows is TTC-censored at the cap."""
    stop = np.zeros_like(PLANS)
    r = _score(plans=stop)
    assert r["share"] == 0.0 and r["n_viol"] == 0
    assert r["n_plan_ttc_finite"] == 5 and r["n_plan_ttc_censored"] == 5
    assert r["ttc_cap_s"] == TTC_CAP_S == 30.0
    np.testing.assert_allclose(
        distance_keeping(stop, LEADS, LENS, SPEEDS, DT)["time_gap_min_s"][:5],
        [19.8, 5.8, 5.8, 5.8, 5.8], atol=1e-9)


def test_closing_plans_are_not_ttc_censored():
    r = _score()
    assert r["n_plan_ttc_finite"] == 4 and r["n_plan_ttc_censored"] == 0     # w4 lost its lead


def test_deliberate_regression_a_deranged_lead_changes_lead_viol():
    """SPEC X7 (f). Roll the leads one window: w1 now sees w0's far lead, so the one true
    violation disappears. True value 0.2; deranged value 0.0, and w0 (now lead-less) is a lead
    window with an undefined human time gap."""
    true = _score()
    assert true["share"] == pytest.approx(0.2)
    # NB: lens must be rolled with the leads or the w0/w5 geometry is wrong
    der = lead_viol_vs_human(PLANS, HUMANS, np.roll(LEADS, 1, axis=0), np.roll(LENS, 1),
                             SPEEDS, DT, eid=EID, lead_valid=LEAD_VALID, n_boot=200)
    assert der["share"] == 0.0 and der["n_viol"] == 0
    assert der["n_human_tg_undefined"] == 1
    assert der["share"] != true["share"]


# --------------------------------------------------------------------------- #
# MUTATIONS -- each one reintroduces a plausible definition error and the analytic
# check must go RED. A mutation the check survives would mean the case cannot see it.
# --------------------------------------------------------------------------- #
def test_the_analytic_check_is_green_on_the_real_implementation():
    _analytic_check()


def _mutants():
    def drop_human(p, h, thr=1.0):
        with np.errstate(invalid="ignore"):
            return np.asarray(p, float).reshape(-1) < thr

    def plan_inclusive(p, h, thr=1.0):
        with np.errstate(invalid="ignore"):
            return (np.asarray(p, float) <= thr) & (np.asarray(h, float) >= thr)

    def human_strict(p, h, thr=1.0):
        with np.errstate(invalid="ignore"):
            return (np.asarray(p, float) < thr) & (np.asarray(h, float) > thr)

    def nan_plan_is_violation(p, h, thr=1.0):
        with np.errstate(invalid="ignore"):
            return ~(np.asarray(p, float) >= thr) & (np.asarray(h, float) >= thr)

    def swapped_roles(p, h, thr=1.0):
        with np.errstate(invalid="ignore"):
            return (np.asarray(h, float) < thr) & (np.asarray(p, float) >= thr)

    def threshold_two(p, h, thr=1.0):
        with np.errstate(invalid="ignore"):
            return (np.asarray(p, float) < 2.0) & (np.asarray(h, float) >= 2.0)

    return {"drops the human condition": drop_human,
            "plan boundary inclusive (<=)": plan_inclusive,
            "human boundary strict (>)": human_strict,
            "undefined plan time gap counted as violation": nan_plan_is_violation,
            "plan and human roles swapped": swapped_roles,
            "threshold 2.0 s instead of 1.0 s": threshold_two}


@pytest.mark.parametrize("name", list(_mutants()))
def test_mutation_of_the_flag_rule_goes_red(name, monkeypatch):
    monkeypatch.setattr(lm, "_viol_flags", _mutants()[name])
    with pytest.raises(AssertionError):
        _analytic_check()


def test_mutation_denominator_of_all_windows_goes_red(monkeypatch):
    monkeypatch.setattr(lm, "_lead_window_mask",
                        lambda hh, lv: (np.ones(len(hh), bool), "mutant"))
    with pytest.raises(AssertionError):
        _analytic_check()


# --------------------------------------------------------------------------- #
# the paired estimator -- the X7 bar is refcv8 - refcv7 < 0, separated
# --------------------------------------------------------------------------- #
def _flag_result(viol_idx, n=20):
    """A lead_viol result over n lead windows (10 episodes x 2) violating exactly at viol_idx."""
    plan_tg = np.full(n, 2.0)
    plan_tg[list(viol_idx)] = 0.5
    eid = np.repeat(np.array([f"e{i}" for i in range(n // 2)], dtype=object), 2)
    return lead_viol(_tg_dict(plan_tg), _tg_dict(np.full(n, 2.0)), eid,
                     lead_valid=np.ones(n, bool), n_boot=300), eid


def test_paired_delta_is_a_literal_and_the_sign_convention_is_a_minus_b():
    # arm A violates window 0 of all 10 episodes + window 1 of episodes 0-5 -> 16 of 20
    a, eid = _flag_result([2 * i for i in range(10)] + [2 * i + 1 for i in range(6)])
    # arm B violates window 1 of episodes 0-3 -> 4 of 20
    b, _ = _flag_result([2 * i + 1 for i in range(4)])
    assert a["n_viol"] == 16 and b["n_viol"] == 4
    d = paired_lead_viol(a, b, eid, names=("A", "B"), n_boot=300)
    assert d["delta"] == pytest.approx(0.6)                      # (16 - 4) / 20
    assert d["separated"] is True and d["x7_direction_met"] is False   # A is WORSE
    assert d["estimator"] == "paired_episode_cluster_bootstrap"
    s = paired_lead_viol(b, a, eid, names=("B", "A"), n_boot=300)
    assert s["delta"] == pytest.approx(-0.6)
    assert s["separated"] is True and s["x7_direction_met"] is True    # fewer violations
    same = paired_lead_viol(a, a, eid, n_boot=300)
    assert same["delta"] == 0.0 and same["x7_direction_met"] is False


def test_paired_refuses_different_denominators():
    a, eid = _flag_result([0])
    b = lead_viol(_tg_dict(np.full(20, 2.0)), _tg_dict(np.full(20, 2.0)), eid,
                  lead_valid=np.arange(20) < 19, n_boot=50)
    with pytest.raises(ValueError):
        paired_lead_viol(a, b, eid)


# --------------------------------------------------------------------------- #
# WIRING -- four_families.longitudinal must carry it (and stay interval-complete)
# --------------------------------------------------------------------------- #
def _win_from(plans, humans, **lead_extra):
    n = plans.shape[0]
    return {"pred": plans.copy(), "gt": humans.copy(), "wp_steps": list(range(1, K + 1)),
            "dt_s": DT,
            "lead": {"leads": LEADS[1:5].copy()[:n], "lead_lens": LENS[1:5][:n].copy(),
                     "speeds": SPEEDS[1:5][:n].copy(), "eid": EID[:n].copy(),
                     "n_boot": 100, **lead_extra}}


def test_wiring_all_families_carries_lead_viol_and_stays_ci_complete():
    from taniteval.four_families import all_families
    w = _win_from(PLANS[1:5], HUMANS[1:5],
                  state=np.array(["LEAD", "LEAD", "LEAD", "LEAD"], dtype=object))
    fam = all_families(w)
    lv = fam["longitudinal"]["distance_keeping"]["lead_viol"]
    # w1 violates, w2 does not, w3's human is already < 1 s, w4's plan left the lane
    assert lv["n_lead_windows"] == 4 and lv["n_viol"] == 1
    assert lv["share"] == pytest.approx(0.25)
    assert lv["denominator_source"] == "caller lead_valid"
    assert lv["ci"]["estimator"] == "episode_cluster_bootstrap"
    cov = fam["_ci_coverage"]["longitudinal"]
    assert cov["complete"] is True
    assert not [u for u in cov["undeclared"] if "lead_viol" in u]


def test_wiring_state_other_than_lead_shrinks_the_denominator():
    from taniteval.four_families import all_families
    w = _win_from(PLANS[1:5], HUMANS[1:5],
                  state=np.array(["LEAD", "NO_LABEL", "LEAD", "LEAD"], dtype=object))
    lv = all_families(w)["longitudinal"]["distance_keeping"]["lead_viol"]
    assert lv["n_lead_windows"] == 3 and lv["n_viol"] == 1       # 1/3: NO_LABEL is not a lead window


def test_wiring_explicit_lead_valid_wins_over_state():
    from taniteval.four_families import all_families
    w = _win_from(PLANS[1:5], HUMANS[1:5],
                  state=np.array(["NO_LABEL"] * 4, dtype=object),
                  lead_valid=np.array([True, False, False, False]))
    lv = all_families(w)["longitudinal"]["distance_keeping"]["lead_viol"]
    assert lv["n_lead_windows"] == 1 and lv["share"] == 1.0      # only w1, which violates


def test_wiring_control_plan_equal_to_human_reads_zero_through_all_families():
    from taniteval.four_families import all_families
    w = _win_from(HUMANS[1:5], HUMANS[1:5],
                  state=np.array(["LEAD"] * 4, dtype=object))
    lv = all_families(w)["longitudinal"]["distance_keeping"]["lead_viol"]
    assert lv["n_lead_windows"] == 4 and lv["share"] == 0.0


def test_wiring_without_state_or_lead_valid_uses_the_flagged_fallback():
    from taniteval.four_families import all_families
    w = _win_from(PLANS[1:5], HUMANS[1:5])
    lv = all_families(w)["longitudinal"]["distance_keeping"]["lead_viol"]
    assert lv["denominator_source"].startswith("FALLBACK") and lv["n_lead_windows"] == 4


def test_wiring_time_join_applies_the_same_path_steps_to_the_human_path():
    """A lead block on a coarser grid selects `path_steps` from BOTH the plan and the human path."""
    import torch
    from taniteval.four_families import _distance_keeping
    n = 4
    plan_d = np.full((n, 2 * K, 2), 9999.0)                      # off-grid steps are poison
    hum_d = np.full((n, 2 * K, 2), 9999.0)
    plan_d[:, 1::2] = PLANS[1:5]
    hum_d[:, 1::2] = HUMANS[1:5]
    lead = {"leads": LEADS[1:5].copy(), "lead_lens": LENS[1:5].copy(), "speeds": SPEEDS[1:5].copy(),
            "eid": EID[:n].copy(), "path_steps": [1, 3, 5, 7], "dt_s": DT, "n_boot": 100,
            "lead_valid": np.ones(n, bool)}
    out = _distance_keeping(torch.as_tensor(plan_d).float(), 0.25, lead,
                            torch.as_tensor(hum_d).float())
    assert out["lead_viol"]["n_viol"] == 1 and out["lead_viol"]["share"] == pytest.approx(0.25)


def test_wiring_flags_are_parked_with_the_other_per_window_arrays_not_nested():
    """Arm tools replace `distance_keeping['_per_window']` with a string before JSON; flags that sat
    in `lead_viol` would leak a numpy array into the record."""
    from taniteval.four_families import all_families
    w = _win_from(PLANS[1:5], HUMANS[1:5], state=np.array(["LEAD"] * 4, dtype=object))
    dk = all_families(w)["longitudinal"]["distance_keeping"]
    assert "_per_window" not in dk["lead_viol"]
    np.testing.assert_array_equal(dk["_per_window"]["lead_viol"], [1, 0, 0, 0])


def test_wiring_paired_delta_runs_on_two_all_families_dicts():
    """The X7 bar end to end: arm minus control, both straight out of four_families."""
    from taniteval.four_families import all_families
    st = np.array(["LEAD"] * 4, dtype=object)
    arm = all_families(_win_from(PLANS[1:5], HUMANS[1:5], state=st))["longitudinal"]
    ctl = all_families(_win_from(HUMANS[1:5], HUMANS[1:5], state=st))["longitudinal"]
    d = paired_lead_viol(arm["distance_keeping"], ctl["distance_keeping"], EID[:4],
                         names=("arm", "human"), n_boot=100)
    assert d["n_viol_a"] == 1 and d["n_viol_b"] == 0
    assert d["delta"] == pytest.approx(0.25) and d["x7_direction_met"] is False
    with pytest.raises(KeyError):
        paired_lead_viol({"_per_window": {}}, ctl["distance_keeping"], EID[:4])


# =========================================================================== #
# 2026-10-11 -- the HUMAN-GAP-CONDITIONED denominator (PI), the PROGRESS CONTROL and the VERDICT
# (Master Mind). Every expected value is a hand-derived LITERAL.
#
# Fixture PR (K = 4, dt = 0.5 s, ego v0 = 10 m/s, lead 4 m long so its rear face is 2 m behind its
# centre, lead stationary on the x axis; time gap = min_step(lead_x - 2 - plan_x) / 10; the
# human always drives x = 5, 10, 15, 20 -> arc 20 m):
#
#   window  lead_x  plan x-track     plan arc  plan_tg  human_tg  lead?  at-risk?  viol  ratio
#   w0      40      5,10,15,20          20       1.8      1.8      yes     yes       0    1.0
#   w1      40      4,8,12,16           16       2.2      1.8      yes     yes       0    0.8
#   w2      40      5,10,15,20          20       1.8      1.8      yes     yes       0    1.0
#   w3      40      3,6,9,12            12       2.6      1.8      yes     yes       0    0.6
#   w4      40      8,16,24,32          32       0.6      1.8      yes     yes       1    1.6   <- VIOLATION
#   w5      25      0,0,0,0 (STOP)       0       2.3      0.3      yes     NO        0    --    human already < 1 s
#   w6      none    5,10,15,20          20       NaN      NaN      no      no        --   --
#
#   n_lead = 6, n_at_risk = 5, n_viol = 1  ->  share = 1/6, share_of_at_risk = 0.2
#   progress, on the 5 at-risk windows only:  mean(1.0, 0.8, 1.0, 0.6, 1.6) = 1.0
#   (w5's STOP arc 0 and w6 must NOT enter it: that is what the denominator mutants below catch)
# =========================================================================== #
H4 = [5, 10, 15, 20]
PR_PLANS = np.stack([_xs(H4), _xs([4, 8, 12, 16]), _xs(H4), _xs([3, 6, 9, 12]),
                     _xs([8, 16, 24, 32]), _xs([0, 0, 0, 0]), _xs(H4)])
PR_HUMANS = np.stack([_xs(H4)] * 7)
PR_LEADS = np.stack([_lead(40)] * 5 + [_lead(25), np.full((K, 2), NAN)])
PR_LENS = np.array([4.0] * 6 + [NAN])
PR_SPEEDS = np.full(7, V0)
PR_VALID = np.array([True] * 6 + [False])
PR_EID = np.array(["a", "a", "b", "b", "c", "c", "d"], dtype=object)


def _pr_score(plans=PR_PLANS, humans=PR_HUMANS, **kw):
    return lead_viol_vs_human(plans, humans, PR_LEADS, PR_LENS, PR_SPEEDS, DT, eid=PR_EID,
                              lead_valid=PR_VALID, n_boot=200, **kw)


def _pr_check():
    """The literal assertions (green on the real code; RED under every mutant below)."""
    r = _pr_score()
    assert r["status"] == "OK"
    assert r["n_lead_windows"] == 6 and r["n_at_risk"] == 5 and r["n_viol"] == 1
    assert r["share"] == pytest.approx(1 / 6, abs=1e-6)             # all lead windows: kept, NOT the bar
    assert r["share_of_at_risk"] == pytest.approx(0.2)              # 1 violation / 5 at-risk windows
    ci = r["share_of_at_risk_ci"]
    assert ci["estimator"] == "episode_cluster_bootstrap"
    assert ci["mean"] == pytest.approx(0.2)                          # the point is the FULL-SET value
    assert ci["n_windows"] == 5 and ci["n_episodes"] == 3            # w0 w1 | w2 w3 | w4  (w5 is out)
    assert ci["lo"] <= 0.2 <= ci["hi"] and "H-ESTIM-SEED-1" in ci["question_answered"]
    pw = r["_per_window"]
    np.testing.assert_array_equal(pw["viol"], [0, 0, 0, 0, 1, 0, NAN])
    np.testing.assert_array_equal(pw["at_risk"], [1, 1, 1, 1, 1, 0, NAN])
    np.testing.assert_allclose(pw["progress_ratio"], [1.0, 0.8, 1.0, 0.6, 1.6, NAN, NAN],
                               atol=1e-12, equal_nan=True)
    p = r["progress"]
    assert p["status"] == "OK" and p["n"] == 5 and p["n_at_risk"] == 5
    assert r["lead_progress_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert p["lead_progress_ratio_clipped"] == pytest.approx(0.88)   # (1 + .8 + 1 + .6 + 1) / 5
    assert p["progress_ratio_of_means"] == pytest.approx(1.0)        # (20+16+20+12+32) / (5 * 20)
    assert p["n_below_min"] == 2 and p["meets_min_progress_mean"] is True
    assert p["overshoot_masks_crawl"] is True                        # mean 1.0 clears 0.9, clipped .88 does not
    # MM ruling 2026-10-11: the DECISION reads the clipped mean, so the overshoot cannot mask the crawl
    assert p["meets_min_progress"] is False and p["decision_reading"] == "lead_progress_ratio_clipped"
    assert r["lead_progress_ratio_clipped"] == pytest.approx(0.88)
    assert lm.lead_viol_verdict(r)["verdict"] == "INADMISSIBLE-CRAWL"
    assert p["ci"]["n_windows"] == 5 and p["ci"]["n_episodes"] == 3
    # the PAIRED read against a STOP baseline that never violates: same windows, both denominators
    base = _pr_score(plans=np.zeros_like(PR_PLANS))
    d = paired_lead_viol(r, base, PR_EID, n_boot=200)
    assert d["delta"] == pytest.approx(1 / 6, abs=1e-4)              # old all-windows reading, unchanged
    assert d["at_risk"]["n_at_risk"] == 5 and d["at_risk"]["delta"] == pytest.approx(0.2)
    assert d["at_risk"]["n_viol_a"] == 1 and d["at_risk"]["n_viol_b"] == 0
    assert d["progress"]["n_used"] == 5
    assert d["progress"]["lead_progress_ratio_a"] == pytest.approx(1.0)
    assert d["progress"]["lead_progress_ratio_b"] == 0.0
    return r


def test_pr_fixture_arcs_and_time_gaps_match_the_hand_table():
    np.testing.assert_allclose(lm.path_arc_length(PR_PLANS), [20, 16, 20, 12, 32, 0, 20], atol=1e-12)
    np.testing.assert_allclose(lm.path_arc_length(PR_HUMANS), [20.0] * 7, atol=1e-12)
    plan = distance_keeping(PR_PLANS, PR_LEADS, PR_LENS, PR_SPEEDS, DT)["time_gap_min_s"]
    human = distance_keeping(PR_HUMANS, PR_LEADS, PR_LENS, PR_SPEEDS, DT)["time_gap_min_s"]
    np.testing.assert_allclose(plan, [1.8, 2.2, 1.8, 2.6, 0.6, 2.3, NAN], atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(human, [1.8, 1.8, 1.8, 1.8, 1.8, 0.3, NAN], atol=1e-9, equal_nan=True)


def test_arc_length_is_the_polyline_from_the_origin_and_nan_propagates():
    # 3-4-5: origin -> (3, 4) is 5 m, (3, 4) -> (3, 10) is 6 m
    np.testing.assert_allclose(lm.path_arc_length(np.array([[[3.0, 4.0], [3.0, 10.0]]])), [11.0])
    nan_path = np.array([[[1.0, 0.0], [NAN, 0.0]]])
    assert np.isnan(lm.path_arc_length(nan_path)).all()
    with pytest.raises(ValueError):
        lm.path_arc_length(np.zeros((3, 2)))


def test_pr_analytic_five_at_risk_windows_one_violation_reads_a_fifth():
    _pr_check()


def test_the_bar_denominator_is_the_at_risk_windows_not_all_lead_windows():
    r = _pr_score()
    assert r["share"] != r["share_of_at_risk"]                       # 1/6 vs 1/5: the two readings differ
    assert "share_of_at_risk" in r["bar_denominator"]


def test_without_an_episode_id_the_at_risk_interval_is_unavailable_with_a_reason():
    plan = distance_keeping(PR_PLANS, PR_LEADS, PR_LENS, PR_SPEEDS, DT)
    human = distance_keeping(PR_HUMANS, PR_LEADS, PR_LENS, PR_SPEEDS, DT)
    r = lead_viol(plan, human, None, lead_valid=PR_VALID)
    assert r["share_of_at_risk"] == pytest.approx(0.2)
    assert r["share_of_at_risk_ci"]["status"] == "UNAVAILABLE" and r["share_of_at_risk_ci"]["n"] == 5
    assert r["share_of_at_risk_ci"]["reason"]


def test_no_at_risk_window_gives_a_not_applicable_interval_not_a_zero():
    # the human is already tight (tg 0.5 s) in every lead window -> nothing can be a violation
    r = lead_viol(_tg_dict([0.1, 0.1]), _tg_dict([0.5, 0.5]), np.array(["a", "b"], dtype=object),
                  lead_valid=[True, True], plan_arc=[1.0, 1.0], human_arc=[1.0, 1.0])
    assert r["n_at_risk"] == 0 and r["share_of_at_risk"] is None
    assert r["share_of_at_risk_ci"]["status"] == "NOT-APPLICABLE" and r["share_of_at_risk_ci"]["n"] == 0
    assert r["progress"]["status"] == "NOT-APPLICABLE" and r["lead_progress_ratio"] is None
    assert lm.lead_viol_verdict(r)["verdict"] == "INADMISSIBLE-NO-PROGRESS-CONTROL"


# --------------------------------------------------------------------------- #
# the progress control and the verdict
# --------------------------------------------------------------------------- #
def test_control_stop_plan_meets_lead_viol_but_reads_zero_progress_and_is_a_crawl():
    r = _pr_score(plans=np.zeros_like(PR_PLANS))
    assert r["n_viol"] == 0 and r["share_of_at_risk"] == 0.0         # STOP never violates ...
    assert r["lead_progress_ratio"] == 0.0                            # ... and has covered nothing
    v = lm.lead_viol_verdict(r)
    assert v["verdict"] == "INADMISSIBLE-CRAWL" and v["bar_decision"] is None
    assert v["lead_progress_ratio"] == 0.0 and v["min_progress"] == 0.9


def test_control_human_as_plan_violates_nothing_and_reads_exactly_one_progress():
    r = _pr_score(plans=PR_HUMANS)
    assert r["n_viol"] == 0 and r["share_of_at_risk"] == 0.0
    assert r["lead_progress_ratio"] == 1.0                            # 20 / 20 on every window, exact
    assert r["progress"]["lead_progress_ratio_clipped"] == 1.0
    assert r["progress"]["overshoot_masks_crawl"] is False
    assert lm.lead_viol_verdict(r)["verdict"] == "ADMISSIBLE-BAR-PENDING-BASELINE"


def _paired_arms(arm_arc, base_viol_idx=range(0, 16, 2), n=20):
    """Two arms over the SAME 20 at-risk windows (10 episodes x 2, human tg 2.0 s everywhere).
    The ARM never violates; the BASELINE violates window 0 of episodes 0-7 (8 of 20). Both carry
    arcs against a 20 m human arc; the baseline walks the human's arc (ratio 1.0)."""
    eid = np.repeat(np.array([f"e{i}" for i in range(n // 2)], dtype=object), 2)
    human = _tg_dict(np.full(n, 2.0))
    base_tg = np.full(n, 2.0)
    base_tg[list(base_viol_idx)] = 0.5
    arm = lead_viol(_tg_dict(np.full(n, 2.0)), human, eid, lead_valid=np.ones(n, bool),
                    n_boot=300, plan_arc=np.full(n, float(arm_arc)), human_arc=np.full(n, 20.0))
    base = lead_viol(_tg_dict(base_tg), human, eid, lead_valid=np.ones(n, bool),
                     n_boot=300, plan_arc=np.full(n, 20.0), human_arc=np.full(n, 20.0))
    return arm, base, eid


def test_human_as_plan_against_a_violating_baseline_is_a_PASS():
    arm, base, eid = _paired_arms(arm_arc=20.0)                       # ratio exactly 1.0
    d = paired_lead_viol(arm, base, eid, names=("arm", "base"), n_boot=300)
    assert d["at_risk"]["delta"] == pytest.approx(-0.4)               # 0 - 8/20
    assert d["at_risk"]["separated"] is True
    assert d["x7_direction_met_at_risk"] is True and d[lm.LEAD_VIOL_BAR_KEY] is True
    v = lm.lead_viol_verdict(arm, paired=d)
    assert v["verdict"] == "PASS" and v["bar_decision"] is True
    assert v["lead_progress_ratio"] == 1.0


def test_a_crawler_that_wins_the_paired_bar_is_inadmissible_not_a_pass():
    """Same violations as above, but the arm covers half the human's distance. The paired bar
    reads MET - and the verdict must refuse to call it a pass."""
    arm, base, eid = _paired_arms(arm_arc=10.0)                       # ratio 0.5
    d = paired_lead_viol(arm, base, eid, n_boot=300)
    assert d["x7_direction_met_at_risk"] is True                      # the bar alone is fooled
    v = lm.lead_viol_verdict(arm, paired=d)
    assert v["verdict"] == "INADMISSIBLE-CRAWL" and v["bar_decision"] is None
    assert v["lead_progress_ratio"] == 0.5


def test_a_worse_arm_that_makes_progress_is_a_FAIL():
    arm, base, eid = _paired_arms(arm_arc=20.0)
    d = paired_lead_viol(base, arm, eid, names=("base-as-arm", "arm"), n_boot=300)
    assert d["x7_direction_met_at_risk"] is False                     # more violations than the other
    v = lm.lead_viol_verdict(base, paired=d)
    assert v["verdict"] == "FAIL" and v["bar_decision"] is False


def test_progress_boundary_is_inclusive_at_exactly_point_nine():
    def block(ratio):
        return {"metric": "LEAD-VIOL", "status": "OK", "lead_progress_ratio": ratio,
                "lead_progress_ratio_clipped": ratio, "progress": {}}
    assert lm.LEAD_PROGRESS_MIN == 0.9
    assert lm.lead_viol_verdict(block(0.9))["verdict"] == "ADMISSIBLE-BAR-PENDING-BASELINE"
    assert lm.lead_viol_verdict(block(0.8999))["verdict"] == "INADMISSIBLE-CRAWL"
    assert lm.lead_viol_verdict(block(0.0))["verdict"] == "INADMISSIBLE-CRAWL"
    # and end to end on one at-risk window: 9 m against the human's 10 m is exactly 0.9
    one = lead_viol(_tg_dict([2.0]), _tg_dict([2.0]), np.array(["a"], dtype=object),
                    lead_valid=[True], plan_arc=[9.0], human_arc=[10.0], n_boot=50)
    assert one["lead_progress_ratio"] == 0.9 and one["progress"]["meets_min_progress"] is True


def test_a_tie_at_point_nine_survives_float_averaging():
    """Twenty windows that each read 18 / 20 = 0.9 average to 0.8999999999999999 in float64; the
    inclusive bar must still read them as a tie, not as a crawl."""
    n = 20
    eid = np.repeat(np.array([f"e{i}" for i in range(n // 2)], dtype=object), 2)
    r = lead_viol(_tg_dict(np.full(n, 2.0)), _tg_dict(np.full(n, 2.0)), eid,
                  lead_valid=np.ones(n, bool), plan_arc=np.full(n, 18.0),
                  human_arc=np.full(n, 20.0), n_boot=50)
    assert r["lead_progress_ratio"] == pytest.approx(0.9, abs=1e-12)
    assert r["progress"]["meets_min_progress"] is True and r["progress"]["n_below_min"] == 0
    assert lm.lead_viol_verdict(r)["verdict"] == "ADMISSIBLE-BAR-PENDING-BASELINE"


def test_no_arcs_means_the_control_is_unavailable_and_the_bar_is_withheld():
    plan = distance_keeping(PR_PLANS, PR_LEADS, PR_LENS, PR_SPEEDS, DT)
    human = distance_keeping(PR_HUMANS, PR_LEADS, PR_LENS, PR_SPEEDS, DT)
    r = lead_viol(plan, human, PR_EID, lead_valid=PR_VALID)
    assert r["progress"]["status"] == "UNAVAILABLE" and r["progress"]["reason"]
    assert r["lead_progress_ratio"] is None
    v = lm.lead_viol_verdict(r)
    assert v["verdict"] == "INADMISSIBLE-NO-PROGRESS-CONTROL" and v["bar_decision"] is None


def test_a_degenerate_human_arc_is_excluded_and_counted_not_clamped():
    # w0's human barely moves (0.4 m < the shared 0.5 m floor): its ratio would be 12.5 and is dropped
    p = lm.lead_progress([5.0, 5.0, 10.0], [0.4, 10.0, 10.0], [True, True, True])
    assert p["n"] == 2 and p["n_human_arc_degenerate"] == 1 and p["n_arc_nonfinite"] == 0
    assert p["lead_progress_ratio"] == pytest.approx(0.75)           # (0.5 + 1.0) / 2
    assert p["meets_min_progress"] is False
    assert lm.MIN_HUMAN_ARC_M == 0.5
    # a non-finite arc is also excluded and counted
    q = lm.lead_progress([NAN, 10.0], [10.0, 10.0], [True, True])
    assert q["n"] == 1 and q["n_arc_nonfinite"] == 1 and q["lead_progress_ratio"] == 1.0
    # a window that is not at-risk never enters, however its arcs look
    z = lm.lead_progress([0.0, 10.0], [10.0, 10.0], [False, True])
    assert z["n"] == 1 and z["lead_progress_ratio"] == 1.0


def test_lead_progress_shape_contract():
    with pytest.raises(ValueError):
        lm.lead_progress([1.0, 2.0], [1.0], [True, True])
    with pytest.raises(ValueError):
        lm.lead_progress([1.0], [1.0], [True], eid=["a", "b"])


# --------------------------------------------------------------------------- #
# the paired estimator -- both arms on the SAME windows, and what it refuses
# --------------------------------------------------------------------------- #
def test_paired_refuses_arms_scored_against_different_human_paths():
    """The at-risk set is a property of the human path. Arm B's human is out of the corridor on w0
    (gap < 0 -> undefined time gap): the lead windows are unchanged but the at-risk sets differ."""
    a = _pr_score()
    hum_b = PR_HUMANS.copy()
    hum_b[0] = _xs([20, 35, 45, 50])
    b = _pr_score(humans=hum_b)
    assert b["n_lead_windows"] == a["n_lead_windows"] == 6 and b["n_at_risk"] == 4
    with pytest.raises(ValueError, match="at-risk sets differ"):
        paired_lead_viol(a, b, PR_EID, n_boot=50)


def test_paired_on_a_pre_ruling_result_keeps_the_old_keys_and_says_the_new_ones_are_unavailable():
    a = _pr_score()
    b = _pr_score(plans=np.zeros_like(PR_PLANS))
    for r in (a, b):                                                  # what an old dump looks like
        r["_per_window"].pop("at_risk")
        r["_per_window"].pop("progress_ratio")
    d = paired_lead_viol(a, b, PR_EID, n_boot=50)
    assert d["status"] == "OK" and d["delta"] == pytest.approx(1 / 6, abs=1e-4)   # back-compat
    assert d["x7_direction_met"] is False
    assert d["at_risk"]["status"] == "UNAVAILABLE" and d[lm.LEAD_VIOL_BAR_KEY] is None
    assert d["progress"]["status"] == "UNAVAILABLE"
    assert lm.lead_viol_verdict(a, paired=d)["verdict"] == "INADMISSIBLE-CRAWL"   # the PR fixture overshoots w4 (1.6) and crawls w1/w3: clipped 0.88 < 0.90 -> CRAWL (MM 2026-10-11)
    # the paired-path plumbing, read past the progress gate (min_progress 0.85 < the fixture's 0.88)
    assert lm.lead_viol_verdict(a, paired=d, min_progress=0.85)["verdict"] == "UNAVAILABLE-PAIRED"


def test_the_bar_key_the_spec_should_cite():
    assert lm.LEAD_VIOL_BAR_KEY == "x7_direction_met_at_risk"
    d = paired_lead_viol(_pr_score(), _pr_score(plans=np.zeros_like(PR_PLANS)), PR_EID, n_boot=50)
    assert d[lm.LEAD_VIOL_BAR_KEY] is False                           # arm violates more than STOP


# --------------------------------------------------------------------------- #
# DELIBERATE REGRESSIONS -- each re-introduces a plausible error; the literals must go RED
# --------------------------------------------------------------------------- #
def test_the_pr_literals_are_green_on_the_real_implementation():
    _pr_check()


def test_mutation_denominator_swapped_to_all_lead_windows_goes_red(monkeypatch):
    monkeypatch.setattr(lm, "_at_risk_mask", lambda lead_win, htg, thr: np.asarray(lead_win, bool))
    with pytest.raises(AssertionError):
        _pr_check()


def test_mutation_denominator_swapped_to_all_windows_goes_red(monkeypatch):
    monkeypatch.setattr(lm, "_at_risk_mask",
                        lambda lead_win, htg, thr: np.ones(len(np.asarray(lead_win)), bool))
    with pytest.raises(AssertionError):
        _pr_check()


def test_mutation_progress_over_all_windows_goes_red(monkeypatch):
    real = lm.lead_progress
    monkeypatch.setattr(lm, "lead_progress",
                        lambda pa, ha, ar, *a, **k: real(pa, ha, np.ones(len(ar), bool), *a, **k))
    with pytest.raises(AssertionError):
        _pr_check()


def _verdict_check(verdict_fn):
    """Crawl handling, literal. Called green on the real verdict and under each mutant."""
    arm, base, eid = _paired_arms(arm_arc=10.0)                       # ratio 0.5, wins the paired bar
    d = paired_lead_viol(arm, base, eid, n_boot=300)
    assert d[lm.LEAD_VIOL_BAR_KEY] is True
    v = verdict_fn(arm, paired=d)
    assert v["verdict"] == "INADMISSIBLE-CRAWL" and v["bar_decision"] is None
    mid, _b, _e = _paired_arms(arm_arc=14.0)                          # ratio 0.7: still a crawl
    assert verdict_fn(mid, paired=d)["verdict"] == "INADMISSIBLE-CRAWL"
    ok, _b, _e = _paired_arms(arm_arc=18.0)                           # ratio 0.9 exactly: admissible
    assert verdict_fn(ok, paired=d)["verdict"] == "PASS"
    stop = _pr_score(plans=np.zeros_like(PR_PLANS))
    assert verdict_fn(stop)["verdict"] == "INADMISSIBLE-CRAWL"


def test_the_verdict_literals_are_green_on_the_real_helper():
    _verdict_check(lm.lead_viol_verdict)


@pytest.mark.parametrize("name,thr", [("the progress control ignored", 0.0),
                                      ("a lax 0.50 bar instead of 0.90", 0.5),
                                      ("a 0.95 bar instead of 0.90", 0.95)])
def test_mutation_of_the_progress_bar_goes_red(name, thr):
    real = lm.lead_viol_verdict
    with pytest.raises(AssertionError):
        _verdict_check(lambda lv, **k: real(lv, **{**k, "min_progress": thr}))


# --------------------------------------------------------------------------- #
# WIRING -- four_families.longitudinal carries the new keys, parks the arrays, stays JSON-able
# --------------------------------------------------------------------------- #
def _pr_win(plans, humans, n=6, **lead_extra):
    return {"pred": plans[:n].copy(), "gt": humans[:n].copy(), "wp_steps": list(range(1, K + 1)),
            "dt_s": DT,
            "lead": {"leads": PR_LEADS[:n].copy(), "lead_lens": PR_LENS[:n].copy(),
                     "speeds": PR_SPEEDS[:n].copy(), "eid": PR_EID[:n].copy(), "n_boot": 100,
                     "state": np.array(["LEAD"] * n, dtype=object), **lead_extra}}


def test_wiring_all_families_carries_the_at_risk_ci_progress_and_verdict():
    import json
    from taniteval.four_families import all_families
    fam = all_families(_pr_win(PR_PLANS, PR_HUMANS))
    dk = fam["longitudinal"]["distance_keeping"]
    lv = dk["lead_viol"]
    assert lv["n_lead_windows"] == 6 and lv["n_at_risk"] == 5 and lv["n_viol"] == 1
    assert lv["share"] == pytest.approx(1 / 6, abs=1e-6)
    assert lv["share_of_at_risk"] == pytest.approx(0.2)
    assert lv["share_of_at_risk_ci"]["estimator"] == "episode_cluster_bootstrap"
    assert lv["lead_progress_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert lv["progress"]["n"] == 5 and lv["progress"]["ci"]["n_episodes"] == 3
    assert lv["verdict"]["verdict"] == "INADMISSIBLE-CRAWL"   # the PR fixture overshoots w4 (1.6) and crawls w1/w3: clipped 0.88 < 0.90 -> CRAWL (MM 2026-10-11)
    assert lv["lead_progress_ratio_clipped"] == pytest.approx(0.88)
    assert lm.lead_viol_verdict(lv, min_progress=0.85)["verdict"] == "ADMISSIBLE-BAR-PENDING-BASELINE"
    # the arrays are parked with the other per-window arrays, never nested in the JSON block
    assert "_per_window" not in lv
    np.testing.assert_array_equal(dk["_per_window"]["lead_at_risk"], [1, 1, 1, 1, 1, 0])
    np.testing.assert_allclose(dk["_per_window"]["lead_progress_ratio"],
                               [1.0, 0.8, 1.0, 0.6, 1.6, NAN], atol=1e-12, equal_nan=True)
    json.dumps(lv)                                                    # plain python types only
    cov = fam["_ci_coverage"]["longitudinal"]
    assert cov["complete"] is True and not [u for u in cov["undeclared"] if "lead_viol" in u]


def test_wiring_stop_arm_is_a_crawl_and_human_arm_is_admissible_through_all_families():
    from taniteval.four_families import all_families
    stop = all_families(_pr_win(np.zeros_like(PR_PLANS), PR_HUMANS))["longitudinal"]
    lv = stop["distance_keeping"]["lead_viol"]
    assert lv["n_viol"] == 0 and lv["lead_progress_ratio"] == 0.0
    assert lv["verdict"]["verdict"] == "INADMISSIBLE-CRAWL"
    hum = all_families(_pr_win(PR_HUMANS, PR_HUMANS))["longitudinal"]["distance_keeping"]["lead_viol"]
    assert hum["n_viol"] == 0 and hum["lead_progress_ratio"] == 1.0
    assert hum["verdict"]["verdict"] == "ADMISSIBLE-BAR-PENDING-BASELINE"


def test_wiring_paired_bar_and_verdict_run_on_two_all_families_dicts():
    """End to end: arm vs STOP baseline through four_families, the way the battery will call it."""
    from taniteval.four_families import all_families
    arm = all_families(_pr_win(PR_PLANS, PR_HUMANS))["longitudinal"]["distance_keeping"]
    base = all_families(_pr_win(np.zeros_like(PR_PLANS), PR_HUMANS))["longitudinal"]["distance_keeping"]
    d = paired_lead_viol(arm, base, PR_EID[:6], names=("arm", "stop"), n_boot=100)
    assert d["at_risk"]["n_at_risk"] == 5 and d["at_risk"]["delta"] == pytest.approx(0.2)
    assert d[lm.LEAD_VIOL_BAR_KEY] is False
    assert lm.lead_viol_verdict(arm, paired=d)["verdict"] == "INADMISSIBLE-CRAWL"   # the PR fixture overshoots w4 (1.6) and crawls w1/w3: clipped 0.88 < 0.90 -> CRAWL (MM 2026-10-11)
    v = lm.lead_viol_verdict(arm, paired=d, min_progress=0.85)       # accepts the distance_keeping dict
    assert v["verdict"] == "FAIL" and v["bar_decision"] is False


def test_DELIBERATE_REGRESSION_the_unclipped_mean_as_the_decision_goes_RED():
    """MM ruling 2026-10-11: the decision reads the CLIPPED mean. A verdict fed the unclipped mean
    (1.0 on the overshoot fixture) would admit a planner that crawls in 2 of 5 at-risk windows."""
    blk = {"metric": "LEAD-VIOL", "status": "OK", "lead_progress_ratio": 1.0,
           "lead_progress_ratio_clipped": 0.88, "progress": {}}
    assert lm.lead_viol_verdict(blk)["verdict"] == "INADMISSIBLE-CRAWL"
    mutant = dict(blk, lead_progress_ratio_clipped=blk["lead_progress_ratio"])   # the old reading
    assert lm.lead_viol_verdict(mutant)["verdict"] != "INADMISSIBLE-CRAWL"
