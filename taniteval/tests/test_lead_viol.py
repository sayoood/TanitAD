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
