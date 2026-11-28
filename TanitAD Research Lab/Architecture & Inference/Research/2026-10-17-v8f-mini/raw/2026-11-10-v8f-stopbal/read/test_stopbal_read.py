"""Tests of stopbal_read.py on SYNTHETIC per-window arrays (no dumps, no torch, no GPU): one planted case per committed outcome (all seven), an exact-zero control,
a sub-floor planted delta that must NOT be a gain, controls 1-4, and the floor-comparison mutation (tests/run_mutation_floor.py runs this file against a
mutated copy and records the log).  Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_stopbal_read.py
The module under test is read from env STOPBAL_READ_SRC when set (the mutation runner), else stopbal_read.py beside this file."""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("STOPBAL_READ_SRC", os.path.join(HERE, "stopbal_read.py"))
_spec = importlib.util.spec_from_file_location("stopbal_read_under_test", SRC)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
CI = S.load_ci()
NB = 200          # the tests plant exact differences; a small B keeps the file fast (the real read uses 2,000)
P_CELL = S.CELL_P[1]
N_EP, PER_EP = 40, 20
N = N_EP * PER_EP
EID = np.repeat(np.arange(N_EP), PER_EP)
IDX = np.arange(N)
OK = np.ones(N, bool)
SP = (IDX % 10 == 0)            # 80 stop-point windows, spread over every episode
SP_I, OTH_I = np.where(SP)[0], np.where(~SP)[0]
rng = np.random.default_rng(7)
BASE = 2.0 + rng.normal(0, 0.3, N)


def arm(d_sp=0.0, d_all=0.0, hit=0.0, fpr=0.0, shift=0.0, score_pos=0.5):
    """BASE track error moved by d_sp on stop-point windows and d_all on the others (+ shift everywhere); decision fires on a share `hit` of SP and `fpr` of the rest."""
    ade = BASE + shift + np.where(SP, d_sp, d_all)
    dec = np.zeros(N, bool)
    dec[SP_I[:int(round(hit * len(SP_I)))]] = True
    dec[OTH_I[:int(round(fpr * len(OTH_I)))]] = True
    score = np.where(dec, 0.9, 0.1) + (SP * score_pos * 0.01)
    return {"cells": {P_CELL: ade, "TAC-LON": np.full(N, 0.8)}, "goal_stop": dec, "lon_stop": dec.copy(), "goal_score": score, "lon_score": score.copy()}


NULL = arm()
REP = arm(shift=0.01)            # the run-to-run floor of every cell is 0.01


def floors():
    return S.compute_floors(CI, REP, NULL, EID, SP, OK)


def lever(kind="BAL-G", seeds=((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)), fl=None):
    fl = floors() if fl is None else fl
    per = [S.read_seed(CI, kind, NULL, arm(*x), fl, EID, SP, OK, NB) for x in seeds]
    return per, S.lever_outcome(per)


GOOD = (-0.5, 0.0, 0.4, 0.01)        # P gain, no cost, M passes
GAINONLY = (-0.5, 0.0, 0.1, 0.01)    # P gain, M fails (hit short)


def test_floor_is_the_replicate_gap():
    f = floors()
    assert f["cells"][("all", P_CELL)] == pytest.approx(0.01)
    assert f["cells"][("sp", P_CELL)] == pytest.approx(0.01)
    assert f["cells"][("all", "TAC-LON")] == 0.0


@pytest.mark.parametrize("kind", ["BAL-G", "BAL-L"])
def test_outcome_adopt(kind):
    per, o = lever(kind, (GOOD, GOOD))
    assert o["label"] == "BAL-ADOPT", o
    assert all(p["flags"]["P"] and p["flags"]["NR_ok"] and p["flags"]["M"] for p in per)
    assert "recipe" in o["trigger"]


def test_outcome_gain_other_path():
    _, o = lever("BAL-G", (GAINONLY, GAINONLY))
    assert o["label"] == "BAL-GAIN-OTHER-PATH" and "third-seed" in o["trigger"], o
    assert "BAL-RECALL-SHORT" in o["holding"]            # also holds literally; the headline follows the precedence (r4)


def test_outcome_decision_only():
    _, o = lever("BAL-G", ((0.0, 0.0, 0.4, 0.01), (0.0, 0.0, 0.4, 0.01)))
    assert o["label"] == "BAL-DECISION-ONLY" and "no training" in o["trigger"], o


def test_outcome_fp_costs():
    _, o = lever("BAL-G", ((0.0, 0.0, 0.4, 0.10), (0.0, 0.0, 0.4, 0.01)))
    assert o["label"] == "BAL-FP-COSTS" and "2.14" in o["trigger"], o          # fpr > 0.05 at EITHER seed


def test_outcome_recall_short():
    per, o = lever("BAL-L", ((0.0, 0.0, 0.1, 0.01), (0.0, 0.0, 0.4, 0.01)))
    assert o["label"] == "BAL-RECALL-SHORT", o                                   # hit < 0.30 at EITHER seed
    assert o["recall_short_branch"] is not None and "20.81" in o["trigger"]


def test_outcome_regresses_whatever_else_reads():
    per, o = lever("BAL-G", ((-0.5, 0.5, 0.4, 0.01), GOOD))                  # a clear gain and a passing M, but NR is a cost at seed 0
    assert o["label"] == "BAL-REGRESSES" and per[0]["flags"]["NR_cost"] and per[0]["named_costs"], o
    assert "halves W" in o["trigger"]


def test_outcome_seeds_disagree():
    _, o = lever("BAL-G", (GOOD, (0.0, 0.0, 0.4, 0.01)))                       # P at seed 0 only, M passes at both: no predicate holds
    assert o["label"] == "SEEDS-DISAGREE" and o["holding"] == [], o
    assert "third training seed" in o["trigger"]


def test_exact_zero_control():
    per, o = lever("BAL-G", ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)))
    for p in per:
        for k, c in p["cells"].items():
            assert c["delta"] == 0.0 and not c["gain"] and not c["cost"], (k, c)
        assert not p["flags"]["P"] and p["flags"]["NR_ok"]
    assert o["label"] == "BAL-RECALL-SHORT"            # hit 0.0 < 0.30 at both seeds


def test_subfloor_delta_is_not_a_gain():
    sub = (-0.005, 0.0, 0.4, 0.01)                      # |delta| 0.005 < floor 0.01, yet the paired interval separates it
    per, o = lever("BAL-G", (sub, sub))
    c = per[0]["P_cell"]
    assert c["separated"] and c["delta"] < 0 and not c["above_floor"], c
    assert not c["gain"] and not per[0]["flags"]["P"], c      # <- the floor comparison; flips under the mutation (run_mutation_floor.py)
    assert o["label"] == "BAL-DECISION-ONLY"
    above = (-0.5, 0.0, 0.4, 0.01)
    per2, _ = lever("BAL-G", (above, above))
    assert per2[0]["P_cell"]["gain"]


def test_missing_floor_is_never_a_gain_or_cost():
    per, _ = lever("BAL-G", (GOOD, GOOD), fl={"cells": {}, "dec": {"BAL-G": {"hit": 0.0, "fpr": 0.0, "auroc": 0.0}, "BAL-L": {"hit": 0.0, "fpr": 0.0, "auroc": 0.0}}})
    assert not per[0]["P_cell"]["gain"] and not per[0]["flags"]["P"]


def test_accuracy_cell_direction():
    a = np.full(N, 0.7)       # accuracy lower than NULL's 0.8: worse for a higher-is-better cell
    c = S.cell_delta(CI, a, np.full(N, 0.8), EID, np.ones(N, bool), True, 0.0, NB)
    assert c["cost"] and not c["gain"]
    c2 = S.cell_delta(CI, np.full(N, 0.9), np.full(N, 0.8), EID, np.ones(N, bool), True, 0.0, NB)
    assert c2["gain"] and not c2["cost"]


def test_recall_short_branches():
    # (a) recall up (hit 0.2 > NULL 0.0 + floor 0) but short, AUROC the same as NULL's within floor
    per, o = lever("BAL-G", ((0.0, 0.0, 0.2, 0.01), (0.0, 0.0, 0.2, 0.01)))
    assert o["label"] == "BAL-RECALL-SHORT" and o["recall_short_branch"]["recall_up_both"]
    # (b) recall unchanged: hit equals NULL's (0.0)
    per, o = lever("BAL-G", ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)))
    assert o["recall_short_branch"]["recall_unchanged_both"] and o["recall_short_branch"]["branch"].startswith("(b)")


def test_both_levers_line():
    assert S.both_levers_line({"BAL-G": {"label": "BAL-ADOPT"}, "BAL-L": {"label": "BAL-ADOPT"}}).startswith("both adopt")
    assert "recipe lever" in S.both_levers_line({"BAL-G": {"label": "BAL-ADOPT"}, "BAL-L": {"label": "BAL-RECALL-SHORT"}})
    assert S.both_levers_line({"BAL-G": {"label": "BAL-RECALL-SHORT"}, "BAL-L": {"label": "BAL-RECALL-SHORT"}}) is None


# ------------------------------------------------------------------------------------------------------------------------------------------ controls
def good_selfcheck(kind):
    g, l = S.CARD_WEIGHTS[kind]
    return {"declared_vs_built": "PASS", "levers": {"goal_pos_weight_stop": g, "lon_class_weight_stop": l},
            "live": {"terms": {"goal": 0.05, "lon": 0.6}, "dead_params": [], "groups_grad": {"T": 1767.4}}}


@pytest.mark.parametrize("kind", ["BAL-NULL", "BAL-G", "BAL-L"])
def test_control1_2_pass(kind):
    assert S.check_selfcheck(kind, good_selfcheck(kind)) == []


def test_control1_refusals():
    sc = good_selfcheck("BAL-G")
    sc["levers"]["goal_pos_weight_stop"] = 1.0
    assert any("did not test it" in p for p in S.check_selfcheck("BAL-G", sc))
    sc = good_selfcheck("BAL-L")
    sc["levers"]["lon_class_weight_stop"] = 4.5616
    assert any("lon_class_weight_stop reads 4.5616" in p for p in S.check_selfcheck("BAL-L", sc))
    sc = good_selfcheck("BAL-NULL")
    sc["levers"]["goal_pos_weight_stop"] = 4.5616                    # a NULL arm that carried a lever
    assert S.check_selfcheck("BAL-NULL", sc)
    sc = good_selfcheck("BAL-G")
    del sc["levers"]["lon_class_weight_stop"]
    assert any("missing" in p for p in S.check_selfcheck("BAL-G", sc))
    sc = good_selfcheck("BAL-G")
    sc["levers"]["goal_pos_weight_stop"] = "4.5616"
    assert S.check_selfcheck("BAL-G", sc)
    sc = good_selfcheck("BAL-G")
    sc["declared_vs_built"] = "FAIL"
    assert any("G-DVB" in p for p in S.check_selfcheck("BAL-G", sc))
    assert S.check_selfcheck("BAL-G", None)


def test_control2_refusals():
    sc = good_selfcheck("BAL-G")
    sc["live"]["dead_params"] = ["T.h_goal.weight"]
    assert any("dead T head" in p for p in S.check_selfcheck("BAL-G", sc))
    sc = good_selfcheck("BAL-G")
    sc["live"]["terms"]["lon"] = float("nan")
    assert any("live.terms.lon" in p for p in S.check_selfcheck("BAL-G", sc))
    sc = good_selfcheck("BAL-G")
    sc["live"]["groups_grad"]["T"] = 0.0
    assert any("groups_grad.T" in p for p in S.check_selfcheck("BAL-G", sc))


def test_control3_instrument():
    ok = {"top1_vs_dump_ol_max_abs_m": 0.0, "g_vs_dump_max_abs_m": 0.0, "v0_vs_dump_max_abs_mps": 0.0, "ident_vs_top1_max_abs_m": 0.0, "own_all_vs_ident_max_abs_m": 0.0,
          "goal_prob_in_0_1": True, "lon_prob_rows_sum_to_1_max_dev": 2e-7, "n_windows": 2399, "n_has": 2365, "group_counts_has": {"STOP_POINT": 117}, "manifest_ok": None}
    assert S.check_instrument(ok) == []
    for k, v in (("top1_vs_dump_ol_max_abs_m", 1e-3), ("goal_prob_in_0_1", False), ("n_windows", 2398), ("manifest_ok", False), ("group_counts_has", {"STOP_POINT": 116})):
        bad = dict(ok, **{k: v})
        assert S.check_instrument(bad), k
    assert S.planted_auroc_controls()["pass"]


def test_window_sets_stop_point_is_has_and_present():
    sd = {"has": np.array([1, 1, 1, 0], bool), "err_mps": np.array([[1.0], [np.nan], [2.0], [3.0]]), "strat_tlstop": np.array([1, 1, 0, 1])}
    ok, sp = S.window_sets(sd)
    assert ok.tolist() == [True, False, True, False] and sp.tolist() == [True, False, False, False]
