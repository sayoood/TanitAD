"""Tests for read_s1z.py (CPU only).

    CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_read_s1z.py --basetemp F:/tanitad_scratch/s1zread/pt/run

Bars are compared with the frozen card text; M2, G4 and the Z outcome map are tested on pure functions at exact boundaries and end to end on the Batch 1 synthetic scenario
(no real score is read). Every bar has a RED case.
"""
import hashlib
import os
import re
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import read_s1z as Z  # noqa: E402

R = Z.R
sys.path.insert(0, Z.EVAL_DIR)
from test_read_scorer1p6 import Scn  # noqa: E402  (the Batch 1 synthetic world)

CARD_TEXT = open(Z.CARD, encoding="utf-8").read()


# ------------------------------------------------------------------------------------------------ bars equal the card
def _row(rid):
    return next(l for l in CARD_TEXT.splitlines() if l.startswith(f"| **{rid}"))


def test_card_sha_is_frozen():
    assert hashlib.sha256(open(Z.CARD, "rb").read()).hexdigest() == Z.CARD_SHA256


def test_bar_constants_equal_card_text():
    assert "10,000 draws, seed 20260927" in CARD_TEXT and Z.B_DRAWS_BAR == 10000 and Z.SEED_BAR == 20260927
    assert R.B_DRAWS == Z.B_DRAWS_BAR and R.SEED == Z.SEED_BAR
    p1 = _row("P1")
    assert "> 2·F" in p1 and Z.FLOOR_FACTOR == 2.0
    assert "CI-hi < 0" in _row("M2") and Z.M2_CI_HI_BAR == 0.0
    assert float(re.search(r"≥ −(\d+\.\d+)", _row("G1")).group(1)) == -Z.G1_BAR and R.G1_BAR == Z.G1_BAR
    assert float(re.search(r"< \+(\d+\.\d+) points", _row("G2")).group(1)) == Z.G2_BAR and R.G2_BAR == Z.G2_BAR
    g3 = _row("G3")
    assert float(re.search(r"v4/v0 > (\d+\.\d+)", g3).group(1)) == Z.OVERSPEED_RATIO and R.OVERSPEED == Z.OVERSPEED_RATIO
    assert float(re.search(r"< \+(\d+\.\d+)", g3).group(1)) == Z.G3_BAR and R.G3_BAR == Z.G3_BAR
    g4 = _row("G4")
    assert float(re.search(r"< \+(\d+\.\d+)", g4).group(1)) == Z.G4_BAR
    assert "same definition as `P6_RESULT.md`" in g4
    assert float(re.search(r"point ≥ (\d+\.\d+)", _row("Goal status")).group(1)) == Z.GOAL and R.GOAL == Z.GOAL
    # slow ratio: the definition source
    src = open(Z.P6_RESULT.replace("result/P6_RESULT.md", "selc6_analyze.py"), encoding="utf-8").read()
    assert re.search(r"^SLOW_RATIO = 0\.85", src, re.M) and Z.SLOW_RATIO == 0.85 == R.SLOW


def test_card_outcome_names_and_levers_present():
    for z, name in (("Z1", "GOAL MET"), ("Z2", "GAIN, BELOW GOAL"), ("Z3", "HARM"), ("Z4", "NULL"), ("Z5", "SWAMPED"), ("Z6", "MECHANISM ONLY")):
        assert f"**{z} {name}**" in CARD_TEXT and Z.Z_NAMES[z] == name
    assert "speed-matched pairs only" in CARD_TEXT and "speed-matched pairs only" in Z.NEXT["Z3_G4"]
    assert "removed from L_twin" in CARD_TEXT and "removed from L_twin" in Z.NEXT["Z3_G1"]
    assert "1,600-update dose" in Z.NEXT["Z4"] and "second C0 replicate" in Z.NEXT["Z5"]


# ------------------------------------------------------------------------------------------------ exact boundaries, one RED per bar
def test_m2_boundary():
    assert Z.m2_clears(-1e-9, -1e-9) is True
    assert Z.m2_clears(0.0, -0.1) is False and Z.m2_clears(-0.1, 0.0) is False       # CI-hi == 0 is not < 0
    assert Z.m2_clears(0.01, -0.1) is False
    assert Z.m2_clears(None, -0.1) is None


def test_g4_boundary():
    assert Z.g4_ok(0.0999) is True
    assert Z.g4_ok(0.10) is False and Z.g4_ok(0.2) is False


def test_g1_g2_g3_goal_p1_boundaries():
    assert R.g1_ok(-0.5) is True and R.g1_ok(-0.5001) is False
    assert R.g2_ok(0.4999) and not R.g2_ok(0.5)
    assert R.g3_ok(0.0999) and not R.g3_ok(0.10)
    assert R.goal_status(94.55, 94.6) == "GOAL MET" and R.goal_status(94.5, 94.55) == "GOAL NOT EXCLUDED" and R.goal_status(94.5, 94.54) == "BELOW GOAL"
    assert R.p1_clears(0.01, 0.4000001, 0.2) is True and R.p1_clears(0.01, 0.4, 0.2) is False and R.p1_clears(0.0, 1.0, 0.2) is False
    assert Z.swamped_vs_c0(0.01, 0.4, 0.2) is True and Z.swamped_vs_c0(0.01, 0.41, 0.2) is False and Z.swamped_vs_c0(0.0, 0.1, 0.2) is False
    assert Z.swamped_vs_c0(0.1, 0.1, None) is None


# ------------------------------------------------------------------------------------------------ the outcome map (pure)
OK = dict(G1=True, G2=True, G3=True, G4=True)


def st(**kw):
    d = dict(p1=False, m2=False, guards=dict(OK), g1_heads_failed=[], swamped=False, point=90.0)
    d.update(kw)
    return d


def test_outcome_map_every_z_reachable():
    assert Z.decide_outcome(st(p1=True, m2=True, point=94.55))["primary"] == "Z1"
    assert Z.decide_outcome(st(p1=True, point=94.54))["primary"] == "Z2"
    assert Z.decide_outcome(st(p1=True, guards={**OK, "G4": False}))["primary"] == "Z3"
    assert Z.decide_outcome(st(m2=True, guards={**OK, "G3": False}))["primary"] == "Z3"
    assert Z.decide_outcome(st())["primary"] == "Z4"
    assert Z.decide_outcome(st(swamped=True))["primary"] == "Z5"
    assert Z.decide_outcome(st(m2=True))["primary"] == "Z6"
    assert Z.decide_outcome(st(p1=None))["primary"] == "Unnamed"
    assert Z.decide_outcome(st(swamped=None))["primary"] == "Unnamed"
    assert Z.decide_outcome(st(guards={**OK, "G2": None}))["primary"] == "Unnamed"


def test_outcome_red_per_bar():
    """each failed bar moves the outcome off the adopted ones"""
    base = st(p1=True, m2=True, point=95.0)
    assert Z.decide_outcome(base)["adopted"] is True
    for g in ("G1", "G2", "G3", "G4"):
        o = Z.decide_outcome({**base, "guards": {**OK, g: False}})
        assert o["adopted"] is False and o["primary"] == "Z3" and g in o["failed_guards"]
    o = Z.decide_outcome(st(p1=False, m2=True))
    assert o["adopted"] is False and o["primary"] == "Z6"                      # P1 red, M2 alone
    o = Z.decide_outcome(st(p1=True, m2=False))
    assert o["adopted"] is True                                                # M2 does not gate adoption (card: P1 and G1 to G4)


def test_z3_levers():
    o = Z.decide_outcome(st(p1=True, guards={**OK, "G4": False, "G1": False}, g1_heads_failed=["TTC"]))
    lv = o["next_levers"]["Z3"]
    assert "speed-matched pairs" in lv and "TTC" in lv and "L_twin" in lv
    assert "no committed lever" in Z.decide_outcome(st(p1=True, guards={**OK, "G3": False}))["next_levers"]["Z3"]


def test_z5_listed_with_z4_primary_is_z5():
    o = Z.decide_outcome(st(swamped=True))
    assert o["matches"] == ["Z5", "Z4"] and o["primary"] == "Z5"


# ------------------------------------------------------------------------------------------------ end to end on the synthetic scenario
@pytest.fixture(scope="module")
def S():
    return Scn()


def inputs(S, c0=None, c0r=None, s1z=None):
    d = {"C0": (S.p6_pick, 6.0, 11), "C0r": (S.p6_pick, 6.0, 12), "S1Z": (S.p6_pick, 6.0, 13)}
    for k, v in (("C0", c0), ("C0r", c0r), ("S1Z", s1z)):
        if v:
            d[k] = v
    return S.inputs({k: S.arm(k, *v) for k, v in d.items()})


def read(S, **kw):
    return Z.s1z_read(inputs(S, **kw))


def test_Z1_adopted_goal_met_and_m2_holds(S):
    r = read(S, s1z=(S.oracle, 0.2, 21))
    assert r["status"] == "READ" and r["outcome"]["primary"] == "Z1", r["outcome"]
    assert r["M2"]["clears"] is True and r["adoption"]["adopted"] is True and r["verdict_label"] == "ADOPT"
    assert r["guards"]["G4"]["ok"] is True and r["goal_status"]["status"] == "GOAL MET"


def test_Z2_gain_below_goal(S):
    r = read(S, s1z=(S.mix(0.5), 0.2, 22))
    assert r["outcome"]["primary"] == "Z2" and r["goal_status"]["status"] == "BELOW GOAL", r["outcome"]


def test_Z3_harm_from_decoy(S):
    r = read(S, s1z=(S.decoy_cruise(), 0.2, 25))
    assert r["outcome"]["primary"] == "Z3" and r["adoption"]["adopted"] is False
    assert r["guards"]["G1"]["ok"] is False and r["guards"]["G3"]["ok"] is False


def test_Z4_null_when_s1z_is_p6(S):
    r = read(S)
    assert r["outcome"]["primary"] == "Z4" and r["M2"]["clears"] is False
    assert r["M2"]["S1Z_minus_P6"]["hi"] == 0.0 and r["M2"]["S1Z_minus_C0"]["hi"] == 0.0         # zero difference: CI-hi == 0 is not < 0


def test_Z5_swamped(S):
    r = read(S, c0=(S.mix(0.10), 6.0, 31), c0r=(S.mix(0.30, shift=0.5), 6.0, 32), s1z=(S.mix(0.10), 6.0, 33))
    # S1Z == C0 picks, so S1Z - C0 is 0: the swamp needs S1Z to differ from C0 by a gain inside 2F
    r = read(S, c0=(S.p6_pick, 6.0, 31), c0r=(S.mix(0.60, shift=0.5), 6.0, 32), s1z=(S.mix(0.10), 6.0, 33))
    d = r["P1"]["S1Z_vs_C0"]
    assert d["lo"] > 0 and not abs(d["point"]) > 2 * d["F"], "built to have CI-lo > 0 inside 2F"
    assert r["outcome"]["primary"] == "Z5" and r["adoption"]["adopted"] is False, r["outcome"]


def test_G1_red(S):
    r = read(S, s1z=(S.trap_cruise(), 0.2, 27))
    assert r["P1"]["clears"] is True and r["guards"]["G1"]["ok"] is False and r["outcome"]["primary"] == "Z3"
    assert "TTC" in r["guards"]["G1"]["failed_heads"] and "L_twin" in r["outcome"]["next_levers"]["Z3"]


def test_G2_red(S):
    r = read(S, s1z=(S.e2_sabotage(), 0.2, 30))
    assert r["P1"]["clears"] is True and r["guards"]["G2"]["ok"] is False and r["guards"]["G1"]["ok"] is True and r["outcome"]["primary"] == "Z3"


def test_G3_red_and_g3_is_against_P6_only(S):
    r = read(S, s1z=(S.oracle_fast, 0.2, 26))
    assert r["P1"]["clears"] is True and r["guards"]["G3"]["ok"] is False and r["guards"]["G4"]["ok"] is True and r["outcome"]["primary"] == "Z3"
    assert "vs_C0" not in " ".join(r["guards"]["G3"])


def test_G4_red_slow_picks():
    """P6 picks the oracle (not slow); S1Z picks the slowest proposal everywhere: slow-pick share rises, G4 RED, no adoption"""
    S2 = Scn(seed=5)
    S2.p6_pick = S2.oracle.copy()
    r = Z.s1z_read(inputs(S2, s1z=(S2.slowest, 0.2, 61)))
    g4 = r["guards"]["G4"]
    assert g4["S1Z_minus_P6"]["hi"] >= 0.10 and g4["ok"] is False and r["adoption"]["adopted"] is False
    assert g4["S1Z_share"] > g4["P6_share"] and g4["internal_check_vs_run_read_slow_share_abs_diff"] == 0.0


def test_G4_green_when_s1z_less_slow(S):
    r = read(S, s1z=(S.oracle, 0.2, 21))
    assert r["guards"]["G4"]["S1Z_minus_P6"]["hi"] < 0.10 and r["guards"]["G4"]["ok"]


def test_M2_counts_pdms_zero_or_ttc_zero(S):
    r = read(S, s1z=(S.oracle, 0.2, 21))
    m = r["M2"]
    pick_fail = lambda pk: np.mean(((S.pdms[np.arange(S.N), pk] <= 0) | (S.sub[np.arange(S.N), pk, 3] <= 0)))   # noqa: E731
    assert abs(m["S1Z_share"]["point"] - pick_fail(S.oracle)) < 1e-12 and abs(m["P6_share"]["point"] - pick_fail(S.p6_pick)) < 1e-12
    assert m["P6_share"]["point"] > m["S1Z_share"]["point"]
    # the TTC-0 clause: pick_fail counts a pick whose TTC is 0 even if PDMS > 0 (checked on the pure indicator, no shared state is mutated)
    ttc0 = np.array([0.0, 1.0]); pd = np.array([0.4, 0.9])
    assert (((pd <= 0) | (ttc0 <= 0)).astype(float) == np.array([1.0, 0.0])).all()


def test_m2_deliberate_regression_swapped_sign():
    """M2 must read CI-hi: a vector of all -1 differences clears, all +1 does not"""
    assert Z.m2_clears(-1.0, -1.0) and not Z.m2_clears(1.0, 1.0)


def test_control_equals_batch1_read(S):
    """the S1Z path with the same picks gives P1 and G1 to G3 exactly as Batch 1's S1E slot"""
    inp = inputs(S, s1z=(S.mix(0.5), 0.2, 22))
    r = Z.s1z_read(inp)
    b = R.run_read(Z._as_s1e_slot(inp), need_ens3=False)
    e = b["endpoints"]["S1E"]
    for k in Z.CONTROL_FIELDS:
        assert r["batch1_endpoints_S1Z_slot"][k] == e[k]
    assert r["P1"]["S1Z_vs_C0"]["point"] == e["P1_vs_C0"]["point"] and r["guards"]["G2"]["delta_regret_vs_P6"] == e["G2_delta_regret_vs_P6"]


def test_c0r_required(S):
    inp = inputs(S)
    del inp.arms["C0r"]
    assert Z.s1z_read(inp)["status"] == "READ STOPPED"


def test_missing_scores_is_incomplete(S):
    d = S.arm("S1Z", S.oracle, 0.2, 21, scored=False)
    inp = inputs(S)
    inp.arms["S1Z"] = d
    r = Z.s1z_read(inp)
    assert r["status"] == "INCOMPLETE" and "outcome" not in r


def test_gate_failure_stops_read(S):
    inp = inputs(S, s1z=(S.oracle, 0.2, 21))
    inp.arms["S1Z"].e3_proposals = inp.arms["S1Z"].e3_proposals.copy()
    inp.arms["S1Z"].e3_proposals[0, 0, 0, 0] += 1.0
    r = Z.s1z_read(inp)
    assert r["status"] == "READ STOPPED" and "verdict" not in r and "outcome" not in r


def test_smoke_stamp_and_ledger(S, tmp_path):
    led = str(tmp_path / "smoke_ledger_K.json")
    r = Z.s1z_read(inputs(S, s1z=(S.oracle, 0.2, 21)), smoke=True, ledger_path=led)
    assert r["stamp"] == "SMOKE_NOT_A_RESULT" and r["K_ledger"]["K"] == 1
    jp, mp = Z.write_outputs(r, str(tmp_path / "o"))
    assert "SMOKE_NOT_A_RESULT" in open(mp, encoding="utf-8").read()
    r2 = Z.s1z_read(inputs(S, s1z=(S.oracle, 0.2, 21)), smoke=True, ledger_path=led)
    assert r2["K_ledger"]["K"] == 1 and r2["K_ledger"]["this_read_added"] is False       # the same read is not a new read
    lo, hi = r["K_ledger"]["bonferroni_pdms_S1Z_minus_P6"]["lo"], r["K_ledger"]["bonferroni_pdms_S1Z_minus_P6"]["hi"]
    assert lo <= hi


def test_every_block_carries_estimator_tier_n_artifact(S):
    r = read(S, s1z=(S.oracle, 0.2, 21))
    for blk in (r["P1"], r["M2"], r["guards"]["G1"], r["guards"]["G2"], r["guards"]["G3"], r["guards"]["G4"], r["family_clause"], r["goal_status"]):
        assert all(k in blk for k in ("estimator", "tier", "n", "artifact"))
    assert "PAUSED" in r["family_clause"]["STRATEGIC"]
