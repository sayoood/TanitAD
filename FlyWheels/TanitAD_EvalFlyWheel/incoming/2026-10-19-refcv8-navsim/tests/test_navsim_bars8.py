"""The refcv8 NavSim bars layer against LITERAL targets (TANITAD VENV, CPU, no GPU, no checkpoint):

    pytest -q -p no:cacheprovider tests/test_navsim_bars8.py -k "not e2e and not fullsplit"     # the fast literal suite
    pytest -q -p no:cacheprovider tests/test_navsim_bars8.py                                     # + the real-artifact controls

Expectations are LITERALS written here (hand arithmetic, registered edges, banked numbers), never an expression over the code under
test -- CLAUDE.md: "a check that shares the defect it checks for is green forever".  ``tests/test_mutation_bars8.py`` re-runs this
file in fresh interpreters with ``NAVSIM_BARS8_MUTATION`` set and proves the named tests go RED.

Tests whose name contains ``e2e`` or ``fullsplit`` read real artifacts (the landed smoke, the banked full-split CSVs) and SKIP when
those disks are absent -- a skipped control is stated, it is not a pass.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "code"))
import navsim_bars8 as B  # noqa: E402

SG, US = "sg-one-north", "us-ma-boston"
SPEEDS = [0.0, 1.999, 2.0, 5.0, 8.33, 10.0, 13.89, 20.0]          # cycled over the tokens: 2 per band -> 16 tokens per band
CMDS = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0)]    # left, forward, right, forward -> 32 straight / 32 turn
ELIGIBLE = {"label": "refcv8 final read", "step": 48852, "weights": "ema"}
CFG = {"n_full": 64, "sub_intervals": False, "strata_pairs": False}


def mk_info(n_logs: int = 16, per: int = 4) -> pd.DataFrame:
    rows = {}
    for i in range(n_logs * per):
        k = i // per
        rows[f"t{i:04d}"] = {"log": f"2021.01.01.00.{k:02d}.00_veh-1_00000_00001",       # one drive per log (16 drives >= 8)
                             "map_name": SG if k < 4 else US, "command": list(CMDS[i % 4]), "v0": SPEEDS[i % 8],
                             "mcp": 3.0 if i % 2 == 0 else 50.0}
    return pd.DataFrame.from_dict(rows, orient="index")


def mk_arm(vals, toks) -> pd.DataFrame:
    """NC carries the target PDMS, everything else is 1 -> PDMS = NC * DAC * (5 + 5 + 2) / 12 = NC exactly."""
    v = np.broadcast_to(np.asarray(vals, float), (len(toks),)).copy()
    return pd.DataFrame({"NC": v, "DAC": 1.0, "EP": 1.0, "TTC": 1.0, "C": 1.0, "DDC": 1.0, "PDMS": v}, index=list(toks))[list(B.METRICS)]


def ev(vals: dict, meta=None, cfg=None, info=None, **kw):
    info = mk_info() if info is None else info
    arms = {k: mk_arm(v, info.index) for k, v in vals.items()}
    return B.evaluate_navtest(arms, info, ELIGIBLE if meta is None else meta, {**CFG, **(cfg or {})}, **kw)


# ---- 1. hand-computed PDMS on three rows --------------------------------------------------------------------------------------------
def _three_rows(tmp_path, score1=0.7916666666666666):
    # r1 NC1 DAC1 EP.5 TTC1 C1 -> (5*1 + 5*.5 + 2*1)/12 = 9.5/12 ; r2 NC0 -> 0 ; r3 NC1 DAC1 EP1 TTC0 C1 -> (0 + 5 + 2)/12 = 7/12
    df = pd.DataFrame({"token": ["a", "b", "c", "average"], "valid": [True] * 4,
                       "no_at_fault_collisions": [1.0, 0.0, 1.0, 2 / 3], "drivable_area_compliance": [1.0, 1.0, 1.0, 1.0],
                       "ego_progress": [0.5, 0.3, 1.0, 0.6], "time_to_collision_within_bound": [1.0, 1.0, 0.0, 2 / 3],
                       "comfort": [1.0, 1.0, 1.0, 1.0], "driving_direction_compliance": [1.0, 1.0, 1.0, 1.0],
                       "score": [score1, 0.0, 0.5833333333333334, 0.4583333333333333]})
    p = tmp_path / "three.csv"
    df.to_csv(p)
    return str(p)


def test_pdms_hand_computed_on_three_rows(tmp_path):
    d = B.read_scores_csv(_three_rows(tmp_path))
    rec = B.arm_record(d, ["a", "b", "c"], ["la", "lb", "lc"], {**B.DEFAULT_CFG, "sub_intervals": False})
    assert (rec["NC"], rec["DAC"], rec["TTC"], rec["EP"], rec["C"], rec["DDC"], rec["PDMS"]) == \
           (66.6667, 100.0, 66.6667, 60.0, 100.0, 100.0, 45.8333)
    assert rec["zeroed_by_NC_or_DAC"] == 1
    assert rec["pdms_recompute_max_abs"] < 1e-12                       # PDMS re-derived from the sub-scores agrees with the CSV
    assert rec["intervals_x100"]["PDMS"]["status"] == "UNAVAILABLE"      # 3 logs < 8: no interval is invented
    assert rec["intervals_x100"]["PDMS"]["n_clusters"] == 3


def test_a_corrupted_score_column_is_caught_by_the_independent_recompute(tmp_path):
    d = B.read_scores_csv(_three_rows(tmp_path, score1=0.9))           # 0.9 instead of 0.7916666...
    rec = B.arm_record(d, ["a", "b", "c"], ["la", "lb", "lc"], {**B.DEFAULT_CFG, "sub_intervals": False})
    assert 0.108 < rec["pdms_recompute_max_abs"] < 0.109


def test_pdm_score_column_is_refused(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"token": ["a"], "valid": [True], "pdm_score": [0.5]}).to_csv(p)
    with pytest.raises(B.BarsError):
        B.read_scores_csv(str(p))


# ---- 2. the constant-arm control and the keyed join ---------------------------------------------------------------------------------
def test_constant_arm_control_reads_exactly_zero_with_zero_width_interval():
    r = ev({"R8_LEGAL": 0.5, "STOP": 0.5})
    p = r["pairs"]["R8_LEGAL__minus__STOP"]
    assert p["delta_x100"] == 0.0 and (p["wins"], p["ties"], p["losses"]) == (0, 64, 0)
    iv = p["interval"]
    assert iv["status"] == "OK" and iv["lo"] == 0.0 and iv["hi"] == 0.0 and iv["positive"] is False and iv["negative"] is False


def test_two_constant_arms_read_an_exact_difference_with_zero_width_interval():
    r = ev({"R8_LEGAL": 0.6, "STOP": 0.35})
    iv = r["pairs"]["R8_LEGAL__minus__STOP"]["interval"]
    assert r["pairs"]["R8_LEGAL__minus__STOP"]["delta_x100"] == 25.0
    assert iv["lo"] == iv["hi"] == 25.0 and iv["positive"] is True


def test_join_is_keyed_by_token_not_by_position():
    info = mk_info()
    base = 0.30 + 0.01 * np.arange(64)                                   # a different value on every token
    a = mk_arm(base, info.index)
    b = mk_arm(base, info.index).sample(frac=1.0, random_state=7)       # SAME numbers per token, rows shuffled
    assert list(b.index) != list(a.index)
    r = B.evaluate_navtest({"R8_LEGAL": a, "STOP": b}, info, ELIGIBLE, {**B.DEFAULT_CFG, **CFG})
    p = r["pairs"]["R8_LEGAL__minus__STOP"]
    assert p["delta_x100"] == 0.0 and (p["wins"], p["ties"], p["losses"]) == (0, 64, 0)
    assert p["interval"]["lo"] == 0.0 and p["interval"]["hi"] == 0.0


def test_an_arm_missing_tokens_is_refused_for_pairing_not_silently_joined():
    info = mk_info()
    full = mk_arm(0.6, info.index)
    short = mk_arm(0.5, info.index[:60])
    r = B.evaluate_navtest({"R8_LEGAL": full, "STOP": short}, info, ELIGIBLE, {**B.DEFAULT_CFG, **CFG})
    assert "STOP" in r["refused_pairing"] and "STOP" not in r["arms"]


# ---- 3. strata edges and levels, each at its boundary ---------------------------------------------------------------------------------
@pytest.mark.parametrize("v,band", [(0.0, "[0,2)"), (1.999, "[0,2)"), (2.0, "[2,8.33)"), (8.3299, "[2,8.33)"), (8.33, "[8.33,13.89)"),
                                    (13.8899, "[8.33,13.89)"), (13.89, ">=13.89"), (40.0, ">=13.89"), (-0.1, "invalid"),
                                    (float("nan"), "invalid")])
def test_speed_band_edges_literal(v, band):
    assert B.speed_band(v) == band


@pytest.mark.parametrize("oh,key,stratum", [((1, 0, 0, 0), "left", "turn"), ((0, 1, 0, 0), "forward", "straight"),
                                            ((0, 0, 1, 0), "right", "turn"), ((0, 0, 0, 1), "unknown", "unknown"),
                                            ((1, 1, 0, 0), "malformed", "malformed"), ((0, 0, 0), "malformed", "malformed")])
def test_command_strata_literal(oh, key, stratum):
    assert (B.command_key(oh), B.command_stratum(oh)) == (key, stratum)


@pytest.mark.parametrize("m,fired", [(0.0, True), (4.9999, True), (5.0, True), (5.0000001, False), (26.7, False)])
def test_ep_clause_is_inclusive_at_5_m(m, fired):
    assert B.ep_clause_fired(m) is fired


def test_city_stratum_literal():
    assert (B.city_stratum("sg-one-north"), B.city_stratum("us-ma-boston"), B.city_stratum("us-pa-pittsburgh-hazelwood")) == \
           ("Singapore", "US", "US")


def test_strata_partition_counts_literal():
    r = ev({"R8_LEGAL": 0.6, "STOP": 0.5})["strata"]
    P = r["partitions"]
    assert r["partition_sums_ok"] is True
    assert {k: v["n"] for k, v in P["city"].items()} == {"Singapore": 16, "US": 48}
    assert {k: v["n"] for k, v in P["command"].items()} == {"straight": 32, "turn": 32, "unknown": 0, "malformed": 0}
    assert {k: v["n"] for k, v in P["t0_speed_band"].items()} == {"[0,2)": 16, "[2,8.33)": 16, "[8.33,13.89)": 16, ">=13.89": 16, "invalid": 0}
    assert {k: v["n"] for k, v in P["ep_clause"].items()} == {"fired": 32, "not_fired": 32}
    rr = ev({"R8_LEGAL": 0.6, "STOP": 0.5})
    assert rr["ep_clause_fired"]["n"] == 32 and rr["ep_clause_fired"]["fraction"] == 0.5


def test_seed_floor_flag_in_strata_literal():
    # LEGAL 0.60 everywhere; s1 0.50 on the 16 Singapore tokens only -> global floor 2.5, Singapore floor 10.0, US floor 0.0
    info = mk_info()
    sg = np.array([m == SG for m in info["map_name"]])
    r = ev({"R8_LEGAL": 0.6, "R8_LEGAL_s1": np.where(sg, 0.5, 0.6), "STOP": np.where(sg, 0.59, 0.5)}, info=info, cfg={"strata_pairs": True})
    assert r["inference_seed_floor"]["x100"] == 2.5
    s = r["strata"]["partitions"]["city"]
    assert s["Singapore"]["seed_floor_x100_in_stratum"] == 10.0 and s["US"]["seed_floor_x100_in_stratum"] == 0.0
    d_sg = s["Singapore"]["pairs"]["R8_LEGAL__minus__STOP"]
    d_us = s["US"]["pairs"]["R8_LEGAL__minus__STOP"]
    assert (d_sg["delta_x100"], d_sg["floor_used_x100"], d_sg["exceeds_seed_floor"]) == (1.0, 10.0, False)     # 1.0 <= 10.0: not an effect
    assert (d_us["delta_x100"], d_us["floor_used_x100"], d_us["exceeds_seed_floor"]) == (10.0, 2.5, True)       # 10.0 > max(0.0, 2.5)
    # an EFFECT needs the floor AND a separated interval: Singapore has 4 logs (< 8, no interval) and is inside its floor; US has 12 logs
    assert d_sg["is_effect"] is False and d_sg["interval"]["status"] == "UNAVAILABLE"
    assert d_us["is_effect"] is True and d_us["interval"]["positive"] is True


# ---- 4. BAR-R8-N1 / N4 on synthetic arms with known answers -----------------------------------------------------------------------------
def test_n1_and_n4_pass_literal():
    r = ev({"R8_LEGAL": 0.60, "R8_LEGAL_s1": 0.60, "STOP": 0.50, B.R7_ARM: 0.55})
    n1, n4 = r["bars"]["BAR-R8-N1"], r["bars"]["BAR-R8-N4"]
    assert n1["verdict"] == "PASS" and n1["arithmetic"]["R8_LEGAL"]["margin_x100"] == 10.0 and n1["seed_floor_x100"] == 0.0
    assert n4["verdict"] == "PASS" and n4["arithmetic"]["R8_LEGAL"]["margin_x100"] == 5.0


def test_n1_inside_2x_floor_is_not_proven():
    # LEGAL .60 / s1 .54 -> floor 6.0 ; margin 10.0 <= 2 x 6.0 = 12.0 -> NOT PROVEN even though the interval is separated
    r = ev({"R8_LEGAL": 0.60, "R8_LEGAL_s1": 0.54, "STOP": 0.50})
    n1 = r["bars"]["BAR-R8-N1"]
    assert n1["seed_floor_x100"] == 6.0 and n1["verdict"].startswith("NOT PROVEN")
    assert r["pairs"]["R8_LEGAL__minus__STOP"]["interval"]["positive"] is True


def test_n4_inside_2x_floor_is_not_proven():
    # A-R8-NS1 AMENDMENT 2.  LEGAL .60 / s1 .59 -> floor 1.0 ; R7 .585 : seed-0 margin 1.5 is ABOVE 1x the floor but inside 2x,
    # so it reads NOT PROVEN (the builder's 1x gate read PASS here -- the n4_floor_1x mutation restores it and must go red)
    r = ev({"R8_LEGAL": 0.60, "R8_LEGAL_s1": 0.59, "STOP": 0.40, B.R7_ARM: 0.585})
    n4 = r["bars"]["BAR-R8-N4"]
    assert n4["seed_floor_x100"] == 1.0 and n4["arithmetic"]["R8_LEGAL"]["margin_x100"] == 1.5
    assert n4["per_arm_verdict"]["R8_LEGAL"] == "NOT PROVEN (margin inside 2x the inference-seed floor)"
    assert n4["verdict"].startswith("NOT PROVEN")
    assert n4["margin_exceeds_1x_floor_BESIDE"] is True


def test_n4_above_2x_floor_at_both_seeds_passes():
    # LEGAL .60 / s1 .59 -> floor 1.0 ; R7 .56 : margins 4.0 (seed 0) and 3.0 (seed 1), both > 2.0 -> PASS
    r = ev({"R8_LEGAL": 0.60, "R8_LEGAL_s1": 0.59, "STOP": 0.40, B.R7_ARM: 0.56})
    n4 = r["bars"]["BAR-R8-N4"]
    assert n4["per_arm_verdict"] == {"R8_LEGAL": "PASS", "R8_LEGAL_s1": "PASS"}
    assert n4["verdict"] == "PASS"


def test_n1_fails_when_the_arm_is_below_stop():
    r = ev({"R8_LEGAL": 0.40, "R8_LEGAL_s1": 0.40, "STOP": 0.50})
    assert r["bars"]["BAR-R8-N1"]["verdict"] == "FAILED"


def test_n1_fails_when_the_margin_is_positive_but_not_separated():
    # per-log deltas alternate +0.30 / -0.20 over 16 logs: mean +0.05, sd 0.25 -> the log-cluster interval straddles 0
    info = mk_info()
    even = np.array([(i // 4) % 2 == 0 for i in range(64)])
    r = ev({"R8_LEGAL": np.where(even, 0.8, 0.3), "R8_LEGAL_s1": np.where(even, 0.8, 0.3), "STOP": 0.5}, info=info)
    p = r["pairs"]["R8_LEGAL__minus__STOP"]
    assert p["delta_x100"] == 5.0 and p["interval"]["positive"] is False
    assert r["bars"]["BAR-R8-N1"]["verdict"] == "FAILED"


def test_n1_needs_the_floor_arm_to_pass():
    r = ev({"R8_LEGAL": 0.60, "STOP": 0.50})
    assert r["bars"]["BAR-R8-N1"]["verdict"].startswith("NOT PROVEN (inference-seed floor not measured")


def test_n1_on_a_subset_is_not_evaluated_not_failed():
    info = mk_info()
    arms = {k: mk_arm(v, info.index) for k, v in {"R8_LEGAL": 0.6, "R8_LEGAL_s1": 0.6, "STOP": 0.5}.items()}
    r = B.evaluate_navtest(arms, info, ELIGIBLE, {**B.DEFAULT_CFG, "sub_intervals": False, "strata_pairs": False})   # default n_full 12,146
    assert r["bars"]["BAR-R8-N1"]["verdict"].startswith("NOT_EVALUATED -- SUBSET")


def test_validation_only_run_carries_no_verdict():
    r = ev({"R8_LEGAL": 0.6, "R8_LEGAL_s1": 0.6, "STOP": 0.5, B.R7_ARM: 0.55}, meta={"label": "VALIDATION ONLY (x)", "step": 40000, "weights": "ema"})
    for k in ("BAR-R8-N1", "BAR-R8-N4"):
        assert r["bars"][k]["verdict"] == "NOT_EVALUATED -- VALIDATION ONLY"
    assert r["bars"]["BAR-R8-N1"]["verdict_if_eligible_ARITHMETIC_ONLY"] == "PASS"           # arithmetic is shown, labelled as such
    assert B.classify({"label": "x", "step": 40000, "weights": "ema"}, B.DEFAULT_CFG) == "VALIDATION"
    assert B.classify({"label": "VALIDATION ONLY", "step": 48852, "weights": "ema"}, B.DEFAULT_CFG) == "VALIDATION"


def test_raw_weights_are_beside_and_carry_no_bar():
    r = ev({"R8_LEGAL": 0.6, "R8_LEGAL_s1": 0.6, "STOP": 0.5}, meta={"label": "refcv8", "step": 48852, "weights": "raw"})
    assert r["bars"]["BAR-R8-N1"]["verdict"].startswith("NOT_EVALUATED -- BESIDE")
    assert B.classify({"label": "refcv8", "step": 48852, "weights": "raw"}, B.DEFAULT_CFG) == "BESIDE"
    assert B.classify(ELIGIBLE, B.DEFAULT_CFG) == "RESULT"


def test_swapped_arm_labels_would_flip_the_n1_verdict_literal():
    # the literal that the mutation arm 'swap_arm_labels' must break: LEGAL .60 beats STOP .50 -> PASS, never FAILED
    r = ev({"R8_LEGAL": 0.60, "R8_LEGAL_s1": 0.60, "STOP": 0.50})
    assert r["bars"]["BAR-R8-N1"]["arithmetic"]["R8_LEGAL"]["PDMS_x100"] == 60.0
    assert r["bars"]["BAR-R8-N1"]["arithmetic"]["R8_LEGAL"]["margin_x100"] == 10.0
    assert r["arms"]["STOP"]["PDMS"] == 50.0


# ---- 5. R8_RT and R8_PRIV ---------------------------------------------------------------------------------------------------------------
GOOD_SHA = "f18359be021dd400f4865cdcb3ca053bb8c0910b54c0611ba09d44da9a97219e"
CLIPFED_SHA = "b208c9c1" + "0" * 56


def test_rt_carries_no_bar_unless_ship_r8_passed():
    r = ev({"R8_LEGAL": 0.6, "R8_LEGAL_s1": 0.6, "STOP": 0.5, B.R7_ARM: 0.55, "R8_RT": 0.7}, meta={**ELIGIBLE, "rt_table_sha256": GOOD_SHA})
    assert r["bars"]["R8_RT"]["verdict"].startswith("NO_BAR")
    assert "BAR-R8-N1[R8_RT]" not in r["bars"]["R8_RT"]
    assert "R8_RT__minus__R8_LEGAL" in r["pairs"]                                              # reported beside


def test_rt_bar_when_ship_passed_and_the_table_is_per_frame():
    r = ev({"R8_LEGAL": 0.6, "R8_LEGAL_s1": 0.6, "STOP": 0.5, B.R7_ARM: 0.55, "R8_RT": 0.7},
           meta={**ELIGIBLE, "rt_table_sha256": GOOD_SHA}, cfg={"ship_r8_passed": True})
    assert r["bars"]["R8_RT"]["BAR-R8-N1[R8_RT]"]["verdict"] == "PASS"
    assert r["bars"]["R8_RT"]["BAR-R8-N1[R8_RT]"]["arithmetic"]["R8_RT"]["margin_x100"] == 20.0


@pytest.mark.parametrize("sha", [CLIPFED_SHA, ""])
def test_rt_bar_refused_for_the_clip_fed_or_unrecorded_table(sha):
    r = ev({"R8_LEGAL": 0.6, "R8_LEGAL_s1": 0.6, "STOP": 0.5, "R8_RT": 0.7}, meta={**ELIGIBLE, "rt_table_sha256": sha}, cfg={"ship_r8_passed": True})
    assert r["bars"]["R8_RT"]["verdict"].startswith("REFUSED")


def test_priv_never_carries_a_bar():
    r = ev({"R8_LEGAL": 0.6, "R8_LEGAL_s1": 0.6, "STOP": 0.5, "R8_PRIV": 0.8}, cfg={"ship_r8_passed": True})
    pv = r["bars"]["R8_PRIV"]
    assert pv["verdict"].startswith("NO_BAR") and not [k for k in pv if k.startswith("BAR-")]
    assert pv["PRIV_minus_LEGAL"]["delta_x100"] == 20.0                                       # reported as PRIV - LEGAL with a paired CI
    assert pv["PRIV_minus_LEGAL"]["interval"]["status"] == "OK"
    assert r["arms"]["R8_PRIV"]["role"].startswith("DIAGNOSTIC ONLY")


# ---- 6. the interval's conjunction (pure, on hand-made estimator records) -------------------------------------------------------------------
def _fake(lo, hi, dlo, dhi, dstatus="OK"):
    return {"status": "OK", "delta": 1.0, "lo": lo, "hi": hi, "n_clusters": 136, "separated_log_name": lo > 0 or hi < 0,
            "sensitivity_nuplan_drive": ({"status": "OK", "n_clusters": 44, "lo": dlo, "hi": dhi} if dstatus == "OK"
                                         else {"status": dstatus, "reason": "x"})}


def test_pair_positive_needs_both_levels():
    assert B.compact_pair(_fake(1.0, 2.0, 0.5, 2.5))["positive"] is True
    assert B.compact_pair(_fake(1.0, 2.0, -0.5, 2.5))["positive"] is False      # log level separated, drive level straddles 0
    assert B.compact_pair(_fake(-1.0, 2.0, 0.5, 2.5))["positive"] is False
    cr = B.compact_pair(_fake(1.0, 2.0, 0, 0, dstatus="CANNOT-RULE"))
    assert cr["positive"] is True and cr["drive_level"]["status"] == "CANNOT-RULE"       # stated, never silently passed
    assert B.compact_pair({"status": "UNAVAILABLE", "reason": "r", "n_clusters": 5})["status"] == "UNAVAILABLE"


# ---- 7. N2 / N3 from summaries (literals) ---------------------------------------------------------------------------------------------------
def _pi(lo, dlo, hi=0.002):
    # the parse7 / parse8 paired record shape (log_name level + the nuplan_drive sensitivity)
    return {"status": "OK", "delta": round((lo + hi) / 2, 6), "lo": lo, "hi": hi, "n_clusters": 76,
            "sensitivity_nuplan_drive": {"status": "OK", "n_clusters": 34, "lo": dlo, "hi": hi}}


def _nh(legal=0.30, lo=0.0005, sf=0.0004, dlo=None, s1=None, s1_lo=None, s1_dlo=None, s1_pair=True):
    # seed 1 follows seed 0 unless a test moves it (A-R8-NS1 AMENDMENT 3: N2 is read at both inference seeds)
    dlo = lo if dlo is None else dlo
    s1 = legal if s1 is None else s1
    s1_lo = lo if s1_lo is None else s1_lo
    s1_dlo = dlo if s1_dlo is None else s1_dlo
    pis = {"R8_LEGAL__minus__STOP_zero": _pi(lo, dlo)}
    if s1_pair:
        pis["R8_LEGAL_s1__minus__STOP_zero"] = _pi(s1_lo, s1_dlo)
    return {"arms": {"R8_LEGAL": {"official_two_stage_EPDMS": legal}, "R8_LEGAL_s1": {"official_two_stage_EPDMS": s1},
                     "CV_official": {"official_two_stage_EPDMS": 0.10},
                     "STOP_zero": {"official_two_stage_EPDMS": 0.2985}, "ECHO_ha0_ext": {"official_two_stage_EPDMS": 0.20}},
            "paired_intervals": pis,
            "pairs": {"R8_LEGAL__minus__R8_LEGAL_s1": {"official_two_stage_EPDMS_delta": sf}}}


@pytest.mark.parametrize("kw,expect", [({}, "PASS"), ({"sf": 0.001}, "NOT PROVEN"), ({"lo": -0.001}, "FAILED"), ({"legal": 0.2269}, "FAILED")])
def test_n2_literal(kw, expect):
    # margin = 0.30 - max(0.10, 0.2985, 0.20) = 0.0015 ; floor 0.0004 -> 0.0015 > 0.0008 PASS ; floor 0.001 -> 0.0015 <= 0.002 NOT PROVEN
    assert B.navhard_bar(_nh(**kw), ELIGIBLE, B.DEFAULT_CFG)["verdict"].startswith(expect)


def test_n2_needs_the_drive_level():
    # A-R8-NS1 AMENDMENT 3: log level lo +0.0005 excludes 0, the nuplan_drive level lo -0.0001 does not -> not separated -> FAILED
    # (the builder's log-only gate read PASS here; the n2_log_only mutation restores it and must go red)
    r = B.navhard_bar(_nh(dlo=-0.0001), ELIGIBLE, B.DEFAULT_CFG)
    assert r["per_arm_verdict"] == {"R8_LEGAL": "FAILED", "R8_LEGAL_s1": "FAILED"}
    assert r["verdict"] == "FAILED"
    assert r["paired_vs_STOP"]["positive"] is False and r["paired_vs_STOP"]["drive_level"]["lo"] == -0.0001


def test_n2_both_seeds_must_agree():
    # seed 1 at 0.29 sits below max(STOP 0.2985, CV, ECHO) -> its own verdict FAILED; seed 0 PASSES -> NOT PROVEN, never PASS
    r = B.navhard_bar(_nh(s1=0.29), ELIGIBLE, B.DEFAULT_CFG)
    assert r["per_arm_verdict"]["R8_LEGAL"] == "PASS" and r["per_arm_verdict"]["R8_LEGAL_s1"] == "FAILED"
    assert r["verdict"] == "NOT PROVEN (inference seeds disagree: seed0=PASS, seed1=FAILED)"


def test_n2_seed1_pair_missing_is_not_a_pass():
    r = B.navhard_bar(_nh(s1_pair=False), ELIGIBLE, B.DEFAULT_CFG)
    assert r["per_arm_verdict"]["R8_LEGAL"] == "PASS" and r["per_arm_verdict"]["R8_LEGAL_s1"].startswith("UNAVAILABLE")
    assert r["verdict"].startswith("NOT PROVEN (seed-1 mirror not evaluable")


def test_n2_is_gated_by_the_label_and_unavailable_without_a_summary():
    assert B.navhard_bar(_nh(), {"label": "VALIDATION ONLY", "step": 40000, "weights": "ema"}, B.DEFAULT_CFG)["verdict"] == "NOT_EVALUATED -- VALIDATION ONLY"
    assert B.navhard_bar(None, ELIGIBLE, B.DEFAULT_CFG)["verdict"].startswith("UNAVAILABLE")


def _wu(blind=0.30, legal=0.60, s1="same"):
    arms = {"R8_LEGAL": legal, "CV_official": 0.20, "STOP_zero": 0.5212, "ECHO_ha0_ext": 0.50}
    if s1 is not None:
        arms["R8_LEGAL_s1"] = legal if s1 == "same" else s1
    if blind is not None:
        arms["R8_BLIND"] = blind
    return {"arms": {k: {"S2_EPDMS_u": {"value": v}} for k, v in arms.items()}, "seed_floor_S2_EPDMS_u": 0.001}


def test_n3_with_the_blind_control_literal():
    ok = B.warmup_bar(_wu(), ELIGIBLE, B.DEFAULT_CFG)                       # margin .60 - .5212 = .0788 > 2 x .001 ; BLIND .30 < floor -> FAILED
    assert ok["verdict"] == "PASS" and ok["blind_control_verdict"] == "FAILED"
    void = B.warmup_bar(_wu(blind=0.65), ELIGIBLE, B.DEFAULT_CFG)           # the deliberate regression PASSES -> the panel cannot see frames
    assert void["verdict"].startswith("VOID") and void["panel"].startswith("VOID")
    none = B.warmup_bar(_wu(blind=None), ELIGIBLE, B.DEFAULT_CFG)           # control absent -> N3 cannot PASS
    assert none["verdict"].startswith("UNAVAILABLE")
    assert B.warmup_bar(_wu(legal=0.40), ELIGIBLE, B.DEFAULT_CFG)["verdict"] == "FAILED"


def test_n3_both_seeds():
    # A-R8-NS1 AMENDMENT 3: seed 1 at 0.50 < max(CV .20, STOP .5212, ECHO .50) -> FAILED while seed 0 PASSES -> NOT PROVEN
    r = B.warmup_bar(_wu(s1=0.50), ELIGIBLE, B.DEFAULT_CFG)
    assert r["per_arm_verdict"] == {"R8_LEGAL": "PASS", "R8_LEGAL_s1": "FAILED"}
    assert r["verdict"] == "NOT PROVEN (inference seeds disagree: seed0=PASS, seed1=FAILED)"
    assert B.warmup_bar(_wu(s1=None), ELIGIBLE, B.DEFAULT_CFG)["verdict"].startswith("NOT PROVEN (seed-1 mirror not evaluable")


# ---- 8. artifact guards, the pinned estimator ---------------------------------------------------------------------------------------------
def test_count_guard_refuses_a_failed_or_partial_scorer():
    good = {"status": "PASS", "log_failed": 0, "C3_missing": 0, "C3_unexpected": 0}
    B.check_counts_guard("x.csv", good, "A")
    for bad in ({**good, "status": "FAIL"}, {**good, "log_failed": 3}, {**good, "C3_missing": 1}):
        with pytest.raises(B.BarsError):
            B.check_counts_guard("x.csv", bad, "A")


def test_the_estimator_is_the_refcv7_one_and_is_pinned():
    r = ev({"R8_LEGAL": 0.6, "R8_LEGAL_s1": 0.6, "STOP": 0.5})
    e = r["estimator"]
    assert (e["name"], e["paired"], e["cluster_unit"], e["sensitivity_unit"], e["n_boot"], e["seed"]) == \
           ("navsim_log_cluster_bootstrap", "paired_navsim_log_cluster_bootstrap", "log_name", "nuplan_drive", 2000, 0)
    assert "another draw of LOGS" in e["question_answered"]
    assert r["n_logs"] == 16 and "tier" in r and "NEVER closed loop" in r["tier"]


def test_every_arm_reports_all_five_subscores_and_pdms_never_pdms_alone():
    r = ev({"R8_LEGAL": 0.6, "R8_LEGAL_s1": 0.6, "STOP": 0.5}, cfg={"sub_intervals": True})
    a = r["arms"]["R8_LEGAL"]
    assert all(m in a for m in ("NC", "DAC", "TTC", "EP", "C", "DDC", "PDMS"))
    assert set(a["intervals_x100"]) == {"NC", "DAC", "TTC", "EP", "C", "PDMS"}


def test_city_census_cross_check_refuses_a_mismatch(tmp_path):
    # two independent derivations of 'Singapore' must agree; a census saying 1,913 against inputs saying 1,914 is a refusal.
    inp = tmp_path / "in.json.gz"
    import gzip
    toks = {f"t{i}": {"log_name": "l", "map_name": SG if i < 3 else US, "ego_statuses": [{"ego_velocity": [1.0, 0.0], "driving_command": [0, 1, 0, 0]}],
                      "human_future_poses": [[0, 0, 0]] * 8, "cv_poses": [[0, 0, 0]] * 8} for i in range(5)}
    with gzip.open(inp, "wt", encoding="utf-8") as fh:
        json.dump({"tokens": toks}, fh)
    ti = B.load_inputs(str(inp))
    assert int((ti["map_name"] == B.SG_MAP).sum()) == 3 and ti.loc["t0", "v0"] == 1.0
    bad = tmp_path / "cen.json"
    bad.write_text(json.dumps({"n_tokens": 5, "tokens_by_city": {SG: 2}}))
    run = tmp_path / "run"
    (run / "scores_navtest" / "x").mkdir(parents=True)
    with pytest.raises(B.BarsError):                                  # no R8_LEGAL in the dir: refused before anything is scored
        B.load_run(str(run), inputs=str(inp), floors_dir=str(tmp_path), r7_csv="", census=str(bad))


# ---- 9. real artifacts (skip when the disks are absent) ---------------------------------------------------------------------------------------
SMOKE = ("C:/Users/Admin/qland/landed_hold_r8navsim2/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/raw/"
         "smoke_cpu_ema40k_VALIDATION_ONLY")
_have_smoke = os.path.exists(os.path.join(SMOKE, "INDEX.json")) and os.path.exists(B.DEFAULT_INPUTS) and os.path.exists(B.DEFAULT_W3)


@pytest.mark.skipif(not _have_smoke, reason="the landed 5-token smoke / the navtest inputs / the banked floors are not on this disk")
def test_e2e_smoke_runs_end_to_end_and_is_labelled_validation_only(tmp_path):
    rc = B.main(["--run-dir", SMOKE, "--tag", "smoke5_VALIDATION_ONLY", "--out-dir", str(tmp_path)])
    assert rc == 0
    d = json.load(open(tmp_path / "NAVSIM_BARS_smoke5_VALIDATION_ONLY.json", encoding="utf-8"))
    assert d["label"].startswith("VALIDATION ONLY") and d["n_tokens"] == 5 and d["validation_only"] is True
    # the arms' PDMS (x100, 4 dp) are the literals the landed smoke published (INDEX.json summary_x100_4dp) -- NOT a refcv8 result
    assert {k: d["arms"][k]["PDMS"] for k in ("R8_LEGAL", "R8_LEGAL_s1", "R8_PRIV", "R8_RT", "PRIOR_ha0p")} == \
           {"R8_LEGAL": 41.6371, "R8_LEGAL_s1": 40.9421, "R8_PRIV": 46.9929, "R8_RT": 29.5351, "PRIOR_ha0p": 14.5337}
    assert all(d["arms"][a]["pdms_recompute_max_abs"] < 1e-12 for a in d["arms"])
    assert all(a["intervals_x100"]["PDMS"]["status"] == "UNAVAILABLE" for a in d["arms"].values())      # 5 logs < 8: no interval
    assert d["bars"]["BAR-R8-N1"]["verdict"] == d["bars"]["BAR-R8-N4"]["verdict"] == "NOT_EVALUATED -- VALIDATION ONLY"
    assert d["bars"]["R8_RT"]["verdict"].startswith("NO_BAR") and d["bars"]["R8_PRIV"]["verdict"].startswith("NO_BAR")
    assert d["controls"]["ep_clause_source_consistent"]["status"] == "OK" and d["controls"]["v0_inputs_vs_scorer_rows"]["status"] == "OK"
    assert d["strata"]["partition_sums_ok"] is True
    assert os.path.exists(tmp_path / "NAVSIM_BARS_smoke5_VALIDATION_ONLY.md")


R7_THOR = os.path.dirname(os.path.dirname(B.DEFAULT_R7))
_have_full = all(os.path.exists(p) for p in (B.DEFAULT_R7, B.DEFAULT_INPUTS, os.path.join(R7_THOR, "r7thor_s50400_R7_A1", "r7thor_s50400_R7_A1.csv"),
                                             os.path.join(R7_THOR, "r7thor_s50400_R7_A1_s1", "r7thor_s50400_R7_A1_s1.csv"),
                                             os.path.join(B.DEFAULT_W3, "STOP_navtest", "STOP_navtest_rows.jsonl")))


@pytest.mark.skipif(not _have_full, reason="the banked full-split refcv7 / W3 CSVs are not on this disk")
def test_fullsplit_control_reproduces_the_banked_refcv7_numbers():
    """The layer must READ KNOWN VALUES on the real 12,146-token split.  refcv7-50,400 LEGAL is the stand-in 'R8_LEGAL' here (a control on
    the layer, NOT a refcv8 number): banked PDMS 70.2048 and paired vs STOP +8.3846 [6.36, 10.24] (MODEL_REGISTRY 15.4.1, INHERITED);
    the banked R7_A1 vs R7_A1_s1 inference floor is 0.1398 (summary_navtest_vmaxoff_legal.THOR.json)."""
    ti = B.load_inputs(B.DEFAULT_INPUTS)
    stop_rows = B.read_rows_jsonl(os.path.join(B.DEFAULT_W3, "STOP_navtest", "STOP_navtest_rows.jsonl"))
    ti["mcp"] = pd.Series({t: float(r["max_compliant_progress_m"]) for t, r in stop_rows.items()})
    assert len(ti) == 12146 and int(ti["mcp"].notna().sum()) == 12146
    rd = lambda p: B.read_scores_csv(p)                                                       # noqa: E731
    stop = rd(os.path.join(B.DEFAULT_W3, "STOP_navtest", "STOP_navtest.csv"))
    leg = rd(B.DEFAULT_R7)
    a1 = rd(os.path.join(R7_THOR, "r7thor_s50400_R7_A1", "r7thor_s50400_R7_A1.csv"))
    a1s1 = rd(os.path.join(R7_THOR, "r7thor_s50400_R7_A1_s1", "r7thor_s50400_R7_A1_s1.csv"))
    meta = {"label": "control", "step": 48852, "weights": "ema"}
    r = B.evaluate_navtest({"R8_LEGAL": leg, "STOP": stop, B.R7_ARM: leg}, ti, meta, {"sub_intervals": False, "strata_pairs": False})
    assert round(100 * float(stop["PDMS"].mean()), 4) == 61.8202 and r["arms"]["R8_LEGAL"]["PDMS"] == 70.2048
    p = r["pairs"]["R8_LEGAL__minus__STOP"]
    assert p["delta_x100"] == 8.3846 and abs(p["interval"]["lo"] - 6.36) < 0.01 and abs(p["interval"]["hi"] - 10.24) < 0.01
    assert p["interval"]["positive"] is True and r["pairs"]["R8_LEGAL__minus__REFCV7_50400_LEGAL"]["delta_x100"] == 0.0
    assert r["n_tokens"] == 12146 and r["n_logs"] == 136 and r["strata"]["partition_sums_ok"] is True
    assert r["strata"]["partitions"]["city"]["Singapore"]["n"] == 1914                         # MEASURED navtest_city_census.json
    clause = r["strata"]["partitions"]["ep_clause"]["fired"]["n"]
    assert clause == 1006                                                                       # 8.28 % of 12,146, MEASURED on the STOP rows
    r2 = B.evaluate_navtest({"R8_LEGAL": a1, "R8_LEGAL_s1": a1s1, "STOP": stop}, ti, meta, {"sub_intervals": False, "strata_pairs": False})
    assert r2["inference_seed_floor"]["x100"] == 0.1398
