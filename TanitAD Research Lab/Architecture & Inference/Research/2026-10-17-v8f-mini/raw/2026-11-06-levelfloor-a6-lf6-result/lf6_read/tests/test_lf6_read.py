"""Tests of the LF6 reader (PREREG_LEVELFLOOR_A6 sha256 fbf5de02...): the joint reading of card section 5 in its order 1 to 5, the 24-rollout table and flags, the bounded-lift instruments and the bound control, the reuse of lfo_read.py's ceiling labels and lf_read.py's
dose tables, the consistency of the reader's table with lf6_precheck.py and with lf6_chain.sh, and the constants of the card (doses, kappa, tolerance).
Run:  CUDA_VISIBLE_DEVICES=-1 python -I -m pytest -q -p no:cacheprovider --basetemp F:/TanitAD-scratch/motion_c/lf6_read/pt/run <this file>   (LF6_READER = the reader under test; code/run_mutations_lf6_read.py points it at a mutated copy)."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PKG = HERE.parent
READER = Path(os.environ.get("LF6_READER", str(PKG / "code" / "lf6_read.py")))
spec = importlib.util.spec_from_file_location("lf6_read_under_test", str(READER))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
R.init()


def test_the_constants_of_the_card():
    assert R.DELTAS == {"05": 0.5, "10": 1.0} and R.DELTA_FLAG == {"05": "0.5", "10": "1.0"} and R.REPORT_ORDER == ("05", "10")
    assert R.KAPPA == 1.0 and R.KAPPA_FLAG == "1.0" and R.BASE_SMOOTH == "sg72" and R.SEEDS == (0, 1)
    assert R.BOUND_TOL_M == 1e-6
    assert R.LFR.KAPPA["B05"] == 1.0 and R.LFR.KAPPA["B10"] == 1.0 and R.LFR.KAPPA_FLAG["B05"] == "1.0"          # the dose keys the A3 rules are called with
    assert "five doses" in R.MULTIPLICITY and "0.50, 0.75 and 1.0" in R.MULTIPLICITY and "adaptivity" in R.MULTIPLICITY.lower()


def test_rollouts_are_the_twenty_four_of_the_card_in_priority_order():
    rows = R.rollouts()
    tags = [r[0] for r in rows]
    assert len(rows) == 24 and len(set(tags)) == 24
    assert tags[:4] == ["RILFB10_s0", "RILFB10_s1", "RILFB10C_s0", "RILFB10C_s1"]                                   # the admissible arms first, delta 1.0 then 0.5
    assert tags[4:8] == ["RILFB05_s0", "RILFB05_s1", "RILFB05C_s0", "RILFB05C_s1"]
    assert tags[8:12] == ["RILFOB10_s0", "RILFOB10_s1", "RILFOB10C_s0", "RILFOB10C_s1"] and tags[12:16][0] == "RILFOB05_s0"
    assert tags[16:20] == ["LFB10_s0", "LFB10_s1", "LFB10C_s0", "LFB10C_s1"] and tags[20:] == ["LFB05_s0", "LFB05_s1", "LFB05C_s0", "LFB05C_s1"]
    assert all(r[4] == "logged" for r in rows[16:]) and all(r[4] == "reindexed" for r in rows[:16])
    assert [r[6] for r in rows[:8]] == ["heads"] * 8 and [r[6] for r in rows[8:16]] == ["logged"] * 8 and [r[6] for r in rows[16:]] == ["heads"] * 8
    fl = {r[0]: R.expected_flags(r[6], r[7], r[4], r[5], r[3]) for r in rows}
    assert all("--floor-kappa 1.0 " in f for f in fl.values()) and all("--smooth sg72 " in f for f in fl.values())
    assert fl["RILFB10_s0"].endswith("--floor-vhat-source heads --floor-lift-cap 1.0") and fl["RILFB05C_s1"].endswith("--floor-vhat-source heads --floor-lift-cap 0.5") and "--feasible kbox" in fl["RILFB05C_s1"]
    assert fl["RILFOB05_s0"].endswith("--floor-vhat-source logged --floor-lift-cap 0.5") and "--floor-vhat-noise" not in "".join(fl.values())
    assert "--scene logged" in fl["LFB10C_s1"] and "--feasible kbox" in fl["LFB10C_s1"] and "--feasible" not in fl["LFB05_s0"] and "--scene reindexed" in fl["RILFOB10_s1"]
    assert "M2b_s1_recft_LLR" in fl["LFB05_s1"] and "M2b_s0_recft_LLR" in fl["RILFB10_s0"]


def test_the_chain_table_is_the_readers():
    out = subprocess.run(["C:/Program Files/Git/bin/bash.exe", "-c", f"bash '{(PKG / 'lf6_chain.sh').as_posix()}' --dry-run"], capture_output=True, text=True, stdin=subprocess.DEVNULL, env=dict(os.environ, LF6_PK=PKG.as_posix().replace("E:/", "/e/"))).stdout
    rows = [ln.split("\t") for ln in out.splitlines() if ln.count("\t") == 5]
    want = [[r[0], str(r[3]), r[4], r[5], r[6], R.DELTA_FLAG[R.delta_key(r[7])]] for r in R.rollouts()]
    assert rows == want


def test_the_precheck_parses_every_tag_as_the_reader_builds_it():
    for tag, base, comp, s, scene, feas, source, delta in R.rollouts():
        t = R.PCO.parse_tag(tag)
        assert (t["base"], t["composite"], t["seed"], t["source"], t["delta"]) == (base, comp, s, source, delta), tag
    assert R.PCO.DELTA_FLAG == {0.5: "0.5", 1.0: "1.0"}


def test_joint_reading_follows_card_section_5_in_its_order():
    j = R.joint_reading
    none = {"05": "P-FAILS-M1-MOVED", "10": "P-FAILS-M1-MOVED"}
    # 1. a counted DOMINANT first, whatever the ceiling says; both doses -> both reported, delta 0.5 named first
    one = j({"05": "P-FAILS-M1-MOVED", "10": "D-HOLDS"}, {"05": "CEILING-HARMS", "10": "CEILING-HARMS"})
    assert one["case"] == "LFB-COUNTED-DOMINANT" and one["counted_dominant"] is True and one["doses"] == [1.0] and "delta 1.0" in one["next"] and "scale" in one["next"] and "five were read" in one["next"]
    both = j({"05": "D-HOLDS", "10": "D-HOLDS"}, {"05": "CEILING-CLEARS", "10": "CEILING-CLEARS"})
    assert both["doses"] == [0.5, 1.0] and both["next"].index("delta 0.5") < both["next"].index("delta 1.0") and "smaller intervention" in both["next"]
    assert j({"05": "LEAK-BORNE", "10": "LEAK-BORNE"}, {"05": "CEILING-HARMS", "10": "CEILING-HARMS"})["counted_dominant"] is False            # a LEAK-BORNE arm is not a DOMINANT
    # 2. otherwise a CLEARS at either dose
    c = j(none, {"05": "CEILING-SHORT", "10": "CEILING-CLEARS"})
    assert c["case"] == "BOUNDED-FORM-WORKS-GIVEN-A-GOOD-LEVEL" and c["estimator_levers_signalled"] is True and not c["counted_dominant"] and c["doses"] == [1.0] and "temporal filter" in c["next"] and "A40" in c["next"]
    assert j(none, {"05": "CEILING-CLEARS", "10": "F-UNREACHED"})["case"] == "BOUNDED-FORM-WORKS-GIVEN-A-GOOD-LEVEL"
    # 3. otherwise HARMS at both doses
    h = j(none, {"05": "CEILING-HARMS", "10": "CEILING-HARMS"})
    assert h["case"] == "BOUNDING-DOES-NOT-REMOVE-THE-HARM" and h["estimator_levers_signalled"] is False and "exhausted on CPU" in h["next"] and "T2" in h["next"] and "A40" in h["next"] and "BACKLOG.md" in h["next"]
    # 4. otherwise SHORT at both, or SHORT at one and HARMS at the other
    for lab in ({"05": "CEILING-SHORT", "10": "CEILING-SHORT"}, {"05": "CEILING-SHORT", "10": "CEILING-HARMS"}, {"05": "CEILING-HARMS", "10": "CEILING-SHORT"}):
        s = j(none, lab)
        assert s["case"] == "THE-BOUND-REMOVES-THE-CLOSURE" and s["estimator_levers_signalled"] is False and "T2" in s["next"] and "A40" in s["next"] and "exhausted on CPU" in s["next"]
    # 5. F-UNREACHED at a dose (and no case above): the instrument first
    for lab in ({"05": "F-UNREACHED", "10": "CEILING-HARMS"}, {"05": "CEILING-SHORT", "10": "F-UNREACHED"}, {"05": "F-UNREACHED", "10": "F-UNREACHED"}):
        u = j(none, lab)
        assert u["case"] == "F-UNREACHED" and "instrument is checked first" in u["next"] and not u["counted_dominant"]
    assert "delta 0.5" in j(none, {"05": "F-UNREACHED", "10": "CEILING-HARMS"})["next"]


def test_ceiling_labels_are_lfo_reads_unchanged():
    reach_bad, ok = {"PASS": False}, {"PASS": True}
    assert R.LFO.decide(reach_bad)["label"] == "F-UNREACHED"
    assert R.LFO.decide(ok, {"PASS": False}, {"PASS": True})["label"] == "CEILING-SHORT"
    assert R.LFO.decide(ok, {"PASS": True}, {"PASS": False, "per_seed": {"s0": {"non_crawl_separated_worse": ["LAT"], "ade6_within_guard": True}, "s1": {"non_crawl_separated_worse": [], "ade6_within_guard": True}}})["label"] == "CEILING-HARMS"
    assert R.LFO.decide(ok, {"PASS": True}, {"PASS": True})["label"] == "CEILING-CLEARS"


def _ticks(n=4):
    tk = 0.2 * np.arange(1, 31)[None, :]
    s = np.cumsum(np.full((n, 30), 1.0), axis=1)
    sp = s.copy()
    sp[0] = s[0] + np.minimum(0.5 * tk[0], 0.3)                                                    # bound, delta term binds
    sp[1] = s[1] + 0.1                                                                              # bound, delta term does not bind
    return {"lf_bind": np.array([True, True, False, False][:n]), "lf_dterm": np.array([True, False, False, False][:n]), "lf_dterm_knots": np.array([5, 0, 0, 0][:n]), "lf_s": s, "lf_sp": sp, "floor_lift_cap": np.array(0.5)}


def test_bound_block_reads_the_dterm_share_the_bound_and_its_tolerance():
    b = R.bound_block(_ticks(), 0.5)
    assert b["n_ticks"] == 4 and b["n_bound"] == 2 and b["dterm_share_of_bound"] == 0.5 and b["n_dterm_bound_ticks"] == 1 and abs(b["mean_dterm_knots_where_bound"] - 2.5) < 1e-12
    assert b["bound_holds"] and b["sp_ge_s"] and b["unbound_ticks_have_sp_equal_s"] and b["dterm_only_on_bound_ticks"] and b["max_sp_minus_s_minus_delta_t_m"] <= 1e-6
    for excess, holds in ((5e-7, True), (5e-6, False)):                                            # the tolerance is 1e-6 m, no looser and no tighter
        t = _ticks()
        t["lf_sp"] = t["lf_sp"].copy()
        t["lf_sp"][0, 29] = t["lf_s"][0, 29] + 0.5 * 6.0 + excess
        assert R.bound_block(t, 0.5)["bound_holds"] is holds
    t = _ticks()
    t["lf_sp"] = t["lf_sp"].copy()
    t["lf_sp"][2, 3] = t["lf_s"][2, 3] + 0.01                                                      # an unbound tick that moved
    assert R.bound_block(t, 0.5)["unbound_ticks_have_sp_equal_s"] is False
    t = _ticks()
    t["lf_sp"] = t["lf_sp"].copy()
    t["lf_sp"][0, 0] = t["lf_s"][0, 0] - 1e-3                                                      # s' below s
    assert R.bound_block(t, 0.5)["sp_ge_s"] is False
    t = _ticks()
    t["lf_dterm"] = np.array([True, False, True, False])                                           # the delta term cannot bind on an unbound tick
    assert R.bound_block(t, 0.5)["dterm_only_on_bound_ticks"] is False


def test_the_bound_control_uses_the_cards_delta_for_the_arm():
    t = _ticks()
    assert R.bound_block(t, 0.5)["bound_holds"] is True and R.bound_block(t, 0.1)["bound_holds"] is False                  # the same ticks break a smaller delta


def test_err_block_reads_the_heads_overshoot():
    h = R._err_block(np.full(10, 8.0), np.full(10, 6.0), np.ones(10, bool))
    assert h["share_vhat_exceeds_logged"] == 1.0 and abs(h["mae_mps"] - 2.0) < 1e-12 and abs(h["relative_mae"] - 1 / 3) < 1e-12
    u = R._err_block(np.array([5.0, 7.0]), np.array([6.0, 6.0]), np.ones(2, bool))
    assert u["share_vhat_exceeds_logged"] == 0.5 and u["mae_mps"] == 1.0
    assert R._err_block(np.array([np.nan, 7.0]), np.array([6.0, 6.0]), np.ones(2, bool))["n"] == 1


def test_relabel_names_the_attribution_for_what_it_is_here():
    b = {"seeds": {"s0": {"RILFO_minus_RILF075": 1, "RILFOC_minus_RILF075C": 2, "RILF075_minus_RIC0_beside": 3, "attribution_rows_changed_vs_RILF075": 4, "family_tables": {"RILFO_minus_RILF075": 5}}}}
    d = R.relabel(b)["seeds"]["s0"]
    assert d["ARM_minus_RILFO100"] == 1 and d["ARMC_minus_RILFO100C"] == 2 and d["RILFO100_minus_RIC0_beside"] == 3 and d["attribution_rows_changed_vs_RILFO100"] == 4 and d["family_tables"] == {"ARM_minus_RILFO100": 5}
    assert not any("075" in k for k in d)
