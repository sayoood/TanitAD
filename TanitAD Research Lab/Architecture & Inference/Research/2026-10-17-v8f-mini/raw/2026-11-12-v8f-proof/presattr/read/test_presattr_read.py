"""pytest of the reader kit of card ADD114 (PRESATTR10K): synthetic data only (tiny worlds of 36 join rows). The model reader of card ADD106 is imported as a sibling module. One test per control with a planted
fault, every name branch of card section 4 (with the four over-all names), the clause of note N1 on the real-format lines (reference without and with the optional fields), the refusal of the names stage, the run of the
start-parse line of the driver on the marker as the pod wrote it, the wording scan and the path length."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import read_presattr10k as RE                    # noqa: E402

RE.bind()
BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
PRE = "PRESATTR10K"
LOW, HIGH, ZERO = [-0.05, -0.01], [0.01, 0.05], [-0.01, 0.01]
CK = "ab" * 32
W16 = "28bc31f21ecb4ccb" + "0" * 48
REAL_RR_K0 = "arm I10RQ50KF5Hb20_s0 form RR replan steps 10 rule every 5 readout causal_g kappa 0.2 model seed 0 reader seed 0 ckpt sha256 %s wrapper sha256 1234 started 2026-11-01T00:00:00Z" % CK
REAL_RLC_K0 = REAL_RR_K0.replace("I10RQ50KF5Hb20_s0", "I10RLcKF5Hb20_s0").replace("form RR", "form RLc")
REAL_RLC_K30 = ("arm I10RLcKF5P30Hb20_s0 form RLc replan steps 10 rule every 5 readout causal_g kappa 0.2 preroll 30 exec plan nav deadreckon grid grid_p30.json model seed 0 reader seed 0 "
                "ckpt sha256 %s wrapper sha256 1234 started 2026-11-01T00:00:00Z" % CK)
# the lines of the queue of card ADD114 (pod_presattr10k_queue.sh): optional fields present, the rule-join tail or the gap-file tail
NEW_LA_K0 = ("arm I10RQ50KF5Hb20aLA_s0 form RR replan steps 10 rule every 5 readout causal_g kappa 0.2 preroll 0 exec plan nav deadreckon grid eval_grid_windows.json model seed 0 reader seed 0 "
             "ckpt sha256 %s wrapper sha256 %s started 2026-11-22T14:21:50Z vis-gap-file /w/VGAPA_LA.npz gap file sha256 %s vis-closing 1" % (CK, W16, "cd" * 32))
NEW_PL_K0 = ("arm I10RLcKF5Hb20pLE_s0 form RLc replan steps 10 rule every 5 readout causal_g kappa 0.2 preroll 0 exec plan nav deadreckon grid eval_grid_windows.json model seed 0 reader seed 0 "
             "ckpt sha256 %s wrapper sha256 %s started 2026-11-22T14:21:50Z rule-join /w/JLE.npz rule join sha256 %s" % (CK, W16, "ef" * 32))
NEW_PL_K30 = NEW_PL_K0.replace("I10RLcKF5Hb20pLE", "I10RLcKF5P30Hb20pLE").replace("preroll 0", "preroll 30").replace("eval_grid_windows.json", "grid_p30.json")


@pytest.fixture()
def world(tmp_path):
    return RE.make_world(str(tmp_path / "w"))


def ev(w):
    a, m, wsha = w
    return RE.compute_controls(a, m, wsha)


# ------------------------------------------------------------------------------------------------ selftest and controls
def test_selftest_ok(capsys):
    assert RE.selftest() == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "ZZ READ-PRESATTR10K SELFTEST OK"


def test_clean_world_all_pass(world):
    c = ev(world)
    assert sorted(c) == sorted(RE.CONTROLS) == ["e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7"]
    assert all(v["pass"] is True for v in c.values()), {k: v["failures"] for k, v in c.items()}


@pytest.mark.parametrize("fault", RE.CONTROLS)
def test_planted_fault_flips_exactly_that_control(tmp_path, fault):
    c = ev(RE.make_world(str(tmp_path / "f"), fault))
    assert {k: v["pass"] for k, v in c.items()} == {k: (k != fault) for k in RE.CONTROLS}
    assert c[fault]["failures"]


@pytest.mark.parametrize("fault,ctl", [("e0_clause", "e0"), ("e1_run", "e1"), ("e2_default", "e2"), ("e6_done", "e6"), ("e7_present", "e7"), ("e7_missing", "e7"), ("e7_above", "e7"),
                                       ("e7_la", "e7"), ("e7_sha", "e7"), ("e7_closing", "e7"), ("e7_lafile", "e7")])
def test_further_planted_faults(tmp_path, fault, ctl):
    c = ev(RE.make_world(str(tmp_path / "f"), fault))
    assert c[ctl]["pass"] is False and c[ctl]["failures"]
    assert all(v["pass"] for k, v in c.items() if k != ctl), {k: v["failures"] for k, v in c.items() if k != ctl and not v["pass"]}


def test_e7_failure_text_counts_the_disagreeing_steps(tmp_path):
    c = ev(RE.make_world(str(tmp_path / "f"), "e7_present"))
    txt = " ".join(c["e7"]["failures"])
    assert "K0_PL_s1" in txt and "1 of 36 logged steps disagree" in txt
    c = ev(RE.make_world(str(tmp_path / "g"), "e7"))
    assert "differs from the lead_gap" in " ".join(c["e7"]["failures"])


def test_e0_hashes_and_clauses():
    rec = {"pass_e0": True, "clauses": {"a": True, "b": True}, "join": {"sha256": "aa" * 32}, "base": {"sha256": "bb" * 32}, "eval_join_sha256": "cc" * 32, "counts": {"n_replaced_rows_above_gap_max": 7}}
    ok = RE.ctl_e0(rec, "aa" * 32, "bb" * 32, "cc" * 32)
    assert ok["pass"] and ok["values"]["n_replaced_rows_above_gap_max"] == 7
    assert not RE.ctl_e0(dict(rec, pass_e0=False), "aa" * 32, "bb" * 32, "cc" * 32)["pass"]
    assert not RE.ctl_e0(dict(rec, clauses={"a": True, "b": False}), "aa" * 32, "bb" * 32, "cc" * 32)["pass"]
    assert not RE.ctl_e0(dict(rec, clauses={}), "aa" * 32, "bb" * 32, "cc" * 32)["pass"]
    for i in range(3):
        got = ["aa" * 32, "bb" * 32, "cc" * 32]
        got[i] = "00" * 32
        assert not RE.ctl_e0(rec, *got)["pass"], i
    assert not RE.ctl_e0(rec, None, "bb" * 32, "cc" * 32)["pass"] and not RE.ctl_e0(None, "aa" * 32, "bb" * 32, "cc" * 32)["pass"]


def test_e7_has_shares_for_the_six_runs_and_the_count(world):
    c = ev(world)
    sh = c["e7"]["values"]["rollout_shares"]
    assert sorted(sh) == sorted(RE.new_keys()) and len(RE.new_keys()) == 6
    assert all(v["share_of_rule_steps_with_a_lead"] > 0 for v in sh.values())
    assert c["e7"]["values"]["n_replaced_rows_above_gap_max"] == 1


def test_pl_check_by_a_plain_loop(world):
    """the vectorised pl_check agrees with a plain loop over the logged steps"""
    a = world[0]
    JL = RE.load_join_arrays(a.jle_file)
    logs = RE.load_logs_setting(a.pull, RE.run_name("K0", "PL"), 0, "K0")
    ci, j, pres, g = RE.rows4(logs)
    bad = 0
    for i in range(len(ci)):
        r = int(JL["off"][ci[i]]) + int(j[i])
        gg = float(JL["lead_gap"][r])
        val = int(JL["lead_valid"][r]) == 1 and np.isfinite(gg) and 0.0 <= gg <= 60.0
        bad += int(bool(pres[i]) != bool(val))
        if pres[i] and val:
            bad += int(abs(g[i] - gg) > 1e-3)
    assert bad == 0
    assert RE.pl_check((ci, j, pres, g), JL)["PASS"]
    j2 = j.copy()
    j2[0] = 12                                                       # past the last frame of the clip
    r = RE.pl_check((ci, j2, pres, g), JL)
    assert not r["PASS"] and r["n_unmapped"] == 1


def test_rows4_tolerates_a_log_without_a_closing_column():
    z = dict(ci=np.array([0, 1]), j=np.array([2, 3]), present=np.array([True, False]), g_src=np.array([5.0, np.nan]))
    ci, j, pres, g = RE.rows4([z, z])
    assert len(ci) == 4 and pres.sum() == 2


def test_e1_clauses():
    full = {k: RE.WRAPPER_SHA + "0" * 48 for k in RE.new_keys()}
    ok = RE.ctl_e1(RE.WRAPPER_SHA, "x\nZZ CHECK-WRAPPER-V PASS\n\n", "h " + RE.WRAPPER_SHA, True, full)
    assert ok["pass"]
    assert not RE.ctl_e1("f" * 16, "ZZ CHECK-WRAPPER-V PASS", "h " + RE.WRAPPER_SHA, True, full)["pass"]
    assert not RE.ctl_e1(RE.WRAPPER_SHA, "ZZ CHECK-WRAPPER-V PASS", "h " + RE.WRAPPER_SHA, False, full)["pass"]
    assert not RE.ctl_e1(RE.WRAPPER_SHA, "ZZ CHECK-WRAPPER-V FAIL", "h " + RE.WRAPPER_SHA, True, full)["pass"]
    assert not RE.ctl_e1(RE.WRAPPER_SHA, "ZZ CHECK-WRAPPER-V PASS", "h other", True, full)["pass"]
    k0 = RE.new_keys()[0]
    assert not RE.ctl_e1(RE.WRAPPER_SHA, "ZZ CHECK-WRAPPER-V PASS", "h " + RE.WRAPPER_SHA, True, dict(full, **{k0: "f" * 64}))["pass"]
    assert not RE.ctl_e1(RE.WRAPPER_SHA, "ZZ CHECK-WRAPPER-V PASS", "h " + RE.WRAPPER_SHA, True, dict(full, **{k0: None}))["pass"]


# ------------------------------------------------------------------------------------------------ e2 under note N1: real-format lines
def test_e2_real_format_lines_pass_against_a_reference_without_the_optional_fields():
    assert RE.d2_line_failures(RE.parse_run_line(NEW_LA_K0), RE.parse_run_line(REAL_RR_K0)) == []
    assert RE.d2_line_failures(RE.parse_run_line(NEW_PL_K0), RE.parse_run_line(REAL_RLC_K0)) == []


def test_e2_real_format_line_passes_against_a_reference_with_the_optional_fields():
    assert RE.d2_line_failures(RE.parse_run_line(NEW_PL_K30), RE.parse_run_line(REAL_RLC_K30)) == []


def test_e2_parse_of_the_real_lines():
    p = RE.parse_run_line(REAL_RR_K0)
    assert p == {"form": "RR", "replan": "10", "every": "5", "readout": "causal_g", "kappa": "0.2", "model_seed": "0", "reader_seed": "0", "ckpt": CK}
    p = RE.parse_run_line(NEW_PL_K30)
    assert p["preroll"] == "30" and p["exec"] == "plan" and p["nav"] == "deadreckon" and p["grid"] == "grid_p30.json" and p["form"] == "RLc" and "wrapper" not in p
    assert p["every"] == "5"                                          # the `rule-join` tail does not disturb `rule every`


def test_e2_a_new_run_with_a_pre_roll_fails_against_a_reference_without_the_field():
    bad = RE.d2_line_failures(RE.parse_run_line(NEW_PL_K0.replace("preroll 0", "preroll 30")), RE.parse_run_line(REAL_RLC_K0))
    assert any("preroll" in x for x in bad)


def test_e2_default_grid_and_nav_for_a_reference_without_them():
    bad = RE.d2_line_failures(RE.parse_run_line(NEW_LA_K0.replace("eval_grid_windows.json", "grid_p30.json")), RE.parse_run_line(REAL_RR_K0))
    assert any("grid" in x for x in bad)
    bad = RE.d2_line_failures(RE.parse_run_line(NEW_LA_K0.replace("nav deadreckon", "nav logged")), RE.parse_run_line(REAL_RR_K0))
    assert any("nav" in x for x in bad)
    assert any("grid" in x for x in RE.d2_line_failures(RE.parse_run_line(NEW_LA_K0.replace(" grid eval_grid_windows.json", "")), RE.parse_run_line(REAL_RR_K0)))


def test_e2_the_wrong_form_or_readout_fails():
    assert any("form" in x for x in RE.d2_line_failures(RE.parse_run_line(NEW_PL_K0), RE.parse_run_line(REAL_RR_K0)))
    assert any("readout" in x for x in RE.d2_line_failures(RE.parse_run_line(NEW_LA_K0.replace("readout causal_g", "readout pinned")), RE.parse_run_line(REAL_RR_K0)))


@pytest.mark.parametrize("field", ["form", "replan steps", "rule every", "readout", "kappa", "model seed", "reader seed"])
def test_e2_required_fields_must_be_in_both_lines(field):
    rx = {"form": r" form \S+", "replan steps": r" replan steps \S+", "rule every": r" rule every \S+", "readout": r" readout \S+", "kappa": r" kappa \S+", "model seed": r" model seed \S+",
          "reader seed": r" reader seed \S+"}[field]
    assert RE.d2_line_failures(RE.parse_run_line(re.sub(rx, "", NEW_PL_K30, count=1)), RE.parse_run_line(REAL_RLC_K30)) != []
    assert RE.d2_line_failures(RE.parse_run_line(NEW_PL_K30), RE.parse_run_line(re.sub(rx, "", REAL_RLC_K30, count=1))) != []


def test_e2_checkpoint_numbers_and_wrapper_hash():
    assert any("ckpt" in x for x in RE.d2_line_failures(RE.parse_run_line(NEW_LA_K0.replace(CK, "cd" * 32)), RE.parse_run_line(REAL_RR_K0)))
    assert RE.d2_line_failures(RE.parse_run_line(NEW_LA_K0.replace("kappa 0.2", "kappa 0.20")), RE.parse_run_line(REAL_RR_K0)) == []
    assert RE.d2_line_failures(RE.parse_run_line(NEW_LA_K0), RE.parse_run_line(REAL_RR_K0.replace("wrapper sha256 1234", "wrapper sha256 ffff"))) == []


def test_e2_reference_arms():
    assert RE.ref_arm("K0", "LA") == "RR" and RE.ref_arm("K0", "PL") == "RLc" and RE.ref_arm("K30", "PL") == "RLc"


def test_e2_rule_share_above_zero_and_a_missing_line(world):
    a = world[0]
    lines = {"%s_%s_s%d" % (S, arm, k): (RE.read_text(RE.run_text_path(a.pull, RE.run_name(S, arm), k)), RE.read_text(RE.run_text_path(a.pull, RE.run_name(S, RE.ref_arm(S, arm)), k)))
             for S, arm in RE.new_pairs() for k in RE.SEEDS}
    rc = {k: {"share_of_rule_steps_with_a_lead": 0.5} for k in lines}
    assert RE.ctl_e2(lines, rc)["pass"]
    assert not RE.ctl_e2(lines, dict(rc, K30_PL_s1={"share_of_rule_steps_with_a_lead": 0.0}))["pass"]
    assert not RE.ctl_e2(dict(lines, K30_PL_s1=("", lines["K30_PL_s1"][1])), rc)["pass"]
    assert not RE.ctl_e2(dict(lines, K30_PL_s1=(lines["K30_PL_s1"][0], "")), rc)["pass"]


# ------------------------------------------------------------------------------------------------ e3 to e6
def banked_levels():
    return {S: {arm: dict(zip("CUF", v)) for arm, v in arms.items()} for S, arms in RE.EXPECTED.items()}


def test_e3_each_value():
    assert RE.EXPECTED["K30"]["LA"] == (0.0, 0.0839, 0.4877) and RE.EXPECTED["K0"]["RR"] == (0.0296, 0.0738, 0.1682)
    base = banked_levels()
    assert RE.ctl_e3(base)["pass"]
    for S in base:
        for arm in base[S]:
            for c in "CUF":
                x = json.loads(json.dumps(base))
                x[S][arm][c] += 0.00011
                assert not RE.ctl_e3(x)["pass"], (S, arm, c)
                x[S][arm][c] -= 0.00002
                assert RE.ctl_e3(x)["pass"], (S, arm, c)
    assert not RE.ctl_e3(None)["pass"] and not RE.ctl_e3({"K0": {"GEc": {}}})["pass"]
    x = json.loads(json.dumps(base))
    del x["K0"]["RLc"]
    assert not RE.ctl_e3(x)["pass"]


def test_e3_is_read_by_the_function_of_the_new_cells():
    src = (HERE / "read_presattr10k.py").read_text(encoding="utf-8")
    assert "for S in SETTINGS:" in src and "measure_setting(a, E, S, keys, ctx_cache, log)" in src and src.count("measure_setting(") == 2


def test_e4_times_and_text():
    q, r = "..\n12 passed in 3s\n", "..\n30 passed, 1 warning in 4s\n"
    assert RE.ctl_e4(q, r, 1000, 1010, "1100", 1200)["pass"]
    assert RE.ctl_e4(q, r, 1100, 1200, "1100", 1200)["pass"]                                  # not later than: equal is allowed
    assert not RE.ctl_e4(q, r, 1101, 1010, "1100", 1200)["pass"]
    assert not RE.ctl_e4(q, r, 1000, 1201, "1100", 1200)["pass"]
    assert not RE.ctl_e4(q, r, 1000, 1010, None, 1200)["pass"]
    assert not RE.ctl_e4(q, r, 1000, 1010, "1100", "")["pass"]
    assert RE.ctl_e4(q, r, "2026-11-22T10:00:00Z", "2026-11-22T10:01:00Z", "1795345027", "2026-11-22T11:00:00Z")["pass"]
    assert not RE.ctl_e4("..\n0 passed\n", r, 1000, 1010, "1100", 1200)["pass"]
    assert not RE.ctl_e4("", r, 1000, 1010, "1100", 1200)["pass"]
    assert not RE.ctl_e4("..\n1 failed, 11 passed in 3s\n", r, 1000, 1010, "1100", 1200)["pass"]
    assert not RE.ctl_e4(q, "..\n29 passed, 1 error in 3s\n", 1000, 1010, "1100", 1200)["pass"]


def test_e5_each_part():
    ok = {S: {"grid_all_pass": True, "strategic_a": True, "strategic_b": True} for S in RE.SETTINGS}
    assert RE.ctl_e5(ok)["pass"]
    for S in RE.SETTINGS:
        for k in ("grid_all_pass", "strategic_a", "strategic_b"):
            x = {s: dict(v) for s, v in ok.items()}
            x[S][k] = False
            assert not RE.ctl_e5(x)["pass"], (S, k)
    x = dict(ok)
    del x["K30"]
    assert not RE.ctl_e5(x)["pass"]


def test_e6_counts_and_done():
    keys = RE.new_keys()
    done = {k: True for k in keys}
    nd = {k: 141 for k in keys}
    assert RE.ctl_e6(done, nd, 141)["pass"]
    assert not RE.ctl_e6(dict(done, K0_LA_s1=False), nd, 141)["pass"]
    assert not RE.ctl_e6(done, dict(nd, K30_PL_s0=140), 141)["pass"]
    assert not RE.ctl_e6(done, dict(nd, K30_PL_s0=None), 141)["pass"]
    assert RE.EXPECT_DUMPS == 141
    assert RE.ctl_e6(done, nd, 141, "ZZ PRESATTR10K DONE 2026-11-22T12:00:00Z all steps done\n")["pass"]
    assert not RE.ctl_e6(done, nd, 141, "ZZ PRESATTR10K DONE\n")["pass"]
    assert not RE.ctl_e6(done, nd, 141, "ZZ CANDSET10K DONE 2026-11-22T12:00:00Z all steps done\n")["pass"]


# ------------------------------------------------------------------------------------------------ names (card section 4)
def payload(ivs=None):
    return RE.mk_payload(ivs)


C1, C2, C3, C4 = "LA-VS-RLc", "GEc-VS-PL", "PL-VS-RLc", "GEc-VS-LA"


@pytest.mark.parametrize("c,pat", [(HIGH, "RAISES-%s"), (LOW, "LOWERS-%s"), (ZERO, "%s-NOT-SEPARATED"), ([-0.05, 0.0], "%s-NOT-SEPARATED"), ([0.0, 0.05], "%s-NOT-SEPARATED")])
@pytest.mark.parametrize("S", ["K0", "K30"])
@pytest.mark.parametrize("cn,pn", list(RE.CONTRASTS.items()))
@pytest.mark.parametrize("mk,mw", [("U", "UNSAFE"), ("C", "CONTACT"), ("F", "FALLING-BEHIND")])
def test_every_contrast_measure_branch(c, pat, S, cn, pn, mk, mw):
    nm = RE.all_names(payload({(S, pn, mk): c}))
    assert nm["%s_%s_%s" % (S, cn, mw)] == "%s-%s-%s-%s" % (PRE, S, cn, pat % mw)
    others = [v for k, v in nm.items() if k != "%s_%s_%s" % (S, cn, mw) and not k.endswith("OVERALL")]
    assert all("NOT-SEPARATED" in v for v in others)                  # nothing else moved


def test_contrast_names_are_the_ones_of_the_card():
    assert RE.CONTRASTS == {"PRESENCE-AT-LABEL-GAP": "LA-VS-RLc", "PRESENCE-AT-CAMERA-GAP": "GEc-VS-PL", "GAP-AT-LABEL-PRESENCE": "PL-VS-RLc", "GAP-AT-CAMERA-PRESENCE": "GEc-VS-LA"}
    nm = RE.all_names(payload({("K0", C2, "U"): HIGH}))
    assert nm["K0_PRESENCE-AT-CAMERA-GAP_UNSAFE"] == "PRESATTR10K-K0-PRESENCE-AT-CAMERA-GAP-RAISES-UNSAFE"
    assert nm["K30_GAP-AT-LABEL-PRESENCE_CONTACT"] == "PRESATTR10K-K30-GAP-AT-LABEL-PRESENCE-CONTACT-NOT-SEPARATED"
    assert len(nm) == 2 * (4 * 3 + 1)


def test_the_unsafe_name_reads_only_the_unsafe_interval():
    nm = RE.all_names(payload({("K0", C1, "C"): HIGH, ("K0", C3, "F"): HIGH}))
    assert nm["K0_OVERALL"] == PRE + "-K0-UNSAFE-NOT-ATTRIBUTED"


@pytest.mark.parametrize("raising,word", [
    ([], "UNSAFE-NOT-ATTRIBUTED"),
    ([C1], "PRESENCE-CARRIES-UNSAFE"), ([C2], "PRESENCE-CARRIES-UNSAFE"), ([C1, C2], "PRESENCE-CARRIES-UNSAFE"),
    ([C3], "GAP-CARRIES-UNSAFE"), ([C4], "GAP-CARRIES-UNSAFE"), ([C3, C4], "GAP-CARRIES-UNSAFE"),
    ([C1, C3], "BOTH-CARRY-UNSAFE"), ([C2, C4], "BOTH-CARRY-UNSAFE"), ([C1, C2, C3, C4], "BOTH-CARRY-UNSAFE")])
def test_over_all_names(raising, word):
    nm = RE.all_names(payload({("K0", pn, "U"): HIGH for pn in raising}))
    assert nm["K0_OVERALL"] == "%s-K0-%s" % (PRE, word)
    assert nm["K30_OVERALL"] == PRE + "-K30-UNSAFE-NOT-ATTRIBUTED"


def test_a_lowering_contrast_does_not_count_as_raising():
    assert RE.all_names(payload({("K0", C1, "U"): LOW, ("K0", C3, "U"): HIGH}))["K0_OVERALL"] == PRE + "-K0-GAP-CARRIES-UNSAFE"
    assert RE.all_names(payload({("K0", C1, "U"): [0.0, 0.05]}))["K0_OVERALL"] == PRE + "-K0-UNSAFE-NOT-ATTRIBUTED"


def test_the_pair_GEc_VS_RLc_has_no_name():
    nm = RE.all_names(payload({("K0", "GEc-VS-RLc", m): HIGH for m in "CUF"}))
    assert all("GEc" not in v for v in nm.values()) and nm["K0_OVERALL"] == PRE + "-K0-UNSAFE-NOT-ATTRIBUTED"
    assert "GEc-VS-RLc" in RE.PAIRS


def test_two_derivations_agree_on_random_intervals():
    import random
    rnd = random.Random(5)
    pool = [LOW, HIGH, ZERO, [-0.02, 0.0], [0.0, 0.02]]
    for _ in range(300):
        ivs = {(S, nm, m): rnd.choice(pool) for S in RE.SETTINGS for nm in RE.PAIRS for m in "CUF"}
        p = payload(ivs)
        assert RE.names_of(p) == RE.derive_all(p)


def test_disagreement_is_refused(monkeypatch):
    monkeypatch.setattr(RE, "derive_all", lambda p: dict(RE.names_of(p), K0_OVERALL="x"))
    with pytest.raises(SystemExit):
        RE.all_names(payload())


# ------------------------------------------------------------------------------------------------ stages
def good_controls(tmp_path):
    p = payload({("K0", C2, "U"): HIGH})
    for S in RE.SETTINGS:
        d = p["settings"][S]
        d["levels_per_seed"] = {a: {c: [0.0, 0.0] for c in "CUF"} for a in RE.ARMS}
        d["counts"] = {"n_windows": 10, "n_lead_windows": 5, "n_episodes_with_lead": 3}
        d["families"], d["strategic"] = {}, {"pairs": {}}
        d["strategic_reason"] = "x"
        for nm in RE.PAIRS:
            for m in "CUF":
                d["pairs"][nm][m].update({"pooled": 0.0, "per_seed": [0.0, 0.0]})
    ctl = {k: {"pass": True, "failures": []} for k in RE.CONTROLS}
    res = {"card": RE.CARD, "all_pass": True, "controls": ctl, "payload": dict(p, control_values={"e7": {"rollout_shares": {}, "n_replaced_rows_above_gap_max": 12}})}
    f = tmp_path / "c.json"
    f.write_text(json.dumps(res), encoding="utf-8")
    return f, res


def test_names_stage_refuses_when_not_all_pass(tmp_path, capsys):
    f, res = good_controls(tmp_path)
    for k in RE.CONTROLS:
        bad = json.loads(json.dumps(res))
        bad["controls"][k]["pass"] = False
        g = tmp_path / ("b_%s.json" % k)
        g.write_text(json.dumps(bad), encoding="utf-8")
        assert RE.main(["--stage", "names", "--controls", str(g), "--out", str(tmp_path / ("o_%s.json" % k))]) == 3
        assert not (tmp_path / ("o_%s.json" % k)).exists()
    bad = json.loads(json.dumps(res))
    del bad["controls"]["e7"]
    g = tmp_path / "m.json"
    g.write_text(json.dumps(bad), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(g), "--out", str(tmp_path / "o_m.json")]) == 3
    bad = json.loads(json.dumps(res))
    bad["all_pass"] = False
    bad["withheld"] = bad.pop("payload")
    g = tmp_path / "w.json"
    g.write_text(json.dumps(bad), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(g), "--out", str(tmp_path / "o_w.json")]) == 3
    assert "ZZ READ-PRESATTR10K names refused" in capsys.readouterr().out


def test_names_stage_writes_result_and_log(tmp_path, capsys):
    f, res = good_controls(tmp_path)
    o = tmp_path / "r.json"
    assert RE.main(["--stage", "names", "--controls", str(f), "--out", str(o)]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "ZZ READ-PRESATTR10K names done"
    assert out[0].startswith("TIER T1-kin/hb0.2 (kinematic rollout on the model's own actions")
    assert "no pre-roll" in " ".join(out) and "pre-roll of 30 steps (6 s)" in " ".join(out)
    assert any(x == PRE + "-K0-PRESENCE-AT-CAMERA-GAP-RAISES-UNSAFE" for x in out) and any(x == PRE + "-K0-PRESENCE-CARRIES-UNSAFE" for x in out)
    assert any(x.startswith("count n_replaced_rows_above_gap_max 12 ") for x in out)
    r = json.loads(o.read_text(encoding="utf-8"))
    assert r["controls_sha256"] == RE.sha_file(str(f)) and "K0_OVERALL" in r["names"] and set(r) >= {"card", "names", "tier", "values"}
    assert (tmp_path / "r_names.log").exists()
    assert RE.main(["--stage", "names", "--controls", str(f), "--out", str(tmp_path / "r.json")]) == 2          # refuses to write into an existing out


def test_names_stage_refuses_a_controls_file_of_another_card(tmp_path):
    f, res = good_controls(tmp_path)
    res["card"] = "PREREG_V8F_PROOF_ADD113"
    g = tmp_path / "x.json"
    g.write_text(json.dumps(res), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(g), "--out", str(tmp_path / "ox.json")]) == 3


def test_assemble_controls_withholds_on_fail():
    ctl = {k: {"pass": True, "failures": []} for k in RE.CONTROLS}
    assert "payload" in RE.assemble_controls(ctl, {"x": 1}, {}) and "withheld" not in RE.assemble_controls(ctl, {"x": 1}, {})
    ctl["e3"] = {"pass": False, "failures": ["x"]}
    r = RE.assemble_controls(ctl, {"x": 1}, {})
    assert r["all_pass"] is False and "withheld" in r and "payload" not in r


def test_stdout_of_controls_has_only_the_allowed_lines(tmp_path, monkeypatch, capsys):
    w = RE.make_world(str(tmp_path / "w"))
    monkeypatch.setattr(RE, "run_controls", lambda a: RE.assemble_controls(RE.compute_controls(w[0], w[1], w[2]), {"settings": w[1]["settings"]}, {}))
    out = tmp_path / "C.json"
    assert RE.main(["--stage", "controls", "--out", str(out)]) == 0
    assert capsys.readouterr().out.strip().splitlines() == ["e%d PASS" % i for i in range(8)] + ["ZZ READ-PRESATTR10K controls all_pass True"]
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["all_pass"] is True and "payload" in res and "withheld" not in res and sorted(res["controls"]) == sorted(RE.CONTROLS)


def test_stdout_of_a_failed_controls_stage_withholds(tmp_path, monkeypatch, capsys):
    w = RE.make_world(str(tmp_path / "w"), "e3")
    monkeypatch.setattr(RE, "run_controls", lambda a: RE.assemble_controls(RE.compute_controls(w[0], w[1], w[2]), {"settings": w[1]["settings"]}, {}))
    out = tmp_path / "C.json"
    assert RE.main(["--stage", "controls", "--out", str(out)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert "e3 FAIL" in lines and lines[-1] == "ZZ READ-PRESATTR10K controls all_pass False" and len(lines) == 9
    res = json.loads(out.read_text(encoding="utf-8"))
    assert "withheld" in res and "payload" not in res


def test_settings_table_matches_the_card():
    S = RE.SPEC
    assert S["K0"]["runs"] == {"RLc": "RLcKF5Hb20", "LA": "RQ50KF5Hb20aLA", "PL": "RLcKF5Hb20pLE", "GEc": "RQ50KF5Hb20gGEc", "RR": "RQ50KF5Hb20"} and S["K0"]["P"] == 0 and S["K0"]["grid_file"] == "pinned"
    assert S["K30"]["runs"] == {"RLc": "RLcKF5P30Hb20", "LA": "RQ50KF5P30Hb20aLA", "PL": "RLcKF5P30Hb20pLE", "GEc": "RQ50KF5P30Hb20gGEc", "RR": "RQ50KF5P30Hb20"} and S["K30"]["P"] == 30
    assert S["K30"]["grid_file"] == "grid_p30.json"
    assert RE.preroll_text(0) == "no pre-roll" and RE.preroll_text(30) == "pre-roll of 30 steps (6 s)"
    assert RE.PAIRS == {"LA-VS-RLc": ("LA", "RLc"), "GEc-VS-PL": ("GEc", "PL"), "PL-VS-RLc": ("PL", "RLc"), "GEc-VS-LA": ("GEc", "LA"), "GEc-VS-RLc": ("GEc", "RLc")}
    assert RE.MD.WRAPPER_SHA == RE.WRAPPER_SHA == "28bc31f21ecb4ccb"
    assert len(RE.new_keys()) == 6 and RE.NEW_ARMS == {"K0": ("LA", "PL"), "K30": ("PL",)}


def test_model_is_imported_not_edited():
    src = (HERE / "read_presattr10k.py").read_text(encoding="utf-8")
    assert "spec_from_file_location" in src and "def w2_one" not in src and "def lead_multi" not in src and "def read_grid" not in src
    for fn in ("rows_of", "load_logs", "lead_multi", "w2_one", "gapfile_ref", "meta_of", "run_file_sha", "ctl_g8"):
        assert hasattr(RE.MD, fn)
        assert "MD." + fn in src
    for fn in ("read_grid", "strategic_grid", "ucf", "reading", "cells_of", "family_of", "grid_keys", "engine"):
        assert "R." + fn in src


# ------------------------------------------------------------------------------------------------ the start-parse line of the driver
def start_line():
    ls = [x for x in (HERE / "podread_presattr10k.sh").read_text(encoding="utf-8").splitlines() if "QS=$(" in x]
    assert len(ls) == 1, ls
    return ls[0].strip()


def run_start_parse(tmp_path, first_line):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("no bash")
    with open(str(tmp_path / "PRESATTR10K_START.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(first_line + "\nsecond line\n")
    with open(str(tmp_path / "t.sh"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write('set -u\nGE=.\nstop() { echo "STOP $*"; exit 3; }\n' + start_line() + '\necho "QS=$QS"\n')
    return subprocess.run([bash, "t.sh"], cwd=str(tmp_path), capture_output=True, encoding="utf-8")


def test_start_parse_line_on_the_marker_as_the_pod_wrote_it(tmp_path):
    r = run_start_parse(tmp_path, "1795357310 2026-11-22T14:21:50Z pid 782478")
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "QS=1795357310", (r.stdout, r.stderr)
    assert RE.to_epoch("1795357310") == 1795357310.0


def test_start_parse_line_stops_on_an_iso_first_token(tmp_path):
    r = run_start_parse(tmp_path, "2026-11-22T14:21:50Z 1795357310 pid 782478")
    assert r.returncode == 3 and "STOP START-MARKER-UNREADABLE" in r.stdout and "QS=" not in r.stdout


def test_start_parse_line_stops_on_an_empty_marker(tmp_path):
    r = run_start_parse(tmp_path, "")
    assert r.returncode == 3 and "STOP START-MARKER-UNREADABLE" in r.stdout


def test_queue_marker_formats_and_run_names_of_the_queue_script():
    q = (HERE.parent / "queue" / "pod_presattr10k_queue.sh").read_text(encoding="utf-8")
    assert 'echo "$(date +%s) $(ts) pid $$" > "$GP/PRESATTR10K_START.txt"' in q
    assert 'echo "ZZ PRESATTR10K DONE $(ts) all steps done" > "$GP/PRESATTR10K_DONE.txt"' in q
    for nm in ("I10RQ50KF5Hb20aLA", "I10RLcKF5Hb20pLE", "I10RLcKF5P30Hb20pLE"):
        assert nm in q
    for S, arm in RE.new_pairs():
        assert "I10" + RE.run_name(S, arm) in q


# ------------------------------------------------------------------------------------------------ files: wording, paths, scripts
def kit_files():
    return [p for p in HERE.iterdir() if p.is_file() and p.suffix in (".py", ".sh", ".md")]


def test_wording_scan_and_path_length():
    names = {p.name for p in kit_files()}
    assert {"read_presattr10k.py", "podread_presattr10k.sh", "pull_presattr.sh", "push_presattr_read.sh", "test_presattr_read.py", "PRESATTR_READ_DEFS.md"} <= names
    for p in kit_files():
        low = p.read_text(encoding="utf-8").lower()
        for w in BANNED:
            assert w not in low, (p.name, w)
        rel = "/f/TanitAD-scratch/v8proof/presattr/read/" + p.name
        assert len(rel) <= 200 and p.stem.lower() not in ("aux", "con", "prn", "nul")


def test_no_file_above_5_mb():
    for p in HERE.rglob("*"):
        if p.is_file():
            assert p.stat().st_size <= 5 * 1024 * 1024, p


def test_shell_files_are_lf_only_and_parse():
    bash = shutil.which("bash")
    for nm in ("podread_presattr10k.sh", "pull_presattr.sh", "push_presattr_read.sh"):
        assert b"\r" not in (HERE / nm).read_bytes(), nm
        if bash:
            r = subprocess.run([bash, "-n", str(HERE / nm)], capture_output=True, encoding="utf-8")
            assert r.returncode == 0, (nm, r.stderr)


def test_driver_rules():
    pod = (HERE / "podread_presattr10k.sh").read_text(encoding="utf-8")
    for must in ("PRESATTR10K_DONE.txt", "PRESATTR10K_STOP.txt", "PRESATTR10K_START.txt", "stat -c %Y", "--selftest", "-ge 20", "ALREADY-READ", "CUDA_VISIBLE_DEVICES=-1", "GE=$V/presattr",
                 "K=$GE/kit_read", "OUT=$GE/read", "LT=$GE/lead_tests", "lead_pytest_queue.out", "lead_pytest_read.out", "PRESATTR10K_CONTROLS.json", "PRESATTR10K_RESULT.json", "read_names.out",
                 "ZZ PODREAD-PRESATTR10K", "^e[0-7] (PASS|FAIL)|^ZZ READ-PRESATTR10K", "--queue-start \"$QS\"", "--controls-start \"$CSTART\"", "VGAPLE.npz", "VGAPA_LA.npz", "JLE.npz", "JLE_RECORD.json",
                 "GAPERR_GAPFILES_RECORD.json", "read_presattr10k.py", "v8f_t1_eval_reader_rule_rzgfhqetjv.py", "I10RQ50KF5Hb20aLA", "I10RLcKF5Hb20pLE", "I10RLcKF5P30Hb20pLE", "I10R0Hb20",
                 "grid_p30.json", "eval_grid_windows.json", "CS=$V/cropscale", "MODEL=$CS/kit_read/read_cropscale10k.py", "$(ts)", "PJ=$GE/joins", "--la-file", "--jle-file", "--jle-record", "--eval-join"):
        assert must in pod, must
    for stale in ("CANDSET", "GAPERR10K", "FALLATTR", "grid_p60", "d[0-6]"):
        assert stale not in pod, stale
    for banned_cmd in ("rm ", "mv ", "kill", "pkill"):
        assert banned_cmd not in pod, banned_cmd
    pull = (HERE / "pull_presattr.sh").read_text(encoding="utf-8")
    assert "sed '/hf_/d'" in pull and "*.npz|*.pt) continue" in pull and "pod_add114" in pull and "sha256sum" in pull and "/workspace/v8proof/presattr" in pull and "REFUSED" in pull
    assert "add113" not in pull and ".npz " not in pull.split("FILES=")[1].split("\n")[0]
    push = (HERE / "push_presattr_read.sh").read_text(encoding="utf-8")
    assert "set -C" in push and "sha256sum" in push and "kit_read" in push and "read_presattr10k.py podread_presattr10k.sh" in push and "/workspace/v8proof/presattr" in push


def test_reader_does_not_touch_lead_test_outputs_or_delete():
    src = (HERE / "read_presattr10k.py").read_text(encoding="utf-8")
    for bad in ("os.remove", "os.unlink", "shutil.rmtree", "os.rename", "shutil.move"):
        assert bad not in src
