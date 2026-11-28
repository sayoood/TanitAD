"""pytest of the reader kit of card ADD113 (CANDSET10K): synthetic data only (tiny worlds of 36 join rows). The model reader of card ADD106 is imported as a sibling module. One test per control with a planted
fault, every name branch of card section 4, the presence clause of note N1 on the real-format lines, the refusal of the names stage, the run of the start-parse line of the driver on the marker as the pod wrote it,
the wording scan and the path length."""
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import read_candset10k as RE                    # noqa: E402

RE.bind()
BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
PRE = "CANDSET10K"
LOW, HIGH, ZERO = [-0.05, -0.01], [0.01, 0.05], [-0.01, 0.01]
CK = "ab" * 32
REAL_CONV = "arm I10RQ50Hb20_s0 form RR replan steps 10 rule every 10 readout pinned kappa 0.2 model seed 0 reader seed 0 ckpt sha256 %s wrapper sha256 1234 started 2026-11-01T00:00:00Z" % CK
REAL_K0 = "arm I10RQ50KF5Hb20_s0 form RR replan steps 10 rule every 5 readout causal_g kappa 0.2 model seed 0 reader seed 0 ckpt sha256 %s wrapper sha256 1234 started 2026-11-01T00:00:00Z" % CK
REAL_K60 = ("arm I10RQ50KF5P60Hb20_s0 form RR replan steps 10 rule every 5 readout causal_g kappa 0.2 preroll 60 exec plan nav deadreckon grid grid_p60.json model seed 0 reader seed 0 "
            "ckpt sha256 %s wrapper sha256 1234 started 2026-11-01T00:00:00Z" % CK)
NEW_CONV = ("arm I10RQ50Hb20gGEc_s0 form RR replan steps 10 rule every 10 readout pinned kappa 0.2 preroll 0 exec plan nav deadreckon grid eval_grid_windows.json model seed 0 reader seed 0 "
            "ckpt sha256 %s wrapper sha256 28bc31f21ecb4ccb%s started x vis-gap-file /w/VGAPLE.npz gap file sha256 55de2f7c vis-closing 1" % (CK, "0" * 48))
NEW_K0 = NEW_CONV.replace("I10RQ50Hb20gGEc", "I10RQ50KF5Hb20gGEc").replace("rule every 10 readout pinned", "rule every 5 readout causal_g")
NEW_K60 = NEW_K0.replace("I10RQ50KF5Hb20gGEc", "I10RQ50KF5P60Hb20gGEc").replace("preroll 0", "preroll 60").replace("eval_grid_windows.json", "grid_p60.json")


@pytest.fixture()
def world(tmp_path):
    return RE.make_world(str(tmp_path / "w"))


def ev(w):
    a, rec, m, wsha = w
    return RE.compute_controls(a, rec, m, wsha)


# ------------------------------------------------------------------------------------------------ selftest and controls
def test_selftest_ok(capsys):
    assert RE.selftest() == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "ZZ READ-CANDSET10K SELFTEST OK"


def test_clean_world_all_pass(world):
    c = ev(world)
    assert sorted(c) == sorted(RE.CONTROLS) == ["d0", "d1", "d2", "d3", "d4", "d5", "d6"]
    assert all(v["pass"] is True for v in c.values()), {k: v["failures"] for k, v in c.items()}


@pytest.mark.parametrize("fault", RE.CONTROLS)
def test_planted_fault_flips_exactly_that_control(tmp_path, fault):
    c = ev(RE.make_world(str(tmp_path / "f"), fault))
    assert {k: v["pass"] for k, v in c.items()} == {k: (k != fault) for k in RE.CONTROLS}
    assert c[fault]["failures"]


@pytest.mark.parametrize("fault,ctl", [("d0_sha", "d0"), ("d0_closing", "d0"), ("d1_run", "d1"), ("d2_default", "d2"), ("d6_done", "d6")])
def test_further_planted_faults(tmp_path, fault, ctl):
    c = ev(RE.make_world(str(tmp_path / "f"), fault))
    assert c[ctl]["pass"] is False and c[ctl]["failures"]
    assert all(v["pass"] for k, v in c.items() if k != ctl), {k: v["failures"] for k, v in c.items() if k != ctl and not v["pass"]}


def test_d0_record_hash_must_match_disk(world):
    a, rec, m, wsha = world
    rec["base"]["sha256"] = "00" * 32
    assert not RE.compute_controls(a, rec, m, wsha)["d0"]["pass"]
    rec["base"]["sha256"] = rec["base"]["sha256"][:8]
    assert not RE.compute_controls(a, rec, m, wsha)["d0"]["pass"]


def test_d0_reference_without_the_reader_gap_fails(tmp_path):
    w = RE.make_world(str(tmp_path / "w"))
    import numpy as np
    a = w[0]
    p = Path(a.pull) / "I10RQ50KF5P60Hb20_s0" / "ticks" / "rule_log_mid_I10RQ50KF5P60Hb20_s0.npz"
    p.unlink()
    assert not ev(w)["d0"]["pass"]
    assert np is not None


def test_d0_has_shares_for_the_six_runs(world):
    c = RE.ctl_d0(world[1], RE.sha_file(world[0].gap_file), RE.w2_new_runs(world[0], RE.gap_ref_once(world[0]), RE.sha_file(world[0].gap_file)))
    assert sorted(c["values"]["rollout_shares"]) == sorted(RE.new_keys()) and len(RE.new_keys()) == 6
    assert all(v["share_of_rule_steps_with_a_lead"] > 0 for v in c["values"]["rollout_shares"].values())


def test_d1_clauses():
    ok = RE.ctl_d1(RE.WRAPPER_SHA, "x\nZZ CHECK-WRAPPER-V PASS\n\n", "h " + RE.WRAPPER_SHA, True, {k: RE.WRAPPER_SHA + "0" * 48 for k in RE.new_keys()})
    assert ok["pass"]
    full = {k: RE.WRAPPER_SHA + "0" * 48 for k in RE.new_keys()}
    assert not RE.ctl_d1("f" * 16, "ZZ CHECK-WRAPPER-V PASS", "h " + RE.WRAPPER_SHA, True, full)["pass"]
    assert not RE.ctl_d1(RE.WRAPPER_SHA, "ZZ CHECK-WRAPPER-V PASS", "h " + RE.WRAPPER_SHA, False, full)["pass"]
    assert not RE.ctl_d1(RE.WRAPPER_SHA, "ZZ CHECK-WRAPPER-V FAIL", "h " + RE.WRAPPER_SHA, True, full)["pass"]
    assert not RE.ctl_d1(RE.WRAPPER_SHA, "ZZ CHECK-WRAPPER-V PASS", "h other", True, full)["pass"]
    assert not RE.ctl_d1(RE.WRAPPER_SHA, "ZZ CHECK-WRAPPER-V PASS", "h " + RE.WRAPPER_SHA, True, dict(full, CONV_s0="f" * 64))["pass"]
    assert not RE.ctl_d1(RE.WRAPPER_SHA, "ZZ CHECK-WRAPPER-V PASS", "h " + RE.WRAPPER_SHA, True, dict(full, CONV_s0=None))["pass"]


# ------------------------------------------------------------------------------------------------ d2 under note N1: real-format lines
def test_d2_real_format_lines_conv_and_k0_pass():
    assert RE.d2_line_failures(RE.parse_run_line(NEW_CONV), RE.parse_run_line(REAL_CONV)) == []
    assert RE.d2_line_failures(RE.parse_run_line(NEW_K0), RE.parse_run_line(REAL_K0)) == []


def test_d2_real_format_line_k60_passes():
    assert RE.d2_line_failures(RE.parse_run_line(NEW_K60), RE.parse_run_line(REAL_K60)) == []


def test_d2_parse_of_the_real_lines():
    p = RE.parse_run_line(REAL_CONV)
    assert p == {"form": "RR", "replan": "10", "every": "10", "readout": "pinned", "kappa": "0.2", "model_seed": "0", "reader_seed": "0", "ckpt": CK}
    p = RE.parse_run_line(REAL_K60)
    assert p["preroll"] == "60" and p["exec"] == "plan" and p["nav"] == "deadreckon" and p["grid"] == "grid_p60.json" and "wrapper" not in p


def test_d2_conv_new_run_with_a_pre_roll_fails_against_a_reference_without_the_field():
    bad = RE.d2_line_failures(RE.parse_run_line(NEW_CONV.replace("preroll 0", "preroll 30")), RE.parse_run_line(REAL_CONV))
    assert any("preroll" in x for x in bad)


def test_d2_default_grid_and_nav_for_a_reference_without_them():
    bad = RE.d2_line_failures(RE.parse_run_line(NEW_K0.replace("eval_grid_windows.json", "grid_p30.json")), RE.parse_run_line(REAL_K0))
    assert any("grid" in x for x in bad)
    bad = RE.d2_line_failures(RE.parse_run_line(NEW_K0.replace("nav deadreckon", "nav logged")), RE.parse_run_line(REAL_K0))
    assert any("nav" in x for x in bad)
    new_no_grid = NEW_K0.replace(" grid eval_grid_windows.json", "")
    assert any("grid" in x for x in RE.d2_line_failures(RE.parse_run_line(new_no_grid), RE.parse_run_line(REAL_K0)))


def test_d2_a_differing_readout_fails():
    bad = RE.d2_line_failures(RE.parse_run_line(NEW_K0.replace("readout causal_g", "readout pinned")), RE.parse_run_line(REAL_K0))
    assert any("readout" in x for x in bad)


@pytest.mark.parametrize("field", ["form", "replan steps", "rule every", "readout", "kappa", "model seed", "reader seed"])
def test_d2_required_fields_must_be_in_both_lines(field):
    import re
    rx = {"form": r" form \S+", "replan steps": r" replan steps \S+", "rule every": r" rule every \S+", "readout": r" readout \S+", "kappa": r" kappa \S+", "model seed": r" model seed \S+",
          "reader seed": r" reader seed \S+"}[field]
    assert not RE.d2_line_failures(RE.parse_run_line(re.sub(rx, "", NEW_K60, count=1)), RE.parse_run_line(REAL_K60)) == []
    assert not RE.d2_line_failures(RE.parse_run_line(NEW_K60), RE.parse_run_line(re.sub(rx, "", REAL_K60, count=1))) == []


def test_d2_a_reference_line_with_fewer_fields_is_tolerated_where_n1_says():
    ref = "arm x form RR replan steps 10 rule every 5 readout causal_g kappa 0.2 model seed 0 reader seed 0"           # no ckpt, no preroll, no grid, no nav
    assert RE.d2_line_failures(RE.parse_run_line(NEW_K0), RE.parse_run_line(ref)) == []


def test_d2_checkpoint_and_numbers():
    bad = RE.d2_line_failures(RE.parse_run_line(NEW_K0.replace(CK, "cd" * 32)), RE.parse_run_line(REAL_K0))
    assert any("ckpt" in x for x in bad)
    assert RE.d2_line_failures(RE.parse_run_line(NEW_K0.replace("kappa 0.2", "kappa 0.20")), RE.parse_run_line(REAL_K0)) == []
    bad = RE.d2_line_failures(RE.parse_run_line(NEW_K0.replace("rule every 5", "rule every 10")), RE.parse_run_line(REAL_K0))
    assert any("every" in x for x in bad)


def test_d2_wrapper_hash_is_not_compared():
    assert RE.d2_line_failures(RE.parse_run_line(NEW_K0), RE.parse_run_line(REAL_K0.replace("wrapper sha256 1234", "wrapper sha256 ffff"))) == []


def test_d2_rule_share_above_zero_is_required(world):
    a, rec, m, wsha = world
    lines = {"%s_s%d" % (S, k): (RE.read_text(RE.run_text_path(a.pull, RE.run_name(S, "GEc"), k)), RE.read_text(RE.run_text_path(a.pull, RE.run_name(S, "RR"), k))) for S in RE.NEW_SETTINGS for k in RE.SEEDS}
    w2 = {k: {"share_of_rule_steps_with_a_lead": 0.5} for k in lines}
    assert RE.ctl_d2(lines, w2)["pass"]
    w2["K60_s1"] = {"share_of_rule_steps_with_a_lead": 0.0}
    assert not RE.ctl_d2(lines, w2)["pass"]
    w2["K60_s1"] = {"share_of_rule_steps_with_a_lead": 0.5}
    lines["K60_s1"] = ("", lines["K60_s1"][1])
    assert not RE.ctl_d2(lines, w2)["pass"]


# ------------------------------------------------------------------------------------------------ d3 to d6
def test_d3_each_value():
    base = {k: dict(v) for k, v in RE.EXPECTED_K30.items()}
    assert RE.ctl_d3(base)["pass"]
    for arm in base:
        for c in "CFU":
            x = {k: dict(v) for k, v in base.items()}
            x[arm][c] += 0.00011
            assert not RE.ctl_d3(x)["pass"], (arm, c)
            x[arm][c] -= 0.00002
            assert RE.ctl_d3(x)["pass"], (arm, c)
    assert not RE.ctl_d3(None)["pass"] and not RE.ctl_d3({"GEc": {}})["pass"]


def test_d3_is_read_by_the_function_of_the_new_settings():
    src = (HERE / "read_candset10k.py").read_text(encoding="utf-8")
    assert "for S in SETTINGS:" in src and "measure_setting(a, E, S, keys, ctx_cache, log)" in src and src.count("measure_setting(") == 2
    assert "K30" in RE.SETTINGS and "K30" not in RE.NEW_SETTINGS


def test_d4_times_and_text():
    q, r = "..\n12 passed in 3s\n", "..\n30 passed, 1 warning in 4s\n"
    assert RE.ctl_d4(q, r, 1000, 1010, "1100", 1200)["pass"]
    assert RE.ctl_d4(q, r, 1100, 1200, "1100", 1200)["pass"]                                  # not later than: equal is allowed
    assert not RE.ctl_d4(q, r, 1101, 1010, "1100", 1200)["pass"]
    assert not RE.ctl_d4(q, r, 1000, 1201, "1100", 1200)["pass"]
    assert not RE.ctl_d4(q, r, 1000, 1010, None, 1200)["pass"]
    assert not RE.ctl_d4(q, r, 1000, 1010, "1100", "")["pass"]
    assert RE.ctl_d4(q, r, "2026-11-22T10:00:00Z", "2026-11-22T10:01:00Z", "1795345027", "2026-11-22T11:00:00Z")["pass"]
    assert not RE.ctl_d4("..\n0 passed\n", r, 1000, 1010, "1100", 1200)["pass"]
    assert not RE.ctl_d4("", r, 1000, 1010, "1100", 1200)["pass"]
    assert not RE.ctl_d4("..\n1 failed, 11 passed in 3s\n", r, 1000, 1010, "1100", 1200)["pass"]
    assert not RE.ctl_d4(q, "..\n29 passed, 1 error in 3s\n", 1000, 1010, "1100", 1200)["pass"]


def test_d5_each_part():
    ok = {S: {"grid_all_pass": True, "strategic_a": True, "strategic_b": True} for S in RE.SETTINGS}
    assert RE.ctl_d5(ok)["pass"]
    for S in RE.SETTINGS:
        for k in ("grid_all_pass", "strategic_a", "strategic_b"):
            x = {s: dict(v) for s, v in ok.items()}
            x[S][k] = False
            assert not RE.ctl_d5(x)["pass"], (S, k)
    x = dict(ok)
    del x["K60"]
    assert not RE.ctl_d5(x)["pass"]


def test_d6_counts_and_done():
    keys = RE.new_keys()
    done = {k: True for k in keys}
    nd = {k: 141 for k in keys}
    cl = {S: 141 for S in RE.NEW_SETTINGS}
    assert RE.ctl_d6(done, nd, cl)["pass"]
    assert not RE.ctl_d6(dict(done, K0_s1=False), nd, cl)["pass"]
    assert not RE.ctl_d6(done, dict(nd, K60_s0=140), cl)["pass"]
    assert not RE.ctl_d6(done, nd, dict(cl, K60=None))["pass"]
    assert RE.ctl_d6(done, nd, cl, "ZZ CANDSET10K DONE 2026-11-22T12:00:00Z all steps done\n")["pass"]
    assert not RE.ctl_d6(done, nd, cl, "ZZ CANDSET10K DONE\n")["pass"]


# ------------------------------------------------------------------------------------------------ names (card section 4)
def iv(**kw):
    return {(S, nm.replace("_", "-").replace("GEc-VS-", "GEc-VS-"), m): v for (S, nm, m), v in kw.items()}


def payload(rr_c=0.1, ivs=None):
    return RE.mk_payload(rr_c, ivs)


@pytest.mark.parametrize("c,word", [(LOW, "CONTACT-CUT-HOLDS"), (HIGH, "CONTACT-RAISED"), (ZERO, "CONTACT-NOT-SEPARATED"), ([-0.05, 0.0], "CONTACT-NOT-SEPARATED"), ([0.0, 0.05], "CONTACT-NOT-SEPARATED")])
def test_contact_branches(c, word):
    nm = RE.all_names(payload(0.1, {("K0", "GEc-VS-RR", "C"): c}))
    assert nm["K0_CONTACT"] == "%s-K0-%s" % (PRE, word)


def test_no_contact_to_cut_threshold():
    assert RE.all_names(payload(0.0099))["CONV_CONTACT"] == PRE + "-CONV-NO-CONTACT-TO-CUT"
    assert RE.all_names(payload(0.01))["CONV_CONTACT"] == PRE + "-CONV-CONTACT-NOT-SEPARATED"
    # below the floor the interval is not consulted
    assert RE.all_names(payload(0.005, {("CONV", "GEc-VS-RR", "C"): HIGH}))["CONV_CONTACT"] == PRE + "-CONV-NO-CONTACT-TO-CUT"


@pytest.mark.parametrize("u,word", [(LOW, "UNSAFE-CUT-HOLDS"), (HIGH, "UNSAFE-RAISED"), (ZERO, "UNSAFE-NOT-SEPARATED")])
def test_unsafe_branches(u, word):
    assert RE.all_names(payload(0.1, {("K60", "GEc-VS-RR", "U"): u}))["K60_UNSAFE"] == "%s-K60-%s" % (PRE, word)


@pytest.mark.parametrize("f,word", [(HIGH, "FALLS-BEHIND-MORE"), (LOW, "NO-SEPARATED-PRICE"), (ZERO, "NO-SEPARATED-PRICE"), ([0.0, 0.05], "NO-SEPARATED-PRICE")])
def test_price_branches(f, word):
    assert RE.all_names(payload(0.1, {("CONV", "GEc-VS-RLc", "F"): f}))["CONV_PRICE"] == "%s-CONV-%s" % (PRE, word)


@pytest.mark.parametrize("c,word", [(HIGH, "CONTACT-ABOVE-LABEL-FORM"), (LOW, "CONTACT-AT-LABEL-FORM"), (ZERO, "CONTACT-AT-LABEL-FORM")])
def test_label_form_contact_branches(c, word):
    assert RE.all_names(payload(0.1, {("K0", "GEc-VS-RLc", "C"): c}))["K0_VS_LABEL"] == "%s-K0-%s" % (PRE, word)


def test_over_all_names():
    hold = {(S, "GEc-VS-RR", "C"): LOW for S in RE.NEW_SETTINGS}
    assert RE.all_names(payload(0.1, hold))["OVERALL"] == PRE + "-CUT-HOLDS-AT-EVERY-SETTING-WITH-CONTACT"
    # a setting with no contact is not counted; the others all hold
    assert RE.all_names(payload({"CONV": 0.001, "K0": 0.1, "K60": 0.1}, hold))["OVERALL"] == PRE + "-CUT-HOLDS-AT-EVERY-SETTING-WITH-CONTACT"
    assert RE.all_names(payload({"CONV": 0.001, "K0": 0.1, "K60": 0.1}, {("K0", "GEc-VS-RR", "C"): LOW, ("K60", "GEc-VS-RR", "C"): LOW}))["OVERALL"].endswith("CUT-HOLDS-AT-EVERY-SETTING-WITH-CONTACT")
    # no setting with contact: no setting that holds
    assert RE.all_names(payload(0.001, hold))["OVERALL"] == PRE + "-CUT-NOT-AT-EVERY-SETTING"
    # one not separated
    assert RE.all_names(payload(0.1, {**hold, **{("K60", "GEc-VS-RR", "C"): ZERO}}))["OVERALL"] == PRE + "-CUT-NOT-AT-EVERY-SETTING"
    # one raised wins over holds
    assert RE.all_names(payload(0.1, {**hold, **{("K60", "GEc-VS-RR", "C"): HIGH}}))["OVERALL"] == PRE + "-CUT-RAISED-AT-A-SETTING"
    # raised at a setting with no contact is not read (the interval is not consulted)
    assert RE.all_names(payload({"CONV": 0.001, "K0": 0.1, "K60": 0.1}, {**hold, **{("CONV", "GEc-VS-RR", "C"): HIGH}}))["OVERALL"] == PRE + "-CUT-HOLDS-AT-EVERY-SETTING-WITH-CONTACT"
    assert RE.all_names(payload(0.1))["OVERALL"] == PRE + "-CUT-NOT-AT-EVERY-SETTING"


def test_k30_has_no_name():
    nm = RE.all_names(payload(0.1))
    assert not any("K30" in v for v in nm.values())
    assert len(nm) == 3 * 4 + 1


def test_two_derivations_agree_on_random_intervals():
    import random
    rnd = random.Random(5)
    pool = [LOW, HIGH, ZERO, [-0.02, 0.0], [0.0, 0.02]]
    for _ in range(300):
        ivs = {(S, nm, m): rnd.choice(pool) for S in RE.NEW_SETTINGS for nm in ("GEc-VS-RR", "GEc-VS-RLc") for m in "CUF"}
        rr = {S: rnd.choice([0.0, 0.0099, 0.01, 0.2]) for S in RE.NEW_SETTINGS}
        p = payload(rr, ivs)
        assert RE.names_of(p) == RE.derive_all(p)


def test_disagreement_is_refused(monkeypatch):
    monkeypatch.setattr(RE, "derive_all", lambda p: dict(RE.names_of(p), OVERALL="x"))
    with pytest.raises(SystemExit):
        RE.all_names(payload(0.1))


# ------------------------------------------------------------------------------------------------ stages
def good_controls(tmp_path):
    p = payload(0.1, {("K0", "GEc-VS-RR", "C"): LOW})
    for S in RE.SETTINGS:
        d = p["settings"][S]
        d["levels"]["GEc"] = {"C": 0.0, "U": 0.1, "F": 0.5}
        d["levels_per_seed"] = {a: {c: [0.0, 0.0] for c in "CUF"} for a in RE.ARMS}
        d["counts"] = {"n_windows": 10, "n_lead_windows": 5, "n_episodes_with_lead": 3}
        d["families"], d["strategic"] = {}, {"pairs": {}}
        d["strategic_reason"] = "x"
        for nm in RE.PAIRS:
            d["pairs"].setdefault(nm, {m: {"interval": ZERO, "pooled": 0.0, "per_seed": [0.0, 0.0]} for m in "CUF"})
    for S in RE.NEW_SETTINGS:
        for nm in RE.PAIRS:
            for m in "CUF":
                p["settings"][S]["pairs"][nm][m].update({"pooled": 0.0, "per_seed": [0.0, 0.0]})
    p["settings"]["K30"]["pairs"] = {nm: {m: {"interval": ZERO, "pooled": 0.0, "per_seed": [0.0, 0.0]} for m in "CUF"} for nm in RE.PAIRS}
    ctl = {k: {"pass": True, "failures": []} for k in RE.CONTROLS}
    res = {"card": RE.CARD, "all_pass": True, "controls": ctl, "payload": dict(p, control_values={"d0": {"rollout_shares": {}}})}
    f = tmp_path / "c.json"
    f.write_text(json.dumps(res), encoding="utf-8")
    return f, res


def test_names_stage_refuses_each_missing_or_failed_control(tmp_path, capsys):
    f, res = good_controls(tmp_path)
    for k in RE.CONTROLS:
        bad = json.loads(json.dumps(res))
        bad["controls"][k]["pass"] = False
        g = tmp_path / ("b_%s.json" % k)
        g.write_text(json.dumps(bad), encoding="utf-8")
        assert RE.main(["--stage", "names", "--controls", str(g), "--out", str(tmp_path / ("o_%s.json" % k))]) == 3
        assert not (tmp_path / ("o_%s.json" % k)).exists()
    bad = json.loads(json.dumps(res))
    del bad["controls"]["d6"]
    g = tmp_path / "m.json"
    g.write_text(json.dumps(bad), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(g), "--out", str(tmp_path / "o_m.json")]) == 3
    bad = json.loads(json.dumps(res))
    bad["all_pass"] = False
    bad["withheld"] = bad.pop("payload")
    g = tmp_path / "w.json"
    g.write_text(json.dumps(bad), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(g), "--out", str(tmp_path / "o_w.json")]) == 3
    out = capsys.readouterr().out
    assert "ZZ READ-CANDSET10K names refused" in out


def test_names_stage_writes_result_and_log(tmp_path, capsys):
    f, res = good_controls(tmp_path)
    o = tmp_path / "r.json"
    assert RE.main(["--stage", "names", "--controls", str(f), "--out", str(o)]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "ZZ READ-CANDSET10K names done"
    assert out[0].startswith("TIER T1-kin/hb0.2 (kinematic rollout on the model's own actions")
    assert "no pre-roll" in out[0] and "pre-roll of 60 steps (12 s)" in out[0]
    assert any(x == PRE + "-K0-CONTACT-CUT-HOLDS" for x in out)
    assert any(x.startswith("setting K60 (pre-roll of 60 steps (12 s)") for x in out) and any(x.startswith("setting CONV (no pre-roll") for x in out)
    assert any("windows with a lead 5" in x for x in out)
    r = json.loads(o.read_text(encoding="utf-8"))
    assert r["controls_sha256"] == RE.sha_file(str(f)) and "OVERALL" in r["names"]
    assert (tmp_path / "r_names.log").exists()
    assert RE.main(["--stage", "names", "--controls", str(f), "--out", str(tmp_path / "r.json")]) == 2          # refuses to write into an existing out


def test_names_stage_refuses_a_controls_file_of_another_card(tmp_path, capsys):
    f, res = good_controls(tmp_path)
    res["card"] = "PREREG_V8F_PROOF_ADD111"
    g = tmp_path / "x.json"
    g.write_text(json.dumps(res), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(g), "--out", str(tmp_path / "ox.json")]) == 3


def test_assemble_controls_withholds_on_fail():
    ctl = {k: {"pass": True, "failures": []} for k in RE.CONTROLS}
    assert "payload" in RE.assemble_controls(ctl, {"x": 1}, {}) and "withheld" not in RE.assemble_controls(ctl, {"x": 1}, {})
    ctl["d3"] = {"pass": False, "failures": ["x"]}
    r = RE.assemble_controls(ctl, {"x": 1}, {})
    assert r["all_pass"] is False and "withheld" in r and "payload" not in r


def test_stdout_of_controls_has_only_the_allowed_lines(tmp_path, monkeypatch, capsys):
    w = RE.make_world(str(tmp_path / "w"))
    a0 = w[0]
    monkeypatch.setattr(RE, "run_controls", lambda a: RE.assemble_controls(RE.compute_controls(a0, w[1], w[2], w[3]), {"settings": w[2]["settings"]}, {}))
    out = tmp_path / "C.json"
    assert RE.main(["--stage", "controls", "--out", str(out)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ["d%d PASS" % i for i in range(7)] + ["ZZ READ-CANDSET10K controls all_pass True"]
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["all_pass"] is True and "payload" in res and "withheld" not in res and sorted(res["controls"]) == sorted(RE.CONTROLS)


def test_stdout_of_a_failed_controls_stage_withholds(tmp_path, monkeypatch, capsys):
    w = RE.make_world(str(tmp_path / "w"), "d3")
    monkeypatch.setattr(RE, "run_controls", lambda a: RE.assemble_controls(RE.compute_controls(w[0], w[1], w[2], w[3]), {"settings": w[2]["settings"]}, {}))
    out = tmp_path / "C.json"
    assert RE.main(["--stage", "controls", "--out", str(out)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert "d3 FAIL" in lines and lines[-1] == "ZZ READ-CANDSET10K controls all_pass False" and len(lines) == 8
    res = json.loads(out.read_text(encoding="utf-8"))
    assert "withheld" in res and "payload" not in res


def test_settings_table_matches_the_card():
    S = RE.SPEC
    assert S["CONV"]["runs"] == {"GEc": "RQ50Hb20gGEc", "RR": "RQ50Hb20", "RLc": "RLcHb20"} and S["CONV"]["P"] == 0 and S["CONV"]["readout"] == "pinned"
    assert S["K0"]["runs"] == {"GEc": "RQ50KF5Hb20gGEc", "RR": "RQ50KF5Hb20", "RLc": "RLcKF5Hb20"} and S["K0"]["P"] == 0
    assert S["K60"]["runs"] == {"GEc": "RQ50KF5P60Hb20gGEc", "RR": "RQ50KF5P60Hb20", "RLc": "RLcKF5P60Hb20"} and S["K60"]["P"] == 60 and S["K60"]["grid_file"] == "grid_p60.json"
    assert S["K30"]["runs"] == {"GEc": "RQ50KF5P30Hb20gGEc", "RR": "RQ50KF5P30Hb20", "RLc": "RLcKF5P30Hb20"} and S["K30"]["P"] == 30
    assert RE.preroll_text(0) == "no pre-roll" and RE.preroll_text(60) == "pre-roll of 60 steps (12 s)"
    assert RE.EXPECTED_K30 == {"GEc": {"C": 0.0062, "F": 0.5154, "U": 0.0839}, "RR": {"C": 0.0954, "F": 0.4954, "U": 0.1339}, "RLc": {"C": 0.0, "F": 0.4908, "U": 0.0714}}
    assert RE.MD.WRAPPER_SHA == RE.WRAPPER_SHA == "28bc31f21ecb4ccb"


def test_model_is_imported_not_edited():
    src = (HERE / "read_candset10k.py").read_text(encoding="utf-8")
    assert "spec_from_file_location" in src and "def w2_one" not in src and "def lead_multi" not in src and "def read_grid" not in src
    for fn in ("rows_of", "load_logs", "lead_multi", "w2_one", "gapfile_ref", "meta_of", "run_file_sha", "ctl_g8"):
        assert hasattr(RE.MD, fn)
        assert "MD." + fn in src
    for fn in ("read_grid", "strategic_grid", "ucf", "reading", "cells_of", "family_of", "grid_keys", "engine"):
        assert "R." + fn in src


# ------------------------------------------------------------------------------------------------ the start-parse line of the driver
def start_line():
    ls = [x for x in (HERE / "podread_candset10k.sh").read_text(encoding="utf-8").splitlines() if "QS=$(" in x]
    assert len(ls) == 1, ls
    return ls[0].strip()


def run_start_parse(tmp_path, first_line):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("no bash")
    with open(str(tmp_path / "CANDSET10K_START.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(first_line + "\nsecond line\n")
    with open(str(tmp_path / "t.sh"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write('set -u\nGE=.\nstop() { echo "STOP $*"; exit 3; }\n' + start_line() + '\necho "QS=$QS"\n')
    return subprocess.run([bash, "t.sh"], cwd=str(tmp_path), capture_output=True, encoding="utf-8")


def test_start_parse_line_on_the_marker_as_the_pod_wrote_it(tmp_path):
    r = run_start_parse(tmp_path, "1795345027 2026-11-22T10:57:07Z pid 777199")
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "QS=1795345027", (r.stdout, r.stderr)


def test_start_parse_line_stops_on_an_iso_first_token(tmp_path):
    r = run_start_parse(tmp_path, "2026-11-22T10:57:07Z 1795345027 pid 777199")
    assert r.returncode == 3 and "STOP START-MARKER-UNREADABLE" in r.stdout and "QS=" not in r.stdout


def test_start_parse_line_stops_on_an_empty_marker(tmp_path):
    r = run_start_parse(tmp_path, "")
    assert r.returncode == 3 and "STOP START-MARKER-UNREADABLE" in r.stdout


def test_queue_start_marker_format_of_the_queue_script():
    q = (HERE.parent / "queue" / "pod_candset10k_queue.sh").read_text(encoding="utf-8")
    assert 'echo "$(date +%s) $(ts) pid $$" > "$GP/CANDSET10K_START.txt"' in q
    assert 'echo "ZZ CANDSET10K DONE $(ts) all steps done" > "$GP/CANDSET10K_DONE.txt"' in q


# ------------------------------------------------------------------------------------------------ files: wording, paths, scripts
def kit_files():
    return [p for p in HERE.iterdir() if p.is_file() and p.suffix in (".py", ".sh", ".md")]


def test_wording_scan_and_path_length():
    names = {p.name for p in kit_files()}
    assert {"read_candset10k.py", "podread_candset10k.sh", "pull_candset.sh", "push_candset_read.sh", "test_candset_read.py", "CANDSET_READ_DEFS.md"} <= names
    for p in kit_files():
        low = p.read_text(encoding="utf-8").lower()
        for w in BANNED:
            assert w not in low, (p.name, w)
        rel = "/f/TanitAD-scratch/v8proof/candset/read/" + p.name
        assert len(rel) <= 200 and p.stem.lower() not in ("aux", "con", "prn", "nul")


def test_no_file_above_5_mb_and_no_lead_test_output_asserted():
    for p in HERE.rglob("*"):
        if p.is_file():
            assert p.stat().st_size <= 5 * 1024 * 1024, p


def test_shell_files_are_lf_only_and_parse():
    bash = shutil.which("bash")
    for nm in ("podread_candset10k.sh", "pull_candset.sh", "push_candset_read.sh"):
        assert b"\r" not in (HERE / nm).read_bytes(), nm
        if bash:
            r = subprocess.run([bash, "-n", str(HERE / nm)], capture_output=True, encoding="utf-8")
            assert r.returncode == 0, (nm, r.stderr)


def test_driver_rules():
    pod = (HERE / "podread_candset10k.sh").read_text(encoding="utf-8")
    for must in ("CANDSET10K_DONE.txt", "CANDSET10K_STOP.txt", "CANDSET10K_START.txt", "stat -c %Y", "--selftest", "-ge 20", "ALREADY-READ", "CUDA_VISIBLE_DEVICES=-1", "GE=$V/candset", "K=$GE/kit_read",
                 "OUT=$GE/read", "LT=$GE/lead_tests", "lead_pytest_queue.out", "lead_pytest_read.out", "CANDSET10K_CONTROLS.json", "CANDSET10K_RESULT.json", "read_names.out", "ZZ PODREAD-CANDSET10K",
                 "^d[0-6] (PASS|FAIL)|^ZZ READ-CANDSET10K", "--queue-start \"$QS\"", "--controls-start \"$CSTART\"", "VGAPLE.npz", "GAPERR_GAPFILES_RECORD.json", "read_candset10k.py",
                 "v8f_t1_eval_reader_rule_rzgfhqetjv.py", "I10RQ50Hb20gGEc", "I10RQ50KF5Hb20gGEc", "I10RQ50KF5P60Hb20gGEc", "I10RQ50KF5P30Hb20gGEc", "I10R0Hb20", "grid_p60.json", "eval_grid_windows.json",
                 "CS=$V/cropscale", "MODEL=$CS/kit_read/read_cropscale10k.py", "$(ts)"):
        assert must in pod, must
    for stale in ("GAPERR10K", "FALLATTR", "b[0-5]"):
        assert stale not in pod, stale
    for banned_cmd in ("rm ", "mv ", "kill", "pkill"):
        assert banned_cmd not in pod, banned_cmd
    pull = (HERE / "pull_candset.sh").read_text(encoding="utf-8")
    assert "sed '/hf_/d'" in pull and "*.npz|*.pt) continue" in pull and "pod_add113" in pull and "sha256sum" in pull and "/workspace/v8proof/candset" in pull and "REFUSED" in pull
    assert "add111" not in pull
    push = (HERE / "push_candset_read.sh").read_text(encoding="utf-8")
    assert "set -C" in push and "sha256sum" in push and "kit_read" in push and "read_candset10k.py podread_candset10k.sh" in push and "/workspace/v8proof/candset" in push
