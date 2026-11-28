"""test_llr_read.py -- the reader of arm LLR (code/llr_read.py): the A2 card's outcomes, the rules, the frozen-file controls, the reused (banked CLR base) arrays, and the three known answers on EXISTING arrays.
Every test is a mutation target (run_mutations_llr.py).  Env override: LLR_READ_PATH.  Run: CUDA_VISIBLE_DEVICES=-1 python -I -m pytest tests/test_llr_read.py -q --basetemp F:/TanitAD-scratch/motion_c/llr_read/pt/bt"""
import importlib.util
import json
import os
import re
import shutil

import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
SRC = f"{PK}/code/llr_read.py"


def _load(path):
    spec = importlib.util.spec_from_file_location("llr_read_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load(os.environ.get("LLR_READ_PATH", SRC))


@pytest.fixture(scope="module", autouse=True)
def heavy():
    C.init()


def cell(better, share, xy=-0.03, yp=0.05):
    return {"better": better, "xy_delta": xy, "yp_delta": yp, "yp_separated_worse": False, "share_closed": share}


def test_P_is_per_seed_and_needs_the_share_bar():
    ok = C.rule_P({"s0": cell(True, 0.6), "s1": cell(True, 0.5)})
    assert ok["PASS"] and ok["per_seed"]["s1"]["holds"]                               # 0.5 is the bar (>=)
    assert not C.rule_P({"s0": cell(True, 0.9), "s1": cell(True, 0.4999)})["PASS"]
    assert not C.rule_P({"s0": cell(True, 0.9), "s1": cell(False, 0.9)})["PASS"]
    assert not C.rule_P({"s0": cell(True, 0.9), "s1": cell(True, None)})["PASS"]    # no excess to close: fails
    assert C.SHARE_BAR == 0.5 and C.LEAK_FRACTION == 0.5


def test_leak_borne_rule():
    lg = {"s0": cell(True, 0.6), "s1": cell(True, 0.6)}
    assert not C.rule_leak(lg, {"s0": cell(True, 0.4), "s1": cell(True, 0.4)})["LEAK_BORNE"]            # drop 0.2 <= half of 0.6
    assert C.rule_leak(lg, {"s0": cell(True, 0.4), "s1": cell(True, 0.29)})["LEAK_BORNE"]               # drop 0.31 > 0.3 at one seed
    assert not C.rule_leak({"s0": cell(True, 0.0), "s1": cell(True, -0.1)}, {"s0": cell(True, -0.5), "s1": cell(True, 0.0)})["LEAK_BORNE"]    # a logged share that is not positive cannot be leak-borne
    assert C.rule_leak({"s0": cell(True, 0.5), "s1": cell(True, 0.5)}, {"s0": cell(True, None), "s1": cell(True, 0.5)})["LEAK_BORNE"]      # an undefined reindexed share counts 0


def test_decision_table_is_the_a2_cards_outcomes_for_llr():
    d = C.decide
    assert d(True, True, True, 0, False)["label"] == "D-HOLDS" and d(True, True, True, 0, False)["counts_as_result"]
    assert d(True, True, False, 0, False)["label"] == "P-HOLDS-G-OR-D-FAILS" and "G holds, D fails" in d(True, True, False, 0, False)["notes"]
    assert d(True, False, False, 0, False)["label"] == "P-HOLDS-G-OR-D-FAILS" and "G fails" in d(True, False, False, 0, False)["notes"]
    f = d(False, True, False, 0, False)
    assert f["label"] == "P-FAILS-FAMILY-EXHAUSTED" and "exhausted" in f["next"] and "T2 re-perception" in f["next"] and "A40" in f["next"] and f["counts_as_result"]
    assert "G fails too" in d(False, False, False, 0, False)["notes"][0]
    assert set(C.NEXT_ARM) == {"D-HOLDS", "P-HOLDS-G-OR-D-FAILS", "P-FAILS-FAMILY-EXHAUSTED"}
    assert "LLR-W2" in C.NEXT_ARM["P-HOLDS-G-OR-D-FAILS"] and "does not say whether" in C.NEXT_ARM["P-HOLDS-G-OR-D-FAILS"]          # the card's W / 2 is ambiguous for LLR: named, not guessed


def test_attribution_is_reported_and_never_branches_the_outcome():
    for n in (0, 1, 2):
        o = C.decide(False, True, False, n, False)
        assert o["label"] == "P-FAILS-FAMILY-EXHAUSTED" and any("separated-better at %d of 2" % n in x for x in o["notes"])
    a = C.rule_attr({"s0": cell(True, 0.1), "s1": cell(False, 0.0)}, {"s0": cell(False, 0.0), "s1": cell(False, 0.0)})
    assert a["SEPARATED_BETTER_AT_ANY_SEED"] and a["n_seeds_separated_better_reindexed"] == 1 and a["n_seeds_separated_better_logged_beside"] == 0 and a["deciding_scene"] == "reindexed"
    assert not C.rule_attr({"s0": cell(False, 0.1), "s1": cell(False, 0.0)}, {"s0": cell(True, 0.5), "s1": cell(True, 0.5)})["SEPARATED_BETTER_AT_ANY_SEED"]   # the logged scene never decides


def test_leak_borne_overrides_the_label_and_counts_for_nothing():
    for P, D in ((False, False), (True, False), (True, True)):
        o = C.decide(P, True, D, 1, True)
        assert o["label"] == "LEAK-BORNE" and not o["counts_as_result"] and o["underlying"] != "LEAK-BORNE" and "camera-clock" in o["next"]


def test_D_needs_read_a_at_both_seeds_and_P_and_G():
    v = lambda dom: {"DOMINANT": dom, "DOMINANT_three_families_strategic_paused": dom, "D1": dom, "D2": dom, "D3": dom, "D3_by_family": {}, "separated_worse_rows": [], "failing_terms": []}
    both = {"s0": v(True), "s1": v(True)}
    assert C.rule_D(both, True, True)["DOMINANT_COUNTED"]
    assert not C.rule_D(both, False, True)["DOMINANT_COUNTED"] and not C.rule_D(both, True, False)["DOMINANT_COUNTED"]
    assert not C.rule_D({"s0": v(True), "s1": v(False)}, True, True)["DOMINANT_COUNTED"]


def test_roles_runs_and_the_banked_base_are_what_the_card_says():
    assert set(C.NEW_ROLES) == {"LLR", "LLRC", "FZLLR", "RILLR", "RILLRC"} and set(C.RUN_OF.values()) == {"LLR"} and len(C.RUN_OF) == 5          # five roles, one run: NO attribution-base rollout
    assert C.NEW_ROLES["RILLR"] == ("reindexed", "off") and C.NEW_ROLES["LLR"] == ("logged", "off") and C.NEW_ROLES["FZLLR"] == ("hold_now", "off") and C.NEW_ROLES["RILLRC"] == ("reindexed", "kbox")
    assert set(C.REUSED) == {"C0", "C0C", "FZC0", "C0REP", "RIC0", "RIC0C", "CLR", "RICLR"} and C.REUSED["CLR"].endswith("clr_read/raw/arrays") and C.REUSED["RICLR"].endswith("clr_read/raw/arrays")
    assert len(C.REUSED_SHA) == 16 and len(C.CLR_ARR_SHA) == 10 and all(C.CLR_ARR_SHA[k] == C.REUSED_SHA[k] for k in ("CLR_s0", "CLR_s1", "RICLR_s0", "RICLR_s1"))
    src = open(SRC, encoding="utf-8").read()
    assert "RAPRP01" not in src.replace("RAPRP01", "RAPRP01") or True
    assert not re.search(r"RIRAPRP01|\"RAPRP01\"", src), "the RAPR-P01 attribution base belongs to the CLR read, not to this one"


def test_expected_flags_are_the_cards_scene_modes():
    assert C.expected_flags("RILLR") == "--smooth sg72 --scene reindexed" and C.expected_flags("RILLRC") == "--smooth sg72 --scene reindexed --feasible kbox"
    assert C.expected_flags("LLR") == "--smooth sg72 --scene logged" and C.expected_flags("LLRC").endswith("--feasible kbox") and C.expected_flags("FZLLR") == "--smooth sg72 --scene hold_now"


def test_reused_arrays_are_the_pinned_files_including_the_banked_clr_base():
    for s in (0, 1):
        _, ctl = C.reused_arrays(s)
        assert len(ctl) == 8 and all(v["PASS"] for v in ctl.values()), [k for k, v in ctl.items() if not v["PASS"]]
        assert {"CLR_s%d" % s, "RICLR_s%d" % s} <= set(ctl)
    assert C.sha256_file(f"{C.CLR_ARR}/RICLR_s0.npz") == C.REUSED_SHA["RICLR_s0"]


def test_a_changed_pin_fails_the_reused_arrays_control(monkeypatch):
    bad = dict(C.REUSED_SHA)
    bad["RICLR_s1"] = "0" * 64
    monkeypatch.setattr(C, "REUSED_SHA", bad)
    _, ctl = C.reused_arrays(1)
    assert not ctl["RICLR_s1"]["PASS"] and ctl["CLR_s1"]["PASS"]


def test_frozen_files_controls(tmp_path, monkeypatch):
    d = tmp_path / "a2"
    shutil.copytree(C.A2_DIR, d)
    monkeypatch.setattr(C, "A2_DIR", str(d))
    assert C.frozen_sha_of("A2_RULING_1.md") == C.sha256_file(str(d / "A2_RULING_1.md")) and C.frozen_sha_of("nope") is None
    assert C.parse_frozen_llr() == {"WIRING": "token", "W_LEAD": "3.362"}
    ctl = C.run_controls(str(tmp_path / "none"), str(tmp_path))
    assert ctl["frozen_A2_RULING_1.md"]["PASS"] and ctl["frozen_FROZEN_LLR.txt"]["PASS"] and ctl["frozen_llr_values"]["PASS"] and ctl["card_a2"]["PASS"]
    (d / "A2_RULING_1.md").write_text("edited\n", encoding="utf-8")
    (d / "FROZEN_LLR.txt").write_text("WIRING=add\nW_LEAD=3.362\n", encoding="utf-8")
    ctl = C.run_controls(str(tmp_path / "none"), str(tmp_path))
    assert not ctl["frozen_A2_RULING_1.md"]["PASS"] and not ctl["frozen_FROZEN_LLR.txt"]["PASS"] and not ctl["frozen_llr_values"]["PASS"]


def test_production_without_llr_rollouts_refuses_and_names_the_controls(tmp_path):
    empty = tmp_path / "out"
    empty.mkdir()
    (tmp_path / "raw").mkdir()
    cp = json.load(open(C.CLR_PRECHECK_JSON, encoding="utf-8"))["seeds"]
    json.dump({"PASS": True, "w_lead": "3.362", "seeds": {k: {"ckpt_sha256": "a" * 64, "clr_ckpt_sha256": cp[k]["ckpt_sha256"]} for k in ("s0", "s1")}}, open(tmp_path / "raw" / "LLR_PRECHECK.json", "w", encoding="utf-8"))
    rc = C.main(["--mode", "production", "--out-root", str(empty), "--raw", str(tmp_path / "raw"), "--results-md", str(tmp_path / "R.md")])
    assert rc == 3
    res = json.load(open(tmp_path / "raw" / "LLR_READ.json", encoding="utf-8"))
    assert "refused" in res and all(n in res["refused"] for n in ("rollout_RILLR_s0", "rollout_FZLLR_s1", "rollout_LLRC_s1", "LLR_PRECHECK", "one_checkpoint_per_seed"))
    assert not os.path.exists(tmp_path / "raw" / "arrays") and not os.path.exists(tmp_path / "R.md")                      # nothing was extracted, no RESULTS.md


def test_ka_new2_control_needs_the_lead_head_blobs_and_all_three_deciding_cases(tmp_path, monkeypatch):
    shutil.copytree(f"{PK}/raw", tmp_path / "pk" / "raw", ignore=shutil.ignore_patterns("arrays"))
    monkeypatch.setattr(C, "PK", str(tmp_path / "pk"))
    out = tmp_path / "o"
    out.mkdir()
    assert C.run_controls(str(out), str(tmp_path))["KA_NEW2"]["PASS"]
    k = json.load(open(tmp_path / "pk" / "raw" / "KA_NEW2.json", encoding="utf-8"))
    for mutate in (lambda d: d["cases"]["C"].__setitem__("PASS", False), lambda d: d.__setitem__("model_sha256", "0" * 64), lambda d: d.__setitem__("PASS", False), lambda d: d.__setitem__("tool_sha256", "1" * 64)):
        d = json.loads(json.dumps(k))
        mutate(d)
        json.dump(d, open(tmp_path / "pk" / "raw" / "KA_NEW2.json", "w", encoding="utf-8"))
        assert not C.run_controls(str(out), str(tmp_path))["KA_NEW2"]["PASS"]


@pytest.mark.parametrize("mode", ["ka-null", "ka-ri", "ka-clr"])
def test_reader_known_answers(tmp_path, mode):
    assert C.main(["--mode", mode, "--raw", str(tmp_path)]) == 0
    res = json.load(open(tmp_path / f"KA_READER_{mode}.json", encoding="utf-8"))
    assert res["KA_PASS"] is True and all(res["ka_checks"].values())
    assert res["outcome"]["label"] == "P-FAILS-FAMILY-EXHAUSTED"       # null arm / null reindexed slots / CLR's own arrays (CLR failed P): nothing cleared the bar
    if mode in ("ka-ri", "ka-clr"):
        assert len(res["ka_checks"]) >= 30
    if mode == "ka-clr":                                                # CLR's five arrays fed as LLR's against the REAL banked base: every CLR contrast back, every attribution row exactly 0
        assert res["ka_checks"]["rows_compared_at_least_400"] and any(k.endswith("equals_CLR_READ[RICLR_minus_RIC0]") for k in res["ka_checks"])
        assert all(r["delta"] == 0.0 and not r["separated"] for k in ("s0", "s1") for r in res["seeds"][k]["RILLR_minus_RICLR"].values())
