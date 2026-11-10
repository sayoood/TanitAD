"""Tests for read_s1zsm.py (CPU only; one process at a time).

    CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_read_s1zsm.py --basetemp F:/tanitad_scratch/s1zsmread/pt/run

(a) CONTROL on the real files: S1Z's own pass dir as the arm equals the banked s1z_read.json leaf for leaf, and M3 (S1Z - S1Z) is exactly 0.0 with CI [0, 0];
(b) RED: a card with one byte changed is refused; (c) RED: an arm whose proposals are not bit-identical fails G0 (as read_s1z does); (d) the SM outcome map from synthetic inputs, overlaps included.
Nothing here writes selectors/ledger_K.json (a temp copy is used for the ledger test, and the real file's sha256 is asserted unchanged).
"""
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import read_s1zsm as Z  # noqa: E402

S, R = Z.S, Z.R
sys.path.insert(0, S.EVAL_DIR)
from test_read_scorer1p6 import Scn  # noqa: E402  (the Batch 1 synthetic world)

CARD_TEXT = open(Z.CARD, encoding="utf-8").read()
LEDGER = Z.DEFAULT_LEDGER
REAL = dict(c0="F:/tanitad_scratch/p6arms/pass/C0", c0r="F:/tanitad_scratch/p6arms/pass/C0r", s1z="F:/tanitad_scratch/p6arms/pass_s1z/S1Z")
REF = HERE / "control_out" / "s1z_read_ref.json"


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


LEDGER_SHA0 = _sha(LEDGER)


def teardown_module(module):
    assert _sha(LEDGER) == LEDGER_SHA0, "the programme's K ledger must not be modified by these tests"


def _row(rid):
    return next(l for l in CARD_TEXT.splitlines() if l.startswith(f"| **{rid}"))


# ------------------------------------------------------------------------------------------------ the card
def test_card_sha_is_frozen():
    assert _sha(Z.CARD) == Z.CARD_SHA256 == "ac75b2891e794de800102a6d5a0f0353e8092e5f976f31b543ac7700811ad6b8"


def test_bars_equal_card_text():
    assert "10,000 draws, seed 20260927" in CARD_TEXT and R.B_DRAWS == 10000 and R.SEED == 20260927
    assert "> 2·F" in _row("P1, primary") and "S1ZSM − C0" in _row("P1, primary") and S.FLOOR_FACTOR == 2.0
    assert "S1ZSM − P6: CI-hi < 0" in _row("M2, mechanism kept") and Z.M2_CI_HI_BAR == 0.0
    assert "S1ZSM − S1Z: CI-lo > 0" in _row("M3, mechanism of this lever") and "EP sub-score" in _row("M3, mechanism of this lever") and Z.M3_CI_LO_BAR == 0.0
    assert float(re.search(r"≥ −(\d+\.\d+)", _row("G1, guard")).group(1)) == -S.G1_BAR
    assert float(re.search(r"< \+(\d+\.\d+) points", _row("G2, guard")).group(1)) == S.G2_BAR
    g3 = _row("G3, guard")
    assert float(re.search(r"v4/v0 > (\d+\.\d+)", g3).group(1)) == S.OVERSPEED_RATIO and float(re.search(r"< \+(\d+\.\d+)", g3).group(1)) == S.G3_BAR
    g4 = _row("G4, guard")
    assert float(re.search(r"< \+(\d+\.\d+)", g4).group(1)) == S.G4_BAR and "definition of `PREREG_S1Z_P6`" in g4
    assert float(re.search(r"point ≥ (\d+\.\d+)", _row("Goal status")).group(1)) == S.GOAL
    assert "F = |C0r − C0| = 0.3796" in CARD_TEXT


def test_outcome_names_and_levers_are_the_cards():
    sec5 = CARD_TEXT.split("## 5 ·")[1].split("## 6 ·")[0]
    for z, name in Z.SM_NAMES.items():
        if z != "Unnamed":
            assert f"**{z} {name}" in sec5, z
    norm = lambda s: re.sub(r"[\s`*]+", " ", s.replace("−", "-")).strip().lower()   # noqa: E731
    sec5n = norm(sec5)
    for z, text in Z.NEXT.items():
        if z == "Unnamed":
            continue
        # the committed lever text appears in the card's bullet (first clause is enough to pin it, whole text for short ones)
        probe = norm(text)[:60]
        assert probe in sec5n, (z, probe)
    assert "Every match is listed. The primary label follows SM1, SM2, SM3, SM4, SM5, SM6, then Unnamed." in sec5 and Z.SM_ORDER == ("SM1", "SM2", "SM3", "SM4", "SM5", "SM6")


# ------------------------------------------------------------------------------------------------ pure bars at exact boundaries
def test_m2_m3_boundaries():
    assert Z.m2_clears(-1e-12) is True and Z.m2_clears(0.0) is False and Z.m2_clears(1e-9) is False and Z.m2_clears(None) is None
    assert Z.m3_clears(1e-12) is True and Z.m3_clears(0.0) is False and Z.m3_clears(-1e-9) is False and Z.m3_clears(None) is None


# ------------------------------------------------------------------------------------------------ (d) outcome map
GOK = {"G1": True, "G2": True, "G3": True, "G4": True}


def st(**kw):
    d = dict(p1=False, m2=False, m3=False, swamped=False, guards=dict(GOK), g1_heads_failed=[], point=91.0)
    d.update(kw)
    return d


def test_SM1_goal_met():
    o = Z.decide_outcome(st(p1=True, m2=True, m3=True, point=94.55))
    assert o["matches"] == ["SM1"] and o["primary"] == "SM1" and o["adopted"] is True and o["next_levers"]["SM1"] == Z.NEXT["SM1"]
    assert Z.decide_outcome(st(p1=True, m2=True, m3=True, point=94.55 - 1e-6))["primary"] == "SM2"


def test_SM2_adopted_below_goal():
    o = Z.decide_outcome(st(p1=True, m2=True, m3=True, point=92.0))
    assert o["matches"] == ["SM2"] and o["name"] == "ADOPTED, BELOW GOAL"


def test_SM2_with_failed_guard_is_not_adopted():
    o = Z.decide_outcome(st(p1=True, m2=True, m3=True, guards={**GOK, "G4": False}))
    assert o["adopted"] is False and o["matches"] == ["Unnamed"] and "G4" in o["note_failed_guards"] and o["failed_guards"] == ["G4"]


def test_SM3_swamped_and_overlap_with_SM5():
    o = Z.decide_outcome(st(swamped=True, m2=True, m3=False))
    assert o["matches"] == ["SM3", "SM5"] and o["primary"] == "SM3" and set(o["next_levers"]) == {"SM3", "SM5"}


def test_SM3_excludes_SM4():
    o = Z.decide_outcome(st(swamped=True, m2=True, m3=True))
    assert o["matches"] == ["SM3"], "SM4 requires 'not SM3'"


def test_SM4_mechanism_no_pdms():
    o = Z.decide_outcome(st(m2=True, m3=True))
    assert o["matches"] == ["SM4"] and o["name"] == "MECHANISM, NO PDMS"


def test_SM5_ep_not_recovered():
    o = Z.decide_outcome(st(m2=True, m3=False))
    assert o["matches"] == ["SM5"]


def test_SM6_failures_back():
    o = Z.decide_outcome(st(m2=False, m3=True))
    assert o["matches"] == ["SM6"] and o["next_levers"]["SM6"] == Z.NEXT["SM6"]
    assert Z.decide_outcome(st(m2=False, m3=False))["matches"] == ["SM6"]


def test_overlap_SM2_SM6_primary_follows_card_order():
    o = Z.decide_outcome(st(p1=True, m2=False, m3=False, point=92.0))
    assert o["matches"] == ["SM2", "SM6"] and o["primary"] == "SM2"
    o = Z.decide_outcome(st(p1=True, m2=False, m3=True, point=95.0))
    assert o["matches"] == ["SM1", "SM6"] and o["primary"] == "SM1"
    o = Z.decide_outcome(st(swamped=True, m2=False))
    assert o["matches"] == ["SM3", "SM6"] and o["primary"] == "SM3"


def test_unnamed_when_nothing_matches_and_when_input_undefined():
    o = Z.decide_outcome(st(p1=True, m2=True, m3=True, guards={**GOK, "G2": False}))
    assert o["primary"] == "Unnamed" and o["matches"] == ["Unnamed"] and o["next_levers"]["Unnamed"] == Z.NEXT["Unnamed"]
    for k in ("p1", "m2", "m3", "swamped"):
        assert Z.decide_outcome(st(**{k: None}))["primary"] == "Unnamed"
    assert Z.decide_outcome(st(guards={**GOK, "G3": None}))["primary"] == "Unnamed"
    assert Z.decide_outcome(st(guards={}))["primary"] == "Unnamed"


def test_every_label_reachable_and_each_has_a_lever():
    seen = set()
    for p1 in (True, False):
        for m2 in (True, False):
            for m3 in (True, False):
                for sw in (True, False):
                    for g in (True, False):
                        for pt in (91.0, 95.0):
                            o = Z.decide_outcome(st(p1=p1, m2=m2, m3=m3, swamped=sw, guards={**GOK, "G1": g}, point=pt))
                            assert o["primary"] in o["matches"] and all(z in Z.NEXT for z in o["matches"]) and set(o["next_levers"]) == set(o["matches"])
                            seen.update(o["matches"])
    assert seen == set(Z.SM_NAMES)


def test_relabel_only_whole_tokens():
    assert Z.relabel({"S1Z_minus_P6": "S1Z e3_logits S1ZSM", "x": ["S1Z"]}) == {"S1ZSM_minus_P6": "S1ZSM e3_logits S1ZSM", "x": ["S1ZSM"]}


# ------------------------------------------------------------------------------------------------ synthetic end to end
@pytest.fixture(scope="module")
def Sc():
    return Scn()


def inputs(Sc, c0=None, c0r=None, s1z=None, s1zsm=None):
    d = {"C0": (Sc.p6_pick, 6.0, 11), "C0r": (Sc.p6_pick, 6.0, 12), "S1Z": (Sc.p6_pick, 6.0, 13), "S1ZSM": (Sc.p6_pick, 6.0, 14)}
    for k, v in (("C0", c0), ("C0r", c0r), ("S1Z", s1z), ("S1ZSM", s1zsm)):
        if v:
            d[k] = v
    return Sc.inputs({k: Sc.arm(k, *v) for k, v in d.items()})


def read(Sc, **kw):
    return Z.s1zsm_read(inputs(Sc, **kw))


def test_e2e_goal_met_with_m3(Sc):
    r = read(Sc, s1zsm=(Sc.oracle, 0.2, 21))
    assert r["status"] == "READ", r.get("why")
    o = r["outcome"]
    assert o["primary"] == "SM1" and r["M2"]["clears"] is True and r["M3"]["S1ZSM_minus_S1Z"]["lo"] > 0 and r["M3"]["clears"] is True
    assert o["inputs"]["M3"] is True and r["adoption"]["adopted"] is True and r["goal_status"]["status"] == "GOAL MET"
    assert "S1Z" not in json.dumps(r["P1"]).replace("S1ZSM", "")


def test_e2e_s1zsm_equal_s1z_gives_m3_exactly_zero(Sc):
    r = read(Sc, s1z=(Sc.mix(0.5), 0.2, 22), s1zsm=(Sc.mix(0.5), 0.2, 23))
    d = r["M3"]["S1ZSM_minus_S1Z"]
    assert d["point"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0 and r["M3"]["clears"] is False
    rp = r["reported_not_deciding"]
    assert rp["S1ZSM_minus_S1Z_PDMS_x100_E3"]["point"] == 0.0 and rp["S1ZSM_minus_S1Z_failure_share_E3"]["point"] == 0.0


def test_e2e_m3_red_when_arm_is_worse_than_s1z_on_ep(Sc):
    r = read(Sc, s1z=(Sc.oracle_fast, 0.2, 24), s1zsm=(Sc.p6_pick, 6.0, 25))
    assert r["M3"]["S1ZSM_minus_S1Z"]["point"] < 0 and r["M3"]["clears"] is False


def test_e2e_sm6_when_arm_is_p6(Sc):
    r = read(Sc)
    assert r["M2"]["clears"] is False and "SM6" in r["outcome"]["matches"] and r["M2"]["S1ZSM_minus_P6"]["hi"] == 0.0


def test_e2e_m2_ignores_the_c0_clause(Sc):
    # arm beats P6 on failures; C0 is the oracle (so arm - C0 CI-hi >= 0): the S1Z reader's M2 would fail on its C0 clause, this card's must not
    r = read(Sc, c0=(Sc.oracle, 0.2, 26), c0r=(Sc.oracle, 0.2, 27), s1zsm=(Sc.mix(0.5), 0.2, 28))
    assert r["M2"]["S1ZSM_minus_P6"]["hi"] < 0 and r["M2"]["clears"] is True


def test_e2e_scores_missing_is_incomplete(Sc):
    inp = inputs(Sc)
    inp.arms["S1ZSM"] = Sc.arm("S1ZSM", Sc.oracle, 0.2, 29, scored=False)
    r = Z.s1zsm_read(inp)
    assert r["status"] == "INCOMPLETE" and "outcome" not in r
    inp = inputs(Sc)
    inp.arms["S1Z"] = Sc.arm("S1Z", Sc.oracle, 0.2, 30, scored=False)
    assert Z.s1zsm_read(inp)["status"] == "INCOMPLETE"


def test_pair_shares_and_rep_read_are_reported(Sc, tmp_path):
    pw = tmp_path / "pair_weight.json"
    pw.write_text(json.dumps({"sets_with_pair_share": {"NC": 0.4, "DAC": 0.5, "TTC": 0.3}}), encoding="utf-8")
    lg = tmp_path / "run.jsonl"
    lg.write_text('{"event": "x"}\n{"event": "pair_calibrated", "sets_with_pair_share": {"NC": 0.1}}\nnot json\n', encoding="utf-8")
    rr = tmp_path / "rep_read.json"
    rr.write_text(json.dumps({"seed": {"s_p": 0.2, "df": 3, "t_0.975": 3.182}}), encoding="utf-8")
    r = Z.s1zsm_read(inputs(Sc, s1zsm=(Sc.oracle, 0.2, 31)), run_log=str(pw), rep_read=str(rr))
    rp = r["reported_not_deciding"]
    assert rp["pair_shares_of_the_run"]["sets_with_pair_share"] == {"NC": 0.4, "DAC": 0.5, "TTC": 0.3}
    assert rp["seed_sd_next_to_2F"]["t_times_s_p"] == pytest.approx(3.182 * 0.2) and rp["seed_sd_next_to_2F"]["2F"] == pytest.approx(2 * r["P1"]["S1ZSM_vs_C0"]["F"])
    assert Z.read_pair_shares(str(lg))["sets_with_pair_share"] == {"NC": 0.1}
    assert Z.read_pair_shares(None) == {"given": False} and Z.read_pair_shares(str(tmp_path / "nope"))["found"] is False
    assert Z.read_rep(str(tmp_path / "nope"), 0.4)["ok"] is False


def test_every_block_carries_estimator_tier_n_artifact(Sc):
    r = read(Sc, s1zsm=(Sc.oracle, 0.2, 32))
    for blk in (r["M2"], r["M3"], r["P1"], r["guards"]["G1"], r["guards"]["G4"], r["reported_not_deciding"]):
        assert all(k in blk for k in ("estimator", "tier", "n", "artifact"))
    assert "PAUSED" in r["strategic"]


def test_ledger_written_to_a_temp_copy_only(Sc, tmp_path):
    led = tmp_path / "ledger_K.json"
    if os.path.exists(LEDGER):
        shutil.copy(LEDGER, led)
    r = Z.s1zsm_read(inputs(Sc, s1zsm=(Sc.oracle, 0.2, 33)), ledger_path=str(led))
    k = r["K_ledger"]
    assert k["this_read_added"] is True and k["key"].startswith("S1ZSM:") and k["K"] >= 1 and k["bonferroni_pdms_S1ZSM_minus_P6"]["hi"] >= k["bonferroni_pdms_S1ZSM_minus_P6"]["lo"]
    assert "S1ZSM:" in open(led, encoding="utf-8").read()
    assert _sha(LEDGER) == LEDGER_SHA0
    r2 = Z.s1zsm_read(inputs(Sc, s1zsm=(Sc.oracle, 0.2, 33)))
    assert r2["K_ledger"]["K"] is None


# ------------------------------------------------------------------------------------------------ (b) RED: one changed byte in the card
def test_RED_card_one_byte_changed_refuses(Sc, tmp_path):
    b = bytearray(open(Z.CARD, "rb").read())
    b[1000] = b[1000] ^ 0x01
    bad = tmp_path / "PREREG_S1ZSM_P6.md"
    bad.write_bytes(bytes(b))
    r = Z.s1zsm_read(inputs(Sc, s1zsm=(Sc.oracle, 0.2, 34)), card_path=str(bad))
    assert r["status"] == "READ STOPPED" and "REFUSED" in r["why"] and "outcome" not in r and "M3" not in r
    assert r["card"]["sha256_actual"] != Z.CARD_SHA256
    r2 = Z.s1zsm_read(inputs(Sc, s1zsm=(Sc.oracle, 0.2, 34)), card_path=str(tmp_path / "missing.md"))
    assert r2["status"] == "READ STOPPED"
    md = Z.to_markdown(r)
    assert "No verdict is written" in md


def test_RED_cli_refuses_changed_card(tmp_path):
    if not all(os.path.isdir(p) for p in REAL.values()):
        pytest.skip("real inputs not on this box")
    b = bytearray(open(Z.CARD, "rb").read())
    b[500] ^= 0x01
    bad = tmp_path / "card.md"
    bad.write_bytes(bytes(b))
    rc = Z.main(["read", "--c0", REAL["c0"], "--c0r", REAL["c0r"], "--s1z", REAL["s1z"], "--s1zsm", REAL["s1z"], "--card", str(bad), "--no-ledger", "--out", str(tmp_path / "o")])
    assert rc == 2
    j = json.load(open(tmp_path / "o" / "s1zsm_read.json", encoding="utf-8"))
    assert j["status"] == "READ STOPPED" and "outcome" not in j


# ------------------------------------------------------------------------------------------------ (c) RED: G0 gate
def test_RED_arm_proposals_not_bit_identical_fails_g0(Sc):
    inp = inputs(Sc, s1zsm=(Sc.oracle, 0.2, 35))
    a = inp.arms["S1ZSM"]
    a.e3_proposals = a.e3_proposals.copy()
    a.e3_proposals[0, 0, 0, 0] += 1.0
    r = Z.s1zsm_read(inp)
    assert r["status"] == "READ STOPPED" and "outcome" not in r and "M3" not in r
    assert any(not g["PASS"] for g in r["gates"].values()) and "G0-E3" in json.dumps(r["gates"])


def test_RED_reference_s1z_not_bit_identical_fails_g0(Sc):
    inp = inputs(Sc, s1zsm=(Sc.oracle, 0.2, 36))
    a = inp.arms["S1Z"]
    a.e3_proposals = a.e3_proposals.copy()
    a.e3_proposals[3, 5, 1, 1] += 0.5
    r = Z.s1zsm_read(inp)
    assert r["status"] == "READ STOPPED" and r["gates"]["G0-E3[S1Z reference]"]["PASS"] is False and "outcome" not in r


def test_RED_banked_p6_pick_mismatch_fails_g0(Sc):
    inp = inputs(Sc, s1zsm=(Sc.oracle, 0.2, 37))
    inp.p6_pick_banked = inp.p6_pick_banked.copy()
    inp.p6_pick_banked[0] = (inp.p6_pick_banked[0] + 1) % 64
    r = Z.s1zsm_read(inp)
    assert r["status"] == "READ STOPPED" and r["gates"]["G0-PICK"]["PASS"] is False


def test_RED_missing_reference_or_c0r_stops(Sc):
    for drop in ("S1Z", "C0r", "S1ZSM", "C0"):
        inp = inputs(Sc)
        del inp.arms[drop]
        assert Z.s1zsm_read(inp)["status"] == "READ STOPPED"


# ------------------------------------------------------------------------------------------------ (a) CONTROL on the real files
@pytest.mark.skipif(not (all(os.path.isdir(p) for p in REAL.values()) and REF.exists()), reason="real inputs not on this box")
def test_CONTROL_s1z_as_the_arm_equals_banked_s1z_read(tmp_path):
    rc = Z.main(["read", "--c0", REAL["c0"], "--c0r", REAL["c0r"], "--s1z", REAL["s1z"], "--s1zsm", REAL["s1z"], "--control-s1z-read", str(REF), "--out", str(tmp_path / "ctl")])
    assert rc == 0
    j = json.load(open(tmp_path / "ctl" / "s1zsm_read.json", encoding="utf-8"))
    c = j["control"]
    assert c["EQUAL"] is True and c["n_mismatch"] == 0 and c["n_leaves_compared"] > 80
    d = c["M3_S1Z_minus_S1Z"]
    assert d["point"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0 and c["PASS"] is True and j["M3"]["clears"] is False
    # the same numbers the banked read holds, spot-checked independently of the leaf walk
    ref = json.load(open(REF, encoding="utf-8"))
    assert j["P1"]["PDMS_x100_E3"] == ref["P1"]["PDMS_x100_E3"] and j["M2"]["S1ZSM_minus_P6"] == ref["M2"]["S1Z_minus_P6"]
    assert j["reported_not_deciding"]["EP_subscore_vs_P6"] == ref["reported_not_deciding"]["EP_subscore_vs_P6"]
    assert {k: v["ok"] for k, v in j["guards"].items()} == {k: v["ok"] for k, v in ref["guards"].items()}
    assert j["K_ledger"]["K"] is None and _sha(LEDGER) == LEDGER_SHA0


@pytest.mark.skipif(not (all(os.path.isdir(p) for p in REAL.values()) and REF.exists()), reason="real inputs not on this box")
def test_CONTROL_red_a_changed_reference_number_is_detected(tmp_path):
    ref = json.load(open(REF, encoding="utf-8"))
    ref["P1"]["PDMS_x100_E3"]["lo"] += 1e-9
    bad = tmp_path / "ref.json"
    bad.write_text(json.dumps(ref), encoding="utf-8")
    rc = Z.main(["read", "--c0", REAL["c0"], "--c0r", REAL["c0r"], "--s1z", REAL["s1z"], "--s1zsm", REAL["s1z"], "--control-s1z-read", str(bad), "--out", str(tmp_path / "ctl")])
    j = json.load(open(tmp_path / "ctl" / "s1zsm_read.json", encoding="utf-8"))
    assert rc == 0 and j["control"]["EQUAL"] is False and j["control"]["PASS"] is False and j["control"]["n_mismatch"] >= 1
