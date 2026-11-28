"""Tests for read_dose.py (CPU only).

    CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_read_dose.py --basetemp F:/tanitad_scratch/dose_read_work/pt/run

(the parent of --basetemp must exist). Pure arithmetic is tested on hand-made numbers at strict boundaries. The end-to-end tests read the real arm dirs under F:/tanitad_scratch/p6arms
(controls: S1Z1 fed as S1ZD, P6's own picks fed as S1ZD) and write only under F:/tanitad_scratch/dose_read_work; they point at a COPY of the K ledger and assert the real ledger's sha256 is unchanged.
"""
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import read_dose as X  # noqa: E402

R, RR = X.R, X.RR
PA = "F:/tanitad_scratch/p6arms"
WORK = X.CONTROL_ROOT
CARD_TEXT = open(X.CARD, encoding="utf-8").read()
REAL_LEDGER = X.REAL_LEDGER
REP_JSON = "F:/tanitad_scratch/repread_real/rep_read.json"
ARGS = ["read", "--p6-pass", PA + "/pass", "--c0", PA + "/pass/C0", "--c0r", PA + "/pass/C0r", "--s1z", PA + "/pass_s1z/S1Z", "--s1z1", PA + "/pass_rep_c0r2_s1e1_s1z1/S1Z1"]
S1Z1 = PA + "/pass_rep_c0r2_s1e1_s1z1/S1Z1"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------------ the card
def test_card_sha_is_frozen():
    assert sha(X.CARD) == X.CARD_SHA256
    assert open(X.CARD.replace(".md", ".md.sha256")).read().split()[0] == X.CARD_SHA256


def test_constants_equal_card_text():
    assert "2.776 · 0.1791 = **0.4972**" in CARD_TEXT and X.S_P == 0.1791 and X.T_975_DF4 == 2.776
    assert "= **0.6089**" in CARD_TEXT and round(X.D1_THR, 4) == 0.4972 and round(X.D2_THR, 4) == 0.6089
    assert "94.55" in CARD_TEXT and X.GOAL == 94.55
    assert "10,000 draws, seed 20260927" in CARD_TEXT and R.B_DRAWS == 10000 and R.SEED == 20260927
    assert "Ordering of labels when several match: A, B, D, C" in CARD_TEXT and X.ORDER == ("DOSE-A", "DOSE-B", "DOSE-D", "DOSE-C")
    assert "t · s_p · √(1 + 1/2)" in CARD_TEXT
    assert math.isclose(X.D2_THR, X.D1_THR * math.sqrt(1.5))


def test_next_levers_quote_the_card():
    for k, frag in (("DOSE-A", "6,400-update dose"), ("DOSE-B", "seed-1 replicate"), ("DOSE-D", "Shapley over heads"), ("DOSE-C", "zero-multiplier picks (2.81 of 6.70 points)")):
        assert frag in X.NEXT[k] and frag in CARD_TEXT


# ------------------------------------------------------------------------------------------------ bars at strict boundaries
def test_d1_boundaries():
    t = X.D1_THR
    assert X.d1_proven(0.01, t + 1e-9)
    assert not X.d1_proven(0.01, t)            # strict
    assert not X.d1_proven(0.0, t + 1.0)       # CI-lo must be > 0
    assert not X.d1_proven(-0.01, t + 1.0)
    assert not X.d1_proven(None, 1.0)


def test_d2_boundaries():
    t = X.D2_THR
    assert X.d2_proven(0.01, t + 1e-9) and not X.d2_proven(0.01, t) and not X.d2_proven(0.0, t + 1)


def test_goal_boundary():
    assert RR.goal_label(94.55) == "GOAL MET" and RR.goal_label(94.5) == "BELOW GOAL"


# ------------------------------------------------------------------------------------------------ outcome map and label ordering
@pytest.mark.parametrize("d1,d2,g,want,allm", [
    (True, True, True, "DOSE-A", ["DOSE-A"]),
    (True, False, True, "DOSE-B", ["DOSE-B"]),
    (True, True, False, "DOSE-D", ["DOSE-D"]),
    (True, False, False, "DOSE-B", ["DOSE-B", "DOSE-D"]),     # ordering A, B, D, C: B before D
    (False, False, True, "DOSE-C", ["DOSE-C"]),
    (False, True, False, "DOSE-C", ["DOSE-C"]),
])
def test_outcome_map(d1, d2, g, want, allm):
    o = X.decide_outcome(d1, d2, g)
    assert o["primary"] == want and o["matches"] == allm


def test_unnamed_when_undefined():
    o = X.decide_outcome(True, True, True, defined=False)
    assert o["primary"] == "Unnamed" and o["matches"] == ["Unnamed"]


# ------------------------------------------------------------------------------------------------ regret split on hand-made numbers
class FakeB3:
    def stat(self, v, mask=None):
        v = np.asarray(v, float)
        return float(v.mean()), np.array([v.mean(), v.mean()]), len(v)

    def diff(self, a, b, mask=None):
        return self.stat(np.asarray(a) - np.asarray(b))


class FakeTable:
    pass


def test_regret_classes_hand_numbers():
    # 5 tokens x 3 proposals; sub order NC DAC EP TTC C DDC
    P = np.array([[0.0, 0.8, 0.9],      # pick 0: zero pick, a non-zero proposal exists -> zero-multiplier, regret 0.9
                  [0.0, 0.0, 0.0],      # pick 0: all zero -> zero_pick_proposal, regret 0
                  [0.5, 0.7, 0.9],      # pick 0: non-zero, TTC 0 -> regret 0.4
                  [0.6, 0.7, 0.9],      # pick 0: non-zero, TTC 1 -> rest, regret 0.3
                  [0.9, 0.2, 0.1]])     # pick 0: best -> rest, regret 0
    S_ = np.ones((5, 3, 6))
    S_[2, 0, 3] = 0.0
    reg, m, best, pk = X.regret_masks(P, S_, np.zeros(5, int))
    assert [int(m[k].sum()) for k in X.CLASSES] == [1, 1, 1, 2]
    assert math.isclose(100 * (reg * m["zero_pick_selection"]).mean(), 100 * 0.9 / 5)
    assert math.isclose(100 * (reg * m["nonzero_pick_TTC0"]).mean(), 100 * 0.4 / 5)
    assert math.isclose(100 * (reg * m["rest"]).mean(), 100 * 0.3 / 5)
    t = FakeTable()
    t.pdms, t.sub = P, S_
    out = X.regret_split(t, np.arange(5), np.zeros(5, int), FakeB3())
    assert math.isclose(out["sum_of_components_x100"], out["regret_x100"]) and out["n_tokens"] == 5


def test_sft_log_parser(tmp_path):
    p = tmp_path / "sft.log"
    rows = [{"event": "start", "updates": 1600}, {"event": "eval", "tag": "step0", "n": 3137, "base": {"pick": 90.0, "zero_picks": 5}, "B": {"pick": 90.0, "zero_picks": 5, "pick_minus_base": {"mean": 0.0, "ci95": [0.0, 0.0]}}},
            {"event": "eval", "tag": "update200", "n": 3137, "base": {"pick": 90.0, "zero_picks": 5}, "B": {"pick": 91.5, "zero_picks": 3, "pick_minus_base": {"mean": 1.5, "ci95": [1.0, 2.0]}}}]
    p.write_text("noise line\n" + "\n".join("2026-11-10 " + json.dumps(r) for r in rows) + "\n{broken\n", encoding="utf-8")
    got = X.parse_sft_log(str(p))
    assert [g["tag"] for g in got] == ["step0", "update200"] and got[1]["B_pick_x100"] == 91.5 and got[1]["B_minus_base_ci95"] == [1.0, 2.0] and got[1]["B_zero_picks"] == 3


# ------------------------------------------------------------------------------------------------ end to end (real arm dirs)
def run_main(out_name, s1zd, extra, monkeypatch=None):
    out = f"{WORK}/test_{out_name}"
    shutil.rmtree(out, ignore_errors=True)
    rc = X.main(ARGS + ["--s1zd", s1zd, "--out", out] + extra)
    return rc, out


@pytest.fixture(scope="module")
def ledger_copy():
    before = sha(REAL_LEDGER)
    cp = f"{WORK}/test_ledger_copy.json"
    shutil.copy(REAL_LEDGER, cp)
    yield cp
    assert sha(REAL_LEDGER) == before, "the REAL K ledger changed"
    os.remove(cp)


@pytest.fixture(scope="module")
def ctl_a(ledger_copy):
    k0 = len(json.load(open(ledger_copy))["reads"])
    rc, out = run_main("ctl_a", S1Z1, ["--control-standin", "--ledger", ledger_copy])
    return rc, json.load(open(out + "/dose_read.json")), out, k0, ledger_copy


def test_control_a_s1z1_as_s1zd_equals_replicate_read(ctl_a):
    rc, d, out, k0, led = ctl_a
    assert rc == 0 and d["status"] == "READ" and d["stamp"] == "CONTROL_NOT_A_RESULT"
    ref = json.load(open(REP_JSON))["runs"]["S1Z1"]["E3_pdms_vs_P6"]
    got = d["bars_read"]["D1"]["d_S1ZD_minus_P6"]
    assert got == ref                                     # every digit
    assert (round(got["point"], 4), round(got["lo"], 4), round(got["hi"], 4)) == (0.3861, 0.1705, 0.6148)
    assert d["bars_read"]["D1"]["PROVEN"] is False and d["bars_read"]["D1"]["d_gt_threshold"] is False and d["outcome"]["primary"] == "DOSE-C"
    assert d["p6_pass_gate"]["PASS"] and d["p6_pass_gate"]["n_pick_equal_banked_p6"] == d["p6_pass_gate"]["n_tokens"]
    assert d["bars_read"]["goal"]["status"] == "BELOW GOAL"


def test_control_a_ledger_one_control_key_on_the_copy(ctl_a):
    rc, d, out, k0, led = ctl_a
    reads = json.load(open(led))["reads"]
    assert len(reads) == k0 + 1 and reads[-1]["key"].startswith("CONTROL:DOSE:S1ZD:") and reads[-1]["key"].endswith(":E3") and reads[-1]["family"] == "DOSE"
    assert d["K_ledger"]["K"] == k0 + 1 and len(d["K_ledger"]["keys_added"]) == 1


def test_control_a_regret_split_is_p6s(ctl_a):
    rc, d, out, k0, led = ctl_a
    c = d["regret_split"]["P6"]["components_x100"]
    assert round(c["zero_pick_selection"]["regret_contribution"], 2) == 2.81 and round(c["nonzero_pick_TTC0"]["regret_contribution"], 2) == 1.58 and round(c["rest"]["regret_contribution"], 2) == 2.32
    assert round(d["regret_split"]["P6"]["regret_x100"], 2) == 6.70 and round(d["regret_split"]["P6"]["mean_best_x100"], 2) == 97.90
    assert (c["zero_pick_selection"]["n_tokens"], c["zero_pick_proposal"]["n_tokens"], c["nonzero_pick_TTC0"]["n_tokens"], c["rest"]["n_tokens"]) == (355, 29, 449, 11313)
    # S1ZD (= S1Z1 here) regret falls by exactly its PDMS gain
    assert math.isclose(d["regret_split"]["S1ZD_minus_P6_contribution_x100"]["total_regret"]["point"], -d["bars_read"]["D1"]["d_S1ZD_minus_P6"]["point"], abs_tol=1e-9)


def test_control_a_strategic_paused_quoted_as_read_rep(ctl_a):
    rc, d, out, k0, led = ctl_a
    assert d["strategic"] == R.STRATEGIC and "PAUSED" in d["strategic"] and "2026-09-19" in d["strategic"]
    md = open(out + "/dose_read.md", encoding="utf-8", newline="").read()
    assert "\r" not in md and "PAUSED" in md
    assert b"\r" not in open(out + "/dose_read.json", "rb").read()


def test_control_b_p6_picks_as_s1zd(ledger_copy):
    rc, out = run_main("ctl_b", PA + "/pass/P6_control", ["--control-standin", "--ledger", ledger_copy])
    d = json.load(open(out + "/dose_read.json"))
    assert rc == 0
    assert d["bars_read"]["D1"]["d_S1ZD_minus_P6"] == {"point": 0.0, "lo": 0.0, "hi": 0.0, "n": 12146}
    rg = d["regret_split"]
    c = rg["S1ZD"]["components_x100"]
    assert [round(c[k]["regret_contribution"], 2) for k in ("zero_pick_selection", "nonzero_pick_TTC0", "rest")] == [2.81, 1.58, 2.32]
    assert all(v["point"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 for v in rg["S1ZD_minus_P6_contribution_x100"].values())
    assert d["outcome"]["primary"] == "DOSE-C"


def test_deliberate_regression_threshold_zero_changes_the_verdict(ctl_a, monkeypatch):
    """control a reads D1 NOT PROVEN (0.3861 < 0.4972); with the D1 threshold at 0 the same data must read PROVEN and the outcome must change"""
    rc, d, out, k0, led = ctl_a
    d1 = d["bars_read"]["D1"]["d_S1ZD_minus_P6"]
    assert not X.d1_proven(d1["lo"], d1["point"])
    monkeypatch.setattr(X, "D1_THR", 0.0)
    assert X.d1_proven(d1["lo"], d1["point"])
    assert X.decide_outcome(True, False, True)["primary"] == "DOSE-B" != d["outcome"]["primary"]


def test_deliberate_regression_altered_card_is_refused(tmp_path, ledger_copy):
    p = tmp_path / "card.md"
    p.write_text(CARD_TEXT.replace("0.4972", "0.0000"), encoding="utf-8")
    rc, out = run_main("card_alt", S1Z1, ["--control-standin", "--ledger", ledger_copy, "--card", str(p)])
    d = json.load(open(out + "/dose_read.json"))
    assert rc == 2 and d["status"] == "READ STOPPED" and "sha256" in d["why"] and "outcome" not in d


def test_real_read_refuses_duplicate_arm(tmp_path):
    import tempfile
    out = tempfile.mkdtemp(prefix="dose_dup_")      # C: temp, outside the control folder; removed below
    inp_args = ARGS + ["--s1zd", S1Z1, "--out", out, "--no-ledger"]
    rc = X.main(inp_args)
    d = json.load(open(out + "/dose_read.json"))
    shutil.rmtree(out, ignore_errors=True)
    assert rc == 2 and d["status"] == "REFUSED" and "bit-identical" in d["why"]


def test_control_refuses_real_ledger_and_real_read_refuses_control_folder():
    with pytest.raises(SystemExit):
        X.main(ARGS + ["--s1zd", S1Z1, "--out", WORK + "/x", "--control-standin"])        # default ledger = the real one
    with pytest.raises(SystemExit):
        X.main(ARGS + ["--s1zd", S1Z1, "--out", "F:/tanitad_scratch/elsewhere", "--control-standin", "--no-ledger"])
    with pytest.raises(SystemExit):
        X.main(ARGS + ["--s1zd", S1Z1, "--out", WORK + "/x", "--no-ledger"])


def test_missing_run_is_refused():
    class Inp:
        arms = {"C0": 1}
        provenance = {"arms": {}}
    res = X.dose_read(Inp(), None)
    assert res["status"] == "REFUSED" or res["status"] == "READ STOPPED"
