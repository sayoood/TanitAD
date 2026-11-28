"""Tests for read_rep.py (CPU only).

    mkdir -p F:/tanitad_scratch/repread/pt
    CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_read_rep.py --basetemp F:/tanitad_scratch/repread/pt/run

Pure arithmetic is tested on hand-made numbers at exact (strict) boundaries; the end-to-end tests read the real C0, C0r, S1E arm dirs (Batch 1) and so need F:/tanitad_scratch/p6arms/pass.
"""
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import read_rep as X  # noqa: E402

R = X.R
PASS = "F:/tanitad_scratch/p6arms/pass"
VERDICT = "F:/tanitad_scratch/p6arms/read/verdict/scorer1p6_read.json"
CARD_TEXT = open(X.CARD, encoding="utf-8").read()


# ------------------------------------------------------------------------------------------------ the card
def test_card_sha_is_frozen():
    assert hashlib.sha256(open(X.CARD, "rb").read()).hexdigest() == X.CARD_SHA256


def test_constants_equal_card_text():
    assert "4.303" in CARD_TEXT and "3.182" in CARD_TEXT and "2.776" in CARD_TEXT and X.T_TABLE == {2: 4.303, 3: 3.182, 4: 2.776}
    assert "2 · 0.3796 = 0.759" in CARD_TEXT and X.F_BATCH1 == 0.3796
    assert "94.55" in CARD_TEXT and X.GOAL == 94.55
    assert "10,000 draws, seed 20260927" in CARD_TEXT and R.B_DRAWS == 10000 and R.SEED == 20260927
    assert "d > t · s_p / √n_R" in CARD_TEXT and "t · s_p · √(1/n_R + 1/3)" in CARD_TEXT


def test_refused_card_sha(tmp_path, monkeypatch):
    p = tmp_path / "card.md"
    p.write_text(CARD_TEXT + "\nedited\n", encoding="utf-8")
    monkeypatch.setattr(X, "CARD", str(p))

    class Inp:
        provenance = {"arms": {}}
        arms = {}

    res = X.rep_read(Inp(), None)
    assert res["status"] == "READ STOPPED" and "sha256" in res["why"]
    assert "outcome" not in res


# ------------------------------------------------------------------------------------------------ s_p, df, t on hand-made numbers
def test_seed_stats_hand_numbers():
    ss = X.seed_stats({"C0": [1.0, 2.0, 3.0], "S1E": [10.0, 12.0]})
    assert ss["df"] == 3
    assert ss["s_R"]["C0"] == pytest.approx(1.0) and ss["s_R"]["S1E"] == pytest.approx(math.sqrt(2.0))
    assert ss["s_p"] == pytest.approx(math.sqrt((2 * 1.0 + 1 * 2.0) / 3.0))     # (n-1)-weighted pooled variance
    ss = X.seed_stats({"C0": [1.0, 2.0, 3.0], "S1E": [10.0, 12.0], "S1Z": [5.0, 5.0]})
    assert ss["df"] == 4 and ss["s_p"] == pytest.approx(math.sqrt((2 * 1.0 + 2.0 + 0.0) / 4.0))
    ss = X.seed_stats({"C0": [4.0, 4.0, 4.0], "S1E": [7.0]})                      # a one-run recipe adds no df
    assert ss["df"] == 2 and ss["s_p"] == 0.0 and "S1E" not in ss["s_R"]


def test_df_zero_is_refused():
    with pytest.raises(X.Refused, match="df = 0"):
        X.seed_stats({"C0": [91.0], "S1E": [91.2]})
    with pytest.raises(X.Refused):
        X.seed_stats({})
    with pytest.raises(X.Refused):
        X.t_value(0)


def test_t_values(monkeypatch):
    assert X.t_value(2) == 4.303 and X.t_value(3) == 3.182 and X.t_value(4) == 2.776
    assert X.t_value(1) == pytest.approx(12.706, abs=1e-3) and X.t_value(5) == pytest.approx(2.5706, abs=1e-3)
    from scipy import stats
    for df in (2, 3, 4):
        assert abs(stats.t.ppf(0.975, df) - X.T_TABLE[df]) < 5e-4          # the card's printed values are scipy's, rounded
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    assert X.t_value(3) == 3.182                                         # no scipy needed for the card's df
    with pytest.raises(X.Refused, match="scipy"):
        X.t_value(7)


def test_f3():
    assert X.f3_of([91.229, 91.608, 91.0]) == pytest.approx(0.608)
    assert X.f3_of([1.0]) is None and X.f3_of([]) is None
    thr = 2 * 0.3796
    assert X.floor_clause(thr)["triggered"] is False                      # strict: F3 > 0.7592
    assert X.floor_clause(float(np.nextafter(thr, 1.0)))["triggered"] is True
    assert X.floor_clause(0.5)["text"] is None and "understated training noise" in X.floor_clause(0.9)["text"]


# ------------------------------------------------------------------------------------------------ R1 and R2 at strict boundaries
def test_r1_boundaries():
    t, s_p, n = 3.182, 0.2, 3
    thr = X.r1_threshold(t, s_p, n)
    assert thr == pytest.approx(3.182 * 0.2 / math.sqrt(3))
    up = float(np.nextafter(thr, np.inf))
    assert X.bar_proven(0.01, up, thr) is True                            # both clear
    assert X.bar_proven(0.01, thr, thr) is False                          # d == threshold: strict
    assert X.bar_proven(0.01, float(np.nextafter(thr, 0)), thr) is False
    assert X.bar_proven(0.0, up, thr) is False                            # CI-lo == 0: strict
    assert X.bar_proven(-0.01, 5.0, thr) is False
    assert X.bar_proven(0.01, up, thr) and not X.bar_proven(None, up, thr) and not X.bar_proven(0.01, None, thr)


def test_r2_boundaries():
    t, s_p = 3.182, 0.2
    thr = X.r2_threshold(t, s_p, 2, 3)
    assert thr == pytest.approx(3.182 * 0.2 * math.sqrt(1 / 2 + 1 / 3))
    assert X.r2_threshold(t, s_p, 2, 3) > X.r1_threshold(t, s_p, 2)      # a difference of two means is harder than a mean against a constant
    up = float(np.nextafter(thr, np.inf))
    assert X.bar_proven(0.001, up, thr) and not X.bar_proven(0.001, thr, thr) and not X.bar_proven(0.0, up, thr)


def test_goal_label():
    assert X.goal_label(94.55) == "GOAL MET" and X.goal_label(94.549) == "BELOW GOAL"


def test_outcome_map():
    o = X.decide_outcome({"C0": False, "S1E": False}, {"C0": False, "S1E": False}, {"S1E": False})
    assert o["primary"] == "REP-B" and o["matches"] == ["REP-B"] and "dose arm" in o["next_levers"]["REP-B"]
    o = X.decide_outcome({"C0": True, "S1E": False}, {"C0": True, "S1E": False}, {"S1E": False})
    assert o["primary"] == "REP-A" and "ENS_R" in o["next_levers"]["REP-A"]
    o = X.decide_outcome({"C0": False, "S1E": False}, {"C0": False, "S1E": True}, {"S1E": True})       # S1E proven on R1 but a guard failed: not adoptable
    assert o["matches"] == ["REP-C"] and "stacked" in o["next_levers"]["REP-C"]
    o = X.decide_outcome({"C0": False, "S1E": True}, {"C0": False, "S1E": True}, {"S1E": True})
    assert o["matches"] == ["REP-A", "REP-C"] and o["primary"] == "REP-A"
    o = X.decide_outcome({"C0": False}, {"C0": True}, {})                  # PROVEN on R1 with a failed guard and no REP-C: neither A nor B
    assert o["matches"] == ["Unnamed"] and o["primary"] == "Unnamed"
    o = X.decide_outcome({"C0": False, "S1E": False}, {"C0": False, "S1E": False}, {"S1E": False})
    assert "Unnamed" not in o["matches"]


# ------------------------------------------------------------------------------------------------ real-data tests (Batch 1 arms)
needs_data = pytest.mark.skipif(not os.path.exists(PASS + "/C0/e3_logits.npz"), reason="Batch 1 arm dirs not on this box")


def _read(tmp, runs, extra=(), name="o"):
    out = str(Path(tmp) / name)
    args = ["read", "--no-ledger", "--out", out] + [x for r in runs for x in ("--run", r)] + list(extra)
    rc = X.main(args)
    return rc, json.load(open(out + "/rep_read.json", encoding="utf-8"))


@needs_data
def test_changed_rows_equal_changed_from_table(tmp_path):
    import changed_from_table as C
    tab = X.Table(X.DEFAULT_TABLE)
    assert tab.sha256.startswith(X.TABLE_SHA256_PREFIX)
    with np.load(PASS + "/C0/e3_logits.npz", allow_pickle=False) as Z:
        tok, L = [str(t) for t in Z["token"]], Z["logits"]
    tix = np.array([tab.pos[t] for t in tok])

    class W:
        token = np.array(tok)

    with np.load(X.DEFAULT_TABLE, allow_pickle=False) as T:
        p6 = R.pick_logits(T["logits"])[tix]
    rows = X.changed_rows(tab, tix, W, R.pick_logits(L), p6)
    ref = C.build(X.DEFAULT_TABLE, PASS + "/C0/e3_logits.npz")
    assert len(rows) == len(ref) == 5998
    for (t, v, sub, s) in ref:
        mv, ms, msub = rows[t]
        assert mv == v and ms == s and [float(x) for x in msub] == sub


@needs_data
def test_control_equals_batch1_verdict(tmp_path):
    rc, j = _read(tmp_path, [f"C0={PASS}/C0", f"C0r={PASS}/C0r", f"S1E={PASS}/S1E"], ["--control-verdict", VERDICT])
    c = j["control"]
    assert rc == 0 and c["EQUAL"] and c["n_mismatch"] == 0 and c["n_leaves_compared"] > 100
    assert j["runs"]["C0"]["E3_pdms"]["point"] == pytest.approx(91.229, abs=5e-4)
    assert j["runs"]["C0r"]["E3_pdms_vs_P6"]["lo"] == pytest.approx(0.1971, abs=5e-5) and j["runs"]["S1E"]["E3_pdms_vs_P6"]["point"] == pytest.approx(0.0744, abs=5e-5)
    assert j.get("stamp_partial") == "PARTIAL_NOT_THE_CARD_READ" and j["card_complete"] is False


@needs_data
def test_deliberate_regression_shuffled_logits_fail_the_control(tmp_path):
    d = Path(tmp_path) / "C0shuf"
    d.mkdir(parents=True)
    with np.load(PASS + "/C0/e3_logits.npz", allow_pickle=False) as Z:
        arrs = {k: Z[k] for k in Z.files}
    perm = np.random.default_rng(0).permutation(len(arrs["logits"]))
    arrs["logits"] = arrs["logits"][perm]                                  # one run's logits shuffled across tokens: proposals still bit-identical, picks wrong
    np.savez(d / "e3_logits.npz", **arrs)
    rc, j = _read(tmp_path, [f"C0={d}", f"C0r={PASS}/C0r", f"S1E={PASS}/S1E"], ["--control-verdict", VERDICT])
    c = j["control"]
    assert c["EQUAL"] is False and c["n_mismatch"] > 0
    assert any(m[0].startswith("C0") for m in c["mismatches"])
    import shutil
    shutil.rmtree(d)


@needs_data
def test_recipe_with_one_run_everywhere_is_refused_df0(tmp_path):
    rc, j = _read(tmp_path, [f"C0={PASS}/C0", f"S1E={PASS}/S1E"])
    assert rc == 2 and j["status"] == "REFUSED" and "df = 0" in j["why"] and "outcome" not in j


@needs_data
def test_duplicate_arm_refused_without_smoke(tmp_path):
    rc, j = _read(tmp_path, [f"C0={PASS}/C0", f"C0r={PASS}/C0r", f"C0r2={PASS}/C0r"])
    assert rc == 2 and j["status"] == "REFUSED" and "bit-identical" in j["why"]


@needs_data
def test_unknown_run_name_refused(tmp_path):
    rc, j = _read(tmp_path, [f"C0={PASS}/C0", f"C7={PASS}/C0r"])
    assert rc == 2 and j["status"] == "REFUSED"


def test_smoke_path_rules(tmp_path):
    with pytest.raises(SystemExit, match="smoke"):
        X.main(["read", "--run", f"C0={PASS}/C0", "--smoke-standin", "--no-ledger", "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit, match="smoke folder"):
        X.main(["read", "--run", f"C0={PASS}/C0", "--no-ledger", "--out", X.SMOKE_ROOT + "/y"])
    with pytest.raises(SystemExit, match="no-ledger"):
        X.main(["read", "--run", f"C0={PASS}/C0", "--smoke-standin", "--out", X.SMOKE_ROOT + "/y"])


@needs_data
def test_ledger_keys_one_per_run_and_recipe_mean(tmp_path):
    led = Path(tmp_path) / "ledger.json"
    argv = ["read", "--ledger", str(led), "--out", str(Path(tmp_path) / "L")] + [x for r in (f"C0={PASS}/C0", f"C0r={PASS}/C0r", f"S1E={PASS}/S1E") for x in ("--run", r)]
    assert X.main(argv) == 0
    d = json.load(open(led, encoding="utf-8"))
    keys = [r["key"] for r in d["reads"]]
    assert len(keys) == 3 + 2 and sum(k.startswith("REP:") and k.endswith(":E3") for k in keys) == 5 and sum(":mean:" in k for k in keys) == 2
    assert X.main(argv) == 0                                               # the same read again is not a new read
    assert len(json.load(open(led, encoding="utf-8"))["reads"]) == 5
