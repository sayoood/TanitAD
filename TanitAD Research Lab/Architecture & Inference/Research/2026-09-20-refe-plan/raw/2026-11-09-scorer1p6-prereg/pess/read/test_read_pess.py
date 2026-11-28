"""Tests for read_pess.py (CPU only; synthetic data, no navtest truth is read).

    CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_read_pess.py --basetemp F:/tanitad_scratch/pess_pytest/tmp
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
import read_pess as P  # noqa: E402

R = P.R
LEDGER = P.DEFAULT_LEDGER
CARD_TEXT = open(P.CARD, encoding="utf-8").read()


def rng_logits(n_members, N=5, seed=0):
    g = np.random.default_rng(seed)
    return [g.normal(size=(N, 64, 6)).astype(np.float32) for _ in range(n_members)]


# ------------------------------------------------------------------------------------------------ the card
def test_card_sha_is_frozen_value():
    assert hashlib.sha256(open(P.CARD, "rb").read()).hexdigest() == P.CARD_SHA256
    assert P.card_ok()[0]


def test_bar_constants_match_card_text():
    assert "0.4971" in CARD_TEXT and P.P1_FLOOR == 0.4971
    assert "s_p 0.1791, df 4, t 2.776" in CARD_TEXT and (P.P1_S_P, P.P1_T) == (0.1791, 2.776)
    assert abs(P.P1_S_P * P.P1_T - P.P1_FLOOR) < 1e-4
    assert "CI-hi < +0.10" in CARD_TEXT and P.G4_BAR == 0.10
    assert R.G1_BAR == -0.5 and R.G2_BAR == 0.5 and R.G3_BAR == 0.10
    assert "seed 20260927" in CARD_TEXT and R.SEED == 20260927 and R.B_DRAWS == 10000
    assert "minimum" in CARD_TEXT and "mean minus 1 x SD (ddof 0)" in CARD_TEXT
    assert P.PESSIMISTIC_HEADS == (0, 1, 3) and P.HEAD["NC"] == 0 and P.HEAD["DAC"] == 1 and P.HEAD["TTC"] == 3 and P.HEAD["EP"] == 2


def test_sha_mismatch_refusal(tmp_path):
    bad = tmp_path / "card.md"
    bad.write_text(CARD_TEXT + "x", encoding="utf-8", newline="\n")
    ok, act = P.card_ok(str(bad))
    assert not ok and act != P.CARD_SHA256
    res = P.pess_read([("A", "nowhere")], "nowhere", card_path=str(bad))
    assert res["status"] == "READ STOPPED" and "REFUSED" in res["why"]
    # CLI refuses with rc 2 before touching any input
    assert P.main(["read", "--member", "A=nowhere", "--p6", "nowhere", "--out", str(tmp_path / "o"), "--card", str(bad), "--no-ledger"]) == 2
    assert not (tmp_path / "o").exists()


def test_missing_card_refused(tmp_path):
    ok, act = P.card_ok(str(tmp_path / "absent.md"))
    assert not ok and act is None


# ------------------------------------------------------------------------------------------------ combine rules on hand-made logits
def test_rules_on_handmade_logits():
    # one token, one proposal, three members, six heads (NC DAC EP TTC C DDC)
    a = np.zeros((1, 64, 6), np.float32)
    b = np.zeros((1, 64, 6), np.float32)
    c = np.zeros((1, 64, 6), np.float32)
    a[0, 0] = [1, 2, 3, 4, 5, 6]
    b[0, 0] = [3, 2, 9, 0, 5, 12]
    c[0, 0] = [5, 8, 0, 2, 5, 0]
    ens = P.combine([a, b, c], "ENS9")[0, 0]
    pes = P.combine([a, b, c], "PESS")[0, 0]
    lcb = P.combine([a, b, c], "LCB1")[0, 0]
    np.testing.assert_allclose(ens, [3, 4, 4, 2, 5, 6], rtol=1e-6)
    # min on NC, DAC, TTC; mean on EP, C, DDC
    np.testing.assert_allclose(pes, [1, 2, 4, 0, 5, 6], rtol=1e-6)
    # SD ddof 0: NC values 1,3,5 -> sqrt(8/3); DAC 2,2,8 -> sqrt(8); TTC 4,0,2 -> sqrt(8/3)
    np.testing.assert_allclose(lcb[[0, 1, 3]], [3 - np.sqrt(8 / 3), 4 - np.sqrt(8.0), 2 - np.sqrt(8 / 3)], rtol=1e-5)
    np.testing.assert_allclose(lcb[[2, 4, 5]], [4, 5, 6], rtol=1e-6)
    for x in (ens, pes, lcb):
        assert x.dtype == np.float32
    assert P.combine([a, b, c], "ENS9").dtype == np.float32


def test_ddof_is_zero_not_one():
    a = np.zeros((1, 64, 6), np.float32)
    b = np.full((1, 64, 6), 2.0, np.float32)
    lcb = P.combine([a, b], "LCB1")
    assert lcb[0, 0, 0] == pytest.approx(1.0 - 1.0)      # mean 1, SD(ddof 0) = 1 (ddof 1 would give 1.414)
    assert lcb[0, 0, 2] == pytest.approx(1.0)            # EP: mean only


def test_one_member_collapses_exactly():
    (m,) = rng_logits(1)
    for rule in ("ENS9", "PESS", "LCB1", "PESS_MAX"):
        out = P.combine([m], rule)
        assert R.bits_equal(out, m), rule
    # and the same picks
    assert (R.pick_logits(P.combine([m], "PESS")) == R.pick_logits(m)).all()


def test_two_equal_members_collapse_exactly():
    (m,) = rng_logits(1, seed=3)
    for rule in ("ENS9", "PESS", "LCB1"):
        assert R.bits_equal(P.combine([m, m.copy()], rule), m), rule


def test_min_not_max_and_labelled_different_arm():
    ms = rng_logits(4, N=300, seed=7)
    c = P.labelfree_compare(ms, "PESS", "PESS_MAX")
    assert c["different_arms"] and c["sha256_a"] != c["sha256_b"]
    assert c["n_pick_differs"] > 0 and c["n_tokens"] == 300
    # MIN <= MEAN <= MAX on the pessimistic heads, elementwise
    mn, mx, mean = P.combine(ms, "PESS"), P.combine(ms, "PESS_MAX"), P.combine(ms, "ENS9")
    assert (mn[..., [0, 1, 3]] <= mean[..., [0, 1, 3]] + 1e-6).all() and (mx[..., [0, 1, 3]] >= mean[..., [0, 1, 3]] - 1e-6).all()
    # EP, C, DDC are mean for both
    assert R.bits_equal(mn[..., [2, 4, 5]], mean[..., [2, 4, 5]]) and R.bits_equal(mx[..., [2, 4, 5]], mean[..., [2, 4, 5]])


def test_control_c3_pass_and_fail_paths():
    ms = rng_logits(8, N=200, seed=11)
    assert P.control_c3(ms)["PASS"]
    # a single member: MAX == MIN, the control must read FAIL (0 % differ, sha equal)
    c1 = P.control_c3(ms[:1])
    assert c1["share_pick_differs"] == 0.0 and not c1["different_arms"] and not c1["PASS"]


def test_ties_pick_first_index_and_min_creates_ties():
    # two members whose per-proposal minimum ties two proposals exactly: the pick is the lower index
    a = np.zeros((1, 64, 6), np.float32)
    b = np.zeros((1, 64, 6), np.float32)
    a[0, :, :] = -1.0
    b[0, :, :] = -1.0
    a[0, 5, :] = 2.0
    b[0, 9, :] = 2.0
    a[0, 9, :] = 2.0
    b[0, 5, :] = 2.0
    pes = P.combine([a, b], "PESS")
    pk = R.pick_logits(pes)
    assert pk[0] == 5            # proposals 5 and 9 carry identical combined logits; first index wins
    assert R.bits_equal(pes[0, 5], pes[0, 9])
    # ties are rule-deterministic: a repeated call gives the same pick and sha
    assert R.pick_logits(P.combine([a, b], "PESS"))[0] == 5
    assert P.sha256_array(pes) == P.sha256_array(P.combine([a, b], "PESS"))


def test_pessimism_can_change_the_pick():
    # member a loves proposal 0 on NC/DAC/TTC; member b hates it: the mean keeps 0, the minimum does not
    a = np.zeros((1, 64, 6), np.float32)
    b = np.zeros((1, 64, 6), np.float32)
    a[0, 0, [0, 1, 3]] = 8.0
    b[0, 0, [0, 1, 3]] = -2.0
    a[0, 1, [0, 1, 3]] = 1.0
    b[0, 1, [0, 1, 3]] = 1.0
    a[0, :, [2, 4, 5]] = 1.0
    b[0, :, [2, 4, 5]] = 1.0
    pe, pp = R.pick_logits(P.combine([a, b], "ENS9")), R.pick_logits(P.combine([a, b], "PESS"))
    assert pp[0] == 1 and pe[0] != 1


def test_combine_rejects_bad_input():
    with pytest.raises(ValueError):
        P.combine([], "PESS")
    with pytest.raises(ValueError):
        P.combine([np.zeros((1, 64, 6), np.float32), np.zeros((2, 64, 6), np.float32)], "PESS")
    with pytest.raises(ValueError):
        P.combine(rng_logits(2), "NOPE")


# ------------------------------------------------------------------------------------------------ bars and outcome table
def test_bar_functions_strict():
    assert P.p1_clears(0.4972, 0.01) is True
    assert P.p1_clears(0.4971, 0.01) is False        # d > 0.4971 strict
    assert P.p1_clears(0.9, 0.0) is False            # CI-lo > 0 strict
    assert P.p1_clears(0.9, -0.1) is False
    assert P.p1_clears(None, 0.1) is None
    assert P.p2_clears(1e-9) is True and P.p2_clears(0.0) is False and P.p2_clears(None) is None
    assert P.m_clears(-1e-9) is True and P.m_clears(0.0) is False and P.m_clears(None) is None


G_OK = {"G1": True, "G2": True, "G3": True, "G4": True}


def test_outcome_every_branch():
    d = P.decide_outcome
    # PE1: P1, P2, M and all guards
    o = d(True, True, True, G_OK)
    assert o["primary"] == "PE1" and o["adopted"] and o["matches"] == ["PE1"]
    # PE2: P2 holds, P1 fails (M free)
    o = d(False, True, True, G_OK)
    assert o["primary"] == "PE2" and not o["adopted"] and o["matches"] == ["PE2"]
    o = d(False, True, False, G_OK)
    assert o["matches"] == ["PE2"]
    # PE3: M holds, P2 fails
    o = d(False, False, True, G_OK)
    assert o["primary"] == "PE3" and o["matches"] == ["PE3"]
    o = d(True, False, True, G_OK)                    # P1 holding without P2 is still PE3
    assert o["primary"] == "PE3"
    # PE4: neither
    o = d(False, False, False, G_OK)
    assert o["primary"] == "PE4" and o["matches"] == ["PE4"]
    o = d(True, False, False, G_OK)
    assert o["primary"] == "PE4"
    # Unnamed: P1, P2 hold, M fails -> no label
    o = d(True, True, False, G_OK)
    assert o["primary"] == "Unnamed" and o["matches"] == ["Unnamed"] and not o["adopted"]
    # Unnamed: everything but one guard
    bad = dict(G_OK, G4=False)
    o = d(True, True, True, bad)
    assert o["primary"] == "Unnamed" and o["failed_guards"] == ["G4"] and not o["adopted"]
    # a failed guard does not stop PE2 / PE3 / PE4 labels
    assert d(False, True, True, bad)["primary"] == "PE2"
    # undefined inputs -> Unnamed with a reason
    for args in ((None, True, True, G_OK), (True, None, True, G_OK), (True, True, None, G_OK), (True, True, True, dict(G_OK, G2=None)), (True, True, True, {})):
        o = d(*args)
        assert o["primary"] == "Unnamed" and "why" in o
    # all five labels reachable
    seen = {d(*a)["primary"] for a in [(True, True, True, G_OK), (False, True, False, G_OK), (False, False, True, G_OK), (False, False, False, G_OK), (True, True, False, G_OK)]}
    assert seen == {"PE1", "PE2", "PE3", "PE4", "Unnamed"}


def test_outcome_names_and_next_levers_in_card():
    for k, name in P.PE_NAMES.items():
        if k != "Unnamed":
            assert f"**{k} {name}**" in CARD_TEXT
    assert set(P.PE_NEXT) == {"PE1", "PE2", "PE3", "PE4", "Unnamed"}
    assert P.PE_ORDER == ("PE1", "PE2", "PE3", "PE4")


# ------------------------------------------------------------------------------------------------ truth lookup
def test_truth_lookup_refuses_invalid(tmp_path):
    N = 4
    pdms = np.random.default_rng(0).uniform(size=(N, 64))
    sub = np.random.default_rng(1).uniform(size=(N, 64, 6))
    valid = np.ones((N, 64), bool)
    valid[2, 7] = False
    f = tmp_path / "t.npz"
    np.savez(f, token=np.array(["a", "b", "c", "d"]), log=np.array(["l"] * N), pdms=pdms, sub=sub, valid=valid, pick=np.zeros(N, np.int64), logits=np.zeros((N, 64, 6), np.float32))
    T = P.Truth(str(f))
    d, s = T.lookup(np.array([1, 2, 3, 4]))
    assert d[1] == pdms[1, 2] and s.shape == (N, 6)
    with pytest.raises(SystemExit):
        T.lookup(np.array([1, 2, 7, 4]))


# ------------------------------------------------------------------------------------------------ the ledger and gpu are never touched by tests
def test_ledger_file_untouched_and_no_cuda():
    before = hashlib.sha256(open(LEDGER, "rb").read()).hexdigest()
    P.combine(rng_logits(3), "PESS")
    assert hashlib.sha256(open(LEDGER, "rb").read()).hexdigest() == before
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"


def test_pess_read_writes_no_ledger_in_control_mode_signature():
    src = open(HERE / "read_pess.py", encoding="utf-8").read()
    assert "if ledger_path and not control:" in src            # control runs never reach the ledger
    assert "ledger = None if (a.no_ledger or a.control) else a.ledger" in src


def test_lf_line_endings_in_package():
    for f in ("read_pess.py", "test_read_pess.py"):
        assert b"\r" not in open(HERE / f, "rb").read(), f
