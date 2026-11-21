"""Tests for read_div.py (CPU only; synthetic data, no navtest truth is read; control C3 of the task).

    CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_read_div.py --basetemp F:/tanitad_scratch/divread_build/tmp
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import read_div as D  # noqa: E402

R, RP = D.R, D.RP
CARD_TEXT = open(D.CARD, encoding="utf-8").read()
OK = {"G1": True, "G2": True, "G3": True, "G4": True}


def mk_B3(n_logs=40, per=30):
    log = np.repeat(np.arange(n_logs), per).astype(str)
    return R.BootSet(log), n_logs * per


# ------------------------------------------------------------------------------------------------ the card
def test_card_sha_is_frozen_value():
    assert hashlib.sha256(open(D.CARD, "rb").read()).hexdigest() == D.CARD_SHA256 == "e5fb2f8e4123fcfe3a599b0cc71537bab187716c6a5d3e46735ce094bacf6fac"
    assert D.card_ok()[0]


def test_bar_constants_match_card_text():
    assert "d > 0.4971" in CARD_TEXT and D.P1_FLOOR == 0.4971
    assert "seed 20260927" in CARD_TEXT and R.SEED == 20260927 and R.B_DRAWS == 10000
    assert "8912f1d0" in CARD_TEXT and D.BANKED_PESS8_SHA256_PREFIX.startswith("8912f1d0")
    assert "PESS-DV** = PESS over {DV0, DV1}" in CARD_TEXT and D.ARMS["PESS-DV"] == ["DV0", "DV1"]
    assert "PESS-SEED** = PESS over {C0, C0r}" in CARD_TEXT and D.ARMS["PESS-SEED"] == ["C0", "C0r"]
    assert "PESS-10** = PESS over the eight PESS-read members plus DV0 and DV1" in CARD_TEXT and D.ARMS["PESS-10"] == D.EIGHT + ["DV0", "DV1"] and len(D.TEN) == 10
    assert "K: + 5" in CARD_TEXT and len(D.LEDGER_ARMS) == 5 and set(D.LEDGER_ARMS) == {"DV0", "DV1", "PESS-DV", "PESS-SEED", "PESS-10"}
    assert "CI-hi < 0" in CARD_TEXT and "CI-lo > 0" in CARD_TEXT
    for k in ("DV1 ADOPT", "DV2 DIVERSITY HELPS, TOO SMALL", "DV3 NO DIVERSITY GAIN", "Unnamed"):
        assert k in CARD_TEXT
    assert D.EIGHT == json.load(open(D.PESS_BANKED_JSON, encoding="utf-8"))["members"]


def test_member_dirs_default_to_the_documented_ones():
    assert D.MEMBER_DIRS["DV0"] == "F:/tanitad_scratch/p6arms/pass_div/DV0" and D.MEMBER_DIRS["DV1"] == "F:/tanitad_scratch/p6arms/pass_div/DV1"


# ------------------------------------------------------------------------------------------------ C3: every outcome branch
def test_dv1_adopt():
    o = D.decide_outcome(True, True, True, OK)
    assert o["primary"] == "DV1" and o["adopted"] and o["matches"] == ["DV1"] and o["name"] == "ADOPT"


def test_dv1_needs_both_q2_parts_and_all_guards():
    assert D.decide_outcome(True, True, False, OK)["primary"] == "DV2"          # q2b fails -> Q2 fails
    assert D.decide_outcome(True, False, True, OK)["primary"] == "DV2"          # q2a fails
    o = D.decide_outcome(True, True, True, {**OK, "G3": False})
    assert o["primary"] == "Unnamed" and o["failed_guards"] == ["G3"] and not o["adopted"]


def test_dv2_diversity_helps_too_small():
    o = D.decide_outcome(True, False, False, OK)
    assert o["primary"] == "DV2" and o["matches"] == ["DV2"] and not o["adopted"] and "Decision A" in o["next_levers"]["DV2"]
    assert D.decide_outcome(True, False, True, {**OK, "G1": False})["primary"] == "DV2"      # a failed guard does not hide DV2


def test_dv3_no_diversity_gain():
    o = D.decide_outcome(False, False, False, OK)
    assert o["primary"] == "DV3" and o["matches"] == ["DV3"] and o["name"] == "NO DIVERSITY GAIN"


def test_overlap_dv1_and_dv3_primary_dv1_listed_both():
    o = D.decide_outcome(False, True, True, OK)
    assert o["matches"] == ["DV1", "DV3"] and o["primary"] == "DV1" and o["adopted"]


def test_unnamed_branches():
    assert D.decide_outcome(True, True, True, {**OK, "G2": False})["primary"] == "Unnamed"
    for args in ((None, True, True, OK), (True, None, True, OK), (True, True, None, OK), (True, True, True, {**OK, "G4": None}), (True, True, True, {})):
        o = D.decide_outcome(*args)
        assert o["primary"] == "Unnamed" and o["matches"] == ["Unnamed"] and not o["adopted"]


def test_every_label_is_reachable():
    seen = set()
    for q1 in (True, False):
        for a in (True, False):
            for b in (True, False):
                for g in (True, False):
                    seen.add(D.decide_outcome(q1, a, b, {**OK, "G1": g})["primary"])
    assert seen == {"DV1", "DV2", "DV3", "Unnamed"}


# ------------------------------------------------------------------------------------------------ the bar functions
def test_bar_predicates_are_strict():
    assert D.q1_clears(1e-9) and not D.q1_clears(0.0) and D.q1_clears(None) is None
    assert D.q2a_clears(0.4972, 0.01) and not D.q2a_clears(0.4971, 0.01) and not D.q2a_clears(1.0, 0.0) and D.q2a_clears(None, 1) is None
    assert D.q2b_clears(0.01) and not D.q2b_clears(0.0)
    assert D.m_clears(-1e-9) and not D.m_clears(0.0) and D.m_clears(None) is None


def test_bars_from_arrays_synthetic_signs():
    B3, N = mk_B3()
    base = np.random.default_rng(1).uniform(0.5, 1.0, N)
    pd = {"P6": base, "PESS-SEED": base.copy(), "PESS-8": base.copy(), "PESS-DV": np.clip(base + 0.02, 0, 1), "PESS-10": np.clip(base + 0.02, 0, 1)}
    b = D.bars_from_arrays(B3, pd)
    assert b["Q1"]["clears"] is True and b["Q1"]["point"] > 0
    assert b["Q2a_vs_P6"]["point"] > 0 and b["Q2b_vs_PESS8"]["clears"] is True
    b2 = D.bars_from_arrays(B3, {k: base.copy() for k in pd})
    assert b2["Q1"]["clears"] is False and b2["Q2a_vs_P6"]["clears"] is False and b2["M"]["clears"] is False and b2["Q1"]["point"] == 0.0
    zs = base.copy()
    zs[::3] = 0.0
    zd = zs.copy()
    zd[::3] = 0.5
    b3 = D.bars_from_arrays(B3, {**{k: base.copy() for k in pd}, "PESS-SEED": zs, "PESS-DV": zd})
    assert b3["M"]["clears"] is True and b3["M"]["point"] < 0 and b3["M"]["n_pick_pdms_zero_PESS-DV"] == 0


def test_synthetic_end_to_end_outcomes_via_bars():
    B3, N = mk_B3()
    base = np.random.default_rng(2).uniform(0.5, 1.0, N)

    def outcome(dv, seed, p10, p8, guards=OK):
        b = D.bars_from_arrays(B3, {"P6": base, "PESS-DV": dv, "PESS-SEED": seed, "PESS-10": p10, "PESS-8": p8})
        return D.decide_outcome(b["Q1"]["clears"], b["Q2a_vs_P6"]["clears"], b["Q2b_vs_PESS8"]["clears"], guards)["primary"]
    big, mid, tiny = np.clip(base + 0.05, 0, 1), np.clip(base + 0.02, 0, 1), np.clip(base + 0.003, 0, 1)
    assert outcome(mid, base, big, mid) == "DV1"                  # Q1 holds, +5 points over P6 and over PESS-8
    assert outcome(mid, base, tiny, tiny) == "DV2"                # Q1 holds, Q2 point below 0.4971
    assert outcome(base, mid, tiny, tiny) == "DV3"                # Q1 fails
    assert outcome(mid, base, big, mid, {**OK, "G4": False}) == "Unnamed"


# ------------------------------------------------------------------------------------------------ arms and the combine (read_pess rules, tiny logits)
def rng_logits(names, N=5, seed=0):
    g = np.random.default_rng(seed)
    return {k: g.normal(size=(N, 64, 6)).astype(np.float32) for k in names}


def test_pess_dv_is_min_on_nc_dac_ttc_mean_elsewhere_and_single_is_identity():
    L = rng_logits(["DV0", "DV1", "C0", "C0r"])
    c = D.combine_arm(L, D.ARMS["PESS-DV"])
    S = np.stack([L["DV0"], L["DV1"]])
    for h in (0, 1, 3):
        assert np.array_equal(c[..., h], S[..., h].min(0))
    for h in (2, 4, 5):
        assert np.array_equal(c[..., h], np.mean(S[..., h], axis=0, dtype=np.float32))
    assert R.bits_equal(D.combine_arm(L, ["DV0"]), L["DV0"])         # DV0 as an arm is the member's own logits
    assert D.sha256_array(D.combine_arm(L, D.ARMS["PESS-SEED"])) != D.sha256_array(c)


def test_member_order_is_pinned():
    assert D.TEN[:8] == D.EIGHT and D.TEN[8:] == ["DV0", "DV1"]


def test_controls_may_only_score_already_read_arms():
    for arms in D.CONTROL_ARMS.values():
        assert set(arms) <= {"PESS-8", "S1ZSM"}
    assert D.CONTROL_MEMBERS["c1"] == D.EIGHT and D.CONTROL_MEMBERS["c2"] == ["S1ZSM"]


# ------------------------------------------------------------------------------------------------ refusals
def test_div_read_refuses_a_changed_card(tmp_path):
    p = tmp_path / "card.md"
    p.write_text(CARD_TEXT + "\nedit\n", encoding="utf-8")
    res = D.div_read([(n, "x") for n in D.TEN], "x", card_path=str(p))
    assert res["status"] == "READ STOPPED" and "REFUSED" in res["why"] and "arms" not in res
    assert D.main(["read", "--card", str(p), "--out", str(tmp_path / "o")]) == 2
    assert not (tmp_path / "o").exists()


def test_main_refuses_missing_card(tmp_path):
    assert D.main(["read", "--card", str(tmp_path / "nope.md"), "--out", str(tmp_path / "o")]) == 2


def test_div_read_refuses_wrong_member_set_before_loading():
    assert "members must be exactly" in D.div_read([("C0", "x"), ("C0r", "x")], "x")["why"]
    assert D.div_read([(n, "x") for n in D.TEN[::-1]], "x")["status"] == "READ STOPPED"      # order is part of the sha
    assert D.div_read([(n, "x") for n in D.TEN], "x", control="c1")["status"] == "READ STOPPED"


def test_main_refuses_missing_member_logits(tmp_path):
    assert D.main(["read", "--member", f"DV0={tmp_path}/nodir", "--out", str(tmp_path / "o")]) == 2


def test_markdown_renders_a_stopped_read():
    res = {"card": {"path": "p", "sha256_actual": "a", "sha256_expected": "b"}, "estimator": "e", "n_members": 0, "members": [], "status": "READ STOPPED", "why": "w"}
    assert "No verdict is written." in D.to_markdown(res)


def test_ledger_is_written_in_one_place_and_never_in_a_control():
    src = open(HERE / "read_div.py", encoding="utf-8").read()
    assert src.count("led.add(") == 1
    assert "ledger = None if (a.no_ledger or a.control) else a.ledger" in src
    assert b"\r" not in open(HERE / "read_div.py", "rb").read()


# ------------------------------------------------------------------------------------------------ post_div.py (never run here: pure pieces only)
def test_post_div_defaults_and_zero_byte_cleanup(tmp_path, monkeypatch):
    import post_div as P
    a = P.parse(["2026-11-11T08:00:00"])
    assert a.chain_log == "/workspace/data/refe_div/chain_div.log" and a.done_pattern == "ZZDIV_CHAIN_DONE" and a.fail_pattern == ["ZZDIV_CHAIN_FAIL", "REFUSED"]
    cmd = P.armpass_cmd({"DV0": "a.pt", "DV1": "b.pt"})
    assert cmd[-4:] == ["--arm", "DV0=a.pt", "--arm", "DV1=b.pt"] and "devbox_gpu.lock" not in open(P.__file__, encoding="utf-8").read().replace("never touches the lock file", "")
    assert P.read_cmd()[-2:] == ["--out", P.READ_OUT] and P.read_cmd(True)[-1] == "--no-ledger"
    monkeypatch.setattr(P, "LOG", str(tmp_path / "log.txt"))
    d = tmp_path / "_progress"
    d.mkdir()
    (d / "part_000003.npz.tmp.npz").write_bytes(b"")
    (d / "part_000004.npz.tmp.npz").write_bytes(b"x")
    (d / "part_000005.npz").write_bytes(b"")                      # a finished part is never touched
    removed = P.clean_zero_byte_tmp(str(d))
    assert [os.path.basename(r) for r in removed] == ["part_000003.npz.tmp.npz"]
    assert sorted(os.listdir(d)) == ["part_000004.npz.tmp.npz", "part_000005.npz"]
    assert "removed 0-byte leftover" in open(P.LOG, encoding="utf-8").read()
    assert "ZZDIV_CHAIN_DONE" in P.chain_query("/x.log", "2026-11-11T08:00:00", "ZZDIV_CHAIN_DONE", ["REFUSED"])


def test_post_div_rejects_a_bad_t0():
    import post_div as P
    import pytest
    with pytest.raises(SystemExit):
        P.parse(["tomorrow"])
