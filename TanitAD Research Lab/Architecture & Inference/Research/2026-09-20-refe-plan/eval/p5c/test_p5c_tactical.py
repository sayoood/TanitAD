"""Tests for p5c_tactical.py (B12). The wrapper must CONSUME the existing tools' output (identity with command_adherence.run_read on the same seams), never pool
families, and say NOT COMPUTED with reason and n where it cannot compute. RED arms: pooled score, absent family, STRATEGIC not paused, a goal row without a
reason, mismatched token sets (INVALID, never numbers), --tokens-from without --subset-ok.
Integration tests use the banked P6 / u400 / deployed seams and the W3 export; they skip when those are not on this machine.
Run: CUDA_VISIBLE_DEVICES=-1 python -m pytest test_p5c_tactical.py -q"""
import copy
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import p5c_tactical as T   # noqa: E402

PK = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_DIR = os.path.join(PK, "raw", "p2_r2_instrument", "baseline")
SEAMS = {"C1": os.path.join(BASE_DIR, "p6_navtest_full_seam.npz"), "C0": os.path.join(BASE_DIR, "u400_navtest_full_seam.npz"),
         "base": os.path.join(BASE_DIR, "deployed_navtest_full_seam.npz")}
NAVSIM = os.environ.get("TANITAD_NAVSIM_ROOT", "F:/Archive/devbox-C/navsim")
EXPORT = NAVSIM + "/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
REPO = os.environ.get("TANITAD_REPO", "E:/Projects/TanitAD")
HAVE = all(os.path.exists(p) for p in list(SEAMS.values()) + [EXPORT, REPO + "/stack/tanitad/refs/refc_tactical.py"])
needs = pytest.mark.skipif(not HAVE, reason="banked seams / W3 export / repo labeller not on this machine")


def good_block():
    b = T.not_computed("synthetic", 12146)
    b["command_adherence"] = {"status": "COMPUTED", "n_tokens": 12146}
    return b


# ----------------------------------------------------------------------------------------------- the block contract
def test_not_computed_block_is_valid_and_names_reason_and_n():
    b = T.not_computed("no tactical.json supplied", 500)
    assert T.validate_block(b) == []
    assert b["command_adherence"]["reason"] and b["command_adherence"]["n"] == 500
    assert b["goal_setting"]["status"] == "NOT COMPUTABLE" and "no tactical goal head" in b["goal_setting"]["reason"]
    assert b["strategic"]["status"] == "PAUSED"


def test_good_block_validates():
    assert T.validate_block(good_block()) == []


def test_RED_pooled_score_refused():
    b = good_block()
    b["pooled_score"] = 0.9
    assert T.validate_block(b)
    b = good_block()
    b["pooled"] = True
    assert T.validate_block(b)


@pytest.mark.parametrize("k", list(T.FAMILY_KEYS) + ["strategic"])
def test_RED_absent_family_refused(k):
    b = good_block()
    del b[k]
    assert T.validate_block(b)


def test_RED_strategic_never_a_pass():
    b = good_block()
    b["strategic"] = {"status": "PASS"}
    assert any("PAUSED" in x for x in T.validate_block(b))


def test_RED_goal_setting_needs_reason_and_n():
    b = good_block()
    b["goal_setting"] = {"status": "NOT COMPUTABLE"}
    assert T.validate_block(b)
    b = good_block()
    b["selection_quality"]["C1"] = {"status": "NOT COMPUTED"}
    assert T.validate_block(b)


def test_not_a_tactical_block():
    assert T.validate_block({}) and T.validate_block(None)


def test_selection_block_without_tables_says_not_computed_with_n():
    s = T.selection_block({}, 12146)
    assert all(v["status"] == "NOT COMPUTED" and v["n"] == 12146 and "64" in v["reason"] for v in s.values())


# ----------------------------------------------------------------------------------------------- integration on the banked seams
@needs
def test_subset_without_subset_ok_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="subset-ok"):
        T.build(SEAMS, tokens=["x"], subset_ok=False, workdir=str(tmp_path))


@needs
def test_reach_wrapper_equals_the_existing_tool_on_the_same_seams(tmp_path):
    """the tactical block's numbers ARE command_adherence.run_read's: same seams, same tokens, same estimator."""
    z = np.load(SEAMS["C1"], allow_pickle=False)
    toks = sorted(str(t) for t in z["token"])[::97][:120]
    blk = T.build(SEAMS, subset_ok=True, tokens=toks, workdir=str(tmp_path), n_boot=500)
    assert T.validate_block(blk) == [] and blk["rehearsal"] is True and blk["command_adherence"]["registered_read"] is False
    ca = blk["command_adherence"]
    assert ca["status"] == "COMPUTED" and ca["n_tokens"] == 120 and all(g["ok"] for g in ca["gates"].values())
    import command_adherence as CA
    sub = {k: T.subset_seam(v, toks, str(tmp_path / f"d_{k}.npz")) for k, v in SEAMS.items()}
    direct = CA.run_read({"name": "C1", "seam": sub["C1"]}, [{"role": "C0", "name": "C0", "seam": sub["C0"]}, {"role": "base", "name": "base", "seam": sub["base"]}],
                         exp=CA.load_export(), expect_full=False, n_boot=500)
    for role in ("C1", "C0", "base"):
        want = direct["arms"][role]["executed_plan_A7"]["turn_active"]["accuracy"] if role in direct["arms"] else None
        if want is not None:
            assert ca["arms"][role]["turn_active"]["accuracy"] == want
    assert ca["paired_C1_minus_x"]["C1-C0"]["turn_active"]["accuracy_delta"] == direct["paired_vs_comparators"]["C1_minus_C0_C0"]["turn_active"]["accuracy_delta"]
    assert set(ca["paired_C1_minus_x"]) == {"C1-C0", "C1-base", "C0-base"}
    assert ca["random_proposal_floor"]["status"] == "NOT COMPUTED" and ca["random_proposal_floor"]["n"] == 120 and ca["incomplete"]
    assert blk["selection_quality"]["C1"]["status"] == "NOT COMPUTED" and blk["goal_setting"]["n"] == 120 and blk["strategic"]["status"] == "PAUSED"


@needs
def test_RED_mismatched_token_sets_give_INVALID_never_numbers(tmp_path):
    import command_adherence as CA
    z = np.load(SEAMS["C1"], allow_pickle=False)
    toks = sorted(str(t) for t in z["token"])[::211][:40]
    a = T.subset_seam(SEAMS["C1"], toks, str(tmp_path / "a.npz"))
    b = T.subset_seam(SEAMS["C0"], toks[:-1], str(tmp_path / "b.npz"))
    r = CA.run_read({"name": "C1", "seam": a}, [{"role": "C0", "name": "C0", "seam": b}], exp=CA.load_export(), expect_full=False, n_boot=50)
    blk = T.compact_adherence(r, None, {"C1": "C1"}, 40, registered=False)
    assert blk["status"] == "INVALID" and "arms" not in blk and blk["invalid"]


@needs
def test_RED_registered_read_enforces_the_full_class_counts(tmp_path):
    """without --subset-ok the ruling's 12,146 / 136-log counts are enforced: a 3-token read is INVALID (G_COUNTS), not a number."""
    import command_adherence as CA
    z = np.load(SEAMS["C1"], allow_pickle=False)
    toks = sorted(str(t) for t in z["token"])[:30]
    p = {k: T.subset_seam(v, toks, str(tmp_path / f"{k}.npz")) for k, v in SEAMS.items()}
    r = CA.run_read({"name": "C1", "seam": p["C1"]}, [{"role": "C0", "name": "C0", "seam": p["C0"]}], exp=CA.load_export(), expect_full=True, n_boot=50)
    blk = T.compact_adherence(r, None, {"C1": "C1", "C0": "C0"}, 30, registered=True)
    assert blk["status"] == "INVALID" and any("12,146" in x or "counts" in x for x in blk["invalid"])
