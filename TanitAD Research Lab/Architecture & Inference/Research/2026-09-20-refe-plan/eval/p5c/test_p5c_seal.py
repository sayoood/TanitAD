"""Tests for p5c_seal.py (B11): SEALED.json and the K-ledger entry writer / validator. Controls read known values (the real ledger validates at K = 50;
entries are n = K+1, K+2). RED arms: a tampered sealed file, a duplicate / gapped / out-of-order ledger, a wrong n, a wrong sha, a missing arm entry,
an in-place merge, rows older than the seal each must FAIL / REFUSE.
Run: CUDA_VISIBLE_DEVICES=-1 python -m pytest test_p5c_seal.py -q"""
import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5c_seal as S   # noqa: E402

REAL_LEDGER = os.path.join(HERE, "..", "..", "raw", "2026-11-09-scorer1p6-prereg", "selectors", "ledger_K.json")
H = lambda c: c * 64   # noqa: E731
BASE, CARD = H("b"), H("c")


def sealed_doc(**kw):
    arms = {"C1": {"model_final_sha256": H("1"), "hash_source": "declared"}, "C0": {"model_final_sha256": H("2"), "hash_source": "declared"}}
    arms.update(kw.pop("extra_arms", {}))
    return S.build_sealed(arms, kw.get("card", CARD), kw.get("base", BASE), kw.get("utc", "2026-11-13T20:00:00Z"))


def ledger(k=3, t0=10):
    return {"card": "x", "reads": [{"key": f"K{i}:E3", "n": i, "time": f"2026-11-13T10:{t0 + i:02d}:00Z", "family": "F", "set": "E3", "smoke": False} for i in range(1, k + 1)], "seeded": "s"}


# ----------------------------------------------------------------------------------------------- SEALED.json
def test_CONTROL_good_seal_verifies():
    r = S.verify_sealed(sealed_doc())
    assert r["pass"] is True and r["control"] == "B11_sealed" and r["arms"] == {"C1": H("1"), "C0": H("2")}


def test_self_hash_is_the_hash_of_the_content():
    d = sealed_doc()
    assert d["sealed_sha256"] == hashlib.sha256(S.canon({k: v for k, v in d.items() if k != "sealed_sha256"})).hexdigest()


def test_RED_tampered_sha_is_refused():
    d = sealed_doc()
    d["arms"]["C1"]["model_final_sha256"] = H("9")
    r = S.verify_sealed(d)
    assert r["pass"] is False and any("edited after sealing" in p for p in r["problems"])


@pytest.mark.parametrize("field,val", [("card_sha256", H("d")), ("base_sha256", H("e")), ("sealed_utc", "2026-11-13T19:00:00Z"), ("set", "E2")])
def test_RED_any_field_edit_is_refused(field, val):
    d = sealed_doc()
    d[field] = val
    assert S.verify_sealed(d)["pass"] is False


def test_RED_edit_and_rehash_is_caught_by_the_file_recheck(tmp_path):
    f = tmp_path / "c1.pt"
    f.write_bytes(b"model-one")
    g = tmp_path / "c0.pt"
    g.write_bytes(b"model-two")
    arms = {"C1": {"model_final_sha256": S.sha256_file(str(f)), "hash_source": "file", "path": str(f)},
            "C0": {"model_final_sha256": S.sha256_file(str(g)), "hash_source": "file", "path": str(g)}}
    d = S.build_sealed(arms, CARD, BASE, "2026-11-13T20:00:00Z")
    assert S.verify_sealed(d, recheck_files=True)["pass"] is True
    f.write_bytes(b"model-one-retrained")                               # the checkpoint changed after the seal
    r = S.verify_sealed(d, recheck_files=True)
    assert r["pass"] is False and any("no longer hashes" in p for p in r["problems"])


def test_RED_declared_hash_cannot_be_rechecked():
    assert S.verify_sealed(sealed_doc(), recheck_files=True)["pass"] is False


def test_RED_missing_arm_same_sha_base_sha_and_bad_hex():
    d = sealed_doc()
    del d["arms"]["C0"]
    d["sealed_sha256"] = S.self_hash(d)
    assert S.verify_sealed(d)["pass"] is False                             # one arm sealed, not both
    same = sealed_doc(extra_arms={"C0": {"model_final_sha256": H("1"), "hash_source": "declared"}})
    assert S.verify_sealed(same)["pass"] is False
    assert S.verify_sealed(sealed_doc(base=H("1")))["pass"] is False       # C1 == base: it did not move
    bad = sealed_doc(extra_arms={"C0": {"model_final_sha256": "XYZ", "hash_source": "declared"}})
    assert S.verify_sealed(bad)["pass"] is False


def test_card_sha_must_match_the_file(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("frozen card")
    d = sealed_doc(card=S.sha256_file(str(card)))
    assert S.verify_sealed(d, str(card))["pass"] is True
    card.write_text("frozen card, edited")
    assert S.verify_sealed(d, str(card))["pass"] is False


def test_RED_rows_older_than_the_seal(tmp_path):
    d = sealed_doc(utc=S.utc_now())
    rows = tmp_path / "rows.jsonl"
    rows.write_text("{}")
    os.utime(rows, (1.0e9, 1.0e9))                                         # 2001: the read started before the seal
    r = S.verify_sealed(d, rows_paths=[str(rows)])
    assert r["pass"] is False and any("OLDER than the seal" in p for p in r["problems"])
    d2 = sealed_doc(utc="2001-01-01T00:00:00Z")
    assert S.verify_sealed(d2, rows_paths=[str(rows)])["pass"] is True


# ----------------------------------------------------------------------------------------------- the ledger
def test_CONTROL_real_ledger_validates_at_K_50():
    if not os.path.exists(REAL_LEDGER):
        pytest.skip("ledger_K.json not in this tree")
    r = S.validate_ledger(json.load(open(REAL_LEDGER, encoding="utf-8")))
    assert r["pass"] is True and r["K"] == 50, r


def test_RED_ledger_duplicate_gap_reorder_time():
    L = ledger()
    assert S.validate_ledger(L)["pass"] is True
    d = copy.deepcopy(L)
    d["reads"][2]["key"] = d["reads"][0]["key"]
    assert S.validate_ledger(d)["pass"] is False                           # duplicate key
    g = copy.deepcopy(L)
    g["reads"][2]["n"] = 4
    assert S.validate_ledger(g)["pass"] is False                           # gap
    o = copy.deepcopy(L)
    o["reads"][0], o["reads"][1] = o["reads"][1], o["reads"][0]
    assert S.validate_ledger(o)["pass"] is False                           # out of order n
    t = copy.deepcopy(L)
    t["reads"][2]["time"] = "2026-11-13T09:00:00Z"
    assert S.validate_ledger(t)["pass"] is False                           # time runs backwards
    m = copy.deepcopy(L)
    del m["reads"][1]["family"]
    assert S.validate_ledger(m)["pass"] is False


def test_entries_follow_the_cards_template_with_n_K_plus_1():
    L = ledger(5)
    e = S.make_entries(sealed_doc(), L, "2026-11-13T20:30:00Z")
    assert [x["n"] for x in e] == [6, 7] and [x["nominee"] for x in e] == ["C1", "C0"]
    assert e[0]["key"] == f"P5C:C1:{H('1')}:E3" and e[1]["key"] == f"P5C:C0:{H('2')}:E3"
    for x in e:
        assert x["family"] == "P5C" and x["set"] == "E3" and x["smoke"] is False and x["scorers"] == ["lqr_tracked", "tracker_transparent"]
        assert x["card_sha256"] == CARD and x["base_sha256"] == BASE and x["time"] == "2026-11-13T20:30:00Z"
    assert S.check_entries(L, e, sealed_doc())["pass"] is True


def test_entries_on_the_real_ledger_are_51_and_52():
    if not os.path.exists(REAL_LEDGER):
        pytest.skip("ledger_K.json not in this tree")
    L = json.load(open(REAL_LEDGER, encoding="utf-8"))
    e = S.make_entries(sealed_doc(), L, "2026-12-01T00:00:00Z")
    assert [x["n"] for x in e] == [51, 52]
    merged = copy.deepcopy(L)
    merged["reads"] += e
    assert S.validate_ledger(merged)["pass"] is True and len(L["reads"]) == 50     # the input was not touched


def test_RED_entries_refused():
    L = ledger()
    with pytest.raises(ValueError, match="duplicate"):
        L2 = copy.deepcopy(L)
        L2["reads"].append({"key": f"P5C:C1:{H('1')}:E3", "n": 4, "time": "2026-11-13T12:00:00Z", "family": "P5C", "set": "E3", "smoke": False})
        S.make_entries(sealed_doc(), L2, "2026-11-13T20:30:00Z")
    with pytest.raises(ValueError, match="SEALED"):
        d = sealed_doc()
        d["arms"]["C1"]["model_final_sha256"] = H("7")
        S.make_entries(d, L)
    with pytest.raises(ValueError, match="earlier"):
        S.make_entries(sealed_doc(), L, "2026-11-13T10:00:00Z")
    with pytest.raises(ValueError, match="C0r"):
        S.make_entries(sealed_doc(extra_arms={"C0r": {"model_final_sha256": H("3"), "hash_source": "declared"}}), L, "2026-11-13T20:30:00Z")
    e = S.make_entries(sealed_doc(extra_arms={"C0r": {"model_final_sha256": H("3"), "hash_source": "declared"}}), L, "2026-11-13T20:30:00Z", c0r_approved=True)
    assert [x["n"] for x in e] == [4, 5, 6]


def test_RED_check_entries_catches_wrong_n_sha_missing_arm_and_replay():
    L = ledger(4)
    good = S.make_entries(sealed_doc(), L, "2026-11-13T20:30:00Z")
    sd = sealed_doc()
    assert S.check_entries(L, good, sd)["pass"] is True
    b = copy.deepcopy(good)
    b[0]["n"] = 9
    assert S.check_entries(L, b, sd)["pass"] is False                      # not the next free n
    b = copy.deepcopy(good)
    b[1]["n"], b[0]["n"] = b[0]["n"], b[1]["n"]
    assert S.check_entries(L, b, sd)["pass"] is False                      # out of order
    b = copy.deepcopy(good)
    b[0]["key"] = f"P5C:C1:{H('8')}:E3"
    assert S.check_entries(L, b, sd)["pass"] is False                      # sha is not the sealed one
    assert S.check_entries(L, good[:1], sd)["pass"] is False               # C0 sealed but never entered
    assert S.check_entries(ledger(4) | {"reads": L["reads"] + good}, good, sd)["pass"] is False      # replaying the same entries: duplicate keys
    b = copy.deepcopy(good)
    b[0]["scorers"] = ["lqr_tracked"]
    assert S.check_entries(L, b, sd)["pass"] is False
    b = copy.deepcopy(good)
    b[0]["card_sha256"] = H("0")
    assert S.check_entries(L, b, sd)["pass"] is False


# ----------------------------------------------------------------------------------------------- CLI
def cli(*a):
    return subprocess.run([sys.executable, os.path.join(HERE, "p5c_seal.py"), *map(str, a)], capture_output=True, text=True, encoding="utf-8")


def test_cli_chain_seal_verify_entries_check_merge(tmp_path):
    (tmp_path / "c1.pt").write_bytes(b"one")
    (tmp_path / "c0.pt").write_bytes(b"two")
    (tmp_path / "card.md").write_text("card")
    (tmp_path / "ledger.json").write_text(json.dumps(ledger(6)))
    assert cli("seal", "--arm", f"C1={tmp_path / 'c1.pt'}", "--arm", f"C0={tmp_path / 'c0.pt'}", "--card", tmp_path / "card.md", "--base-sha", BASE, "--out", tmp_path / "SEALED.json").returncode == 0
    assert cli("seal", "--arm", f"C1={tmp_path / 'c1.pt'}", "--arm", f"C0={tmp_path / 'c0.pt'}", "--card", tmp_path / "card.md", "--base-sha", BASE, "--out", tmp_path / "SEALED.json").returncode == 2   # written once
    assert cli("verify", "--sealed", tmp_path / "SEALED.json", "--card", tmp_path / "card.md", "--recheck-files", "--out", tmp_path / "v.json").returncode == 0
    assert cli("entries", "--sealed", tmp_path / "SEALED.json", "--ledger", tmp_path / "ledger.json", "--out", tmp_path / "e.json", "--time", "2026-12-01T00:00:00Z").returncode == 0
    assert cli("check", "--ledger", tmp_path / "ledger.json", "--entries", tmp_path / "e.json", "--sealed", tmp_path / "SEALED.json", "--out", tmp_path / "chk.json").returncode == 0
    assert cli("merge", "--ledger", tmp_path / "ledger.json", "--entries", tmp_path / "e.json", "--out", tmp_path / "ledger.json").returncode == 2        # never in place
    assert cli("merge", "--ledger", tmp_path / "ledger.json", "--entries", tmp_path / "e.json", "--out", tmp_path / "merged.json").returncode == 0
    m = json.load(open(tmp_path / "merged.json"))
    assert [r["n"] for r in m["reads"]] == list(range(1, 9)) and len(json.load(open(tmp_path / "ledger.json"))["reads"]) == 6
    assert cli("validate-ledger", "--ledger", tmp_path / "merged.json").returncode == 0
    # RED: the merged ledger plus the same entries again is refused
    assert cli("merge", "--ledger", tmp_path / "merged.json", "--entries", tmp_path / "e.json", "--out", tmp_path / "m2.json").returncode == 2
    # RED: a tampered SEALED.json fails verify
    d = json.load(open(tmp_path / "SEALED.json"))
    d["arms"]["C1"]["model_final_sha256"] = H("a")
    (tmp_path / "T.json").write_text(json.dumps(d))
    assert cli("verify", "--sealed", tmp_path / "T.json", "--out", tmp_path / "vt.json").returncode == 1
    assert cli("entries", "--sealed", tmp_path / "T.json", "--ledger", tmp_path / "ledger.json", "--out", tmp_path / "et.json").returncode == 2


def test_cli_validate_ledger_red(tmp_path):
    L = ledger(4)
    L["reads"][3]["n"] = 7
    (tmp_path / "l.json").write_text(json.dumps(L))
    assert cli("validate-ledger", "--ledger", tmp_path / "l.json").returncode == 1
