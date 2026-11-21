"""Unit tests of read_pess_a1 (synthetic; no navtest read)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read_pess_a1 as A  # noqa: E402
import read_pess as RP  # noqa: E402


def _m():
    rng = np.random.default_rng(0)
    return [rng.normal(size=(5, 64, 6)).astype(np.float32) for _ in range(4)]


def test_lcb1nd_touches_only_nc_dac():
    m = _m(); out = A.combine(m, "LCB1ND"); mean = np.mean(np.stack(m), 0, dtype=np.float32)
    for h in (RP.HEAD["TTC"], RP.HEAD["EP"], RP.HEAD["C"], RP.HEAD["DDC"]):
        assert np.array_equal(out[..., h], mean[..., h])
    for h in (RP.HEAD["NC"], RP.HEAD["DAC"]):
        assert (out[..., h] <= mean[..., h] + 1e-6).all() and (out[..., h] < mean[..., h]).any()


def test_red_ttc_put_back_equals_lcb1_bits():
    m = _m()
    assert np.array_equal(A.lcb_on(m, [RP.HEAD["NC"], RP.HEAD["DAC"], RP.HEAD["TTC"]]), A._orig_combine(m, "LCB1"))
    assert not np.array_equal(A.combine(m, "LCB1ND"), A._orig_combine(m, "LCB1"))


def test_one_member_collapses():
    m = _m()[:1]
    assert np.array_equal(A.combine(m, "LCB1ND"), m[0])


def test_other_rules_unchanged():
    m = _m()
    for r in ("ENS9", "PESS", "LCB1"):
        assert np.array_equal(A.combine(m, r), A._orig_combine(m, r))


def test_outcome_table():
    assert A.decide_a1(True, True, True, True, True)["label"] == "A1-1"
    assert A.decide_a1(False, False, True, True, True)["label"] == "A1-2"
    assert A.decide_a1(False, True, True, False, True)["label"] == "A1-3"
    assert A.decide_a1(True, False, True, True, True)["label"] == "Unnamed"
    assert A.decide_a1(True, True, True, False, False)["label"] == "Unnamed"


def test_card_sha_refusal(tmp_path, monkeypatch):
    p = tmp_path / "card.md"; p.write_bytes(b"not the card")
    monkeypatch.setattr(A, "A1_CARD", str(p))
    assert A.main(["--member", "X=Y", "--p6", "Z", "--out", str(tmp_path / "o"), "--no-ledger"]) == 2
