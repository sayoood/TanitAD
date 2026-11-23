"""tests of read_labelunsafe10k_b.py (card PREREG_V8F_PROOF_ADD136). The tiny synthetic worlds of the test of card ADD135 are reused (its test module is imported).
Run one file at a time. LABELUNSAFE_KIT names the folder of the kit of card ADD135 (default: ../kit)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.environ.get("LABELUNSAFE_KIT") or os.path.join(os.path.dirname(HERE), "kit")
for p in (KIT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import test_read_labelunsafe10k as TL  # noqa: E402
import read_labelunsafe10k_b as LB  # noqa: E402
from test_read_labelunsafe10k import E  # noqa: E402,F401

LU = LB.RL
SAVED = ("CARD", "PRE", "MECH_THRESH", "MECH_CLOSER", "MECH_MIXED", "REP_SAME", "REP_DIFF", "ctl_u2", "ctl_u3")


@pytest.fixture()
def unbound():
    """the imported tool is set back to card ADD135 after a test that bound this card"""
    old = {k: getattr(LU, k) for k in SAVED}
    yield
    for k, v in old.items():
        setattr(LU, k, v)


def outs(tmp_path, n=4):
    out = []
    for i in range(n):
        p = tmp_path / ("t%d.out" % i)
        p.write_text(TL.pytest_text() if i < 2 else "test_read_labelunsafe10k_b.py .....\n\n5 passed in 1.00s\n", encoding="utf-8")
        out.append(str(p))
    return out


def card_a_file(tmp_path, **kw):
    r = {"card": "PREREG_V8F_PROOF_ADD135", "pass_u0": True, "pass_u1": True, "pass_u2": False, "pass_u3": True, "pass_u4": True, "all_pass": False}
    r.update(kw)
    p = tmp_path / "add135_controls.json"
    p.write_text(json.dumps(r), encoding="utf-8")
    return str(p)


def args(p, out, tests, stage, a135):
    return TL.stage_args(p, out, tests, stage) + ["--add135-controls", a135]


def test_order_this_card_is_not_run_unless_u2_alone_failed(tmp_path, capsys):
    assert LB.order_ok(card_a_file(tmp_path))
    for kw in ({"pass_u2": True}, {"pass_u0": False}, {"pass_u4": False}, {"card": "PREREG_V8F_PROOF_ADD134"}):
        assert not LB.order_ok(card_a_file(tmp_path, **kw))
    assert not LB.order_ok(str(tmp_path / "none.json"))
    p = TL.build(tmp_path / "w", TL.W1)
    rc = LB.main(args(p, tmp_path / "out", outs(tmp_path), "controls", card_a_file(tmp_path, pass_u2=True)), TL.tiny_cfg())
    assert rc == 3 and "REFUSED" in capsys.readouterr().out and not (tmp_path / "out").exists()
    assert LU.CARD == "PREREG_V8F_PROOF_ADD135"


def test_stages_pass_and_the_names_carry_this_card(E, tmp_path, capsys, unbound):
    p = TL.build(tmp_path / "w", TL.W1)
    out, tests, a135 = tmp_path / "out", outs(tmp_path), card_a_file(tmp_path)
    assert LB.main(args(p, out, tests, "controls", a135), TL.tiny_cfg()) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert [x for x in lines if x.startswith("ctl ")] == ["ctl u%d PASS" % i for i in range(5)] and lines[-1].endswith("controls all_pass True")
    rec = json.load(open(out / LU.CONTROLS_NAME, encoding="utf-8"))
    assert rec["card"] == "PREREG_V8F_PROOF_ADD136" and rec["all_pass"] is True
    assert LB.main(args(p, out, tests, "names", a135), TL.tiny_cfg()) == 0
    res = json.load(open(out / LU.RESULT_NAME, encoding="utf-8"))
    assert res["card"] == "PREREG_V8F_PROOF_ADD136" and len(res["names"]) == 2 and all(n.startswith("LABELUNSAFE10KB-K0-") for n in res["names"])
    assert res["measures"]["M1"]["value"] == pytest.approx(14 / 17)


def test_u3_needs_four_outputs_and_the_name_of_this_test_file(tmp_path):
    t = outs(tmp_path)
    assert LB.ctl_u3(t)[0] and not LB.ctl_u3(t[:2])[0]
    with open(t[3], "w", encoding="utf-8") as f:
        f.write(TL.pytest_text())
    assert not LB.ctl_u3(t)[0]


def test_exempt_windows_slow_or_braking():
    tl = np.ones((3, 5))
    tl[:, 4] = np.nan
    vl = np.array([[5.0, 5.0, 5.0, 5.0, 0.1], [5.0, 4.7, 4.6, 4.6, 4.6], [5.0, 5.0, 0.9, 5.0, 5.0]])
    assert LB.exempt_windows({"tl": tl, "vl": vl}).tolist() == [False, True, True]


def test_u2_share_and_exemption(E, tmp_path, monkeypatch):
    p = TL.build(tmp_path / "w", TL.W1)
    w = LU.load_world(E, TL.tiny_cfg(), p["root"], p["join"], p["grid"])
    assert LB.ctl_u2(E, w)[0]
    ok, why = LB.ctl_u2(E, w, plant_s=0.0)
    assert not ok and "no window unsafe" in why
    monkeypatch.setattr(LU, "M4_DELTA", 50.0)
    ok, why = LB.ctl_u2(E, w)
    assert not ok and "fewer than 0.90" in why
    monkeypatch.setattr(LB, "M4_SHARE", 0.0)
    ok, why = LB.ctl_u2(E, w)
    assert not ok and "neither slow nor braking" in why
