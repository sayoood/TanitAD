"""tests of read_boxfuse_loc.py (card PREREG_V8F_PROOF_ADD130). Tiny synthetic worlds of the kits of cards ADD127 to ADD129 (their test modules are imported).
Run one file at a time. BOXREAD_KIT_READ, BOXREAD_KIT_C and BOXREAD_KIT_F name the folders of those kits (defaults: ../read, ../read2, ../fuse, or the pod paths)."""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for env, rel, pod, mark in (("BOXREAD_KIT_F", "fuse", "/workspace/v8proof/boxread/kit_f", "read_boxfuse.py"),):
    for cand in (os.environ.get(env), os.path.join(os.path.dirname(HERE), rel), pod):
        if cand and os.path.isfile(os.path.join(cand, mark)):
            KITF = cand
            break
    else:
        raise RuntimeError("the kit of card ADD129 was not found")
for p in (KITF, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import test_read_boxfuse as TF  # noqa: E402
import read_boxfuse_loc as RL  # noqa: E402
from test_read_boxfuse import fast_render, small_floor  # noqa: E402,F401

RF, RB, BS, T = RL.RF, RL.RB, RL.BS, TF.T


@pytest.fixture
def low_f3(monkeypatch):
    monkeypatch.setattr(RF, "F3_MIN_SHARE", 0.05)


def world(tmp_path, capsys):
    """a world with the files of cards ADD128 and ADD129 and two further locator files -> (argv builder per fit, info)"""
    fa, info = TF.world(tmp_path, capsys)
    assert RF.main(fa("controls")) == 0 and RF.main(fa("names")) == 0
    capsys.readouterr()
    loc = info["dirs"]["loc"]
    z = dict(np.load(os.path.join(loc, "LOC_s0.npz")))
    for i, f in enumerate(RL.FITS):
        zz = dict(z)
        zz["az_pred"] = z["az_pred"] + 0.15 * (i + 1)
        p = os.path.join(loc, f + ".npz")
        np.savez(p, **zz)
        BS.wj(os.path.join(loc, f + "_meta.json"), {"pred_sha256": BS.sha_file(p)})
    r129 = os.path.join(info["fout"], "FUSE_RESULT.json")
    outs = {f: str(tmp_path / ("out_" + f)) for f in RL.FITS}

    def la(mode, fit, extra=(), other=True):
        base = fa(mode)
        base[base.index("--out-dir") + 1] = outs[fit]
        e = ["--locator-fit", fit, "--add129-result", r129] + list(extra)
        if other:
            o = [x for x in RL.FITS if x != fit][0]
            e += ["--other-controls", os.path.join(outs[o], "FUSE_CONTROLS.json")]
        return base + e
    info.update({"outs": outs, "r129": r129, "sum": str(tmp_path / "out_sum")})
    return la, info


def run(argv, capsys):
    rc = RL.main(argv)
    return rc, capsys.readouterr().out


def test_names_rule():
    ok = {"LOC_s1": {"R": 0.89, "iv": [0.8, 0.95]}, "LOC_s2": {"R": 0.90, "iv": [0.8, 0.99]}}
    assert RL.derive_names(ok) == ["BOXFUSE-T0-LOCFIT-REPLICATED", "BOXFUSE-T0-LOCFIT-BAR-3-OF-3"]
    ok["LOC_s2"]["R"] = 0.93
    assert RL.derive_names(ok) == ["BOXFUSE-T0-LOCFIT-REPLICATED", "BOXFUSE-T0-LOCFIT-BAR-NOT-3-OF-3"]
    ok["LOC_s2"]["iv"] = [0.85, 1.01]
    assert RL.derive_names(ok) == ["BOXFUSE-T0-LOCFIT-NOT-REPLICATED", "BOXFUSE-T0-LOCFIT-BAR-NOT-3-OF-3"]


def test_full_course_controls_for_both_fits_then_values_then_names(tmp_path, capsys, fast_render, small_floor, low_f3):
    la, info = world(tmp_path, capsys)
    rc, out = run(la("controls", "LOC_s1"), capsys)
    assert rc == 0 and out.strip().splitlines()[-1] == "ZZ BOXFUSE CONTROLS ALL-PASS", out
    rc, out = run(la("names", "LOC_s1"), capsys)
    assert rc == 3 and "REFUSED: the controls file of the other fit" in out
    rc, out = run(["summary", info["outs"]["LOC_s1"], info["outs"]["LOC_s2"], info["sum"], info["r129"]], capsys)
    assert rc == 3 and "REFUSED" in out
    rc, out = run(la("controls", "LOC_s2"), capsys)
    assert rc == 0, out
    for f in RL.FITS:
        rc, out = run(la("names", f), capsys)
        assert rc == 0 and "ZZ BOXFUSE NAME" not in out, out
        cj = json.load(open(os.path.join(info["outs"][f], "FUSE_CONTROLS.json"), encoding="utf-8"))
        assert cj["card"] == "PREREG_V8F_PROOF_ADD130" and cj["inputs"]["locator_fit"] == f
        assert set(cj["controls"]["f0"]["g6_fg_of_row_198_reproduced"].values()) == {True}
    rc, out = run(["summary", info["outs"]["LOC_s1"], info["outs"]["LOC_s2"], info["sum"], info["r129"]], capsys)
    names = [x.split()[-1] for x in out.splitlines() if x.startswith("ZZ LOCFIT NAME ")]
    assert rc == 0 and len(names) == 2 and all(n.startswith("BOXFUSE-T0-LOCFIT-") for n in names), out
    rj = json.load(open(os.path.join(info["sum"], "LOCFIT_RESULT.json"), encoding="utf-8"))
    assert rj["card"] == "PREREG_V8F_PROOF_ADD130" and len(rj["R_FG_three_fits"]) == 3 and set(rj["per_fit"]) == {"LOC_s0", "LOC_s1", "LOC_s2"}
    txt = open(os.path.join(info["sum"], "LOCFIT_VERDICT.md"), encoding="utf-8").read()
    assert "ADD130" in txt and "read a third time" in txt and "STRATEGIC: not computed, n = 0" in txt
    assert RF.CARD == "PREREG_V8F_PROOF_ADD129" and RF.ctl_f0 is RL.ORIG["ctl_f0"] and RF.derive_names is RL.ORIG["derive_names"]


def test_g0_fails_for_the_first_fit_of_the_locator(tmp_path, capsys, fast_render, small_floor, low_f3):
    la, info = world(tmp_path, capsys)
    argv = la("controls", "LOC_s1")
    argv[len(argv) - 1 - argv[::-1].index("--locator-fit") + 1] = "LOC_s0"
    rc, out = run(argv, capsys)
    assert rc == 2 and re.search(r"^ctl f0 FAIL the locator's fit is not one of", out, re.M), out


def test_g6_fails_when_row_198_is_not_reproduced(tmp_path, capsys, fast_render, small_floor, low_f3):
    la, info = world(tmp_path, capsys)
    rj = json.load(open(info["r129"], encoding="utf-8"))
    rj["measures"]["E"]["FG_1"] += 0.01
    bad = os.path.join(str(tmp_path), "r129_changed.json")
    BS.wt(bad, json.dumps(rj))
    argv = la("controls", "LOC_s2")
    argv[len(argv) - 1 - argv[::-1].index("--add129-result") + 1] = bad
    rc, out = run(argv, capsys)
    assert rc == 2 and re.search(r"^ctl f0 FAIL FG_1 with LOC_s0: its error is not the banked value of row 198", out, re.M), out


def test_g0_fails_when_the_locator_file_is_not_its_banked_record(tmp_path, capsys, fast_render, small_floor, low_f3):
    la, info = world(tmp_path, capsys)
    BS.wj(os.path.join(info["dirs"]["loc"], "LOC_s1_meta.json"), {"pred_sha256": "0" * 64})
    rc, out = run(la("controls", "LOC_s1"), capsys)
    assert rc == 2 and re.search(r"^ctl f0 FAIL the locator's file does not hash to its banked record", out, re.M), out


def test_tool_identity_is_checked(monkeypatch):
    assert RL._sha(RF.__file__).startswith(RL.RF_PREFIX)
    monkeypatch.setattr(RL, "RF_PREFIX", "0" * 16)
    with pytest.raises(SystemExit):
        RL.load_fuse()
