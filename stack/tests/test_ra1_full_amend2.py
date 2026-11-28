"""PREREG_RA1_FULL Amendment 2's reader (code/ra1_full_amend2.py): IDM at gamma = 200. CPU only, < 60 s.

What is pinned, and the control that must go red for each:
  * IDM-200 = A1g200 - A0 per seed (A1g200 is 0.25 m below A0 on every window here): PASS only when BOTH seeds are < 0, separated and
    beyond 2 x max(F_A1g200, F_A0) -- a wide A0 floor FAILS it, one unseparated seed FAILS it, a separated WORSE A1g200 FAILS it;
  * a run whose BUILT gamma is not the cell's (an A1g200 directory holding a gamma-50 model) is REFUSED, and so is a missing run;
  * the family block pairs A1g200 with A0 component by component when both carry per-window family arrays, and says UNAVAILABLE
    (never crashes, never reads 0) when one does not;
  * A1g200 - A1g50 and the sensitivity rows are reported and decide nothing.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-refav1-hj-mini"
READER = PKG / "code" / "ra1_full_amend2.py"
pytestmark = pytest.mark.skipif(not READER.exists(), reason="reader not in this checkout")
DIRS = {"A0": ("A0_b0.08_s{s}", "A0", None), "A1g200": ("A1_b0.08g200_s{s}", "A1", 200.0), "A1g50": ("A1_b0.08g50_s{s}", "A1", 50.0)}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_run = _load("test_ra1_full_report_helpers", Path(__file__).resolve().parent / "test_ra1_full_report.py")._run
FAM = _load("ra1_hjepa_eval_for_test", REPO / "stack" / "scripts" / "ra1_hjepa_eval.py").FAMILY_COMPONENTS


def _cell(d, arm, gamma, s, ade6, hade6, sens, fam=None, built_gamma="same"):
    _run(d, arm, 3000, s, ade6, hade6)
    z = dict(np.load(d / "perwindow.npz"))
    z["sens_yaw_rate_k10"] = np.full(ade6.size, sens)
    if fam is not None:                                    # per-window family arrays, as ra1_hjepa_eval saves them (fam_arm_*)
        for _f, name, kind, *_ in FAM:
            if kind == "mean":
                z[f"fam_arm_{name}"] = fam + 0 * ade6
            else:
                z[f"fam_arm_{name}__sum"], z[f"fam_arm_{name}__cnt"] = (fam + 0 * ade6) * 30.0, np.full(ade6.size, 30.0)
    np.savez_compressed(d / "perwindow.npz", **z)
    ev = json.loads((d / "eval.json").read_text(encoding="utf-8"))
    ev["cfg"] = {"gamma": gamma if built_gamma == "same" else built_gamma, "gamma_up": 50.0, "beta": 0.08}
    (d / "eval.json").write_text(json.dumps(ev), encoding="utf-8")


def _world(tmp, a0_gap=0.0, a0_spread=0.0, g200_built=200.0, drop=None, fam=True, g200_off=-0.25):
    rng = np.random.default_rng(2)
    base = 2.6 + rng.normal(0, 0.3, 200) + np.repeat(rng.normal(0, 0.2, 20), 10)
    st = tmp / "full" / "runs" / "st3000"
    for cell, (tmpl, arm, gamma) in DIRS.items():
        for s in (0, 1):
            if drop == tmpl.format(s=s):
                continue
            off = {"A0": 0.0, "A1g200": g200_off, "A1g50": -0.05}[cell] + 0.01 * s
            if cell == "A0" and s == 1:
                off = off + a0_gap + np.repeat(np.tile([a0_spread, -a0_spread], 10), 10)
            f = {"A0": 1.0, "A1g200": 0.8, "A1g50": None}[cell] if fam else None
            _cell(st / tmpl.format(s=s) / "eval", arm, gamma, s, base + off, base + 0.4 + off, {"A0": 0.25, "A1g200": 0.8, "A1g50": 0.76}[cell],
                  fam=f, built_gamma=g200_built if cell == "A1g200" else "same")
    return tmp / "full"


def _main(root):
    out = root / "a2.json"
    rc = _load("ra1_full_amend2", READER).main(["--root", str(root), "--S", "3000", "--out", str(out)])
    return rc, (json.loads(out.read_text(encoding="utf-8")) if rc == 0 else None)


def test_idm200_passes_only_at_both_seeds_beyond_the_larger_floor(tmp_path):
    rc, r = _main(_world(tmp_path / "ok"))
    assert rc == 0 and r["IDM200"]["PASS"] is True and "IDM-200 PASS" in r["next"] and "IDM-200 PASS" in r["reading"]
    rows = r["IDM200"]["rows"]
    assert [round(x["A1g200_minus_A0"]["delta"], 4) for x in rows] == [-0.25, -0.25]
    assert [round(x["A1g200_minus_A1g50_DESCRIPTIVE"]["delta"], 4) for x in rows] == [-0.2, -0.2]
    assert rows[0]["sens_yaw_rate_k10"]["A1g200"] == pytest.approx(0.8) and rows[0]["sens_yaw_rate_k10"]["A0"] == pytest.approx(0.25)
    rc, r = _main(_world(tmp_path / "floor", a0_gap=0.2))          # F_A0 = 0.21 -> 2F = 0.42 > 0.25: FAIL at both seeds
    assert rc == 0 and r["IDM200"]["PASS"] is False and r["IDM200"]["two_F"] == pytest.approx(0.42, abs=1e-6)
    assert all(x["separated"] for x in r["IDM200"]["rows"]) and "no further gamma arms" in r["next"]
    rc, r = _main(_world(tmp_path / "one", a0_spread=0.6))         # seed 1 not separated: FAIL, BOTH seeds are required
    assert rc == 0 and r["IDM200"]["rows"][0]["separated"] is True and r["IDM200"]["rows"][1]["separated"] is False
    assert r["IDM200"]["PASS"] is False
    rc, r = _main(_world(tmp_path / "worse", g200_off=+0.25))     # separated and beyond the floor, but the WRONG way: FAIL
    rows = r["IDM200"]["rows"]
    assert rc == 0 and r["IDM200"]["PASS"] is False and all(x["separated"] and x["beats_2x_floor"] and not x["below"] for x in rows)


def test_a_run_built_at_the_wrong_gamma_and_a_missing_run_are_refused(tmp_path, capsys):
    assert _main(_world(tmp_path / "g", g200_built=50.0))[0] == 2
    assert "built (gamma, gamma_up) = (50.0, 50.0)" in capsys.readouterr().err
    assert _main(_world(tmp_path / "m", drop="A1_b0.08g200_s1"))[0] == 2
    assert "missing ['A1_b0.08g200_s1']" in capsys.readouterr().err


def test_families_are_paired_by_component_or_reported_unavailable(tmp_path):
    rc, r = _main(_world(tmp_path / "f"))
    blk = r["IDM200"]["rows"][0]["families_A1g200_minus_A0"]
    assert rc == 0 and blk["status"] == "OK"
    for fam in ("longitudinal", "lateral"):
        for name, c in blk[fam].items():
            assert c["delta_exact"] == pytest.approx(-0.2, abs=1e-9), name    # the saved sums are already in published units
    rc, r = _main(_world(tmp_path / "n", fam=False))
    assert rc == 0 and r["IDM200"]["rows"][0]["families_A1g200_minus_A0"]["status"] == "UNAVAILABLE"
