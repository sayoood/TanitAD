"""PREREG_RA1_FULL Amendment 1's reader (code/ra1_full_amend1.py). CPU only, < 30 s.

Run: PYTHONPATH=<repo>/stack;<repo>/taniteval pytest -q stack/tests/test_ra1_full_amend1.py

What is pinned, and the control that must go red for each:
  * SENS is COPIED from the verdict, and a seed the verdict could not evaluate is REFUSED (a missing run is never a FAIL);
  * the gate replicate is A1x_s1 - A1_s1 on eval-141, paired: REPLICATES only when > 0 AND separated; otherwise the branch text
    names A1a, and K is untouched;
  * the K arm is re-picked on the HOLDOUT among R0 < A1 < A2 with a 0.01 m tie to the simpler; K for A2 is read only when A2 is
    picked, against 2 x max(F_A1, F_A2) -- a wide A2 floor FAILS an A2 that a narrow A1 floor would pass;
  * a missing run is REFUSED.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-refav1-hj-mini"
READER = PKG / "code" / "ra1_full_amend1.py"
pytestmark = pytest.mark.skipif(not READER.exists(), reason="reader not in this checkout")
DIRS = {"R0": "R0_s{s}", "A0": "A0_b0.08_s{s}", "A1": "A1_b0.08g50_s{s}", "A1x": "A1x_b0.08g50_s{s}", "A2": "A2_b0.08g50_s{s}"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_run = _load("test_ra1_full_report_helpers", Path(__file__).resolve().parent / "test_ra1_full_report.py")._run


def _world(tmp, x_rep=0.03, x_spread=0.0, a2_hold=0.05, a2_gap=0.0, sens_pass=True, sens_seed1=True, drop=None):
    rng = np.random.default_rng(1)
    base = 2.6 + rng.normal(0, 0.3, 200) + np.repeat(rng.normal(0, 0.2, 20), 10)
    st = tmp / "full" / "runs" / "st3000"
    off = {"R0": (-0.2, 0.5), "A0": (-0.25, 0.45), "A1": (-0.3, 0.4), "A2": (-0.3, 0.4 + a2_hold)}
    for arm, (e, h) in off.items():
        for s in (0, 1):
            if drop == f"{arm}_s{s}":
                continue
            g = 0.01 * s + (a2_gap * s if arm == "A2" else 0.0)
            _run(st / DIRS[arm].format(s=s) / "eval", arm, 3000, s, base + e + g, base + h + 0.01 * s)
    spread = np.repeat(np.tile([x_spread, -x_spread], 10), 10)   # zero-mean episode spread: widens the CI, leaves the mean at x_rep
    for s, dx in ((0, 0.03), (1, x_rep + spread)):           # A1x = A1 + dx on every window
        if drop != f"A1x_s{s}":
            _run(st / DIRS["A1x"].format(s=s) / "eval", "A1x", 3000, s, base - 0.3 + 0.01 * s + dx, base + 0.4 + 0.01 * s)
    row = lambda: {"A1_yaw_rate_k10": 0.76, "A0_yaw_rate_k10": 0.1, "passes_level": True, "separated_above_A0": sens_pass}
    v = dict(kin_mode="yaw", deciding=True, instrument_ok_all=True,
             SENS={"bar": "b", "PASS": sens_pass, "per_seed": {"seed0": row(), "seed1": row() if sens_seed1 else "NOT EVALUABLE: missing run"}})
    vp = tmp / "full" / "full4_verdict.json"
    vp.write_text(json.dumps(v), encoding="utf-8")
    return tmp / "full", vp


def _main(root, vp):
    out = root / "a1.json"
    rc = _load("ra1_full_amend1", READER).main(["--root", str(root), "--verdict", str(vp), "--S", "3000", "--out", str(out)])
    return rc, (json.loads(out.read_text(encoding="utf-8")) if rc == 0 else None)


def test_sens_pass_and_a_replicating_gate_name_refav2(tmp_path):
    rc, r = _main(*_world(tmp_path))
    assert rc == 0
    assert r["SENS"]["PASS"] is True and "refav2" in r["next"] and "A1a" not in r["next"]
    assert r["REPLICATE"]["verdict"] == "REPLICATES"
    assert r["REPLICATE"]["eval141"]["delta"] == pytest.approx(0.03, abs=1e-4)
    assert r["A2"]["K_arm"] == "A1" and r["A2"]["K_for_A2"]["read"] is False and r["A2"]["K_for_A2"]["PASS"] is None


def test_a_gate_that_does_not_replicate_names_a1a_and_sens_fail_names_the_gamma_ladder(tmp_path):
    rc, r = _main(*_world(tmp_path / "neg", x_rep=-0.01, sens_pass=False))
    assert rc == 0
    assert r["REPLICATE"]["verdict"] == "DOES NOT REPLICATE" and "A1a" in r["next"]
    assert "gamma 200" in r["next"] and "refav2" not in r["next"]
    rc, r = _main(*_world(tmp_path / "ns", x_rep=0.005, x_spread=0.2))   # > 0 but NOT separated: does not replicate
    assert rc == 0 and r["REPLICATE"]["eval141"]["delta"] == pytest.approx(0.005, abs=1e-4)
    assert r["REPLICATE"]["eval141"]["separated"] is False and r["REPLICATE"]["verdict"] == "DOES NOT REPLICATE"


def test_holdout_repick_tie_goes_to_the_simpler_arm_and_a2_is_read_against_the_larger_floor(tmp_path):
    rc, r = _main(*_world(tmp_path / "tie", a2_hold=-0.005))          # A2 better by 0.005 m on the holdout: a tie -> A1
    assert rc == 0 and r["A2"]["K_arm"] == "A1"
    rc, r = _main(*_world(tmp_path / "pick", a2_hold=-0.05))          # A2 better by 0.05 m: picked, K read for A2
    assert rc == 0 and r["A2"]["K_arm"] == "A2" and r["A2"]["K_for_A2"]["PASS"] is True
    rc, r = _main(*_world(tmp_path / "wide", a2_hold=-0.05, a2_gap=0.5))  # F_A2 = 0.51 -> 2F = 1.02 > |~0.7|: FAIL on A2's own floor
    assert rc == 0 and r["A2"]["K_arm"] == "A2" and r["A2"]["K_for_A2"]["PASS"] is False
    assert r["A2"]["two_F"] == pytest.approx(2 * 0.51, abs=1e-6)


def test_an_unevaluable_sens_seed_and_a_missing_run_are_refused(tmp_path):
    assert _main(*_world(tmp_path / "s", sens_seed1=False))[0] == 2
    assert _main(*_world(tmp_path / "m", drop="A0_s1"))[0] == 2
    assert _main(*_world(tmp_path / "x", drop="A1x_s1"))[0] == 2
