"""PREREG_V8F_SMOOTH's reader (code/v8f_smooth_read.py). CPU only, < 60 s.

Identities, each a reading that would be wrong without it:
  * an arm equal to P10 (zero residual) reads EXACTLY 0 on every delta, so D is not met (0 is not < 0) and P is;
  * a residual that is a quadratic in time is left unchanged by SG(., 2): every delta is 0 to float precision;
  * white noise on the residual is reduced: accel and yaw-rate MAE fall, separated, so D is met;
  * an arm that IS the (slightly jittery) human path is made WORSE by S: the dynamics deltas are > 0 (D not met, whatever their
    separation) and ADE rises above the +0.01 m bar (P not met), while control C still passes;
  * one seed read is INTERIM, two seeds carry a verdict; a jagged human path fails control C and makes every arm INVALID;
  * a run whose saved ade6 is not ||pred - gt|| is REFUSED.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "code" / "v8f_smooth_read.py"
pytestmark = pytest.mark.skipif(not CODE.exists(), reason="reader not in this checkout")


def _load():
    spec = importlib.util.spec_from_file_location("v8f_smooth_read", CODE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _world(tmp, jagged_human=False, bad_ade=None, human_jitter=0.0):
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    n, H = 200, 30
    t = (np.arange(1, H + 1) * 0.2)[None]
    v0 = rng.uniform(4, 14, n)[:, None]
    k = rng.normal(0, 0.01, n)[:, None]
    s = v0 * t + 0.1 * rng.normal(0, 1, n)[:, None] * t ** 2
    gt = np.stack([np.sin(k * s + 1e-9) / (k + 1e-9), (1 - np.cos(k * s)) / (k + 1e-9)], -1)
    p10 = gt + np.stack([0.05 * t ** 2 * np.ones((n, 1)), 0.02 * t * np.ones((n, 1))], -1)        # a smooth, wrong prior
    if jagged_human:                                                       # the HUMAN path jitters, the prior does not
        gt = gt + rng.normal(0, 0.4, gt.shape)
    gt = gt + rng.normal(0, human_jitter, gt.shape) if human_jitter else gt
    eid = np.repeat(np.arange(20), 10)
    np.savez(tmp / "gt.npz", gt=gt, eid=eid, dt=0.2)
    np.save(tmp / "p10.npy", p10)
    quad = np.stack([0.3 * t ** 2 - 0.2 * t, -0.1 * t ** 2], -1) * np.ones((n, 1, 1))
    runs = {"zero": p10, "quad": p10 + quad, "noisy": p10 + rng.normal(0, 0.3, gt.shape), "truth": gt}
    ev = tmp / "eval"
    ev.mkdir()
    for arm, p in runs.items():
        for sd in (0, 1):
            a = np.linalg.norm(p - gt, axis=-1).mean(1)
            np.savez(ev / f"eval_{arm}_s{sd}.npz", pred=p, ade6=a + (0.01 if bad_ade == f"{arm}_s{sd}" else 0.0), eid=eid)
    return tmp


def _run(tmp, tags):
    rc = _load().main(["--eval-dir", str(tmp / "eval"), "--tags", *tags, "--out", str(tmp / "o.json"), "--gt-npz", str(tmp / "gt.npz"),
                       "--p10-pred", str(tmp / "p10.npy"), "--n-boot", "300"])
    return rc, json.loads((tmp / "o.json").read_text(encoding="utf-8"))


def test_zero_and_quadratic_residuals_are_fixed_points_and_noise_is_removed(tmp_path):
    rc, r = _run(_world(tmp_path), ["zero_s0", "zero_s1", "quad_s0", "quad_s1", "noisy_s0", "noisy_s1"])
    assert rc == 0 and r["control_C"]["ok"] is True
    for arm in ("zero", "quad"):
        for s in ("seed0", "seed1"):
            S = r["arms"][arm]["per_seed"][s]["S"]
            assert abs(S["ade6_S_minus_arm"]["delta"]) < 1e-9, arm
            assert all(abs(v["delta"]) < 1e-6 and not v["ok"] for v in S["D"].values()), arm
            assert S["P_ok"] is True
        assert r["arms"][arm]["verdict"] == "D FAIL"
    N = r["arms"]["noisy"]
    assert all(N["per_seed"][s]["S"]["D_ok"] for s in ("seed0", "seed1"))
    assert N["per_seed"]["seed0"]["S"]["roughness"]["accel_mean_mps2"] < 0.5 * N["per_seed"]["seed0"]["roughness_arm"]["accel_mean_mps2"]
    assert N["verdict"] in ("SMOOTH-PASS", "D PASS, P FAIL")


def test_one_seed_is_interim_and_a_jagged_human_path_is_invalid(tmp_path):
    rc, r = _run(_world(tmp_path / "a"), ["noisy_s0"])
    assert rc == 0 and r["arms"]["noisy"]["verdict"].startswith("INTERIM")
    rc, r = _run(_world(tmp_path / "b", jagged_human=True), ["noisy_s0", "noisy_s1"])
    assert rc == 0 and r["control_C"]["ok"] is False and r["arms"]["noisy"]["verdict"].startswith("INVALID")


def test_a_wrong_saved_ade6_is_refused(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        _run(_world(tmp_path, bad_ade="quad_s1"), ["quad_s0", "quad_s1"])
    assert e.value.code == 2 and "the ground truth is not its grid" in capsys.readouterr().err


def test_smoothing_a_correct_wiggle_fails_both_d_and_p(tmp_path):
    rc, r = _run(_world(tmp_path, human_jitter=0.03), ["truth_s0", "truth_s1"])
    assert rc == 0 and r["control_C"]["ok"] is True
    for s in ("seed0", "seed1"):
        S = r["arms"]["truth"]["per_seed"][s]["S"]
        assert all(v["delta"] > 0 and v["separated"] and not v["ok"] for v in S["D"].values())
        assert S["ade6_S_minus_arm"]["hi"] > 0.01 and S["P_ok"] is False
    assert r["arms"]["truth"]["verdict"] == "D FAIL"
