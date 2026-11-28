"""Controls for collapse_monitor.py. Planted controls need numpy only; the real-path test needs the banked R35H_s0 checkpoint, the eval frames and the slowwm tree (skipped when absent)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
import collapse_monitor as CM  # noqa: E402

RUN = "C:/Users/Admin/t2v8f_pulls/ladder_R35H_s0"
FRAMES = "F:/TanitAD-scratch/ladder_read/cache/frames_s0.npz"


def test_isotropic_pr_equals_dim():
    Z = np.random.default_rng(1).standard_normal((4000, 48))
    s = CM.spectrum_stats(Z)
    assert s["dim"] == 48 and abs(s["pr"] - 48) / 48 < 0.05
    assert s["dead_dims"] == 0 and s["dims_for_99pct"] >= 45


def test_rank1_pr_is_1():
    rng = np.random.default_rng(2)
    Z = np.outer(rng.standard_normal(3000), rng.standard_normal(64))
    s = CM.spectrum_stats(Z)
    assert abs(s["pr"] - 1.0) < 0.05 and s["dims_for_90pct"] == 1 and s["top20_frac"][0] > 0.999


@pytest.mark.parametrize("r", [2, 5, 12, 30])
def test_rank_r_pr_is_r(r):
    rng = np.random.default_rng(3 + r)
    Q, _ = np.linalg.qr(rng.standard_normal((64, r)))
    s = CM.spectrum_stats(rng.standard_normal((5000, r)) @ Q.T)
    assert abs(s["pr"] - r) / r < 0.05 and s["dims_for_99pct"] == r


def test_dead_dims_counted():
    Z = np.random.default_rng(4).standard_normal((1000, 32))
    Z[:, [3, 9, 20]] = 7.0                      # constant, non-zero mean: still dead
    assert CM.spectrum_stats(Z)["dead_dims"] == 3


def test_reading_rule_is_fixed():
    assert CM.RULE == {"collapsed": 0.02, "partial": 0.10}
    assert [CM.reading(x) for x in (0.0, 0.019, 0.02, 0.099, 0.10, 0.9)] == ["collapsed", "collapsed", "partial", "partial", "healthy", "healthy"]


def test_selfcheck_green_on_the_real_monitor():
    assert CM.selfcheck()["ok"]


def test_red_arm_wrong_axis_mutation(monkeypatch):
    """A monitor whose covariance reads time as the feature axis (Z transposed) must fail the planted self-check."""
    orig = CM._cov
    monkeypatch.setattr(CM, "_cov", lambda Z: orig(np.asarray(Z).T))
    sc = CM.selfcheck()
    assert not sc["ok"]
    assert not sc["offset_iso_pr_within_5pct_of_dim"]           # the per-feature offset survives a wrong-axis centring
    assert not sc["iso_pr_within_5pct_of_dim"]                  # and 'dim' is read as N


def test_red_arm_is_detected_by_the_plain_stats_too(monkeypatch):
    orig = CM._cov
    Z = np.random.default_rng(5).standard_normal((300, 16))
    monkeypatch.setattr(CM, "_cov", lambda Z_: orig(np.asarray(Z_).T))
    assert CM.spectrum_stats(Z)["dim"] == 300 != 16


@pytest.mark.skipif(not (os.path.exists(RUN) and os.path.exists(FRAMES)), reason="banked R35H_s0 or frames absent")
def test_real_path_z1_bit_equal_to_lp_lib_clip_z1():
    import torch
    CM._setup_paths()
    import lr_lib as LR
    import lp_lib as LP
    import tanitad
    assert os.path.abspath(tanitad.__file__).replace("\\", "/").lower().startswith("f:/tanitad-scratch/slowwm/tree"), tanitad.__file__
    model, cfg, argv, ck = LR.load_arm(RUN)
    fr = LR.load_eval_frames(FRAMES)
    idx = [0, 17, 40, 63, 90, 100, 120, 140]
    Z, z1 = CM.collect_latents(model, fr, idx)
    sub = {k: ([fr[k][i] for i in idx] if isinstance(fr[k], list) and k in ("clips", "T", "zmean", "act", "ok") else fr[k]) for k in fr}
    ref = LP.clip_z1(model.wm, sub)
    assert list(ref) == [fr["clips"][i] for i in idx]
    for c in ref:
        assert np.array_equal(z1[c].numpy(), ref[c].numpy())
    assert np.array_equal(Z[1], np.concatenate([ref[c].numpy() for c in ref], 0))
    # the encoded level 1 of the stack is the same z1 (informative: the levels 2, 3 read the same latents)
    zl, _ = LR.encode_clip(model, LR.z_in(model, sub, 0), sub["act"][0])
    assert zl[0][0].shape == ref[sub["clips"][0]].shape
    assert Z[2].shape[1] == Z[3].shape[1] == Z[1].shape[1]
    # a mutated reader (frames of another clip handed to z1) must differ
    wrong = LP.clip_z1(model.wm, {**sub, "zmean": sub["zmean"][::-1]})
    assert not np.array_equal(z1[sub["clips"][0]].numpy(), wrong[sub["clips"][0]].numpy())
