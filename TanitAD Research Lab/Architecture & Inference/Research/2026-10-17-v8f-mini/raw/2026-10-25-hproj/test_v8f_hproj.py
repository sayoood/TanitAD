"""Self-tests of the HPROJ arithmetic (analytic identities + deliberate-regression arms). CPU, numpy + torch, tiny, no data files needed
(the last two tests read the banked control / score jsons when present).

  python -m pytest -q test_v8f_hproj.py --basetemp C:/Users/Admin/pt_hproj -p no:cacheprovider
"""
import json
import os
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v8f_hproj as HP  # noqa: E402

H = HP.H


def _res60(nb=12, n=64, seed=0):
    rng = np.random.default_rng(seed)
    basis = HP.basis_np(nb)
    c = rng.normal(size=(n, 2, nb)) * np.linspace(2.0, 0.1, nb)[None, None, :]
    return basis, c, np.einsum("tk,nck->ntc", basis, c)


def test_basis_matches_the_model_formula_and_is_full_rank_on_the_eval_steps():
    b = HP.basis_np(12)
    i = np.arange(60)
    assert b.shape == (60, 12)
    assert np.allclose(b[:, 3], np.cos(3 * np.pi * (i + 0.5) / 60) * (i + 1) / 60)
    assert np.linalg.matrix_rank(b[1::2]) == 12                       # 30 eval rows identify 12 coefficients


def test_up_np_matches_torch_upsample_prior_and_keeps_the_knots():
    import torch
    from tanitad.models import v8f_mini as M
    p = np.random.default_rng(1).normal(size=(7, 30, 2)) * 10
    u = HP.up_np(p)
    assert u.shape == (7, 60, 2) and np.allclose(u[:, 1::2], p)
    assert np.allclose(u[:, 0], 0.5 * p[:, 0])                        # first midpoint is halfway to the origin
    assert np.abs(u - M.upsample_prior(torch.as_tensor(p, dtype=torch.float64)).numpy()).max() < 1e-12


def test_coefficients_are_recovered_exactly_including_the_unobserved_midpoints():
    basis, c, res60 = _res60()
    rec_c, diag = HP.recover_c(res60[:, 1::2], basis)
    assert np.abs(rec_c - c).max() < 1e-9 and diag["rank"] == 12
    p = np.random.default_rng(2).normal(size=(len(c), 30, 2)) * 5
    tau60_true = HP.up_np(p) + res60
    tau60_rec, _, _ = HP.recon_traj60(tau60_true[:, 1::2], p, basis)
    assert np.abs(tau60_rec - tau60_true).max() < 1e-9               # knots AND the 0.1 s midpoints


def test_deliberate_regression_wrong_basis_is_caught_by_the_knot_control():
    """A basis without the ramp spans a different space: its reconstruction cannot reproduce the knots -> control C1 (<= 1e-4 m) must FAIL."""
    basis, c, res60 = _res60()
    p = np.random.default_rng(3).normal(size=(len(c), 30, 2)) * 5
    tau30 = (HP.up_np(p) + res60)[:, 1::2]
    ok, _, _ = HP.recon_traj60(tau30, p, basis)
    bad, _, _ = HP.recon_traj60(tau30, p, HP.basis_np(12, ramp=False))
    assert np.abs(ok[:, 1::2] - tau30).max() <= HP.TOL_RECON
    assert np.abs(bad[:, 1::2] - tau30).max() > HP.TOL_RECON


def test_w60_is_the_linear_interpolation_with_w01_equal_w02_and_matches_hblend_derived_grid():
    rng = np.random.default_rng(4)
    w = np.sort(rng.uniform(0, 1, size=(30, 2)), axis=0)
    w60 = HP.w60_from_w30(w)
    assert w60.shape == (60, 2)
    assert np.allclose(w60[1::2], w) and np.array_equal(w60[0], w60[1])
    assert np.allclose(w60[2], 0.5 * (w[0] + w[1]))                   # t = 0.3 s sits halfway between the 0.2 s and 0.4 s knots
    assert np.allclose(w60, H.w_midpoints(w))                          # independent of HBLEND's derived-grid helper, equal to it
    assert (np.diff(w60, axis=0) >= -1e-12).all()


def test_blend_identities():
    basis, c, res60 = _res60(n=10)
    p = np.random.default_rng(5).normal(size=(10, 30, 2)) * 5
    up = HP.up_np(p)
    tau60 = up + res60
    assert np.allclose(HP.blend60(up, tau60, np.ones((60, 2))), tau60)
    assert np.allclose(HP.blend60(up, tau60, np.zeros((60, 2))), up)
    w = np.stack([np.full(60, 0.25), np.full(60, 0.75)], 1)
    h = HP.blend60(up, tau60, w)
    assert np.allclose(h - up, w[None] * res60)                       # along keeps 25 %, cross keeps 75 %, in the ego frame


def _line(v, n=60, dt=0.1):
    t = dt * np.arange(1, n + 1)
    return np.stack([v * t, np.zeros_like(t)], 1)[None]


def test_projection_identity_below_the_cap_and_arc_length_clamp_above_it():
    traj = _line(10.0)
    assert np.abs(HP.project_f32(traj, np.array([20.0])) - traj).max() < 1e-4        # below the cap: untouched
    assert np.abs(HP.project_f32(traj, np.array([1e9])) - traj).max() < 1e-4         # no limit (1e9): untouched
    out = HP.project_f32(traj, np.array([5.0]))
    assert abs(out[0, -1, 0] - 5.0 * 6.0) < 1e-3 and np.abs(out[0, :, 1]).max() < 1e-6   # re-timed to 5 m/s, geometry (the x axis) preserved
    assert (np.diff(out[0, :, 0]) <= 5.0 * 0.1 + 1e-4).all()


def test_projection_kamm_circle_slows_a_tight_turn_but_not_a_straight_line():
    t = 0.1 * np.arange(1, 61)
    r = 20.0                                                             # 20 m radius at 15 m/s: v^2 / r = 11.25 m/s^2 > 0.7 g = 6.87
    th = 15.0 * t / r
    turn = np.stack([r * np.sin(th), r * (1 - np.cos(th))], 1)[None]
    out = HP.project_f32(turn, np.array([1e9]))
    arc = lambda p: np.linalg.norm(np.diff(np.concatenate([np.zeros((1, 1, 2)), p], 1), axis=1), axis=-1).sum()
    assert arc(out) < arc(turn) - 5.0
    assert abs(arc(HP.project_f32(_line(15.0), np.array([1e9]))) - 15.0 * 6.0) < 1e-3


def test_cap_from_J_matches_the_limit_onehots_and_marks_no_limit_as_1e9():
    J = {"limit": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]], np.float32),
         "limit_ok": np.array([True, True, True, True, False])}
    cap, lim = HP.cap_from_J(J, np.array([0, 1, 2, 3, 4, 1]))
    assert np.allclose(cap[:4], np.array([30, 50, 100, 120]) / 3.6, atol=1e-5)
    assert cap[4] == HP.BIG and np.isnan(lim[4])                         # no valid limit -> 1e9, as the eval's torch.where does
    assert np.isclose(cap[5], 50 / 3.6, atol=1e-5)


def test_banded_projection_identity_and_band_semantics():
    import diag_band as DB
    traj = _line(10.0)
    cap = np.array([5.0])
    assert np.abs(DB.project_banded(traj, cap, start=0) - HP.project_f32(traj, cap)).max() < 1e-5      # identity control (band 0 == project_speed)
    out = DB.project_banded(traj, cap, start=20)
    v = np.linalg.norm(np.diff(np.concatenate([np.zeros((1, 1, 2)), out], 1), axis=1), axis=-1)[0] / 0.1
    assert v[:20].mean() > 9.0 and v[25:].max() < 5.0 + 1e-3             # untouched for the first 2 s, capped after


def _mk_score(c_hi, k2, k6, sep=True):
    row = lambda m, lo, hi: {"mean": m, "lo": lo, "hi": hi, "separated": sep}
    z = lambda: row(0.0, -0.1, 0.1)
    single = {n: {"ade2": {"mean": 0.2}, "ade6": {"mean": 2.0}, "fw_lon_accel_mae": {"mean": 0.7}} for n in ("HP", "P10")}
    return {"controls": {"binding": {"all_pass": True}}, "estimator": "e", "tier": "t", "single": single,
            "paired": {"HP_minus_B": {"ade2": row(k2, k2 - 0.05, k2 + 0.05), "ade6": row(k6, k6 - 0.05, k6 + 0.05)},
                       "HP_minus_P10": {"ade2": z(), "ade6": z(), "fw_lon_accel_mae": z()}},
            "consistency": {"paired": {"HP_minus_B": {"viol_any": {"mean": c_hi - 0.01, "lo": 0.0, "hi": c_hi}}}}}


@pytest.mark.parametrize("c_hi,k,sep,expect", [(0.015, -0.1, True, "PASS"), (0.05, -0.1, True, "CONSISTENCY NOT RESTORED"),
                                               (0.015, 0.05, True, "GAIN LOST"), (0.05, 0.05, True, "CONSISTENCY NOT RESTORED"),
                                               (0.0200001, -0.1, True, "CONSISTENCY NOT RESTORED"), (0.015, -0.1, False, "GAIN LOST")])
def test_outcome_table_logic_verbatim(tmp_path, monkeypatch, c_hi, k, sep, expect):
    monkeypatch.setattr(HP, "OUT", str(tmp_path))
    for s in HP.SEEDS:
        json.dump(_mk_score(c_hi, k, k, sep), open(tmp_path / f"score_{s}.json", "w"))
    assert HP.cmd_read(None) == 0
    assert json.load(open(tmp_path / "READ.json"))["outcome"] == expect


def test_banked_control_jsons_pass_if_present():
    for s in HP.SEEDS:
        p = os.path.join(HP.OUT, f"control_{s}.json")
        if not os.path.exists(p):
            pytest.skip("control not run yet")
        c = json.load(open(p))
        assert c["binding"]["all_pass"] and c["binding"]["C1_recon_knots_vs_banked_traj30_maxabs_m"] <= HP.TOL_RECON
        d = c["diagnostics"]
        assert d["R1_must_exceed_C1_bar_and_does"] and d["R2_must_exceed_C2_bar_and_does"] and d["R3_must_exceed_C2_bar_and_does"]
