"""Contract tests for the 0-GPU latent screen (``tanitad.eval.latent_screen``).

⭐ WHY THESE EXIST. The screen's job is to REJECT a temporally-collapsed latent before a
training run is authorised. A screen that silently passes everything is worse than no screen,
so the load-bearing test is :func:`test_collapsed_latent_fails_on_jitter` — a latent built to
have NO 100 ms temporal resolution must fail, and must fail specifically on the jitter ratio.

The synthetic latents below are the two poles the reference measurement found:
  ORACLE     the true speed track is linearly readable at each window position
             (frozen-v1's oracle arm: jitter 1.0x, derivative corr +0.99997)
  COLLAPSED  the window's nine positions are near-copies of ONE scene vector plus isotropic
             noise -- frozen-v1's actual behaviour (jitter 51.0x, per-position corr +0.0061)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tanitad.eval import latent_screen as LS


DT = 0.1
W = 9
C = W // 2


def _speed_tracks(n: int, seed: int) -> torch.Tensor:
    """[n, W] plausible speed tracks: a base speed plus a per-window constant accel."""
    g = np.random.default_rng(seed)
    v0 = g.uniform(2.0, 30.0, size=(n, 1))
    acc = g.normal(0.0, 1.2, size=(n, 1))                 # m/s^2
    t = (np.arange(W) - C)[None, :] * DT
    return torch.from_numpy((v0 + acc * t).astype(np.float32))


def _oracle_latent(Q: torch.Tensor, d: int, seed: int) -> torch.Tensor:
    """Channel 0 IS the speed at that position; the rest is window-constant scene noise."""
    g = np.random.default_rng(seed)
    n = len(Q)
    scene = g.normal(0, 1, size=(n, 1, d - 1)).repeat(W, axis=1)
    z = np.concatenate([Q.numpy()[:, :, None], scene], axis=2)
    return torch.from_numpy(z.astype(np.float32))


def _collapsed_latent(Q: torch.Tensor, d: int, seed: int, jitter: float = 6.0
                      ) -> torch.Tensor:
    """The pathology: every window position carries the CENTRE speed (no 100 ms resolution),
    plus large isotropic per-position noise. This is what a 51x jitter ratio looks like."""
    g = np.random.default_rng(seed)
    n = len(Q)
    base = np.zeros((n, W, d), dtype=np.float64)
    base[:, :, 0] = Q.numpy()[:, C][:, None]              # constant within the window
    base[:, :, 1:] = g.normal(0, 1, size=(n, 1, d - 1))   # window-constant scene
    return torch.from_numpy((base + g.normal(0, jitter, size=(n, W, d))).astype(np.float32))


def _splits(Z, Q, n_fit=260, n_sel=140):
    return (Z[:n_fit], Q[:n_fit], Z[n_fit:n_fit + n_sel], Q[n_fit:n_fit + n_sel],
            Z[:n_fit + n_sel], Q[:n_fit + n_sel], Z[n_fit + n_sel:], Q[n_fit + n_sel:])


# --------------------------------------------------------------------------- #
def test_savgol_first_derivative_is_exact_on_a_polynomial():
    """An order-2 SG first derivative reproduces d/dt of a quadratic exactly."""
    for half, order in ((1, 1), (2, 2), (4, 2), (4, 3)):
        c = LS.savgol_coeffs(half, order, 1, DT)
        x = (np.arange(-half, half + 1) * DT)
        y = 3.0 + 2.0 * x + (0.0 if order < 2 else 5.0 * x ** 2)
        assert float((c * y).sum()) == pytest.approx(2.0, abs=1e-8)


def test_savgol_order1_half1_is_the_centred_difference():
    c = LS.savgol_coeffs(1, 1, 1, DT)
    np.testing.assert_allclose(c, np.array([-1, 0, 1]) / (2 * DT), atol=1e-12)


def test_savgol_rejects_too_short_a_window():
    with pytest.raises(ValueError):
        LS.savgol_coeffs(1, 3)


# --------------------------------------------------------------------------- #
def test_oracle_latent_passes_the_screen():
    Q = _speed_tracks(600, seed=0)
    Z = _oracle_latent(Q, d=12, seed=1)
    res = LS.screen_latent(*_splits(Z, Q), name="oracle")
    assert res["verdict"] == "PASS", res["failed_screens"]
    assert res["screens"]["jitter_ratio"]["value"] < 2.0
    assert res["screens"]["derivative_corr"]["value"] > 0.5
    assert res["screens"]["derived_accel_r2"]["value"] > 0.5


def test_collapsed_latent_fails_on_jitter():
    """⭐ THE LOAD-BEARING TEST — the screen must reject a latent with no 100 ms resolution.

    ``dynenc-branchB`` spent 40 000 steps on such a latent. If this test ever passes a
    collapsed latent the instrument is worthless and the gate must not be trusted.
    """
    Q = _speed_tracks(600, seed=2)
    Z = _collapsed_latent(Q, d=12, seed=3)
    res = LS.screen_latent(*_splits(Z, Q), name="collapsed")
    assert res["verdict"] == "FAIL"
    assert "jitter_ratio" in res["failed_screens"]
    assert res["screens"]["jitter_ratio"]["value"] > 2.0
    # and the diagnostic must show the pathology's signature: near-copies within the window
    assert res["screens"]["cos_adjacent_100ms"]["value"] > \
        res["screens"]["cos_adjacent_100ms"]["cos_random_other_row"]


def test_collapsed_latent_can_still_read_scene_speed():
    """The pathology is NOT 'the latent cannot read speed'. A collapsed latent reads the
    SCENE's speed well while carrying no usable derivative — which is exactly why a speed-R²
    number alone can never be the gate."""
    Q = _speed_tracks(600, seed=4)
    Z = _collapsed_latent(Q, d=12, seed=5, jitter=3.0)
    res = LS.screen_latent(*_splits(Z, Q), name="collapsed_readable")
    assert res["fit"]["window_ridge_heldout_speed_r2_centre"] > 0.5
    assert res["verdict"] == "FAIL"


# --------------------------------------------------------------------------- #
def test_sigma_estimator_travels_with_every_sigma():
    """⚠️ REGRESSION GUARD. 'sigma <= 0.28 m/s' and 'sigma <= 0.1 m/s' are the SAME physical
    requirement under two different derivative estimators, and the programme has mis-quoted
    across them. Any dict that reports a sigma must carry its estimator."""
    Q = _speed_tracks(500, seed=6)
    Z = _oracle_latent(Q, d=8, seed=7)
    res = LS.screen_latent(*_splits(Z, Q, 220, 120), name="sigma")
    assert "Savitzky-Golay" in res["sigma_estimator"]
    assert "2-POINT CENTRED DIFFERENCE" in res["sigma_estimator"]
    assert res["screens"]["speed_sigma_mps"]["estimator"] == res["sigma_estimator"]
    assert res["screens"]["derived_accel_r2"]["estimator"] == res["sigma_estimator"]


def test_upper_bound_stencil_is_labelled_as_cheating():
    Q = _speed_tracks(500, seed=8)
    Z = _oracle_latent(Q, d=8, seed=9)
    res = LS.screen_latent(*_splits(Z, Q, 220, 120), name="ub")
    ub = res["screens"]["derived_accel_r2"]
    assert ub["upper_bound_best_stencil"] >= ub["value"] - 1e-9
    assert "CHEATING BY CONSTRUCTION" in ub["upper_bound_note"]


def test_gates_are_not_mutated_by_a_call():
    before = dict(LS.SCREEN_GATES)
    Q = _speed_tracks(400, seed=10)
    Z = _oracle_latent(Q, d=8, seed=11)
    LS.screen_latent(*_splits(Z, Q, 180, 100), name="x",
                     gates={**LS.SCREEN_GATES, "jitter_ratio_max": 99.0})
    assert LS.SCREEN_GATES == before


def test_shape_contract_is_enforced():
    Q = _speed_tracks(120, seed=12)
    Z = _oracle_latent(Q, d=8, seed=13)
    s = list(_splits(Z, Q, 60, 30))
    with pytest.raises(ValueError):
        LS.screen_latent(s[0][:, 0], *s[1:])                      # Z not [N, W, D]
    s2 = list(_splits(Z, Q, 60, 30))
    s2[7] = s2[7][:, :3]                                          # Q_hold W mismatch
    with pytest.raises(ValueError):
        LS.screen_latent(*s2)


def test_reference_record_carries_the_measured_frozen_v1_fail():
    """The calibration is thin (ONE encoder + an oracle) and must stay visible."""
    ref = LS.REFERENCE_LATENTS["flagship_v1_step29999_comma2k19"]
    assert ref["verdict"] == "FAIL"
    assert ref["jitter_ratio"] == pytest.approx(51.0)
    Q = _speed_tracks(400, seed=14)
    Z = _oracle_latent(Q, d=8, seed=15)
    res = LS.screen_latent(*_splits(Z, Q, 180, 100), name="calib")
    assert res["calibration_n_latents"] == len(LS.REFERENCE_LATENTS)
    assert "never by moving a threshold" in res["calibration_note"]


def test_format_screen_is_printable_and_names_failures():
    Q = _speed_tracks(400, seed=16)
    Z = _collapsed_latent(Q, d=8, seed=17)
    res = LS.screen_latent(*_splits(Z, Q, 180, 100), name="print_me")
    txt = LS.format_screen(res)
    assert "print_me" in txt and "FAIL" in txt and "jitter_ratio" in txt
