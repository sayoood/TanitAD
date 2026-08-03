"""Contract tests for the capacity/architecture probe instrument.

The instrument's job is to adjudicate "unrecoverable vs unrecovered", so its own
correctness has to be established BEFORE any verdict is quoted from it:

* the dual ridge must equal a primal ridge solved independently (otherwise the
  "capacity ladder" is a bug, not a ladder);
* the probe loss must equal the SHIPPED ``idm_head.idm_loss`` when unmasked
  (otherwise an architecture sweep is secretly an objective sweep);
* the injection control must be detectable when injected and undetectable when
  not (otherwise the sensitivity floor it reports is meaningless);
* ``r2_score`` must return 0.0, not NaN, on a constant target — one comma2k19
  episode really does ship ``long_accel`` identically 0.0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.eval import accel_probe as AP  # noqa: E402


def _toy(n=200, w=9, d=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    Z = torch.randn(n, w, d, generator=g)
    y = Z[:, w // 2, 0] * 2.0 - Z[:, w // 2 + 1, 1]
    return Z, y


def test_r2_constant_target_is_zero_not_nan():
    assert AP.r2_score(np.zeros(10), np.zeros(10)) == 0.0
    assert AP.r2_score(np.arange(10.0), np.full(10, 3.0)) == 0.0
    assert AP.r2_score(np.arange(10.0), np.arange(10.0)) == 1.0


def test_window_features_shapes_and_diff_basis():
    Z, _ = _toy()
    n, w, d = Z.shape
    assert AP.window_features(Z, "centre").shape == (n, d)
    assert AP.window_features(Z, "window").shape == (n, w * d)
    # offsets 1, 2 and w//2 -> three blocks
    assert AP.window_features(Z, "diff").shape == (n, 3 * d)
    assert AP.window_features(Z, "centre_diff").shape == (n, 4 * d)
    # the diff basis really is antisymmetric window positions
    got = AP.window_features(Z, "diff")[:, :d]
    assert torch.allclose(got, Z[:, w // 2 + 1] - Z[:, w // 2 - 1])


def test_dual_ridge_equals_primal_ridge():
    """The dual solution must reproduce an independently solved primal ridge."""
    torch.manual_seed(0)
    n, d = 60, 8
    X = torch.randn(n, d, dtype=torch.float64)
    Y = (X @ torch.randn(d, 2, dtype=torch.float64)
         + 0.1 * torch.randn(n, 2, dtype=torch.float64))
    Xte = torch.randn(25, d, dtype=torch.float64)
    alpha = 0.7
    # primal on CENTRED features with an intercept from the target mean
    mu = X.mean(0, keepdim=True)
    Xc, Yc = X - mu, Y - Y.mean(0, keepdim=True)
    Wp = torch.linalg.solve(Xc.T @ Xc + alpha * torch.eye(d, dtype=torch.float64),
                            Xc.T @ Yc)
    want = (Xte - mu) @ Wp + Y.mean(0, keepdim=True)
    got = AP.DualRidge(X, Y, kernel="linear").predict(Xte, alpha)
    assert torch.allclose(got, want, atol=1e-8), (got - want).abs().max()


def test_dual_ridge_fp32_gram_matches_the_exact_fp64_reference():
    """The fast Gram path must not move a prediction beyond fp32 rounding."""
    torch.manual_seed(7)
    X = torch.randn(120, 40, dtype=torch.float64)
    Y = X @ torch.randn(40, 3, dtype=torch.float64)
    Xte = torch.randn(30, 40, dtype=torch.float64)
    exact = AP.DualRidge(X, Y, kernel="linear").predict(Xte, 1.0)
    fast = AP.DualRidge(X, Y, kernel="linear", matmul_device="cpu",
                        matmul_dtype=torch.float32).predict(Xte, 1.0)
    rel = float((exact - fast).abs().max() / exact.abs().max())
    assert rel < 1e-4, rel


def test_dual_ridge_alpha_path_is_monotone_in_shrinkage():
    torch.manual_seed(1)
    X = torch.randn(80, 6, dtype=torch.float64)
    Y = X @ torch.randn(6, 1, dtype=torch.float64)
    r = AP.DualRidge(X, Y, kernel="linear")
    norms = [float(r.predict(X, a).std()) for a in (1e-6, 1.0, 1e3, 1e8)]
    assert norms == sorted(norms, reverse=True), norms
    assert norms[-1] < 1e-3 * norms[0]          # alpha -> inf collapses to mean


def test_rbf_kernel_ridge_beats_linear_on_a_nonlinear_target():
    torch.manual_seed(2)
    X = torch.randn(300, 3, dtype=torch.float64)
    Y = (X[:, :1] ** 2 + torch.sin(3 * X[:, 1:2]))
    tr, te = slice(0, 200), slice(200, 300)
    lin = AP.DualRidge(X[tr], Y[tr], kernel="linear")
    rbf = AP.DualRidge(X[tr], Y[tr], kernel="rbf", gamma=0.5)
    best_l = max(AP.r2_score(lin.predict(X[te], a), Y[te])
                 for a in AP.DualRidge.alpha_grid())
    best_r = max(AP.r2_score(rbf.predict(X[te], a), Y[te])
                 for a in AP.DualRidge.alpha_grid())
    assert best_r > best_l + 0.2, (best_r, best_l)


def test_probe_loss_matches_shipped_idm_loss_when_unmasked():
    import idm_head as ih
    torch.manual_seed(3)
    b, d = 16, 10
    pred = {"scalars": torch.randn(b, 4), "traj": torch.randn(b, 4, 2)}
    scal, traj = torch.randn(b, 4), torch.randn(b, 4, 2)
    std = ih.Standardizer.fit(torch.randn(64, 4))
    a = ih.idm_loss(pred, scal, traj, std)["loss"]
    c = AP.probe_loss(pred, scal, traj, std)["loss"]
    assert torch.allclose(a, c, atol=1e-7), (float(a), float(c), d)


def test_probe_loss_mask_selects_only_that_channel():
    import idm_head as ih
    torch.manual_seed(4)
    b = 12
    pred = {"scalars": torch.randn(b, 4, requires_grad=True),
            "traj": torch.randn(b, 4, 2)}
    scal, traj = torch.randn(b, 4), torch.randn(b, 4, 2)
    std = ih.Standardizer.fit(torch.randn(64, 4))
    m = torch.tensor([0.0, 0.0, 0.0, 1.0])
    AP.probe_loss(pred, scal, traj, std, mask=m, traj_weight=0.0)["loss"].backward()
    g = pred["scalars"].grad
    assert float(g[:, :3].abs().max()) == 0.0
    assert float(g[:, 3].abs().max()) > 0.0


def test_inject_signal_is_detectable_and_absent_when_not_injected():
    """The control must FIRE when the answer is planted and stay silent when not."""
    torch.manual_seed(5)
    n, w, d = 400, 9, 32
    Z = torch.randn(n, w, d, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)               # unrelated to Z
    tr, te = slice(0, 300), slice(300, n)

    def probe(Zin):
        X = AP.window_features(Zin, "centre")
        Xtr, Xte = AP.standardize(X[tr], X[te])
        r = AP.DualRidge(Xtr, y[tr, None], kernel="linear")
        return max(AP.r2_score(r.predict(Xte, a), y[te])
                   for a in AP.DualRidge.alpha_grid())

    assert probe(Z) < 0.15, "an unrelated target must not be recoverable"
    Zi, meta = AP.inject_signal(Z, y.numpy(), frac=0.5, seed=0)
    assert meta["injected_rms"] > 0
    assert probe(Zi) > 0.8, "a planted signal must be recovered"


def test_inject_signal_shared_gain_across_two_blocks():
    """Train and held-out blocks must get the SAME direction and the SAME gain."""
    torch.manual_seed(9)
    A = torch.randn(50, 5, 8, dtype=torch.float64)
    B = torch.randn(30, 5, 8, dtype=torch.float64) * 3.0        # different scale
    ya, yb = torch.randn(50, dtype=torch.float64), torch.randn(30, dtype=torch.float64)
    mu, sd = float(ya.mean()), float(ya.std())
    rms = float(A.pow(2).mean().sqrt())
    Ai, ma = AP.inject_signal(A, ((ya - mu) / sd).numpy(), 0.2, seed=1,
                              standardize_y=False, rms=rms)
    Bi, mb = AP.inject_signal(B, ((yb - mu) / sd).numpy(), 0.2, seed=1,
                              standardize_y=False, rms=rms)
    assert ma["latent_rms"] == mb["latent_rms"] == rms
    # the added component must be the same direction: recover it and compare
    ua = (Ai - A)[:, 2][0]
    ub = (Bi - B)[:, 2][0]
    cos = float(torch.dot(ua, ub) / (ua.norm() * ub.norm()))
    assert abs(abs(cos) - 1.0) < 1e-9, cos


def test_mlp_and_gru_heads_honour_the_output_contract():
    for make in (lambda: AP.MLPHead(12, hidden=16, depth=2, mode="centre"),
                 lambda: AP.MLPHead(12, hidden=16, depth=1, mode="window"),
                 lambda: AP.GRUHead(12, d_model=8, layers=1)):
        mod = make()
        out = mod(torch.randn(5, 9, 12))
        assert out["scalars"].shape == (5, 4)
        assert out["traj"].shape == (5, 4, 2)


def test_fit_probe_head_learns_a_recoverable_scalar():
    """End-to-end: the generic trainer must drive a learnable target's R² up."""
    torch.manual_seed(6)
    n, w, d = 512, 9, 16
    Z = torch.randn(n, w, d)
    scal = torch.stack([Z[:, 4, 0], Z[:, 4, 1], Z[:, 4, 2], Z[:, 4, 3]], 1)
    traj = torch.randn(n, 4, 2) * 0.01
    tr, te = slice(0, 400), slice(400, n)
    mod, _ = AP.fit_probe_head(
        lambda: AP.MLPHead(d, hidden=64, depth=2, mode="centre"),
        (Z[tr], scal[tr], traj[tr]), epochs=40, batch=64, lr=3e-3)
    ps, _ = AP.predict_head(mod, Z[te])
    assert AP.r2_score(ps[:, 0].numpy(), scal[te, 0].numpy()) > 0.8
