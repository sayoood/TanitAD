"""CAPLADDER controls on synthetic data (PREREG_V8F_CAPLADDER control 3 a/b/c, the dimension and calibration logic, control 4 logic, the dual ridge against the primal).
Smoke size: R = 39 reference draws, 6 calibration draws (20 for the calibration test), B = 500. CPU only (CUDA_VISIBLE_DEVICES=-1). Run: pytest -q test_capladder_controls.py"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pytest
import torch

import capladder_lib as CL

R_T, NC_T, B_T = 39, 6, 500


class World:
    """n_ep episodes x per frames; y has an episode level plus a frame part (clustered, like lead_gap)."""

    def __init__(self, seed=0, n_ep=40, per=8):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.n_ep, self.per = n_ep, per
        self.ep = np.repeat([f"e{i:03d}" for i in range(n_ep)], per)
        a = rng.normal(size=n_ep)
        self.y = np.repeat(a, per) * 0.8 + rng.normal(size=n_ep * per) * 0.6
        self.ys = (self.y - self.y.mean()) / self.y.std()
        self.n = n_ep * per
        self.fold = CL.outer_fold(self.ep)
        self.inner = CL.inner_fold(self.ep)
        self.uniq = np.unique(self.ep)
        self.pos = np.searchsorted(self.uniq, self.ep)
        self.W = CL.boot_weights_local(self.uniq, B_T, 0)
        self.rows = np.arange(self.n)

    def rung(self, X, R=R_T, NC=NC_T):
        K5 = CL.gram_from_matrix(X, {"A": (self.rows, self.fold)})["A"]
        rg, res = CL.rung_from_K(K5, self.rows, self.y, self.ep, self.fold, self.inner, self.pos, self.n_ep, self.W, R, NC)
        return rg


def tokens(w, dt=2, seed=1):
    return torch.as_tensor(np.random.default_rng(seed).normal(size=(w.n, 640, dt)).astype(np.float32))


def flat(t):
    return t.reshape(t.shape[0], -1).numpy()


def checker():
    r, c = np.meshgrid(np.arange(16), np.arange(40), indexing="ij")
    return torch.as_tensor(((-1.0) ** (r + c)).reshape(-1).astype(np.float32))


# ------------------------------------------------------------------------------------------------ units
def test_folds_and_boot():
    w = World()
    assert set(np.unique(w.fold)) <= set(range(5)) and len(np.unique(w.fold)) >= 3
    assert (w.fold == CL.outer_fold(w.ep)).all()                                         # deterministic
    assert (w.W.sum(1) == w.n_ep).all()


def test_perm_labels_is_a_derangement_of_episodes():
    w = World()
    F = np.flatnonzero(w.fold != 0)
    yp = CL.perm_labels(w.y[F], w.ep[F], np.random.default_rng(3))
    assert not np.allclose(yp, w.y[F])
    for e in np.unique(w.ep[F]):
        assert not np.allclose(yp[w.ep[F] == e], w.y[F][w.ep[F] == e])                   # no episode keeps its own sequence


def test_dual_ridge_equals_primal_ridge():
    rng = np.random.default_rng(0)
    n, d = 90, 12
    X = rng.normal(size=(n, d)) * rng.uniform(0.5, 3.0, size=d) + rng.normal(size=d)
    y = X[:, 0] * 0.7 - X[:, 3] + rng.normal(size=n)
    fold = np.arange(n) % 5
    K5 = CL.gram_from_matrix(X, {"A": (np.arange(n), fold)})["A"]
    k = 2
    F, S = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    sp = CL.Split(K5[k], F, S, y[F][:, None])
    mu, sd = X[F].mean(0), X[F].std(0) + 1e-8
    Xs = (X - mu) / sd
    for c in (1e-2, 1.0, 100.0):
        lam = c * np.trace(Xs[F] @ Xs[F].T) / len(F)
        w_ = np.linalg.solve(Xs[F].T @ Xs[F] + lam * np.eye(d), Xs[F].T @ (y[F] - y[F].mean()))
        prim = Xs[S] @ w_ + y[F].mean()
        assert np.allclose(prim, sp.predict(c)[:, 0], atol=2e-4), np.abs(prim - sp.predict(c)[:, 0]).max()


# ------------------------------------------------------------------------------------------------ control 3 (planted)
def test_control3a_planted_channel_in_R2_is_readable():
    w = World()
    t = tokens(w)
    r2 = CL.pool_field(t, 4, 10)
    r2[:, 0, 0] += torch.as_tensor(w.ys.astype(np.float32))                              # SNR 1: unit signal variance on a unit-variance feature
    plain = CL.pool_field(tokens(w, seed=9), 4, 10)
    assert w.rung(flat(r2)).point[0] < 0.8
    ra = w.rung(flat(r2))
    assert CL.readable_row(ra)[0] is True
    assert CL.readable_row(w.rung(flat(plain)))[0] is False                              # regression arm: no plant -> not readable


def test_control3b_checkerboard_gives_loss_at_the_pooling_step_it_crosses():
    w = World()
    t = tokens(w)
    sig = torch.as_tensor(w.ys.astype(np.float32))
    t0 = t.clone()
    t0[:, :, 0] += 0.5 * sig[:, None] * checker()[None, :]
    r0, r1, r2 = flat(t0), flat(CL.pool_field(t0, 8, 20)), flat(CL.pool_field(t0, 4, 10))
    g0, g1, g2 = w.rung(r0), w.rung(r1), w.rung(r2)
    assert np.abs(r1[:, ::2]).max() < 20 and g0.point[0] < 0.5 and g1.point[0] > 0.9      # the signal is gone after 2 x 2 averaging
    assert CL.readable_row(g0)[0] and not CL.readable_row(g1)[0] and not CL.readable_row(g2)[0]
    assert CL.loss_row(g0, g1)[0] is True                                                # LOSS at the 16x40 -> 8x20 step
    assert CL.loss_row(g1, g2)[0] is False                                               # nothing left to lose afterwards
    # deliberate-regression arm: a pattern that survives averaging (constant sign) gives NO loss
    t1 = t.clone()
    t1[:, :, 0] += 0.5 * sig[:, None]
    h0, h1 = w.rung(flat(t1)), w.rung(flat(CL.pool_field(t1, 8, 20)))
    assert CL.readable_row(h0)[0] and CL.readable_row(h1)[0] and CL.loss_row(h0, h1)[0] is False


def test_control3c_same_channel_on_every_rung_gives_no_loss():
    w = World()
    ch = (w.ys + np.random.default_rng(5).normal(size=w.n)).astype(np.float32)           # SNR 1 channel, identical values on every rung
    rungs = []
    for i in range(5):                                                                    # R0 .. R4: the channel + 63 independent noise features (identical width)
        X = np.random.default_rng(100 + i).normal(size=(w.n, 64)).astype(np.float32)
        X[:, 0] = ch
        rungs.append(w.rung(X))
    assert all(CL.readable_row(r)[0] for r in rungs)
    for a in range(4):
        assert CL.loss_row(rungs[a], rungs[a + 1])[0] is False, a
    assert CL.loss_row(rungs[1], rungs[4])[0] is False


# ------------------------------------------------------------------------------------------------ control 2 (dimension)
def test_control2_gaussian_is_not_readable_and_a_planted_matrix_is_caught():
    w = World()
    for d in (32, 400, 2000):
        G = np.random.default_rng(0).normal(size=(w.n, d)).astype(np.float32)
        assert CL.readable_row(w.rung(G))[0] is False, d
    P = np.random.default_rng(0).normal(size=(w.n, 400)).astype(np.float32)
    P[:, 0] = w.ys + np.random.default_rng(1).normal(size=w.n) * 0.5
    caught = CL.readable_row(w.rung(P))[0]
    assert caught is True                                                                # regression arm: control 2's check would FAIL on a matrix that carries the label


# ------------------------------------------------------------------------------------------------ control 1 (calibration)
def test_control1_calibration_share_is_small_and_no_derangement_fails():
    w = World()
    rungs = {f"R{i}": w.rung(np.random.default_rng(200 + i).normal(size=(w.n, 60 + 30 * i)).astype(np.float32), R=R_T, NC=20) for i in range(4)}
    pairs = [("R0", "R1"), ("R1", "R2"), ("R2", "R3")]
    out = CL.calibration_share(rungs, pairs)
    assert out["READABLE"]["rows"] == 80 and out["LOSS"]["rows"] == 60
    assert out["READABLE"]["share"] <= 0.05 and out["LOSS"]["share"] <= 0.05, out
    # deliberate-regression arm: derangement switched off (the "calibration" columns carry the REAL labels of a planted world) -> the share explodes -> control 1 FAILS
    X = np.random.default_rng(7).normal(size=(w.n, 60)).astype(np.float32)
    X[:, 0] = w.ys + 0.5 * np.random.default_rng(8).normal(size=w.n)
    planted = {"P0": w.rung(X, R=R_T, NC=20), "P1": w.rung(X.copy(), R=R_T, NC=20)}
    for rg in planted.values():
        for c in rg.cal_cols():
            rg.point[c] = rg.point[0]
            rg.boot[:, c] = rg.boot[:, 0]
    bad = CL.calibration_share(planted, [("P0", "P1")])
    assert bad["READABLE"]["share"] > 0.05


# ------------------------------------------------------------------------------------------------ control 4 (R0 identity logic)
def test_control4_pool_identity_and_logic():
    rng = np.random.default_rng(0)
    f = torch.as_tensor(rng.normal(size=(7, 640, 16)).astype(np.float32)) * 3
    p8, p4 = CL.pool_field(f, 8, 20), CL.pool_field(f, 4, 10)
    ref8 = f.reshape(7, 8, 2, 20, 2, 16).mean(dim=(2, 4)).reshape(7, 160, 16)
    assert torch.equal(p8, ref8)
    c8, c4 = p8.to(torch.float8_e4m3fn).float(), p4.to(torch.float8_e4m3fn).float()
    e8, e4 = CL.rel_l2(p8, c8), CL.rel_l2(p4, c4)
    assert e8.max() < 0.08 and e4.max() < 0.08 and CL.control4_pass(e8, e4)              # fp8 rounding alone stays inside the card's 0.08
    sh = torch.roll(p8, 1, dims=1)                                                        # a one-cell shift (wrong grid / cell order) fails
    assert CL.rel_l2(sh, c8).min() > 0.5 and not CL.control4_pass(CL.rel_l2(sh, c8), e4)
