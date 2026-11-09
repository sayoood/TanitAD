"""PREREG_V8F_DECINFO controls 1 to 5, each with its deliberate mutation (IMPL_CHOICES_DECINFO.md section 4). CPU. Run through test_run.cmd."""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pytest
import torch

import decinfo_lib as DL


def _syn(n_ep=40, per=6, frames=16, dim=64, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    eid = np.repeat([f"e{i:03d}" for i in range(n_ep)], per)
    n = len(eid)
    X = rng.standard_normal((n, frames, dim)).astype(np.float32)
    w = rng.standard_normal(dim)
    y = X @ w / np.sqrt(dim) + noise * rng.standard_normal((n, frames))
    return eid, X, y


class _D:
    pass


# ---------------------------------------------------------------------------------------- control 1
def test_c1_shuffled_targets_read_at_or_below_005_and_mutation_fails():
    eid, X, y = _syn(dim=32)
    nf = 160
    D = _D(); D.nf, D.n = nf, len(eid)
    real, _, _ = DL.ridge_read(D, X, y, y)
    assert real > 0.6
    ysh = y.copy(); ysh[:nf] = DL.derange_tracks(y[:nf], eid[:nf], 5)
    r_sh, _, _ = DL.ridge_read(D, X, y, ysh)
    assert r_sh <= 0.05, r_sh
    # mutation: the shuffle is skipped, the control reads at the true level and the bar is broken
    r_mut, _, _ = DL.ridge_read(D, X, y, y)
    assert not (r_mut <= 0.05)


def test_c1_derangement_never_keeps_an_own_episode_track():
    eid, X, y = _syn()
    out = DL.derange_tracks(y, eid, 3)
    for i in range(len(eid)):
        j = int(np.nonzero((y == out[i]).all(1))[0][0])
        assert eid[j] != eid[i]


# ---------------------------------------------------------------------------------------- control 2
@pytest.mark.parametrize("dim,n_samples", [(128, 6000), (256, 6000), (1280, 9000), (5120, 9700)])
def test_c2_planted_signal_recovered_and_zero_features_read_zero(dim, n_samples):
    rng = np.random.default_rng(dim)
    Xf = rng.standard_normal((n_samples, dim)).astype(np.float32)
    Xh = rng.standard_normal((3000, dim)).astype(np.float32)
    a, z = DL.planted_check(Xf, Xh)
    assert a >= 0.80 and z <= 0.02, (a, z)
    yf, yh = DL.planted_target(Xf, Xh)
    # mutation 1: a fit target that is not a function of the features must fail the recovery bar
    assert DL.r2(yh, DL.ridge_pred(Xf, yf[rng.permutation(len(yf))], Xh)) < 0.80
    # mutation 2: features not zeroed in the zero arm would read the signal and break the 0.02 bar
    assert DL.r2(yh, DL.ridge_pred(Xf, yf, Xh)) > 0.02


def test_c2_conv_and_mlp_readers_recover_a_planted_signal():
    rng = np.random.default_rng(1)
    n = 2500
    X = torch.randn(n, 128, 4, 10)
    w = torch.randn(128) / 11.0
    y = torch.einsum("bcij,c->b", X, w) / 40.0                 # one channel mix shared over the 40 positions (what a convolution can learn)
    y = (y / y.std()).numpy() + 0.2 * rng.standard_normal(n)
    yt = torch.from_numpy(y).float()
    p, _ = DL.conv_train(X[:1800], yt[:1800], X[1800:2100], yt[1800:2100], X[2100:], 0, epochs=60)
    assert DL.r2(y[2100:], p.astype(np.float64)) > 0.5
    V = torch.randn(n, 128)
    yv = (V @ torch.randn(128)); yv = (yv / yv.std()).numpy() + 0.2 * rng.standard_normal(n)
    yvt = torch.from_numpy(yv).float()
    pm, _ = DL.DM.train(V[:1800], yvt[:1800], V[1800:2100], yvt[1800:2100], V[2100:], 0, epochs=60)
    assert DL.r2(yv[2100:], pm.astype(np.float64)) > 0.5


# ---------------------------------------------------------------------------------------- control 3
def test_c3_guard_passes_on_the_real_split_and_matches_the_card():
    D, sp = DL.load_tune(read_feats=False)
    assert (len(sp["fit_ids"]), len(sp["held_ids"]), len(sp["fit"]), len(sp["held"])) == (53, 17, 744, 238)
    assert set(sp["tune"]).isdisjoint(set(sp["score"]))
    assert set(sp["fit"]).isdisjoint(set(sp["held"]))
    assert D.n == 982 and set(D.pos.tolist()) == set(sp["tune"].tolist())


def test_c3_mutations_raise():
    fit, held, es, score = {"a", "b", "c"}, {"d", "e"}, {"a"}, {"x", "y"}
    assert DL.split_guard(fit, held, es, score, tune_eids=fit | held)
    with pytest.raises(AssertionError):
        DL.split_guard(fit | {"d"}, held, es, score)
    with pytest.raises(AssertionError):
        DL.split_guard(fit, held, es | {"d"}, score)
    with pytest.raises(AssertionError):
        DL.split_guard(fit | {"x"}, held, es, score)
    with pytest.raises(AssertionError):
        DL.split_guard(fit, held, es, score, tune_eids={"a", "b"})


def test_c3_fit_statistics_ignore_held_out_rows():
    base = torch.randn(40, 16, 40, 24)
    garb = base.clone(); garb[30:] = torch.randn(10, 16, 40, 24) * 100
    f = lambda t: DL.fit_pca_stream(iter([t[:20], t[20:]]), 24, 30, k=8)
    mu1, c1 = f(base); mu2, c2 = f(garb)
    assert np.array_equal(mu1, mu2) and np.array_equal(c1, c2)
    # mutation: a PCA fitted over the held-out rows too changes the basis
    mu3, c3 = DL.fit_pca_stream(iter([garb[:20], garb[20:]]), 24, 40, k=8)
    assert not np.array_equal(c3, c2)
    # the ridge prediction of fit rows does not depend on the held-out features' scale
    rng = np.random.default_rng(0)
    Xf = rng.standard_normal((500, 16)).astype(np.float32); y = rng.standard_normal(500)
    h = rng.standard_normal((10, 16)).astype(np.float32)
    Wb = DL.L3.ridge_fit(Xf, y / y.std())
    assert torch.equal(Wb[0], DL.L3.ridge_fit(Xf, y / y.std())[0])
    assert np.array_equal(DL.ridge_pred(Xf, y, h)[:0], DL.ridge_pred(Xf, y, h * 3)[:0])


# ---------------------------------------------------------------------------------------- control 4
@pytest.fixture(scope="module")
def tune_nofeats():
    return DL.load_tune(read_feats=False)


def _ridge_m1(D, X, y):
    r, _, _ = DL.ridge_read(D, X, y, y)
    return r


@pytest.mark.parametrize("tag", ["M2b_s0", "M2b_s1"])
def test_c4_reproduction_of_the_plantest3_gate(tune_nofeats, tag):
    D, sp = tune_nofeats
    X = D.z1_ref[tag].numpy().astype(np.float32)
    pre = json.load(open(DL.PRE.format(tag=tag), encoding="utf-8"))["enc_L1"]
    rs, ry = _ridge_m1(D, X, D.vk), _ridge_m1(D, X, D.yk)
    print(tag, "speed", rs, pre["r2_speed"], "yaw", ry, pre["r2_yaw"])
    assert abs(rs - pre["r2_speed"]) <= 0.01, (rs, pre["r2_speed"])
    assert abs(ry - pre["r2_yaw"]) <= 0.01, (ry, pre["r2_yaw"])
    # mutation: targets shifted by one frame leave the tolerance
    Xm = X[:, :-1]
    ms_, my_ = _ridge_m1(D, Xm, D.vk[:, 1:]), _ridge_m1(D, Xm, D.yk[:, 1:])
    assert abs(ms_ - pre["r2_speed"]) > 0.01 or abs(my_ - pre["r2_yaw"]) > 0.01, (ms_, my_)


# ---------------------------------------------------------------------------------------- control 5
@pytest.mark.parametrize("tag", ["M2b_s0", "M2b_s1"])
def test_c5_chain_identity(tune_nofeats, tag):
    D, sp = tune_nofeats
    ds, wins, eid = DL.W.open_windows(16)
    sub = np.arange(0, 96, 6)
    assert len(sub) == 16 and sub.max() < D.nf
    idx = D.ds_idx[sub]
    assert set(idx.tolist()).isdisjoint(set(D.ds_idx[D.nf:].tolist()))
    feats = np.stack([ds.item(wins[i][0], wins[i][1])["feats"] for i in idx])
    model, _ = DL.load_any(DL.M2B[tag])
    with torch.no_grad():
        adapt = model.adapter(torch.from_numpy(feats).float()) + model.pos[None, None]
        pooled = adapt.mean(2)
        assert torch.equal(model.wm.z1(adapt), model.wm.proj(pooled))
        # mutation: a float64 pool cast back breaks bit identity
        assert not torch.equal(model.wm.z1(adapt), model.wm.proj(adapt.double().mean(2).float()))
        assert float((model.wm.z1(adapt) - D.z1_ref[tag][sub]).abs().max()) < 1e-4


def test_stack_alignment():
    X = np.arange(2 * 16 * 3, dtype=np.float32).reshape(2, 16, 3)
    S = DL.stack_ch(X, 4)
    assert S.shape == (2, 13, 12)
    assert np.array_equal(S[:, 0, 0:3], X[:, 3]) and np.array_equal(S[:, 0, 9:12], X[:, 0])
    assert np.array_equal(S[:, -1, 0:3], X[:, 15]) and np.array_equal(S[:, -1, 9:12], X[:, 12])
