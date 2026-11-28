"""Fast synthetic unit tests for fullres_probe.py (CPU only: run with CUDA_VISIBLE_DEVICES=-1)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fullres_probe as FP  # noqa: E402
import resprobe as RP  # noqa: E402


def test_pool_maps_lo():
    r, c = np.meshgrid(np.arange(16), np.arange(40), indexing="ij")
    x = torch.tensor((r * 100 + c).reshape(1, 640, 1), dtype=torch.float32)
    u = FP.pool_tokens(x, "U640")
    assert u.shape == (1, 640, 1)
    l40 = FP.pool_tokens(x, "L40")
    assert l40.shape == (1, 40, 1)
    for rb in range(4):
        for cb in range(10):
            want = np.mean([i * 100 + j for i in range(rb * 4, rb * 4 + 4) for j in range(cb * 4, cb * 4 + 4)])
            assert abs(float(l40[0, rb * 10 + cb, 0]) - want) < 1e-4
    l160 = FP.pool_tokens(x, "L160")
    assert l160.shape == (1, 160, 1)
    for rb in range(8):
        for cb in range(20):
            want = np.mean([i * 100 + j for i in range(rb * 2, rb * 2 + 2) for j in range(cb * 2, cb * 2 + 2)])
            assert abs(float(l160[0, rb * 20 + cb, 0]) - want) < 1e-4


def test_pool_maps_hi_center_crop():
    r, c = np.meshgrid(np.arange(26), np.arange(64), indexing="ij")
    x = torch.tensor((r * 100 + c).reshape(1, 1664, 1), dtype=torch.float32)
    c640 = FP.pool_tokens(x, "C640")
    assert c640.shape == (1, 640, 1)
    for k in (0, 39, 40, 321, 639):
        assert int(c640[0, k, 0]) == (5 + k // 40) * 100 + (12 + k % 40)
    assert int(c640[0, 0, 0]) == 512 and int(c640[0, 639, 0]) == 2051
    assert FP.pool_tokens(x, "H1664").shape == (1, 1664, 1)


def test_row_mapping():
    clips = ["a", "b", "c"]
    cpos = {c: i for i, c in enumerate(clips)}
    row0, T = np.array([0, 101, 202]), np.array([101, 101, 50])
    rows, ok = FP.map_rows(["b", "c", "c", "zzz", "a", "b"], np.array([3, 49, 50, 0, 100, 101]), cpos, row0, T)
    assert list(ok) == [True, True, False, False, True, False]
    assert rows[0] == 104 and rows[1] == 251 and rows[4] == 100


def test_masked_loss_ignores_masked_rows():
    torch.manual_seed(0)
    out = torch.randn(10, 3)
    y = torch.randn(10, 3)
    y[:, 2] = (y[:, 2] > 0).float()
    m = torch.ones(10, 3, dtype=torch.bool)
    m[5:, 0] = False
    m[:, 1] = False
    l1 = FP.masked_loss(out, y, m, [False, False, True])
    out2, y2 = out.clone(), y.clone()
    out2[5:, 0] = 99.0
    y2[5:, 0] = -99.0
    out2[:, 1] = 7.0
    l2 = FP.masked_loss(out2, y2, m, [False, False, True])
    assert abs(float(l1) - float(l2)) < 1e-7
    want = (torch.nn.functional.mse_loss(out[:5, 0], y[:5, 0]) + torch.nn.functional.binary_cross_entropy_with_logits(out[:, 2], y[:, 2])) / 2
    assert abs(float(l1) - float(want)) < 1e-6
    # no observed target at all: zero loss with a graph
    assert float(FP.masked_loss(out, y, torch.zeros(10, 3, dtype=torch.bool), [False, False, True])) == 0.0


def _planted(n, T, C, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T, C)).astype(np.float32)
    flag = rng.random(n) < 0.5
    v = rng.standard_normal(C)
    v = (v / np.linalg.norm(v) * np.sqrt(C)).astype(np.float32)
    pos = rng.integers(0, T, n)
    for i in np.flatnonzero(flag):
        X[i, pos[i]] += v
    return X, flag.astype(np.float64)


def test_attentive_probe_recovers_planted_object_and_beats_mean_pool():
    n, T, C = 5000, 48, 64
    X, y = _planted(n, T, C, 1)
    E = np.random.default_rng(2).standard_normal((n, 8))
    epi = np.arange(n) // 7
    tr, te = np.arange(0, 4000), np.arange(4000, n)
    Xt = torch.tensor(X)
    Y, M = y[:, None], np.ones((n, 1), bool)
    pa, _ = FP.p3_train_predict(lambda ib: Xt[ib], C, E, Y, M, [True], tr, te, epi, seed=0, epochs=40, dev="cpu")
    Xm = Xt.mean(1, keepdim=True)
    pm, _ = FP.p3_train_predict(lambda ib: Xm[ib], C, E, Y, M, [True], tr, te, epi, seed=0, epochs=40, dev="cpu")
    a_att, a_mean = RP.auc_pooled(y[te], pa[:, 0]), RP.auc_pooled(y[te], pm[:, 0])
    assert a_att >= 0.95, a_att
    assert a_mean < a_att - 0.10, (a_att, a_mean)


def test_multitask_regression_masked_targets_fit():
    n, T, C = 3000, 16, 8
    rng = np.random.default_rng(3)
    X = rng.standard_normal((n, T, C)).astype(np.float32)
    E = rng.standard_normal((n, 8))
    y0 = X[:, :, 0].mean(1).astype(np.float64) * 8.0 + 5.0
    Y = np.stack([y0, np.zeros(n)], 1)
    M = np.stack([np.ones(n, bool), np.zeros(n, bool)], 1)
    Xt = torch.tensor(X)
    p, used = FP.p3_train_predict(lambda ib: Xt[ib], C, E, Y, M, [False, False], np.arange(2400), np.arange(2400, n), np.arange(n) // 5, seed=0, epochs=40, dev="cpu")
    assert p.shape == (600, 2) and np.isfinite(p).all()
    assert abs(p[:, 0].mean() - 5.0) < 1.0           # de-standardised to the original units
    assert RP.r2_pooled(y0[2400:], p[:, 0]) > 0.3


def test_plant_add_positions_and_l40_dilution():
    n, T, C = 4, 640, 3
    ts = FP.TokSet("U640", torch.zeros(n, T, C, dtype=torch.int8), torch.ones(C))
    plant = {"flag": torch.tensor([True, False, True, False]), "pos": torch.tensor([5, 7, 639, 0]), "vec_white": torch.tensor([1.0, 2.0, 3.0]), "sigma": torch.ones(C) * 2}
    x = ts.get(torch.arange(n), "white", plant)
    assert torch.allclose(x[0, 5], torch.tensor([1.0, 2.0, 3.0])) and torch.allclose(x[2, 639], torch.tensor([1.0, 2.0, 3.0]))
    assert float(x[1].abs().sum()) == 0 and float(x[3].abs().sum()) == 0
    xt = ts.get(torch.arange(n), "true", plant)
    assert torch.allclose(xt[0, 5], torch.tensor([2.0, 4.0, 6.0]))
    ts40 = FP.TokSet("L40", torch.zeros(n, 40, C, dtype=torch.float16), torch.ones(C))
    x4 = ts40.get(torch.arange(n), "white", plant)
    # pos 639 = row 15, col 39 -> block (3, 9) -> token 39 ; pos 5 = row 0, col 5 -> block (0, 1) -> token 1; each carries vec / 16
    assert torch.allclose(x4[2, 39], torch.tensor([1.0, 2.0, 3.0]) / 16) and torch.allclose(x4[0, 1], torch.tensor([1.0, 2.0, 3.0]) / 16)


def test_bootstrap_pairing():
    rng = np.random.default_rng(0)
    n_ep, per = 30, 20
    epi = np.repeat(np.arange(n_ep), per)
    y = rng.standard_normal(len(epi))
    good = y + 0.1 * rng.standard_normal(len(y))
    bad = y + 1.0 * rng.standard_normal(len(y))
    stop = np.zeros(len(y), bool)
    stop[::7] = True
    b = FP.Boot(epi, n_ep, stop, B=300)
    same = b.pair("r2", y, good, good)
    assert same["diff"] == 0.0 and same["ci95"] == [0.0, 0.0] and not same["separated_pos"] and not same["separated_neg"]
    d = b.pair("r2", y, good, bad)
    assert d["diff"] > 0 and d["separated_pos"] and not d["separated_neg"]
    d2 = b.pair("r2", y, bad, good)
    assert abs(d2["diff"] + d["diff"]) < 1e-12 and d2["separated_neg"]
    yb = (y > 0).astype(float)
    assert b.pair("auc", yb, good, bad)["separated_pos"]
    # STOPDIST form: positive = a (smaller error) better; identical predictions give 0
    ds = np.where(stop, np.abs(y) * 10 + 5, np.nan)
    pa, pb = np.where(stop, ds + 0.2, np.nan), np.where(stop, ds + 3.0, np.nan)
    r = b.pair("mae", ds, pa, pb)
    assert r["diff"] > 2.5 and r["separated_pos"]
    assert b.pair("mae", ds, pa, pa)["diff"] == 0.0
    # the same draw is shared: two calls give identical intervals
    assert b.pair("r2", y, good, bad)["ci95"] == d["ci95"]


def test_fine_over_coarse_rule():
    pos = {"separated_pos": True, "separated_neg": False}
    neg = {"separated_pos": False, "separated_neg": True}
    flat = {"separated_pos": False, "separated_neg": False}
    assert FP.decide_fine_over_coarse(pos, flat) and FP.decide_fine_over_coarse(pos, pos)
    assert not FP.decide_fine_over_coarse(pos, neg) and not FP.decide_fine_over_coarse(flat, flat)


def test_gram_matches_numpy_and_symmetric():
    rng = np.random.default_rng(0)
    X = rng.integers(-100, 100, (50, 6, 128)).astype(np.int8)
    scale = rng.random(128).astype(np.float32) + 0.5
    ts = FP.TokSet("U640", torch.tensor(X), torch.tensor(scale))
    K = FP.gram_gpu(ts, blk=16)
    A = (X.astype(np.float32) * scale).reshape(50, -1)
    mx = np.abs(A @ A.T).max()
    assert np.abs(K - A @ A.T).max() < 1e-5 * mx and np.abs(K - K.T).max() < 1e-5 * mx
