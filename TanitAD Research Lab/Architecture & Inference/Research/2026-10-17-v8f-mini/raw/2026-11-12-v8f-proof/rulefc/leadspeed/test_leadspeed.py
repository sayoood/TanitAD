"""Tests of leadspeed.py (card ADD148): label and clock units, and the smoke controls s2 (planted closing), s3 (permuted labels), s4 (causal) on synthetic data, CPU."""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import leadspeed as L  # noqa: E402


# ---------------------------------------------------------------- units
def test_token_labels_clock_and_slope():
    T = 12
    k = np.arange(0, 2 * T)                                  # 0.1 s clock
    gap = 30.0 - 2.0 * (k * 0.1)                             # closing -2 m/s
    pres, g, clos = L.token_labels(k, np.ones_like(k), np.ones_like(k), gap, T)
    assert pres.all() and abs(g[3] - (30.0 - 2.0 * 0.6)) < 1e-12        # token frame 3 = k 6
    assert np.isnan(clos[:2]).all() and np.allclose(clos[2:], -2.0)


def test_token_labels_gap_bounds_and_breaks():
    T = 8
    k = np.arange(0, 16)
    gap = np.full(16, 20.0)
    gap[8] = 70.0                                            # token frame 4 above 60 m: no lead
    pres, g, clos = L.token_labels(k, np.ones_like(k), np.ones_like(k), gap, T)
    assert not pres[4] and np.isnan(clos[4]) and np.isnan(clos[5]) and np.isnan(clos[6]) and clos[7] == 0.0


def test_clock_check_finds_planted_shift():
    v = np.sin(np.arange(0, 200) * 0.05) * 5 + 10            # 0.1 s clock
    k = np.arange(0, 200)
    vt = v[0::2][:100]
    r = L.clock_check(k, v, vt)
    assert min(r, key=lambda s: r[s][0]) == 0 and r[0][0] == 0.0
    vt2 = v[2::2][:99]                                       # token clock shifted by one token frame
    r2 = L.clock_check(k, v, vt2)
    assert min(r2, key=lambda s: r2[s][0]) == 1


# ---------------------------------------------------------------- planted data (s2, s3)
def planted(n_clips=160, T=40, n_tok=8, d=32, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, 2)))
    u1, u2 = q[:, 0], q[:, 1]
    X, P, G, C = [], [], [], []
    for _ in range(n_clips):
        w, ph, amp = rng.uniform(0.3, 1.0), rng.uniform(0, 2 * np.pi), rng.uniform(2.0, 10.0)
        t = np.arange(2 * T) * 0.1
        gap = 30.0 + amp * np.sin(w * t + ph)                 # 20 to 40 m, closing up to about 10 m/s
        k = np.arange(2 * T)
        pres, g, clos = L.token_labels(k, np.ones_like(k), np.ones_like(k), gap, T)
        x = rng.normal(scale=0.1, size=(T, n_tok, d))
        th = (g / L.GAP_SCALE) * (np.pi / 2)                  # the gap in the token's DIRECTION (a LayerNorm divides out its size)
        x[:, :4, :] += 2.0 * (np.cos(th)[:, None, None] * u1[None, None, :] + np.sin(th)[:, None, None] * u2[None, None, :])
        X.append(x.astype(np.float32)); P.append(pres); G.append(g); C.append(clos)
    return X, P, G, C


def samples(X, P, G, C, clips):
    xs, ps, gs, cs = [], [], [], []
    for i in clips:
        for t in L.windows(len(X[i])):
            xs.append(X[i][t - L.H + 1:t + 1]); ps.append(P[i][t]); gs.append(G[i][t]); cs.append(C[i][t])
    return (torch.tensor(np.stack(xs)), torch.tensor(np.array(ps, np.float32)), torch.tensor(np.array(gs, np.float32)), torch.tensor(np.array(cs, np.float32)))


def fit(head, data, steps=1500, bs=128, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
    x, p, g, c = data
    n = len(x)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), generator=gen)
        t = L.loss_terms(head(x[idx]), p[idx], g[idx], c[idx])
        loss = t["pres"] + t["gap"] + t["clos"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head


def held_out_closing(head, data):
    x, _, _, c = data
    with torch.no_grad():
        out = head(x)[:, 2].numpy() * L.CLOS_SCALE
    m = np.isfinite(c.numpy())
    return c.numpy()[m], out[m]


@pytest.fixture(scope="module")
def pdata():
    X, P, G, C = planted()
    return samples(X, P, G, C, range(0, 140)), samples(X, P, G, C, range(140, 160))


def test_s2_planted_closing_learned(pdata):
    tr, ho = pdata
    torch.manual_seed(1)
    h = fit(L.make_head(d_in=32, width=32, temporal=True), tr)
    y, p = held_out_closing(h, ho)
    torch.manual_seed(1)
    hb = fit(L.make_head(d_in=32, width=32, temporal=False), tr)
    yb, pb = held_out_closing(hb, ho)
    print("s2 temporal R2", L.r2(y, p), "no-temporal R2", L.r2(yb, pb))
    assert L.r2(y, p) >= 0.80
    assert L.r2(yb, pb) < 0.30


def test_s3_permuted_labels(pdata):
    tr, ho = pdata
    x, p, g, c = tr
    perm = torch.tensor(np.random.default_rng(148).permutation(len(c)))
    h = fit(L.make_head(d_in=32, width=32, temporal=True), (x, p, g, c[perm]))
    y, q = held_out_closing(h, ho)
    print("s3 permuted r", L.pearson(y, q))
    assert abs(L.pearson(y, q)) <= 0.10


def test_s4_causal():
    torch.manual_seed(0)
    h = L.make_head(d_in=16, width=16).eval()
    X = torch.randn(30, 4, 16)
    t = 20
    with torch.no_grad():
        a = h(X[t - L.H + 1:t + 1][None])
        X2 = X.clone()
        X2[t + 1:] = torch.randn(30 - t - 1, 4, 16) * 100
        b = h(X2[t - L.H + 1:t + 1][None])
    assert torch.equal(a, b)
    with torch.no_grad():                                    # the window's own frames do reach the output (the probe is not blind)
        X3 = X.clone()
        X3[t - 3] += 5.0
        c = h(X3[t - L.H + 1:t + 1][None])
    assert not torch.equal(a, c)
