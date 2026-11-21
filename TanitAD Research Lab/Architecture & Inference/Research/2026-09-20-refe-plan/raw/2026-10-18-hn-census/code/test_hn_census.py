#!/usr/bin/env python3
"""Unit tests + controls for hn_census.py / hn_load.py (PREREG s5).  Plain asserts; run:
    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I test_hn_census.py
Exit code 0 only if every test passes.  run_mutants.py re-runs this file against mutated copies and requires each mutant to turn a named test RED.
"""
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hn_census as H  # noqa: E402
import hn_load as L  # noqa: E402

T = 20


def straight(v=1.0):
    """expert/proposal straight along x at v m/s, 0.2 s grid, [T,3]"""
    t = np.arange(1, T + 1) * 0.2
    return np.stack([v * t, np.zeros(T), np.zeros(T)], axis=-1).astype(np.float32)


def test_ade_fde():
    E = straight(5.0)[None]                                           # [1,T,3]
    P = np.stack([straight(5.0)[:, :2],                               # identical
                  straight(5.0)[:, :2] + np.array([0.0, 0.5]),        # constant 0.5 m lateral offset
                  straight(6.0)[:, :2]])[None]                        # speed 6 vs 5: gap grows with time
    ade, fde = H.ade_fde(P, E)
    t = np.arange(1, T + 1) * 0.2
    assert abs(ade[0, 0]) < 1e-9 and abs(fde[0, 0]) < 1e-9
    assert abs(ade[0, 1] - 0.5) < 1e-6 and abs(fde[0, 1] - 0.5) < 1e-6
    assert abs(ade[0, 2] - (1.0 * t).mean()) < 1e-6, ade[0, 2]        # mean over ALL 20 poses, not pose 0
    assert abs(fde[0, 2] - 4.0) < 1e-6


def test_counts_inclusive():
    dist = np.array([[0.5, 0.5000001, 0.49, 3.0]])
    hn = np.array([[True, True, False, True]])
    npx, nhn = H.counts_at(dist, hn, 0.5)
    assert int(npx[0]) == 2 and int(nhn[0]) == 1, (npx, nhn)          # 0.5 is INCLUDED, 0.5000001 is not


def test_hard_negative_definition():
    nc = np.array([1.0, 0.0, 0.5, 1.0])
    dac = np.array([0.0, 1.0, 1.0, 1.0])
    pdms = np.array([0.0, 0.0, 0.7, 0.9])
    hn = H.hard_negative(nc, dac, pdms)
    assert hn.tolist() == [True, True, False, False], hn               # DAC-only, NC-only are hard negatives; NC=0.5 is not


def test_band_mask():
    d = np.array([0.0, 0.5, 0.5000001, 1.0, 4.0, 9.0])
    assert H.band_mask(d, 0.0, 0.5).tolist() == [True, True, False, False, False, False]
    assert H.band_mask(d, 0.5, 1.0).tolist() == [False, False, True, True, False, False]
    assert H.band_mask(d, 4.0, float("inf")).tolist() == [False, False, False, False, False, True]


def test_expert_class():
    stat = np.zeros((1, T, 3), np.float32)
    left = straight(5.0)[None].copy(); left[0, -1, 2] = np.radians(40)
    right = straight(5.0)[None].copy(); right[0, -1, 2] = np.radians(-40)
    st = straight(5.0)[None].copy(); st[0, -1, 2] = np.radians(5)
    E = np.concatenate([stat, left, right, st])
    cls, Lp, yaw = H.expert_class(E)
    assert cls.tolist() == ["STATIONARY", "LEFT", "RIGHT", "STRAIGHT"], cls


def test_boot_matches_bruteforce():
    rng = np.random.default_rng(1)
    li = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    num = rng.random(9); den = rng.random(9) + 0.5
    B = H.Boot(li, 3, B=500, seed=7)
    r = B.ratio(num, den)
    sn = np.array([num[li == k].sum() for k in range(3)]); sd = np.array([den[li == k].sum() for k in range(3)])
    v = np.array([sn[B.draws[b]].sum() / sd[B.draws[b]].sum() for b in range(500)])
    assert abs(r["point"] - sn.sum() / sd.sum()) < 1e-12
    assert abs(r["lo"] - np.percentile(v, 2.5)) < 1e-12 and abs(r["hi"] - np.percentile(v, 97.5)) < 1e-12
    # the draw-count matrix is consistent with the draws
    assert (B.C.sum(1) == 3).all() and (B.C[0] == np.bincount(B.draws[0], minlength=3)).all()


def test_hist_quantile_matches_numpy():
    rng = np.random.default_rng(3)
    li = rng.integers(0, 5, 4000)
    v = rng.lognormal(0.0, 0.8, 4000)
    B = H.Boot(li, 5, B=200, seed=11)
    Hh = B.per_log_hist(v, li)
    q = B.hist_quantiles(Hh)
    for k, name in zip(H.QS, ("q05", "q25", "q50", "q75", "q95")):
        assert abs(q[name]["point"] - np.quantile(v, k)) < 3 * H.BIN, (name, q[name]["point"], np.quantile(v, k))
        assert q[name]["lo"] <= q[name]["point"] + 3 * H.BIN and q[name]["hi"] >= q[name]["point"] - 3 * H.BIN
    # bootstrap quantile of a single resample equals the quantile of the concatenated resampled logs (brute force, one draw)
    d0 = B.draws[0]
    vv = np.concatenate([v[li == k] for k in d0])
    bf = np.quantile(vv, 0.5)
    from_hist = H.quantiles_from_hist((B.C[0] @ Hh)[None, :], (0.5,))[0, 0]
    assert abs(bf - from_hist) < 3 * H.BIN, (bf, from_hist)


def test_pair_stats():
    P = np.zeros((4, T, 2), np.float32)
    P[:, :, 0] = (np.arange(1, T + 1) * 1.0)[None]                    # all straight
    P[1, :, 1] = 0.2                                                  # B: 0.2 m lateral of A
    P[2, :, 1] = 3.0                                                  # C: 3 m away
    P[3, :, 1] = 0.25                                                 # D: 0.25 m from A, 0.05 from B
    dac = np.array([0.0, 1.0, 1.0, 0.0]); pdms = np.array([0.0, 0.8, 0.8, 0.0])
    D = H.pair_dist(P)
    s = H.pair_stats(D, dac, pdms, 0.3)
    # DAC0 = {A, D}; DAC1 = {B, C}; twins within 0.3: (A,B), (D,B)  -> 2 pairs, both DAC0 proposals twinned
    assert s["n_pairs_dac"] == 2 and s["n_dac0"] == 2 and s["n_dac0_twinned"] == 2, s
    s2 = H.pair_stats(D, dac, pdms, 0.1)
    assert s2["n_pairs_dac"] == 1 and s2["n_dac0_twinned"] == 1, s2     # only (D,B) at 0.05
    s3 = H.pair_stats(D, dac, pdms, 0.01)
    assert s3["n_pairs_dac"] == 0


def test_cyclic_wrong_expert():
    E = np.arange(6 * T * 3, dtype=np.float32).reshape(6, T, 3)
    li = np.array([0, 0, 0, 1, 1, 1])
    W = H.cyclic_wrong_expert(E, li)
    for i in range(6):
        assert not np.array_equal(W[i], E[i])                         # nobody keeps their own expert
        assert any(np.array_equal(W[i], E[j]) for j in np.where(li == li[i])[0])   # and it comes from the same log


def test_a1_block_toy():
    # 2 logs, 4 sets, 4 proposals.  Set 0: one proximate HN; set 1: proximate but safe; set 2: HN only far away; set 3: nothing proximate
    dist = np.array([[0.2, 3, 3, 3], [0.3, 3, 3, 3], [3, 3, 0.4, 3.0], [3, 3, 3, 3.0]])
    hn = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]], bool)
    dac0 = hn.copy(); z = np.zeros_like(hn)
    B = H.Boot(np.array([0, 0, 1, 1]), 2, B=100, seed=1)
    R = H.a1_block(B, dist, hn, dac0, z, z)
    e = R["d=0.5"]
    assert abs(e["share_sets_with_proximate"]["point"] - 0.75) < 1e-12
    assert abs(e["share_sets_with_proximate_HN"]["point"] - 0.25) < 1e-12
    assert abs(e["pooled_HN_rate_among_proximate"]["point"] - 1 / 3) < 1e-12
    assert abs(e["share_HN_given_proximate"]["point"] - 1 / 3) < 1e-12
    assert abs(e["share_of_all_HN_that_are_proximate"]["point"] - 1 / 3) < 1e-12


def test_join_key_includes_rank():
    a = {"log_name": "L", "token": "t", "step": 0, "rank": 0}
    b = {"log_name": "L", "token": "t", "step": 0, "rank": 1}
    assert L.key_of(a) != L.key_of(b)
    assert L.key_of({"key": ["L", "t", 0, 1]}) == L.key_of(b)


def test_identity_control():
    rng = np.random.default_rng(5)
    P = rng.normal(size=(3, 8, T, 2)).astype(np.float32)
    E = np.concatenate([P[:, 0], np.zeros((3, T, 1), np.float32)], -1)
    ade, _ = H.ade_fde(P, E)
    assert ade.min(1).max() < 1e-6 and (H.counts_at(ade, np.zeros_like(ade, bool), 0.5)[0] >= 1).all()


if __name__ == "__main__":
    names = sorted(n for n in dir() if n.startswith("test_"))
    bad = []
    for n in names:
        try:
            globals()[n]()
            print("PASS", n)
        except Exception:
            bad.append(n)
            print("FAIL", n)
            traceback.print_exc()
    print(f"{len(names) - len(bad)}/{len(names)} passed")
    sys.exit(1 if bad else 0)
