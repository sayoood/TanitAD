#!/usr/bin/env python3
"""Self-test of eval/aux_s1_readout.py.  Run (driverl-eval venv; no pytest):
    PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe eval/test_aux_s1_readout.py     -> ZZTEST_PASS <n>

Expected values are DERIVED BY HAND from the fixtures below (never from the module under test).  Each RED mutation edits the module's SOURCE (the snippet must occur
exactly once -- a vacuous mutation proves nothing) and the named check(s) must then FAIL.

FIXTURE P (P1 / P2).  Label: a corridor along x, fine rows with |y| < 6.5 are inside, SDF = 6.5 - |y| (0.1 m int8).  At the head's 2 m grid the TRUE mask is the
6 coarse rows 13..18 (y in [-6, 6)) in all 40 columns; the true coarse SDF is 6.5 - |y_centre| (clipped +-6): rows 11, 12, 13, 18, 19, 20 are the 6 cells per column within +-3 m of the boundary.
Columns: ix 0..3 are x < 0 (behind the rear axle), the band [0,20) is ix 4..13, [20,40) is ix 14..23, [40,72) is ix 24..39 (centre x = -8 + 2 (ix + 0.5)).
Predictions (+1 inside, -1 outside) = the truth corridor except
    A (log L1, token a)  none                     B (L1, b)  shifted +1 row for ix 4..13, +2 rows for ix 14..23
    C (log L2, token c)  nothing predicted for ix >= 24     D (L2, d)  everything predicted for ix 0..3         E (L3, e)  no label (garbage prediction)
Per column the truth has 6 cells.  Band [0,20) (10 columns): A 60/60, B 50/70, C 60/60, D 60/60 -> 230/250 = 0.92        [inter/union]
Band [20,40) (10 columns): A 60/60, B 40/80 (rows 15..18 shared, 13..20 united), C 60/60, D 60/60 -> 220/260 = 11/13
Band [40,72) (16 columns): A 96/96, B 96/96, C 0/96, D 96/96 -> 288/384 = 0.75      Band [0,40): 450/510 = 15/17
Behind (ix 0..3): A, B, C 24/24 each, D 24/128 -> with the bands: all cells 834/1094.
Split by "u400 DAC = 0" = {b, c}: [0,20) (50+60)/(70+60) = 110/130, [20,40) 100/140, [40,72) 96/192 = 0.5, [0,40) 210/270.  "none" = {a, d}: IoU 1 in every band.
P2: sdf prediction = truth + 0.5 on the near-boundary cells, truth + 100 elsewhere: near MAE 0.5 over 6 * 40 * 4 = 960 cells, overall MAE (6/32) 0.5 + (26/32) 100 = 81.34375.
"""
from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(HERE, "aux_s1_readout.py")
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PKG, "refe"))
import torch  # noqa: E402
import aux_drivable_head as AUX  # noqa: E402
import proxy_eval as PE  # noqa: E402
import rl_val_eval as RV  # noqa: E402


def load(src: str, name: str):
    m = types.ModuleType(name)
    m.__file__ = SRC_PATH
    sys.modules[name] = m
    exec(compile(src, SRC_PATH, "exec"), m.__dict__)
    return m


# ------------------------------------------------------------------------------------------------------------------ fixture P
def corridor_label(half=6.5) -> np.ndarray:
    y = -32.0 + 0.25 + 0.5 * np.arange(128)
    row = np.clip(np.round(10.0 * (half - np.abs(y))), -127, 127).astype(np.int8)
    return np.tile(row[None, :], (160, 1))


class FakeSource:
    def __init__(self, d, all_keys=None):
        self.d, self._all = d, all_keys

    def labels(self, keys):
        out = np.zeros((len(keys), 160, 128), np.int8)
        valid = np.zeros(len(keys), bool)
        for i, k in enumerate(keys):
            if tuple(k) in self.d:
                out[i], valid[i] = self.d[tuple(k)], True
        return torch.from_numpy(out), torch.from_numpy(valid)

    def all_keys(self):
        return list(self._all if self._all is not None else self.d)


def corridor_pred(rows_lo=13, rows_hi=18):
    p = -np.ones((40, 32), np.float32)
    p[:, rows_lo:rows_hi + 1] = 1.0
    return p


def shifted(cols, shift):
    p = corridor_pred()
    for ix in cols:
        p[ix] = -1.0
        p[ix, 13 + shift:19 + shift] = 1.0
    return p


KEYS = [("L1", "a"), ("L1", "b"), ("L2", "c"), ("L2", "d"), ("L3", "e")]


def fixture_p(logit_override=None):
    lab = corridor_label()
    src = FakeSource({k: lab for k in KEYS[:4]})
    t = AUX.build_targets(torch.from_numpy(lab[None]), None)
    truth = t.sdf_m.numpy()[0].astype(np.float32)
    near = np.abs(truth) <= 3.0
    A = corridor_pred()
    B = corridor_pred()
    for ix in range(4, 14):
        B[ix] = shifted([ix], 1)[ix]
    for ix in range(14, 24):
        B[ix] = shifted([ix], 2)[ix]
    C = corridor_pred()
    C[24:] = -1.0
    D = corridor_pred()
    D[:4] = 1.0
    E = np.random.default_rng(7).normal(size=(40, 32)).astype(np.float32)
    logit = np.stack([A, B, C, D, E]) if logit_override is None else logit_override
    sdf = np.where(near, truth + 0.5, truth + 100.0).astype(np.float32)
    sdf = np.stack([sdf] * 5)
    return {"keys": list(KEYS), "logit": logit.astype(np.float32), "sdf": sdf, "dropped_step_nonzero": 0, "duplicate_keys_identical": 0}, src


DZ = {"b", "c"}
I = {"all": {"0-20": 230 / 250, "20-40": 220 / 260, "40-72": 288 / 384, "0-40": 450 / 510, "all-cells": 834 / 1094},
     "dac0": {"0-20": 110 / 130, "20-40": 100 / 140, "40-72": 96 / 192, "0-40": 210 / 270},
     "dac_ok": {"0-20": 1.0, "20-40": 1.0, "40-72": 1.0, "0-40": 1.0}}


# ------------------------------------------------------------------------------------------------------------------ checks
def chk_band_geometry(m):
    for (lo, hi), cols in zip(m.BANDS, (range(4, 14), range(14, 24), range(24, 40))):
        b = m.band_cells(lo, hi)
        assert b.shape == (40, 32) and {int(i) for i in np.flatnonzero(b[:, 0])} == set(cols), (lo, hi)
        assert (b == b[:, :1]).all(), "a band must not depend on y"
    assert {int(i) for i in np.flatnonzero(m.band_cells(*m.K2_BAND)[:, 0])} == set(range(4, 24))
    assert abs(m.cell_x_centres()[4] - 1.0) < 1e-12 and abs(m.cell_x_centres()[39] - 71.0) < 1e-12


def chk_p1_exact(m):
    pred, src = fixture_p()
    st = m.frame_stats(pred, src)
    r = m.perception_reads(pred, st, DZ, n_boot=200)
    for sname, d in I.items():
        for nm, want in d.items():
            got = r["p1"][sname][nm]["est"]
            assert abs(got - want) < 1e-12, (sname, nm, got, want)
    assert r["p1"]["all"]["0-20"]["n_cells"] == 1280.0 and r["p1"]["all"]["all-cells"]["n_cells"] == 5120.0
    assert r["p1"]["dac0"]["n_frames"] == 2 and r["p1"]["dac_ok"]["n_frames"] == 2 and r["p1"]["all"]["n_frames"] == 4
    ci = r["p1"]["all"]["0-20"]["ci95"]
    assert ci[0] <= r["p1"]["all"]["0-20"]["est"] + 1e-12 and ci[1] >= r["p1"]["all"]["0-20"]["est"] - 1e-12


def chk_p1_split_direction(m):
    pred, src = fixture_p()
    r = m.perception_reads(pred, m.frame_stats(pred, src), DZ, n_boot=100)
    assert abs(r["p1"]["dac0"]["40-72"]["est"] - 0.5) < 1e-12 and abs(r["p1"]["dac_ok"]["40-72"]["est"] - 1.0) < 1e-12


def chk_unlabelled(m):
    pred, src = fixture_p()
    r = m.perception_reads(pred, m.frame_stats(pred, src), None, n_boot=100)
    assert r["n_frames"] == 5 and r["n_labelled"] == 4 and r["n_unlabelled"] == 1 and r["n_logs"] == 2 and list(r["p1"]) == ["all"]


def chk_matches_head_metrics(m):
    pred, src = fixture_p()
    st = m.frame_stats(pred, src)
    r = m.perception_reads(pred, st, None, n_boot=100)
    lab, valid = src.labels(pred["keys"])
    hm = AUX.head_metrics(torch.from_numpy(pred["logit"]), torch.from_numpy(pred["sdf"]), lab, valid)
    assert abs(r["p1"]["all"]["all-cells"]["est"] - hm["iou"]) < 1e-12, (r["p1"]["all"]["all-cells"]["est"], hm["iou"])
    assert abs(r["p2"]["overall_mae_m"]["est"] - hm["sdf_mae_m"]) < 1e-3 and r["p1"]["all"]["all-cells"]["n_cells"] == hm["n_cells"]


def chk_p2_exact(m):
    pred, src = fixture_p()
    r = m.perception_reads(pred, m.frame_stats(pred, src), None, n_boot=100)
    p2 = r["p2"]["near_boundary_mae_m"]
    assert p2["n_cells"] == 960.0, p2["n_cells"]
    assert abs(p2["est"] - 0.5) < 1e-5, p2["est"]
    assert abs(r["p2"]["overall_mae_m"]["est"] - 81.34375) < 1e-4, r["p2"]["overall_mae_m"]["est"]


def chk_k2(m):
    pred, src = fixture_p()
    st = m.frame_stats(pred, src)
    assert m.perception_reads(pred, st, None, n_boot=50)["k2"]["decision"] == "K2_CLEARED"
    dead, _ = fixture_p(np.full((5, 40, 32), -1.0, np.float32))
    r = m.perception_reads(dead, m.frame_stats(dead, src), None, n_boot=50)
    assert r["p1"]["all"]["0-40"]["est"] == 0.0 and r["k2"]["decision"] == "K2_FIRES", r["k2"]
    assert m.k2_decision({"est": float("nan"), "ci95": [0, 0]})["decision"] == "K2_UNDEFINED"


def chk_boot_scheme_is_pe_boot(m):
    rng = np.random.default_rng(3)
    logs = np.array([f"L{i % 7}" for i in range(60)])
    vals = rng.normal(size=60)
    mu, lo, hi = PE.boot({str(i): v for i, v in enumerate(vals)}, {str(i): logs[i] for i in range(60)}, n=2000, seed=20260927)
    ul, (num, den) = m.by_log(logs, vals, np.ones(60))
    W = m.boot_weights(len(ul), 2000, 20260927)
    r = m.ratio_read(num, den, W)
    assert abs(r["est"] - mu) < 1e-12 and abs(r["ci95"][0] - lo) < 1e-12 and abs(r["ci95"][1] - hi) < 1e-12, (r, mu, lo, hi)


def _two_arms():
    rng = np.random.default_rng(5)
    G = 9
    return rng.uniform(1, 5, G), rng.uniform(6, 9, G), rng.uniform(1, 5, G), rng.uniform(6, 9, G)


def chk_selfpair_zero(m):
    na, da, nb, db = _two_arms()
    W = m.boot_weights(len(na), 500, 20260927)
    d = m.diff_read(na, da, na, da, W)
    assert d["est"] == 0.0 and d["ci95"] == [0.0, 0.0], d
    S = np.random.default_rng(1).uniform(0, 3, (9, 9))
    Nm = S + 2.0
    d = m.auc_diff_read(S, Nm, S, Nm, W)
    assert d["est"] == 0.0 and d["ci95"] == [0.0, 0.0], d
    pred, src = fixture_p()
    st = m.frame_stats(pred, src)
    for k, v in m.head_fit_diff(pred, st, pred, st, 300).items():
        assert v["est"] == 0.0 and v["ci95"] == [0.0, 0.0], (k, v)


def chk_swap_flips(m):
    na, da, nb, db = _two_arms()
    W = m.boot_weights(len(na), 500, 20260927)
    f, r = m.diff_read(na, da, nb, db, W), m.diff_read(nb, db, na, da, W)
    assert r["est"] == -f["est"] and abs(r["ci95"][0] + f["ci95"][1]) < 1e-12 and abs(r["ci95"][1] + f["ci95"][0]) < 1e-12, (f, r)
    assert abs(f["est"] - (na.sum() / da.sum() - nb.sum() / db.sum())) < 1e-12
    S1, S2 = np.random.default_rng(1).uniform(0, 3, (9, 9)), np.random.default_rng(2).uniform(0, 3, (9, 9))
    N1 = N2 = np.full((9, 9), 4.0)
    f, r = m.auc_diff_read(S1, N1, S2, N2, W), m.auc_diff_read(S2, N2, S1, N1, W)
    assert r["est"] == -f["est"] and abs(r["ci95"][0] + f["ci95"][1]) < 1e-12 and abs(r["ci95"][1] + f["ci95"][0]) < 1e-12, (f, r)
    pred, src = fixture_p()
    pred2, _ = fixture_p()
    pred2["logit"] = pred2["logit"].copy()
    pred2["logit"][2][24:] = 1.0                                           # C now predicts the corridor + everything outside it for ix >= 24
    sta, stb = m.frame_stats(pred, src), m.frame_stats(pred2, src)
    f, r = m.head_fit_diff(pred, sta, pred2, stb, 300), m.head_fit_diff(pred2, stb, pred, sta, 300)
    assert any(abs(v["est"]) > 1e-3 for v in f.values()), "the fixture pair must differ"
    for k in f:
        assert r[k]["est"] == -f[k]["est"] and abs(r[k]["ci95"][0] + f[k]["ci95"][1]) < 1e-12, k


def chk_pairing_is_paired(m):
    """a strongly correlated pair: the paired interval of A - B must be far narrower than the interval of two independent bootstraps (the unpaired estimator)."""
    rng = np.random.default_rng(11)
    G = 40
    base = rng.uniform(0.2, 0.9, G)
    na, nb = base * 10, base * 10 + rng.normal(0, 0.02, G)
    den = np.full(G, 10.0)
    W = m.boot_weights(G, 2000, 20260927)
    paired = m.diff_read(na, den, nb, den, W)
    wid_p = paired["ci95"][1] - paired["ci95"][0]
    ra, rb = m.ratio_read(na, den, W), m.ratio_read(nb, den, m.boot_weights(G, 2000, 99))
    wid_u = math.hypot(ra["ci95"][1] - ra["ci95"][0], rb["ci95"][1] - rb["ci95"][0])          # the width an UNPAIRED difference of the two would have
    assert wid_p < 0.1 * wid_u, (wid_p, wid_u)


def chk_auc_hand(m):
    score = np.array([3.0, 2.0, 1.0, 5.0, 4.0, 2.0])
    passes = np.array([1, 1, 0, 1, 0, 0], bool)
    cl = np.array([0, 0, 0, 1, 1, 1])
    S, Nm = m.auc_cluster_stats(score, passes, cl, 2)
    assert np.array_equal(S, np.array([[2.0, 1.5], [1.0, 2.0]])) and np.array_equal(Nm, np.array([[2.0, 4.0], [1.0, 2.0]])), (S, Nm)
    assert abs(m.auc_value(S, Nm) - 6.5 / 9) < 1e-15
    assert m.auc_value(S, Nm, np.array([2.0, 0.0])) == 1.0 and m.auc_value(S, Nm, np.array([0.0, 2.0])) == 1.0
    assert abs(m.set_auc([3, 2, 1, 5, 4, 2], passes) - (1 + 0 + 1 + 1 + 0 + 0.5 + 1 + 1 + 1) / 9) < 1e-15
    assert m.set_auc([1.0, 2.0], [True, True]) is None
    # brute force on random weights: the pair-count decomposition IS the pooled AUC of the resampled rows
    rng = np.random.default_rng(2)
    sc, ps = rng.integers(0, 6, 80).astype(float), rng.random(80) < 0.4
    cl = rng.integers(0, 5, 80)
    S, Nm = m.auc_cluster_stats(sc, ps, cl, 5)
    for _ in range(5):
        w = rng.integers(0, 4, 5).astype(float)
        rows = np.concatenate([np.flatnonzero(cl == g).repeat(int(w[g])) for g in range(5)])
        a = m.set_auc(sc[rows], ps[rows])
        assert a is not None
        assert abs(m.auc_value(S, Nm, w) - a) < 1e-12, (w, m.auc_value(S, Nm, w), a)


def _corridor_sdf():
    yc = -31.0 + 2.0 * np.arange(32)
    return np.tile(np.clip(6.5 - np.abs(yc), -6.0, 6.0)[None, :], (40, 1)).astype(np.float32)


def _p8(final_y):
    p = np.zeros((8, 3))
    p[:, 0] = 5.0 * (np.arange(8) + 1)
    p[7, 1] = final_y
    return p


def chk_corner_min_sdf(m):
    sdf = _corridor_sdf()
    props = np.stack([_p8(0.0), _p8(6.0)])[None]                              # [1, 2, 8, 3]
    got, oob = m.corner_min_sdf(sdf[None], props)
    hw = 2.297 / 2.0
    assert abs(got[0, 0] - (6.5 - (1.0 + (hw - 1.0)))) < 1e-4 and abs(got[0, 0] - 5.3515) < 1e-4, got
    assert abs(got[0, 1] - (6.5 - (6.0 + hw))) < 1e-4 and abs(got[0, 1] + 0.6485) < 1e-4, got
    assert oob.shape == (1, 2) and (oob == 0).all()
    # the head's own feed uses the SAME lookup: 20 poses at 5 Hz for the straight proposal (x = 2 (n + 1), speed 10 m/s)
    tr = np.zeros((1, 1, 20, 3), np.float32)
    tr[0, 0, :, 0] = 2.0 * (np.arange(20) + 1)
    feats, _ = AUX.feed_scalars(torch.from_numpy(sdf[None]), torch.from_numpy(tr))
    assert abs(float(feats[0, 0, 0]) - got[0, 0]) < 1e-4, (float(feats[0, 0, 0]), got[0, 0])
    far = m.corner_min_sdf(sdf[None], (props + np.array([100.0, 0, 0]) * np.array([1, 0, 0]))[...])
    assert far[1].min() > 0.0                                                  # out-of-extent corners are counted, read as +6


def chk_p3_integration(m):
    sdf = np.stack([_corridor_sdf()] * 2)
    props = np.stack([np.stack([_p8(0.0), _p8(6.0)])] * 2)
    dac = np.array([[1.0, 0.0], [1.0, 0.0]])
    valid = np.ones((2, 2), bool)
    scorer = np.array([[0.0, 1.0], [1.0, 0.0]])
    r = m.p3_read(sdf, props, dac, valid, scorer, ["L1", "L2"], n_boot=200)
    p = r["pooled"]
    assert p["head"]["est"] == 1.0 and p["scorer"]["est"] == 0.5 and abs(p["head_minus_scorer"]["est"] - 0.5) < 1e-15, p
    assert p["n_proposals"] == 4 and p["n_pass"] == 2 and p["n_fail"] == 2 and p["n_logs"] == 2
    # A2: with no initial-head prediction the thresholds are met but the floor is unread => FLOOR_UNREAD, never a pass (the floor's own cases are chk_p3_floor_*)
    assert r["decision"]["decision"] == "FLOOR_UNREAD" and r["decision"]["decision_without_floor"] == "H2_EVALUATED" and abs(r["decision"]["evaluate_threshold"] - 0.9) < 1e-15
    valid2 = valid.copy()
    valid2[0, 1] = False                                                       # drop token 1's failing proposal: one fail left (token 2's veer)
    r2 = m.p3_read(sdf, props, dac, valid2, scorer, ["L1", "L2"], n_boot=100)
    assert r2["pooled"]["n_proposals"] == 3 and r2["pooled"]["n_fail"] == 1
    bad = dac.copy()
    bad[0, 0] = 0.5
    try:
        m.p3_read(sdf, props, bad, valid, scorer, ["L1", "L2"], n_boot=10)
    except ValueError:
        pass
    else:
        raise AssertionError("a DAC label of 0.5 must be refused")


def chk_p3_decision(m):
    # the THRESHOLDS (0.90 / scorer + 0.05 / head < scorer), with a clearly positive floor so only the thresholds decide; S1 R3: the old UNCOMMITTED_BAND is H2_PROVISIONAL
    cases = [((0.95, 0.80), "H2_EVALUATED", 0.90), ((0.95, 0.93), "H2_PROVISIONAL", 0.98), ((0.70, 0.78), "H2_KILLED", 0.90), ((0.86, 0.80), "H2_PROVISIONAL", 0.90)]
    for (h, s), want, thr in cases:
        d = m.p3_decision(h, s, 0.10)
        assert d["decision"] == want and d["decision_without_floor"] == want and abs(d["evaluate_threshold"] - thr) < 1e-12, (h, s, d)
        assert d["decision"] != "UNCOMMITTED_BAND"


def chk_p3_floor_rule(m):
    """A2.1 on the pure rule: the floor is an ADDED condition on the two promoting outcomes (lower bound strictly > 0), a missing floor is never a pass, a kill is unchanged."""
    for (h, s), base in (((0.95, 0.80), "H2_EVALUATED"), ((0.86, 0.80), "H2_PROVISIONAL")):
        assert m.p3_decision(h, s, 1e-9)["decision"] == base                        # strictly positive lower bound: the base outcome stands
        assert m.p3_decision(h, s, 0.0)["decision"] == "P3_AT_FLOOR"                # exactly 0 is not > 0
        assert m.p3_decision(h, s, -0.02)["decision"] == "P3_AT_FLOOR"
        assert m.p3_decision(h, s, float("nan"))["decision"] == "P3_AT_FLOOR"       # an unreadable bound is not a cleared floor
        d = m.p3_decision(h, s, None)
        assert d["decision"] == "FLOOR_UNREAD" and d["decision_without_floor"] == base
    for fl in (0.3, 0.0, -0.3, None):                                               # head < scorer is a kill whatever the floor says
        d = m.p3_decision(0.70, 0.78, fl)
        assert d["decision"] == "H2_KILLED" and abs(d["evaluate_threshold"] - 0.90) < 1e-12
    assert abs(m.p3_decision(0.95, 0.93, 0.1)["evaluate_threshold"] - 0.98) < 1e-12   # scorer + 0.05 beats 0.90 when the scorer is above 0.85: unchanged
    assert m.p3_decision(0.95, 0.93, 0.1)["decision"] == "H2_PROVISIONAL"


# ---- the floor on synthetic C3-like data.  Every token has the same 8 proposals (final lateral offsets YS_F; the first 4 pass DAC, the last 4 fail) so every pooled AUC below is a
# fraction you can count by hand.  Fields: C = the corridor SDF (min along the footprint strictly decreases with the offset => the ranking is perfect), Z = all zeros (every
# proposal ties => AUC exactly 1/2).
YS_F = [0.5, 1.0, 1.5, 2.0, 4.0, 5.0, 6.0, 7.0]
DAC_PERFECT_F = np.array([1, 1, 1, 1, 0, 0, 0, 0], float)
DAC_BAND_F = np.array([1, 1, 0, 1, 1, 0, 0, 0], float)       # pass {0,1,3,4} fail {2,5,6,7}: wins 4 + 4 + 3 + 3 of 16 pairs = 14/16 = 0.875 for the corridor ranking


def _floor_case(m, trained, init, dac=DAC_PERFECT_F, scorer=None, n_logs=12, tpl=2, n_boot=400):
    """trained / init: a field [40, 32] for every log (or a list of n_logs fields); init=None => the floor is not given."""
    N = n_logs * tpl
    one = lambda f: [f] * n_logs if isinstance(f, np.ndarray) else list(f)         # noqa: E731
    th = np.stack([one(trained)[i // tpl] for i in range(N)])
    ih = None if init is None else np.stack([one(init)[i // tpl] for i in range(N)])
    props = np.stack([np.stack([_p8(y) for y in YS_F])] * N)
    sc = np.zeros((N, len(YS_F))) if scorer is None else scorer(N)
    r = m.p3_read(th, props, np.tile(dac[None], (N, 1)), np.ones((N, len(YS_F)), bool), sc, [f"L{i // tpl:02d}" for i in range(N)], n_boot=n_boot, init_sdf=ih)
    return r, (th, ih, props, sc)


def chk_p3_floor_decisions(m):
    C, Z = _corridor_sdf(), np.zeros((40, 32), np.float32)
    # (a) trained beats its init clearly: AUC 1.0 vs 1/2, scorer ties (1/2) => thr 0.90 cleared, floor lower bound 0.5 => H2_EVALUATED
    r, _ = _floor_case(m, C, Z)
    p, f, d = r["pooled"], r["floor"], r["decision"]
    assert p["head"]["est"] == 1.0 and p["scorer"]["est"] == 0.5 and f["status"] == "READ" and f["init"]["est"] == 0.5, (p, f)
    assert f["trained_minus_init"]["est"] == 0.5 and f["trained_minus_init"]["ci95"] == [0.5, 0.5], f["trained_minus_init"]
    assert d["decision"] == "H2_EVALUATED" and d["floor_lower_bound"] == 0.5 and abs(d["evaluate_threshold"] - 0.9) < 1e-15, d
    # (b) trained == init (identical predictions): its AUC clears 0.90 and the scorer, yet the paired difference is exactly 0 => P3_AT_FLOOR
    r, _ = _floor_case(m, C, C)
    d = r["decision"]
    assert r["pooled"]["head"]["est"] == 1.0 and r["floor"]["init"]["est"] == 1.0
    assert r["floor"]["trained_minus_init"]["est"] == 0.0 and r["floor"]["trained_minus_init"]["ci95"] == [0.0, 0.0]
    assert d["decision"] == "P3_AT_FLOOR" and d["decision_without_floor"] == "H2_EVALUATED", d
    # (c) the band: the corridor ranking on DAC_BAND_F reads 14/16 = 0.875, between the scorer (1/2) and 0.90 => H2_PROVISIONAL only if the floor holds
    r, _ = _floor_case(m, C, Z, dac=DAC_BAND_F)
    assert r["pooled"]["head"]["est"] == 0.875 and r["decision"]["decision"] == "H2_PROVISIONAL" and r["floor"]["trained_minus_init"]["est"] == 0.375, r["decision"]
    r, _ = _floor_case(m, C, C, dac=DAC_BAND_F)
    assert r["decision"]["decision"] == "P3_AT_FLOOR" and r["decision"]["decision_without_floor"] == "H2_PROVISIONAL", r["decision"]
    # (d) the floor is missing: FLOOR_UNREAD, never a pass, in both outcomes that would promote
    for dac, base in ((DAC_PERFECT_F, "H2_EVALUATED"), (DAC_BAND_F, "H2_PROVISIONAL")):
        r, _ = _floor_case(m, C, None, dac=dac)
        assert r["decision"]["decision"] == "FLOOR_UNREAD" and r["decision"]["decision_without_floor"] == base and r["floor"]["status"] == "FLOOR_UNREAD", r["decision"]
    # a kill is a kill: the scorer ranks perfectly (AUC 1.0), the head ties (1/2) => H2_KILLED with an init, with an identical init, and with none
    perfect_scorer = lambda N: np.tile(-np.asarray(YS_F)[None], (N, 1))              # noqa: E731
    for init in (C, Z, None):
        r, _ = _floor_case(m, Z, init, scorer=perfect_scorer)
        assert r["pooled"]["scorer"]["est"] == 1.0 and r["decision"]["decision"] == "H2_KILLED", r["decision"]
    # an init of another shape (other tokens) is refused
    try:
        m.p3_read(np.stack([C] * 4), np.stack([np.stack([_p8(y) for y in YS_F])] * 4), np.tile(DAC_PERFECT_F[None], (4, 1)), np.ones((4, 8), bool), np.zeros((4, 8)),
                  ["L0", "L0", "L1", "L1"], n_boot=10, init_sdf=np.stack([C] * 3))
    except ValueError:
        pass
    else:
        raise AssertionError("an init prediction of another token count must be refused")


def chk_p3_floor_same_draws(m):
    """The floor difference rides the SAME log resamples as head-minus-scorer.  Make the init head's AUC equal the scorer's on EVERY resample (the scorer's logit IS the init
    head's score, so the pair-count matrices are identical): the two paired differences are then the same numbers, interval included.  Different weights break the equality.
    The fixture is deliberately irregular (14 logs of 1-5 tokens, continuous offsets, labels noisy in the offset) so the resampled differences are CONTINUOUS: on a regular
    fixture they take few distinct values and two different draws can land on the same percentile atoms (MEASURED: the unpaired mutation then stayed GREEN)."""
    rng = np.random.default_rng(11)
    C, Z = _corridor_sdf(), np.zeros((40, 32), np.float32)
    n_logs = 14
    sizes = rng.integers(1, 6, n_logs)
    tok_logs = [f"L{g:02d}" for g in range(n_logs) for _ in range(int(sizes[g]))]
    N = len(tok_logs)
    ys = rng.uniform(0.3, 7.5, (N, 8))
    props = np.stack([np.stack([_p8(y) for y in row]) for row in ys])
    dac = (ys + rng.normal(0.0, 1.5, ys.shape) < 3.0).astype(float)
    log_i = np.asarray([int(l[1:]) for l in tok_logs])
    trained = np.stack([Z if g == 0 else C for g in log_i])                    # at chance in log 0 only,
    init = np.stack([Z if g in (1, 2, 3, 4, 5) else C for g in log_i])         # at chance in logs 1-5: trained beats init on average, not on every draw
    init_score = m.corner_min_sdf(init, props)[0]
    r = m.p3_read(trained, props, dac, np.ones(dac.shape, bool), init_score, tok_logs, n_boot=3000, init_sdf=init)
    a, b = r["floor"]["trained_minus_init"], r["pooled"]["head_minus_scorer"]
    assert a["ci95"][1] - a["ci95"][0] > 0.05 and a["est"] > 0.0, a             # a non-degenerate interval: the check is not vacuous
    assert a["est"] == b["est"] and a["ci95"] == b["ci95"], (a, b)
    assert r["floor"]["init"]["est"] == r["pooled"]["scorer"]["est"] and r["floor"]["init"]["ci95"] == r["pooled"]["scorer"]["ci95"]
    assert len({round(x, 12) for x in (a["ci95"][0], a["ci95"][1], r["floor"]["init"]["ci95"][0])}) == 3


def chk_p3_floor_ci_bound(m):
    """The condition is on the 95 % LOWER BOUND, not on the point estimate: trained beats init on average (+0.083) but one draw of logs may not."""
    C, Z = _corridor_sdf(), np.zeros((40, 32), np.float32)
    r, _ = _floor_case(m, [Z if i == 0 else C for i in range(12)], [Z if i in (1, 2, 3) else C for i in range(12)], n_boot=4000)
    f = r["floor"]["trained_minus_init"]
    assert f["est"] > 0.05 and f["ci95"][0] < 0.0 < f["ci95"][1], f               # the premise (hand-built: trained is at chance in log 0, the init in logs 1-3)
    assert r["decision"]["decision_without_floor"] == "H2_EVALUATED" and r["pooled"]["head"]["est"] > 0.9
    assert r["decision"]["decision"] == "P3_AT_FLOOR" and r["decision"]["floor_lower_bound"] == f["ci95"][0] < 0.0, r["decision"]


def chk_floor_key_guard(m):
    """A2 guard: the init prediction must cover EXACTLY the trained arm's keys.  Fewer, more, or other keys are refused; a permutation is aligned to the trained order."""
    def pred(keys, off=0.0):
        return {"keys": list(keys), "logit": np.stack([np.full((40, 32), i + off, np.float32) for i in range(len(keys))]),
                "sdf": np.stack([np.full((40, 32), 10.0 * i + off, np.float32) for i in range(len(keys))]), "dropped_step_nonzero": 0, "duplicate_keys_identical": 0}
    keys = [("L1", "a"), ("L1", "b"), ("L2", "c")]
    trained = pred(keys)
    for bad in (keys[:2], keys + [("L9", "z")], [keys[0], keys[1], ("L2", "d")], keys[:2] + [("L2", "d"), ("L9", "z")]):
        try:
            m.align_init_pred(trained, pred(bad))
        except m.FloorRefused:
            pass
        else:
            raise AssertionError(f"an init prediction over {bad} must be refused")
    perm = [keys[2], keys[0], keys[1]]                                        # same SET, another order: row i of the result is the init row of trained key i
    al = m.align_init_pred(trained, pred(perm, off=0.5))
    assert al["keys"] == keys
    assert al["logit"][0, 0, 0] == 1.5 and al["logit"][1, 0, 0] == 2.5 and al["logit"][2, 0, 0] == 0.5, al["logit"][:, 0, 0]
    assert al["sdf"][0, 0, 0] == 10.5 and al["sdf"][2, 0, 0] == 0.5


def chk_cli_pred_init_refusals(m):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.npz")
        np.savez(p, log=np.array(["A"]), token=np.array(["t"]), logit=np.zeros((1, 40, 32), np.float32), sdf_m=np.zeros((1, 40, 32), np.float32))
        for pred_args, init_args, want in ((["H2-s0=" + p], ["N-s0=" + p], "no floor"), (["H2-s0=" + p], ["H1-s0=" + p], "no --pred arm")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = m.main(["perception", "--pred", *pred_args, "--pred-init", *init_args, "--labels-set", "navtest", "--out", os.path.join(d, "o.json")])
            assert rc == 4 and "--pred-init" in buf.getvalue() and want in buf.getvalue(), (rc, buf.getvalue())


def chk_pair_with_n(m):
    got = m.pair_with_n(["N-s0", "H2-s0", "H1-s0", "H2-s1", "N-s2", "H1-s2"])
    assert got == {"H1-s0": "N-s0", "H1-s2": "N-s2", "H2-s0": "N-s0", "H2-s1": None}, got
    assert "N-s0" not in got
    for bad in ("X-s0", "H2-0", "H2-s", "N"):
        try:
            m.parse_arm(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} must not parse")
    assert m.parse_arm("H2-s12") == ("H2", 12)


def chk_leak_key_and_log(m):
    train = [("T1", "x"), ("T1", "y"), ("T2", "z")]
    m.assert_disjoint([("L1", "a"), ("L2", "b")], train)
    for ev in ([("T2", "z")], [("T1", "other_token")]):
        try:
            m.assert_disjoint(ev, train)
        except m.LeakRefused:
            continue
        raise AssertionError(f"leak not refused: {ev}")


def chk_leak_heldout(m):
    train = [("T1", "x"), ("T2", "z")]
    m.assert_disjoint([("H1", "p")], train, heldout_logs=["H1", "H2"])
    try:
        m.assert_disjoint([("H3", "p")], train, heldout_logs=["H1", "H2"])
    except m.LeakRefused:
        return
    raise AssertionError("an evaluated log outside the held-out set must be refused")


def chk_train_keys_for(m):
    src = FakeSource({}, all_keys=[("T1", "x"), ("H1", "p"), ("H1", "q"), ("T2", "z")])
    assert sorted(m.train_keys_for("navtest", src)) == sorted(src.all_keys())
    got = m.train_keys_for("train", src, ["H1"])
    assert sorted(got) == [("T1", "x"), ("T2", "z")], got
    m.assert_disjoint([("H1", "p")], got, heldout_logs=["H1"])
    try:
        m.train_keys_for("train", src, None)
    except ValueError:
        return
    raise AssertionError("a held-out read without --heldout-logs must be refused")


def chk_load_head_pred(m):
    with tempfile.TemporaryDirectory() as d:
        n = 4
        lg = np.random.default_rng(0).normal(size=(n, 40, 32)).astype(np.float32)
        sd = lg * 2
        base = dict(log=np.array(["A", "A", "B", "A"]), token=np.array(["t0", "t1", "t2", "t0"]), logit=lg, sdf_m=sd)
        lg[3], sd[3] = lg[0], sd[0]
        p = os.path.join(d, "a.npz")
        np.savez(p, step=np.array([0, 0, 0, 0]), **base)
        r = m.load_head_pred(p)
        assert r["keys"] == [("A", "t0"), ("A", "t1"), ("B", "t2")] and r["duplicate_keys_identical"] == 1 and r["dropped_step_nonzero"] == 0
        np.savez(p, step=np.array([0, 0, 0, 2]), **base)
        r = m.load_head_pred(p)
        assert r["keys"] == [("A", "t0"), ("A", "t1"), ("B", "t2")] and r["dropped_step_nonzero"] == 1 and r["duplicate_keys_identical"] == 0, r
        lg2 = lg.copy()
        lg2[3] += 3.0                                    # beyond A8's one-logit bound: a different frame under one key
        np.savez(p, step=np.array([0, 0, 0, 0]), **dict(base, logit=lg2))
        try:
            m.load_head_pred(p)
        except ValueError:
            pass
        else:
            raise AssertionError("two different outputs for one key must be refused")
        # A8: a duplicate within bf16 noise (0.03 logit, 0.09 m) is accepted, counted, and the FIRST copy is kept
        lg3, sd3 = lg.copy(), sd.copy()
        lg3[3], sd3[3] = lg[0] + np.float32(0.03), sd[0] + np.float32(0.09)
        np.savez(p, step=np.array([0, 0, 0, 0]), **dict(base, logit=lg3, sdf_m=sd3))
        r = m.load_head_pred(p)
        assert r["keys"] == [("A", "t0"), ("A", "t1"), ("B", "t2")] and r["duplicate_keys_within_tol"] == 1 and r["duplicate_keys_identical"] == 0, r
        assert np.array_equal(r["logit"][0], lg3[0]) and np.array_equal(r["sdf"][0], sd3[0]), "the FIRST copy must be kept"
        assert abs(r["duplicate_max_abs_diff"][0] - 0.03) < 1e-3 and abs(r["duplicate_max_abs_diff"][1] - 0.09) < 1e-3, r["duplicate_max_abs_diff"]
        # ... and one beyond a 2 m cell on the SDF alone is still refused
        sd4 = sd.copy()
        sd4[3] = sd[0] + np.float32(2.5)
        np.savez(p, step=np.array([0, 0, 0, 0]), **dict(base, logit=lg3, sdf_m=sd4))
        try:
            m.load_head_pred(p)
        except ValueError:
            pass
        else:
            raise AssertionError("a duplicate 2.5 m apart on the SDF is a different frame and must be refused")
        np.savez(p, log=base["log"], token=base["token"], logit=lg[:, :39], sdf_m=sd[:, :39])
        try:
            m.load_head_pred(p)
        except ValueError:
            return
        raise AssertionError("a wrong grid shape must be refused")


def _selection_fixture():
    TG = np.array([[1, 1, 1.0, 1, 1, 1], [1, 1, 0.5, 1, 1, 1], [0, 1, 0.0, 1, 1, 1], [1, 1, 0.2, 0, 1, 1], [1, 1, 0.1, 1, 1, 1]], float)
    sets = [(f"LOG{i // 2}", f"tok{i:02d}", 0, 0) for i in range(12)]
    pdm = {k: {"ckpt": 100, "tg": TG} for k in sets}

    def sc(picks):
        out = {}
        for k, pk in zip(sets, picks):
            s = np.full((5, 7), -8.0)
            s[pk, :6] = 8.0
            out[k] = {"ckpt": 100, "scores": s, "space": "logit"}
        return out

    n0 = [1] * 12
    return pdm, sc, n0


def chk_selection_exact(m):
    pdm, sc, n0 = _selection_fixture()
    scores = {"N-s0": sc(n0), "H2-s0": sc([0] * 6 + [1] * 6), "H1-s0": sc(n0), "H1-s1": sc(n0)}
    r = m.selection_reads(scores, pdm, "navsim_v1", n_boot=1000)
    assert r["unpaired_head_arms"] == ["H1-s1"] and sorted(r["pairs"]) == ["H1-s0-minus-N-s0", "H2-s0-minus-N-s0"], r["unpaired_head_arms"]
    h2 = r["pairs"]["H2-s0-minus-N-s0"]
    want = 100.0 * (6 * (1.0 - 9.5 / 12.0)) / 12.0                              # 6 of 12 sets pick P0 (PDMS 1) instead of P1 (9.5/12)
    q = h2["paired_vs_n"]["quantities"]
    assert abs(q["PDMS"]["mean"] - want) < 1e-9 and abs(want - 10.416666666666666) < 1e-9, (q["PDMS"]["mean"], want)
    assert abs(q["EP"]["mean"] - 100.0 * 6 * 0.5 / 12.0) < 1e-9 and q["DAC"]["mean"] == 0.0 and q["NC"]["mean"] == 0.0
    assert q["PDMS"]["better"] == 6 and q["PDMS"]["worse"] == 0 and q["PDMS"]["tied"] == 6 and h2["n_sets"] == 12 and h2["n_logs"] == 6
    assert h2["e2_dac_zero_share_delta"]["mean_pp"] == -0.0 and h2["e3_zero_picks"]["mean"] == 0.0
    assert h2["selfcheck"]["selfpair"]["ok"] and h2["selfcheck"]["swap"]["ok"]
    h1 = r["pairs"]["H1-s0-minus-N-s0"]["paired_vs_n"]["primary"]
    assert h1["mean"] == 0.0 and h1["ci95"] == [0.0, 0.0]                         # the scorer of a feed-off arm identical to N reads exactly 0
    # swap: the reverse pair reads the negation
    rev = m.selection_pair(scores["N-s0"], scores["H2-s0"], pdm, "navsim_v1", 1000)
    assert abs(rev["paired_vs_n"]["primary"]["mean"] + want) < 1e-9
    # E2 sign: an arm that picks P2 (DAC = 1 but NC = 0)... use a DAC-failing proposal to fix the sign convention exactly
    TG2 = pdm[next(iter(pdm))]["tg"].copy()
    TG2[2, 1] = 0.0                                                             # P2 now has DAC = 0
    pdm2 = {k: {"ckpt": 100, "tg": TG2} for k in pdm}
    s2 = {"N-s0": sc(n0), "H2-s0": sc([2] * 3 + [1] * 9)}
    e2 = m.selection_pair(s2["H2-s0"], s2["N-s0"], pdm2, "navsim_v1", 500)
    assert abs(e2["paired_vs_n"]["quantities"]["DAC"]["mean"] + 100.0 * 3 / 12) < 1e-9
    assert abs(e2["e2_dac_zero_share_delta"]["mean_pp"] - 25.0) < 1e-9, e2["e2_dac_zero_share_delta"]
    assert abs(e2["e3_zero_picks"]["mean"] - 25.0) < 1e-9


def chk_seed_floor(m):
    pdm, sc, n0 = _selection_fixture()
    n1 = n0[:11] + [0]                                                          # N-s1 picks P0 on set 11 (+2.5/12 PDMS there)
    scores = {"N-s0": sc(n0), "N-s1": sc(n1), "N-s2": sc(n0), "H2-s0": sc([0] * 6 + [1] * 6)}
    r = m.selection_reads(scores, pdm, "navsim_v1", n_boot=1000)
    f = r["seed_floor"]
    want = 100.0 * (1.0 - 9.5 / 12.0) / 12.0
    assert f["measured"] and f["n_n_seeds"] == 3 and abs(f["floor"]["PDMS"] - want) < 1e-9, f
    assert abs(want - 1.7361111111111112) < 1e-9
    assert r["pairs"]["H2-s0-minus-N-s0"]["classification"]["PDMS"] == "separated and above the seed floor", r["pairs"]["H2-s0-minus-N-s0"]["classification"]
    one = m.selection_reads({"N-s0": sc(n0), "H2-s0": sc([0] * 6 + [1] * 6)}, pdm, "navsim_v1", n_boot=1000)
    assert not one["seed_floor"]["measured"]
    assert one["pairs"]["H2-s0-minus-N-s0"]["classification"]["PDMS"] == "separated, single seed, not yet a lever claim"
    two = m.selection_reads({"N-s0": sc(n0), "N-s1": sc(n1), "H2-s0": sc([0] * 6 + [1] * 6)}, pdm, "navsim_v1", n_boot=500)
    assert not two["seed_floor"]["measured"], "two N seeds are not a measured floor"


def chk_cli_refuses_n_arm_and_unpinned_store(m):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.npz")
        np.savez(p, log=np.array(["A"]), token=np.array(["t"]), logit=np.zeros((1, 40, 32), np.float32), sdf_m=np.zeros((1, 40, 32), np.float32))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = m.main(["perception", "--pred", f"N-s0={p}", "--labels-set", "navtest", "--out", os.path.join(d, "o.json")])
        assert rc == 4 and "no-head arm" in buf.getvalue(), (rc, buf.getvalue())
    assert m.LABEL_PINS["navtest"][1] == 12146 and m.LABEL_PINS["train"][1] == 103139


CHECKS = [chk_band_geometry, chk_p1_exact, chk_p1_split_direction, chk_unlabelled, chk_matches_head_metrics, chk_p2_exact, chk_k2, chk_boot_scheme_is_pe_boot,
          chk_selfpair_zero, chk_swap_flips, chk_pairing_is_paired, chk_auc_hand, chk_corner_min_sdf, chk_p3_integration, chk_p3_decision, chk_pair_with_n,
          chk_leak_key_and_log, chk_leak_heldout, chk_train_keys_for, chk_load_head_pred, chk_selection_exact, chk_seed_floor, chk_cli_refuses_n_arm_and_unpinned_store,
          chk_p3_floor_rule, chk_p3_floor_decisions, chk_p3_floor_same_draws, chk_p3_floor_ci_bound, chk_floor_key_guard, chk_cli_pred_init_refusals]
BYNAME = {f.__name__: f for f in CHECKS}


def run_checks(mod, names=None):
    failing = []
    for f in (CHECKS if names is None else [BYNAME[n] for n in names]):
        try:
            f(mod)
        except Exception as e:                                                   # noqa: BLE001
            failing.append(f"{f.__name__}[{type(e).__name__}]")
    return failing


# ------------------------------------------------------------------------------------------------------------------ real-store checks (this dev box)
def real_store_checks(mod) -> list:
    """Checks against the BANKED stores. Skipped (with the reason) only when the store directory is absent; a store that is present and wrong FAILS."""
    out = []
    nav, trn = mod.LABEL_DIRS["navtest"], mod.LABEL_DIRS["train"]
    if not (os.path.isdir(nav) and os.path.isdir(trn)):
        print(f"SKIP real-store checks: NO_STORE ({nav} / {trn})")
        return out
    src = mod.open_labels("navtest")
    keys = src.all_keys()
    assert len(keys) == 12146 and src.manifest["store_sha256"] == mod.LABEL_PINS["navtest"][0]
    try:
        mod.open_labels("navtest", trn)
    except AUX.StoreRefused:
        pass
    else:
        raise AssertionError("the train store must be refused under the navtest pin")
    # an oracle head on 96 real navtest frames reads IoU 1.0 / MAE 0.0 exactly, a head rolled one row in y does not; H2 - H1 is paired; a TRAIN frame is refused
    keys96 = keys[:96]
    lab, valid = src.labels(keys96)
    t = AUX.build_targets(lab, valid)
    logit = ((t.mask_frac.numpy() - 0.5) * 10 + 1e-3).astype(np.float32)
    sdf = t.sdf_m.numpy().astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        np.savez(os.path.join(d, "h2.npz"), log=np.array([k[0] for k in keys96]), token=np.array([k[1] for k in keys96]), logit=logit, sdf_m=sdf)
        np.savez(os.path.join(d, "h1.npz"), log=np.array([k[0] for k in keys96]), token=np.array([k[1] for k in keys96]), logit=np.roll(logit, 1, 2), sdf_m=np.roll(sdf, 1, 2))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mod.main(["perception", "--pred", f"H2-s0={d}/h2.npz", f"H1-s0={d}/h1.npz", "--labels-set", "navtest", "--n-boot", "200", "--out", f"{d}/o.json"])
        assert rc == 0, buf.getvalue()
        import json
        r = json.load(open(f"{d}/o.json", encoding="utf-8"))
        a = r["arms"]["H2-s0"]
        assert a["p1"]["all"]["0-40"]["est"] == 1.0 and a["p2"]["near_boundary_mae_m"]["est"] == 0.0 and a["k2"]["decision"] == "K2_CLEARED"
        assert r["arms"]["H1-s0"]["p1"]["all"]["0-40"]["est"] < 0.9 and r["arms"]["H1-s0"]["p3"]["status"] == "NOT_COMPUTED"
        pr = r["head_fit_pairs"]["H2-s0-minus-H1-s0"]
        assert pr["iou_0-40"]["est"] > 0.1 and pr["near_boundary_mae_m"]["est"] < 0.0
        assert r["labels"]["store_sha256"] == mod.LABEL_PINS["navtest"][0]
        tk = AUX.AuxLabelSource(trn, pinned_sha256=mod.LABEL_PINS["train"][0], pinned_keys=103139).all_keys()[:3]
        np.savez(f"{d}/leak.npz", log=np.array([k[0] for k in tk]), token=np.array([k[1] for k in tk]), logit=np.zeros((3, 40, 32), np.float32), sdf_m=np.zeros((3, 40, 32), np.float32))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mod.main(["perception", "--pred", f"H2-s0={d}/leak.npz", "--labels-set", "navtest", "--out", f"{d}/o2.json"])
        assert rc == 3 and "ZZAUXS1_REFUSED" in buf.getvalue(), (rc, buf.getvalue())
        # A2 on the CLI: H2's initial head = the oracle rolled one row in y (a weak head); H1 has no init => FLOOR_UNREAD; trained - init is positive on P1; an init over 95 of the 96 keys is refused
        np.savez(os.path.join(d, "i2.npz"), log=np.array([k[0] for k in keys96]), token=np.array([k[1] for k in keys96]), logit=np.roll(logit, 1, 2), sdf_m=np.roll(sdf, 1, 2))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mod.main(["perception", "--pred", f"H2-s0={d}/h2.npz", f"H1-s0={d}/h1.npz", "--pred-init", f"H2-s0={d}/i2.npz", "--labels-set", "navtest", "--n-boot", "200",
                           "--out", f"{d}/o3.json"])
        assert rc == 0 and "floor_read ['H2-s0']" in buf.getvalue() and "floor_unread ['H1-s0']" in buf.getvalue(), (rc, buf.getvalue())
        r3 = json.load(open(f"{d}/o3.json", encoding="utf-8"))
        fl = r3["arms"]["H2-s0"]["floor"]
        assert fl["status"] == "READ" and fl["p1"]["all"]["0-40"]["est"] < 0.9 and fl["trained_minus_init"]["iou_0-40"]["est"] > 0.1 and fl["trained_minus_init"]["near_boundary_mae_m"]["est"] < 0.0
        assert r3["arms"]["H1-s0"]["floor"]["status"] == "FLOOR_UNREAD" and r3["arms"]["H2-s0"]["p1"]["all"]["0-40"]["est"] == 1.0
        np.savez(os.path.join(d, "i3.npz"), log=np.array([k[0] for k in keys96[:95]]), token=np.array([k[1] for k in keys96[:95]]), logit=logit[:95], sdf_m=sdf[:95])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mod.main(["perception", "--pred", f"H2-s0={d}/h2.npz", "--pred-init", f"H2-s0={d}/i3.npz", "--labels-set", "navtest", "--n-boot", "200", "--out", f"{d}/o4.json"])
        assert rc == 3 and "ZZAUXS1_REFUSED H2-s0" in buf.getvalue() and "missing" in buf.getvalue(), (rc, buf.getvalue())
    return ["real_store_navtest_oracle_and_leak_cli", "real_store_a2_floor_cli"]


# ------------------------------------------------------------------------------------------------------------------ RED mutations
M = [
    # (name, class, snippet that must occur exactly once, replacement, the checks that must go RED)
    ("sign_pdms_delta_reversed", "sign", "fwd = RV.read_pair(ph, pn, n_boot, seed)", "fwd = RV.read_pair(pn, ph, n_boot, seed)", ["chk_selection_exact"]),
    ("sign_e2_not_negated", "sign", 'e2 = {"mean_pp": 0.0 - d["mean"], "ci95_pp": [-d["ci95"][1], -d["ci95"][0]]', 'e2 = {"mean_pp": d["mean"], "ci95_pp": [d["ci95"][0], d["ci95"][1]]',
     ["chk_selection_exact"]),
    ("sign_p3_score_inverted", "sign", "return s.min(-1).values.numpy().astype(np.float64), (1.0", "return (-s).min(-1).values.numpy().astype(np.float64), (1.0",
     ["chk_corner_min_sdf"]),
    ("sign_p3_pass_label_inverted", "sign", "passes = dac[m] == 1.0", "passes = dac[m] == 0.0", ["chk_p3_integration"]),
    ("sign_dac_split_inverted", "sign", '"dac0": z, "dac_ok": ~z', '"dac0": ~z, "dac_ok": z', ["chk_p1_exact", "chk_p1_split_direction"]),
    ("unpaired_ratio_bootstrap", "unpaired", "Wb = W\n", "Wb = boot_weights(W.shape[1], W.shape[0], SEED + 1)\n", ["chk_selfpair_zero", "chk_swap_flips", "chk_pairing_is_paired"]),
    ("unpaired_auc_bootstrap", "unpaired", "- ((W @ Sb) * W).sum(1) / ((W @ Nb) * W).sum(1)",
     "- ((np.roll(W, 1, axis=0) @ Sb) * np.roll(W, 1, axis=0)).sum(1) / ((np.roll(W, 1, axis=0) @ Nb) * np.roll(W, 1, axis=0)).sum(1)", ["chk_selfpair_zero", "chk_swap_flips"]),
    ("bootstrap_not_pe_boot", "unpaired", "W[b] = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)",
     "W[b] = np.bincount(rng.integers(0, n_groups, n_groups + 1), minlength=n_groups)", ["chk_boot_scheme_is_pe_boot"]),
    ("mask_lookup_band_axis_swapped", "xy", "xc = cell_x_centres()", "xc = AUX.Y0 + (np.arange(NCX) + 0.5) * CELL_M", ["chk_band_geometry", "chk_p1_exact"]),
    ("mask_lookup_p3_xy_swapped", "xy", "AUX.sample_pred_sdf(torch.from_numpy(np.asarray(sdf_pred, np.float32)), pts)",
     "AUX.sample_pred_sdf(torch.from_numpy(np.asarray(sdf_pred, np.float32)), pts[..., [1, 0]])", ["chk_corner_min_sdf", "chk_p3_integration"]),
    ("leak_key_guard_off", "leak", "    if key_ov or log_ov:", "    if False:", ["chk_leak_key_and_log"]),
    ("leak_heldout_guard_off", "leak", "        if outside:", "        if False:", ["chk_leak_heldout"]),
    ("leak_train_keys_not_filtered", "leak", "    return [k for k in keys if k[0] not in set(heldout_logs)]", "    return keys", ["chk_train_keys_for"]),
    ("pair_seed_ignored", "pair", 'return {n: n_by_seed.get(s) for n, (c, s) in sorted(parsed.items()) if c != "N"}',
     'return {n: next(iter(n_by_seed.values()), None) for n, (c, s) in sorted(parsed.items()) if c != "N"}', ["chk_pair_with_n", "chk_selection_exact"]),
    ("later_step_rows_kept", "io", "keep = step == 0", "keep = step >= 0", ["chk_load_head_pred"]),
    ("dup_bound_dropped", "io", "if d_lg > DUP_LOGIT_ATOL or d_sd > DUP_SDF_ATOL_M:", "if False:", ["chk_load_head_pred"]),
    ("dup_sdf_bound_dropped", "io", "if d_lg > DUP_LOGIT_ATOL or d_sd > DUP_SDF_ATOL_M:", "if d_lg > DUP_LOGIT_ATOL:", ["chk_load_head_pred"]),
    ("dup_keeps_last", "io", "                dup_close += 1\n", "                dup_close += 1\n                seen[k] = i\n                rows[rows.index(j)] = i\n", ["chk_load_head_pred"]),
    ("duplicate_keys_silent", "io", "if not (np.array_equal(lg[i], lg[j]) and np.array_equal(sd[i], sd[j])):", "if False:", ["chk_load_head_pred"]),
    ("unlabelled_counted_as_labelled", "io", 'out["valid"][s:e] = valid.numpy()', 'out["valid"][s:e] = True', ["chk_unlabelled", "chk_p1_exact"]),
    ("k2_bar_dropped", "bar", "K2_IOU_MIN = 0.80", "K2_IOU_MIN = 0.0", ["chk_k2"]),
    ("p3_gap_collapsed_into_kill", "bar", '    elif auc_head < auc_scorer:\n        base = "H2_KILLED"\n    else:\n        base = "H2_PROVISIONAL"', '    else:\n        base = "H2_KILLED"',
     ["chk_p3_decision"]),
    ("p3_band_label_reverted", "bar", 'base = "H2_PROVISIONAL"', 'base = "UNCOMMITTED_BAND"', ["chk_p3_decision", "chk_p3_floor_rule"]),
    # ---- Amendment A2: the paired P3 floor
    ("a2_floor_condition_dropped", "floor", "elif not (floor_lo > 0.0):", "elif False:", ["chk_p3_floor_rule", "chk_p3_floor_decisions", "chk_p3_floor_ci_bound"]),
    ("a2_floor_unread_is_a_pass", "floor", 'd = "FLOOR_UNREAD"', "d = base", ["chk_p3_floor_rule", "chk_p3_floor_decisions", "chk_p3_integration"]),
    ("a2_floor_diff_unpaired_weights", "unpaired", "floor_diff = auc_diff_read(Sh, Nh, Si, Ni, W)",
     "floor_diff = auc_diff_read(Sh, Nh, Si, Ni, boot_weights(len(ul), n_boot, seed + 1))", ["chk_p3_floor_same_draws"]),
    ("a2_floor_bound_is_point_estimate", "floor", 'floor_lo = floor_diff["ci95"][0]', 'floor_lo = floor_diff["est"]', ["chk_p3_floor_ci_bound"]),
    ("a2_floor_read_on_the_trained_head", "floor", "init_score, init_oob = corner_min_sdf(init_sdf, props8)", "init_score, init_oob = corner_min_sdf(head_sdf, props8)",
     ["chk_p3_floor_decisions", "chk_p3_floor_same_draws"]),
    ("a2_floor_key_guard_off", "floor", "    if st != si:", "    if False:", ["chk_floor_key_guard"]),
    ("a2_floor_arm_check_off", "floor", 'if cls == "N" or name not in arms:', "if False:", ["chk_cli_pred_init_refusals"]),
    ("p2_band_unmasked", "metric", "near = kn & (np.abs(true_sdf) <= P2_HALF_BAND_M)", "near = kn & (np.abs(true_sdf) <= 1e9)", ["chk_p2_exact"]),
    ("p2_squared_error", "metric", 'err = np.abs(pred["sdf"][s:e].astype(np.float64) - true_sdf)', 'err = (pred["sdf"][s:e].astype(np.float64) - true_sdf) ** 2', ["chk_p2_exact"]),
    ("floor_measured_with_one_seed", "floor", '"measured": n_seeds >= 3', '"measured": n_seeds >= 1', ["chk_seed_floor"]),
    ("classification_ignores_floor", "floor", 'if floor is None or not floor["measured"]:', "if False:", ["chk_seed_floor"]),
]
CLASSES = {"sign", "unpaired", "xy", "leak", "pair", "io", "bar", "metric", "floor"}


def main() -> int:
    src = open(SRC_PATH, encoding="utf-8").read()
    n = 0
    real = load(src, "aux_s1_readout_real")
    failing = run_checks(real)
    assert not failing, f"the unmutated module FAILS its own checks: {failing}"
    n += len(CHECKS)
    print(f"unmutated module: {len(CHECKS)} checks GREEN")
    extra = real_store_checks(real)
    n += len(extra)
    for e in extra:
        print(f"GREEN {e}")
    print(f"{'mutation':34s} {'class':9s} must go RED -> went RED")
    for i, (name, cls, old, new, expect) in enumerate(M):
        cnt = src.count(old)
        assert cnt == 1, f"mutation {name}: the snippet occurs {cnt} times (must be exactly 1) -- a vacuous mutation proves nothing"
        try:
            mod = load(src.replace(old, new), f"aux_s1_readout_mut{i}")
        except SyntaxError as e:
            raise AssertionError(f"mutation {name} does not compile: {e}")
        red = run_checks(mod, expect)
        red_names = {r.split("[")[0] for r in red}
        missing = [c for c in expect if c not in red_names]
        assert not missing, f"mutation {name} ({cls}) stayed GREEN on {missing}: the suite cannot see it"
        print(f"{name:34s} {cls:9s} {', '.join(red)}")
        n += 1
    assert {c for _, c, *_ in M} >= CLASSES, "a required mutation class is missing"
    n += 1
    print(f"ZZTEST_PASS {n}  ({len(CHECKS)} checks + {len(extra)} real-store + {len(M)} RED mutations + coverage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
