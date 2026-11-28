"""Pins for a0k0_headroom_retime_r8.py (ARC-LENGTH RE-TIMING: keep refcv8's path, re-time only the progress along it).

Controls of PREREG_RETIME, each with a DELIBERATE-REGRESSION arm:
  (a) g == 1 reproduces os bit for bit (and a g != 1 must NOT);
  (b) the re-timed path lies ON the os polyline (<= 1e-6 m) -- and the LON script's t0-frame along-only scaling must FAIL this same guard;
  (c) on a fixture circle every re-timed point is at distance R from the centre up to the chord sagitta (the shape is preserved);
  (d) DELIBERATE REGRESSION: the LON script's along-only scaling fails (b) and (c) on the same fixture;
  (e) g == 0 puts each point on the os polyline at arc length s_prior (INDEPENDENT pure-python locator);
plus an independent slow polyline walker for eval_polyline, a planted-gain recovery with LITERAL expectations, the fit against a 0.002 brute-force grid
(slow walker), the cross-fit leak regression, the extension / degenerate-path rules, the ACC-LAT labels, and the real deliverable's controls
(skipped only if the deliverable is absent).

    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' python -m pytest -q test_a0k0_headroom_retime_r8.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a0k0_headroom_lon_r8 as LON  # noqa: E402  (the banked LON script: the DELIBERATE-REGRESSION arm)
import a0k0_headroom_retime_r8 as M  # noqa: E402

OUT_JSON = HERE.parent / "raw" / "a0k0_retime" / "A0K0_RETIME.json"
FROZEN = HERE.parent / "raw" / "a0k0_retime" / "FROZEN_GAINS_RETIME_ema30k.json"


# ------------------------------------------------------------------ fixtures
def _path(v, w, T, dt=1.0):
    n = len(v)
    x, y, th = np.zeros(n), np.zeros(n), np.zeros(n)
    out = []
    for _ in range(T):
        x, y, th = x + v * np.cos(th) * dt, y + v * np.sin(th) * dt, th + w * dt
        out.append(np.stack([x, y], -1))
    return np.stack(out, 1)


def _windows(n=300, T=6, seed=0):
    rng = np.random.default_rng(seed)
    v, w = rng.uniform(1, 25, n), rng.normal(0, 0.12, n)
    O = _path(v, w, T).astype(np.float32)
    B = _path(v * rng.uniform(0.6, 1.1, n), w * rng.uniform(0.3, 1.0, n), T).astype(np.float32)
    return O, B


def _slow_c(O1, s):
    """INDEPENDENT pure-python polyline walker for ONE window: O1 [S, 2] points, vertices (0,0), O1[0], ...; s >= 0.  Extends along the last positive segment."""
    pts = [(0.0, 0.0)] + [(float(a), float(b)) for a, b in O1]
    segs = []
    for i in range(len(pts) - 1):
        d = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        segs.append(d)
    total = sum(segs)
    if s >= total:
        ux, uy = 1.0, 0.0
        for i in range(len(segs) - 1, -1, -1):
            if segs[i] > 0:
                ux, uy = (pts[i + 1][0] - pts[i][0]) / segs[i], (pts[i + 1][1] - pts[i][1]) / segs[i]
                break
        return (pts[-1][0] + (s - total) * ux, pts[-1][1] + (s - total) * uy)
    acc = 0.0
    for i, d in enumerate(segs):
        if d > 0 and s < acc + d:
            f = (s - acc) / d
            return (pts[i][0] + f * (pts[i + 1][0] - pts[i][0]), pts[i][1] + f * (pts[i + 1][1] - pts[i][1]))
        acc += d
    return pts[-1]


def _slow_arc(O1):
    pts = [(0.0, 0.0)] + [(float(a), float(b)) for a, b in O1]
    out, acc = [], 0.0
    for i in range(len(pts) - 1):
        acc += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        out.append(acc)
    return out


def _circle(R=40.0, S=40, dth=math.radians(1.0), frac=0.7):
    th = dth * np.arange(1, S + 1)
    O = np.stack([R * np.sin(th), R * (1 - np.cos(th))], -1)[None]          # circle through (0,0), centre (0, R), turning left
    chord = 2 * R * math.sin(dth / 2)
    B = np.stack([chord * frac * np.arange(1, S + 1), np.zeros(S)], -1)[None]   # straight prior, slower than os
    return O, B, R * (1 - math.cos(dth / 2))


def _lon_style(O, B, g):
    """The LON script's t0-frame ALONG-ONLY scaling (cross gain fixed at 1): p_x = g o_x + (1-g) b_x, p_y = o_y."""
    P = np.asarray(O, np.float64).copy()
    P[..., 0] = g * np.asarray(O, np.float64)[..., 0] + (1 - g) * np.asarray(B, np.float64)[..., 0]
    return P


# ------------------------------------------------------------------ eval_polyline vs the independent walker
def test_eval_polyline_matches_independent_walker_incl_beyond_end():
    O, B = _windows(40, 6, seed=1)
    geo = M.polyline_geometry(O)
    rng = np.random.default_rng(2)
    for _ in range(3):
        s = rng.uniform(0, 1.6, size=len(O)) * geo["L"][:, -1]          # includes s > L_S (extension)
        P = M.eval_polyline(geo, s)
        for i in range(len(O)):
            q = _slow_c(O[i], float(s[i]))
            assert abs(P[i, 0] - q[0]) <= 1e-9 and abs(P[i, 1] - q[1]) <= 1e-9


def test_arc_length_equals_independent_cumulative_chord_sum():
    O, _B = _windows(30, 6, seed=3)
    L = M.polyline_geometry(O)["L"][:, 1:]
    for i in range(len(O)):
        assert np.allclose(L[i], _slow_arc(O[i]), atol=1e-9, rtol=0)


# ------------------------------------------------------------------ (a) g == 1 reproduces os exactly
def test_a_g1_is_os_bit_for_bit_and_g_not_1_is_not():
    O, B = _windows(300, 6, seed=4)
    T = O.shape[1]
    P1 = M.retime_apply(O, B, np.ones((len(O), T)))
    assert P1.dtype == O.dtype and np.array_equal(P1, O)
    assert np.abs(M.retime_apply(O, B, np.ones((len(O), T)), cast=False) - O.astype(np.float64)).max() <= 1e-6
    P1b = M.retime_apply(O, B, np.ones(T))                                   # shared [T] gains broadcast
    assert np.array_equal(P1b, O)
    # deliberate regression: a g that is NOT 1 must move the plan (the guard has power)
    Pbad = M.retime_apply(O, B, np.full((len(O), T), 1.0001))
    assert not np.array_equal(Pbad, O)


# ------------------------------------------------------------------ (b) on the os polyline; the LON along-only scaling FAILS it
def test_b_retimed_path_lies_on_the_os_polyline_and_lon_style_scaling_fails_the_same_guard():
    O, B = _windows(300, 6, seed=5)
    n, T = O.shape[:2]
    rng = np.random.default_rng(6)
    Gn = rng.uniform(0.0, 2.0, size=(n, T))
    geo = M.polyline_geometry(O)
    P, s = M.retime_positions(geo, M.prior_arc(B), Gn)
    assert M.dist_to_path(P, geo).max() <= 1e-6                              # polyline UNION its final extension ray
    inside = s <= geo["L"][:, -1:] + 1e-12
    assert inside.sum() > 0 and (M.dist_to_path(P, geo, ray=False)[inside] <= 1e-6).all()
    d = M.retime_diag(O, B, Gn)
    assert d["max_dist_to_path_f64_m"] <= 1e-6
    assert d["max_dist_to_path_after_cast_m"] <= d["float32_rounding_bound_m"] + 1e-6
    # DELIBERATE REGRESSION: the LON t0-frame along-only scaling puts points OFF the path
    Plon = _lon_style(O, B, rng.uniform(0.0, 0.9, size=(n, T)))
    assert M.dist_to_path(Plon, geo).max() > 1e-3
    assert (M.dist_to_path(Plon, geo) > 1e-6).mean() > 0.5


def test_b_the_guard_is_not_vacuous_a_moved_point_is_detected():
    O, B = _windows(50, 6, seed=7)
    geo = M.polyline_geometry(O)
    P, _ = M.retime_positions(geo, M.prior_arc(B), np.full((50, 6), 0.6))
    P2 = P.copy()
    P2[..., 1] += 0.01                                                       # 1 cm sideways
    assert M.dist_to_path(P2, geo).max() > 1e-4


# ------------------------------------------------------------------ (c) circle fixture, (d) the LON arm fails it
def test_c_circle_fixture_every_retimed_point_stays_at_distance_R_up_to_the_sagitta():
    R = 40.0
    O, B, sag = _circle(R=R)
    S = O.shape[1]
    rng = np.random.default_rng(8)
    g = rng.uniform(0.0, 1.0, size=(1, S))                                   # g <= 1 and s_prior < s_os: s' <= L_S, no extrapolation
    geo = M.polyline_geometry(O)
    P, s = M.retime_positions(geo, M.prior_arc(B), g)
    assert (s <= geo["L"][:, -1:] + 1e-12).all()
    dist = np.hypot(P[0, :, 0] - 0.0, P[0, :, 1] - R)
    assert np.abs(dist - R).max() <= sag + 1e-9
    assert np.abs(dist - R).max() < 1e-2 * R                                 # LITERAL: far below any along-only distortion (checked next)
    # vertices are exactly on the circle: g == 1 returns os, which IS on the circle
    P1, _ = M.retime_positions(geo, M.prior_arc(B), np.ones((1, S)))
    assert np.abs(np.hypot(P1[0, :, 0], P1[0, :, 1] - R) - R).max() <= 1e-9


def test_d_deliberate_regression_lon_style_scaling_leaves_the_circle_and_the_path():
    R = 40.0
    O, B, sag = _circle(R=R)
    S = O.shape[1]
    g = np.full((1, S), 0.5)
    Plon = _lon_style(O, B, g)
    dist = np.hypot(Plon[0, :, 0], Plon[0, :, 1] - R)
    assert np.abs(dist - R).max() > 100 * sag                                # (c) fails for the LON arm
    assert M.dist_to_path(Plon, M.polyline_geometry(O)).max() > 1e-3         # (b) fails for the LON arm
    # and the banked LON fitter + blend really produce that behaviour (cross gain 1, along gain 0.5)
    Wn = np.ones((S, 2))
    Wn[:, 0] = 0.5
    Pb = LON.blend(O.astype(np.float32), B.astype(np.float32), Wn)
    assert M.dist_to_path(Pb.astype(np.float64), M.polyline_geometry(O)).max() > 1e-3


# ------------------------------------------------------------------ (e) g == 0: on the os polyline at arc length s_prior (independent locator)
def test_e_g0_is_the_os_path_point_at_the_priors_arc_length():
    O, B = _windows(60, 6, seed=9)
    P0 = M.retime_apply(O, B, np.zeros((60, 6)), cast=False)
    for i in range(60):
        sp = _slow_arc(B[i])
        for k in range(6):
            q = _slow_c(O[i], sp[k])
            assert abs(P0[i, k, 0] - q[0]) <= 1e-9 and abs(P0[i, k, 1] - q[1]) <= 1e-9
    # LITERAL: os straight along x to (10, 0), (20, 0); prior (4, 0), (8, 0): g = 0 -> s' = 4, 8 -> points (4, 0), (8, 0)
    Ol = np.array([[[10.0, 0.0], [20.0, 0.0]]])
    Bl = np.array([[[4.0, 3.0], [8.0, 6.0]]])           # arc lengths: 5, 10  (a prior that is not on the x axis)
    P = M.retime_apply(Ol, Bl, np.zeros((1, 2)), cast=False)
    assert np.allclose(P[0], [[5.0, 0.0], [10.0, 0.0]], atol=1e-12)


# ------------------------------------------------------------------ extension / degenerate-path rules (LITERALS)
def test_extension_follows_the_last_positive_segment_and_end_vertex_is_bit_exact():
    geo = M.polyline_geometry(np.array([[[1.0, 0.0], [3.0, 0.0]]]))
    assert np.array_equal(M.eval_polyline(geo, np.array([3.0])), np.array([[3.0, 0.0]]))      # s == L_S -> V_S exactly
    assert np.allclose(M.eval_polyline(geo, np.array([5.0])), [[5.0, 0.0]], atol=1e-12)
    g2 = M.polyline_geometry(np.array([[[1.0, 0.0], [1.0, 0.0]]]))                              # last segment has zero length: use the previous direction
    assert np.allclose(M.eval_polyline(g2, np.array([3.0])), [[3.0, 0.0]], atol=1e-12)
    assert not bool(g2["fallback"][0])
    g3 = M.polyline_geometry(np.array([[[0.0, 0.0], [0.0, 0.0]]]))                              # zero-length path: fallback +x, flagged
    assert bool(g3["fallback"][0]) and np.allclose(M.eval_polyline(g3, np.array([2.0])), [[2.0, 0.0]], atol=1e-12)
    g4 = M.polyline_geometry(np.array([[[0.0, 0.0], [2.0, 0.0]]]))                              # zero-length FIRST segment
    assert np.allclose(M.eval_polyline(g4, np.array([1.0])), [[1.0, 0.0]], atol=1e-12)
    g5 = M.polyline_geometry(np.array([[[0.0, 3.0], [0.0, 7.0]]]))                              # extension along +y
    assert np.allclose(M.eval_polyline(g5, np.array([10.0])), [[0.0, 10.0]], atol=1e-12)


def test_nonfinite_plan_fails_loud():
    with pytest.raises(ValueError):
        M.polyline_geometry(np.array([[[1.0, np.nan]]]))


def test_progress_is_clipped_at_zero_and_g1_is_L_exactly():
    L = np.array([[10.0, 20.0]])
    Mp = np.array([[30.0, 50.0]])
    assert np.array_equal(M.retime_progress(L, Mp, np.ones((1, 2))), L)
    assert (M.retime_progress(L, Mp, np.full((1, 2), 2.0)) >= 0).all()
    assert np.array_equal(M.retime_progress(np.array([[1.0]]), np.array([[5.0]]), np.array([[2.0]])), np.array([[0.0]]))      # 1 + (1 - 2)(5 - 1) = -3 -> clipped to 0


def test_path_membership_is_not_heading_invariance_documented_caveat():
    """PREREG_RETIME 'Where the geometry forces a choice' (1): points lie ON the polyline but the chord between two re-timed points is not an os chord."""
    O = np.array([[[10.0, 0.0], [20.0, 5.0], [30.0, 15.0]]])
    B = np.array([[[6.0, 0.0], [12.0, 0.0], [18.0, 0.0]]])
    g = np.array([[0.0, 1.0, 1.0]])                                           # pull point 1 back to s_prior (6 m), keep vertices 2 and 3
    P = M.retime_apply(O, B, g, cast=False)[0]
    geo = M.polyline_geometry(O)
    assert M.dist_to_path(P[None], geo).max() <= 1e-9
    h_os = math.degrees(math.atan2(O[0, 1, 1] - O[0, 0, 1], O[0, 1, 0] - O[0, 0, 0]))
    h_rt = math.degrees(math.atan2(P[1, 1] - P[0, 1], P[1, 0] - P[0, 0]))
    assert abs(h_os - h_rt) > 1.0                                              # LITERAL: the chord heading moved although the path did not


# ------------------------------------------------------------------ the fit
def test_fit_recovers_a_planted_gain_with_literal_expectation():
    O, B = _windows(500, 6, seed=10)
    g_true = np.array([1.2, 0.8, 0.5, 1.1, 0.9, 1.3])
    geo = M.polyline_geometry(O)
    G, _ = M.retime_positions(geo, M.prior_arc(B), g_true)
    g = M.retime_fit(O, B, G)
    assert np.allclose(g, [1.2, 0.8, 0.5, 1.1, 0.9, 1.3], atol=1e-4)
    P = M.retime_apply(O, B, g, cast=False)
    assert np.abs(P - G).max() < 1e-3


def test_fit_loss_is_not_worse_than_a_slow_brute_force_grid():
    rng = np.random.default_rng(11)
    O, B = _windows(40, 4, seed=12)
    G = M.retime_positions(M.polyline_geometry(O), M.prior_arc(B), np.array([0.7, 1.3, 0.4, 1.6]))[0] + rng.normal(0, 1.0, size=(40, 4, 2))
    g = M.retime_fit(O, B, G)
    sp = [_slow_arc(B[i]) for i in range(40)]
    so = [_slow_arc(O[i]) for i in range(40)]

    def loss(k, gv):
        tot = 0.0
        for i in range(40):
            s = max(so[i][k] + (1.0 - gv) * (sp[i][k] - so[i][k]), 0.0)
            q = _slow_c(O[i], s)
            tot += math.hypot(q[0] - G[i, k, 0], q[1] - G[i, k, 1])
        return tot / 40
    for k in range(4):
        brute = min(loss(k, float(x)) for x in np.arange(0.0, 2.0 + 1e-9, 0.002))
        assert loss(k, float(g[k])) <= brute + 1e-6


def test_flat_loss_ties_go_to_g_equal_1():
    O = np.zeros((20, 3, 2), np.float32)
    B = np.zeros((20, 3, 2), np.float32)
    G = np.zeros((20, 3, 2))
    assert np.array_equal(M.retime_fit(O, B, G), np.ones(3))


def test_insample_never_worse_than_os_known_sign():
    O, B = _windows(400, 6, seed=13)
    rng = np.random.default_rng(14)
    G = O.astype(np.float64) + rng.normal(0, 2.0, size=O.shape)
    Gn, _ = M.retime_insample(O, B, G)
    ade = lambda P: np.linalg.norm(np.asarray(P, np.float64) - G, axis=-1).mean()  # noqa: E731
    assert ade(M.retime_apply(O, B, Gn, cast=False)) <= ade(O) + 1e-9           # g == 1 is inside the box


def test_crossfit_does_not_leak_and_the_leaky_variant_does():
    O, B = _windows(500, 6, seed=15)
    clip = np.arange(500) // 10
    folds = M.episode_folds(clip)
    rng = np.random.default_rng(16)
    G = O.astype(np.float64) + rng.normal(0, 1.5, size=O.shape)
    Gn, _ = M.retime_crossfit(O, B, G, folds)
    f = 2
    G2 = G.copy()
    G2[folds == f] += 5.0                                                       # perturb ONLY fold f's ground truth
    Gn2, _ = M.retime_crossfit(O, B, G2, folds)
    assert np.array_equal(Gn[folds == f], Gn2[folds == f])                      # fold f's applied gains never saw its own GT
    Gl, _ = M.retime_crossfit(O, B, G, folds, leak=True)
    Gl2, _ = M.retime_crossfit(O, B, G2, folds, leak=True)
    assert not np.array_equal(Gl[folds == f], Gl2[folds == f])                  # DELIBERATE REGRESSION: the leaky fit does see it


def test_gated_small_bin_falls_back_to_the_ungated_fold_fit():
    O, B = _windows(300, 4, seed=17)
    G = O.astype(np.float64) + np.random.default_rng(18).normal(0, 1, size=O.shape)
    grp = np.zeros(300, np.int64)
    grp[:10] = 1                                                                # bin 1 has < MIN_BIN_FIT windows
    folds = M.episode_folds(np.arange(300))
    Gn, rec = M.retime_crossfit(O, B, G, folds, groups=grp)
    assert any("fallback" in v for k, v in rec.items() if k.endswith("|bin1"))
    for fd in np.unique(folds):
        m = (folds == fd) & (grp == 1)
        if m.any():
            assert np.allclose(Gn[m], np.broadcast_to(np.asarray(rec[f"fold{int(fd)}"]["g"]), Gn[m].shape), atol=1e-4, rtol=0)   # rec rounds to 4 dp


# ------------------------------------------------------------------ acceptance condition (shared function) and the vacuity guard
def _cell(lo, hi, d=0.0):
    return {"delta": d, "lo": lo, "hi": hi, "separated": lo > 0 or hi < 0}


def _lat(ok=True):
    return {k: _cell(-0.1, 0.1) for k in M.LATERAL_KEYS} if ok else {**{k: _cell(-0.1, 0.1) for k in M.LATERAL_KEYS}, "LAT_heading_mae_deg": _cell(0.05, 0.3, 0.2)}


def test_acc_lat_holds_fails_and_undetermined():
    assert M.acc_lat({0: _lat(), 1: _lat()})["HOLDS"] is True
    r = M.acc_lat({0: _lat(), 1: _lat(False)})
    assert r["HOLDS"] is False and r["separated_worse_at_any_seed"] == ["LAT_heading_mae_deg"] and r["separated_worse_at_both_seeds"] == []
    u = {k: {"delta": 0.0} for k in M.LATERAL_KEYS}
    assert M.acc_lat({0: u, 1: u})["HOLDS"] is None


def test_acc_lat_label_and_the_vacuity_guard():
    ok = M.acc_lat({0: _lat(), 1: _lat()})
    assert M.acc_verdict(ok, True) == "HOLDS"
    assert M.acc_verdict(ok, False).startswith("HOLDS VACUOUSLY")
    assert M.acc_verdict(M.acc_lat({0: _lat(False), 1: _lat(False)}), True) == "FAILS"
    assert M.acc_lat.__doc__ and len(M.LATERAL_KEYS) == 5


def test_arm_contract():
    assert set(M.PRIMARY_ARMS) <= set(M.ARMS)
    assert M.RT_ARMS == ("RETIME_XF", "GATED_RETIME_XF", "RETIME_IS", "GATED_RETIME_IS", "RETIME_WF", "GATED_RETIME_WF")
    assert M.ARMS["RS_XF"][0] == "rs" and not any("LON" in n or "LAT1" in n for n in M.ARMS)
    assert M.RT_BOX == (0.0, 2.0) and M.K_FOLDS == 5 and M.FOLD_SEED == 20261018 and M.MIN_BIN_FIT == 30


# ------------------------------------------------------------------ the real deliverable's controls (skipped if absent)
@pytest.mark.skipif(not OUT_JSON.exists(), reason="A0K0_RETIME.json not produced yet")
def test_deliverable_controls():
    r = M.jload(OUT_JSON)
    assert r["controls_summary"]["ALL_PASS"] is True
    assert r["controls_summary"]["R0926_exact"] is True
    for tree, tr in r["trees"].items():
        for seed, sr in tr["seeds"].items():
            assert sr["mask_sha_equal"] is True
            for surf, ss in sr["surfaces"].items():
                assert ss["decisions_check"]["PASS"] and ss["validation"]["PASS"]
                for st, rs in ss["sets"].items():
                    ct = rs["controls"]
                    assert ct["a_g1_reproduces_banked_os"]["PASS"] and ct["b_g0_is_prior"]["PASS"] and ct["i_retime_on_os_path"]["PASS"], (tree, seed, surf, st)
                    if "h_RS_XF_reproduces_banked" in ct:
                        assert ct["h_RS_XF_reproduces_banked"]["PASS"]
                    assert ct["c_d_optimism"]["IS_known_sign_le_os"] is True
                    for arm in M.RT_ARMS:
                        assert rs["arms"][arm]["geometry"]["max_dist_to_path_f64_m"] <= 1e-6


@pytest.mark.skipif(not FROZEN.exists(), reason="frozen gains not produced yet")
def test_frozen_gain_tables_respect_their_box():
    fz = M.jload(FROZEN)
    for surf, t in fz["surfaces"].items():
        for set_name in ("all", "compliant_info_only"):
            a = t[set_name]
            rt = np.asarray(a["RETIME"])
            assert rt.ndim == 1 and len(rt) == len(t["instants_s"]) and rt.min() >= 0.0 and rt.max() <= 2.0
            for v in a["GATED_RETIME"].values():
                g = np.asarray(v["g"])
                assert g.shape == rt.shape and g.min() >= 0.0 and g.max() <= 2.0
            assert a["in_sample_pooled_mean_ade"]["RETIME"] <= a["in_sample_pooled_mean_ade"]["os"] + 1e-9   # g == 1 is feasible
