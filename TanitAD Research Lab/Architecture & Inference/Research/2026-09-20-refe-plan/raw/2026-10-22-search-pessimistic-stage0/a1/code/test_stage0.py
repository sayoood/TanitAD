"""Tests for stage0.py on SYNTHETIC data only.  No real token, label or feature is read here.

Run:  CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest code/test_stage0.py -q

Layout: (A) pieces, (B) estimator, (C) the controls that must read known values (each with its RED / deliberate-regression arm), (D) the decision rule, (E) reach of every
pre-registered constant, (F) the end-to-end path through the real loader on synthetic files.
"""
import copy
import importlib.util
import itertools
import json
import math
import os
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import calib_a1 as CA  # noqa: E402
import golden_digests as GD  # noqa: E402
import stage0 as S0  # noqa: E402
import synth  # noqa: E402

L1_STEPA = os.environ.get("L1_CODE", "C:/Users/Admin/par18/l1run/code")


# ---------------------------------------------------------------------------------------------------------------------- fixtures (small worlds: 24 logs / 21 sessions / Step A kept, tokens thinned)
@pytest.fixture(scope="module")
def planted():
    w = synth.make_world(seed=3, kappa=2.0, D=32, frac=0.3)
    d = S0.make_design(w)
    return w, d, S0.boot_indices(w.S)


@pytest.fixture(scope="module")
def null_world():
    w = synth.make_world(seed=4, kappa=0.0, D=32, frac=0.3)
    d = S0.make_design(w)
    return w, d, S0.boot_indices(w.S)


@pytest.fixture(scope="module")
def planted_arms(planted):
    w, d, idx = planted
    a = S0.run_arm(w, d, S0.R0, 0, idx)
    m = S0.run_arm(w, d, S0.R0, 0, idx, fit=S0.fit_all_unexcluded(w, d, S0.R0, 0))
    return a, m


# ====================================================================================================================== (A) pieces
def test_session_of():
    assert S0.session_of("2021.06.14.18.42.45_veh-12_01345_01523") == "2021.06.14.18.42.45_veh-12"


def test_real_geometry_folds_are_balanced_and_complete():
    geo = synth.load_geometry()
    names = sorted({S0.session_of(k) for k in geo})
    cnt = {s: 0 for s in names}
    for k, v in geo.items():
        cnt[S0.session_of(k)] += v["n_tokens"]
    assert len(names) == 21 and sum(cnt.values()) == 1705
    f = S0.make_folds(np.array([cnt[s] for s in names]), names)
    assert sorted(set(f.tolist())) == [0, 1, 2, 3, 4]
    tot = np.bincount(f, weights=[cnt[s] for s in names], minlength=5)
    assert tot.sum() == 1705 and tot.min() >= 300 and tot.max() <= 380       # memo: about 341 each
    f2 = S0.make_folds(np.array([cnt[s] for s in names]), names)
    assert np.array_equal(f, f2)                                             # deterministic


def test_split_halves_strict_exact_deterministic():
    w = synth.make_world(seed=1, D=8, frac=0.3)
    h = S0.split_halves(w.token, w.log, S0.SPLIT_SEED)
    assert h.dtype == bool
    for lg in set(w.log.tolist()):
        m = w.log == lg
        assert h[m].sum() == m.sum() // 2                                   # exactly floor(n/2) search tokens per log
    assert np.array_equal(h, S0.split_halves(w.token, w.log, S0.SPLIT_SEED))
    assert not np.array_equal(h, S0.split_halves(w.token, w.log, S0.SPLIT_SEED + 1))


def test_design_every_session_in_one_fold_and_halves_cross_folds(planted):
    w, d, _ = planted
    for s in range(w.S):
        assert len(set(d.fold[w.sess_idx == s].tolist())) == 1
    assert d.info["n_search"] + d.info["n_fit"] == w.N
    assert all(x > 0 for x in d.info["fold_search_tokens"]) and all(x > 0 for x in d.info["fold_search_stepA_tokens"])


def _stepA():
    p = os.path.join(L1_STEPA, "stepA.py")
    if not os.path.exists(p):
        pytest.skip("L1 stepA.py not found")
    spec = importlib.util.spec_from_file_location("stepA_l1", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_aggregate_and_concordance_match_L1_stepA():
    sa = _stepA()
    rng = np.random.default_rng(0)
    z = rng.standard_normal((50, 6)) * 3
    assert np.allclose(S0.aggregate(z), sa.aggregate(z), atol=1e-12, rtol=0)
    ag, pg = rng.random(300), rng.random(300)
    pa, pp = np.round(rng.random(300), 2), np.round(rng.random(300), 2)
    ag[:20] = pg[:20]                                                        # ties
    num_ref, den_ref = sa.concordance_counts(ag, pg, pa, pp)
    num, den = S0.a3_terms(ag[None], pg[None], pa[None], pp[None], np.ones((1, 300), bool))
    assert np.isclose(num[0], num_ref.sum()) and np.isclose(den[0], den_ref.sum())


def test_aggregate_formula():
    z = np.zeros(6)
    assert np.isclose(S0.aggregate(z), 0.5 * 0.5 * (0.5 * 12) / 12)
    z = np.array([50, 50, 50, -50, -50, -50.0])
    assert np.isclose(S0.aggregate(z), 1 * 1 * (5 * 1 + 0 + 0) / 12)         # DDC (row 5) is unused


def test_infan_terms_bruteforce():
    rng = np.random.default_rng(1)
    pd = np.round(rng.random((3, 64)), 2)
    ag = np.round(rng.random((3, 64)), 1)                                   # many score ties
    va = np.ones((3, 64), bool)
    va[1, :5] = False
    num, den = S0.infan_terms(ag, S0.pair_mask(pd, va))
    for n in range(3):
        c = t = 0.0
        for i, j in itertools.permutations(range(64), 2):
            if va[n, i] and va[n, j] and pd[n, i] - pd[n, j] >= 0.05 - 1e-12:
                t += 1
                c += 1.0 if ag[n, i] > ag[n, j] else (0.5 if ag[n, i] == ag[n, j] else 0.0)
        assert np.isclose(num[n], c) and np.isclose(den[n], t)


def test_regret_bruteforce_and_random_pick_expectation_is_exact():
    rng = np.random.default_rng(2)
    pd = rng.random((4, 64))
    ag = rng.random((4, 64))
    va = np.ones((4, 64), bool)
    r = S0.regret_points(ag, pd, va)
    for n in range(4):
        assert np.isclose(r[n], 100 * (pd[n].max() - pd[n][np.argmax(ag[n])]))
    # a random pick's expected regret is max - mean: average the regret over many label permutations of ONE token
    pdn = pd[0]
    perm = np.stack([rng.permutation(64) for _ in range(20000)])
    reg = 100 * (pdn.max() - pdn[perm][:, np.argmax(ag[0])])
    assert abs(reg.mean() - 100 * (pdn.max() - pdn.mean())) < 4 * reg.std() / math.sqrt(len(reg))


def _toy_batch(rng, D=7, B=11, B2=9):
    return (rng.standard_normal((B, D)), rng.random((B, 5)), rng.standard_normal((B2, D)), rng.standard_normal((B2, D)), rng.choice([-1.0, 0.0, 1.0], (B2, 5)))


def test_loss_gradient_matches_finite_differences_and_frozen_row_is_exactly_zero():
    rng = np.random.default_rng(3)
    D = 7
    W0, b0 = rng.standard_normal((6, D)), rng.standard_normal(6)
    W, b = W0 + 0.1 * rng.standard_normal((6, D)), b0 + 0.1 * rng.standard_normal(6)
    xf, tf, xa, xp, y = _toy_batch(rng, D)
    rec = S0.Recipe(prox=0.4)
    tot, _, _, gW, gb = S0.loss_grad(W, b, W0, b0, xf, tf, xa, xp, y, rec)
    assert np.all(gW[S0.DDC] == 0.0) and gb[S0.DDC] == 0.0                   # EXACTLY zero
    eps = 1e-6
    for (r, c) in [(0, 0), (1, 3), (2, 5), (3, 1), (4, 6)]:
        Wp, Wm = W.copy(), W.copy()
        Wp[r, c] += eps
        Wm[r, c] -= eps
        fd = (S0.loss_grad(Wp, b, W0, b0, xf, tf, xa, xp, y, rec)[0] - S0.loss_grad(Wm, b, W0, b0, xf, tf, xa, xp, y, rec)[0]) / (2 * eps)
        assert np.isclose(gW[r, c], fd, rtol=1e-5, atol=1e-8)
    for r in range(5):
        bp, bm = b.copy(), b.copy()
        bp[r] += eps
        bm[r] -= eps
        fd = (S0.loss_grad(W, bp, W0, b0, xf, tf, xa, xp, y, rec)[0] - S0.loss_grad(W, bm, W0, b0, xf, tf, xa, xp, y, rec)[0]) / (2 * eps)
        assert np.isclose(gb[r], fd, rtol=1e-5, atol=1e-8)
    # the probe can read a non-zero at all: the RED mutation routes a term through the frozen row
    _, _, _, gWm, gbm = S0.loss_grad(W, b, W0, b0, xf, tf, xa, xp, y, rec, touch_ddc=True)
    assert np.abs(gWm[S0.DDC]).max() > 0.0
    # every trainable row receives a non-zero gradient
    assert all(np.abs(gW[r]).sum() > 0 for r in S0.ROWS)
    # prox pulls toward P2: zero gradient contribution at the init
    t0 = S0.loss_grad(W0, b0, W0, b0, xf, tf, xa, xp, y, S0.Recipe(prox=0.4))
    t1 = S0.loss_grad(W0, b0, W0, b0, xf, tf, xa, xp, y, S0.Recipe(prox=0.0))
    assert np.array_equal(t0[3], t1[3])


def test_frozen_row_check_fires_when_a_term_touches_it(planted):
    w, d, _ = planted
    tr = S0.train_tokens_for_model(w, d, 0)
    with pytest.raises(RuntimeError, match="frozen DDC"):
        S0.fit_model(w, tr, S0.Recipe(steps=3), 0, touch_ddc=True)


def test_adam_matches_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    D = 6
    W0, b0 = rng.standard_normal((6, D)), rng.standard_normal(6)
    xf, tf, xa, xp, y = _toy_batch(rng, D)
    rec = S0.Recipe(steps=25, lr=3e-3, prox=0.2)
    W, b = W0.copy(), b0.copy()
    st = S0.adam_state(W, b)
    tW, tb = torch.nn.Parameter(torch.tensor(W0)), torch.nn.Parameter(torch.tensor(b0))
    opt = torch.optim.Adam([tW, tb], lr=rec.lr, betas=rec.betas, eps=rec.eps)
    for step in range(1, rec.steps + 1):
        _, _, _, gW, gb = S0.loss_grad(W, b, W0, b0, xf, tf, xa, xp, y, rec)
        S0.adam_update(W, b, gW, gb, st, step, rec)                                  # the function fit_model calls
        _, _, _, tgW, tgb = S0.loss_grad(tW.detach().numpy(), tb.detach().numpy(), W0, b0, xf, tf, xa, xp, y, rec)
        tW.grad, tb.grad = torch.tensor(tgW), torch.tensor(tgb)
        opt.step()
    assert np.allclose(W, tW.detach().numpy(), atol=1e-10) and np.allclose(b, tb.detach().numpy(), atol=1e-10)


def test_fit_model_moves_trainable_rows_keeps_ddc_bit_identical_and_is_seeded(planted):
    w, d, _ = planted
    tr = S0.train_tokens_for_model(w, d, 0)
    m1 = S0.fit_model(w, tr, S0.Recipe(steps=80), 0)
    m2 = S0.fit_model(w, tr, S0.Recipe(steps=80), 0)
    m3 = S0.fit_model(w, tr, S0.Recipe(steps=80), 1)
    assert np.array_equal(m1["W"], m2["W"]) and np.array_equal(m1["b"], m2["b"])           # deterministic
    assert not np.array_equal(m1["W"], m3["W"])                                              # the seed reaches the minibatch stream
    assert np.array_equal(m1["W"][S0.DDC], w.W6[S0.DDC]) and m1["b"][S0.DDC] == w.b6[S0.DDC]
    assert all(np.abs(m1["W"][r] - w.W6[r]).max() > 0 for r in S0.ROWS)
    assert m1["audit"]["min_ep_row_grad_abs_sum"] > 0 and all(n > 0 for n in m1["audit"]["pairs_active_by_row"])


# ====================================================================================================================== (B) the estimator
def test_cluster_gap_ratio_of_sums_and_loso():
    sess = np.array([0, 0, 1, 1, 2])
    A = np.array([3., 1., 2., 2., 5.])
    B = np.array([1., 1., 1., 1., 1.])
    den = np.array([4., 4., 4., 4., 2.])
    idx = S0.boot_indices(3, 2000, 1)
    g = S0.cluster_gap(sess, 3, A, B, den, idx)
    assert np.isclose(g["point"], (13 - 5) / 18)
    assert np.isclose(g["loso_min"], (13 - 5 - (5 - 1)) / (18 - 2)) and np.isclose(g["loso_max"], (13 - 5 - (4 - 2)) / (18 - 8))    # drop session 2 -> 0.25 ; drop session 0 or 1 -> 0.6
    # bootstrap resamples whole SESSIONS: with only session 2 carrying signal, resamples that miss it read exactly 0
    sess2 = np.array([0, 1, 2])
    g2 = S0.cluster_gap(sess2, 3, np.array([0., 0., 1.]), np.zeros(3), np.ones(3), S0.boot_indices(3, 4000, 2))
    assert g2["lo95"] == 0.0 and g2["hi95"] > 0 and g2["point"] == pytest.approx(1 / 3)
    # the same idx must give the same answer (paired draws across measures)
    assert S0.cluster_gap(sess, 3, A, B, den, idx) == g


def test_boot_indices_are_seeded_and_session_shaped():
    a, b = S0.boot_indices(21), S0.boot_indices(21)
    assert a.shape == (S0.BOOT_B, 21) and np.array_equal(a, b) and a.max() == 20


def test_instrument_identity_all_models_equal_P2_gives_exactly_zero_gaps(planted):
    w, d, idx = planted
    a = S0.run_arm(w, d, S0.R0, 0, idx, fit={"models": S0.p2_models(w), "audit": [], "recipe": None, "seed": None})
    for m in ("M1", "M2", "M3"):
        assert a["gaps"][m]["point"] == 0.0
    assert all(a["swap_null"][f"nu_{m}"] == 0.0 for m in ("M1", "M2", "M3"))


# ====================================================================================================================== (C) CONTROLS
# ---- C-STRUCT
def test_cstruct_default_pipeline_excludes_each_models_fold(planted):
    w, d, _ = planted
    fa = S0.fit_all(w, d, S0.Recipe(steps=5), 0)
    for h, a in enumerate(fa["audit"]):
        tr = S0.train_tokens_for_model(w, d, h)
        assert not np.isin(d.fold[tr], [h]).any() and not d.is_search[tr].any()
        assert a["own_fold_tokens"] == 0 and a["own_fold_sessions"] == 0


def test_cstruct_RED_deleting_the_exclusion_line_fails_closed(planted):
    w, d, _ = planted
    mod = S0.load_mutated(S0.MUT_NO_EXCLUDE, "stage0_mut_a")
    with pytest.raises(RuntimeError, match="C-LEAK"):
        mod.fit_all(w, d, S0.Recipe(steps=3), 0)


def test_cstruct_RED_a_search_token_or_a_foreign_session_in_training_is_caught(planted):
    w, d, _ = planted
    tr = S0.train_tokens_for_model(w, d, 0)
    with pytest.raises(RuntimeError, match="C-LEAK"):
        S0.check_exclusion(w, d, 0, np.append(tr, np.nonzero(d.is_search)[0][0]))
    other_fold0_fit = np.nonzero((d.fold == 0) & ~d.is_search)[0][:1]
    with pytest.raises(RuntimeError, match="C-LEAK"):
        S0.check_exclusion(w, d, 0, np.append(tr, other_fold0_fit))


def test_the_mutation_markers_each_match_exactly_one_line():
    for mk in (*S0.MUT_NO_EXCLUDE, *S0.MUT_NO_CHECK):
        src = open(os.path.join(HERE, "stage0.py"), encoding="utf-8").read().replace("\r\n", "\n").split("\n")
        hits = [ln for ln in src if ln.rstrip().endswith(mk) and not ln.lstrip().startswith(("#", '"'))]
        assert len(hits) == 1
    with pytest.raises(RuntimeError):
        S0.load_mutated({"# NO-SUCH-MARKER": "pass"}, "stage0_mut_x")


# ---- C-III : delete the exclusion -> the gap collapses
def test_ciii_planted_gap_is_read_and_collapses_when_the_exclusion_is_deleted(planted_arms):
    a, m = planted_arms
    g, gm = a["gaps"]["M1"]["point"], m["gaps"]["M1"]["point"]
    assert g > 0.01 and a["gaps"]["M1"]["lo_bonf"] > a["swap_null"]["nu_M1"] > 0         # the instrument sees it ... (and the swap null is a real, non-zero floor)
    assert a["gaps"]["M2"]["point"] > 0 and a["gaps"]["M3"]["point"] > 0                  # ... on every measure, in the quality direction
    assert abs(gm) < 0.25 * g                                                               # ... and the mutated pipeline does not
    c = S0.control_collapse(a, m)
    assert c["pass"], c


def test_ciii_RED_control_fails_if_the_gap_does_not_collapse(planted_arms):
    a, _ = planted_arms
    assert not S0.control_collapse(a, a)["pass"]                                             # "mutated" arm that still has the gap -> control must FAIL


# ---- C-II-a : SEEN vs OOF.  Real world: non-reversal.  Planted world: SEEN must BEAT OOF (strict).
def test_cii_a_real_world_gate_is_non_reversal(planted_arms):
    a, _ = planted_arms
    c = S0.control_seen(a)
    assert c["pass"] and c["strict_positive"] and c["token_level_z"] > S0.Z_MAX                     # a gap exists by construction here: SEEN beats OOF at the token level
    flat = copy.deepcopy(a)
    flat["seen_minus_oof"]["M1"]["point"] = -0.002                                                   # a small negative point with a token z of -1.5 (a valid read, found in a synthetic null world)
    flat["seen_minus_oof"]["M1_token"] = {"value": -0.002, "se": 0.0013, "z": -1.5, "n": 800}
    assert S0.control_seen(flat)["pass"] and not S0.control_seen(flat)["strict_positive"]
    rev = copy.deepcopy(a)
    rev["seen_minus_oof"]["M1_token"] = {"value": -0.02, "se": 0.001, "z": -20.0, "n": 800}         # SEEN significantly WORSE than OOF: a fit/search mix-up or a leak -> INVALID
    assert not S0.control_seen(rev)["pass"]


def test_cii_a_strict_rule_is_gated_in_the_planted_world(planted_arms):
    a, m = planted_arms
    assert S0.control_planted(a, m)["pass"] and S0.control_planted(a, m)["seen_beats_oof_planted"]
    blind = copy.deepcopy(a)
    blind["seen_minus_oof"]["M1"]["point"] = 0.0                                                     # a planted world where trained-on tokens do NOT beat unseen ones
    assert not S0.control_planted(blind, m)["pass"]


# ---- C-II-b : planted effect on features, recovered; RED on a world with no planted effect
def test_cii_b_planted_control_passes_and_RED_without_a_plant(planted_arms, null_world):
    a, m = planted_arms
    assert S0.control_planted(a, m)["pass"]
    w0, d0, idx0 = null_world
    a0 = S0.run_arm(w0, d0, S0.R0, 0, idx0)
    m0 = S0.run_arm(w0, d0, S0.R0, 0, idx0, fit=S0.fit_all_unexcluded(w0, d0, S0.R0, 0))
    res = S0.control_planted(a0, m0)
    assert not res["planted_gap_read"] and not res["pass"]


def test_plant_labels_are_synthetic_and_valid(planted):
    w, _, _ = planted
    L = w.labels
    assert np.isfinite(L.sub).all() and L.sub.min() >= 0 and L.sub.max() <= 1
    assert np.allclose(L.pdms, L.sub[..., 0] * L.sub[..., 1] * L.sub[..., 5] * (5 * L.sub[..., 2] + 5 * L.sub[..., 3] + 2 * L.sub[..., 4]) / 12)
    assert np.array_equal(S0.plant_labels(w, 1.0, 5).sub, S0.plant_labels(w, 1.0, 5).sub)
    assert not np.array_equal(S0.plant_labels(w, 1.0, 5).sub, S0.plant_labels(w, 1.0, 6).sub)


# ---- C-I : shuffled labels -> chance
def test_shuffled_labels_keep_the_multiset_and_break_the_relation(planted):
    w, _, _ = planted
    Ls = S0.shuffled_labels(w, 9001)
    non_sa = np.nonzero(w.sa_of_row < 0)[0]
    assert np.allclose(np.sort(Ls.pdms[non_sa], 1), np.sort(w.labels.pdms[non_sa], 1))      # a plain token: its own 64 labels, permuted
    pooled = np.concatenate([w.labels.pdms[w.sa_row], w.labels.p_pdms], 1)
    pooled_s = np.concatenate([Ls.pdms[w.sa_row], Ls.p_pdms], 1)
    assert np.allclose(np.sort(pooled, 1), np.sort(pooled_s, 1))                                # a Step A token: fan + perturbations POOLED and re-dealt
    assert (Ls.pdms != w.labels.pdms).any() and (Ls.p_pdms != w.labels.p_pdms).any()
    assert np.allclose(Ls.pdms, Ls.sub[..., 0] * Ls.sub[..., 1] * Ls.sub[..., 5] * (5 * Ls.sub[..., 2] + 5 * Ls.sub[..., 3] + 2 * Ls.sub[..., 4]) / 12)   # sub and pdms move together
    assert np.allclose(Ls.p_pdms, Ls.p_sub[..., 0] * Ls.p_sub[..., 1] * Ls.p_sub[..., 5] * (5 * Ls.p_sub[..., 2] + 5 * Ls.p_sub[..., 3] + 2 * Ls.p_sub[..., 4]) / 12)
    assert not np.array_equal(S0.shuffled_labels(w, 9001).pdms, S0.shuffled_labels(w, 9002).pdms)
    assert np.array_equal(S0.shuffled_labels(w, 9001).pdms, S0.shuffled_labels(w, 9001).pdms)


def test_pooling_is_what_puts_A3_at_chance():
    """a perturbation is usually WORSE than its anchor, so shuffling the fan and the perturbations separately leaves a blind score away from 0.5; the pooled deal does not"""
    w, s = _stepA_world(seed=2, s_const=-1.0)                               # every perturbation scores LOWER than its anchor, and its true pdms is lower too
    d = S0.make_design(w)
    L = w.labels
    L2 = S0.Labels(L.sub, L.pdms, L.valid, L.p_sub, L.p_pdms, L.pair_ok)
    w2 = w.with_labels(L2)
    cache = S0.build_cache(w2, d, S0.p2_models(w2), "search", roles=(0,))
    sep, pool = [], []
    for sd in range(9001, 9041):
        rng = np.random.default_rng(sd)
        Ls = S0.Labels(L2.sub, np.take_along_axis(L2.pdms, np.argsort(rng.random(L2.pdms.shape), 1), 1), L2.valid, L2.p_sub,
                       np.take_along_axis(L2.p_pdms, np.argsort(rng.random(L2.p_pdms.shape), 1), 1), L2.pair_ok)        # the two sets shuffled SEPARATELY
        r = S0.measure(w2, d, cache, Ls)
        sep.append(r.m2_num[0].sum() / r.m2_den.sum())
        r = S0.measure(w2, d, cache, S0.shuffled_labels(w2, sd))
        pool.append(r.m2_num[0].sum() / r.m2_den.sum())
    assert abs(np.mean(pool) - 0.5) < 0.02 and abs(np.mean(sep) - 0.5) > 0.05


@pytest.fixture()
def relaxed_delta(monkeypatch):
    """the synthetic worlds have far noisier regrets than the real table, so the informativeness guard (4 x SE <= DELTA) is relaxed HERE only; its own test below uses the real DELTA"""
    monkeypatch.setitem(S0.DELTA, "M1", 0.05)
    monkeypatch.setitem(S0.DELTA, "M2", 0.10)
    monkeypatch.setitem(S0.DELTA, "M3", 10.0)


def test_ci_chance_on_shuffled_labels_passes_and_RED_on_true_labels(planted_arms, planted, relaxed_delta):
    w, d, _ = planted
    a, _ = planted_arms
    c = S0.control_shuffle(w, d, a["cache"], seeds=S0.SHUF_SEEDS[:40])
    assert c["pass"], c
    assert abs(c["M1_concordance_minus_half"]["value"]) < 0.01 and not c["uninformative"]
    red = S0.chance_check(w, d, [(a["cache"], w.labels)])                                    # unshuffled labels: a trained head is NOT at chance
    assert not red["pass"] and not red["checks"]["M1_concordance_minus_half"] and not red["checks"]["M1_gap"]


def test_ci_trained_on_shuffled_labels_is_at_chance(planted, relaxed_delta):
    w, d, idx = planted
    c = S0.control_shuffle_world(w, d, S0.Recipe(steps=150), idx, seeds=S0.SHUF_WORLD_SEEDS[:4])
    assert c["pass"], c


def test_ci_RED_a_score_that_carries_the_label_is_caught(planted, relaxed_delta):
    """the full-pipeline control judges the heads against the labels they were trained on; a score that CONTAINS the (shuffled) label must read far from chance"""
    w, d, _ = planted
    ws = w.with_labels(S0.shuffled_labels(w, 9101))
    cache = S0.build_cache(ws, d, S0.p2_models(ws))
    for (g, k) in list(cache.agg):
        cache.agg[(g, k)] = ws.labels.pdms[cache.idx[g]] + 1e-6 * np.random.default_rng(g * 7 + k).standard_normal(cache.agg[(g, k)].shape)
    leaky = S0.chance_check(ws, d, [(cache, ws.labels)])
    assert leaky["M1_concordance_minus_half"]["z"] > S0.Z_MAX and not leaky["pass"]


def test_ci_RED_an_uninformative_control_fails_and_an_uninformative_M3_only_uncertifies(planted_arms, planted, monkeypatch):
    """a control that could not have seen a bias of the size that matters must not pass (M1, M2); for M3 it only withdraws the certificate"""
    w, d, _ = planted
    a, _ = planted_arms
    monkeypatch.setitem(S0.DELTA, "M1", 1e-6)                                                  # M1 cannot be resolved to 1e-6
    c = S0.chance_check(w, d, [(a["cache"], S0.shuffled_labels(w, 9001))])
    assert "M1_gap" in c["uninformative"] and not c["pass"] and c["informative"]["M1"] is False
    monkeypatch.undo()
    monkeypatch.setitem(S0.DELTA, "M1", 0.05)
    monkeypatch.setitem(S0.DELTA, "M2", 0.10)
    c = S0.chance_check(w, d, [(a["cache"], S0.shuffled_labels(w, 9001))])                      # the real M3 delta (0.35) vs the synthetic regret noise of ONE shuffle
    assert c["informative"]["M3"] is False and c["informative"]["M1"] and c["pass"]            # M3 uncertified, the control still passes


def test_uncertified_M3_cannot_trigger_gap_or_power_a_null():
    d = S0.decide(_prim(), _prim(), SWAP, OK, certified={"M1": False, "M2": False, "M3": True})
    assert d["outcome"] == "GAP" and d["gap_measures"] == ["M3"]
    d = S0.decide({"M1": G(0.0, se=0.001), "M2": G(0.0, se=0.01), "M3": G(0.9, se=0.2)}, _prim(), SWAP, OK, certified={"M3": False})
    assert d["outcome"] == "NULL" and d["sub"] == "UNDERPOWERED" and d["per_measure"]["M3"]["certified"] is False
    null = {"M1": G(0.0, se=0.002), "M2": G(0.0, se=0.03), "M3": G(0.0, se=0.05)}
    assert S0.decide(null, null, SWAP, OK)["sub"] == "POWERED"
    assert S0.decide(null, null, SWAP, OK, certified={"M3": False})["sub"] == "UNDERPOWERED"


# ====================================================================================================================== (D) the decision rule
def G(point, lo=None, hi=None, se=0.005, loso_min=None):
    return {"point": point, "lo_bonf": point - 3 * se if lo is None else lo, "hi_bonf": point + 3 * se if hi is None else hi, "se": se,
            "loso_min": point * 0.5 if loso_min is None else loso_min}


def _prim(m1=0.03, m2=0.05, m3=0.9, **kw):
    return {"M1": G(m1, **kw), "M2": G(m2, se=0.03), "M3": G(m3, se=0.2)}


SWAP = {"nu_M1": 0.002, "nu_M2": 0.01, "nu_M3": 0.1}
OK = {"C-A": {"pass": True}, "C-B": {"pass": True}}


def test_decide_gap_requires_all_four_conditions():
    base = S0.decide(_prim(), _prim(), SWAP, OK)
    assert base["outcome"] == "GAP" and "M1" in base["gap_measures"]
    # S : lower bound above the swap null
    p = _prim(); p["M1"]["lo_bonf"] = 0.001
    p["M2"] = G(0.0, se=0.03); p["M3"] = G(0.0, se=0.2)
    assert S0.decide(p, _prim(), SWAP, OK)["outcome"] == "NULL"
    # M : magnitude >= delta
    assert S0.decide({**_prim(), "M1": G(0.005, se=0.001), "M2": G(0.0, se=0.03), "M3": G(0.0, se=0.2)}, _prim(), SWAP, OK)["outcome"] == "NULL"
    # R : the seed replicate must agree
    rep = _prim(); rep["M1"] = G(0.004)
    assert S0.decide({**_prim(), "M2": G(0.0, se=0.03), "M3": G(0.0, se=0.2)}, rep, SWAP, OK)["outcome"] == "NULL"
    # L : one session must not carry it
    assert S0.decide({**_prim(m1=0.03, loso_min=-0.001), "M2": G(0.0, se=0.03), "M3": G(0.0, se=0.2)}, _prim(), SWAP, OK)["outcome"] == "NULL"


def test_decide_invalid_takes_precedence_and_names_the_control():
    d = S0.decide(_prim(), _prim(), SWAP, {"C-I-eval": {"pass": False}, "C-II-a": {"pass": True}})
    assert d["outcome"] == "INVALID" and d["failed_controls"] == ["C-I-eval"]


def test_decide_null_powered_vs_underpowered():
    null = {"M1": G(0.0, se=0.002), "M2": G(0.0, se=0.03), "M3": G(0.0, se=0.05)}
    d = S0.decide(null, null, SWAP, OK)
    assert d["outcome"] == "NULL" and d["sub"] == "POWERED"
    wide = {"M1": G(0.0, se=0.002), "M2": G(0.0, se=0.03), "M3": G(0.0, se=0.5)}      # M3 upper bound 1.5 > delta 0.35
    d = S0.decide(wide, wide, SWAP, OK)
    assert d["outcome"] == "NULL" and d["sub"] == "UNDERPOWERED" and "NOT" in d["routing"]
    assert d["per_measure"]["M3"]["achieved_mde80"] == pytest.approx(S0.MDE_FACTOR * 0.5)


def test_decide_flags_a_significantly_reversed_gap():
    p = _prim()
    p["M1"] = G(-0.03, se=0.005)                                    # OOF significantly BETTER than IN: an anomaly the MM must see, never a GAP
    d = S0.decide(p, _prim(), SWAP, OK)
    assert d["per_measure"]["M1"]["reverse_gap"] is True and "M1" not in d.get("gap_measures", [])
    assert S0.decide(_prim(), _prim(), SWAP, OK)["per_measure"]["M1"]["reverse_gap"] is False


def test_mde_factor_value():
    assert S0.ALPHA_M == pytest.approx(0.025 / 3)
    assert S0.MDE_FACTOR == pytest.approx(2.394 + 0.8416, abs=2e-3)


def test_decide_handles_unavailable_m2():
    p = _prim()
    p["M2"] = {"point": float("nan"), "n_den": 0.0}
    r = _prim()
    r["M2"] = {"point": float("nan"), "n_den": 0.0}
    d = S0.decide(p, r, SWAP, OK)
    assert d["per_measure"]["M2"]["available"] is False and d["outcome"] == "GAP"


# ====================================================================================================================== (E) every pre-registered constant is REACHED
def test_reach_thr_pair_changes_the_pair_set(planted):
    w, d, idx = planted
    base = S0.pair_mask(w.labels.pdms[:20], w.labels.valid[:20]).sum()
    old = S0.THR_PAIR
    try:
        S0.THR_PAIR = 0.5
        assert S0.pair_mask(w.labels.pdms[:20], w.labels.valid[:20]).sum() < base
    finally:
        S0.THR_PAIR = old


def test_reach_split_seed_and_recipe_levers(planted):
    w, d, _ = planted
    assert not np.array_equal(S0.make_design(w, S0.SPLIT_SEED).is_search, S0.make_design(w, S0.SPLIT_SEED + 5).is_search)
    tr = S0.train_tokens_for_model(w, d, 1)
    a = S0.fit_model(w, tr, S0.Recipe(steps=60, prox=0.0), 0)["W"]
    assert not np.array_equal(a, S0.fit_model(w, tr, S0.Recipe(steps=60, prox=1.0), 0)["W"])      # prox reaches
    assert not np.array_equal(a, S0.fit_model(w, tr, S0.Recipe(steps=61, prox=0.0), 0)["W"])      # steps reaches
    assert not np.array_equal(a, S0.fit_model(w, tr, S0.Recipe(steps=60, lam_bce=1.0), 0)["W"])   # lam_bce reaches
    assert not np.array_equal(a, S0.fit_model(w, tr, S0.Recipe(steps=60, lr=1e-2), 0)["W"])        # lr reaches
    assert not np.array_equal(a, S0.fit_model(w, tr, S0.Recipe(steps=60, batch=64), 0)["W"])       # batch reaches
    assert S0.R0.prox == 0.0 and S0.R1.prox == 0.3                                                 # the two registered recipes differ exactly in prox


def test_reach_pairwise_term_reaches_the_head(planted):
    """the pairwise term must change the fit (a 'built, tested, unreachable' check): zero out every Step A pair label and the fit changes"""
    w, d, _ = planted
    tr = S0.train_tokens_for_model(w, d, 2)
    a = S0.fit_model(w, tr, S0.Recipe(steps=60), 0)["W"]
    L = w.labels
    L2 = S0.Labels(L.sub, L.pdms, L.valid, L.p_sub, L.p_pdms, np.zeros_like(L.pair_ok))
    b = S0.fit_model(w.with_labels(L2), tr, S0.Recipe(steps=60), 0)["W"]
    assert not np.array_equal(a, b)


def test_reach_decision_constants(monkeypatch):
    assert S0.decide(_prim(), _prim(), SWAP, OK)["outcome"] == "GAP"
    monkeypatch.setitem(S0.DELTA, "M1", 10.0)
    monkeypatch.setitem(S0.DELTA, "M2", 10.0)
    monkeypatch.setitem(S0.DELTA, "M3", 10.0)
    assert S0.decide(_prim(), _prim(), SWAP, OK)["outcome"] == "NULL"
    monkeypatch.undo()
    monkeypatch.setattr(S0, "REPL_FRAC", 5.0)
    assert S0.decide(_prim(), _prim(), SWAP, OK)["outcome"] == "NULL"


def test_reach_alpha_and_bootstrap_constants():
    sess = np.repeat(np.arange(21), 3)
    rng = np.random.default_rng(0)
    A, B, den = rng.random(63), rng.random(63), np.ones(63)
    idx = S0.boot_indices(21, 3000, 3)
    g1 = S0.cluster_gap(sess, 21, A, B, den, idx)
    old = S0.ALPHA_M
    try:
        S0.ALPHA_M = 0.2
        g2 = S0.cluster_gap(sess, 21, A, B, den, idx)
    finally:
        S0.ALPHA_M = old
    assert g2["lo_bonf"] > g1["lo_bonf"]
    assert S0.boot_indices(21, 10, 1).shape == (10, 21) and S0.BOOT_B == 10000 and len(S0.SHUF_SEEDS) == 100 and len(S0.SHUF_WORLD_SEEDS) == 10


def test_reach_chance_tolerances(planted_arms, planted, relaxed_delta, monkeypatch):
    w, d, _ = planted
    a, _ = planted_arms
    assert S0.control_shuffle(w, d, a["cache"], seeds=S0.SHUF_SEEDS[:20])["pass"]
    monkeypatch.setattr(S0, "Z_MAX", 1e-9)
    assert not S0.control_shuffle(w, d, a["cache"], seeds=S0.SHUF_SEEDS[:20])["pass"]


def test_reach_planted_and_collapse_constants(planted_arms, monkeypatch):
    a, m = planted_arms
    assert S0.control_planted(a, m)["pass"]
    monkeypatch.setattr(S0, "PLANT_MIN_M1", 10.0)
    assert not S0.control_planted(a, m)["pass"]
    monkeypatch.undo()
    monkeypatch.setattr(S0, "COLLAPSE_FRAC", 0.0)
    monkeypatch.undo()
    monkeypatch.setattr(S0, "SEEN_MARGIN", 10.0)
    assert not S0.control_planted(a, m)["pass"]
    assert S0.plant_labels.__defaults__[:2] == (S0.PLANT_KAPPAS[-1], S0.PLANT_SEED) and S0.plant_labels.__defaults__[3] == S0.PLANT_SHARP


def test_constants_digest_changes_with_any_constant(monkeypatch):
    base = S0.constants_sha256()
    assert len(base) == 64
    for name, val in (("SPLIT_SEED", 1), ("BOOT_B", 5), ("REPL_FRAC", 0.9), ("Z_MAX", 5.0), ("PLANT_KAPPAS", (1.0,)), ("COLLAPSE_FRAC", 0.5), ("SEEN_MARGIN", 0.1), ("PLANT_MIN_M1", 0.1),
                      ("N_SEEDS", S0.N_SEEDS + 1), ("REPL_SEED_BASE", 200)):
        monkeypatch.setattr(S0, name, val)
        assert S0.constants_sha256() != base, name
        monkeypatch.undo()
    monkeypatch.setitem(S0.DELTA, "M3", 9.0)
    assert S0.constants_sha256() != base


# ====================================================================================================================== (F) the real entry path on SYNTHETIC files
def test_loader_roundtrip_on_synthetic_files_and_abort_conditions(tmp_path):
    w0 = synth.make_files(str(tmp_path), seed=11, kappa=1.0, D=16, frac=0.2)
    w, man = S0.load_world(str(tmp_path), verify_sha=False)
    assert (w.N, w.D, w.Ns, w.Q, w.S) == (w0.N, w0.D, w0.Ns, w0.Q, w0.S)
    assert np.array_equal(w.sa_row, w0.sa_row) and np.array_equal(w.anchor, w0.anchor)
    assert np.allclose(w.X, w0.X, atol=2e-2, rtol=1e-3)                                      # fp16 round trip
    assert np.allclose(w.labels.pdms, w0.labels.pdms) and w.labels.pair_ok.all()
    # the sha256 gate: synthetic files can never pass the real gate (RED proves the gate is live)
    with pytest.raises(RuntimeError, match="sha256"):
        S0.load_world(str(tmp_path), verify_sha=True)
    # alignment abort: tamper with one fan logit
    p = os.path.join(str(tmp_path), S0.INPUTS["bundle"][0])
    z = dict(np.load(p))
    z["logits"][0, 0, 0] += 1.0
    np.savez(p, **z)
    with pytest.raises(RuntimeError, match="logits differ"):
        S0.load_world(str(tmp_path), verify_sha=False)
    z["logits"][0, 0, 0] -= 1.0
    np.savez(p, **z)                                                                          # restore, then break the labels' scale and finiteness in the table
    pt = os.path.join(str(tmp_path), S0.INPUTS["table"][0])
    zt = dict(np.load(pt))
    bad = dict(zt)
    bad["pdms"] = zt["pdms"] * 120.0
    np.savez(pt, **bad)
    with pytest.raises(RuntimeError, match="scale"):
        S0.load_world(str(tmp_path), verify_sha=False)
    bad = dict(zt)
    bad["sub"] = zt["sub"].copy()
    bad["sub"][0, 0, 0] = np.nan
    np.savez(pt, **bad)
    with pytest.raises(RuntimeError, match="non-finite"):
        S0.load_world(str(tmp_path), verify_sha=False)


def test_run_all_end_to_end_on_synthetic_files(tmp_path):
    synth.make_files(str(tmp_path), seed=12, kappa=2.0, D=16, frac=0.25)
    w, man = S0.load_world(str(tmp_path), verify_sha=False)
    d = S0.make_design(w)
    res = S0.run_all(w, d, man, log=lambda *_: None)
    assert set(res["controls"]) == {"C-STRUCT", "C-I-eval", "C-I-trained", "C-II-a", "C-II-b", "C-III-real"}
    assert res["decision"]["outcome"] in ("GAP", "NULL", "INVALID")
    for k in ("R0_seed0", "R0_seed1", "R1_seed0_descriptive", "R0_seed0_exclusion_deleted", "PLANTED_kappa4_R0_seed0", "PLANTED_kappa4_R0_seed0_exclusion_deleted", "PLANTED_kappa0.5_R0_seed0"):
        assert k in res["arms"] and "gaps" in res["arms"][k]
    assert res["constants_sha256"] == S0.constants_sha256()
    # run_all hands the C-I certificate to the decision: on this tiny noisy world M3's shuffle control is not informative at the real delta, so M3 is uncertified in the verdict as well
    assert res["controls"]["C-I-eval"]["informative"]["M3"] is False and res["decision"]["per_measure"]["M3"]["certified"] is False
    assert res["decision"]["per_measure"]["M1"]["certified"] is True
    json.dumps(S0.jsonable(res))                                                                  # serialisable
    # the exclusion-deleted arm is the only place a leaky audit row may appear
    assert all(a["own_fold_tokens"] == 0 for a in res["arms"]["R0_seed0"]["fit_audit"])
    assert all(a["own_fold_tokens"] == -1 for a in S0.fit_all_unexcluded(w, d, S0.Recipe(steps=3), 0)["audit"])


def test_cli_refuses_without_seal_and_on_wrong_prereg_hash(tmp_path, monkeypatch):
    pre = tmp_path / "PREREG.md"
    pre.write_text("constants_sha256: " + S0.constants_sha256() + "\n", encoding="utf-8")
    out = tmp_path / "r.json"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    with pytest.raises(RuntimeError, match="sealed value"):
        S0.main(["run", "--prereg", str(pre), "--prereg-sha256", "0" * 64, "--out", str(out), "--artifacts", str(tmp_path)])
    pre2 = tmp_path / "PREREG2.md"
    pre2.write_text("no digest here\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="constants_sha256"):
        S0.main(["run", "--prereg", str(pre2), "--prereg-sha256", S0.sha256_file(str(pre2)), "--out", str(out), "--artifacts", str(tmp_path)])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(SystemExit):
        S0.main(["run", "--prereg", str(pre), "--prereg-sha256", S0.sha256_file(str(pre)), "--out", str(out), "--artifacts", str(tmp_path)])
    assert not out.exists()


def test_cli_passes_the_seal_then_stops_at_the_input_sha_gate(tmp_path, monkeypatch):
    synth.make_files(str(tmp_path / "art"), seed=13, D=16, frac=0.2)
    pre = tmp_path / "PREREG.md"
    pre.write_text("constants_sha256: " + S0.constants_sha256() + "\n", encoding="utf-8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    with pytest.raises(RuntimeError, match="sha256/size does not match"):
        S0.main(["run", "--prereg", str(pre), "--prereg-sha256", S0.sha256_file(str(pre)), "--out", str(tmp_path / "r.json"), "--artifacts", str(tmp_path / "art")])


# ====================================================================================================================== (G) planted grid, determinism, loader against the real layout
def test_planted_grid_passes_if_any_kappa_is_read_and_RED_without_a_plant():
    def pair(kappa, seed):
        w = synth.make_world(seed=seed, kappa=kappa, D=32, frac=0.3)
        d = S0.make_design(w)
        idx = S0.boot_indices(w.S)
        return S0.run_arm(w, d, S0.R0, 0, idx), S0.run_arm(w, d, S0.R0, 0, idx, fit=S0.fit_all_unexcluded(w, d, S0.R0, 0))
    ok = S0.control_planted_grid({1.0: pair(1.0, 21), 2.0: pair(2.0, 3)})
    assert ok["pass"] and 2.0 in ok["kappas_passing"] and ok["smallest_kappa_read"] in (1.0, 2.0)
    red = S0.control_planted_grid({1.0: pair(0.0, 23), 4.0: pair(0.0, 24)})
    assert not red["pass"] and red["smallest_kappa_read"] is None


def test_the_pipeline_is_bit_for_bit_deterministic(null_world):
    w, d, idx = null_world
    a = S0.run_arm(w, d, S0.Recipe(steps=40), 0, idx)
    b = S0.run_arm(w, d, S0.Recipe(steps=40), 0, idx)
    assert S0.jsonable(a["gaps"]) == S0.jsonable(b["gaps"]) and S0.jsonable(a["swap_null"]) == S0.jsonable(b["swap_null"])


def test_load_world_sets_step_a_precision_and_pair_ok(tmp_path):
    w0 = synth.make_files(str(tmp_path), seed=14, D=16, frac=0.2)
    w, _ = S0.load_world(str(tmp_path), verify_sha=False)
    assert np.array_equal(w.X[w.sa_row], w0.X[w0.sa_row])               # Step A rows come from the float32 Step A features, not the fp16 table
    assert w.labels.pair_ok.all() and w.sa_of_row[w.sa_row].tolist() == list(range(w.Ns))


# ---- the Step A pair machinery: pairing, anchors and the sign of the pairwise term (found as SURVIVING mutants by mutation_census.py)
def _stepA_world(seed=0, D=32, orthogonal=False, s_const=None):
    """a world whose Step A perturbations are exactly constructed: perturbation q of token i = its anchor's features + s[i,q] * v, and its true sub-scores = the anchor's + 0.2 * s[i,q] on rows 0-4.
    orthogonal=False: v = sum of P2's rows, so P2 scores every pair correctly (A3 concordance 1.0).  orthogonal=True: v is orthogonal to every P2 row, so P2 ties every pair."""
    w = synth.make_world(seed=seed, kappa=0.0, D=D, frac=0.3)
    rng = np.random.default_rng(seed + 5)
    if orthogonal:
        Qm, _ = np.linalg.qr(w.W6.T)
        v = rng.standard_normal(D)
        v -= Qm @ (Qm.T @ v)
        v *= 4.0 / np.linalg.norm(v)
    else:
        v = w.W6[list(S0.ROWS)].sum(0)
        v *= 2.0 / np.linalg.norm(v)
        assert all(w.W6[r] @ v > 0 for r in S0.ROWS)
    s = rng.choice([-1.0, 1.0], size=(w.Ns, w.Q)) if s_const is None else np.full((w.Ns, w.Q), float(s_const))
    a_feat = np.take_along_axis(w.X[w.sa_row], w.anchor[..., None], 1)
    Xp = (a_feat + s[..., None] * v).astype(np.float32)
    L = w.labels
    a_sub = np.take_along_axis(L.sub[w.sa_row], w.anchor[..., None], 1)
    p_sub = a_sub.copy()
    p_sub[..., list(S0.ROWS)] = np.clip(a_sub[..., list(S0.ROWS)] + 0.2 * s[..., None], 0.0, 1.0)
    wv = np.asarray(S0.V1_W)
    p_pdms = p_sub[..., 0] * p_sub[..., 1] * p_sub[..., 5] * (wv[0] * p_sub[..., 2] + wv[1] * p_sub[..., 3] + wv[2] * p_sub[..., 4]) / wv.sum()
    lab = S0.Labels(L.sub, L.pdms, L.valid, p_sub, p_pdms, np.ones_like(L.pair_ok))
    w2 = S0.World(w.token, w.log, w.X, lab, w.sa_row, Xp, w.anchor, w.W6, w.b6, dict(w.meta))
    return w2, s


def _a3_concordance(w, d, models):
    cache = S0.build_cache(w, d, models, "search", roles=(0,))
    rd = S0.measure(w, d, cache)
    return rd.m2_num[0].sum() / rd.m2_den.sum(), rd.m2_den.sum()


def test_a3_pairing_and_anchors_are_exact():
    w, _ = _stepA_world(seed=2)
    d = S0.make_design(w)
    c, n = _a3_concordance(w, d, S0.p2_models(w))
    assert n > 200 and c == pytest.approx(1.0, abs=1e-12)                 # P2 ranks every constructed pair correctly: only a correct anchor / perturbation alignment reads 1.0


def test_pairwise_term_has_the_right_sign_and_learns_what_P2_cannot():
    w, s = _stepA_world(seed=3, orthogonal=True)
    d = S0.make_design(w)
    c0, _ = _a3_concordance(w, d, S0.p2_models(w))
    assert abs(c0 - 0.5) < 0.15                                           # P2 is blind to the perturbation direction
    tr = S0.train_tokens_for_model(w, d, 0)
    m = S0.fit_model(w, tr, S0.Recipe(steps=400, lr=1e-2, lam_bce=0.0), 0)    # the pairwise term only
    models = [(m["W"], m["b"])] * S0.N_FOLDS
    c1, _ = _a3_concordance(w, d, models)
    assert c1 > 0.9                                                       # it learned the direction, with the right sign, and it generalises to the search-half tokens


def test_draft_amendment_quotes_this_codes_constants_digest():
    """the amendment draft (or, once the MM seals it, the amended prereg) must quote the digest of the constants frozen in this file; a constant edit (S included) without an amendment edit is caught here"""
    found = [os.path.join(HERE, "..", n) for n in ("PREREG_STAGE0_A1.md", "DRAFT_AMENDMENT_A1.md") if os.path.exists(os.path.join(HERE, "..", n))]
    if not found:
        pytest.skip("no amendment text next to the code (mutation-census copy)")
    txt = open(found[0], encoding="utf-8").read()
    assert S0.constants_sha256() in txt, os.path.basename(found[0]) + " does not quote constants_sha256 " + S0.constants_sha256()



# ====================================================================================================================== (H) AMENDMENT A1: seed-averaged arms
# GOLDEN digests: produced by `golden_digests.py` run against the SEALED, UNAMENDED stage0.py (sha256 71a9ccf5f7e369e8926d5de95b180ca816995d4473479004bf8c7d8e2161f5c0), before any line of A1 existed
# (raw/golden_unamended.json).  S = 1 must reproduce them bit for bit: the statistics AND the raw bytes of every cached composite score.
GOLDEN_UNAMENDED = {
    "null_seed0": {"stats": "d30f85014f75f5c11b1dc739857472b44c58bc13715f230788f3ee031ee46805", "scores": "b70235083efa7032567daafa9726a208bc3b854ce925293aded2b3ada4f54c2b"},
    "null_seed1": {"stats": "f67678221c010f6bbefa11141f4400d7fd578938a1b143ceffa879da94c5426b", "scores": "50f19aad94af1b10b28755875b81d812cc36cb5e102ae0cf04454a36d852f819"},
    "planted_seed0": {"stats": "df0821ffbea47352f40660635fe42a3bf7a7d6e583ab5214e31b69c8bd934304", "scores": "fa47c65be7a32c9bd95adb678607a3499eafba6144257f385e5acc78882d1f43"},
    "planted_seed1": {"stats": "ebd3ffc169736b1cab09df9d921c465f5b71d77c5bd9dc02c1eb6863ae654edb", "scores": "5dc854b96801f7cb137d7753bf275a4a0e39befaa4ff3e4bbacb7e8d62061198"},
    "planted_exclusion_deleted_seed0": {"stats": "9184637f1acb9cc7e4013aee5d5c9bc0fae73aed3cafc395bf6a1f4e3a7de828", "scores": "29a61fc91ae7d89dcf7d241f52ee7f667871de2cfadcbd41b08c0688ebdb9173"},
    "planted_shuffle_trained_2worlds": "f389535e407d5b499b93db9254fbb79328c513dcb4660797d6d03bcc86daa40f",
}


def test_A1_S1_reproduces_the_unamended_pipeline_bit_for_bit(monkeypatch):
    monkeypatch.setattr(S0, "N_SEEDS", 1)
    assert GD.digests(S0, synth) == GOLDEN_UNAMENDED                                         # the integer-seed API: arms, the exclusion-deleted arm, the trained-on-shuffled-labels control


def test_A1_S1_tuple_seed_form_is_the_same_pipeline(monkeypatch):
    monkeypatch.setattr(S0, "N_SEEDS", 1)
    w = synth.make_world(seed=4, kappa=0.0, D=32, frac=0.3)
    d = S0.make_design(w)
    idx = S0.boot_indices(w.S)
    rec = S0.Recipe(steps=40)
    prim, repl = S0.seed_sets()
    assert prim == (0,) and repl == (100,)
    assert GD._arm_digest(S0, S0.run_arm(w, d, rec, prim, idx)) == GOLDEN_UNAMENDED["null_seed0"]                  # the primary arm of an S = 1 run is the unamended seed-0 arm
    assert GD._arm_digest(S0, S0.run_arm(w, d, rec, (1,), idx)) == GOLDEN_UNAMENDED["null_seed1"]                  # a one-seed tuple is the unamended seed, whatever the seed
    assert GD._arm_digest(S0, S0.run_arm(w, d, rec, repl, idx)) != GOLDEN_UNAMENDED["null_seed1"]                  # the S = 1 replicate is seed 100 (a different draw of the SAME pipeline), not the old seed 1
    m = S0.run_arm(w, d, rec, prim, idx, fit=S0.fit_all_unexcluded(w, d, rec, prim))
    assert GD._arm_digest(S0, m) != GOLDEN_UNAMENDED["null_seed0"]                                                 # the exclusion-deleted twin really is a different fit


def test_A1_a_fold_model_scores_the_mean_of_its_members_composites_not_the_composite_of_averaged_weights(planted):
    w, d, _ = planted
    fa = S0.fit_all_seeds(w, d, S0.Recipe(steps=25), (0, 1, 2))
    assert len(fa["models"]) == S0.N_FOLDS and all(len(m) == 3 for m in fa["models"])
    assert len(fa["audit"]) == 3 * S0.N_FOLDS and sorted({a["seed"] for a in fa["audit"]}) == [0, 1, 2] and all(a["own_fold_tokens"] == 0 for a in fa["audit"])
    assert not np.array_equal(fa["models"][0][0][0], fa["models"][0][1][0])                    # the seed reaches the minibatch stream: members differ
    cache = S0.build_cache(w, d, fa["models"], "search")
    sep = 0.0
    for g in range(S0.N_FOLDS):
        for k in range(S0.N_FOLDS):
            mem = fa["models"][(g + k) % S0.N_FOLDS]
            X = w.X[cache.idx[g]].astype(np.float64)
            ref = sum(S0.aggregate(X @ W.T + b) for W, b in mem) / 3.0                          # hand-rolled: mean over the members of each member's composite
            assert np.allclose(cache.agg[(g, k)], ref, rtol=0, atol=1e-13)
            Wm, bm = np.mean([W for W, _ in mem], 0), np.mean([b for _, b in mem], 0)
            sep = max(sep, float(np.abs(cache.agg[(g, k)] - S0.aggregate(X @ Wm.T + bm)).max()))
            sl = cache.sa_loc[g]
            if len(sl):                                                                         # Step A anchors and perturbations are averaged the same way
                s_idx = w.sa_of_row[cache.idx[g][sl]]
                refp = sum(S0.aggregate(w.Xp[s_idx].astype(np.float64) @ W.T + b) for W, b in mem) / 3.0
                assert np.allclose(cache.agg_p[(g, k)], refp, rtol=0, atol=1e-13)
                assert np.allclose(cache.agg_a[(g, k)], np.take_along_axis(ref[sl], w.anchor[s_idx], 1), rtol=0, atol=1e-13)
    assert sep > 1e-5                                                                           # the two readings genuinely differ here, so the test discriminates 'composite mean' from 'weight mean'


def test_A1_one_member_is_untouched_and_equal_members_average_to_themselves(planted):
    w, d, idx = planted
    X = w.X[:50].astype(np.float64)
    W, b = w.W6, w.b6
    assert np.array_equal(S0.ens_score(X, [(W, b)]), S0.aggregate(X @ W.T + b))                 # bit for bit
    assert np.allclose(S0.ens_score(X, [(W, b)] * 4), S0.aggregate(X @ W.T + b), rtol=0, atol=1e-15)
    a = S0.run_arm(w, d, S0.R0, 0, idx, fit={"models": [[(W.copy(), b.copy())] * 3 for _ in range(S0.N_FOLDS)], "audit": [], "recipe": None, "seed": None})
    for m in ("M1", "M2", "M3"):
        assert a["gaps"][m]["point"] == 0.0                                                       # the instrument identity survives ensembles of P2 heads


def test_A1_replicate_seeds_are_disjoint_by_construction_and_fail_closed(monkeypatch):
    for n in (1, 2, 3, 5, 8, 50, 100):
        p, r = S0.seed_sets(n)
        assert len(p) == len(r) == n and set(p).isdisjoint(r) and p == tuple(range(n)) and r == tuple(range(100, 100 + n))
    monkeypatch.setattr(S0, "N_SEEDS", 3)
    monkeypatch.setattr(S0, "REPL_SEED_BASE", 2)                                                  # replicate {2,3,4} overlaps primary {0,1,2}
    with pytest.raises(RuntimeError, match="overlap"):
        S0.seed_sets()
    with pytest.raises(RuntimeError, match="overlap"):
        S0.constants_dict()
    with pytest.raises(ValueError):
        S0.seed_sets(0)
    with pytest.raises(ValueError):
        S0._as_seeds((1, 1))


def test_A1_every_fit_of_the_primary_and_the_replicate_gets_its_own_seed(planted, monkeypatch):
    w, d, idx = planted
    monkeypatch.setattr(S0, "N_SEEDS", 3)
    prim, repl = S0.seed_sets()
    seen = []
    orig = S0.fit_model

    def spy(w_, tr, rec, seed, *a, **k):
        seen.append(int(seed))
        return orig(w_, tr, rec, seed, *a, **k)
    monkeypatch.setattr(S0, "fit_model", spy)
    rec = S0.Recipe(steps=3)
    S0.run_arm(w, d, rec, prim, idx)
    n0 = len(seen)
    S0.run_arm(w, d, rec, repl, idx)
    p_seen, r_seen = seen[:n0], seen[n0:]
    want = lambda ss: sorted(s * 1_000_003 + 7919 * h + 17 for s in ss for h in range(S0.N_FOLDS))
    assert sorted(p_seen) == want(prim) and sorted(r_seen) == want(repl) and len(p_seen) == len(r_seen) == 3 * S0.N_FOLDS
    assert set(p_seen).isdisjoint(r_seen)                                                          # not one effective RNG seed is shared between the primary and the replicate


def test_A1_the_exclusion_deleted_arm_is_averaged_too(planted):
    w, d, _ = planted
    fa = S0.fit_all_unexcluded(w, d, S0.Recipe(steps=3), (0, 1))
    assert len(fa["audit"]) == 2 * S0.N_FOLDS and sorted({a["seed"] for a in fa["audit"]}) == [0, 1] and all(a["own_fold_tokens"] == -1 for a in fa["audit"])
    assert all(len(m) == 2 for m in fa["models"])


def test_A1_run_all_averages_every_role_at_S2(monkeypatch):
    """OOF, IN, the swap-null models (all one cache), the replicate, the descriptive arm, the exclusion-deleted arm, the planted arms and the trained-on-shuffled-labels control: ALL are S-seed averages"""
    monkeypatch.setattr(S0, "N_SEEDS", 2)
    monkeypatch.setattr(S0, "R0", S0.Recipe(steps=12, prox=0.0))
    monkeypatch.setattr(S0, "R1", S0.Recipe(steps=12, prox=0.3))
    w = synth.make_world(seed=6, kappa=1.5, D=16, frac=0.25)
    d = S0.make_design(w)
    calls, fm = [], []
    orig_all, orig_fm = S0.fit_all_seeds, S0.fit_model
    monkeypatch.setattr(S0, "fit_all_seeds", lambda w_, d_, rec, seeds, *a, **k: (calls.append(S0._as_seeds(seeds)), orig_all(w_, d_, rec, seeds, *a, **k))[1])
    monkeypatch.setattr(S0, "fit_model", lambda w_, tr, rec, seed, *a, **k: (fm.append(int(seed)), orig_fm(w_, tr, rec, seed, *a, **k))[1])
    res = S0.run_all(w, d, None, log=lambda *_: None)
    # fit_all_seeds called by the main module: R0 primary, replicate, R1, 10 shuffled-label worlds, 4 planted = 17 arms; the 5 exclusion-deleted arms run inside the mutated module copy (audited below)
    assert len(calls) == 17 and calls.count((0, 1)) == 16 and calls.count((100, 101)) == 1
    assert len(fm) == 17 * 2 * S0.N_FOLDS
    assert res["n_seeds"] == 2 and res["primary_seeds"] == [0, 1] and res["replicate_seeds"] == [100, 101]
    arms = res["arms"]
    for name, seeds in (("R0_seed0", {0, 1}), ("R0_seed1", {100, 101}), ("R1_seed0_descriptive", {0, 1}), ("R0_seed0_exclusion_deleted", {0, 1}),
                        *[(f"PLANTED_kappa{k:g}_R0_seed0", {0, 1}) for k in S0.PLANT_KAPPAS], *[(f"PLANTED_kappa{k:g}_R0_seed0_exclusion_deleted", {0, 1}) for k in S0.PLANT_KAPPAS]):
        au = arms[name]["fit_audit"]
        assert len(au) == 2 * S0.N_FOLDS and {a["seed"] for a in au} == seeds, name
        assert all(a["own_fold_tokens"] == (-1 if "exclusion_deleted" in name else 0) for a in au), name
    json.dumps(S0.jsonable(res))


def test_A1_shuffled_label_training_control_averages_its_heads(planted, monkeypatch):
    w, d, idx = planted
    monkeypatch.setattr(S0, "N_SEEDS", 2)
    seen, orig = [], S0.fit_model
    monkeypatch.setattr(S0, "fit_model", lambda w_, tr, rec, seed, *a, **k: (seen.append(int(seed)), orig(w_, tr, rec, seed, *a, **k))[1])
    S0.control_shuffle_world(w, d, S0.Recipe(steps=3), idx, seeds=(9101,))
    assert sorted(seen) == sorted(s * 1_000_003 + 7919 * h + 17 for s in (0, 1) for h in range(S0.N_FOLDS))


# ---- the calibration script (calib_a1.py)
def _fake_control(kappa, passed, gap=0.03, lo=0.02, nu=0.01):
    return {"M1_gap": gap, "M1_lo_bonf": lo, "swap_null_M1": nu, "min_required": 0.02, "planted_gap_read": passed, "seen_minus_oof_planted": 0.02, "seen_beats_oof_planted": True,
            "collapse": {"pass": True, "M1": {"gap_mutated": 0.001}}, "pass": passed}


def _planted_recs(passing):
    return [{"kappa": k, "control": _fake_control(k, k in passing), "seconds": 1.0} for k in S0.PLANT_KAPPAS]


def _null_rec(i, gap=False, invalid=False):
    prim = _prim() if gap else {"M1": G(0.0, se=0.002), "M2": G(0.0, se=0.01), "M3": G(0.0, se=0.1)}
    return {"i": i, "label_seed": CA.NULL_SEEDS[i], "primary": prim, "replicate": prim, "swap": SWAP, "controls": {"C-X": {"pass": not invalid}}, "certified": {"M3": True}, "seconds": 1.0,
            "seconds_parts": {"a0": 0.1, "a1": 0.1, "am": 0.1, "c_shuf": 0.1, "c_shuf_world": 0.1, "labels": 0.1}}


def _summ(S, c2b=(2.0,), fp=0, inv=0, n_null=24):
    nulls = [_null_rec(i, gap=(i < fp), invalid=(fp <= i < fp + inv)) for i in range(n_null)]
    return CA.summarise_S(S, _planted_recs(c2b), nulls)


def test_A1_selection_rule_picks_the_smallest_qualifying_S_and_respects_every_bound():
    ok = {S: _summ(S) for S in (3, 5, 8)}
    assert CA.apply_selection_rule(ok)["chosen_S"] == 3
    only_big_kappa = {3: _summ(3, c2b=(4.0,)), 5: _summ(5, c2b=(0.5, 4.0)), 8: _summ(8)}
    assert CA.apply_selection_rule(only_big_kappa)["chosen_S"] == 5                              # kappa = 4 alone does not count: condition (a) is kappa <= 2
    fp = {3: _summ(3, fp=3), 5: _summ(5, fp=2), 8: _summ(8)}
    assert fp[3]["null"]["fp_n"] == 3 and fp[5]["null"]["fp_n"] == 2 and CA.apply_selection_rule(fp)["chosen_S"] == 5      # 2 of 24 passes, 3 of 24 fails
    inv = {3: _summ(3, inv=5), 5: _summ(5, inv=4), 8: _summ(8)}
    assert inv[3]["null"]["invalid_wo_c2b_n"] == 5 and CA.apply_selection_rule(inv)["chosen_S"] == 5                    # 4 of 24 passes, 5 of 24 fails
    none = {3: _summ(3, c2b=()), 5: _summ(5, fp=3), 8: _summ(8, inv=5)}
    r = CA.apply_selection_rule(none)
    assert r["chosen_S"] is None and r["verdict"] == "INSTRUMENT-LIMITED"
    s1 = {1: _summ(1), 3: _summ(3, fp=3), 5: _summ(5, fp=3), 8: _summ(8, fp=3)}
    assert CA.apply_selection_rule(s1)["chosen_S"] is None                                       # S = 1 is the baseline and is never selectable
    incomplete = {3: _summ(3, n_null=23), 5: _summ(5)}
    assert CA.apply_selection_rule(incomplete)["chosen_S"] == 5                                  # an incomplete calibration cannot qualify


def test_A1_a_failed_c2b_is_one_event_not_24_invalid_worlds():
    s = CA.summarise_S(3, _planted_recs(()), [_null_rec(i) for i in range(24)])
    assert s["c2b"]["pass_any"] is False
    assert s["null"]["invalid_with_c2b_n"] == 24 and s["null"]["invalid_wo_c2b_n"] == 0 and s["null"]["fp_n"] == 0


def test_A1_false_positive_is_counted_with_c2b_taken_as_passed():
    s = CA.summarise_S(3, _planted_recs(()), [_null_rec(i, gap=True) for i in range(5)] + [_null_rec(i) for i in range(5, 24)])
    assert s["null"]["fp_n"] == 5 and s["null"]["outcomes_with_c2b"].get("INVALID") == 24            # FP is not hidden by a failed C-II-b


def test_A1_identity_check_reads_the_sealed_excerpt_and_goes_red_on_any_difference():
    if not os.path.exists(CA.SEALED_EXCERPT):
        pytest.skip("sealed excerpt not next to the code")
    sealed = json.load(open(CA.SEALED_EXCERPT))["controls"]["C-II-b"]["per_kappa"]
    recs = [{"kappa": float(k), "control": copy.deepcopy(v)} for k, v in sealed.items()]
    ok = CA.identity_check(recs)
    assert ok["pass"] and ok["bit_identical"] and ok["max_abs_diff"] == 0.0
    bad = copy.deepcopy(recs)
    bad[1]["control"]["M1_gap"] += 1e-6
    r = CA.identity_check(bad)
    assert not r["pass"] and not r["bit_identical"] and r["max_abs_diff"] > 5e-7
    flag = copy.deepcopy(recs)
    flag[0]["control"]["planted_gap_read"] = not flag[0]["control"]["planted_gap_read"]
    assert not CA.identity_check(flag)["pass"]
    assert not CA.identity_check(recs[:3])["pass"]                                                   # a missing kappa is not an identity


def test_A1_calibration_world_is_label_blind_by_construction(tmp_path):
    synth.make_files(str(tmp_path), seed=15, kappa=1.0, D=16, frac=0.2)
    w, _ = CA.blind_world(str(tmp_path), verify_sha=False)
    CA.assert_blind(w)
    assert w.meta["label_blind"] and np.all(w.labels.pdms == 1.0) and np.all(w.labels.p_sub == 1.0)
    real = S0.load_world(str(tmp_path), verify_sha=False)[0]                                           # the loader's own world DOES carry the (here synthetic) labels ...
    assert not np.all(real.labels.pdms == 1.0)
    with pytest.raises(RuntimeError, match="A1-BLIND"):                                                # ... and the guard refuses it
        CA.assert_blind(real.with_labels(real.labels))
    lab = w.with_labels(S0.plant_labels(w, 0.0, seed=CA.NULL_SEEDS[0]))                                # planted / null labels come from the features alone
    assert not np.all(lab.labels.pdms == 1.0) and np.array_equal(lab.X, w.X)


def test_A1_calibration_decision_parts_equal_run_all_on_the_same_world(monkeypatch):
    """the calibration must be the real pipeline: decision_parts + the planted join give run_all's decision and run_all's planted verdicts, number for number"""
    monkeypatch.setattr(S0, "N_SEEDS", 2)
    monkeypatch.setattr(S0, "R0", S0.Recipe(steps=12, prox=0.0))
    monkeypatch.setattr(S0, "R1", S0.Recipe(steps=12, prox=0.3))
    w = synth.make_world(seed=7, kappa=1.5, D=16, frac=0.25)
    d = S0.make_design(w)
    idx = S0.boot_indices(w.S)
    res = S0.run_all(w, d, None, log=lambda *_: None)
    prim, repl = S0.seed_sets()
    parts = CA.decision_parts(w, d, idx, prim, repl)
    c2b = res["controls"]["C-II-b"]["pass"]
    dec = S0.decide(parts["primary"], parts["replicate"], parts["swap"], {**parts["controls"], "C-II-b": {"pass": c2b}}, certified=parts["certified"])
    assert S0.jsonable(dec) == S0.jsonable(res["decision"])
    for k in (0.5, 4.0):
        pp = CA.planted_parts(w, d, idx, prim, k)
        assert S0.jsonable(pp["control"]) == S0.jsonable(res["controls"]["C-II-b"]["per_kappa"][f"{k:g}"])
    assert set(parts["seconds_parts"]) == {"a0", "a1", "am", "c_shuf", "c_shuf_world"}


def test_A1_calibration_worlds_use_the_committed_generators(monkeypatch):
    """planted worlds: kappa from the grid, generator seed 7771; null worlds: kappa = 0, label seeds 8001..8024; S is set per task"""
    monkeypatch.setattr(S0, "R0", S0.Recipe(steps=3, prox=0.0))
    monkeypatch.setattr(S0, "SHUF_SEEDS", S0.SHUF_SEEDS[:3])
    monkeypatch.setattr(S0, "SHUF_WORLD_SEEDS", S0.SHUF_WORLD_SEEDS[:1])
    w = synth.make_world(seed=8, kappa=0.0, D=16, frac=0.2)
    d = S0.make_design(w)
    calls, orig = [], S0.plant_labels
    monkeypatch.setattr(S0, "plant_labels", lambda w_, kappa=S0.PLANT_KAPPAS[-1], seed=S0.PLANT_SEED, *a, **k: (calls.append((float(kappa), int(seed))), orig(w_, kappa, seed, *a, **k))[1])
    monkeypatch.setattr(S0, "N_SEEDS", 1)
    r = CA.run_planted(w, d, 2, 1.0)
    assert calls == [(1.0, 7771)] and S0.N_SEEDS == 2 and r["S"] == 2 and r["kappa"] == 1.0
    calls.clear()
    r = CA.run_null(w, d, 3, 5)
    assert calls == [(0.0, 8006)] and S0.N_SEEDS == 3 and r["S"] == 3 and r["label_seed"] == 8006
    assert CA.NULL_SEEDS[0] == 8001 and CA.NULL_SEEDS[-1] == 8024 and len(CA.NULL_SEEDS) == 24 and CA.KAPPAS == (0.5, 1.0, 2.0, 4.0) and CA.S_LIST == (1, 3, 5, 8)
    assert len(r["controls"]) == 5 and "C-II-b" not in r["controls"]                                    # C-II-b is joined once per S by the master, never recomputed per null world


def test_A1_lazy_shuffles_are_control_shuffle_with_the_label_sets_streamed(planted_arms, planted, relaxed_delta):
    w, d, _ = planted
    a, _ = planted_arms
    seeds = S0.SHUF_SEEDS[:6]
    assert S0.jsonable(S0.control_shuffle(w, d, a["cache"], seeds=seeds)) == S0.jsonable(S0.chance_check(w, d, CA.LazyShuffles(w, a["cache"], seeds)))
    assert len(CA.LazyShuffles(w, a["cache"])) == len(S0.SHUF_SEEDS) == 100


def test_A1_amendment_draft_states_the_committed_selection_rule():
    p = os.path.join(HERE, "..", "DRAFT_AMENDMENT_A1.md")
    if not os.path.exists(p):
        pytest.skip("no DRAFT_AMENDMENT_A1.md next to the code (mutation-census copy)")
    txt = open(p, encoding="utf-8").read()
    for needle in ("smallest S", f"{CA.FP_MAX}/{len(CA.NULL_SEEDS)}", f"{CA.INVALID_MAX}/{len(CA.NULL_SEEDS)}", "INSTRUMENT-LIMITED", "kappa <= 2", "8001", "8024", "7771"):
        assert needle in txt, needle
