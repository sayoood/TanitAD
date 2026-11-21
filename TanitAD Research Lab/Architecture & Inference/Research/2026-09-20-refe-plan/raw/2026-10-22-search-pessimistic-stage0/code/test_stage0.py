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
    for name, val in (("SPLIT_SEED", 1), ("BOOT_B", 5), ("REPL_FRAC", 0.9), ("Z_MAX", 5.0), ("PLANT_KAPPAS", (1.0,)), ("COLLAPSE_FRAC", 0.5), ("SEEN_MARGIN", 0.1), ("PLANT_MIN_M1", 0.1)):
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


def test_draft_prereg_quotes_this_codes_constants_digest():
    """the prereg draft (if it sits next to the code) must quote the digest of the constants that are frozen in this file; a constant edit without a prereg edit is caught here"""
    p = os.path.join(HERE, "..", "DRAFT_PREREG_STAGE0.md")
    if not os.path.exists(p):
        pytest.skip("no DRAFT_PREREG_STAGE0.md next to the code (mutation-census copy)")
    txt = open(p, encoding="utf-8").read()
    assert S0.constants_sha256() in txt, "DRAFT_PREREG_STAGE0.md does not quote constants_sha256 " + S0.constants_sha256()
