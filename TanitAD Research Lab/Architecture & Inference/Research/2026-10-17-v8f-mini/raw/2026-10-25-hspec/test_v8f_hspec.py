"""Tests for v8f_hspec.py (PREREG_V8F_HSPEC): the pre-registered CONTROLS as tests, the optimiser's identities, and MUTATIONS that must go RED.

Run:  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --basetemp <OUT>/pt_basetemp test_v8f_hspec.py
None of these tests writes outside OUT, reads a reading number, or touches git.
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pytest  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import v8f_hspec as S  # noqa: E402

H = S.H
SEEDS = S.SEEDS


@pytest.fixture(scope="module")
def train():
    return {s: S.train_arrays(s) for s in SEEDS}


@pytest.fixture(scope="module")
def fits(train):
    out = {}
    for s in SEEDS:
        T = train[s]
        out[s] = S.fit_v(T["c"], T["w"], T["d"], T["B30"], 0.0, S.d2_matrix())
    return out


@pytest.fixture(scope="module")
def worlds():
    return {s: S.load_world(s) for s in SEEDS}


@pytest.fixture(scope="module")
def ctl(train, fits, worlds):
    """run_controls on stage-1 fits (the controls the PREREG lists, 1-4), per seed."""
    res = {}
    for s in SEEDS:
        T, D2 = train[s], S.d2_matrix()
        perm = np.random.default_rng(0).permutation(len(T["c"]))
        F = {"recov_maxabs_train": T["recov_maxabs"], "kkt_rel": fits[s]["kkt"],
             "control3_regression_fit": {"v": S.fit_v(T["c"], T["w"], T["P"] - T["P"], T["B30"], 0.0, D2)["v"]},
             "control4_permuted_fit": {"v": S.fit_v(T["c"][perm], T["w"], T["d"], T["B30"], 0.0, D2)["v"]}}
        res[s] = S.run_controls(worlds[s], F)
    return res


# ------------------------------------------------------------------------------------------------ the basis and the grid (I2)
def test_basis_is_the_models_and_subsample_is_the_02s_knots():
    import torch
    from tanitad.models import v8f_mini as M
    for s in SEEDS:
        ck = torch.load(H.CKPT[s], map_location="cpu", weights_only=False)
        assert ck["cfg"]["nb"] == S.NB == 12
        assert np.abs(ck["model"]["basis"].numpy().astype(np.float64) - S.basis60()).max() <= 1e-5
    p30 = torch.randn(3, 30, 2)
    up = M.upsample_prior(p30)
    assert up.shape == (3, 60, 2)
    assert torch.equal(M.subsample_02(up), p30), "up(P)[1::2] must be the 0.2 s knots (I2)"
    assert S.basis30().shape == (30, 12) and np.array_equal(S.basis30(), S.basis60()[1::2])
    assert np.linalg.matrix_rank(S.basis30()) == 12 and np.linalg.cond(S.basis30()) < 1e3


# ------------------------------------------------------------------------------------------------ CONTROL 1
def test_control1_recovery_residual_train_and_eval(train, worlds):
    for s in SEEDS:
        assert train[s]["recov_maxabs"] <= 1e-4, (s, train[s]["recov_maxabs"])
        assert worlds[s]["recov_maxabs_eval"] <= 1e-4, (s, worlds[s]["recov_maxabs_eval"])


def test_control1_v_ones_equals_H(ctl):
    for s in SEEDS:
        c = ctl[s]
        assert c["c1_v1_vs_H_plan_maxabs_m"] <= 1e-4
        # every entry of evaluate()'s block matches except the HARD-threshold diagnostic traj.acts.zero_accel_frac (|a| < 1e-3), whose flips the plan perturbation explains
        assert c["c1_v1_vs_H_evaluate_block_n_mismatches_excl_zero_accel_threshold_diagnostic"] == 0, c["c1_v1_vs_H_evaluate_block_mismatches"]
        assert c["c1_v1_vs_H_evaluate_block_n_mismatches"] <= 1
        fl = c["c1_zero_accel_threshold_flips"]
        assert fl["explained_by_plan_perturbation"] and fl["all_flips_straddle_threshold"] and fl["n_flip_cells"] <= 20, fl
        assert c["pass"]["c1_recovery"] and c["pass"]["c1_v1_equals_H"]


# ------------------------------------------------------------------------------------------------ CONTROL 2
def test_control2_v_zeros_equals_P10(ctl, worlds):
    for s in SEEDS:
        c = ctl[s]
        assert c["c2_v0_vs_P_exact_maxabs_m"] == 0.0
        assert c["c2_v0_vs_P10_file_plan_maxabs_m"] <= 1e-4
        assert c["c2_v0_ade6_vs_floors_perwindow_maxabs_m"] <= 1e-5
        assert abs(c["c2_v0_ade6_mean"] - 2.4870) <= 1e-3
        assert c["c2_v0_vs_P10_evaluate_block_n_mismatches_tol1e-4"] == 0, c["c2_v0_vs_P10_evaluate_block_mismatches"]
        assert c["pass"]["c2_v0_equals_P10"]


# ------------------------------------------------------------------------------------------------ CONTROL 3: the deliberate regression
def test_control3_gt_replaced_by_P10_collapses_v_to_zero_and_reads_as_P10(train, ctl):
    D2 = S.d2_matrix()
    for s in SEEDS:
        T = train[s]
        lam = S.lam_rule(T["c"], T["w"], T["d"], T["B30"], D2)
        for lm in (0.0, lam):                                              # stage 1 and stage 2
            v = S.fit_v(T["c"], T["w"], T["P"] - T["P"], T["B30"], lm, D2)["v"]
            assert np.abs(v).max() <= 1e-12
        assert ctl[s]["c3_regression_v_maxabs"] <= 1e-12
        assert abs(ctl[s]["c3_regression_ade6_minus_P10_mean_m"]) <= 1e-4, "the regressed plan must read as P10, never as a gain"
        assert ctl[s]["pass"]["c3_regression_collapses"]


# ------------------------------------------------------------------------------------------------ CONTROL 4: the noise control
def test_control4_permuted_c_collapses_v(ctl):
    for s in SEEDS:
        assert ctl[s]["c4_noise_v_mean"] <= S.NOISE_MEAN_MAX and ctl[s]["c4_noise_v_max"] <= S.NOISE_MAX_MAX, (s, ctl[s]["c4_noise_v_mean"], ctl[s]["c4_noise_v_max"])
        assert ctl[s]["pass"]["c4_noise_collapses"]


def test_all_controls_pass(ctl):
    for s in SEEDS:
        assert ctl[s]["all_pass"], {k: v for k, v in ctl[s]["pass"].items() if not v}


# ------------------------------------------------------------------------------------------------ the optimiser
def test_planted_v_is_recovered_exactly_stage1_and_stage2(train):
    T, D2 = train["M2b_s0"], S.d2_matrix()
    v0 = np.random.default_rng(1).uniform(0.1, 0.9, (S.NB, 2))
    y = S.hs_plan(T["P"], T["c"], T["w"], v0, T["B30"])
    d = y - T["P"]
    for lm in (0.0, 3.7):
        f = S.fit_v(T["c"], T["w"], d, T["B30"], lm, D2)
        assert np.abs(f["v"] - v0).max() <= 1e-5, (lm, np.abs(f["v"] - v0).max())


def test_fit_is_the_bounded_optimum(train, fits):
    from scipy.optimize import lsq_linear
    for s in SEEDS:
        T = train[s]
        assert fits[s]["v"].min() >= 0.0 and fits[s]["v"].max() <= 1.0
        assert max(fits[s]["kkt"]) <= 1e-6, fits[s]["kkt"]
        for a in range(2):                                                 # an independent solver (trf) reaches the same cost
            A = S.design(T["c"], T["w"], T["B30"], a).reshape(-1, S.NB)
            t = T["d"][:, :, a].reshape(-1)
            r2 = lsq_linear(A, t, bounds=(0, 1), method="trf", tol=1e-12)
            c_bvls = 0.5 * float(((A @ fits[s]["v"][:, a] - t) ** 2).sum())
            assert abs(c_bvls - r2.cost) <= 1e-9 * max(1.0, r2.cost)


def test_lambda_rule_balances_the_two_terms_at_v_ones(train):
    D2 = S.d2_matrix()
    for s in SEEDS:
        T = train[s]
        lam = S.lam_rule(T["c"], T["w"], T["d"], T["B30"], D2)
        s1, s2 = S.terms_at(T["c"], T["w"], T["d"], T["B30"], np.ones((S.NB, 2)), D2)
        assert lam > 0 and abs(lam * s2 - s1) <= 1e-9 * s1


def test_d2_matrix_with_origin_prepended():
    D2 = S.d2_matrix()
    assert D2.shape == (29, 30)
    j = np.arange(1, 31, dtype=np.float64)
    assert np.allclose(D2 @ (3.0 * j), 0.0), "a straight, uniform path has zero second difference (row 0 included: e0 = 0)"
    assert np.allclose(D2 @ (j ** 2), 2.0), "quadratic -> constant 2 on every row (row 0: e0 - 2 e1 + e2 = 0 - 2 + 4)"


# ------------------------------------------------------------------------------------------------ MUTATIONS: each must turn a control RED
def test_MUTATION_wrong_subsample_phase_makes_control1_red(train):
    """Using basis rows 0::2 (tau = 0.1, 0.3, ...) instead of 1::2 must break the coefficient recovery (> 1e-4 m): the control has teeth."""
    T = train["M2b_s0"]
    bad = S.basis60()[0::2]
    _, rec = S.recover_coef(T["tau"] - T["P"], bad)
    resid = float(np.abs(rec - (T["tau"] - T["P"])).max())
    assert resid > 1e-2, resid
    assert not (resid <= 1e-4)                                              # i.e. the PREREG's control 1 would FAIL on this build


def test_MUTATION_swapped_axes_breaks_the_planted_recovery(train):
    """A fit that reads w with its axes swapped cannot recover a planted v: the planted-v test is sensitive to the (axis, step) wiring."""
    T, D2 = train["M2b_s0"], S.d2_matrix()
    v0 = np.random.default_rng(2).uniform(0.1, 0.9, (S.NB, 2))
    y = S.hs_plan(T["P"], T["c"], T["w"], v0, T["B30"])
    f = S.fit_v(T["c"], T["w"][:, ::-1], y - T["P"], T["B30"], 0.0, D2)
    assert np.abs(f["v"] - v0).max() > 1e-2


def test_MUTATION_gt_not_replaced_would_not_collapse(train):
    """The regression control only has meaning if the REAL fit does NOT collapse: with the real GT, v is far from 0."""
    T = train["M2b_s0"]
    f = S.fit_v(T["c"], T["w"], T["d"], T["B30"], 0.0, S.d2_matrix())
    assert f["v"].mean() > 0.3


def test_MUTATION_a_real_plan_difference_is_not_excused_by_the_zero_accel_clause(worlds):
    """The zero-accel exception must not swallow a real defect: a plan perturbed by 3e-3 m flips cells that the Lipschitz bound (150 x 3e-3 = 0.45) still explains,
    but the plan-difference bar (<= 1e-4 m) goes RED -- the control is carried by the plan bar and the other block entries, not by that clause."""
    W = worlds["M2b_s0"]
    HS1 = S.hs_plan(W["P"], W["c"], W["w"], np.ones((S.NB, 2)), W["B30"])
    bad = HS1 + 3e-3 * np.random.default_rng(3).standard_normal(HS1.shape)
    assert float(np.abs(bad - W["H"]).max()) > 1e-4
    res_b, _ = W["ev"].evaluate(bad, W["X"], W["ex"], 20, with_families=True)
    res_h, _ = W["ev"].evaluate(W["H"], W["X"], W["ex"], 20, with_families=True)
    other = [m for m in H.deep_diff(H.jround(res_b), H.jround(res_h), tol=1e-6) if not m.startswith("/traj/acts/zero_accel_frac")]
    assert len(other) > 0, "a 3e-3 m perturbation must show up in the evaluate() block beyond the threshold diagnostic"
