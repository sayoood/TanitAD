"""Tests for v8f_hacc.py (PREREG_V8F_HACC): the pre-registered CONTROLS as tests (on synthetic data, plus the two TRAIN-window machinery ties on the real banked arrays),
the optimiser's identities, and the pieces a MUTATION must turn RED (see mutate_hacc.py, which runs this file against textually mutated copies of the module).

Run:  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --basetemp C:/Users/Admin/pt_hacc/<name> test_v8f_hacc.py
HACC_MODULE_DIR (set only by the mutation script) points at the directory holding the module under test; default = this directory.
None of these tests reads an EVAL number or writes outside the pytest basetemp.
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pytest  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MOD_DIR = os.environ.get("HACC_MODULE_DIR", HERE)
sys.path.insert(0, MOD_DIR)
import v8f_hacc as A  # noqa: E402

H, S = A.H, A.S
assert os.path.abspath(os.path.dirname(A.__file__)) == os.path.abspath(MOD_DIR), (A.__file__, MOD_DIR)


# ------------------------------------------------------------------------------------------------ synthetic world
def smooth_plan(rng, N):
    v = 8.0 + np.cumsum(rng.normal(0, 0.15, (N, 30)), 1)
    x = np.cumsum(np.clip(v, 0.5, None) * 0.2, 1)
    yv = np.cumsum(np.cumsum(rng.normal(0, 0.01, (N, 30)), 1), 1) * 0.2
    return np.stack([x, yv], -1)


def synth(N=300, seed=0, noise=0.0, rough=0.4):
    """P (prior), tau = P + r (rough, zero-mean residual), y = P + wtrue (.) r + noise.  wtrue differs per axis, is NOT monotone, and touches both bounds."""
    rng = np.random.default_rng(seed)
    P = smooth_plan(rng, N)
    r = rough * rng.standard_normal((N, 30, 2)) + 0.8 * np.cumsum(rng.standard_normal((N, 30, 2)), 1) * 0.1
    t = np.arange(30)
    wtrue = np.stack([np.clip(0.55 + 0.45 * np.sin(t / 3.0), 0, 1), np.clip(0.25 + 0.3 * np.cos(t / 4.0), 0, 1)], 1)
    wtrue[3, 0], wtrue[4, 0], wtrue[9, 1], wtrue[10, 1] = 0.0, 1.0, 1.0, 0.0
    y = P + wtrue[None] * r + noise * rng.standard_normal((N, 30, 2))
    return {"P": P, "tau": P + r, "y": y, "r": r, "wtrue": wtrue}


@pytest.fixture(scope="module")
def syn():
    return synth()


@pytest.fixture(scope="module")
def syn_noisy():
    return synth(N=400, seed=3, noise=0.3)


def literal_residual(w, tau, P, y, lam, a):
    """The PREREG's residual vector for ONE axis by the LITERAL formula (no matrices): [ H_w - y  ;  sqrt(lam) (D2[0;H_w] - D2[0;y]) ], D2 by np.diff of the ORIGIN-PREPENDED paths."""
    Hw = P[:, :, a] + w[None] * (tau[:, :, a] - P[:, :, a])
    z = np.zeros((len(Hw), 1))
    d2h = np.diff(np.concatenate([z, Hw], 1), n=2, axis=1)                    # [N, 29]: p_(k+1) - 2 p_k + p_(k-1), k = 1..29, p_0 = origin
    d2y = np.diff(np.concatenate([z, y[:, :, a]], 1), n=2, axis=1)
    return np.concatenate([(Hw - y[:, :, a]).reshape(-1), np.sqrt(lam) * (d2h - d2y).reshape(-1)])


# ------------------------------------------------------------------------------------------------ D2 (the origin)
def test_d2_is_the_origin_prepended_second_difference():
    D2 = A.d2_matrix()
    assert D2.shape == (29, 30)
    lit = np.diff(np.concatenate([np.zeros((1, 30)), np.eye(30)], 0), n=2, axis=0)      # the literal second difference of (origin, e_1..e_30), as a matrix
    assert np.array_equal(D2, lit), "D2 must be the second difference with the ORIGIN prepended: 29 cells, row 0 = e_0 - 2 e_1 + e_2 with e_0 = 0"
    assert np.array_equal(D2, S.d2_matrix()), "must equal HSPEC's D2 (same 29 accel cells)"
    j = np.arange(1, 31, dtype=np.float64)
    assert np.allclose(D2 @ (3.0 * j), 0.0) and np.allclose(D2 @ (j ** 2), 2.0)       # row 0 included: 0 - 2 + 4 = 2
    assert D2[0, 0] == -2.0 and D2[0, 1] == 1.0 and D2[0].sum() == -1.0, "row 0 is the cell that touches the origin"


def test_terms_and_lambda0_match_the_literal_formula(syn_noisy):
    d = syn_noisy
    w = np.full((30, 2), 0.6)
    s1, s2 = A.terms_at(d["tau"], d["P"], d["y"], w)
    lit1 = sum(((d["P"][:, :, a] + w[None, :, a] * (d["tau"][:, :, a] - d["P"][:, :, a]) - d["y"][:, :, a]) ** 2).sum() for a in range(2))
    lit2 = 0.0
    for a in range(2):
        Hw = d["P"][:, :, a] + w[None, :, a] * (d["tau"][:, :, a] - d["P"][:, :, a])
        z = np.zeros((len(Hw), 1))
        lit2 += ((np.diff(np.concatenate([z, Hw], 1), n=2, axis=1) - np.diff(np.concatenate([z, d["y"][:, :, a]], 1), n=2, axis=1)) ** 2).sum()
    assert abs(s1 - lit1) <= 1e-9 * lit1 and abs(s2 - lit2) <= 1e-9 * lit2
    T = A.TrainBundle(tau=d["tau"], P=d["P"], y=d["y"], banked_w=w)
    assert abs(A.lambda0(T) - lit1 / lit2) <= 1e-9 * lit1 / lit2


# ------------------------------------------------------------------------------------------------ the fit
@pytest.mark.parametrize("lam", [0.0, 3.7, 400.0])
def test_a_planted_w_is_recovered_on_both_axes_including_both_bounds(syn, lam):
    f = A.fit_w(syn["tau"], syn["P"], syn["y"], lam)
    assert np.abs(f["w"] - syn["wtrue"]).max() <= 1e-9, np.abs(f["w"] - syn["wtrue"]).max()
    assert max(f["kkt"]) <= 1e-6
    assert f["w"].min() >= 0.0 and f["w"].max() <= 1.0


def test_the_axes_are_not_swapped(syn):
    """The two axes carry different planted w: an along/cross swap anywhere in the fit cannot recover them."""
    f = A.fit_w(syn["tau"], syn["P"], syn["y"], 5.0)
    assert np.abs(f["w"][:, 0] - syn["wtrue"][:, 0]).max() <= 1e-9 and np.abs(f["w"][:, 1] - syn["wtrue"][:, 1]).max() <= 1e-9
    assert np.abs(syn["wtrue"][:, 0] - syn["wtrue"][:, 1]).max() > 0.3
    g = A.fit_w(syn["tau"][:, :, ::-1], syn["P"][:, :, ::-1], syn["y"][:, :, ::-1], 5.0)
    assert np.abs(g["w"][:, ::-1] - f["w"]).max() <= 1e-9, "flipping the data's axes must flip the fitted columns"


def test_lambda0_without_isotonic_is_the_clipped_closed_form_and_with_it_is_hblends(syn_noisy):
    """Controls 2 and 1 on synthetic data: lambda = 0 equals the clipped closed form (1e-12); adding HBLEND's isotonic step equals HBLEND's fit_weights w (1e-12)."""
    d = syn_noisy
    f = A.fit_w(d["tau"], d["P"], d["y"], 0.0)
    hb = H.fit_weights(d["tau"], d["P"], d["y"])
    assert np.abs(f["w"] - hb["clipped"]).max() <= 1e-12
    assert (np.diff(hb["clipped"], axis=0) < -1e-6).any(), "the clipped closed form must be NON-monotone here, or the isotonic step is not exercised"
    assert np.abs(A.isotonic(f["w"]) - hb["w"]).max() <= 1e-12
    assert np.abs(f["w"] - hb["w"]).max() > 1e-3, "the arm has NO isotonic step: w must differ from HBLEND's where the clipped form is non-monotone"


def test_the_fit_is_the_optimum_of_the_literal_objective(syn_noisy):
    """An independent solver (lsq_linear 'trf') on a system built from the LITERAL residual (np.diff of the origin-prepended paths, applied to unit vectors) reaches the same
    w: pins D2's origin, the pooling over windows and the lambda scaling, independently of d2_matrix / stack_system."""
    from scipy.optimize import lsq_linear
    d = {k: v[:40] for k, v in syn_noisy.items() if k != "wtrue"}
    for lam in (0.0, 12.0, 300.0):
        f = A.fit_w(d["tau"], d["P"], d["y"], lam)
        for a in range(2):
            r0 = literal_residual(np.zeros(30), d["tau"], d["P"], d["y"], lam, a)
            M = np.stack([literal_residual(np.eye(30)[i], d["tau"], d["P"], d["y"], lam, a) - r0 for i in range(30)], 1)
            res = lsq_linear(M, -r0, bounds=(0, 1), method="trf", tol=1e-14, max_iter=2000)
            cost = lambda w: float((literal_residual(w, d["tau"], d["P"], d["y"], lam, a) ** 2).sum())
            assert cost(f["w"][:, a]) <= cost(res.x) * (1 + 1e-9) + 1e-9, (lam, a, cost(f["w"][:, a]), cost(res.x))        # never worse than the independent optimum
            assert np.abs(res.x - f["w"][:, a]).max() <= 1e-5, (lam, a, np.abs(res.x - f["w"][:, a]).max())


def test_kkt_is_zero_at_the_optimum_and_the_check_raises_on_a_non_optimum(syn_noisy):
    d = syn_noisy
    D2 = A.d2_matrix()
    A_, b_ = A.stack_system(d["tau"][:, :, 0] - d["P"][:, :, 0], d["y"][:, :, 0] - d["P"][:, :, 0], 50.0, D2)
    f = A.fit_w(d["tau"], d["P"], d["y"], 50.0)
    assert A.kkt_rel(A_, b_, f["w"][:, 0]) <= 1e-9
    assert A.kkt_rel(A_, b_, np.clip(f["w"][:, 0] + 0.02, 0, 1)) > 1e-4

    def bad_solver(Am, bm, **kw):
        from scipy.optimize import lsq_linear
        res = lsq_linear(Am, bm, **kw)
        res.x = np.clip(res.x + 0.02, 0.0, 1.0)                                   # a solver that stops short / returns a wrong point
        return res

    with pytest.raises(A.KKTError):
        A.fit_w(d["tau"], d["P"], d["y"], 50.0, solver=bad_solver)


# ------------------------------------------------------------------------------------------------ endpoints and the deliberate regressions
def test_endpoints_w1_is_tau_w0_is_P_and_y_at_an_endpoint_pins_w(syn):
    d = syn
    assert np.abs(H.apply_w(d["P"], d["tau"], 1.0) - d["tau"]).max() <= 1e-12
    assert np.abs(H.apply_w(d["P"], d["tau"], 0.0) - d["P"]).max() == 0.0
    for lam in (0.0, 40.0):
        assert np.abs(A.fit_w(d["tau"], d["P"], d["tau"], lam)["w"] - 1.0).max() <= 1e-9                 # y := tau  =>  w == 1
        assert np.abs(A.fit_w(d["tau"], d["P"], d["P"], lam)["w"]).max() <= 1e-12                         # y := P    =>  w == 0


@pytest.mark.parametrize("lam", [0.0, 7.0, 900.0])
def test_regression_gt_equals_P10_collapses_w_to_zero_and_reads_as_P10(syn_noisy, lam):
    d = syn_noisy
    f = A.fit_w(d["tau"], d["P"], d["P"], lam)                                    # GT := P10 in the fit
    assert np.abs(f["w"]).max() <= 1e-12
    plan = H.apply_w(d["P"], d["tau"], f["w"])
    assert abs(np.linalg.norm(plan - d["y"], axis=-1).mean() - np.linalg.norm(d["P"] - d["y"], axis=-1).mean()) <= 1e-9, "must read as P10's ADE, never as a gain"
    g = A.fit_w(d["tau"], d["P"], d["y"], lam)
    assert g["w"].mean() > 0.2, "the real fit must NOT collapse (else the regression arm proves nothing)"


@pytest.mark.parametrize("lam", [0.0, 25.0])
def test_regression_r_permuted_across_windows_collapses_mean_w(lam):
    d = synth(N=3000, seed=5, noise=0.05)
    real = A.fit_w(d["tau"], d["P"], d["y"], lam)
    assert real["w"].mean() > 0.3
    perm = np.random.default_rng(0).permutation(3000)
    f = A.fit_w(d["P"] + (d["tau"] - d["P"])[perm], d["P"], d["y"], lam)
    assert f["w"].mean() <= 0.10, f["w"].mean()


# ------------------------------------------------------------------------------------------------ the lambda rule (control 6) and its branches
def _row(m, lam, accel, curv, ade2, ade6, p=(1.0, 1.0, 1.0, 1.0)):
    return {"m": m, "lam": lam, "accel": accel, "curv": curv, "ade2": ade2, "ade6": ade6, "p_accel": p[0], "p_curv": p[1], "p_ade2": p[2], "p_ade6": p[3]}


def test_choose_lambda_branches_and_the_098_and_strict_bars():
    ok = dict(curv=0.5, ade2=0.9, ade6=0.9)
    rows = [_row(0, 0.0, 0.995, **ok), _row(1, 1.0, 0.981, **ok), _row(4, 4.0, 0.980, **ok), _row(16, 16.0, 0.5, **ok)]
    r, br = A.choose_lambda(rows)
    assert r["m"] == 4 and br.startswith("qualified"), "0.981 x P10 fails the 0.98 bar; exactly 0.98 passes (<=); the SMALLEST qualifying lambda is chosen"
    assert A.choose_lambda(list(reversed(rows)))[0]["m"] == 4, "the rule sorts by lambda: input order is irrelevant"
    r, br = A.choose_lambda([_row(0, 0.0, 0.5, 0.5, 1.0, 0.9), _row(1, 1.0, 0.5, 0.5, 0.9, 1.0)])
    assert r["m"] == 1 and br.startswith("fallback 1"), "ADE2 == P10's does not qualify; ADE6 == P10's does not qualify -> fallback 1 (the only ADE2 < P10 row)"
    # fallback 1: nobody qualifies; among rows with ADE2 < P10's the lowest accel wins
    rows = [_row(0, 0.0, 0.9, 0.99, 0.95, 0.9), _row(1, 1.0, 0.7, 0.99, 0.97, 0.9), _row(4, 4.0, 0.6, 0.99, 1.0, 0.9), _row(16, 16.0, 0.8, 0.99, 0.5, 0.9)]
    r, br = A.choose_lambda(rows)
    assert r["m"] == 1 and br.startswith("fallback 1"), (r["m"], br)
    # fallback 2: nobody has ADE2 below P10's -> lambda = 0
    r, br = A.choose_lambda([_row(0, 0.0, 0.5, 0.5, 1.1, 0.9), _row(4, 4.0, 0.4, 0.4, 1.2, 0.9)])
    assert r["m"] == 0 and r["lam"] == 0.0 and br.startswith("fallback 2")
    # ADE2 strict: equal to P10's does not qualify even with accel/curv/ADE6 fine
    assert not A.qualifies(_row(1, 1.0, 0.5, 0.5, 1.0, 0.5))
    assert not A.qualifies(_row(1, 1.0, 0.5, 0.5, 0.5, 1.0))
    assert not A.qualifies(_row(1, 1.0, 0.5, 0.99, 0.5, 0.5)) and not A.qualifies(_row(1, 1.0, 0.99, 0.5, 0.5, 0.5))
    assert A.qualifies(_row(1, 1.0, 0.98, 0.98, 0.99, 0.99))


def fake_metrics(plan, y):
    """Light stand-in for arm_pw in the selection tests: the real accel error (the instrument), a roughness proxy for curvature, and ADE."""
    d = np.linalg.norm(np.asarray(plan, np.float64) - y, axis=-1)
    z = np.zeros((len(plan), 1, 2))
    rough = np.abs(np.diff(np.concatenate([z, plan], 1), n=2, axis=1) - np.diff(np.concatenate([z, y], 1), n=2, axis=1)).mean((1, 2))
    return {"ade2": d[:, :10].mean(1), "ade6": d.mean(1), "fw_lon_accel_mae": A.AP.window_mae(plan, y), "fw_lat_curvature_mae_1pm": rough}


def test_select_lambda_is_the_rule_applied_to_an_independent_recomputation(syn_noisy):
    d = syn_noisy
    bw = np.full((30, 2), 0.5)
    T = A.TrainBundle(tau=d["tau"], P=d["P"], y=d["y"], banked_w=bw)
    sel = A.select_lambda(T, metrics_fn=fake_metrics)
    s1, s2 = A.terms_at(d["tau"], d["P"], d["y"], bw)
    assert abs(sel["lambda0"] - s1 / s2) <= 1e-9 * s1 / s2
    assert [g["m"] for g in sel["grid"]] == [0, 1, 4, 16, 64, 256]
    pP = fake_metrics(d["P"], d["y"])
    expect = None
    for g in sel["grid"]:
        f = A.fit_w(d["tau"], d["P"], d["y"], g["m"] * sel["lambda0"])
        pa = fake_metrics(H.apply_w(d["P"], d["tau"], f["w"]), d["y"])
        assert np.abs(f["w"] - g["w"]).max() <= 1e-12
        ok = (pa["fw_lon_accel_mae"].mean() <= 0.98 * pP["fw_lon_accel_mae"].mean() and pa["fw_lat_curvature_mae_1pm"].mean() <= 0.98 * pP["fw_lat_curvature_mae_1pm"].mean()
              and pa["ade2"].mean() < pP["ade2"].mean() and pa["ade6"].mean() < pP["ade6"].mean())
        assert ok == g["qualifies"]
        if ok and expect is None:
            expect = g["m"]
    if expect is not None:
        assert sel["m_star"] == expect and sel["rule_branch"].startswith("qualified")
    assert sel["lam_star"] == sel["m_star"] * sel["lambda0"]


def test_the_lambda_selection_refuses_anything_that_is_not_train(syn):
    """Control 6: its code path receives training arrays alone; passing an eval array (a dict world, a bare array, or an eval-labelled bundle) must RAISE."""
    d = syn
    T = A.TrainBundle(tau=d["tau"], P=d["P"], y=d["y"], banked_w=np.full((30, 2), 0.5))
    A.select_lambda(T, metrics_fn=fake_metrics)                                        # a train bundle is accepted
    eval_world = {"tau": d["tau"], "P": d["P"], "gt": d["y"], "eid": np.arange(len(d["y"]))}
    for bad in (eval_world, d["tau"]):
        try:
            A.select_lambda(bad, metrics_fn=fake_metrics)
        except Exception as e:                                                         # noqa: BLE001
            assert isinstance(e, TypeError), f"an eval input must be refused with TypeError, got {type(e).__name__}: {e}"
        else:
            raise AssertionError("an eval input was accepted by the lambda selection")
    from types import SimpleNamespace
    ns_eval = SimpleNamespace(tau=d["tau"], P=d["P"], y=d["y"], banked_w=np.full((30, 2), 0.5), split="eval", source="eval")   # duck-typed like a bundle, but not one
    with pytest.raises((TypeError, ValueError)):
        A.select_lambda(ns_eval, metrics_fn=fake_metrics)
    ns_train_label = SimpleNamespace(tau=d["tau"], P=d["P"], y=d["y"], banked_w=np.full((30, 2), 0.5), split="train", source="x")  # right label, wrong type
    with pytest.raises(TypeError):
        A.select_lambda(ns_train_label, metrics_fn=fake_metrics)
    with pytest.raises(ValueError):
        A.TrainBundle(tau=d["tau"], P=d["P"], y=d["y"], banked_w=np.zeros((30, 2)), split="eval")
    for bad in (eval_world, ns_eval, ns_train_label):
        try:
            A.lambda0(bad)
        except Exception as e:                                                         # noqa: BLE001
            assert isinstance(e, (TypeError, ValueError)), f"lambda0 must refuse a non-train input, got {type(e).__name__}"
        else:
            raise AssertionError("lambda0 accepted a non-train input")
    relabelled = A.TrainBundle(tau=d["tau"], P=d["P"], y=d["y"], banked_w=np.zeros((30, 2)))
    object.__setattr__(relabelled, "split", "eval")                                    # an eval-labelled object that got past the constructor
    with pytest.raises(ValueError):
        A.select_lambda(relabelled, metrics_fn=fake_metrics)


def test_train_bundle_is_pinned_to_the_banked_train_files():
    for s in A.SEEDS:
        T = A.train_bundle(s)
        assert T.split == "train" and T.tau.shape == (A.N_TRAIN, 30, 2) and T.source.endswith(f"fit_windows_{s}.npz")
        assert H.sha256_file(T.source) == A.TRAIN_SHA[s]["fit"]
        assert A.lambda0(T) > 0


# ------------------------------------------------------------------------------------------------ controls 1 and 2 on the REAL train windows (machinery ties; no eval number)
@pytest.mark.parametrize("seed", ["M2b_s0", "M2b_s1"])
def test_real_train_windows_control1_and_control2_tie_to_hblend(seed):
    T = A.train_bundle(seed)
    f0 = A.fit_w(T.tau, T.P, T.y, 0.0)
    hb = H.fit_weights(T.tau, T.P, T.y)
    assert np.abs(f0["w"] - hb["clipped"]).max() <= 1e-12, "control 2"
    assert np.abs(A.isotonic(f0["w"]) - T.banked_w).max() <= 1e-12, "control 1"
    assert max(f0["kkt"]) <= 1e-6


# ------------------------------------------------------------------------------------------------ the accel instrument (control 7)
def test_accel_instrument_equals_the_family_in_float32():
    rng = np.random.default_rng(11)
    plan, gt = smooth_plan(rng, 200), smooth_plan(rng, 200)
    BE = A._be()
    ref = BE.family_windows(plan, gt, 0.2)["lon_accel_mae"]
    assert A.AP.control(plan, gt, ref) <= A.AP.CONTROL_BAR, A.AP.control(plan, gt, ref)
    bad = A.AP.window_mae(plan + 1e-3, gt)                                           # a perturbed plan must NOT tie
    assert np.abs(bad - ref).max() > A.AP.CONTROL_BAR
    pw = H.arm_pw(plan, gt, BE)                                                      # arm_pw runs on train-shaped arrays [N, 30, 2]
    assert all(np.isfinite(pw[k]).sum() > 0 for k in ("ade2", "ade6", "fw_lon_accel_mae", "fw_lat_curvature_mae_1pm")) and pw["ade2"].shape == (200,)


def test_accel_profile_bands_and_cells():
    rng = np.random.default_rng(2)
    gt = smooth_plan(rng, 50)
    plans = {"HACC": gt + 0.05 * rng.standard_normal(gt.shape), "P10": gt + 0.2 * rng.standard_normal(gt.shape)}
    prof = A.AP.profile(plans, gt)
    assert len(prof["cells"]["HACC"]) == 29 and prof["t_s"][10] == 2.2 and prof["t_s"][12] == 2.6
    b = prof["bands"]["2.2-2.6 s (cells 11-13)"]
    assert abs(b["HACC"] - np.mean(prof["cells"]["HACC"][10:13])) <= 1e-12 and abs(b["HACC_minus_P10"] - (b["HACC"] - b["P10"])) <= 1e-12


# ------------------------------------------------------------------------------------------------ the outcome table, in its order
def _ps(dom, keep, an, cn):
    return {"dom": dom, "keep": keep, "accel_sep_neg": an, "curv_sep_neg": cn}


@pytest.mark.parametrize("s0,s1,want", [
    (_ps(1, 1, 1, 1), _ps(1, 1, 1, 1), "DOMINATES P10"),
    (_ps(1, 1, 0, 0), _ps(1, 1, 0, 0), "DOMINATES P10"),                         # row 1 wins even if H-relative accel is not separated-negative
    (_ps(1, 0, 1, 1), _ps(1, 1, 1, 1), "SMOOTHER, NOT DOMINANT"),               # KEEP fails at one seed -> not row 1; smoother at both
    (_ps(0, 1, 1, 1), _ps(1, 1, 1, 1), "SMOOTHER, NOT DOMINANT"),               # DOM fails at one seed
    (_ps(0, 1, 1, 1), _ps(0, 1, 1, 1), "SMOOTHER, NOT DOMINANT"),
    (_ps(0, 1, 0, 0), _ps(0, 1, 0, 0), "NO SMOOTHING"),
    (_ps(0, 1, 0, 1), _ps(0, 1, 0, 1), "NO SMOOTHING"),                          # curvature alone does not rescue it: accel is the row-3 test
    (_ps(1, 0, 0, 0), _ps(1, 1, 0, 0), "NO SMOOTHING"),
    (_ps(0, 1, 1, 1), _ps(0, 1, 0, 1), "MIXED"),                                 # accel separated-negative at ONE seed only
    (_ps(0, 1, 1, 0), _ps(0, 1, 1, 1), "MIXED"),                                 # accel at both, curvature at one
    (_ps(0, 1, 1, 0), _ps(0, 1, 1, 0), "MIXED"),                                 # accel at both, curvature at neither
    (_ps(0, 0, 1, 1), _ps(0, 0, 0, 0), "MIXED"),
])
def test_outcome_partition_in_prereg_order(s0, s1, want):
    assert A.outcome_of({"M2b_s0": s0, "M2b_s1": s1}) == want
    assert A.NEXT[want]


def test_committed_next_steps_are_the_preregs_verbatim():
    assert A.NEXT["SMOOTHER, NOT DOMINANT"] == "OACC (box 2) runs as queued, and its HX read adds HACC's fit on OACC's checkpoint"
    assert A.NEXT["NO SMOOTHING"] == "the roughness is not a weighting artefact; OACC is the lever"
    assert A.NEXT["MIXED"] == "reported per seed; seeds before windows"
    assert A.NEXT["DOMINATES P10"].startswith("HACC becomes the deployed post-map: 30 x 2 buffers per model, zero training") and "D-V8F-T1KIN-1" in A.NEXT["DOMINATES P10"]


# ------------------------------------------------------------------------------------------------ the dominance reader
def _prow(mean, lo, hi, lower_is_better=True):
    sep = lo > 0 or hi < 0
    return {"mean": mean, "lo": lo, "hi": hi, "separated": sep, "better": bool(sep and (mean < 0 if lower_is_better else mean > 0)),
            "worse": bool(sep and (mean > 0 if lower_is_better else mean < 0))}


def _dom_inputs(accel=(-0.05, -0.08, -0.02), curv=(-0.002, -0.003, -0.001), worse_row=None):
    p = {k: _prow(-0.1, -0.2, -0.05) for k in list(S.DOM_BETTER.values()) + list(S.DOM_NOT_WORSE.values())}
    p["fw_lon_accel_mae"], p["fw_lat_curvature_mae_1pm"] = _prow(*accel), _prow(*curv)
    if worse_row:
        p[worse_row] = _prow(0.2, 0.1, 0.3)
    dkp = {"dk": {"headway_min_m": {"delta": 0.3, "lo": -0.1, "hi": 0.7, "separated": False}, "time_gap_min_s": {"delta": 0.1, "lo": 0.01, "hi": 0.2, "separated": True}},
           "lead_viol": {"at_risk": {"delta": -0.03, "lo": -0.05, "hi": -0.01, "separated": True}}}
    return p, _prow(-0.04, -0.07, -0.01), dkp


def test_dominance_reader_needs_all_four_separated_better_and_no_separated_worse():
    p, cv, dkp = _dom_inputs()
    assert A.dominance(p, cv, dkp)["pass"]
    p, cv, dkp = _dom_inputs(accel=(0.02, -0.01, 0.05))                                  # accel not separated
    r = A.dominance(p, cv, dkp)
    assert not r["pass"] and not r["all_better_ok"] and r["all_not_worse_ok"]
    p, cv, dkp = _dom_inputs(curv=(0.004, 0.001, 0.007))                                  # curvature separated-WORSE
    assert not A.dominance(p, cv, dkp)["pass"]
    p, cv, dkp = _dom_inputs(worse_row="fw_lat_cross_mae")                                # a not-worse row is separated-worse
    r = A.dominance(p, cv, dkp)
    assert not r["pass"] and r["all_better_ok"] and not r["all_not_worse_ok"]
    p, cv, dkp = _dom_inputs()
    dkp["dk"]["headway_min_m"] = {"delta": -0.5, "lo": -0.9, "hi": -0.1, "separated": True}      # less headway, separated: worse
    assert not A.dominance(p, cv, dkp)["pass"]
    p, cv, dkp = _dom_inputs()
    dkp["lead_viol"]["at_risk"] = {"delta": 0.04, "lo": 0.01, "hi": 0.07, "separated": True}     # more lead violations, separated: worse
    assert not A.dominance(p, cv, dkp)["pass"]


def test_keep_bar_is_half_the_exact_seed_floor():
    F = 0.12074
    assert A.keep_read({"mean": 0.0, "lo": -0.01, "hi": 0.0603}, F)["pass"]
    assert not A.keep_read({"mean": 0.0, "lo": -0.01, "hi": 0.0605}, F)["pass"]
    assert abs(A.keep_read({"mean": 0.0, "lo": 0.0, "hi": 0.0}, F)["bar_hi_le"] - 0.06037) <= 1e-9


def test_dominance_reader_reproduces_hspecs_banked_verdicts():
    """Machinery tie X1: from HSPEC's own banked files the reader returns HSPEC's PRIMARY-DOM / PRIMARY-KEEP verdicts and every per-row flag."""
    x = A.x1_hspec_tie()
    assert x["pass"], x["mismatches"][:5]
    assert x["checked"] >= 4 * 15


def test_smooth_flags_read_the_separated_negative_of_accel_and_curvature():
    pv = {"fw_lon_accel_mae": _prow(-0.07, -0.09, -0.06), "fw_lat_curvature_mae_1pm": _prow(-0.006, -0.009, -0.003)}
    sf = A.smooth_flags(pv)
    assert sf["accel_sep_neg"] and sf["curv_sep_neg"]
    pv["fw_lon_accel_mae"] = _prow(-0.07, -0.09, 0.01)
    sf = A.smooth_flags(pv)
    assert not sf["accel_sep_neg"] and sf["curv_sep_neg"], "each flag reads ITS OWN row"
    pv["fw_lon_accel_mae"] = _prow(0.07, 0.04, 0.09)
    sf = A.smooth_flags(pv)
    assert not sf["accel_sep_neg"] and sf["curv_sep_neg"]
    pv = {"fw_lon_accel_mae": _prow(-0.07, -0.09, -0.06), "fw_lat_curvature_mae_1pm": _prow(-0.006, -0.009, 0.001)}
    sf = A.smooth_flags(pv)
    assert sf["accel_sep_neg"] and not sf["curv_sep_neg"]
    pv["fw_lat_curvature_mae_1pm"] = _prow(0.006, 0.003, 0.009)
    sf = A.smooth_flags(pv)
    assert sf["accel_sep_neg"] and not sf["curv_sep_neg"]
