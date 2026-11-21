"""Unit tests of s0c_read.py on SYNTHETIC data only (no real token, no real label)."""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pytest

import s0c_read as C
import s0_read as S0
RP = C.RP

D_TOK, NC, T, N_EP = 64, 40, 40, 20
W_OSC = 0.157                              # fixed angular rate: z(g+10) is an exact linear function of z(g) and z(g-5)


def make_synth(seed=0):
    """20 episodes x 40 contiguous frames. Cell 7 carries a latent z(g) = A sin(w g + phi); gap = 20 + z; ego = noise. d = z(g+10) - z(g)."""
    rng = np.random.default_rng(seed)
    beid, bg, z = [], [], []
    for e in range(N_EP):
        A, ph = 2.0 + rng.random(), rng.random() * 6.28
        for g in range(T):
            beid.append(f"e{e}"); bg.append(g); z.append(A * np.sin(W_OSC * g + ph))
    beid, bg, z = np.array(beid), np.array(bg, np.int64), np.array(z)
    n = len(beid)
    X = (0.3 * rng.standard_normal((n, NC, D_TOK))).astype(np.float32)
    u = rng.standard_normal(D_TOK); u /= np.linalg.norm(u)
    X[:, 7, :] += (z[:, None] * u[None, :]).astype(np.float32)
    uep = sorted(set(beid))
    D = {"X": X, "beid": beid, "bg": bg, "E": rng.standard_normal((n, 8)), "gap": 20.0 + z, "pres": np.ones(n),
         "uep": uep, "epi": np.array([uep.index(e) for e in beid]), "z": z}
    return D


def setup(D, label="main"):
    ri, rm, rk, pc = C.build_pairs(D["beid"], D["bg"], D["gap"])
    d = D["gap"][rk] - D["gap"][ri]
    Pr = {"ri": ri, "rm": rm, "rk": rk, "Eb": C.ego_block(D["E"], ri, rm, rk), "ep": D["epi"][ri]}
    folds_b = RP.make_folds(D["beid"])
    return Pr, d, folds_b, pc


@pytest.fixture(scope="module")
def run_all():
    D = make_synth()
    Pr, d, folds_b, pc = setup(D)
    M = S0.mean_token(D["X"])
    spec = C.k2_setup(D)
    side = C.k2_side(len(Pr["ri"]), spec["seed_side"])
    oof, _ = C.compute_oof(D, M, Pr, {"main": d}, folds_b, spec, side, False, log_=lambda *a: None)
    return {"D": D, "Pr": Pr, "d": d, "oof": oof, "side": side, "spec": spec}


# ------------------------------------------------------------------------------------------------ skill and bootstrap
def test_skill_persistence_zero_and_perfect_one():
    rng = np.random.default_rng(0)
    d = rng.standard_normal(200) * 3
    assert C.skill(d, np.zeros(200)) == 0.0
    assert C.skill(d, d.copy()) == 1.0
    assert C.skill(d, 3 * d) == pytest.approx(-3.0)            # a wrong-sized predictor is worse than persistence


def test_skill_bootstrap_matches_point_value():
    rng = np.random.default_rng(1)
    d = rng.standard_normal(300); p = 0.5 * d
    ep = np.arange(300) % 10
    Cb = RP.boot_counts(10, 50, 0)
    r = C.single_skill(ep, d, p, Cb, 10)
    assert r["skill"] == pytest.approx(C.skill(d, p))
    assert r["ci95"][0] <= r["skill"] <= r["ci95"][1] or abs(r["skill"] - r["ci95"][0]) < 0.2
    one = np.ones((1, 10))                                      # a draw with every episode once equals the point value
    S = C.ep_stats_skill(ep, d, p, 10)
    assert C.boot_skill(S, one)[0] == pytest.approx(C.skill(d, p))


def test_build_pairs_needs_three_rows_and_finite_gaps(monkeypatch):
    monkeypatch.setattr(C, "PAIR_RULE", "all")              # the stage-0c rule
    beid = np.array(["a"] * 6 + ["b"] * 3)
    bg = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2], np.int64)
    gap = np.arange(9, dtype=float)
    ri, rm, rk, pc = C.build_pairs(beid, bg, gap, fwd=2, back=1)
    assert [(int(bg[i]), int(bg[m]), int(bg[k])) for i, m, k in zip(ri, rm, rk)] == [(1, 0, 3), (2, 1, 4), (3, 2, 5)]
    gap2 = gap.copy(); gap2[3] = np.nan
    ri2, _, _, pc2 = C.build_pairs(beid, bg, gap2, fwd=2, back=1)
    assert len(ri2) == 1 and pc2["pairs_with_three_rows"] == 3


# ------------------------------------------------------------------------------------------------ k1
def test_k1_fires_on_a_leak():
    rng = np.random.default_rng(2)
    n = 400
    ep = np.arange(n) % 20
    d_perm = rng.standard_normal(n)
    folds = ep % 5
    clean = rng.standard_normal((n, 6))
    leak = np.concatenate([clean, d_perm[:, None]], 1)         # a feature that carries the permuted target
    def oof(F):
        o = np.full(n, np.nan)
        for f in range(5):
            trp, tep = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            o[tep] = C.arm_oof_fold(F, d_perm, ep, trp, tep, f)
        return o
    ok = C.check_k1({"clean": C.skill(d_perm, oof(clean))})
    bad = C.check_k1({"leak": C.skill(d_perm, oof(leak))})
    assert ok["pass"] and not bad["pass"] and bad["max_skill"] > 0.5


# ------------------------------------------------------------------------------------------------ k2 and the arms
def test_k2_moving_object_read_from_two_grid_frames_only(run_all):
    o, side = run_all["oof"], run_all["side"]
    a_g2, a_m2, a_g1 = RP.auc_pooled(side, o["k2|grid2"]), RP.auc_pooled(side, o["k2|mean2"]), RP.auc_pooled(side, o["k2|grid1"])
    assert a_g2 > 0.95, a_g2
    assert a_m2 <= 0.60, a_m2
    assert a_g1 <= 0.60, a_g1
    assert C.check_k2(a_g2, a_m2, a_g1)["pass"]
    assert not C.check_k2(0.9, a_m2, a_g1)["pass"] and not C.check_k2(a_g2, 0.8, a_g1)["pass"]


def test_k2_cell_is_interior_and_vector_has_the_median_norm(run_all):
    s = run_all["spec"]
    assert 1 <= s["cell"] % 10 <= 8
    assert np.linalg.norm(s["vec"]) == pytest.approx(s["median_token_norm"], rel=1e-3)


def test_linear_change_target_recovered_by_the_arm_that_holds_its_cause(run_all):
    d, o, Pr = run_all["d"], run_all["oof"], run_all["Pr"]
    sk = {a: C.skill(d, o[f"arm|main|{a}"]) for a in C.ARMS}
    assert sk["H-g"] > 0.8, sk
    assert sk["E"] < 0.1, sk                                    # the ego floor holds no cause of d
    assert sk["N-g"] < sk["H-g"] - 0.2, sk                      # one frame cannot tell the phase: history adds
    Cb = RP.boot_counts(len(run_all["D"]["uep"]), 200, 0)
    n_ep = len(run_all["D"]["uep"])
    res = C.analyse_main(d, Pr["ep"], o, Cb, n_ep)
    assert res["outcomes"]["state_g"] == "S0-CHANGE-READ"
    assert res["outcomes"]["history"] == "HISTORY-ADDS"


def test_outcome_names_branches():
    sk = {"E": {"skill": 0.05, "ci95": [0.01, 0.09]}}
    sep = {"diff": 0.1, "ci95": [0.02, 0.2], "separated_pos": True}
    non = {"diff": 0.1, "ci95": [-0.02, 0.2], "separated_pos": False}
    dif = {"H-g minus E": non, "H-m minus E": non, "H-g minus H-m": non, "H-g minus N-g": sep}
    o = C.outcome_names(sk, dif)
    assert o["state_g"] == "S0-CHANGE-EGO" and o["pair"] == "GRID-EQUAL-CHANGE" and o["history"] == "HISTORY-ADDS"
    sk0 = {"E": {"skill": 0.05, "ci95": [-0.01, 0.09]}}
    assert C.outcome_names(sk0, dif)["state_m"] == "S0-CHANGE-FLAT"
    dif2 = {**dif, "H-g minus E": sep, "H-g minus H-m": sep}
    o2 = C.outcome_names(sk0, dif2)
    assert o2["state_g"] == "S0-CHANGE-READ" and o2["pair"] == "GRID-ADDS-CHANGE"


# ------------------------------------------------------------------------------------------------ k3
def test_k3_alignment_exact_and_catches_a_shift(run_all):
    D, Pr = run_all["D"], run_all["Pr"]
    ri, rm, rk = Pr["ri"], Pr["rm"], Pr["rk"]
    args = (D["beid"], D["bg"], D["E"], D["gap"], D["beid"], D["bg"])
    ok = C.check_k3(*args, ri, rm, rk, Pr["Eb"], D["gap"][ri], D["gap"][rk])
    assert ok["pass"] and ok["bad_index"] == 0 and ok["bad_ego"] == 0 and ok["bad_gap"] == 0
    bad = C.check_k3(*args, ri, rm, np.minimum(rk + 1, len(D["gap"]) - 1), Pr["Eb"], D["gap"][ri], D["gap"][rk])
    assert not bad["pass"]
    Eb2 = Pr["Eb"].copy(); Eb2[3, 5] += 1e-12                   # exact means exact
    assert not C.check_k3(*args, ri, rm, rk, Eb2, D["gap"][ri], D["gap"][rk])["pass"]


def test_k4_reference_rounding():
    assert C.check_k4(0.39811, 0.13269)["pass"]
    assert not C.check_k4(0.3985, 0.1327)["pass"]


# ------------------------------------------------------------------------------------------------ the PCA guard and its mutation
def guard_ok():
    """The states of the training rows do not change when the test rows are altered: the PCA saw the training rows only (S0 R5)."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, NC, D_TOK)).astype(np.float32)
    tr = np.arange(150)
    M = S0.mean_token(X)
    a_g, a_m = S0.grid_state(X, tr)[:150], S0.mean_state(M, tr)[:150]
    X2 = X.copy(); X2[150:] *= 50.0
    M2 = S0.mean_token(X2)
    b_g, b_m = S0.grid_state(X2, tr)[:150], S0.mean_state(M2, tr)[:150]
    return bool(np.allclose(a_g, b_g, atol=1e-5) and np.allclose(a_m, b_m, atol=1e-5))


def test_pca_guard_passes_and_the_mutation_turns_it_red(monkeypatch):
    assert guard_ok()
    monkeypatch.setattr(S0, "LEAK_PCA_FIT", True)
    assert not guard_ok()


def test_k2_fit_rows_follow_the_leak_switch(monkeypatch):
    """With the switch on, k2_fold fits its bases on all pairs: its fit selection becomes every pair (mutation visible in the selection itself)."""
    rng = np.random.default_rng(9)
    n = 120
    X = rng.standard_normal((n, NC, D_TOK)).astype(np.float32)
    ri, rm = np.arange(5, 115), np.arange(0, 110)
    trp, tep = np.arange(0, 60), np.arange(60, 110)
    side = (rng.random(len(ri)) < 0.5).astype(np.float64)
    spec = {"cell": 12, "vec": (rng.standard_normal(D_TOK) * 8).astype(np.float32)}
    ep = np.arange(len(ri)) % 10
    rows = np.arange(n)
    M = S0.mean_token(X)
    St = {"g": S0.grid_state(X, rows[:70]), "m": S0.mean_state(M, rows[:70])}
    a = C.k2_fold(X, M, St, trp, tep, ri, rm, spec, side, ep, 0)["grid2"]
    monkeypatch.setattr(S0, "LEAK_PCA_FIT", True)
    St2 = {"g": S0.grid_state(X, rows[:70]), "m": S0.mean_state(M, rows[:70])}
    b = C.k2_fold(X, M, St2, trp, tep, ri, rm, spec, side, ep, 0)["grid2"]
    assert not np.allclose(a, b)


# ------------------------------------------------------------------------------------------------ note N6: the continuity rule of stage 0d
def test_build_pairs_continuity_rule(monkeypatch):
    monkeypatch.setattr(C, "PAIR_RULE", "continuous")
    beid = np.array(["a"] * 8)
    bg = np.arange(8, dtype=np.int64)
    gap = np.array([10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4])
    ri, rm, rk, pc = C.build_pairs(beid, bg, gap, fwd=3, back=1)
    assert [int(bg[i]) for i in ri] == [1, 2, 3, 4] and pc["pairs_removed_by_the_continuity_rule"] == 0
    g2 = gap.copy(); g2[3] = np.nan                         # no lead on a frame in between
    ri2, _, _, pc2 = C.build_pairs(beid, bg, g2, fwd=3, back=1)
    assert [int(bg[i]) for i in ri2] == [4] and pc2["pairs_removed_by_the_continuity_rule"] == 2
    g3 = gap.copy(); g3[5:] += 9.0                          # the label moves to another vehicle between frames 4 and 5
    ri3, _, _, pc3 = C.build_pairs(beid, bg, g3, fwd=3, back=1)
    assert [int(bg[i]) for i in ri3] == [1] and pc3["pairs_removed_by_the_continuity_rule"] == 3
    monkeypatch.setattr(C, "PAIR_RULE", "all")              # mutation: without the rule the broken pairs come back
    ri4, _, _, _ = C.build_pairs(beid, bg, g3, fwd=3, back=1)
    assert len(ri4) == 4

