"""test_trigger_probe3.py -- unit tests of the controls of trigger probe 3 (PREREG_FROZEN.md 6).  Run:
    CUDA_VISIBLE_DEVICES=-1  python -I -m pytest -q -p no:cacheprovider test_trigger_probe3.py
Synthetic data only, except the three tests that read banked probe-1 inputs (marked)."""
from __future__ import annotations

import hashlib
import os
import sys
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
warnings.filterwarnings("ignore")
import numpy as np
import pytest

import g_lib as G
import run_g

L = G.L


# ------------------------------------------------------------------------------------------------ C4: G4 == mean2x2(G8) exactly
def test_c4_g4_equals_loop_reference_bit_for_bit():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 160, 40)).astype(np.float16).astype(np.float32)          # fp16-exact values, like the stored G8
    g4 = G.g4_from_g8(x)
    assert g4.shape == (5, 40, 40) and g4.dtype == np.float32
    for i in range(5):
        assert np.abs(G.g4_reference(x[i]) - g4[i]).max() == 0.0                           # EXACT


def test_c4_g4_is_the_2x2_mean_and_token_order_is_row_major():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 160, 8)).astype(np.float32)
    g4 = G.g4_from_g8(x)
    g = x.reshape(2, 8, 20, 8)
    for r in (0, 3):
        for c in (0, 7, 9):
            want = g[:, 2 * r:2 * r + 2, 2 * c:2 * c + 2].mean((1, 2))
            np.testing.assert_allclose(g4[:, r * 10 + c], want, rtol=1e-6, atol=1e-7)


def test_c4_mutation_of_the_pool_breaks_it():
    """the check has teeth: a wrong pooling (3 of the 4 cells) makes the bit-for-bit comparison fail."""
    x = np.random.default_rng(2).standard_normal((1, 160, 8)).astype(np.float32)
    g = x.reshape(1, 8, 20, 8)
    wrong = ((g[:, 0::2, 0::2] + g[:, 0::2, 1::2] + g[:, 1::2, 0::2]) / 3).reshape(1, 40, 8)
    assert np.abs(G.g4_reference(x[0]) - wrong[0]).max() > 1e-3


# ------------------------------------------------------------------------------------------------ the readout pieces
def test_token_stats_pca_equals_direct_pca():
    rng = np.random.default_rng(3)
    D, k = 48, 5
    A = rng.standard_normal((D, D)).astype(np.float32)
    tok = (rng.standard_normal((300, 16, D)).astype(np.float32) @ A) + 3.0                 # large mean: tests the pilot centring
    outer = rng.integers(0, 5, 300)
    st = G.TokenStats(D=D, nblk=5)
    for b in range(5):
        st.add(b, tok[outer == b].reshape(-1, D))
    for f in (0, 3, 5):
        rows = np.flatnonzero(outer != f) if f < 5 else np.arange(300)
        a = st.pca([b for b in range(5) if b != f], k)
        b_ = G.fit_token_pca(tok, rows, k=k)
        assert a["n"] == b_["n"]
        np.testing.assert_allclose(a["mu"], b_["mu"], atol=2e-4)
        np.testing.assert_allclose(a["ev"], b_["ev"], rtol=1e-3)
        Pa, Pb = a["comp"].T @ a["comp"], b_["comp"].T @ b_["comp"]                          # compare the subspaces (signs / rotations inside the top-k are free)
        np.testing.assert_allclose(Pa, Pb, atol=2e-3)


def test_project_tokens_matches_direct_and_is_token_major():
    rng = np.random.default_rng(4)
    D, k = 24, 4
    x = rng.standard_normal((3, 160, D)).astype(np.float32) + 2.0
    pc = [{"mu": rng.standard_normal(D).astype(np.float32), "comp": rng.standard_normal((k, D)).astype(np.float32)} for _ in range(2)]
    out = G.project_tokens(x, pc)
    assert out.shape == (2, 3, 160 * k)
    for i, p in enumerate(pc):
        want = (x - p["mu"]) @ p["comp"].T                                                    # [3, 160, k]
        np.testing.assert_allclose(out[i].reshape(3, 160, k), want, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out[i][:, 5 * k + 2], want[:, 5, 2], rtol=1e-4, atol=1e-4)  # token t, component j at column k*t + j


# ------------------------------------------------------------------------------------------------ C7: the probe-1 `if src == "V"` trap
@pytest.mark.parametrize("src", ["G4", "G8", "anything_else"])
def test_c7_controls_are_built_for_every_source_name(src):
    names = G.control_arm_names(src)
    assert len(names) == 3 and all(src in n for n in names)
    assert {n.split(":")[0] for n in names} == {"C1", "C2", "C3"}


def test_c7_run_g_builds_c1_c2_c3_for_both_sources_and_the_mlp_c1():
    n = 60
    d = {"E": np.zeros((n, 8)), "y": np.arange(n) % 2, "clip": np.arange(n) // 3}
    S = {"G4": np.zeros((6, n, 640), np.float32), "G8": np.zeros((6, n, 2560), np.float32)}
    lin = run_g.build_arms(d, S, "linear")
    for src in ("G4", "G8"):
        for ctl in G.control_arm_names(src):
            assert ctl in lin, ctl
    mlp = run_g.build_arms(d, S, "mlp")
    assert "C1:shuffle:G4+E" in mlp and "C1:shuffle:G8+E" in mlp
    assert list(lin)[:4] == ["G4+E", "G8+E", "G4", "G8"]                                        # priority order of PREREG 4
    # the mutation of the trap: a builder gated on the source name would drop them; assert the arm functions are genuinely different per source
    f4, f8 = lin["C3:leak:G4+E"][0](0), lin["C3:leak:G8+E"][0](0)
    assert f4.shape[1] == 640 + 8 + 1 and f8.shape[1] == 2560 + 8 + 1


# ------------------------------------------------------------------------------------------------ C5: reachability
def test_c5a_g8_entry_point_rejects_the_4x10_file():
    G.assert_g8_entry((12000, 163840))
    with pytest.raises(AssertionError):
        G.assert_g8_entry((12000, 40960))                                                       # probe 1's V_<pop>.npy
    with pytest.raises(AssertionError):
        G.assert_g8_entry((12000, 163840 // 4))


def test_c5b_upsample_mutation_carries_no_fine_information_and_changes_the_features():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 160, 16)).astype(np.float32)
    g4 = G.g4_from_g8(x)
    up = G.upsample_g4_to_g8(g4)
    assert up.shape == x.shape
    np.testing.assert_allclose(G.g4_from_g8(up), g4, rtol=1e-6, atol=1e-6)                    # G4 of the mutant == G4 of the original: no new information
    assert np.abs(up - x).max() > 0.1                                                          # but the 8x20 tokens differ from the real ones
    pc = [{"mu": np.zeros(16, np.float32), "comp": rng.standard_normal((4, 16)).astype(np.float32)}]
    assert hashlib.sha256(G.project_tokens(x, pc).tobytes()).hexdigest() != hashlib.sha256(G.project_tokens(up, pc).tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ C6: planted-resolution known answer, through the REAL library path
def _synthetic(n=1500, plant=True, seed=0, D=24, k=4):
    """160 tokens of dim D; the label is planted as +s on token (3,4) and -s on token (3,5) along u: the two tokens share one 2x2 cell, so the planted signal cancels EXACTLY in the 4x10 average."""
    rng = np.random.default_rng(seed)
    u = np.zeros(D, np.float32)
    u[0] = 1.0
    y = rng.integers(0, 2, n)
    x = rng.standard_normal((n, 160, D)).astype(np.float32)
    x[:, :, 0] *= 3.0                                                                          # u is a dominant direction of every token: the per-token PCA keeps it
    if plant:
        s = 3.0 * (2 * y - 1).astype(np.float32)
        x[:, 3 * 20 + 4] += s[:, None] * u
        x[:, 3 * 20 + 5] -= s[:, None] * u
    clip = np.arange(n) // 3
    sha = np.array([format(int(hashlib.sha1(str(c).encode()).hexdigest()[:12], 16), "012x") for c in clip])
    return x, y, clip, sha, k, D


def _pipeline_auc(x, y, clip, sha, k, D, grid):
    outer, inner = L.outer_fold(sha), L.inner_fold(sha)
    N = len(y)
    pos8, pos4 = G.sample_positions(N, 7, n_sub=16)
    p8, p4 = G.fit_readout(lambda s: x[s:s + 250], N, 250, outer, pos8, pos4, k=k, D=D)
    S8, S4 = np.zeros((6, N, 160 * k), np.float32), np.zeros((6, N, 40 * k), np.float32)
    G.transform_all(lambda s: x[s:s + 250], N, 250, p8, p4, S8, S4)
    S = S8 if grid == "G8" else S4
    E = np.zeros((N, 2))
    o, _ = L.cv_oof(lambda f: np.concatenate([S[f].astype(np.float64), E], 1), y, outer, inner, "linear")
    return o, outer


def test_c6_planted_resolution_g8_sees_it_g4_does_not_and_the_null_is_flat():
    x, y, clip, sha, k, D = _synthetic(plant=True)
    o8, outer = _pipeline_auc(x, y, clip, sha, k, D, "G8")
    o4, _ = _pipeline_auc(x, y, clip, sha, k, D, "G4")
    cnt = L.boot_counts(clip)
    a8, a4 = L.auc_ci_fold(o8, y, clip, outer, counts=cnt), L.auc_ci_fold(o4, y, clip, outer, counts=cnt)
    assert a8["auc"] > 0.9, a8
    assert a4["lo"] <= 0.5 <= a4["hi"], a4                                                     # the +s / -s cancels exactly in the 2x2 average
    dd = L.dauc_ci_fold(o8, o4, y, clip, outer, counts=cnt)
    assert dd["delta"] > 0.3 and dd["excludes_0"]
    # null: no planted signal -> the paired difference CI covers 0 and both AUCs cover 0.5
    xn, yn, cn, sn, k, D = _synthetic(plant=False, seed=1)
    n8, outer_n = _pipeline_auc(xn, yn, cn, sn, k, D, "G8")
    n4, _ = _pipeline_auc(xn, yn, cn, sn, k, D, "G4")
    cntn = L.boot_counts(cn)
    ddn = L.dauc_ci_fold(n8, n4, yn, cn, outer_n, counts=cntn)
    assert ddn["lo"] <= 0 <= ddn["hi"], ddn
    for o in (n8, n4):
        a = L.auc_ci_fold(o, yn, cn, outer_n, counts=cntn)
        assert a["lo"] <= 0.5 <= a["hi"], a


def test_c1_c3_on_the_pipeline_shuffle_is_flat_leak_is_one():
    x, y, clip, sha, k, D = _synthetic(plant=True, n=1200)
    outer, inner = L.outer_fold(sha), L.inner_fold(sha)
    N = len(y)
    pos8, pos4 = G.sample_positions(N, 7, n_sub=16)
    p8, p4 = G.fit_readout(lambda s: x[s:s + 250], N, 250, outer, pos8, pos4, k=k, D=D)
    S8, S4 = np.zeros((6, N, 160 * k), np.float32), np.zeros((6, N, 40 * k), np.float32)
    G.transform_all(lambda s: x[s:s + 250], N, 250, p8, p4, S8, S4)
    cnt = L.boot_counts(clip)
    ys = L.shuffle_labels(y, 1)
    o, _ = L.cv_oof(lambda f: S4[f].astype(np.float64), ys, outer, inner, "linear")
    a = L.auc_ci_fold(o, ys, clip, outer, counts=cnt)
    assert a["lo"] <= 0.5 <= a["hi"], a
    o, _ = L.cv_oof(lambda f: np.concatenate([S4[f].astype(np.float64), y[:, None].astype(float)], 1), y, outer, inner, "linear")
    assert L.auc_ci_fold(o, y, clip, outer, counts=cnt)["auc"] >= 0.99


# ------------------------------------------------------------------------------------------------ provenance / reuse (read banked probe-1 inputs)
def test_tp_lib_is_byte_identical_to_probe1():
    a = open(f"{G.HERE}/tp_lib.py", "rb").read()
    b = open(f"{G.OLD}/tp_lib.py", "rb").read()
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


def test_windows_and_samples_are_probe1s_and_row_order_matches_the_banked_oof():
    assert G.sha256_file(f"{G.OLDRAW}/windows_train.npz") == G.WINDOWS_SHA
    for pop in G.POPS:
        d = G.load_pop(pop)
        bk = G.banked(pop, "linear")
        assert len(d["y"]) == 12000 and np.array_equal(bk["y"], d["y"]) and np.array_equal(bk["clip"], d["clip"])


def test_c_grid_is_the_frozen_one_and_tp_lib_reads_it():
    assert G.L.C_GRID == (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 5))
    y = (X[:, 0] + rng.standard_normal(300) > 0).astype(int)
    pred, hp, tab = L.fit_probe(X, y, np.arange(300) % 3, "linear")
    assert set(tab) == set(G.L.C_GRID) and hp in G.L.C_GRID
