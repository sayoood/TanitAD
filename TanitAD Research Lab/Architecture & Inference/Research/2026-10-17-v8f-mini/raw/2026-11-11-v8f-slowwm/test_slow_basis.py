"""Planted-data tests of the SLOWWM basis builder (``slow_basis.py``) and the read hook (``slowwm_hook.py``), PREREG_V8F_SLOWWM control 3 (basis identities) and control 4 (the builder's inputs).
CPU only. Run with the patched tree on PYTHONPATH (``stack`` and ``stack/scripts``):
    PYTHONPATH="<tree>/stack;<tree>/taniteval" ZC_REPO=<tree> python -m pytest test_slow_basis.py -q --basetemp <dir>
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slow_basis as SB  # noqa: E402
import slowwm_hook as SH  # noqa: E402

D, K, PHI = 24, 6, 0.985


def planted(n_clips=150, T=101, seed=0, slow_var=0.02, mid_var=4.0, tiny_var=0.01, phi_mid=0.9):
    """Frames x_t = Q [slow AR(1) (phi 0.985, variance 0.02) ; mid AR(1) (phi 0.9, variance 4: PCA's directions) ; white (variance 0.01)] + a per-clip offset. The slow directions carry LITTLE variance, so PCA picks the mid ones
    (lag-1 s autocorrelation 0.9^5 = 0.59) and the temporal statistic picks the slow ones (0.985^5 = 0.93); the tiny-variance white directions are where a basis fit on NOISE statistics lands (a shuffled-time basis)."""
    g = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(g.standard_normal((D, D)))
    clips = []

    def ar1(n, phi, var):
        s = np.zeros((T, n))
        s[0] = g.standard_normal(n) * np.sqrt(var)
        for t in range(1, T):
            s[t] = phi * s[t - 1] + np.sqrt(1 - phi ** 2) * np.sqrt(var) * g.standard_normal(n)
        return s
    for _ in range(n_clips):
        z = np.concatenate([ar1(K, PHI, slow_var), ar1(6, phi_mid, mid_var), g.standard_normal((T, D - K - 6)) * np.sqrt(tiny_var)], 1)
        clips.append((z @ Q.T + 0.3 * g.standard_normal(D)).astype(np.float32))
    return clips, Q


def stats_of(clips, ok=None, F=91, lag=5):
    acc = SB.Acc(D, lag)
    for i, x in enumerate(clips):
        o = np.ones(len(x), bool) if ok is None else ok[i]
        acc.add_clip(x, SB.fit_mask(o, F))
    return acc.finalize()


@pytest.fixture(scope="module")
def data():
    clips, Q = planted()
    return clips, Q, stats_of(clips)


# ------------------------------------------------------------------------------------------------ streaming statistics
def test_streamed_moments_equal_the_direct_formulas(data):
    clips, _, st = data
    X = np.concatenate(clips).astype(np.float64)
    m = X.mean(0)
    C = np.cov(X.T, bias=True)
    A = np.concatenate([c[:-5] for c in clips]).astype(np.float64) - m
    B = np.concatenate([c[5:] for c in clips]).astype(np.float64) - m
    S = (A.T @ B) / len(A)
    S = 0.5 * (S + S.T)
    assert st["n"] == len(X) and st["n_pairs"] == len(A)
    assert np.abs(st["mean"] - m).max() < 1e-9 and np.abs(st["C"] - C).max() < 1e-9 and np.abs(st["S"] - S).max() < 1e-9


def test_fit_mask_and_window_count_follow_the_ok_flag():
    ok = np.ones(101, bool)
    assert SB.n_windows(ok, 91) == 11 and SB.fit_mask(ok, 91).all()
    ok[50] = False                                                   # frame 50 sits inside every window t in [90, 100] -> no window left
    assert SB.n_windows(ok, 91) == 0 and not SB.fit_mask(ok, 91).any()
    ok = np.ones(101, bool)
    ok[0] = False                                                    # only t = 90 holds frame 0
    assert SB.n_windows(ok, 91) == 10
    m = SB.fit_mask(ok, 91)
    assert not m[0] and m[1:].all()
    assert SB.n_windows(np.ones(80, bool), 91) == 0                  # a clip shorter than F has no window


# ------------------------------------------------------------------------------------------------ the bases
def test_pca_basis_is_whitened_exactly(data):
    _, _, st = data
    P, lam = SB.solve_pca(st["C"], K)
    assert np.abs(SB.cov_u(P, st["C"]) - np.eye(K)).max() < 1e-9
    assert np.all(np.diff(lam) <= 1e-12) and lam[0] > 3.0                # high-variance (fast) directions first


def test_slow_basis_is_whitened_in_the_card_metric_and_cov_equals_I_when_the_ridge_is_negligible(data):
    _, _, st = data
    V, mu, ridge = SB.solve_slow(st["C"], st["S"], K)
    B = st["C"] + ridge * np.eye(D)
    assert np.abs(V.T @ B @ V - np.eye(K)).max() < 1e-9                 # the card's normalisation V^T (C + ridge I) V = I
    assert np.all(np.diff(mu) <= 1e-12)
    V0, _, r0 = SB.solve_slow(st["C"], st["S"], K, ridge_rel=1e-12)
    assert np.abs(SB.cov_u(V0, st["C"]) - np.eye(K)).max() < 1e-3
    # the deviation of cov(u) from I under the card's ridge is exactly ridge * V^T V
    assert np.abs(SB.cov_u(V, st["C"]) - (np.eye(K) - ridge * V.T @ V)).max() < 1e-9


def test_slow_recovers_the_slow_subspace_and_beats_pca_top_autocorrelation(data):
    clips, Q, st = data
    V, mu, _ = SB.solve_slow(st["C"], st["S"], K)
    P, lam = SB.solve_pca(st["C"], K)
    chk = SB.slow_vs_pca(st, V, mu, P)
    assert chk["pass"] and chk["slow_top_mu"] > 0.9 and 0.4 < chk["pca_top_autocorr_measured"] < 0.75     # PCA's top direction is mid-slow (0.9^5)
    assert PHI ** 5 - 0.03 < chk["slow_top_mu"] < 1.02                                                     # the planted lag-5 autocorrelation (the per-clip offset adds a static component, so a little above)
    # the SLOW subspace is the planted one: project the true slow directions Q[:, :K] onto span(V)
    Qs = Q[:, :K]
    Vo, _ = np.linalg.qr(V)
    assert np.linalg.norm(Qs - Vo @ (Vo.T @ Qs)) / np.linalg.norm(Qs) < 0.25
    # and the cached u of a clip is a smoother signal than PCA-B's
    u_s, u_p = (clips[0] - st["mean"]) @ V, (clips[0] - st["mean"]) @ P
    ac = lambda u: np.mean([np.corrcoef(u[:-5, j], u[5:, j])[0, 1] for j in range(2)])
    assert ac(u_s) > 0.8 > 0.75 > ac(u_p)


def test_a_shuffled_time_basis_fails_the_ordering_check(data):
    _, _, st = data
    P, _ = SB.solve_pca(st["C"], K)
    Vs, mus, _ = SB.solve_slow(st["C"], st["S_xclip"], K)               # S from frame pairs of two different clips: no time adjacency
    bad = SB.slow_vs_pca(st, Vs, mus, P)
    assert not bad["pass"]
    assert bad["slow_top_mu"] < 0.5                                     # the stored mu of a shuffled-time basis is noise, far below the true one
    # the same builder with the TRUE-time statistic passes (the control's other arm)
    V, mu, _ = SB.solve_slow(st["C"], st["S"], K)
    assert SB.slow_vs_pca(st, V, mu, P)["pass"]


def test_generalised_eigenproblem_matches_scipy(data):
    scipy_linalg = pytest.importorskip("scipy.linalg")
    _, _, st = data
    V, mu, ridge = SB.solve_slow(st["C"], st["S"], K)
    w, _ = scipy_linalg.eigh(st["S"], st["C"] + ridge * np.eye(D))
    assert np.abs(np.sort(w)[::-1][:K] - mu).max() < 1e-9


def test_apply_basis_is_the_float32_affine_map(data):
    clips, _, st = data
    V, mu, _ = SB.solve_slow(st["C"], st["S"], K)
    u = SB.apply_basis(clips[0], st["mean"], V)
    assert u.dtype == np.float32 and u.shape == (101, K)
    assert np.abs(u - ((clips[0] - st["mean"]) @ V)).max() < 1e-4


# ------------------------------------------------------------------------------------------------ files: sha and the trainer's reader
def test_basis_file_roundtrip_and_the_sha_formula_equals_the_trainers(tmp_path, data):
    _, _, st = data
    V, mu, _ = SB.solve_slow(st["C"], st["S"], K)
    path = str(tmp_path / "basis_slow.npz")
    sha = SB.save_basis(path, "slow", st["mean"], V, mu, {"x": 1})
    B = SB.load_basis(path)
    assert B["kind"] == "slow" and B["sha"] == sha == B["stored_sha"] and B["proj"].dtype == np.float32 and np.allclose(B["spectrum"], mu)
    sys.path.insert(0, str(Path(SB.__file__).resolve().parents[0]))
    import v8f_mini_train as TR                                       # the trainer of the patched tree (PYTHONPATH / scripts)
    assert TR.basis_sha256_of_file(path) == sha
    # a changed byte changes the sha (verify by content)
    B2 = dict(kind=np.array("slow"), mean=B["mean"], proj=B["proj"] + np.float32(1e-6))
    assert SB.basis_sha256(B2) != sha


def test_a_tampered_basis_file_is_refused(tmp_path, data):
    _, _, st = data
    V, mu, _ = SB.solve_slow(st["C"], st["S"], K)
    path = str(tmp_path / "b.npz")
    SB.save_basis(path, "slow", st["mean"], V, mu, {})
    z = dict(np.load(path))
    z["proj"] = z["proj"] + np.float32(1e-3)
    np.savez(path, **z)
    with pytest.raises(ValueError):
        SB.load_basis(path)


# ------------------------------------------------------------------------------------------------ control 4: what the builder reads
def _code_of(obj):
    import inspect
    src = inspect.getsource(obj)
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    return chr(10).join(l.split("#")[0] for l in src.splitlines())


def test_builder_reads_no_ego_action_or_label_array():
    """Grep half of control 4. Ego kinematics enter ONLY through the boolean ``ok`` validity mask (which frames count), computed in ``Source.inputs`` exactly as ``theory/ladder/zcache.py`` (the trainer's own window filter);
    no value of any ego / action / label array reaches ``nm``, the statistics, the basis or ``u``. The hash half (the basis is bit-identical when the kinematic VALUES are perturbed and the validity pattern is kept) is in ``controls_slowwm.py``."""
    banned = ('["a"]', '["ch1"]', '"pose"', "p10f", '["v9', "label", "lab_", "tac_", "strat", "situation", "ego_", "speed", "yaw_rate", "act[", "a_t")
    for fn in (SB.Acc, SB.solve_slow, SB.solve_pca, SB.apply_basis, SB.nm_of, SB.fit_mask, SB.n_windows, SB.cmd_fit, SB.cmd_cache, SB.cmd_check3, SB.save_basis, SB.load_basis):
        code = _code_of(fn)
        for bad in banned:
            assert bad not in code, (fn.__name__, bad)
        assert "kin3" not in code and "get_kin" not in code, fn.__name__
    code = _code_of(SB.Source.inputs)
    assert code.count("kin3") >= 1 and "np.isfinite(p10).all(1)" in code           # the ok mask
    assert "norm_mean_tokens" in _code_of(SB.nm_of) and code.count("o[GRID]") == 1 and "nm_of(feats)" in code
    # nothing that would let a kinematic VALUE flow on: the returned tuple is (nm, ok)
    assert "return nm, ok" in code


# ------------------------------------------------------------------------------------------------ the read hook
class FakeWM:
    def __init__(self, kind):
        self.wm_basis = kind


class FakeModel:
    def __init__(self, kind):
        self.wm = FakeWM(kind)


def test_hook_dispatch_and_the_selfcheck_guard(tmp_path, data):
    _, _, st = data
    V, mu, _ = SB.solve_slow(st["C"], st["S"], K)
    path = str(tmp_path / "b.npz")
    SB.save_basis(path, "slow", st["mean"], V, mu, {})
    B = SB.load_basis(path)
    assert SH.is_basis(FakeModel("slow")) and not SH.is_basis(FakeModel("adapter")) and not SH.is_basis(object())
    sc = tmp_path / "selfcheck.json"
    sc.write_text(json.dumps({"levers": {"wm_basis_readback": {"kind": "slow", "basis_sha256": B["sha"], "cache_problems": []}}}), encoding="utf-8")
    assert SH.require_arm_basis(str(sc), path)["sha"] == B["sha"]
    sc.write_text(json.dumps({"levers": {"wm_basis_readback": {"kind": "slow", "basis_sha256": "0" * 64, "cache_problems": []}}}), encoding="utf-8")
    with pytest.raises(SH.BasisMismatch):
        SH.require_arm_basis(str(sc), path)
    sc.write_text(json.dumps({"levers": {}}), encoding="utf-8")
    with pytest.raises(SH.BasisMismatch):
        SH.require_arm_basis(str(sc), path)


def test_hook_frames_must_match_and_clip_z1_is_proj_of_the_arms_u(tmp_path):
    torch = pytest.importorskip("torch")
    fr = {"clips": ["a", "b"], "T": [7, 9]}
    B = {"kind": "slow", "sha": "f" * 64, "mean": np.zeros(1024, np.float32), "proj": np.zeros((1024, 256), np.float32)}
    g = np.random.default_rng(0)
    u = g.standard_normal((16, 256)).astype(np.float32)
    f = tmp_path / "ub.npz"
    np.savez(f, clips=np.array(["a", "b"]), T=np.array([7, 9]), ubasis=u, kind=np.array("slow"), basis_sha=np.array("f" * 64))
    SH.attach_basis_frames(fr, B, str(f))
    assert [len(x) for x in fr["ubasis"]] == [7, 9] and fr["basis_kind"] == "slow"
    with pytest.raises(SH.BasisMismatch):
        SH.attach_basis_frames({"clips": ["a", "c"], "T": [7, 9]}, B, str(f))
    with pytest.raises(SH.BasisMismatch):
        SH.attach_basis_frames({"clips": ["a", "b"], "T": [7, 9]}, dict(B, sha="e" * 64), str(f))

    class WM:
        wm_basis = "slow"

        def z1_from_basis(self, x):
            return x[..., :4] * 2.0
    out = SH.clip_z1_basis(WM(), fr)
    assert np.allclose(out["a"].numpy(), u[:7, :4] * 2.0) and np.allclose(out["b"].numpy(), u[7:, :4] * 2.0)
    with pytest.raises(SH.BasisMismatch):
        SH.z_in_basis(FakeModel("pca"), fr, 0)                           # frames of kind slow under a pca model
    with pytest.raises(SH.BasisMismatch):
        SH.z_in_basis(FakeModel("slow"), {"clips": ["a"], "T": [7]}, 0)  # frames with no basis
