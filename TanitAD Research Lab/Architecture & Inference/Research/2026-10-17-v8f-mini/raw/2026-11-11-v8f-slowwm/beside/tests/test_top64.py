"""Tests of protocol B3 (top-64 mu goal cost), CPU. Run (from beside/):
    PYTHONPATH="<tree>/stack;<tree>/stack/scripts;<tree>/taniteval" CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 python -m pytest tests/test_top64.py -q -p no:cacheprovider --basetemp <dir that exists>
The planted-arm test reads the banked R35H s0 checkpoint and frames and skips when free RAM is under 8 GB."""
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BESIDE = os.path.normpath(os.path.join(HERE, ".."))
THEORY = os.path.normpath(os.path.join(BESIDE, "..", ".."))
TREE = "F:/TanitAD-scratch/slowwm/tree"
os.environ.setdefault("LADDER_READ_CODE_ROOT", TREE)
os.environ.setdefault("WMACT_CODE_ROOT", TREE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("SLOWWM_READ_WORK", "F:/TanitAD-scratch/slowwm_beside_build/b3")
for p in (os.path.join(TREE, "stack"), os.path.join(TREE, "stack", "scripts"), os.path.join(TREE, "taniteval"), os.path.join(THEORY, "slowwm"), os.path.join(THEORY, "slowwm", "read"),
          os.path.join(THEORY, "ladder_plantest"), BESIDE):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import top64_lib as T  # noqa: E402
import top64_run as RUN  # noqa: E402
import top64_report as REP  # noqa: E402


# ----------------------------------------------------------------------------------------------------------------- provenance
def test_tanitad_comes_from_the_f_tree():
    import tanitad
    assert os.path.abspath(tanitad.__file__).replace("\\", "/").lower().startswith(TREE.lower()), tanitad.__file__


def test_rules_file_is_the_frozen_one():
    assert hashlib.sha256(open(os.path.join(THEORY, "slowwm", "BESIDE_RULES_SLOWWM.md"), "rb").read()).hexdigest() == T.RULES_SHA


def test_constants_are_the_rules():
    assert (T.N_FRAMES, T.SEED, T.TOPN, T.SV_TOL, T.FD_STEP, T.FD_REL_TOL) == (4096, 0, 64, 1e-6, 1e-3, 1e-3)


# ----------------------------------------------------------------------------------------------------------------- projector
def test_projector_is_orthogonal_and_ranked_by_the_sv_rule():
    g = np.random.default_rng(0)
    J = g.standard_normal((128, 5)) @ g.standard_normal((5, 64)) + 1e-9 * g.standard_normal((128, 64))      # rank 5 plus noise 1e-9 (below 1e-6 * s_max)
    pr = T.projector_from_jbar(J)
    assert pr["rank"] == 5
    P = pr["P"]
    assert np.abs(P @ P - P).max() < 1e-12 and np.abs(P - P.T).max() < 1e-12
    assert np.abs(P @ J - J).max() < 1e-6                                                                      # the columns of J lie in S
    assert T.projector_from_jbar(J, tol=0.0)["rank"] == 64                                                    # red arm: no threshold keeps the noise directions
    with pytest.raises(ValueError):
        T.projector_from_jbar(np.zeros((8, 3)))


# ----------------------------------------------------------------------------------------------------------------- the wrapper
class _Stack(torch.nn.Module):
    def goal_cost(self, z_pred, goal, level, last_k=None):
        k = last_k or (2 if level == 1 else 1)
        k = min(k, z_pred.shape[-2])
        diff = z_pred[..., -k:, :] - goal.unsqueeze(-2)
        return diff.pow(2).sum(dim=-1).mean(dim=-1)


def test_wrapper_identity_bit_for_bit_and_restored():
    st, g = _Stack(), torch.Generator().manual_seed(0)
    z, goal = torch.randn(3, 5, 12, 128, generator=g), torch.randn(3, 1, 128, generator=g)
    ref = {k: st.goal_cost(z, goal, 1, k) for k in (None, 1, 2, 5)}
    with T.restricted_cost(st, np.eye(128)):
        for k, v in ref.items():
            assert torch.equal(T.make_goal_cost(st.goal_cost, torch.eye(128))(z, goal, 1, k), v)
            assert torch.equal(st.goal_cost(z, goal, 1, k), v)
        with pytest.raises(RuntimeError):
            with T.restricted_cost(st, np.eye(128)):
                pass
    assert "goal_cost" not in st.__dict__
    assert torch.equal(st.goal_cost(z, goal, 1, 1), ref[1])


def test_wrapper_is_the_projected_squared_distance_at_level_1_only():
    st, g = _Stack(), torch.Generator().manual_seed(1)
    z, goal = torch.randn(2, 4, 6, 16, generator=g), torch.randn(2, 1, 16, generator=g)
    U = np.linalg.qr(np.random.default_rng(0).standard_normal((16, 5)))[0]
    P = U @ U.T
    with T.restricted_cost(st, P):
        got = st.goal_cost(z, goal, 1, 2).double().numpy()
        up = st.goal_cost(z, goal, 2, None)
    d = (z[..., -2:, :] - goal.unsqueeze(-2)).double().numpy()
    want = ((d @ P) ** 2).sum(-1).mean(-1)
    assert np.abs(got - want).max() < 1e-4 * np.abs(want).max()
    assert torch.equal(up, _Stack().goal_cost(z, goal, 2, None))                                              # levels above 1 keep the library cost
    # red arm: a projector that drops a direction changes the cost
    P2 = np.eye(16)
    P2[0, 0] = 0
    with T.restricted_cost(st, P2):
        assert not torch.equal(st.goal_cost(z, goal, 1, 2), _Stack().goal_cost(z, goal, 1, 2))


# ----------------------------------------------------------------------------------------------------------------- the Jacobian
class _Enc1(torch.nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.window, self.lin = K, torch.nn.Linear(K * D, D)


class _FakeWM(torch.nn.Module):
    """proj = LayerNorm then Linear, enc1 over a window of K projected frames (front-padded with the first), as ``_z1_post``."""

    def __init__(self, K=3, D=24):
        super().__init__()
        torch.manual_seed(0)
        self.K, self.proj = K, torch.nn.Sequential(torch.nn.LayerNorm(256), torch.nn.Linear(256, D))
        self.enc1 = _Enc1(K, D) if K > 1 else None

    def z1_from_basis(self, u):
        p = self.proj(u)
        if self.enc1 is None:
            return p
        K = self.K
        x = torch.cat([p[:, :1].expand(-1, K - 1, -1), p], 1)
        F = p.shape[1]
        w = torch.stack([x[:, i:i + F] for i in range(K)], 2).flatten(2)
        return torch.tanh(self.enc1.lin(w))


def test_jacobian_autograd_equals_central_difference_and_red_arms():
    wm = T.wm64_of(_FakeWM())
    u = np.random.default_rng(1).standard_normal((64, 3, 256)).astype(np.float32)
    cols = np.arange(8)
    ja, jf = T.jbar_autograd(wm, u, cols), T.jbar_fd(wm, u, cols)
    rel = np.linalg.norm(ja - jf) / np.linalg.norm(ja)
    assert rel < T.FD_REL_TOL
    jc, jcf = T.jbar_autograd(wm, u, cols, joint=False), T.jbar_fd(wm, u, cols, joint=False)
    assert np.linalg.norm(jc - jcf) / np.linalg.norm(jc) < T.FD_REL_TOL
    assert np.linalg.norm(T.jbar_autograd(wm, u, cols + 1) - jf) / np.linalg.norm(jf) > T.FD_REL_TOL         # red: shifted columns
    assert np.linalg.norm(jc - jf) / np.linalg.norm(jf) > T.FD_REL_TOL                                        # red: newest frame only against the joint difference


def test_window_size_and_k1_forms_coincide():
    assert T.window_size(_FakeWM(3)) == 3 and T.window_size(_FakeWM(1)) == 1
    wm = T.wm64_of(_FakeWM(1))
    u = np.random.default_rng(2).standard_normal((16, 1, 256)).astype(np.float32)
    assert np.allclose(T.jbar_autograd(wm, u, np.arange(4)), T.jbar_autograd(wm, u, np.arange(4), joint=False), atol=1e-12)


# ----------------------------------------------------------------------------------------------------------------- train frames
def _cache(root, sha, n_clips=6, n=50, bad_sha=False, seed=0):
    os.makedirs(root, exist_ok=True)
    g, files = np.random.default_rng(seed), {}
    for i in range(n_clips):
        ok = np.ones(n, bool)
        ok[:3] = False
        np.savez(os.path.join(root, f"c{i}.npz"), zbasis=g.standard_normal((n, 256)).astype(np.float32), ok=ok)
        files[f"c{i}"] = T.sha256_file(os.path.join(root, f"c{i}.npz"))
    json.dump(dict(basis_sha256=sha, kind="slow", files_sha256=files), open(os.path.join(root, "BASIS_MANIFEST.json"), "w"))
    if bad_sha:
        files["c0"] = "0" * 64
        json.dump(dict(basis_sha256=sha, kind="slow", files_sha256=files), open(os.path.join(root, "BASIS_MANIFEST.json"), "w"))
    return root


def test_train_frames_seeded_ok_only_windows_and_refusals(tmp_path):
    root = _cache(str(tmp_path / "tc"), "a" * 64)
    p1, _, _ = T.sample_train_frames(root, "a" * 64, 100, 0)
    p2, _, _ = T.sample_train_frames(root, "a" * 64, 100, 0)
    p3, _, _ = T.sample_train_frames(root, "a" * 64, 100, 1)
    assert p1 == p2 and p1 != p3 and len(set(p1)) == 100 and all(j >= 3 for _, j in p1)
    u, prov = T.train_windows(root, "a" * 64, 3, 100, 0)
    assert u.shape == (100, 3, 256) and prov["n_frames"] == 100
    c, j = p1[0]
    z = np.load(os.path.join(root, f"{c}.npz"))["zbasis"]
    assert np.array_equal(u[0], z[j - 2:j + 1])
    with pytest.raises(T.SH.BasisMismatch):
        T.sample_train_frames(root, "b" * 64, 100, 0)                                                         # a cache of another basis
    with pytest.raises(ValueError):
        T.sample_train_frames(root, "a" * 64, 10_000, 0)                                                      # fewer ok frames than asked
    bad = _cache(str(tmp_path / "tc_bad"), "a" * 64, bad_sha=True)
    with pytest.raises(ValueError):
        T.train_windows(bad, "a" * 64, 3, 250, 0)                                                              # a clip file that does not match its manifest sha


def test_clamped_windows_at_the_clip_start(tmp_path):
    root = _cache(str(tmp_path / "tc2"), "a" * 64, n_clips=1, n=50)
    with np.load(os.path.join(root, "c0.npz")) as z:
        ok, zb = z["ok"].copy(), z["zbasis"].copy()
    ok[:] = True                                                                                               # frames 0 and 1 become pickable
    np.savez(os.path.join(root, "c0.npz"), zbasis=zb, ok=ok)
    man = json.load(open(os.path.join(root, "BASIS_MANIFEST.json")))
    man["files_sha256"]["c0"] = T.sha256_file(os.path.join(root, "c0.npz"))
    json.dump(man, open(os.path.join(root, "BASIS_MANIFEST.json"), "w"))
    picks, _, _ = T.sample_train_frames(root, "a" * 64, 50, 0)
    u, _ = T.train_windows(root, "a" * 64, 3, 50, 0)
    z = np.load(os.path.join(root, "c0.npz"))["zbasis"]
    k = [j for _, j in picks].index(0)
    assert np.array_equal(u[k], z[[0, 0, 0]])


# ----------------------------------------------------------------------------------------------------------------- beside numbers
def test_beside_numbers_known_values():
    g = np.random.default_rng(0)
    Q = np.linalg.qr(g.standard_normal((32, 32)))[0]
    z = (g.standard_normal((4000, 32)) * (4.0 * 0.6 ** np.arange(32))) @ Q.T
    b_same = T.beside_numbers(Q[:, :8], z, topn=8)
    assert b_same["rank"] == 8 and b_same["angle_max"] < 3.0 and abs(b_same["variance_share_in_S"] - b_same["variance_share_top64_pca"]) < 0.01
    b_orth = T.beside_numbers(Q[:, 24:], z, topn=8)                                                           # red arm: the lowest-variance directions
    assert b_orth["angle_min"] > 80.0 and b_orth["variance_share_in_S"] < 0.05
    assert abs(T.beside_numbers(np.eye(32), z, topn=8)["variance_share_in_S"] - 1.0) < 1e-12


# ----------------------------------------------------------------------------------------------------------------- the runner and the report
def test_runner_refuses_non_slow_arms_and_skips_finished_units(tmp_path, monkeypatch):
    for bad in ("PCA-B_s0|r|b|u|p", "R35H_s0|r|b|u|p"):
        with pytest.raises(ValueError):
            RUN.parse_arm(bad)
    assert RUN.parse_arm("SLOW_rep_s2|r|b|u|p")["units"] == "all"
    with pytest.raises(AssertionError):
        RUN.parse_arm("SLOW_s0|r|b|u")
    monkeypatch.setattr(RUN.RL, "RUNS", str(tmp_path))
    job = ("p1", "SLOW_s0", "c1", "true")
    np.savez(RUN.RL.path(job), x=np.zeros(1))
    assert RUN.work((job, dict(name="SLOW_s0")))[2] == "skip"                                                  # resume per unit: no model is loaded
    with pytest.raises(ValueError):
        T.build_arm("PCA-B_s0", "r", "b", "u", "t", str(tmp_path / "x.npz"))


def test_the_pool_maps_this_modules_work_not_the_full_width_one():
    # RL.run_stage maps RL.work (full width, all windows): a runner that used it would write full-width units under a B3 label
    assert RUN.run_stage.__globals__["work"] is RUN.work and RUN.work is not RUN.RL.work
    assert "restricted_cost" in RUN.work.__code__.co_names


def _write_units(runs, arm, eid, A, v0, tune, score, noise, seed):
    import plantest2_lib as Q
    os.makedirs(runs, exist_ok=True)
    n = len(eid)
    np.savez(os.path.join(runs, f"meta_{arm}.npz"), A=A, v0=v0, eid=np.array(eid), t=np.arange(n), tune_idx=tune, score_idx=score, hist_ok=np.ones(n, bool), std_raw=np.ones(2), ch1=np.array(1.0),
             basis_kind=np.array("slow"), basis_sha=np.array("s" * 64))
    g = np.random.default_rng(seed)
    for form in ("c1", "c3"):
        for goal, nz in (("true", noise), ("rand", 3.0)):
            plan = (A + nz * g.standard_normal(A.shape)).astype(np.float32)
            np.savez(os.path.join(runs, f"p1_{arm}_{form}_flat1_{goal}.npz"), idx=np.arange(n), plan_raw=plan)
            np.savez(os.path.join(runs, f"score_{arm}_{form}_flat1_{goal}_L0.npz"), idx=score, plan_raw=plan[score])
        json.dump(dict(choice={"LONGITUDINAL": 0.0, "LATERAL": 0.0}), open(os.path.join(runs, f"choice_{arm}_{form}_flat1.json"), "w"))


def test_report_pairs_the_margin_with_the_full_width_margin(tmp_path):
    import plantest2_lib as Q
    g = np.random.default_rng(0)
    eid = [f"e{i}" for i in range(12) for _ in range(5)]
    n = len(eid)
    A = np.concatenate([g.standard_normal((n, 1, 2)) * 0.5 + np.array([0.0, 0.0]), g.standard_normal((n, 11, 2)) * 0.5], 1)
    v0 = 5 + g.random(n)
    tune, score = Q.split_indices(eid)
    b3, full = str(tmp_path / "b3"), str(tmp_path / "full")
    for arm in REP.ARMS:
        _write_units(b3, arm, eid, A, v0, tune, score, noise=0.30, seed=1)
        _write_units(full, arm, eid, A, v0, tune, score, noise=0.05, seed=2)
    res = REP.build(b3, full, b3)
    t = res["tables"]["PLANTEST|c1"]
    for arm in REP.ARMS:
        for f in REP.FAMS:
            r = t[f"{arm}|{f}"]
            assert r["m_top64"] is not None and r["m_full"] is not None
            assert abs(r["top64_minus_full"] - (r["m_top64"] - r["m_full"])) < 1e-12
    assert res["no_bar_no_outcome_label"] is True and "outcome" not in res and all(b is None for b in res["beside"].values())
    md = REP.markdown(res)
    assert "no outcome label" in md and "INCOMPLETE" in md and "PLANNABLE" not in md                                # no projector files: the beside rows are INCOMPLETE, never a label
    # a missing unit gives INCOMPLETE, not a number
    os.remove(os.path.join(b3, "p1_SLOW_s1_c1_flat1_rand.npz"))
    assert REP.build(b3, full, b3)["tables"]["PLANTEST|c1"]["SLOW_s1|LATERAL"]["m_top64"] is None
    # a window set that differs from the deciding read's is refused
    A2 = A.copy()
    A2[0, 0, 0] += 1.0
    _write_units(b3, "SLOW_s0", eid, A2, v0, tune, score, noise=0.30, seed=1)
    with pytest.raises(AssertionError):
        REP.build(b3, full, b3)


def test_projector_npz_roundtrip_and_report_beside_rows(tmp_path):
    g = np.random.default_rng(0)
    jb = g.standard_normal((16, 6))
    pr = T.projector_from_jbar(jb)
    z = g.standard_normal((500, 16))
    bn = T.beside_numbers(pr["U"], z, topn=6)
    res = dict(name="SLOW_s0", kind="slow", basis_sha="s" * 64, window_K=3, n_frames=10, seed=0, jbar=jb, jbar_current=jb, U=pr["U"], s=pr["s"], P=pr["P"], rank=pr["rank"], rank_current_variant=pr["rank"],
               angles_joint_vs_current_deg=np.zeros(6), inputs_sha256=dict(ckpt_last="c" * 64, rules=T.RULES_SHA), train_provenance=dict(x=1), beside=bn, cols=np.arange(6), sv_tol=T.SV_TOL, wall_s=1.0, tier=T.TIER)
    p = str(tmp_path / "proj_SLOW_s0.npz")
    T.save_projector(p, res)
    pj = T.load_projector(p)
    assert pj["rank"] == 6 and np.array_equal(pj["P"], pr["P"]) and pj["meta"]["inputs_sha256"]["rules"] == T.RULES_SHA
    import types
    ctx = types.SimpleNamespace(basis_sha="s" * 64, D=16)
    d = tmp_path / "run"
    d.mkdir()
    (d / "ckpt_last.pt").write_bytes(b"x")
    with pytest.raises(ValueError):                                                                           # checkpoint of another arm
        T.verify_projector(pj, ctx, str(d))
    ctx.basis_sha = "t" * 64
    with pytest.raises(ValueError):                                                                           # basis of another arm
        T.verify_projector(pj, ctx, str(d))


# ----------------------------------------------------------------------------------------------------------------- the artifact and the planted arm
def test_banked_controls_all_pass_when_present():
    p = os.path.join(BESIDE, "CONTROLS_B3.json")
    if not os.path.exists(p):
        pytest.skip("CONTROLS_B3.json not written yet")
    c = json.load(open(p, encoding="utf-8"))
    assert c["all_pass"] is True and c["rules_sha256"] == T.RULES_SHA
    for k in ("i_identity", "ii_bottom64_red_arm", "iii_jacobian"):
        assert c[k]["pass"] is True


def test_planted_arm_identity_on_8_windows_and_a_projector_changes_the_plan(tmp_path):
    import controls_read_slowwm as C
    import top64_controls as TC
    if T.LR.free_gb() < 8.0:
        pytest.skip("free RAM under 8 GB")
    torch.set_num_threads(1)
    P = TC.setup()
    ctx = T.S.prepare_arm_read("SLOW_s0", P["run"], P["pb"]["path"], P["ub"])
    idx = T.LP.sel_arrays(ctx, "c1")["all"][:8]
    full = T.LP.unit3(ctx, "flat1", "true", "c1", idx, 0.0, seed_mode="prior", collect=False)
    with T.restricted_cost(ctx.st, np.eye(ctx.D)):
        ident = T.LP.unit3(ctx, "flat1", "true", "c1", idx, 0.0, seed_mode="prior", collect=False)
    assert np.array_equal(full["plan_raw"], ident["plan_raw"]) and np.array_equal(full["cost1"], ident["cost1"])
    Pm = np.eye(ctx.D)
    Pm[:, :64] = 0.0
    with T.restricted_cost(ctx.st, Pm):
        red = T.LP.unit3(ctx, "flat1", "true", "c1", idx, 0.0, seed_mode="prior", collect=False)
    assert not np.array_equal(full["cost1"], red["cost1"]) and not np.array_equal(full["plan_raw"], red["plan_raw"])
