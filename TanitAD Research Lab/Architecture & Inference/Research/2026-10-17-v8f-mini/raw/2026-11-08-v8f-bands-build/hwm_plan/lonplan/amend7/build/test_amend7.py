"""AMEND_HWMPLAN_7 build tests. CPU, serial, synthetic data only (no real clip, no real target). Each green check has a mutation (the card's regression arm) that must read RED."""
import hashlib, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pytest
import torch
import amend7_env  # noqa: F401  (first)
import amend5_controls as K5
import amend7_controls as C7
import amend7_read as RD7
import fit7 as FT
import fit_ra6 as F
import mlp7 as M
import pairs7 as PR
import ra7_l1
import ra_l1 as R

torch.set_num_threads(2)
ROOT = os.path.dirname(HERE)
TAG = "R35_s0"
ident = lambda x: x
SMALL = [(8, 1.0), (30, 0.01)]                # a tiny grid for the synthetic fits (the real one is M.GRID)


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(M, "GRID", SMALL)
    monkeypatch.setattr(M, "WIDTH", 24)


# ---------------------------------------------------------------------------------------------- synthetic data
def synth(n_clips=14, T=44, seed=0, names=None, w=0.7, d=128):
    """Latents file in the layout of latents_train_R35_s{k}.pt: linear truth, 16-frame windows every 4 frames."""
    r = np.random.RandomState(seed)
    wv = r.randn(d) * 0.05
    Z, A, OK, TT, EID = [], [], [], [], []
    for c in range(n_clips):
        name = names[c] if names else f"clip{c:03d}"
        z = np.cumsum(r.randn(T, d) * 0.3, 0)
        a = np.zeros((T, 2)); a[:, 0] = np.r_[z[1:] - z[:-1], np.zeros((1, d))] @ wv + r.randn(T) * 0.05
        for t in range(15, T, 4):
            Z.append(z[t - 15:t + 1]); A.append(a[t - 15:t + 1]); OK.append(np.ones(16, bool)); TT.append(t); EID.append(name)
    return dict(Z1={TAG: torch.tensor(np.stack(Z), dtype=torch.float32)}, A_raw=torch.tensor(np.stack(A), dtype=torch.float32), ok=torch.tensor(np.stack(OK)), t=np.array(TT), eid=EID)


def synth_table(nonlinear=True, n_clips=40, per=50, d=8, seed=0):
    """A pair table in pairs7's layout. z10 = z0 + h(y) u + noise, with a nonlinear h (y + 0.8 y^3: the readout is nonlinear in D) or h = identity (linear truth); F_true / F_alt are the same transition under the logged / mean acceleration."""
    r = np.random.RandomState(seed)
    n = n_clips * per
    oc = np.repeat(np.arange(n_clips), per)
    z0 = r.randn(n, d)
    u = r.randn(d); u /= np.linalg.norm(u)
    y = r.randn(n)
    h = (lambda a: a + 0.8 * a ** 3) if nonlinear else (lambda a: a)
    z10 = z0 + h(y)[:, None] * u + 0.03 * r.randn(n, d)
    Ft = z0 + h(y)[:, None] * u
    Fa = z0 + h(np.full(n, M.ABAR))[:, None] * u
    frame = np.tile(np.arange(per), n_clips)
    fo = (np.random.RandomState(0).permutation(n_clips) % 5)[oc]
    part = np.empty(n, int)
    for c in range(n_clips):
        part[oc == c] = np.where(oc == c)[0][np.random.RandomState(c).permutation(per)]
    return dict(z0=z0.astype(np.float32), z10=z10.astype(np.float32), y=y, ep=oc.copy(), oc=oc, fold=fo, frame=frame, fr0=np.zeros(n, int), part=part, act=np.zeros((n, 10, 2), np.float32),
                F_true=Ft.astype(np.float32), F_alt=Fa.astype(np.float32), anc=np.arange(0, n, 10), zp=z0[::10].astype(np.float32), zm=z0[::10].astype(np.float32))


def fit_all(T, learn_over=None):
    """RA, KCTL, MLP-A, MLP-B, ZERO, CONST, SHUF-T, PLANT crossfits on a table, small learners (monkeypatched grid). -> dict of results."""
    out = {}
    out["RA"] = FT.crossfit_arm("RA", T, "s0")
    lams = [f["lam"] for f in out["RA"]["folds"]]
    out["KCTL"] = FT.crossfit_arm("KCTL", T, "s0", lams=lams)
    out["MLPA"] = FT.crossfit_arm("MLPA", T, "s0")
    out["MLPB"] = FT.crossfit_arm("MLPB", T, "s0")
    out["ZERO"] = FT.crossfit_arm("ZERO", T, "s0")
    out["CONST"] = FT.crossfit_arm("CONST", T, "s0")
    out["SHUFT"] = FT.crossfit_arm("SHUFT", T, "s0", y_override=M.shuffled(T["y"]))
    return out


# ---------------------------------------------------------------------------------------------- 2.1 / 2.3: the net, seeds, selection
def test_seed_rule_bit_identical_and_sets_differ():
    r = np.random.RandomState(0)
    X, y = r.randn(300, 8), r.randn(300)
    m1 = M.fit_snapshots(X, y, [3], 0.1, 7, 16)[3]
    m2 = M.fit_snapshots(X, y, [3], 0.1, 7, 16)[3]
    m3 = M.fit_snapshots(X, y, [3], 0.1, 507, 16)[3]
    p1, p2, p3 = (M.predict(m, X) for m in (m1, m2, m3))
    assert np.array_equal(p1, p2) and not np.allclose(p1, p3)                   # same seed and config: bit-identical; seed set A vs B: different
    assert not np.array_equal(M.batch_order(300, 7, 0), M.batch_order(300, 8, 0)) and not np.array_equal(M.batch_order(300, 7, 0), M.batch_order(300, 7, 1))
    assert np.array_equal(M.batch_order(300, 7, 2), M.batch_order(300, 7, 2))


def test_snapshots_equal_independent_runs():
    r = np.random.RandomState(1)
    X, y = r.randn(200, 6), r.randn(200)
    snap = M.fit_snapshots(X, y, [2, 5], 0.01, 3, 16)
    alone = M.fit_snapshots(X, y, [2], 0.01, 3, 16)[2]
    assert np.array_equal(M.predict(snap[2], X), M.predict(alone, X))


def test_net_shape_and_fit_recovers_a_nonlinear_map():
    net = M.make_net(256)
    assert [tuple(p.shape) for p in net.parameters()] == [(256, 256), (256,), (256, 256), (256,), (1, 256), (1,)]
    r = np.random.RandomState(2)
    X = r.randn(1500, 4); y = np.sin(2 * X[:, 0]) * X[:, 1]
    m = M.fit_snapshots(X, y, [40], 0.01, 0, 64)[40]
    assert M.r2(M.predict(m, X), y) > 0.6


def test_selection_inner_folds_only_and_tie_goes_to_the_earlier_config(monkeypatch, small):
    T = synth_table(n_clips=20, per=30)
    X = M.build_input("z0z1", T["z0"], T["z10"])
    tr = T["fold"] != 0
    sel = M.select_config(X[tr], T["y"][tr], T["fold"][tr], T["oc"][tr], 0, SMALL, 32)
    outer_test = set(np.unique(T["oc"][T["fold"] == 0]).tolist())
    train_clips = set(np.unique(T["oc"][tr]).tolist())
    assert len(sel["fits"]) == 4 and all(set(f["val_clips"]) <= train_clips and not (set(f["val_clips"]) & outer_test) for f in sel["fits"])
    assert [f["seed"] for f in sel["fits"]] == [1, 2, 3, 4]                     # seed = S0 + 10 f + j with f = 0, j = the inner fold
    monkeypatch.setattr(M, "predict", lambda m, X_: np.zeros(len(X_)))           # every config ties: the earlier one wins
    tie = M.select_config(X[tr], T["y"][tr], T["fold"][tr], T["oc"][tr], 0, [(2, 1.0), (6, 0.1), (2, 0.01)], 32)
    assert tie["chosen"] == [2, 1.0]
    assert M.GRID == SMALL


def test_grid_order_is_the_cards():
    assert M.GRID == [(e, wd) for e in (2, 6, 12, 24) for wd in (1.0, 0.1, 0.01)] and len(M.GRID) == 12 and (M.S0_A, M.S0_B) == (0, 500)


# ---------------------------------------------------------------------------------------------- 3.5 the K-control model
def test_lind_reproduces_ridge_within_1e8():
    r = np.random.RandomState(0)
    z0 = r.randn(400, 12); D = r.randn(400, 12) @ r.randn(12, 12) * 0.5
    y = D @ r.randn(12) + 0.3 * r.randn(400)
    y = y / y.std()                                                    # unit scale, as the scaled accelerations are
    X = np.concatenate([z0, z0 + D], 1)
    for lam in (0.1, 1.0):
        m = M.fit_lind(X, y, lam)
        ref = F.ridge_pred(F.ridge_fit(D, y[:, None], lam), D)[:, 0]
        assert np.abs(M.predict(m, X) - ref).max() < 1e-8


def test_k_control_green_and_misaligned_targets_red():
    T = synth_table(nonlinear=False, n_clips=25, per=40)
    ra = FT.crossfit_arm("RA", T, "s0")
    lams = [f["lam"] for f in ra["folds"]]
    kc = FT.crossfit_arm("KCTL", T, "s0", lams=lams)
    assert M.check_k(kc["pred"], ra["pred"], T["y"])["ok"]
    mis = M.crossfit(M.lind_learner(lams), "z0z1", T["z0"], T["z10"], T["y"][T["part"]], T["fold"], T["oc"])
    assert not M.check_k(mis["pred"], ra["pred"], T["y"])["ok"]


# ---------------------------------------------------------------------------------------------- 8.1 the pure controls and their regression arms
def test_c_pairs7_green_and_mutations_red(tmp_path):
    LT = synth()
    ext, clips = PR.pairs_ext(LT, ident)
    Dr, Yf, EP, clips_ref = F.pairs(LT, ident)
    ref = dict(Dr=Dr, Yf=Yf, EP=EP, clips=clips_ref)
    ok = M.check_pairs(ext, ref, len(Yf), len(clips_ref))
    assert ok["ok"] and clips == clips_ref
    bad = dict(ext, D=ext["D"].copy()); bad["D"][3, 5] += 1e-12
    assert not M.check_pairs(bad, ref)["ok"]
    assert not M.check_pairs(ext, ref, len(Yf) + 1)["ok"] and not M.check_pairs(ext, ref, len(Yf), len(clips_ref) + 1)["ok"]
    # a flipped byte in a chunk refuses (check_chunk_files and chunk_parts)
    order = [f"clipx{i:03d}" for i in range(5)]
    A = synth(3, names=order[:3]); B = synth(2, seed=3, names=order[3:])
    J = dict(chunks=[])
    for i, (d_, cl) in enumerate(((A, order[:3]), (B, order[3:]))):
        fn = f"chunk_{i:03d}.pt"
        torch.save(dict(d_, clips=cl, starts={}, pull_s=[]), tmp_path / fn)
        J["chunks"].append(dict(file=fn, clips=cl, n_windows=len(d_["eid"]), sha256=hashlib.sha256((tmp_path / fn).read_bytes()).hexdigest(), md5_verified=len(cl)))
    (tmp_path / "CHUNKS.json").write_text(json.dumps(J))
    assert M.check_chunk_files(J, str(tmp_path))["ok"]
    raw = bytearray((tmp_path / "chunk_000.pt").read_bytes()); raw[len(raw) // 2] ^= 1
    (tmp_path / "chunk_000.pt").write_bytes(bytes(raw))
    assert not M.check_chunk_files(J, str(tmp_path))["ok"]
    with pytest.raises(RuntimeError):
        F.chunk_parts(TAG, 3, str(tmp_path / "CHUNKS.json"), str(tmp_path))


def test_c_iso7_green_and_planted_score_eid_red():
    c141 = [f"a{i}" for i in range(6)]; c6 = [f"b{i}" for i in range(9)]
    score = ["s1", "s2"]
    assert M.check_iso(c141 + c6[:5], c141, c6, score, n_new=5)["ok"]
    bad = c141 + c6[:5]; bad[2] = "s1"
    assert not M.check_iso(bad, c141, c6, score, n_new=5)["ok"]
    assert not M.check_iso(c141 + c6[:5] + ["s2"], c141, c6, score, n_new=5)["ok"]
    assert not M.check_iso(c141 + c6[:4], c141, c6, score, n_new=5)["ok"]


def test_c_curve7_reproduces_curve_pure_and_sorted_id_fold_map_reads_red():
    names = [f"c{i:02d}" for i in range(25)][::-1]                       # order index runs against the sorted-id index
    LT = synth(25, 60, names=names)
    Dr, Yf, EP, clips = F.pairs(LT, ident)
    order = list(names)
    oc = np.array([order.index(clips[i]) for i in EP])
    fo = (np.random.RandomState(0).permutation(25) % 5)[oc]
    res = M.crossfit(M.ridge_learner(), "D", np.zeros_like(Dr), Dr, Yf[:, 0], fo, oc)       # z0 = 0, so D = z10
    ref = F.curve_pure(Dr, Yf, oc, 25, [25], n_boot=60)["25"]
    mine = M.curve_row(res["pred"], Yf[:, 0], oc, 25, 60, 0)
    assert M.check_curve(mine, ref)["ok"] and M.check_curve(mine, ref)["max_abs_diff"] < 1e-9
    fo_bad = F.fold_map(25, EP)
    bad = M.crossfit(M.ridge_learner(), "D", np.zeros_like(Dr), Dr, Yf[:, 0], fo_bad, oc)
    assert not M.check_curve(M.curve_row(bad["pred"], Yf[:, 0], oc, 25, 60, 0), ref)["ok"]


def test_c_fold7_and_c_sel7_green_and_mutated_arms_red(small):
    T = synth_table(n_clips=25, per=30)
    n_clips = 25
    ml = FT.crossfit_arm("MLPA", T, "s0")
    assert M.check_fold(ml["folds"], n_clips)["ok"] and M.check_sel(ml["folds"])["ok"]
    mut = M.crossfit(M.ridge_learner(), "D", T["z0"], T["z10"], T["y"], T["fold"], T["oc"], mutate="test_in_train")        # fold f's rows added to its own fit
    assert not M.check_fold(mut["folds"], n_clips)["ok"]
    X = M.build_input("z0z1", T["z0"], T["z10"])
    sel_mut = M.crossfit(M.mlp_learner(0, mutate="select_on_test"), "z0z1", T["z0"], T["z10"], T["y"], T["fold"], T["oc"])   # the selection scored on the OUTER TEST fold
    assert not M.check_sel(sel_mut["folds"])["ok"]
    import copy
    bad = copy.deepcopy(ml["folds"]); bad[1]["selection"]["chosen"] = [999, 0.5]
    assert not M.check_sel(bad)["ok"]


def test_c_const7_c_zero7_and_shuf_t(small):
    T = synth_table(n_clips=25, per=30)
    y, fo = T["y"], T["fold"]
    c = FT.crossfit_arm("CONST", T, "s0")
    assert M.check_const(c["pred"], y, fo)["ok"] and not M.check_const(np.full(len(y), y.mean()), y, fo)["ok"]
    z = FT.crossfit_arm("ZERO", T, "s0")
    assert M.check_zero(z["dR"])["ok"]
    dR_bad = np.zeros(len(y))
    for f in range(5):                                                   # ZERO read with the real z10: its fold models read [z0, F]
        te = fo == f
        dR_bad[te] = M.predict(z["models"][f], M.build_input("z0z1", T["z0"][te], T["F_true"][te])) - M.predict(z["models"][f], M.build_input("z0z1", T["z0"][te], T["F_alt"][te]))
    assert not M.check_zero(dR_bad)["ok"]
    sh = FT.crossfit_arm("SHUFT", T, "s0", y_override=M.shuffled(y))
    real = FT.crossfit_arm("MLPA", T, "s0")
    assert M.check_shuf(M.r2(sh["pred"], M.shuffled(y)))["ok"] and not M.check_shuf(M.r2(real["pred"], y))["ok"]      # the identity shuffle reads RED


def test_c_wm7_sign_exchange_and_roll_causality():
    T = synth_table(n_clips=25, per=30)
    ra = FT.crossfit_arm("RA", T, "s0")
    y, oc = T["y"], T["oc"]
    x = ra["dR"]
    bt = M.Boot(oc, 25, dict(y=y, yy=y * y, x=x, xx=x * x, xy=x * y, nx=-x, nxx=x * x, nxy=-x * y))
    _, lo, hi = bt.run(lambda t: np.array([M.t_corr(t, "x", "xx", "xy"), M.t_corr(t, "nx", "nxx", "nxy")]), 100, 0)
    assert M.check_wm_sign([lo[0], hi[0]])["ok"] and not M.check_wm_sign([lo[1], hi[1]])["ok"]          # F_true and F_alt exchanged reads RED
    ctx = K5.toy_ctx(D=8)
    r = np.random.RandomState(0)
    z0 = torch.randn(16, 8, dtype=torch.float64); act = torch.randn(16, 10, 2, dtype=torch.float64)
    st = ctx.st
    ok = M.check_roll_causal(lambda z, a: R.roll(st, z, a), lambda z, a: st.rollout(1, z.unsqueeze(1), a), z0, act)
    assert ok["ok"]

    class NonCausal(K5.ToyStack):
        def rollout(self, level, z0, a, context=1):
            s = super().rollout(level, z0, a, context)
            return s + (a[..., 10:, :].sum((-2, -1)).unsqueeze(-1).unsqueeze(-2) if a.shape[-2] > 10 else 0)
    nc = NonCausal(ctx.st.B)
    bad = M.check_roll_causal(lambda z, a: R.roll(nc, z, a), lambda z, a: nc.rollout(1, z.unsqueeze(1), a), z0, act)
    assert not bad["ok"]


def test_planted_target_is_fixed_and_plant_controls(small):
    T = synth_table(n_clips=40, per=60, d=16)
    z0, D = T["z0"].astype(np.float64), T["z10"].astype(np.float64) - T["z0"].astype(np.float64)
    yp, r2o, f = M.planted(z0, D)
    yp2, r2o2, _ = M.planted(z0, D)
    assert np.array_equal(yp, yp2) and r2o == r2o2 and 0.3 < r2o < 0.7 and abs(yp.std() - 1.2) < 0.3
    assert np.abs(f).max() < 20
    ra = FT.crossfit_arm("RA", T, "s0", y_override=yp, rolls=False)
    kc = FT.crossfit_arm("KCTL", T, "s0", y_override=yp, lams=[f_["lam"] for f_ in ra["folds"]], rolls=False)
    assert not M.check_plant_p1(M.r2(kc["pred"], yp), r2o)["ok"]            # P3: the linear K-control model misses P1 (the regression arm reads RED)
    good = dict(point=0.1, lo=0.05, hi=0.15)
    assert M.check_plant(0.8 * r2o, r2o, good, 0.01)["ok"]
    assert not M.check_plant(0.5 * r2o, r2o, good, 0.01)["ok"]                                          # P1 missed
    assert not M.check_plant(0.9 * r2o, r2o, dict(point=0.1, lo=-0.01, hi=0.15), 0.01)["ok"]           # P2: CI not above 0
    assert not M.check_plant(0.9 * r2o, r2o, dict(point=0.005, lo=0.001, hi=0.01), 0.01)["ok"]          # P2: point below M_k


# ---------------------------------------------------------------------------------------------- the gate (section 4)
def _gate_for(T, tmp_path, monkeypatch, B=150):
    monkeypatch.setattr(FT, "ROOT", str(tmp_path)); monkeypatch.setattr(FT, "FITD", str(tmp_path / "fit")); monkeypatch.setattr(PR, "PAIRS_DIR", str(tmp_path / "pairs"))
    os.makedirs(tmp_path / "pairs", exist_ok=True); os.makedirs(tmp_path / "fit", exist_ok=True)
    p = tmp_path / "pairs" / "pairs7_s0.npz"
    np.savez(p, **T)
    (tmp_path / "PAIRS7_s0.json").write_text(json.dumps(dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
    for arm in ("RA", "KCTL", "MLPA", "MLPB", "ZERO", "CONST"):
        FT.cmd_crossfit("s0", arm)
    S, A = FT.chassis_stats("s0", B)
    g = M.chassis_gate(dict(dR2_A=S["dR2_A"], dR2_B=S["dR2_B"], dr_A=S["dr_A"], dr_B=S["dr_B"]), 0.01, S["F_T_r2"], S["F_T_r_wm"])
    return S, g


def test_gate_reads_pass_for_a_planted_nonlinear_truth_and_fit_fail_for_a_linear_one(small, tmp_path, monkeypatch):
    (tmp_path / "nl").mkdir(); (tmp_path / "lin").mkdir()
    S, g = _gate_for(synth_table(True, 50, 80), tmp_path / "nl", monkeypatch)
    assert S["dR2_A"]["lo"] > 0 and g["a"] and g["b"] and g["label"] == "R7-FIT-PASS", (S["dR2_A"], S["dr_A"], g)
    S2, g2 = _gate_for(synth_table(False, 50, 80), tmp_path / "lin", monkeypatch)
    assert not g2["a"] and g2["label"] == "R7-FIT-FAIL", (S2["dR2_A"], g2)


def test_chassis_gate_and_fit_outcome_table_row_by_row():
    ci = lambda p, lo, hi: dict(point=p, lo=lo, hi=hi)
    good = dict(dR2_A=ci(0.06, 0.04, 0.08), dR2_B=ci(0.06, 0.04, 0.08), dr_A=ci(0.1, 0.05, 0.15), dr_B=ci(0.1, 0.05, 0.15))
    assert M.chassis_gate(good, 0.0238, 0.01, 0.01)["label"] == "R7-FIT-PASS"
    assert M.chassis_gate(dict(good, dR2_A=ci(0.02, 0.01, 0.03)), 0.0238, 0.01, 0.01)["a"] is False             # point below M_k
    assert M.chassis_gate(dict(good, dR2_A=ci(0.06, 0.04, 0.08)), 0.0238, 0.07, 0.01)["a"] is False            # point below F_T
    assert M.chassis_gate(dict(good, dR2_B=ci(0.06, -0.01, 0.08)), 0.0238, 0.01, 0.01)["a"] is False           # seed set B's CI not above 0
    state = dict(good, dr_A=ci(0.1, -0.01, 0.15))
    assert M.chassis_gate(state, 0.0238, 0.01, 0.01)["label"] == "R7-STATE-ONLY"
    fail_b = dict(good, dR2_A=ci(0.0, -0.02, 0.02))
    assert M.chassis_gate(fail_b, 0.0238, 0.01, 0.01) == dict(a=False, b=True, label="R7-FIT-FAIL")
    per = lambda a, b: M.chassis_gate(a, 0.0238, 0.01, 0.01)
    P, S_, Fb, Fn = per(good, 0), per(state, 0), per(fail_b, 0), per(dict(good, dR2_A=ci(0.0, -0.02, 0.02), dr_A=ci(0.0, -0.1, 0.1)), 0)
    assert M.fit_outcome(dict(s0=P, s1=P))["text"] == "R7-FIT-PASS"
    assert M.fit_outcome(dict(s0=S_, s1=S_))["text"] == "R7-STATE-ONLY"
    assert M.fit_outcome(dict(s0=Fb, s1=Fb))["text"] == "R7-FIT-FAIL TRANS-ONLY"
    assert M.fit_outcome(dict(s0=Fn, s1=Fn))["text"] == "R7-FIT-FAIL"
    o = M.fit_outcome(dict(s0=P, s1=Fn))                                  # one chassis only: SEEDS-DISAGREE, the less favourable label applies
    assert o["outcome"] == "R7-FIT-FAIL" and o["seeds_disagree"] and "SEEDS-DISAGREE" in o["text"]
    assert M.fit_outcome(dict(s0=P, s1=S_))["outcome"] == "R7-STATE-ONLY"
    assert M.fit_outcome(dict(s0=P, s1=P), controls_ok=False)["text"] == "R7-FIT-INVALID"


# ---------------------------------------------------------------------------------------------- pairs7 plumbing
def test_pairs7_table_rollouts_partner_and_anchors():
    LT = synth(12, 60)
    st = K5.toy_ctx(D=128, dtype=torch.float32).st
    clips = sorted(set(LT["eid"]))
    t, ck = PR.build_table(LT, ident, st, clips[:4], clips[4:], ["not-a-clip"], delta=0.42)
    n = len(t["y"])
    assert ck["C_PAIRS7"]["ok"] and ck["C_ISO7"]["ok"] and t["F_true"].shape == (n, 128) and t["F_alt"].shape == (n, 128)
    assert (t["oc"] >= 0).all() and t["fold"].min() >= 0 and t["fold"].max() <= 4
    same = t["ep"] == t["ep"][t["part"]]
    far = np.abs(t["frame"] - t["frame"][t["part"]]) >= 5
    assert same.all() and (far | (t["part"] == np.arange(n))).all()
    anc = t["anc"]
    assert ((t["frame"][anc] - t["fr0"][anc]) % 10 == 0).all() and len(anc) == int(((t["frame"] - t["fr0"]) % 10 == 0).sum())
    assert t["zp"].shape == (len(anc), 128) and not np.allclose(t["zp"], t["zm"])
    # identity: with the logged acceleration equal to abar everywhere, F_true == F_alt
    act = t["act"].copy(); act[..., 0] = M.ABAR
    Ft, Fa = PR.rollouts(st, t["z0"], act)
    assert np.array_equal(Ft, Fa)
    with pytest.raises(AssertionError):                                   # C-ISO7: a SCORE episode among the fit clips refuses
        PR.build_table(LT, ident, st, clips[:4], clips[4:], [clips[7]])


def test_pairs7_load_refuses_a_table_whose_sha_differs(tmp_path):
    p = tmp_path / "pairs7_s0.npz"; np.savez(p, y=np.zeros(3))
    rec = tmp_path / "rec.json"; rec.write_text(json.dumps(dict(sha256="0" * 64)))
    with pytest.raises(RuntimeError):
        PR.load_table(str(p), str(rec))
    rec.write_text(json.dumps(dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
    assert len(PR.load_table(str(p), str(rec))["y"]) == 3


# ---------------------------------------------------------------------------------------------- C-GRAD7, C-PLANT7
def test_c_grad7_toy_green_and_two_regressions_red():
    ctx = C7.toy_ctx7()
    world, base = K5.toy_world(ctx)
    z0 = torch.from_numpy(world["z"][:, 0]); hc = ctx.hold_c(world["A"][:, 0])[:, :12]
    src = torch.from_numpy(base["a1_raw"][:, :, 0] / R.ACC)
    with torch.no_grad():
        tgt = ctx.ra.read(R.roll(ctx.st, z0, torch.stack([src, hc[:, :, 1]], -1).unsqueeze(1))[:, 0] - z0, z0)
    G = lambda **kw: C7.grad_of7(ctx.st, ctx.ra, z0, hc[:, :, 0], hc[:, :, 1], tgt, ctx.lo1, ctx.hi1, **kw)
    g = K5.check_grad(G())
    assert g["ok"] and g["max_abs_knots_11_12"] == 0.0 and g["max_abs_yaw"] == 0.0 and g["min_abs_knots_1_10"] > 0
    assert not K5.check_grad(G(detach=True))["ok"]                       # RA7 detached inside the cost
    assert not K5.check_grad(G(read_after=12))["ok"]                     # RA7 reading after 12 actions


def test_c_plant7_i_and_its_mutation():
    ctx = C7.toy_ctx7()
    w = K5.toy_world(ctx)
    ok = C7.check_plant_toy7(ctx, w)
    assert ok["ok"], ok
    assert not C7.check_plant_toy7(ctx, w, mutate="own_actions")["ok"]


# ---------------------------------------------------------------------------------------------- ra7_l1, amend7_env
def _linear_pair(D=8, seed=0):
    """An RA (ra_l1.Readout) and an RA7-class readout that is exactly linear in D."""
    ctx = K5.toy_ctx(D=D, seed=seed)
    W = ctx.ra.W
    net = torch.nn.Linear(2 * D, 1)
    with torch.no_grad():
        net.weight.copy_(torch.cat([-W[:, 0], W[:, 0]])[None]); net.bias.zero_()
    ra7 = ra7_l1.MLPReadout(net, torch.zeros(2 * D), torch.ones(2 * D), "toy", "toy", torch.float64)
    return ctx, ra7


@pytest.mark.parametrize("mode,src", [("l1", "l1"), ("neg", "l1"), ("shuf", "l1"), ("oracle", "none"), ("off", "none")])
def test_plan_arrays_fork_equals_ra_l1_for_a_readout_linear_in_d(mode, src):
    ctx, ra7 = _linear_pair()
    win, base = K5.toy_world(ctx)
    ref, b0 = R._orig_plan_arrays(ctx, base, win, mode, src)
    ctx7 = R.Ctx(st=ctx.st, h1=ctx.h1, lo1=ctx.lo1, hi1=ctx.hi1, ra=ra7, hold_c=ctx.hold_c, ch1_scale=ctx.ch1_scale, tag="toy")
    new, b7 = R.plan_arrays(ctx7, base, win, mode, src)                   # the patched (forked) function
    for key in ("plan_raw", "a1_raw", "cost1"):
        assert np.abs(np.asarray(new[key], np.float64) - np.asarray(ref[key], np.float64)).max() < 1e-6, (mode, key)
    if mode != "off":
        assert np.abs(new["ra_target_scaled"] - ref["ra_target_scaled"]).max() < 1e-6
        assert b7["kind"] == "mlp" and b7["mirror"] == b0["mirror"] and b7["perm"] == b0["perm"] and b7["src_key"] == b0["src_key"]


def test_readout_with_a_wrong_sha_refuses_and_a_changed_byte_refuses(tmp_path):
    net = M.make_net(16, 8)
    p = tmp_path / "readout7_MLP_R35_s0.pt"
    torch.save(dict(kind="mlp", width=8, mu=np.zeros(16), sd=np.ones(16), net_state=net.state_dict()), p)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    r = ra7_l1.load_readout7(str(p), "s0", sha)
    assert r.sha256 == sha and r.read(torch.zeros(3, 8), torch.zeros(3, 8)).shape == (3,)
    with pytest.raises(R.RefusedRun):
        ra7_l1.load_readout7(str(p), "s0", "0" * 64)
    raw = bytearray(p.read_bytes()); raw[len(raw) // 2] ^= 1
    q = tmp_path / "flip.pt"; q.write_bytes(bytes(raw))
    with pytest.raises(R.RefusedRun):
        ra7_l1.load_readout7(str(q), "s0", sha)
    assert not C7.check_ra7_sha(str(p), "s0", sha, flip=True, tmp=str(tmp_path))["ok"] and C7.check_ra7_sha(str(p), "s0", sha)["ok"]


def test_env_with_no_fit_record_gives_zero_sha_and_never_falls_back(tmp_path):
    import amend5_stage as ST
    assert ST.READOUT_FALLBACKS == [] and "amend5" not in ST.readout_path("s0") and "amend6" not in ST.readout_path("s0") and ST.readout_path("s0").endswith("readout7_MLP_R35_s0.pt")
    assert ST.SPEC.endswith("AMEND_HWMPLAN_7.md") and ST.SPEC_SHA == open(ROOT + "/AMEND_HWMPLAN_7.sha256").read().split()[0] == "95c97b39f2692aeb9bff67383e8f8ba58771aeee18739d8693130e80297f7be9"
    assert R.plan_arrays is ra7_l1.plan_arrays7 and R.load_readout is ra7_l1.load_readout7
    code = ("import amend7_env, amend5_stage as ST, ra_l1 as R, sys\n"
            "print(ST.RA_SHA['s0'], ST.RA_SHA['s1'], amend7_env.sha_B('s0'))\n"
            "p = ST.readout_path('s0')\n"
            "try:\n    R.load_readout(p, 's0')\nexcept Exception as e:\n    print(type(e).__name__)\n")
    e = dict(os.environ, AMEND7_ROOT=str(tmp_path), PYTHONPATH=os.pathsep.join([HERE, os.environ.get("PYTHONPATH", "")]))
    (tmp_path / "readout_in").mkdir()
    open(tmp_path / "readout_in" / "readout7_MLP_R35_s0.pt", "wb").write(b"not a readout")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", env=e, cwd=HERE)
    out = r.stdout.split()
    assert out[:3] == ["0" * 64] * 3 and out[3] == "RefusedRun", (r.stdout, r.stderr[-500:])


# ---------------------------------------------------------------------------------------------- the SCORE outcome table (section 7) and the floors
def _rows(delta, lo, hi, fam, band, met, a="A", b="B"):
    return dict(family=fam, band=band, metric=met, a=a, b=b, delta=delta, lo=lo, hi=hi)


def _js(d0, dH, repX, repB, fam="LONGITUDINAL", band="0-2s", met="acc_mae"):
    """A contrast file in the shape RD.cell reads: X - HOLD, X - ref, and the two replicate rows."""
    return dict(rows=[_rows(*d0, fam, band, met), _rows(*dH, fam, band, met, "A", "HOLD")], replicate=[dict(family=fam, band=band, metric=met, arm="A", delta=repX), dict(family=fam, band=band, metric=met, arm="B", delta=repB)])


def test_floor_k_is_max_of_inference_and_readout_training_floors_and_b_point_counts():
    bx = lambda d: dict(rows=[_rows(d, -1, 1, "LONGITUDINAL", "0-2s", "acc_mae"), _rows(d, -1, 1, "LATERAL", "0-2s", "xtrack")], replicate=[])
    js76 = _js((-0.03, -0.05, -0.01), (0.0, 0, 0), 0.004, 0.003)
    js76["rows"] += [_rows(0.0, 0.0, 0.0, "LATERAL", "0-2s", "xtrack")]
    js76["replicate"] += [dict(family="LATERAL", band="0-2s", metric="xtrack", arm="A", delta=0.0), dict(family="LATERAL", band="0-2s", metric="xtrack", arm="B", delta=0.0)]
    d = RD7.delta7(js76, bx(0.002))
    assert d["F_I"] == 0.004 and d["F_T"] == 0.002 and d["Floor"] == 0.004 and d["better"]
    d2 = RD7.delta7(js76, bx(0.010))                                      # the readout-training floor now binds, |Delta_7| still beyond it
    assert d2["Floor"] == 0.010 and d2["better"]
    d3 = RD7.delta7(js76, bx(0.035))                                      # F_T larger than |Delta_7|: not beyond the floor (and B's point -0.03 + 0.035 is above 0)
    assert d3["Floor"] == 0.035 and d3["readout_b_point"] > 0 and not d3["better"]
    assert RD7.sep_better7(dict(delta=-0.03, lo=-0.05, hi=-0.01), 0.01, -0.01, +0.002) is False and RD7.sep_better7(dict(delta=-0.03, lo=-0.05, hi=-0.01), 0.01, -0.01, -0.002) is True
    assert RD7.worse7(dict(delta=0.03, lo=0.01, hi=0.05), 0.01) and not RD7.worse7(dict(delta=0.03, lo=-0.01, hi=0.05), 0.01)


def test_outcome7_table_row_by_row():
    ok, no, mix = {"s0": True, "s1": True}, {"s0": False, "s1": False}, {"s0": True, "s1": False}
    assert RD7.outcome7("CEILING", ok, ok, no) == "R7-CEILING"
    assert RD7.outcome7("HARM", ok, ok, no) == "HARM"
    assert RD7.outcome7("NARROWS", ok, ok, no) == "R7-PRECISION"
    assert RD7.outcome7("NARROWS", ok, mix, no) == "R7-SAME"
    assert RD7.outcome7("NARROWS", ok, no, no) == "R7-SAME" and RD7.outcome7("NULL", no, no, no) == "R7-SAME"
    assert RD7.outcome7("NARROWS", ok, no, mix) == "R7-WORSE"
    assert RD7.outcome7("NULL", no, ok, no) == "R7-SAME"                  # Delta_7 better but Delta_B not: not PRECISION
    assert RD7.outcome7("NARROWS", ok, ok, no, invalid=True) == "INVALID"
    assert set(RD7.NEXT) == {"R7-CEILING", "R7-PRECISION", "R7-SAME", "R7-WORSE", "HARM", "INVALID"}


# ---------------------------------------------------------------------------------------------- the chain script
def test_chain_scripts_parse_and_carry_the_sentinels():
    sh = open(HERE + "/run_amend7.sh", encoding="utf-8").read()
    r = subprocess.run(["C:/Program Files/Git/usr/bin/bash.exe", "-n", HERE + "/run_amend7.sh"], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    for s in ("ZZ AMEND7 FAIL ZZ", "ZZ AMEND7 FIT DONE ZZ", "ZZ AMEND7 FIT FAIL ZZ", "ZZ AMEND7 FIT INVALID ZZ", "ZZ AMEND7 READ DONE ZZ", "AMEND_HWMPLAN_7.sha256"):
        assert s in sh, s
    cmd = open(HERE + "/amend7.cmd", encoding="utf-8").read()
    assert "Git\\usr\\bin\\bash.exe" in cmd and "Git\\bin\\bash.exe" not in cmd
    ps = open(HERE + "/launch7.ps1", encoding="utf-8").read()
    assert "0x08000200" in ps and "< NUL" in ps
