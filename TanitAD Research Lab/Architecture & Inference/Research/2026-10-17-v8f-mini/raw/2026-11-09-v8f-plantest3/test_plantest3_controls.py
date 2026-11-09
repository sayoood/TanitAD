"""PREREG_V8F_PLANTEST3 controls 1 to 6, each with its deliberate-regression arm (IMPL_CHOICES_PLANTEST3.md section 8). CPU. Run through test_run.cmd."""
import os, sys, types, tempfile
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pytest
import torch
import plantest3_lib as L3
import plantest3_real as R3
import plantest2_lib as Q
import plantest2_real as R2
import plantest_lib as P
import run_plantest3 as RUN

TOL, FACTOR = 0.05, 0.2
IDX = np.arange(0, 1976, 124)[:16]


@pytest.fixture(scope="module")
def ctx():
    c = R3.prepare3("M2b_s0")
    R3.get_dec(c)
    return c


def _plan3(ctx, kind, g, cost, idx=IDX, lam=0.0, mode="hold", seed=0):
    return L3.plan_all3(ctx.st, kind, g, ctx.z_now, ctx.z_fut, ctx.z_path, ctx.donor, R2.hold_c_of(ctx), idx=idx, seed=seed, lam=lam, seed_mode=mode, cost=cost)


def _plan2(ctx, kind, g, idx=IDX, lam=0.0, mode="hold", seed=0):
    return Q.plan_all2(ctx.st, kind, g, ctx.z_now, ctx.z_fut, ctx.donor, R2.hold_c_of(ctx), idx=idx, seed=seed, lam=lam, seed_mode=mode)


def _same(a, b, tag):
    assert set(a) == set(b), (tag, set(a) ^ set(b))
    for k in a:
        assert np.array_equal(a[k], b[k]), (tag, k)


# --------------------------------------------------------------------------- control 1: reduction
def test_c1_reduction_bit_identical_to_plantest2(ctx):
    for kind in ("td3", "flat1"):
        for lam in (0.0, 0.1):
            ref = _plan2(ctx, kind, "true", lam=lam)
            _same(ref, _plan3(ctx, kind, "true", None, lam=lam), (kind, lam, "cost=None"))
        ref = _plan2(ctx, kind, "true", lam=0.1)
        _same(ref, _plan3(ctx, kind, "true", L3.CostSpec("latent"), lam=0.1), (kind, "latent spec"))
        _same(ref, _plan3(ctx, kind, "true", L3.CostSpec("latent", c=1.0), lam=0.1), (kind, "latent spec c=1"))
        print("reduction bit-identical:", kind)


def test_c1_mutation_decoded_cost_differs(ctx):
    for kind in ("flat1", "td3"):
        a = _plan2(ctx, kind, "true", lam=0.1)["actions"]
        b = _plan3(ctx, kind, "true", L3.CostSpec("end", None, ctx.dec), lam=0.1)["actions"]
        assert not np.array_equal(a, b), "the reduction check cannot fail"


# --------------------------------------------------------------------------- control 2: planted dynamics + planted decoder
@pytest.fixture(scope="module")
def planted():
    st, (A1, B1) = P.planted_stack(); P.planted_fit_priors(st)
    st_bad, _ = P.planted_stack(disconnect=True); P.planted_fit_priors(st_bad)
    return st, st_bad, L3.planted_decoder(B1)


def _planted(st_truth, st_plan, dec, kind, arm, lam, n=64, seed=777, std=0.5):
    """goal from the CONNECTED world st_truth; the planner uses st_plan (the regression arm plans on a world that ignores the actions)."""
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(n, st_plan.cfg[0].latent_dim, generator=g)
    a = P.planted_actions(n, seed + 1, std)
    with torch.no_grad():
        path = st_truth.rollout(1, z0, a)
    zT = path[:, -1]
    goal = st_plan.initial_latents(zT, 3)[2] if kind == "td3" else (path if arm == "path" else zT)
    hold_c = torch.max(torch.min(a[:, :1].repeat(1, P.H1, 1), st_plan.prior_hi_1), st_plan.prior_lo_1)
    res = L3.run_planner3(st_plan, kind, z0, goal, P.make_gen(0, 0), hold_c=hold_c, lam=lam, seed_mode="hold", cost=L3.CostSpec(arm, None, dec))
    rec = res.actions.detach()
    if arm == "path":
        err = (rec - a).abs().mean().item()
        base = (st_plan.prior_mean_1.expand(n, P.H1, 2) - a).abs().mean().item()
    else:
        with torch.no_grad():
            endr = st_plan.rollout(1, z0, rec)[:, -1]
            endb = st_plan.rollout(1, z0, st_plan.prior_mean_1.expand(n, P.H1, 2).contiguous())[:, -1]
        err = ((endr - zT) @ dec.W1).abs().mean().item()
        base = ((endb - zT) @ dec.W1).abs().mean().item()
    return err, base


CASES = (("flat1", "path"), ("flat1", "end"), ("td3", "end"))


def test_c2_planted_recovery(planted):
    st, _, dec = planted
    for kind, arm in CASES:
        for lam in (0.0, 0.01):
            err, base = _planted(st, st, dec, kind, arm, lam)
            print("planted", kind, arm, "lambda", lam, "error", err, "baseline", base)
            assert err <= TOL and err <= FACTOR * base, (kind, arm, lam, err, base)


def test_c2_mutation_actions_disconnected_goes_red(planted):
    st, st_bad, dec = planted
    for kind, arm in CASES:
        for lam in (0.0, 0.01):
            err, base = _planted(st, st_bad, dec, kind, arm, lam)
            print("DISCONNECTED", kind, arm, "lambda", lam, "error", err, "baseline", base)
            assert not (err <= TOL and err <= FACTOR * base), "the regression arm passed the planted control: the control cannot fail"


# --------------------------------------------------------------------------- control 3: decoder
def _synth(m=400, T=16, D=128, noise=0.01, seed=0):
    g = np.random.default_rng(seed)
    Z = torch.tensor(g.normal(size=(m, T, D)), dtype=torch.float32)
    w = g.normal(size=(D, 2)) * 0.3
    Y = Z.reshape(-1, D).double().numpy() @ w
    Y = Y + noise * g.normal(size=Y.shape)
    return Z, Y[:, 0].reshape(m, T) * 2.0 + 5.0, Y[:, 1].reshape(m, T) * 0.1, w


def _recovered(dec, w, r2s):
    rel = np.abs(dec.W1.numpy().astype(np.float64) * dec.ystd[None, :] - w * np.array([2.0, 0.1])[None, :]).max() / np.abs(w * np.array([2.0, 0.1])[None, :]).max()
    return bool(min(r2s[0], r2s[1]) >= 0.99 and rel <= 0.05), rel


def test_c3_decoder_recovers_known_map_and_shuffled_reads_zero():
    Z, sp, yw, w = _synth()
    a, b = slice(0, 300), slice(300, 400)
    dec = L3.fit_decoder(None, Z[a], sp[a], yw[a], with_l2=False)
    r = L3.heldout_r2(None, dec, 1, Z[b], sp[b], yw[b])
    ok, rel = _recovered(dec, w, r)
    print("decoder synthetic: held-out R2", r[:2], "max rel weight err", rel)
    assert ok
    dsh, rsh = L3.shuffled_r2(None, Z, sp, yw)
    print("decoder shuffled targets: held-out R2", rsh[:2])
    assert max(rsh[0], rsh[1]) <= 0.02


def test_c3_mutation_shuffled_fit_fails_the_recovery_bar():
    Z, sp, yw, w = _synth()
    dsh, rsh = L3.shuffled_r2(None, Z, sp, yw)
    ok, _ = _recovered(dsh, w, rsh)
    assert not ok, "the recovery bar cannot fail"


# --------------------------------------------------------------------------- control 4: split
def _ctx_garbage_score(ctx):
    c2 = types.SimpleNamespace(**vars(ctx))
    rng = np.random.default_rng(0)
    sc = ctx.score_idx
    Z1 = ctx.Z1.clone()
    Z1[sc] = torch.tensor(rng.normal(size=Z1[sc].shape) * 3.0, dtype=torch.float32)
    c2.Z1 = Z1
    c2.z_now, c2.z_fut, c2.z_path = Z1[:, 0], Z1[:, 12], Z1[:, 1:13].contiguous()
    c2.vk = ctx.vk.copy(); c2.vk[sc] = rng.normal(size=c2.vk[sc].shape) * 7.0 + 20.0
    c2.yk = ctx.yk.copy(); c2.yk[sc] = rng.normal(size=c2.yk[sc].shape) * 5.0
    c2.A = ctx.A.copy(); c2.A[sc] = rng.normal(size=c2.A[sc].shape) * 5.0
    c2.dec = None
    return c2


def test_c4_split_and_decoder_never_read_score(ctx):
    e = np.asarray(ctx.eid)
    assert set(e[ctx.tune_idx]).isdisjoint(set(e[ctx.score_idx])) and len(set(e[ctx.tune_idx])) == 70 and len(set(e[ctx.score_idx])) == 71
    c0 = types.SimpleNamespace(**vars(ctx)); c0.dec = None
    c2 = _ctx_garbage_score(ctx)
    R3.FIT_LOG.clear()
    d0, d2 = R3.get_dec(c0), R3.get_dec(c2)
    assert all(set(r) <= set(ctx.tune_idx) and set(r).isdisjoint(set(ctx.score_idx)) for r in R3.FIT_LOG), "the fitter was handed a SCORE row"
    assert d0.digest() == d2.digest() and torch.equal(d0.W2, d2.W2) and d0.n_samples == d2.n_samples
    tune16 = ctx.tune_idx[:16]
    t0 = R3.tune_lambdas3(c0, "flat1", "end", "plain", (0.0, 1.0), tune16)
    t2b = R3.tune_lambdas3(c2, "flat1", "end", "plain", (0.0, 1.0), tune16)
    assert t0 == t2b and all(np.isfinite(v["acc_mae"]) for v in t0.values()) and R2.choose_all(t0) == R2.choose_all(t2b)


def test_c4_driver_tune_branch_passes_only_tune_rows(ctx, monkeypatch):
    seen = []
    real = R3.unit3

    def rec(c, kind, gname, idx, lam, arm, form, **kw):
        seen.append(np.asarray(idx).copy())
        return real(c, kind, gname, idx, lam, arm, form, **kw)
    monkeypatch.setattr(R3, "prepare3", lambda tag: ctx)
    monkeypatch.setattr(R3, "unit3", rec)
    monkeypatch.setattr(RUN, "RUNS", tempfile.mkdtemp(prefix="pt3test_", dir="F:/TanitAD-scratch/pt3"))
    monkeypatch.setattr(RUN, "wait_ram", lambda *a, **k: None)
    sub = ctx.tune_idx[:16]
    monkeypatch.setattr(ctx, "tune_idx", sub)
    RUN._CTX.clear()
    job, dt, st = RUN.work(("tune", "M2b_s0", "end", "plain", 0.0))
    RUN._CTX.clear()
    assert st == "ok", st
    assert len(seen) == 1 and set(seen[0]) <= set(sub) and set(seen[0]).isdisjoint(set(ctx.score_idx))


def test_c4_mutation_fit_over_all_windows_goes_red(ctx):
    c2 = _ctx_garbage_score(ctx)
    good = R3.get_dec(types.SimpleNamespace(**{**vars(ctx), "dec": None}))
    bad = R3.fit_rows(c2, np.arange(ctx.n))
    assert bad.digest() != good.digest(), "the no-read check cannot fail"
    mixed = np.concatenate([ctx.tune_idx[:8], ctx.score_idx[:8]])
    garbage_mae = R3.tune_lambdas3(c2, "flat1", "end", "plain", (0.0,), mixed)
    clean_mae = R3.tune_lambdas3(ctx, "flat1", "end", "plain", (0.0,), mixed)
    assert garbage_mae != clean_mae, "a tuner reading SCORE rows must differ"


# --------------------------------------------------------------------------- control 5: shrink identity
def _grad_max(ctx, spec, idx=IDX, S=3):
    st = ctx.st
    z0 = ctx.z_now[idx]
    g = torch.Generator().manual_seed(3)
    a = (st.prior_mean_1 + st.prior_std_1 * torch.randn(len(idx), S, P.H1, 2, generator=g)).clamp(st.prior_lo_1, st.prior_hi_1).requires_grad_(True)
    z0e = z0.unsqueeze(1).expand(len(idx), S, -1)
    with torch.enable_grad():
        roll = spec.roll(st, 1, z0e, a)
        cost = spec.top_cost(st, roll, ctx.z_fut[idx], 1, 1).sum()
        gr = torch.autograd.grad(cost, a)[0]
    return roll.detach(), float(gr.abs().max())


def test_c5_shrink_identity_and_zero(ctx):
    st = ctx.st
    z0 = ctx.z_now[IDX]
    g = torch.Generator().manual_seed(3)
    a = (st.prior_mean_1 + st.prior_std_1 * torch.randn(len(IDX), 3, P.H1, 2, generator=g)).clamp(st.prior_lo_1, st.prior_hi_1)
    z0e = z0.unsqueeze(1).expand(len(IDX), 3, -1)
    with torch.no_grad():
        plain = st.rollout(1, z0e, a)
        one = L3.CostSpec("end", 1.0, ctx.dec).roll(st, 1, z0e, a)
    assert torch.equal(plain, one)
    r0, g0 = _grad_max(ctx, L3.CostSpec("end", 0.0, ctx.dec))
    assert torch.equal(r0, z0e.unsqueeze(2).expand(-1, -1, P.H1, -1)) and g0 == 0.0, g0
    _, g5 = _grad_max(ctx, L3.CostSpec("end", 0.5, ctx.dec))
    _, g1 = _grad_max(ctx, L3.CostSpec("end", 1.0, ctx.dec))
    print("decoded-cost gradient max abs: c=0", g0, "c=0.5", g5, "c=1", g1)
    assert g5 > 0.0 and g1 > 0.0
    p_none = _plan3(ctx, "flat1", "true", L3.CostSpec("end", None, ctx.dec), lam=0.1)
    p_one = _plan3(ctx, "flat1", "true", L3.CostSpec("end", 1.0, ctx.dec), lam=0.1)
    _same(p_none, p_one, "plan c=1 vs plain")


def test_c5_mutations_go_red(ctx, monkeypatch):
    p_none = _plan3(ctx, "flat1", "true", L3.CostSpec("end", None, ctx.dec), lam=0.1)["actions"]
    p_999 = _plan3(ctx, "flat1", "true", L3.CostSpec("end", 0.999, ctx.dec), lam=0.1)["actions"]
    assert not np.array_equal(p_none, p_999), "the identity check cannot fail"
    p_half = _plan3(ctx, "flat1", "true", L3.CostSpec("end", 0.5, ctx.dec), lam=0.1)["actions"]
    assert not np.array_equal(p_none, p_half)
    monkeypatch.setattr(L3, "shrunk_rollout", lambda st, level, z0, actions, c: st.rollout(level, z0, actions))       # a shrink that ignores c
    r0, g0 = _grad_max(ctx, L3.CostSpec("end", 0.0, ctx.dec))
    print("shrink-ignores-c mutant: c=0 gradient", g0)
    assert g0 > 0.0, "the c = 0 check cannot fail"


# --------------------------------------------------------------------------- control 6: goal plumbing, bit-repeat, a0-only on the new cost
def _direct(ctx, kind, arm, gname, idx, lam=0.1, seed=0):
    cost = L3.CostSpec(arm, None, ctx.dec)
    zf, zn = ctx.z_fut[idx], ctx.z_now[idx]
    d = torch.as_tensor(ctx.donor[idx])
    g = L3.goal3(ctx.st, kind, gname, cost, zf, zn, ctx.z_fut[d], ctx.z_path[idx], ctx.z_path[d])
    return L3.run_planner3(ctx.st, kind, zn, g, P.make_gen(seed, 0), hold_c=R2.hold_c_of(ctx)[idx], lam=lam, cost=cost)


def test_c6_goal_plumbing_swap(ctx):
    for kind, arm in CASES:
        outs = {}
        for g in ("true", "rand"):
            r = _plan3(ctx, kind, g, L3.CostSpec(arm, None, ctx.dec), lam=0.1)
            d = _direct(ctx, kind, arm, g, IDX)
            assert np.array_equal(r["actions"], d.actions.detach().numpy().astype(np.float64)), (kind, arm, g)
            outs[g] = r["actions"]
        assert not np.array_equal(outs["true"], outs["rand"]), (kind, arm)
        print("goal plumbing ok:", kind, arm)


def test_c6_mutation_goal_ignored_goes_red(ctx, monkeypatch):
    real = L3.goal3
    monkeypatch.setattr(L3, "goal3", lambda st, kind, gname, *a, **k: real(st, kind, "true", *a, **k))
    r = _plan3(ctx, "flat1", "rand", L3.CostSpec("path", None, ctx.dec), lam=0.1)
    monkeypatch.undo()
    d = _direct(ctx, "flat1", "path", "rand", IDX)
    assert not np.array_equal(r["actions"], d.actions.detach().numpy().astype(np.float64)), "the swap check cannot fail"


def test_c6_bit_repeat_and_seed_mutation(ctx):
    for kind, arm in CASES:
        c = L3.CostSpec(arm, None, ctx.dec)
        a, b = _plan3(ctx, kind, "true", c, lam=0.1), _plan3(ctx, kind, "true", c, lam=0.1)
        _same(a, b, (kind, arm, "repeat"))
    c = L3.CostSpec("end", None, ctx.dec)
    d = _plan3(ctx, "flat1", "true", c, lam=0.1, seed=1)
    assert not np.array_equal(_plan3(ctx, "flat1", "true", c, lam=0.1)["actions"], d["actions"]), "the repeat check cannot fail"


def test_c6_only_a0_reaches_the_planner(ctx):
    c = L3.CostSpec("path", None, ctx.dec)
    base = _plan3(ctx, "flat1", "true", c, lam=0.1)["actions"]
    saved = ctx.A.copy()
    ctx.A[:, 1:] = np.random.default_rng(0).normal(size=ctx.A[:, 1:].shape) * 5.0
    same = _plan3(ctx, "flat1", "true", c, lam=0.1)["actions"]
    ctx.A[:, 0] = saved[:, 0] + 0.7
    moved = _plan3(ctx, "flat1", "true", c, lam=0.1)["actions"]
    ctx.A[:] = saved
    assert np.array_equal(base, same)
    assert not np.array_equal(base, moved), "a0 is the one logged value the planner is meant to read"


def test_c6_mutation_planner_seeded_with_all_logged_actions_goes_red(ctx):
    idx = IDX
    c = L3.CostSpec("end", None, ctx.dec)
    clean = _plan3(ctx, "flat1", "true", c, lam=0.0)["actions"]
    A_sc = ctx.wm.scale_actions(torch.as_tensor(ctx.A[idx][:, :12]).float())
    leaky = L3.plan_topdown3(ctx.st, ctx.z_now[idx], ctx.z_fut[idx], top_horizon=12, goal_level=1, num_candidates=P.S_CAND, n_iters=P.N_ITERS, lr=P.LR, lam=0.0, subgoal_K=None,
                             patience=P.PATIENCE, rel_tol=P.REL_TOL, last_k=1, generator=P.make_gen(0, 0), seeds={1: A_sc}, cost=c).actions.detach().numpy()
    assert not np.array_equal(clean, leaky)
    Lg = ctx.A[idx][:, :12]
    assert np.abs(R2.R.to_raw(ctx, leaky) - Lg).mean() < np.abs(R2.R.to_raw(ctx, clean) - Lg).mean()
