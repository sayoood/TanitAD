"""PREREG_V8F_PLANTEST2 controls 1 to 5, each with its deliberate-regression arm. CPU. Run through test_run.cmd (env_plantest2.cmd, PYTHONPATH with ';')."""
import os, sys, types, tempfile
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pytest
import torch
import plantest2_lib as Q
import plantest2_real as R2
import plantest_lib as P
import run_plantest2 as RUN

TOL, FACTOR = 0.05, 0.2                       # IMPL section 8, control 3
LIMIT_TOL = 1e-4                              # IMPL section 8, control 2 (units of the prior std)
IDX = np.arange(0, 1976, 124)[:16]


@pytest.fixture(scope="module")
def ctx():
    return R2.prepare2("M2b_s0")


def _plan2(ctx, kind, g, idx=IDX, lam=0.0, mode="hold", seed=0):
    return Q.plan_all2(ctx.st, kind, g, ctx.z_now, ctx.z_fut, ctx.donor, R2.hold_c_of(ctx), idx=idx, seed=seed, lam=lam, seed_mode=mode)


# --------------------------------------------------------------------------- control 1: reduction
def test_c1_reduction_bit_identical_to_plantest(ctx):
    for kind in ("td3", "flat1"):
        k = "flat" if kind == "flat1" else kind
        a = P.plan_all(ctx.st, k, "true", ctx.z_now, ctx.z_fut, ctx.donor, idx=IDX, last_k=1 if kind == "flat1" else None)
        b = _plan2(ctx, kind, "true", mode="prior")
        assert set(a) == set(b), (kind, set(a) ^ set(b))
        for key in a:
            assert np.array_equal(a[key], b[key]), (kind, key)


def test_c1_mutation_hold_seed_differs(ctx):
    a = P.plan_all(ctx.st, "flat", "true", ctx.z_now, ctx.z_fut, ctx.donor, idx=IDX, last_k=1)["actions"]
    b = _plan2(ctx, "flat1", "true", mode="hold")["actions"]
    assert not np.array_equal(a, b), "the reduction check cannot fail"


# --------------------------------------------------------------------------- control 2: limit
def _dev(ctx, kind):
    r = _plan2(ctx, kind, "true", lam=Q.LAMBDA_LIMIT)
    hc = R2.hold_c_of(ctx)[IDX].numpy().astype(np.float64)
    s = Q.prior_std_orig(ctx.st).numpy().astype(np.float64)
    return float((np.abs(r["actions"] - hc) / s).max())


def test_c2_large_lambda_plan_equals_hold(ctx):
    for kind in ("td3", "flat1"):
        d = _dev(ctx, kind)
        print("limit", kind, "max |plan - HOLD_c| / prior std", d)
        assert d <= LIMIT_TOL, (kind, d)


def test_c2_mutation_reg_off_goes_red(ctx, monkeypatch):
    monkeypatch.setattr(Q, "reg_term", lambda a, ref, std, lam: torch.zeros(a.shape[:2], dtype=a.dtype))
    d = _dev(ctx, "flat1")
    print("limit mutant (REG off) flat1 max dev", d)
    assert d > LIMIT_TOL, "the limit check cannot fail"


# --------------------------------------------------------------------------- control 3: planted dynamics through SEED / REG
@pytest.fixture(scope="module")
def planted():
    st, _ = P.planted_stack(); P.planted_fit_priors(st)
    st_bad, _ = P.planted_stack(disconnect=True); P.planted_fit_priors(st_bad)
    return st, st_bad


def _planted(st, kind, lam, n=64, seed=777, std=0.5):
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(n, st.cfg[0].latent_dim, generator=g)
    a = P.planted_actions(n, seed + 1, std)
    zT = P.planted_roll(st, z0, a)
    goal = st.initial_latents(zT, 3)[2] if kind == "td3" else zT
    hold_c = torch.max(torch.min(a[:, :1].repeat(1, P.H1, 1), st.prior_hi_1), st.prior_lo_1)         # HOLD of the planted world = the first planted action held
    res = Q.run_planner2(st, kind, z0, goal, P.make_gen(0, 0), hold_c=hold_c, lam=lam, seed_mode="hold")
    mae = (res.actions.detach() - a).abs().mean().item()
    base = (st.prior_mean_1.expand(n, P.H1, 2) - a).abs().mean().item()
    return mae, base


def test_c3_planted_recovery_with_seed_and_reg(planted):
    st, _ = planted
    for kind in ("td3", "flat1"):
        for lam in (0.0, 0.01):
            mae, base = _planted(st, kind, lam)
            print("planted", kind, "lambda", lam, "MAE", mae, "baseline", base)
            assert mae <= TOL and mae <= FACTOR * base, (kind, lam, mae, base)


def test_c3_mutation_actions_disconnected_goes_red(planted):
    _, st_bad = planted
    for kind in ("td3", "flat1"):
        for lam in (0.0, 0.01):
            mae, base = _planted(st_bad, kind, lam)
            print("DISCONNECTED", kind, "lambda", lam, "MAE", mae, "baseline", base)
            assert not (mae <= TOL), "the regression arm passed the planted control: the control cannot fail"


# --------------------------------------------------------------------------- control 4: split
def test_c4_split_disjoint_stable_and_order_free(ctx):
    it, isc = ctx.tune_idx, ctx.score_idx
    e = np.asarray(ctx.eid)
    assert set(e[it]).isdisjoint(set(e[isc])) and len(np.intersect1d(it, isc)) == 0 and len(it) + len(isc) == len(e)
    assert len(set(e[it])) == 70 and len(set(e[isc])) == 71, (len(set(e[it])), len(set(e[isc])))
    perm = np.random.default_rng(1).permutation(len(e))
    t2, s2 = Q.split_episodes(e[perm])
    t1, s1 = Q.split_episodes(e)
    assert t1 == t2 and s1 == s2
    assert Q.split_episodes(e, salt="other")[0] != t1                       # the salt matters
    print("split: TUNE windows", len(it), "episodes", len(set(e[it])), "; SCORE windows", len(isc), "episodes", len(set(e[isc])))


def _ctx_without_score(ctx):
    c2 = types.SimpleNamespace(**vars(ctx))
    c2.z_now, c2.z_fut = ctx.z_now.clone(), ctx.z_fut.clone()
    c2.A = ctx.A.copy()
    sc = ctx.score_idx
    c2.z_now[sc] = float("nan"); c2.z_fut[sc] = float("nan")
    c2.A[sc] = np.random.default_rng(0).normal(size=c2.A[sc].shape) * 5.0
    return c2


def test_c4_tuning_never_reads_score(ctx, monkeypatch):
    tune16 = ctx.tune_idx[:16]
    base, _ = R2.tune_lambdas(ctx, "flat1", (0.0, 0.1), tune16)
    c2 = _ctx_without_score(ctx)
    again, _ = R2.tune_lambdas(c2, "flat1", (0.0, 0.1), tune16)
    assert base == again and all(np.isfinite(v["acc_mae"]) for v in base.values())
    assert R2.choose_all(base) == R2.choose_all(again)
    # the driver's tune branch hands the planner only TUNE indices
    seen = []
    real = R2.unit2

    def rec(c, kind, gname, idx, lam, **kw):
        seen.append(np.asarray(idx).copy())
        return real(c, kind, gname, idx, lam, **kw)

    monkeypatch.setattr(R2, "prepare2", lambda tag: ctx)
    monkeypatch.setattr(R2, "unit2", rec)
    monkeypatch.setattr(RUN, "RUNS", tempfile.mkdtemp(prefix="pt2test_", dir="F:/TanitAD-scratch"))
    RUN._CTX.clear()
    job, dt, st = RUN.work(("tune", "M2b_s0", "flat1", 0.1))
    RUN._CTX.clear()
    assert st == "ok", st
    assert len(seen) == 1 and set(seen[0]) <= set(ctx.tune_idx) and set(seen[0]).isdisjoint(set(ctx.score_idx))
    assert len(seen[0]) == len(ctx.tune_idx)


def test_c4_mutation_tuner_that_reads_score_goes_red(ctx):
    c2 = _ctx_without_score(ctx)
    mixed = np.concatenate([ctx.tune_idx[:8], ctx.score_idx[:8]])
    bad, _ = R2.tune_lambdas(c2, "flat1", (0.0,), mixed)
    good, _ = R2.tune_lambdas(ctx, "flat1", (0.0,), mixed)
    assert set(mixed) & set(ctx.score_idx), "the mutant must read SCORE windows"
    assert not (np.isfinite(bad[0.0]["acc_mae"]) and bad == good), "the no-read check cannot fail"


# --------------------------------------------------------------------------- control 5: goal plumbing, bit-repeat, a0-only
def _direct(ctx, kind, gname, idx, lam=0.1, seed=0):
    k = "flat" if kind == "flat1" else kind
    lv = 3 if k == "td3" else 1
    src = {"true": ctx.z_fut[idx], "rand": ctx.z_fut[torch.as_tensor(ctx.donor[idx])], "stay": ctx.z_now[idx]}[gname]
    with torch.no_grad():
        g = ctx.st.initial_latents(src, lv)[lv - 1]
    return Q.run_planner2(ctx.st, kind, ctx.z_now[idx], g, P.make_gen(seed, 0), hold_c=R2.hold_c_of(ctx)[idx], lam=lam)


def test_c5_goal_plumbing_swap(ctx):
    for kind in ("td3", "flat1"):
        outs = {}
        for g in ("true", "rand"):
            r = _plan2(ctx, kind, g, lam=0.1)
            d = _direct(ctx, kind, g, IDX)
            assert np.array_equal(r["actions"], d.actions.detach().numpy().astype(np.float64)), (kind, g)
            outs[g] = r["actions"]
        assert not np.array_equal(outs["true"], outs["rand"])


def test_c5_mutation_goal_ignored_goes_red(ctx, monkeypatch):
    real = P.goal_for
    monkeypatch.setattr(P, "goal_for", lambda st, kind, gname, zf, zn, zfd: real(st, kind, "true", zf, zn, zfd))
    r = _plan2(ctx, "flat1", "rand", lam=0.1)
    monkeypatch.undo()
    d = _direct(ctx, "flat1", "rand", IDX)
    assert not np.array_equal(r["actions"], d.actions.detach().numpy().astype(np.float64)), "the swap check cannot fail"


def test_c5_bit_repeat_and_seed_mutation(ctx):
    for kind in ("td3", "flat1"):
        a, b = _plan2(ctx, kind, "true", lam=0.1), _plan2(ctx, kind, "true", lam=0.1)
        for k in a:
            assert np.array_equal(a[k], b[k]), (kind, k)
    c = _plan2(ctx, "flat1", "true", lam=0.1, seed=1)
    assert not np.array_equal(_plan2(ctx, "flat1", "true", lam=0.1)["actions"], c["actions"]), "the repeat check cannot fail"


def test_c5_only_a0_reaches_the_planner(ctx):
    base = _plan2(ctx, "flat1", "true", lam=0.1)["actions"]
    saved = ctx.A.copy()
    ctx.A[:, 1:] = np.random.default_rng(0).normal(size=ctx.A[:, 1:].shape) * 5.0              # garbage for actions 1..end
    same = _plan2(ctx, "flat1", "true", lam=0.1)["actions"]
    ctx.A[:, 0] = saved[:, 0] + 0.7                                                          # a0 itself changed
    moved = _plan2(ctx, "flat1", "true", lam=0.1)["actions"]
    ctx.A[:] = saved
    assert np.array_equal(base, same)
    assert not np.array_equal(base, moved), "a0 is the one logged value the planner is meant to read"


def test_c5_mutation_planner_seeded_with_all_logged_actions_goes_red(ctx):
    idx = IDX
    clean = _plan2(ctx, "flat1", "true", lam=0.0)["actions"]
    A_sc = ctx.wm.scale_actions(torch.as_tensor(ctx.A[idx][:, :12]).float())
    with torch.no_grad():
        g = ctx.st.initial_latents(ctx.z_fut[idx], 1)[0]
    leaky = Q.plan_topdown2(ctx.st, ctx.z_now[idx], g, top_horizon=12, goal_level=1, num_candidates=P.S_CAND, n_iters=P.N_ITERS, lr=P.LR, lam=0.0, subgoal_K=None,
                            patience=P.PATIENCE, rel_tol=P.REL_TOL, last_k=1, generator=P.make_gen(0, 0), seeds={1: A_sc}).actions.detach().numpy()
    assert not np.array_equal(clean, leaky)
    L = ctx.A[idx][:, :12]
    assert np.abs(R2.R.to_raw(ctx, leaky) - L).mean() < np.abs(R2.R.to_raw(ctx, clean) - L).mean()
