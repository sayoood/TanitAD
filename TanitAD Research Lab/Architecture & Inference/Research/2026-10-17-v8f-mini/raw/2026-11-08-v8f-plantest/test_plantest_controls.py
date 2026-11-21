"""PREREG_V8F_PLANTEST controls 1 to 4, each with its mutation going red. CPU. Run: pytest -q test_plantest_controls.py  (env_plantest.cmd / PYTHONPATH with ';' set; tanitad.__file__ asserted under the export)."""
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pytest
import torch
import plantest_lib as P
import plantest_real as R

TOL, FACTOR = 0.05, 0.2                    # IMPL_CHOICES section 6, control 1


# --------------------------------------------------------------------------- control 1: planted dynamics
@pytest.fixture(scope="module")
def planted():
    st, _ = P.planted_stack(); P.planted_fit_priors(st)
    st_bad, _ = P.planted_stack(disconnect=True); P.planted_fit_priors(st_bad)
    return st, st_bad


def test_c1_planted_recovery_passes(planted):
    st, _ = planted
    for kind, lk in (("td3", None), ("flat", 1)):
        mae, base, _, _ = P.planted_recovery(st, kind, n=64, last_k=lk)
        print("planted", kind, "last_k", lk, "MAE", float(mae.mean()), "baseline", float(base.mean()))
        assert mae.mean() <= TOL and mae.mean() <= FACTOR * base.mean(), (kind, mae.mean(), base.mean())


def test_c1_production_flat_cost_beside(planted):
    st, _ = planted
    mae, base, _, _ = P.planted_recovery(st, "flat", n=64, last_k=None)
    print("planted FLAT production last-2 cost MAE", float(mae.mean()), "baseline", float(base.mean()), "(beside, decides nothing)")


def test_c1_mutation_actions_disconnected_goes_red(planted):
    _, st_bad = planted
    for kind, lk in (("td3", None), ("flat", 1)):
        mae, base, _, _ = P.planted_recovery(st_bad, kind, n=64, last_k=lk)
        print("DISCONNECTED", kind, "MAE", float(mae.mean()), "baseline", float(base.mean()))
        assert not (mae.mean() <= TOL), "the regression arm passed the planted control: the control cannot fail"


# --------------------------------------------------------------------------- controls 2 to 4 on the real pipeline (16 windows, M2b_s0)
@pytest.fixture(scope="module")
def ctx():
    return R.prepare("M2b_s0", n_eval=None)


IDX = np.arange(0, 1976, 124)[:16]         # 16 windows spread over the set


def _direct(ctx, kind, gname, idx, seed=0):
    """The planner called directly on goal arrays built here (not through unit/plan_all's goal selection), one batch of 16."""
    lv = 3 if kind == "td3" else 1
    src = {"true": ctx.z_fut[idx], "rand": ctx.z_fut[torch.as_tensor(ctx.donor[idx])], "stay": ctx.z_now[idx]}[gname]
    with torch.no_grad():
        g = ctx.st.initial_latents(src, lv)[lv - 1]
    return P.run_planner(ctx.st, kind, ctx.z_now[idx], g, P.make_gen(seed, 0))


def test_c2_goal_plumbing_swap(ctx):
    for kind in ("td3", "flat"):
        for g in ("true", "rand"):
            r = P.plan_all(ctx.st, kind, g, ctx.z_now, ctx.z_fut, ctx.donor, idx=IDX)
            d = _direct(ctx, kind, g, IDX)
            assert np.array_equal(r["actions"], d.actions.detach().numpy().astype(np.float64)), (kind, g)
        # swapping the two goal arrays swaps the outputs: plan(true) != plan(rand) and each equals its own direct call (above)
        a = P.plan_all(ctx.st, kind, "true", ctx.z_now, ctx.z_fut, ctx.donor, idx=IDX)["actions"]
        b = P.plan_all(ctx.st, kind, "rand", ctx.z_now, ctx.z_fut, ctx.donor, idx=IDX)["actions"]
        assert not np.array_equal(a, b)


def test_c2_mutation_goal_ignored_goes_red(ctx, monkeypatch):
    real = P.goal_for
    monkeypatch.setattr(P, "goal_for", lambda st, kind, gname, zf, zn, zfd: real(st, kind, "true", zf, zn, zfd))     # the mutant ignores the goal name
    r = P.plan_all(ctx.st, "flat", "rand", ctx.z_now, ctx.z_fut, ctx.donor, idx=IDX)
    d = _direct(ctx, "flat", "rand", IDX)
    assert not np.array_equal(r["actions"], d.actions.detach().numpy().astype(np.float64)), "the swap check cannot fail"


def test_c3_bit_repeat(ctx):
    for kind in ("td3", "flat"):
        a = P.plan_all(ctx.st, kind, "true", ctx.z_now, ctx.z_fut, ctx.donor, idx=IDX)
        b = P.plan_all(ctx.st, kind, "true", ctx.z_now, ctx.z_fut, ctx.donor, idx=IDX)
        for k in a:
            assert np.array_equal(a[k], b[k]), (kind, k)


def test_c3_mutation_other_seed_differs(ctx):
    a = P.plan_all(ctx.st, "flat", "true", ctx.z_now, ctx.z_fut, ctx.donor, idx=IDX, seed=0)["actions"]
    b = P.plan_all(ctx.st, "flat", "true", ctx.z_now, ctx.z_fut, ctx.donor, idx=IDX, seed=1)["actions"]
    assert not np.array_equal(a, b), "the repeat check cannot fail"


def test_c4_no_leak_logged_actions_do_not_reach_the_planner(ctx):
    base = P.plan_all(ctx.st, "td3", "true", ctx.z_now, ctx.z_fut, ctx.donor, idx=IDX)["actions"]
    saved = ctx.A.copy()
    ctx.A[:] = np.random.default_rng(0).normal(size=ctx.A.shape) * 5.0             # garbage logged actions on every eval window
    again = P.plan_all(ctx.st, "td3", "true", ctx.z_now, ctx.z_fut, ctx.donor, idx=IDX)["actions"]
    ctx.A[:] = saved
    assert np.array_equal(base, again)
    import inspect
    assert "A_raw" not in inspect.getsource(P.plan_all) + inspect.getsource(P.run_planner) + inspect.getsource(P.goal_for)


def test_c4_mutation_planner_seeded_with_logged_actions_goes_red(ctx):
    idx = IDX
    clean = _direct(ctx, "flat", "true", idx).actions.detach().numpy()
    A_sc = ctx.wm.scale_actions(torch.as_tensor(ctx.A[idx][:, :12]).float())
    with torch.no_grad():
        g = ctx.st.initial_latents(ctx.z_fut[idx], 1)[0]
    leaky = ctx.st.plan_topdown(ctx.z_now[idx], g, top_horizon=12, goal_level=1, num_candidates=P.S_CAND, n_iters=P.N_ITERS, lr=P.LR, last_k=None,
                                init_actions={1: A_sc}, init_noise=0.05, generator=P.make_gen(0, 0)).actions.detach().numpy()
    assert not np.array_equal(clean, leaky), "the no-leak check cannot fail"
    # and the leak would be visible in the read: the leaky plan is closer to the logged actions than the clean one
    L = ctx.A[idx][:, :12]
    raw = lambda a: R.to_raw(ctx, a)
    assert np.abs(raw(leaky) - L).mean() < np.abs(raw(clean) - L).mean()
