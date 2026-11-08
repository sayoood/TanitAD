"""HWM-PLAN harness controls, each with a deliberate-regression arm that must go red. CPU. Run through test_run.cmd (env_hwm.cmd, PYTHONPATH with ';').
(a) reduction to PLANTEST, (b) planted dynamics, (c) goal plumbing and the sub-goal hand-off, (d) band stitching, (e) bit-repeat and seed sensitivity, plus (f) no-leak of the logged actions,
(g) kinematic model and band metrics, (h) geometry read from the BUILT model, (i) admissibility of the goal path, (j) the report. Tier T0 for anything that touches a true-future goal; the toy worlds are synthetic."""
import os, sys, types
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pytest
import torch
import torch.nn as nn
import hwm_plan as H
import hwm_eval as E
import hwm_data as D
import plantest_lib as P
import plantest2_lib as Q

TOL, FACTOR = 0.05, 0.2            # planted recovery: MAE <= 0.05 (10 % of the planted std) and <= 0.2 x the prior-mean baseline (as PLANTEST control 1)
IDX = np.arange(0, 1976, 124)[:16]


# ------------------------------------------------------------------------------------------ toy world with a known action effect
class _Cat(nn.Module):
    """Macro encoder of the toy: concatenation of the lower-level action embeddings (the planted dynamics act on exactly this)."""
    def __init__(self, stride, e):
        super().__init__(); self.stride, self.embed_dim = stride, e

    def forward(self, e):
        return e.reshape(*e.shape[:-2], self.stride * e.shape[-1])


class _Shim:
    def __init__(self, st):
        self.stack, self.ch1_scale = st, 1.0

    def scale_actions(self, a):
        return a


def toy(disconnect=False):
    st, _ = P.planted_stack(disconnect=disconnect)
    st.macro_encoders[0] = _Cat(2, 2); st.macro_encoders[1] = _Cat(2, 4)
    P.planted_fit_priors(st)
    return _Shim(st)


TOY_GEOM = H.Geometry(2, 2, 3, 3, 3, 12, 6).check()      # level 1 plans 6 steps (1.2 s), level 2 plans 6 macro steps, the goal lies 12 steps ahead; knots 1-6 from L1, 7-12 from L2


def toy_world(n=64, seed=777):
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(n, 32, generator=g)
    a = P.planted_actions(n, seed + 1)
    return z0, a


def toy_plan(wm, geom, z0, goal, mode="hier", seed=0, solver=None, planner_cls=H.HWMPlanner, **kw):
    pl = planner_cls(wm, geom, solver or H.SolverCfg.plantest(), **kw)
    return pl.plan_windows(mode, lambda ii: goal[torch.as_tensor(ii)], z0, np.zeros((z0.shape[0], 2)), seed=seed)


def planted_mae(wm, geom=TOY_GEOM, **kw):
    z0, a = toy_world()
    zT = P.planted_roll(wm.stack, z0, a)
    r = toy_plan(wm, geom, z0, zT, **kw)
    pr = r["plan_raw"].copy(); pr[..., 0] /= H.ACC_SCALE                       # the toy has scale 1; to_raw doubles channel 0
    err = np.abs(pr - a.numpy())
    base = (wm.stack.prior_mean_1.expand(a.shape[0], 12, 2) - a).abs().mean((1, 2)).numpy()
    return err, base, r


# ------------------------------------------------------------------------------------------ random-init ladder model (3,5), built, not trained
@pytest.fixture(scope="module")
def ladder():
    from tanitad.models import v8f_mini as M
    ck = torch.load(D.CK["M2b_s0"], map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"]); cfg["wm_strides"] = (3, 5); cfg["hmax"] = 91
    torch.manual_seed(5)
    m = M.V8FMini(M.V8FMiniConfig(**cfg), M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30))).eval()
    Z = torch.randn(64, 31, 128) * 0.5
    A = torch.randn(64, 31, 2)
    D.fit_priors(m.wm.stack, m.wm, Z, A)
    return types.SimpleNamespace(m=m, wm=m.wm, st=m.wm.stack, Z=Z, A=A)


FAST = dict(S=4, n_iters=3, decode_iters=40)


@pytest.fixture(scope="module")
def real():
    """M2b s0 (2,2), PLANTEST's windows and TRAIN prior (zero training; priors fitted on TRAIN windows only)."""
    m, cfg, ck = D.load_model(D.CK["M2b_s0"])
    st = m.wm.stack
    trZ, trA = D.trainset_from_latents_pt(D.TRAIN16, "M2b_s0")
    D.fit_priors(st, m.wm, trZ, trA)
    ev = P.load_latents(D.EVAL16, "M2b_s0")
    return types.SimpleNamespace(m=m, wm=m.wm, st=st, z_now=ev["Z1"][:, 0], z_fut=ev["Z1"][:, 12], A=ev["A"].numpy().astype(np.float64), eid=ev["eid"])


# ------------------------------------------------------------------------------------------ (a) reduction to PLANTEST
def _plantest_plans(real, kind, last_k=None):
    donor = np.arange(len(real.z_now))
    return P.plan_all(real.st, kind, "true", real.z_now, real.z_fut, donor, idx=IDX, last_k=last_k)


def _hwm_plans(real, mode, solver):
    geom = H.Geometry.plantest(real.st)
    pl = H.HWMPlanner(real.wm, geom, solver)
    return pl.plan_windows(mode, lambda ii: real.z_fut[torch.as_tensor(ii)], real.z_now, real.A[:, 0], idx=IDX)


def test_a_reduction_bit_identical_to_plantest(real):
    """HWM-PLAN on PLANTEST's geometry (top plan 3 steps, K = 3 and 6, level 1 plans all 12 knots) with PLANTEST's solver values and initialisation equals PLANTEST's TD3 plan, and the flat arm equals FLAT / FLAT1."""
    for mode, kind, lk in (("hier", "td3", None), ("flat", "flat", None), ("flat", "flat", 1)):
        a = _plantest_plans(real, kind, lk)
        b = _hwm_plans(real, mode, H.SolverCfg(seed_mode="prior", last_k=lk))
        assert np.array_equal(b["plan_raw"], H.to_raw(real.wm, a["actions"])), (mode, lk)
        assert np.array_equal(b["cost1"], a["cost1"]), (mode, lk)
        assert b["plan_raw"].shape == (len(IDX), 12, 2)


def test_a_mutations_each_go_red(real):
    """The equality above cannot pass by construction: a different seed rule or a moved solver value breaks it."""
    a = _plantest_plans(real, "td3")
    for solver in (H.SolverCfg(seed_mode="hold"), H.SolverCfg(seed_mode="prior", lam_sub=0.4), H.SolverCfg(seed_mode="prior", S=8)):
        b = _hwm_plans(real, "hier", solver)
        assert not np.array_equal(b["plan_raw"], H.to_raw(real.wm, a["actions"])), solver


# ------------------------------------------------------------------------------------------ (b) planted dynamics
def test_b_three_level_plan_recovers_planted_actions():
    wm = toy()
    err, base, r = planted_mae(wm)
    assert err.mean() <= TOL and err.mean() <= FACTOR * base.mean(), (err.mean(), base.mean())
    assert err[:, :6].mean() <= TOL and err[:, 6:].mean() <= TOL, "both the level-1 knots and the decoded level-2 knots must recover"
    assert r["decode_resid"].max() < 1e-2, r["decode_resid"].max()
    open(os.path.join(HERE, "RESULT_controls_numbers.txt"), "a").write(f"b planted hier MAE {err.mean():.4f} L1-part {err[:, :6].mean():.4f} L2-part {err[:, 6:].mean():.4f} prior-baseline {base.mean():.4f} decode_resid_max {r['decode_resid'].max():.2e}\n")


def test_b_mutation_disconnected_actions_fail():
    wm = toy(disconnect=True)
    err, base, _ = planted_mae(wm)
    assert err.mean() > TOL, err.mean()


def test_b_mutation_decode_removed_fails_the_l2_knots():
    wm = toy()
    z0, a = toy_world(32)
    zT = P.planted_roll(wm.stack, z0, a)
    run = lambda: H.HWMPlanner(wm, TOY_GEOM, H.SolverCfg.plantest()).plan_windows("hier", lambda ii: zT[torch.as_tensor(ii)], z0, np.zeros((32, 2)))["plan_raw"]
    good = run()
    orig = H.decode_macro
    try:
        H.decode_macro = lambda st, macro, ref, lo, hi, std, cfg: (ref[:, :macro.shape[1] * 2].clone(), torch.zeros(macro.shape[0]))
        bad = run()
    finally:
        H.decode_macro = orig
    assert np.array_equal(good[:, :6], bad[:, :6]) and not np.allclose(good[:, 6:], bad[:, 6:]), "the L2 knots must come from the decode"


# ------------------------------------------------------------------------------------------ (c) goal plumbing and the sub-goal hand-off
def _plans_two_goals(wm, mutate_handoff=False):
    z0, a = toy_world(32)
    zT = P.planted_roll(wm.stack, z0, a)
    zR = zT.roll(7, 0)                                   # the goals of other windows
    st = wm.stack
    orig = st.rollout
    if mutate_handoff:
        def cut(level, z, actions, **kw):
            out = orig(level, z, actions, **kw)
            return out * 0.0 if (level == 3 and not torch.is_grad_enabled()) else out       # the CHOSEN level-3 roll (no-grad) is what level 2 receives as sub-goals
        st.rollout = cut
    try:
        pt = toy_plan(wm, TOY_GEOM, z0, zT)["plan_raw"]
        pr = toy_plan(wm, TOY_GEOM, z0, zR)["plan_raw"]
    finally:
        if mutate_handoff:
            del st.rollout
    return pt, pr


def _goal_reaches_every_band(pt, pr):
    return (not np.allclose(pt[:, :6], pr[:, :6])) and (not np.allclose(pt[:, 6:], pr[:, 6:]))


def test_c_goal_changes_the_plan_in_both_bands():
    pt, pr = _plans_two_goals(toy())
    assert _goal_reaches_every_band(pt, pr)


def test_c_mutation_dropped_subgoal_handoff_goes_red():
    pt, pr = _plans_two_goals(toy(), mutate_handoff=True)
    assert not _goal_reaches_every_band(pt, pr), "with the hand-off cut the goal must not reach the lower levels, so the plumbing check goes red"


def test_c_goal_identity_oracle_vs_shuffled():
    z0, a = toy_world(16)
    zT = P.planted_roll(toy().stack, z0, a)
    donor = np.roll(np.arange(16), 3)
    assert not torch.equal(H.ShuffledGoal(zT, 12, donor)(np.arange(16)), H.OracleGoal(zT, 12)(np.arange(16)))
    assert torch.equal(H.OracleGoal(zT, 12)(np.array([2, 5])), zT[[2, 5]])


# ------------------------------------------------------------------------------------------ (d) band stitching
def _stitch_parts(ladder):
    st, wm = ladder.st, ladder.wm
    solver = H.SolverCfg(**FAST)
    pl = H.HWMPlanner(wm, solver=solver)
    g = pl.geom
    n = 4
    hc = pl.hold_c(ladder.A[:n, 0].numpy().astype(np.float64))
    res = pl.plan_batch("hier", ladder.Z[:n, 0], ladder.Z[:n, 30], hc, P.make_gen(0, 0))
    lo, hi = pl.level1_box(); std = Q.prior_std_orig(st)
    return pl, g, res, hc, lo, hi, std, solver


def test_d_knots_1_10_from_L1_and_11_30_from_L2(ladder):
    pl, g, res, hc, lo, hi, std, solver = _stitch_parts(ladder)
    assert (g.n_op, g.n_traj, g.h1, g.h2) == (10, 30, 12, 10)
    traj, resid = H.stitch(ladder.st, g, res.actions.detach(), res.level_actions[2].detach(), hc, lo, hi, std, solver)
    assert traj.shape == (4, 30, 2)
    assert torch.equal(traj[:, :10], res.actions.detach()[:, :10])                                  # knots 1-10: level 1, exactly
    j0 = 10 // g.c2
    prim, _ = H.decode_macro(ladder.st, res.level_actions[2].detach()[:, j0:10], hc[:, j0 * 3:30], lo, hi, std, solver)
    assert torch.equal(traj[:, 10:], prim[:, 1:21])                                                 # knots 11-30: level 2 decoded (macro steps 3..9, first primitive of step 3 sits inside the L1 band)
    m2 = res.level_actions[2].detach().clone(); m2[:, j0:] += 0.5
    traj2, _ = H.stitch(ladder.st, g, res.actions.detach(), m2, hc, lo, hi, std, solver)
    assert torch.equal(traj2[:, :10], traj[:, :10]) and not torch.equal(traj2[:, 10:], traj[:, 10:])  # the L2 part moves with the level-2 macro-actions and the L1 part does not


def test_d_mutations_deleting_the_L2_part_go_red(ladder):
    pl, g, res, hc, lo, hi, std, solver = _stitch_parts(ladder)
    a1, m2 = res.actions.detach(), res.level_actions[2].detach()
    good, _ = H.stitch(ladder.st, g, a1, m2, hc, lo, hi, std, solver)
    zero_dec = lambda st, macro, ref, lo, hi, std, cfg: (torch.zeros(macro.shape[0], macro.shape[1] * 3, 2), torch.zeros(macro.shape[0]))
    hold_dec = lambda st, macro, ref, lo, hi, std, cfg: (ref[:, :macro.shape[1] * 3].clone(), torch.zeros(macro.shape[0]))
    for dec in (zero_dec, hold_dec):
        bad, _ = H.stitch(ladder.st, g, a1, m2, hc, lo, hi, std, solver, decode=dec)
        assert not torch.equal(bad[:, 10:], good[:, 10:])
    # L1-only stand-in: knots 11-12 from L1, nothing else (what a planner without the tactical level would hold)
    g2 = H.Geometry(g.s2, g.s3, g.K1, g.K2, g.T3, 10, 10)
    short, _ = H.stitch(ladder.st, g2, a1, m2, hc, lo, hi, std, solver)
    assert short.shape[1] == 10 and good.shape[1] == 30


def test_d_decode_roundtrip_fits_the_macro_actions(ladder):
    st, wm = ladder.st, ladder.wm
    a_sc = wm.scale_actions(ladder.A[:8, :30])
    with torch.no_grad():
        enc = st.encode(ladder.Z[:8, :31], a_sc)
    lo, hi = st.prior_lo_1, st.prior_hi_1
    ref = H.hold_full(wm, ladder.A[:8, 0].numpy(), lo, hi, 30)
    p, res = H.decode_macro(st, enc.act[1][:, 3:10], ref[:, 9:30], lo, hi, Q.prior_std_orig(st), H.SolverCfg())
    hold_fit = (ref[:, 9:30] - a_sc[:, 9:30]).abs().mean().item()
    assert res.max() < 0.2 and (p - a_sc[:, 9:30]).abs().mean().item() < 0.2 * hold_fit, (res, hold_fit)


# ------------------------------------------------------------------------------------------ (e) bit-repeat and seed sensitivity
@pytest.mark.parametrize("mode", ["hier", "flat"])
def test_e_bit_repeat_and_seed_changes_the_plan(mode):
    wm = toy()
    z0, a = toy_world(16)
    zT = P.planted_roll(wm.stack, z0, a)
    s0a = toy_plan(wm, TOY_GEOM, z0, zT, mode, seed=0)
    s0b = toy_plan(wm, TOY_GEOM, z0, zT, mode, seed=0)
    s1 = toy_plan(wm, TOY_GEOM, z0, zT, mode, seed=1)
    for k in ("plan_raw", "cost1"):
        assert np.array_equal(s0a[k], s0b[k]), k
    assert not np.array_equal(s0a["plan_raw"], s1["plan_raw"])


def test_e_mutation_planner_ignoring_the_seed_goes_red():
    wm = toy()
    z0, a = toy_world(16)
    zT = P.planted_roll(wm.stack, z0, a)
    fixed = lambda st, *args, generator=None, **kw: Q.plan_topdown2(st, *args, generator=P.make_gen(0, 0), **kw)
    p0 = toy_plan(wm, TOY_GEOM, z0, zT, seed=0, plan_fn=fixed)["plan_raw"]
    p1 = toy_plan(wm, TOY_GEOM, z0, zT, seed=1, plan_fn=fixed)["plan_raw"]
    assert np.array_equal(p0, p1), "the mutant ignores the inference seed, so the seed-sensitivity check of test_e would fail on it"


# ------------------------------------------------------------------------------------------ (f) no-leak of the logged actions
def _ws_toy(n=16):
    wm = toy()
    z0, a = toy_world(n)
    z = torch.stack([z0, P.planted_roll(wm.stack, z0, a)], 1)                          # frame 0 = NOW, frame 1 stands in for the goal frame (not used by the planner)
    A = np.random.default_rng(0).normal(size=(n, 30, 2))
    return wm, z, A


def test_f_planner_reads_only_a0_of_the_logged_actions():
    wm, z, A = _ws_toy()
    pl = H.HWMPlanner(wm, TOY_GEOM, H.SolverCfg(seed_mode="hold"))
    goal = lambda ii: z[torch.as_tensor(ii), 1]
    r1 = pl.plan_windows("hier", goal, z[:, 0], A[:, 0])
    A2 = A.copy(); A2[:, 1:] = np.random.default_rng(1).normal(size=A2[:, 1:].shape)
    r2 = pl.plan_windows("hier", goal, z[:, 0], A2[:, 0])
    assert np.array_equal(r1["plan_raw"], r2["plan_raw"])
    A3 = A.copy(); A3[:, 0] += 0.3
    r3 = pl.plan_windows("hier", goal, z[:, 0], A3[:, 0])
    assert not np.array_equal(r1["plan_raw"], r3["plan_raw"]), "a0 is the measured ego state at NOW and does enter (HOLD seed)"


def test_f_mutation_planner_seeded_from_future_logged_actions_goes_red():
    wm, z, A = _ws_toy()
    class Leaky(H.HWMPlanner):
        def hold_c(self, A_raw0):                                              # the mutant reads a window of logged actions instead of a0
            lo, hi = self.level1_box()
            a = torch.as_tensor(self._A_full[:, :self.geom.n_hold]).float()
            return torch.max(torch.min(self.wm.scale_actions(a), hi), lo)
    goal = lambda ii: z[torch.as_tensor(ii), 1]
    def run(Afull):
        pl = Leaky(wm, TOY_GEOM, H.SolverCfg(seed_mode="hold")); pl._A_full = Afull
        return pl.plan_windows("hier", goal, z[:, 0], Afull[:, 0])["plan_raw"]
    A2 = A.copy(); A2[:, 1:] = np.random.default_rng(1).normal(size=A2[:, 1:].shape)
    assert not np.array_equal(run(A), run(A2)), "the leak check would see this mutant"


# ------------------------------------------------------------------------------------------ (g) kinematic model and band metrics
def test_g_integrator_closed_form_and_bounds():
    v, psi, xy, fr = H.integrate(np.array([10.0]), np.tile([[1.0, 0.0]], (1, 10, 1)))
    assert abs(v[0, -1] - 12.0) < 1e-12 and abs(xy[0, -1, 0] - (10 * 2.0 + 0.5 * 1.0 * 0.2 * 0.2 * 10 * 9)) < 1e-9 and fr == 0.0
    v, psi, xy, fr = H.integrate(np.array([10.0]), np.tile([[50.0, 5.0]], (1, 10, 1)))
    assert fr == 1.0 and v[0, 1] - 10.0 <= H.MU * H.G_ACC * H.DT + 1e-12 and abs(np.diff(psi[0]).max()) <= H.MU * H.G_ACC / 10.0 * H.DT + 1e-9
    v, *_ = H.integrate(np.array([1.0]), np.tile([[-9.0, 0.0]], (1, 10, 1)))
    assert v.min() >= 0.0


def _ref_logged(n=24):
    r = np.random.default_rng(3)
    A = np.stack([r.normal(0, 0.6, (n, 30)), r.normal(0, 0.05, (n, 30))], -1)
    return A, r.uniform(3, 20, n)


def test_g_identical_plan_reads_zero_and_bands_separate():
    A, v0 = _ref_logged()
    m, _ = E.band_metrics(A.copy(), A, v0)
    for b in E.BANDS:
        assert np.isnan(m[b]["hw_err"]).all() and np.isnan(m[b]["ttc_err"]).all()                  # no lead block attached: n = 0, never a zero
        for k, x in m[b].items():
            if k.startswith("_") or np.isnan(x).all():
                continue
            assert np.nanmax(np.abs(x)) == 0.0, (b, k)
    P2 = A.copy(); P2[:, 10:, 0] += 1.0                                            # differs only in knots 11-30
    m2, _ = E.band_metrics(P2, A, v0)
    assert m2["0-2s"]["acc_mae"].max() == 0.0 and m2["2-6s"]["acc_mae"].min() > 0.99 and 0.6 < m2["full6s"]["acc_mae"].mean() < 0.7
    assert m2["0-2s"]["speed_err"].max() == 0.0 and m2["2-6s"]["speed_err"].min() > 0.0


def test_g_mutation_swapped_band_edges_go_red():
    A, v0 = _ref_logged()
    P2 = A.copy(); P2[:, 10:, 0] += 1.0
    good, _ = E.band_metrics(P2, A, v0)
    old = dict(E.BANDS)
    try:
        E.BANDS["0-2s"], E.BANDS["2-6s"] = old["2-6s"], old["0-2s"]
        bad, _ = E.band_metrics(P2, A, v0)
    finally:
        E.BANDS.update(old)
    assert bad["0-2s"]["acc_mae"].max() > 0.0 and good["0-2s"]["acc_mae"].max() == 0.0


# ------------------------------------------------------------------------------------------ (h) geometry from the BUILT model
def test_h_geometry_is_read_from_the_built_stack(real, ladder):
    g22 = H.Geometry.from_stack(real.st, real.wm)
    g35 = H.Geometry.from_stack(ladder.st, ladder.wm)
    assert (g22.s2, g22.s3, g22.h1, g22.h2, g22.goal_frame) == (2, 2, 10, 16, 32)
    assert (g35.s2, g35.s3, g35.h1, g35.h2, g35.goal_frame) == (3, 5, 12, 10, 30) and g35.steps_s == pytest.approx((0.2, 0.6, 3.0))
    st = ladder.st
    old = st.macro_encoders[0].stride
    st.macro_encoders[0].stride = 2                                                # the BUILT modules disagree with the level config: refused
    try:
        with pytest.raises(AssertionError):
            H.Geometry.from_stack(st, ladder.wm)
    finally:
        st.macro_encoders[0].stride = old


# ------------------------------------------------------------------------------------------ (i) admissibility of the goal path
def test_i_goal_path_is_information_disjoint_from_the_situation_classifier():
    from tanitad.models.v8f_mini import scan_source_for_forbidden
    for f in ("hwm_plan.py", "hwm_eval.py", "hwm_data.py", "goal_head.py"):
        assert scan_source_for_forbidden(open(os.path.join(HERE, f), encoding="utf-8").read()) == [], f
    assert scan_source_for_forbidden("def goal(situation_logits):\n    return situation_logits") != []             # the scan can fail
    assert H.GoalInputs._fields == ("z_hist", "nav", "ego_hist", "ctx")                  # ctx: the vision-only strategic token (G5), nothing else
    z = torch.randn(5, 128)
    seen = {}
    def fn(gi):
        seen["shape"] = tuple(gi.z_hist.shape); return gi.z_hist[:, -1]
    out = H.PredictedGoal(fn, 30, z)(np.arange(5))
    assert seen["shape"] == (5, 1, 128) and torch.equal(out, z)                      # the slot hands the predictor the latents up to NOW only
    assert torch.equal(H.PredictedGoal(H.stub_goal_fn, 30, z)(np.arange(5)), z)


# ------------------------------------------------------------------------------------------ (j) the report
def test_j_report_runs_pairs_arms_and_states_the_pause():
    r = np.random.default_rng(0)
    n, eps = 120, 24
    ws = types.SimpleNamespace(eid=np.array([f"e{i % eps}" for i in range(n)]), n=n, lead=dict(state=np.array(["LEAD", "NO_LEAD", "NOT_STRAIGHT", "NO_LABEL"] * (n // 4))))
    def mk(shift):
        out = {}
        for b in E.BANDS:
            d = {m: np.abs(r.normal(1.0 + shift, 0.2, n)) for m in ("acc_mae", "speed_err", "yaw_mae", "head_err", "curv_err", "xtrack", "ade", "tac_man_dis", "tac_lat_dis", "tac_lon_dis")}
            for m in E.LEAD_METRICS + ("_hw_l",):
                d[m] = np.abs(r.normal(1.0 + shift, 0.2, n)) if b == "0-2s" else np.full(n, np.nan)
            for k, hi in zip(E.CLS_KEYS, (3, 3, 5, 3, 3, 5)):
                d[k] = r.integers(0, hi, n)
            out[b] = d
        return out
    base = mk(0.0)
    M = {"hold": base, "prior": mk(0.5), "hwm_oracle": {b: {m: v - 0.5 for m, v in d.items()} for b, d in base.items()}, "flat_oracle": mk(0.2)}
    rep = E.build_report(M, ws, n_boot=200, seeds_M={"hwm_oracle": {0: M["hwm_oracle"], 1: M["hwm_oracle"]}})
    md = E.to_markdown(rep)
    assert "PAUSED" in md and "TACTICAL" in md and "T0" in md
    assert rep["coverage"]["lead_states"]["LEAD"] == 30 and rep["coverage"]["bands"]["2-6s"]["hold"]["n_headway_finite"] == 0 and "outside the 2 s lead track" in rep["coverage"]["bands"]["2-6s"]["hold"]["reason"]
    z = [x for x in rep["rows"] if x["family"] == "LONGITUDINAL" and x["metric"] == "hw_err" and x["band"] == "2-6s"]
    assert z and all(x["n_windows"] == 0 and np.isnan(x["lo"]) for x in z)                                      # n = 0 rows are kept, with the reason, not dropped
    assert rep["tactical"]["hold"]["0-2s"]["maneuver_5way_collapsed"]["per_class"]["lane_keep"]["n_true"] >= 0
    c = [x for x in rep["rows"] if x["kind"] == "contrast" and x["a"] == "hwm_oracle" and x["b"] == "hold" and x["metric"] == "acc_mae" and x["band"] == "0-2s"][0]
    assert c["label"] == "better" and c["delta"] < -0.4
    rr = [x for x in rep["replicate"] if x["metric"] == "acc_mae"][0]
    assert rr["delta"] == 0.0
    banned = ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn", "closed loop", "closed-loop")
    assert not [w for w in banned if w in md.lower()]


# ------------------------------------------------------------------------------------------ (k) LONGITUDINAL headway and TACTICAL (plan-derived), same windows, same bands
LEAD_G0 = np.linspace(14.0, 40.0, 24)                 # initial gap per window, distinct so a misaligned lead shows
LEAD_CLOSE = 2.0                                      # the lead is 2 m/s slower than the ego's logged speed


def _straight_world(n=24):
    """Logged: constant speed, straight (a = 0, yaw rate 0). Lead: dead ahead, rear face at gap g0_i at NOW, closing at 2 m/s; so the logged headway_min is g0_i - 4.0 (knot 10) and the logged min TTC is (g0_i - 3.6) / 2."""
    r = np.random.default_rng(5)
    v0 = r.uniform(8.0, 15.0, n)
    A = np.zeros((n, 30, 2))
    k = np.arange(1, 11)[None, :]
    lens = np.full(n, 4.5)
    x = v0[:, None] * 0.2 * k + LEAD_G0[:n, None] + lens[:, None] / 2 - LEAD_CLOSE * 0.2 * k
    leads = np.stack([x, np.zeros_like(x)], -1)
    lead = dict(leads=leads, lens=lens, state=np.array(["LEAD"] * n), speeds=v0.copy(), ts_rel=0.2 * np.arange(1, 11), dt=0.2)
    return A, v0, lead


def _plan_const_accel(a, n=24):
    p = np.zeros((n, 30, 2))
    p[..., 0] = a
    return p


def _is_diag(blk):
    cm = np.array(blk["confusion_gt_rows_pred_cols"])
    return int(cm.sum() - np.trace(cm)) == 0 and int(cm.sum()) > 0


def _diag_all(M):
    blocks = E.tactical_blocks({"x": M})["x"]
    return [_is_diag(blocks[b][f]) for b in E.BANDS for f in ("lateral", "longitudinal", "maneuver_5way_collapsed")]


def test_k_plan_equal_to_logged_reads_zero_headway_error_and_a_diagonal_confusion():
    A, v0, lead = _straight_world()
    M, _ = E.band_metrics(A.copy(), A, v0, lead)
    m = M["0-2s"]
    assert np.allclose(m["_hw_l"], LEAD_G0 - 4.0, atol=1e-9) and np.allclose(m["_ttc_l"], (LEAD_G0 - 3.6) / LEAD_CLOSE, atol=1e-9)         # the known lead geometry is recovered
    for k in ("hw_err", "tg_err", "ttc_err", "hw_d", "ttc_d"):
        assert np.isfinite(m[k]).all() and np.abs(m[k]).max() == 0.0, k
    for b in ("2-6s", "full6s"):                                                                      # outside the 2 s lead track: n = 0 with the reason, never a zero
        assert np.isnan(M[b]["hw_err"]).all() and np.isnan(M[b]["ttc_err"]).all()
    for b in E.BANDS:
        assert all(M[b][k].max() == 0.0 for k in ("tac_man_dis", "tac_lat_dis", "tac_lon_dis"))
    assert all(_diag_all(M))


def test_k_mutation_a_different_plan_leaves_the_diagonal_and_the_zero():
    A, v0, lead = _straight_world()
    M, _ = E.band_metrics(_plan_const_accel(-6.0), A, v0, lead)
    assert np.abs(M["0-2s"]["hw_err"]).min() > 0.0 and not all(_diag_all(M))                          # the checks of the previous test go red on a plan that differs


def test_k_hard_brake_in_front_of_a_lead_moves_headway_and_ttc_the_right_way():
    A, v0, lead = _straight_world()
    brake, _ = E.band_metrics(_plan_const_accel(-4.0), A, v0, lead)
    m = brake["0-2s"]
    assert (m["hw_d"] > 0).all() and (m["ttc_d"] > 0).all() and (m["tg_d"] > 0).all()                 # braking opens the gap and slows the closing (the minimum TTC grows)
    assert (m["hw_err"] > 0).all() and (m["_ttc_p"] > m["_ttc_l"]).all()
    assert m["tac_lon_dis"].mean() > 0.5                                                              # and the plan reads as brake_stop where the logged path is steady
    # regression arm: an accelerating plan reads the opposite sign; the direction check goes red on it
    gas, _ = E.band_metrics(_plan_const_accel(+3.0), A, v0, lead)
    assert not (gas["0-2s"]["hw_d"] > 0).all() and (gas["0-2s"]["hw_d"] < 0).all() and (gas["0-2s"]["ttc_d"] <= 0).all()


def test_k_mutation_lead_signal_shuffled_across_windows_goes_red():
    A, v0, lead = _straight_world()
    good, _ = E.band_metrics(_plan_const_accel(-4.0), A, v0, lead)
    perm = np.roll(np.arange(24), 7)
    bad_lead = dict(lead, leads=lead["leads"][perm], lens=lead["lens"][perm])
    bad, _ = E.band_metrics(_plan_const_accel(-4.0), A, v0, bad_lead)
    expected = LEAD_G0 - 4.0
    assert np.allclose(good["0-2s"]["_hw_l"], expected, atol=1e-9)                                    # alignment check passes with the true lead ...
    assert not np.allclose(bad["0-2s"]["_hw_l"], expected, atol=1e-6)                                 # ... and goes red when the lead belongs to another window
    assert not np.allclose(bad["0-2s"]["hw_err"], good["0-2s"]["hw_err"], atol=1e-6)                  # the family depends on the lead signal


def test_k_lead_join_uses_raw_frame_two_times_now_and_marks_missing_rows(tmp_path):
    n = 6
    clip = np.array(["ea"] * n + ["eb"] * n)
    frame = np.concatenate([np.arange(n), np.arange(n)])
    path = str(tmp_path / "lead.npz")
    np.savez(path, clip_id=clip, frame=frame, leads=np.full((2 * n, 10, 2), np.nan), lead_lens=frame.astype(float), state=np.array(["LEAD"] * (2 * n)), speeds=frame.astype(float) + 100.0,
             ts_rel_s=0.2 * np.arange(1, 11), dt_s=np.array([0.2]))
    lead = D.lead_for(np.array(["ea", "eb", "ea"]), np.array([1, 2, 5]), path)                       # raw frames 2, 4, 10: the last is absent
    assert list(lead["lens"][:2]) == [2.0, 4.0] and list(lead["state"]) == ["LEAD", "LEAD", "NO_ROW"] and np.isnan(lead["lens"][2]) and np.isnan(lead["leads"][2]).all()
    bad = D.lead_for(np.array(["ea", "eb"]), np.array([1, 2]) + 1, path)                              # regression arm: a shifted key reads other rows, the alignment assertion goes red
    assert list(bad["lens"]) != [2.0, 4.0]
    assert D.lead_join_check(lead, np.array([102.0, 104.0, 0.0])) == (0.0, 0.0, 2)


def test_k_no_lead_in_corridor_is_nan_with_a_reason_not_a_zero():
    A, v0, lead = _straight_world()
    lead2 = dict(lead, state=np.array(["NO_LEAD"] * 12 + ["NOT_STRAIGHT"] * 12))
    M, _ = E.band_metrics(A.copy(), A, v0, lead2)
    assert np.isnan(M["0-2s"]["hw_err"]).all()
    ws = types.SimpleNamespace(eid=np.array([f"e{i % 6}" for i in range(24)]), n=24, lead=lead2)
    cov = E.coverage_of({"hold": M}, ws)
    assert cov["lead_states"]["NO_LEAD"] == 12 and cov["lead_states"]["NOT_STRAIGHT"] == 12 and cov["bands"]["0-2s"]["hold"]["n_headway_finite"] == 0 and "n = 0" in cov["bands"]["0-2s"]["hold"]["reason"]


def test_k_tactical_rule_is_geometry_only_and_one_rule_for_both_sides():
    src = open(os.path.join(HERE, "hwm_eval.py"), encoding="utf-8").read()
    body = src[src.index("def tac_band"):src.index("def band_metrics")]
    assert body.count("classes(xp, pp)") == 1 and body.count("classes(xl, pl)") == 1                  # the same inner function labels both sides
    code = body.split('"""')[2].lower()                                                               # the code after the docstring
    for forbidden in ("goal", "situation", "nav", "oracle", "ws."):
        assert forbidden not in code, forbidden


def _straight_world_6s(n=24):
    """As _straight_world on the 30-knot lead track: the logged gap at knot k is g0_i - 0.4 k."""
    A, v0, _ = _straight_world(n)
    k = np.arange(1, 31)[None, :]
    lens = np.full(n, 4.5)
    x = v0[:, None] * 0.2 * k + LEAD_G0[:n, None] + lens[:, None] / 2 - LEAD_CLOSE * 0.2 * k
    return A, v0, dict(leads=np.stack([x, np.zeros_like(x)], -1), lens=lens, state=np.array(["LEAD"] * n), speeds=v0.copy(), ts_rel=0.2 * np.arange(1, 31), dt=0.2)


def _check_6s(mod):
    A, v0, lead = _straight_world()
    _, _, lead6 = _straight_world_6s()
    M, _ = mod.band_metrics(A.copy(), A, v0, lead, lead6)
    for b in ("2-6s", "full6s"):
        assert np.allclose(M[b]["_hw_l"], LEAD_G0 - 12.0, atol=1e-9), b                               # gap at knot 30
        assert np.allclose(M[b]["_ttc_l"], (LEAD_G0 - 11.6) / LEAD_CLOSE, atol=1e-9), b
        for k in ("hw_err", "tg_err", "ttc_err"):
            assert np.isfinite(M[b][k]).all() and np.abs(M[b][k]).max() == 0.0, (b, k)
    assert np.allclose(M["0-2s"]["_hw_l"], LEAD_G0 - 4.0, atol=1e-9)                                  # the 0-2 s band keeps the K = 10 block


def test_k_6s_lead_block_gives_the_known_gap_in_the_later_bands():
    _check_6s(E)


def test_k_later_band_reads_the_lead_at_its_own_knots_mutation_goes_red():
    import importlib.util
    src = open(E.__file__, encoding="utf-8").read()
    for a, b in (('lead["leads"][:, lo:hi]', 'lead["leads"][:, :hi - lo]'), ("LM.distance_keeping(xl[:, lo + 1:hi + 1]", "LM.distance_keeping(xl[:, 1:hi - lo + 1]")):
        assert src.count(a) == 1, a
        spec = importlib.util.spec_from_loader("hwm_eval_mut", loader=None)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = E.__file__
        exec(compile(src.replace(a, b), "hwm_eval_mut", "exec"), mod.__dict__)
        with pytest.raises(AssertionError):
            _check_6s(mod)


def test_k_plan_that_brakes_keeps_a_larger_gap_in_the_later_band():
    A, v0, lead = _straight_world()
    _, _, lead6 = _straight_world_6s()
    M, _ = E.band_metrics(_plan_const_accel(-1.0), A, v0, lead, lead6)
    assert np.isfinite(M["2-6s"]["hw_d"]).any() and (M["2-6s"]["hw_d"][np.isfinite(M["2-6s"]["hw_d"])] > 0).all()


# ------------------------------------------------------------------------------------------ (l) the band streams: G1 plan_ctx, G2 T3, G3 stitch, G6 vocabulary seeds, G7 constraints
import contextlib
import goal_head as GH


def _lad_plan(ladder, solver=None, geom=None, n=4, goal_frame=30, plan_fn=None, Z=None, **kw):
    Z = ladder.Z if Z is None else Z
    pl = H.HWMPlanner(ladder.wm, geom, solver or H.SolverCfg(**FAST), plan_fn=plan_fn, **kw)
    return pl, pl.plan_windows("hier", lambda ii: Z[torch.as_tensor(ii), goal_frame], Z[:n, 0], ladder.A[:n, 0].numpy().astype(np.float64), idx=np.arange(n))


def test_l_g1_plan_ctx_default_is_today_and_3_reaches_the_rollout(ladder):
    base = _lad_plan(ladder)[1]
    one = _lad_plan(ladder, H.SolverCfg(**FAST, plan_ctx=1))[1]
    assert np.array_equal(base["plan_raw"], one["plan_raw"]) and np.array_equal(base["cost1"], one["cost1"])                 # the default is today's planner, bit for bit
    three = _lad_plan(ladder, H.SolverCfg(**FAST, plan_ctx=3))[1]
    assert not np.array_equal(base["plan_raw"], three["plan_raw"]), "context 3 must reach the rollouts of the plan"
    st = ladder.st
    seen = set()

    def spy(level, z0, a, **kw):
        seen.add((level, kw.get("context", 1)))
        return type(st).rollout(st, level, z0, a, **kw)

    st.__dict__["rollout"] = spy
    try:
        _lad_plan(ladder, H.SolverCfg(**FAST, plan_ctx=(1, 3, 2)))
    finally:
        st.__dict__.pop("rollout", None)
    assert seen == {(1, 1), (2, 3), (3, 2)}, seen                                                                          # every level got its own context
    with pytest.raises(ValueError):                                                                                       # more than the predictor was built for: refused, and the shadow is removed again
        _lad_plan(ladder, H.SolverCfg(**FAST, plan_ctx=4))
    assert "rollout" not in st.__dict__ and "_optimize" not in st.__dict__


def test_l_g1_mutation_shadow_not_installed_makes_context_3_a_no_op(ladder):
    base = _lad_plan(ladder)[1]
    orig = H.shadow_solver
    H.shadow_solver = lambda *a, **k: contextlib.nullcontext()
    try:
        mut = _lad_plan(ladder, H.SolverCfg(**FAST, plan_ctx=3))[1]
    finally:
        H.shadow_solver = orig
    assert np.array_equal(base["plan_raw"], mut["plan_raw"]), "with the shadow cut the context-3 arm equals the context-1 arm, so the check above would go red"


def test_l_g2_t3_arm_geometry_top_horizon_and_goal_reach(ladder):
    st, wm = ladder.st, ladder.wm
    g4, g2 = H.Geometry.from_stack(st, wm, goal_steps=4), H.Geometry.from_stack(st, wm)
    assert (g4.T3, g4.goal_frame, g4.K2, g4.h2, g4.K1) == (4, 60, 2, 10, 4) and g4.goal_frame * H.DT == pytest.approx(12.0) and (g2.T3, g2.goal_frame) == (2, 30)
    Z = torch.randn(4, 61, 128) * 0.5
    tops = []

    def spy(st_, *a, top_horizon, **kw):
        tops.append(top_horizon)
        return Q.plan_topdown2(st_, *a, top_horizon=top_horizon, **kw)

    pa = _lad_plan(ladder, geom=g4, goal_frame=60, Z=Z, plan_fn=spy)[1]["plan_raw"]
    pb = _lad_plan(ladder, geom=g4, goal_frame=60, Z=Z.roll(1, 0), plan_fn=spy)[1]["plan_raw"]
    assert tops == [4, 4] and not np.array_equal(pa, pb), "T3 = 4 plans four level-3 steps, and the goal at 12 s reaches the plan"
    tops.clear()
    _lad_plan(ladder, geom=g2, goal_frame=30, Z=Z, plan_fn=spy)
    assert tops == [2]
    tops.clear()
    import dataclasses
    _lad_plan(ladder, geom=dataclasses.replace(g4, T3=g4.K2), goal_frame=30, Z=Z, plan_fn=spy)                              # mutation: T3 pinned to K2 (a planner that ignores the T3 arm)
    assert tops == [2] and tops != [4], "the T3-ignoring mutant plans 2 level-3 steps, so the tops == [4, 4] check above goes red"


class _FakeStore:
    def __init__(self, d): self.d = d
    def get(self, e): return self.d[e]


def test_l_g2_window_counts_at_each_t3():
    ok = np.ones((100, 2))
    nan = np.ones((70, 2)); nan[40] = np.nan
    st = _FakeStore({"A": dict(pose=np.zeros(100), p10f=ok), "B": dict(pose=np.zeros(70), p10f=nan)})
    pairs = [("A", 0), ("A", 30), ("A", 50), ("A", 40), ("B", 0), ("B", 20), ("B", 35)]
    c = D.goal_frame_counts([e for e, _ in pairs], [g for _, g in pairs], [30, 60], store=st)
    assert c["counts"] == {30: 5, 60: 2} and c["n_pairs"] == 7, c                                                         # hand count: A 4 and B 1 at frame 30, A 2 at frame 60
    c2 = D.goal_frame_counts([e for e, _ in pairs], [g for _, g in pairs], [31, 61], store=st)
    assert c2["counts"] != c["counts"], "a shifted frame changes the count (an off-by-one would go red against the hand count)"


def test_l_g3_l1_full_stitch_plans_every_knot_from_level_1():
    wm = toy()
    g_full = H.Geometry(2, 2, 6, 3, 3, 12, 6, "l1_full").check()
    g_dec = H.Geometry(2, 2, 6, 3, 3, 12, 6, "decode").check()
    z0, a = toy_world(32)
    zT = P.planted_roll(wm.stack, z0, a)
    rf = toy_plan(wm, g_full, z0, zT)
    rd = toy_plan(wm, g_dec, z0, zT)
    assert "decode_resid" not in rf and np.array_equal(rf["plan_raw"], rf["a1_raw"][:, :12]), "l1_full: all 12 knots are level 1's actions, no decode"
    assert np.array_equal(rd["plan_raw"][:, :6], rd["a1_raw"][:, :6]) and not np.array_equal(rd["plan_raw"][:, 6:], rd["a1_raw"][:, 6:12]), "decode: knots 7-12 come from the decoded level-2 part"
    pr = rf["plan_raw"].copy(); pr[..., 0] /= H.ACC_SCALE
    err = np.abs(pr - a.numpy())
    base = (wm.stack.prior_mean_1.expand(32, 12, 2) - a).abs().mean((1, 2)).numpy()
    assert err.mean() <= TOL and err.mean() <= FACTOR * base.mean(), (err.mean(), base.mean())                             # level 1 against all level-2 sub-goals recovers the planted actions
    with pytest.raises(AssertionError):
        H.Geometry(2, 2, 3, 3, 3, 12, 6, "l1_full").check()                                                                # a level-1 horizon of 6 cannot hold 12 knots: refused


def test_l_g3_l1_full_on_the_built_ladder_model(ladder):
    g = H.Geometry.from_stack(ladder.st, ladder.wm, stitch="l1_full")
    assert (g.K1, g.h1, g.stitch_mode) == (10, 30, "l1_full")
    r = _lad_plan(ladder, geom=g)[1]
    assert r["plan_raw"].shape == (4, 30, 2) and np.isfinite(r["plan_raw"]).all() and np.array_equal(r["plan_raw"], r["a1_raw"][:, :30]) and "decode_resid" not in r


# ---- G6: the tactical vocabulary seeds level 2
def _vocab_world(n=36):
    wm = toy()
    amp = 0.5
    tpl = np.array([[[lo * amp, la * amp]] * 12 for la in (0, 1, -1) for lo in (-1, 0, 1)], np.float64)             # 9 classes: (lat) x (lon), constant patterns the toy can reach
    voc = H.Vocabulary(tuple(f"c{i}" for i in range(9)), tpl)
    g = torch.Generator().manual_seed(11)
    a = torch.tensor(tpl[np.arange(n) % 9], dtype=torch.float32) + 0.02 * torch.randn(n, 12, 2, generator=g)
    z0 = torch.randn(n, 32, generator=torch.Generator().manual_seed(5))
    return wm, voc, z0, a, P.planted_roll(wm.stack, z0, a)


def _l2_init_cost(wm, voc, z0, zT, S=16, **kw):
    pl = H.HWMPlanner(wm, TOY_GEOM, H.SolverCfg(seed_mode="hold", S=S), vocab=voc, **kw)
    res = pl.plan_batch("hier", z0, zT, pl.hold_c(np.zeros((z0.shape[0], 2))), P.make_gen(0, 0), ii=np.arange(z0.shape[0]))
    return res.init_costs[2].numpy(), res


def test_l_g6_vocabulary_seeds_beat_hold_seeds_on_a_planted_vocabulary_world():
    wm, voc, z0, a, zT = _vocab_world()
    c_voc, _ = _l2_init_cost(wm, voc, z0, zT)
    c_none, _ = _l2_init_cost(wm, None, z0, zT)
    assert c_voc.mean() < 0.2 * c_none.mean(), (c_voc.mean(), c_none.mean())                                              # one template is the planted class: the best level-2 candidate starts at its goal
    # mutation: the seeds replaced by HOLD (zero templates) must lose the advantage
    voc_hold = H.Vocabulary(voc.names, np.zeros_like(voc.templates))
    c_mut, _ = _l2_init_cost(wm, voc_hold, z0, zT)
    assert not (c_mut.mean() < 0.2 * c_none.mean()), (c_mut.mean(), c_none.mean())
    # the edge reaches the output: another vocabulary (a different candidate set) changes the level-2 plan when the solver is short
    short = H.SolverCfg(seed_mode="hold", n_iters=1)
    run = lambda v: H.HWMPlanner(wm, TOY_GEOM, short, vocab=v).plan_windows("hier", lambda ii: zT[torch.as_tensor(ii)], z0, np.zeros((36, 2)))["plan_raw"]
    assert not np.array_equal(run(voc), run(H.Vocabulary(voc.names, 0.5 * voc.templates))) and not np.array_equal(run(voc), run(None))


def test_l_g6_top_classes_from_scores_and_seed_encoding_is_action_only():
    wm, voc, z0, a, zT = _vocab_world(18)
    scores = np.zeros((18, 9)); scores[np.arange(18), np.arange(18) % 9] = 1.0                                              # a banked posterior that puts the planted class first
    c_top, _ = _l2_init_cost(wm, voc, z0, zT, S=4, vocab_scores=scores)                                                      # S = 4 holds three template candidates
    c_all, _ = _l2_init_cost(wm, voc, z0, zT, S=4)
    assert c_top.mean() < 0.5 * c_all.mean(), (c_top.mean(), c_all.mean())                                                   # the top classes of the posterior carry the planted class; the fixed order does so for a third of the windows
    assert voc.order(scores, np.arange(3), 2).tolist() == [[0, 1], [1, 0], [2, 0]] and voc.order(None, np.arange(2), 3).tolist() == [[0, 1, 2], [0, 1, 2]]
    s1 = H.vocab_macro_seeds(wm.stack, wm, voc, z0[:1], 12, 6)
    s2 = H.vocab_macro_seeds(wm.stack, wm, voc, z0[5:6] * 3.0, 12, 6)
    assert torch.equal(s1, s2) and s1.shape == (9, 6, 4)                                                                    # one reference latent serves all windows: the macro path reads actions only


# ---- G7: T's constraints as a cost
class _ShimRaw(_Shim):
    def scale_actions(self, a):
        return a / torch.tensor([H.ACC_SCALE, 1.0])


def _cons_world(n=16, acc=0.3):
    wm = toy()
    wm = _ShimRaw(wm.stack)
    a = torch.zeros(n, 12, 2); a[..., 0] = acc
    z0 = torch.randn(n, 32, generator=torch.Generator().manual_seed(9))
    zT = P.planted_roll(wm.stack, z0, a)
    Zf = torch.randn(64, 13, 32, generator=torch.Generator().manual_seed(3))
    Af = P.planted_actions(64, 4).repeat(1, 2, 1)[:, :13] * torch.tensor([H.ACC_SCALE, 1.0])
    ro = H.fit_kin_readout(wm.stack, wm, Zf, Af)
    return wm, z0, zT, ro


def _cons(n, ro, weight, cap=10.5):
    v0 = np.full(n, 10.0)
    return H.Constraints(v0, np.full(n, cap), np.full(n, 3.0), np.full(n, 4.0), weight, ro)


def test_l_g7_penalty_is_exactly_zero_inside_the_caps_and_positive_outside_and_has_a_gradient():
    wm, z0, zT, ro = _cons_world()
    cb = _cons(4, ro, 1.0).batch(np.arange(4))
    ok = torch.zeros(4, 3, 12, 2)
    bad = ok.clone(); bad[..., 0] = 0.4                                                                                     # raw 0.8 m/s^2 for 2.4 s: v passes 10.5 m/s
    assert float(H.constraint_penalty(1, ok, cb, 1.0, 2).abs().max()) == 0.0                                                # exact-0.0 control
    cand = bad.clone().requires_grad_(True)
    pen = H.constraint_penalty(1, cand, cb, 1.0, 2)
    assert float(pen.detach().min()) > 0.0
    (g,) = torch.autograd.grad(pen.sum(), cand)
    assert float(g.abs().sum()) > 0.0
    hard = ok.clone(); hard[..., 0] = -3.0                                                                                  # raw -6 m/s^2: beyond the 4 m/s^2 brake cap
    assert float(H.constraint_penalty(1, hard, cb, 1.0, 2).min()) > 0.0
    turn = ok.clone(); turn[..., 1] = 1.0                                                                                   # 10 m/s at 1 rad/s = 10 m/s^2 lateral: beyond 3
    assert float(H.constraint_penalty(1, turn, cb, 1.0, 2).min()) > 0.0
    # level 2 through the readout: a macro-action that decodes to a fast acceleration is penalised, one that decodes to none is not
    with torch.no_grad():
        enc_bad = wm.stack.encode(torch.zeros(1, 13, 32), bad[:1, 0])
        enc_ok = wm.stack.encode(torch.zeros(1, 13, 32), ok[:1, 0])
    assert float(ro.r2.min()) > 0.9, ro.r2
    m_bad, m_ok = enc_bad.act[1][:, None], enc_ok.act[1][:, None]
    assert float(H.constraint_penalty(2, m_bad.expand(4, 1, -1, -1), cb, 1.0, 2, ro).min()) > 0.0 and float(H.constraint_penalty(2, m_ok.expand(4, 1, -1, -1), cb, 1.0, 2, ro).abs().max()) == 0.0
    with pytest.raises(AssertionError):
        H.constraint_penalty(2, m_ok.expand(4, 1, -1, -1), cb, 1.0, 2, None)


def _speed_of(r, v0=10.0):
    v, *_ = H.integrate(np.full(len(r["plan_raw"]), v0), r["plan_raw"])
    return v[:, -1]


def test_l_g7_weight_zero_is_today_bit_for_bit_and_a_violating_plan_is_penalised():
    wm, z0, zT, ro = _cons_world()
    n = z0.shape[0]
    run = lambda **kw: H.HWMPlanner(wm, TOY_GEOM, H.SolverCfg.plantest(), **kw).plan_windows("hier", lambda ii: zT[torch.as_tensor(ii)], z0, np.zeros((n, 2)))
    free = run()
    zero = run(constraints=_cons(n, ro, 0.0))
    assert np.array_equal(free["plan_raw"], zero["plan_raw"]) and np.array_equal(free["cost1"], zero["cost1"]), "weight 0 with the hooks installed equals no constraints, bit for bit"
    held = run(constraints=_cons(n, ro, 1.0e4))
    assert _speed_of(held).mean() < _speed_of(free).mean() - 0.3, (_speed_of(held).mean(), _speed_of(free).mean())            # the planted goal asks for a speed above the cap; the constraint pulls the plan under it
    assert _speed_of(free).mean() > 10.5
    # mutation: the penalty zeroed (the constraint hook wired to nothing) leaves the plan where the free one is
    orig = H.constraint_penalty
    H.constraint_penalty = lambda level, a, *r, **k: a.sum((-1, -2)) * 0.0
    try:
        dead = run(constraints=_cons(n, ro, 1.0e4))
    finally:
        H.constraint_penalty = orig
    assert np.array_equal(dead["plan_raw"], free["plan_raw"]), "with the penalty cut the weighted run equals the free run, so the speed check above goes red"


def test_l_g7_constraints_reach_both_levels_through_the_hooks():
    wm, z0, zT, ro = _cons_world()
    n = z0.shape[0]
    seen = []
    orig = H.constraint_penalty
    def spy(level, a, *r, **k):
        seen.append(level)
        return orig(level, a, *r, **k)
    H.constraint_penalty = spy
    try:
        H.HWMPlanner(wm, TOY_GEOM, H.SolverCfg(S=4, n_iters=2), constraints=_cons(n, ro, 1.0)).plan_windows("hier", lambda ii: zT[torch.as_tensor(ii)], z0, np.zeros((n, 2)))
        lv_h = set(seen); seen.clear()
        H.HWMPlanner(wm, TOY_GEOM, H.SolverCfg(S=4, n_iters=2), constraints=_cons(n, ro, 1.0)).plan_windows("flat", lambda ii: zT[torch.as_tensor(ii)], z0, np.zeros((n, 2)))
        lv_f = set(seen)
    finally:
        H.constraint_penalty = orig
    assert lv_h == {1, 2} and lv_f == {1}


# ---- G4, G5: the goal head
def _gh_world(n=640, steps=3, d=32, ctx_dim=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    z, nav, ctx = torch.randn(n, d, generator=g), torch.randn(n, 5, generator=g), torch.randn(n, ctx_dim, generator=g)
    Wc, Wn = torch.randn(ctx_dim, d, generator=g) * 0.7, torch.randn(5, d, generator=g) * 0.3
    tg = torch.stack([0.9 * z + j * 0.5 * (ctx @ Wc) + 0.5 * j * (nav @ Wn) + 0.03 * torch.randn(n, d, generator=g) for j in range(1, steps + 1)], 1)
    valid = torch.rand(n, steps, generator=g) < 0.8
    return dict(z_now=z.numpy(), z_tgt=tg.numpy(), valid=valid.numpy(), nav=nav.numpy(), stok=ctx.numpy(), ego=torch.randn(n, 1, 2, generator=g).numpy(), eid=np.array([f"e{i // 4}" for i in range(n)]))


@pytest.fixture(scope="module")
def gh():
    st = toy().stack
    d = _gh_world()
    tr, va = GH.split_by_episode(d["eid"])
    ntr, nva = GH.NpzBatches(d, tr, 128, 0), GH.NpzBatches(d, va, 4096, 0)
    head = GH.GoalHead(32, 32, 3, ctx_dim=6, hidden=64)
    res = GH.train(head, st, ntr.get, 500, lr=3e-3, log=lambda s: None)
    return types.SimpleNamespace(st=st, d=d, tr=tr, va=va, ntr=ntr, nva=nva, head=head, res=res)


def test_l_g45_split_is_by_episode_and_the_head_beats_the_stub_with_shuffle_and_context_swap_controls(gh):
    assert not (set(gh.d["eid"][gh.tr]) & set(gh.d["eid"][gh.va])) and len(gh.va) > 0
    c = GH.controls(gh.head, gh.st, gh.nva.get(0))
    assert c["head"] < 0.3 * c["stub"], c                                                                                  # the head reads latents, nav and context: far under the stub goal
    assert c["shuffled"] > 2.0 * c["head"], c                                                                              # a shuffled goal is worse
    assert c["ctx_swap"] > 2.0 * c["head"] and c["no_ctx"] > 2.0 * c["head"], c                                            # the context carries information here, and a swap of it costs
    hn = GH.GoalHead(32, 32, 3, ctx_dim=0, hidden=64)
    GH.train(hn, gh.st, lambda s: {k: v for k, v in gh.ntr.get(s).items() if k != "ctx"}, 500, lr=3e-3, log=lambda s: None)
    cn = GH.controls(hn, gh.st, {k: v for k, v in gh.nva.get(0).items() if k != "ctx"})
    assert "ctx_swap" not in cn and cn["head"] > 1.5 * c["head"], (cn, c)                                                  # mutation: a head with no context path cannot fit the context-dependent target, and has no swap control


def test_l_g45_training_is_head_only_world_model_bit_equal_and_selfcheck_readback(gh):
    r = gh.res
    assert r["wm_fingerprint_before"] == r["wm_fingerprint_after"] == GH.wm_fingerprint(gh.st)
    rb = r["selfcheck"]
    assert rb["red"] == [] and rb["built"]["target_level"] == 3 and rb["built"]["ctx_path"] and rb["built"]["detach_ctx"] and rb["built"]["wm_trainable"] == 0 and rb["built"]["source_scan"] == []
    assert rb["probe"]["min_param_grad_abs_sum"] > 0.0 and rb["probe"]["n_params_with_zero_grad"] == 0                          # non-zero gradient on every head parameter
    assert rb["control_exact_zero"]["loss"] == 0.0 and rb["control_exact_zero"]["max_grad_abs_sum"] == 0.0 and rb["ctx_grad_is_none"]   # exact-0.0 control; the strategic token gets no gradient
    assert all(p.requires_grad for p in gh.st.parameters())                                                                 # the freeze is undone after the run


def _fresh_head():
    return GH.GoalHead(32, 32, 3, ctx_dim=6, hidden=32)


def _freeze(st):
    return GH.freeze_wm(st)


def test_l_g45_mutations_each_selfcheck_goes_red(gh):
    st, b = gh.st, gh.ntr.get(0)
    restore = _freeze(st)
    try:
        GH.selfcheck(_fresh_head(), st, b)                                                                                  # the healthy head passes
        orig_fwd, orig_loss = GH.GoalHead.forward, GH.goal_loss
        GH.GoalHead.forward = lambda self, gi: orig_fwd(self, gi) * 0.0                                                     # zero gradient on the head
        with pytest.raises(GH.GoalHeadSelfcheckError, match="zero gradient"):
            GH.selfcheck(_fresh_head(), st, b)
        GH.GoalHead.forward = orig_fwd
        GH.goal_loss = lambda pred, tgt, valid: orig_loss(pred, tgt, torch.ones_like(valid))                               # the mask is not applied: the exact-0.0 control reads non-zero
        with pytest.raises(GH.GoalHeadSelfcheckError, match="exact-0.0"):
            GH.selfcheck(_fresh_head(), st, b)
        GH.goal_loss = orig_loss
        GH.DETACH_CTX = False                                                                                               # the detach removed: the loss reaches the token
        with pytest.raises(GH.GoalHeadSelfcheckError, match="strategic token|detached"):
            GH.selfcheck(_fresh_head(), st, b)
        GH.DETACH_CTX = True
        from tanitad.models.v8f_mini import scan_source_for_forbidden
        assert scan_source_for_forbidden(open(GH.__file__, encoding="utf-8").read() + "\nsituation_x = 1\n") == ["situation"]    # the source scan can fail
    finally:
        GH.GoalHead.forward, GH.goal_loss, GH.DETACH_CTX = orig_fwd, orig_loss, True
        restore()
    with pytest.raises(GH.GoalHeadSelfcheckError, match="trainable"):
        GH.selfcheck(_fresh_head(), st, b)                                                                                  # an unfrozen world model is refused before step 1


def test_l_g45_mutation_a_trainer_that_updates_the_world_model_changes_its_fingerprint():
    st = toy().stack
    d = _gh_world(128)
    nb = GH.NpzBatches(d, None, 64, 0)
    fp = GH.wm_fingerprint(st)
    orig = (GH.freeze_wm, GH.trainable_params, GH.encode_now, GH.selfcheck)
    def enc_grad(s, z):
        return s.initial_latents(z, 3)[2]
    GH.freeze_wm = lambda s: (lambda: None)
    GH.trainable_params = lambda head, s: list(head.parameters()) + list(s.encoders[0].parameters())
    GH.encode_now = enc_grad
    GH.selfcheck = lambda *a, **k: {}
    try:
        st.encoders[0] = nn.Linear(32, 32)                                                                                  # identity encoders have no parameters; a trainable one stands in for the world model's
        fp = GH.wm_fingerprint(st)
        with pytest.raises(GH.GoalHeadSelfcheckError, match="world model changed"):
            GH.train(GH.GoalHead(32, 32, 3, ctx_dim=6, hidden=16), st, nb.get, 5, lr=1e-2, log=lambda s: None)
    finally:
        GH.freeze_wm, GH.trainable_params, GH.encode_now, GH.selfcheck = orig
    assert GH.wm_fingerprint(st) != fp


def test_l_g45_goal_head_wired_as_predicted_goal_the_context_edge_reaches_the_plan(gh):
    st = gh.st
    wm = toy()
    z0 = torch.from_numpy(gh.d["z_now"][:16]).float()
    nav, ctx = torch.from_numpy(gh.d["nav"][:16]).float(), torch.from_numpy(gh.d["stok"][:16]).float()
    fn = GH.make_goal_fn(gh.head, st, 3)
    assert fn.level == 3
    pl = H.HWMPlanner(wm, TOY_GEOM, H.SolverCfg.plantest())
    run = lambda c, **kw: pl.plan_windows("hier", H.PredictedGoal(fn, 12, z0, nav=nav, ctx=c, level=3), z0, np.zeros((16, 2)), **kw)["plan_raw"]
    p0, p1 = run(ctx), run(ctx.roll(5, 0))
    assert _goal_reaches_every_band(p0, p1), "a context swap changes the goal and so the knots of both bands"
    p_none = run(None)
    assert not np.array_equal(p0, p_none), "the context edge off (the off arm) changes the plan"
    # mutation: the slot hands the head no context: swapping it changes nothing, so the check above goes red
    class NoCtx(H.PredictedGoal):
        def __call__(self, idx, inputs=None):
            self.ctx = None
            return super().__call__(idx, inputs)
    cut = lambda c: pl.plan_windows("hier", NoCtx(fn, 12, z0, nav=nav, ctx=c, level=3), z0, np.zeros((16, 2)))["plan_raw"]
    assert np.array_equal(cut(ctx), cut(ctx.roll(5, 0)))
    with pytest.raises(AssertionError):                                                                                    # a level-3 goal has no level-1 pre-image: the flat arm refuses it
        pl.plan_windows("flat", H.PredictedGoal(fn, 12, z0, nav=nav, ctx=ctx, level=3), z0, np.zeros((16, 2)))


def test_l_g45_predicted_goal_slot_passes_the_context_and_nothing_else():
    seen = {}
    def fn(gi):
        seen.update(fields=gi._fields, ctx=None if gi.ctx is None else tuple(gi.ctx.shape), nav=None if gi.nav is None else tuple(gi.nav.shape))
        return gi.z_hist[:, -1]
    z = torch.randn(5, 16)
    H.PredictedGoal(fn, 30, z, nav=torch.zeros(5, 5), ctx=torch.zeros(5, 7))(np.arange(3))
    assert seen["fields"] == ("z_hist", "nav", "ego_hist", "ctx") and seen["ctx"] == (3, 7) and seen["nav"] == (3, 5)


def test_l_g45_selfcheck_reads_and_clears_stale_grads_on_the_world_model(gh):
    st = gh.st
    p0 = next(iter(st.parameters())) if any(True for _ in st.parameters()) else None
    if p0 is None:                                                                                                             # the identity-encoder toy has no parameters of its own: give the stack one stale gradient through a stand-in
        pytest.skip("toy stack has no parameters")
    restore = _freeze(st)
    try:
        p0.grad = torch.ones_like(p0)
        rb = GH.selfcheck(_fresh_head(), st, gh.ntr.get(0))
        assert rb["wm_stale_grads_before_probe"] >= 1 and p0.grad is None
    finally:
        restore()
