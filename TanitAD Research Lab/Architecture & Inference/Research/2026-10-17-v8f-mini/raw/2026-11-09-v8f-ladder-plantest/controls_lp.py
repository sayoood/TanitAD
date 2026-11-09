"""Evidence that the ladder PLANTEST runner is right (IMPL_CHOICES_LADDER_PLANTEST.md section 6, tolerances fixed before this ran).
Usage: python controls_lp.py [name ...]   (no names = all). Writes CONTROLS_LADDER_PLANTEST.json / .md. Tier T0. CPU, 1 thread."""
import json, os, sys, time, traceback
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import lp_lib as L
import plantest_lib as P
import plantest2_lib as Q

TH = L.THEORY
IDX16 = np.arange(0, 1976, 124)[:16]
ARMS = ("R35_s0", "R35_s1", "R35H_s0", "R35H_s1", "C0_s0", "C0_s1")
TOL_PLAN, TOL_ACC, TOL_YAW = 1e-3, 1e-4, 1e-5
TOL_LATENT, TOL_HIST, TOL_REG_RED = 1e-4, 1e-6, 1e-3
TOL_PLANT, FACTOR, TOL_LIMIT, TOL_GRAD = 0.05, 0.2, 1e-4, 1e-6
_CACHE = {}


def arm(name):
    if name not in _CACHE:
        _CACHE.clear()
        _CACHE[name] = L.prepare_arm(name)
    return _CACHE[name]


def eq(a, b):
    return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)


# ------------------------------------------------------------------------------------------- 1. M2b reproduction through this code path
def c_repro():
    c = L.prepare_m2b("M2b_s0")
    out = {}
    # PLANTEST (prior init), first 64 windows = batch 0 of the banked run
    bank = np.load(f"{TH}/plantest/runs/M2b_s0_flat1_true_s0.npz")["plan_raw"][:64]
    idx = np.arange(64)
    r = L.unit3(c, "flat1", "true", "c1", idx, 0.0, seed_mode="prior", collect=False)
    mb, mn = P.window_metrics(bank, c.A[idx], c.v0[idx]), P.window_metrics(r["plan_raw"], c.A[idx], c.v0[idx])
    out["PLANTEST_flat1_true"] = dict(max_plan_diff=float(np.abs(r["plan_raw"] - bank).max()), d_acc=float(abs(mn["acc_mae"].mean() - mb["acc_mae"].mean())), d_yaw=float(abs(mn["yaw_mae"].mean() - mb["yaw_mae"].mean())),
                                      banked_acc_mae=float(mb["acc_mae"].mean()), mine_acc_mae=float(mn["acc_mae"].mean()), banked_yaw_mae=float(mb["yaw_mae"].mean()), mine_yaw_mae=float(mn["yaw_mae"].mean()), n=64)
    # PLANTEST2 (SEED, lambda 1.0), first 64 SCORE windows
    bank2 = np.load(f"{TH}/plantest2/runs/score_M2b_s0_flat1_true_L1_p0.npz")
    idx2 = c.score_idx[:64]
    assert np.array_equal(bank2["idx"][:64], idx2)
    r2 = L.unit3(c, "flat1", "true", "c1", idx2, 1.0, seed_mode="hold", collect=False)
    mb, mn = P.window_metrics(bank2["plan_raw"][:64], c.A[idx2], c.v0[idx2]), P.window_metrics(r2["plan_raw"], c.A[idx2], c.v0[idx2])
    out["PLANTEST2_flat1_true_lam1"] = dict(max_plan_diff=float(np.abs(r2["plan_raw"] - bank2["plan_raw"][:64]).max()), d_acc=float(abs(mn["acc_mae"].mean() - mb["acc_mae"].mean())),
                                            d_yaw=float(abs(mn["yaw_mae"].mean() - mb["yaw_mae"].mean())), banked_acc_mae=float(mb["acc_mae"].mean()), mine_acc_mae=float(mn["acc_mae"].mean()), n=64)
    ok = all(v["max_plan_diff"] <= TOL_PLAN and v["d_acc"] <= TOL_ACC and v["d_yaw"] <= TOL_YAW for v in out.values())
    # the mutant: the wrong seed's priors (M2b_s1 TRAIN prior on s0 latents is the same data; instead use the planner seed 1) must NOT reproduce
    rb = L.unit3(c, "flat1", "true", "c1", idx, 0.0, seed=1, seed_mode="prior", collect=False)
    out["mutant_planner_seed1_max_plan_diff"] = float(np.abs(rb["plan_raw"] - np.load(f"{TH}/plantest/runs/M2b_s0_flat1_true_s0.npz")["plan_raw"][:64]).max())
    red = out["mutant_planner_seed1_max_plan_diff"] > TOL_PLAN
    return dict(passed=bool(ok and red), detail=out, regression_goes_red=bool(red))


def c_latent_route():
    a = L.prepare_m2b("M2b_s0", whole_clip=False)
    b = L.prepare_m2b("M2b_s0", whole_clip=True)
    d = (a.Zw - b.Zw).abs()
    per_knot = d.amax(dim=(0, 2)).tolist()
    return dict(passed=bool(d.max() <= TOL_LATENT), detail=dict(max_abs_diff=float(d.max()), per_knot_max=per_knot, action_max_abs_diff_vs_cache=a.action_max_abs_diff_vs_cache))


# ------------------------------------------------------------------------------------------- 2. reductions, history, reachability
def c_reduction():
    out = {}
    for name, kinds in (("C0_s0", ("flat1", "td3")), ("R35_s0", ("flat1",))):
        c = arm(name)
        hc = L.hold_c_of(c)
        for kind in kinds:
            k = "flat" if kind == "flat1" else kind
            lk = 1 if kind == "flat1" else None
            a = P.plan_all(c.st, k, "true", c.z_now, c.z_fut, c.donor, idx=IDX16, last_k=lk)
            b = L.plan_all3(c, kind, "true", "c1", idx=IDX16, lam=0.0, seed_mode="prior")
            out[f"{name}|{kind}|PLANTEST"] = bool(eq(a, b))
            a2 = Q.plan_all2(c.st, kind, "true", c.z_now, c.z_fut, c.donor, hc, idx=IDX16, lam=0.3)
            b2 = L.plan_all3(c, kind, "true", "c1", idx=IDX16, lam=0.3, seed_mode="hold")
            out[f"{name}|{kind}|PLANTEST2_lam0.3"] = bool(eq(a2, b2))
    # mutant: the HOLD seed on one side only must differ from the PLANTEST initialisation
    c = arm("R35_s0")
    m = L.plan_all3(c, "flat1", "true", "c1", idx=IDX16, lam=0.0, seed_mode="hold")["actions"]
    p = L.plan_all3(c, "flat1", "true", "c1", idx=IDX16, lam=0.0, seed_mode="prior")["actions"]
    red = not np.array_equal(m, p)
    return dict(passed=bool(all(out.values()) and red), detail=out, regression_goes_red=bool(red))


def c_history():
    out = {}
    ok = True
    for name in ("R35H_s0", "C0_s0"):
        c = arm(name)
        st = c.st
        ii = np.nonzero(c.hist_ok)[0][:8]
        g = torch.Generator().manual_seed(3)
        a = torch.randn(8, 1, 12, 2, generator=g) * 0.3
        z0 = c.z_now[ii].unsqueeze(1)
        hist = dict(z=c.z_hist[ii], a=c.a_hist[ii])
        with torch.no_grad():
            r3 = L.rollout_h(st, z0, a, hist, 3)[:, 0, 0]                                       # first predicted state
            zc = torch.stack([c.z_hist[ii][:, 0], c.z_hist[ii][:, 1], c.z_now[ii]], 1)
            ec = st.act_embed(torch.cat([c.a_hist[ii], a[:, 0, :1]], 1))
            direct = st.predictors[0](zc, ec)[:, -1]
            rep = dict(z=c.z_now[ii].unsqueeze(1).expand(-1, 2, -1).contiguous(), a=a[:, 0, :1].expand(-1, 2, -1).contiguous())
            r3_rep = L.rollout_h(st, z0, a, rep, 3)[:, 0, 0]                                    # the regression: repeated latent instead of the real history
            n1 = L.rollout_h(st, z0, a, None, 1)
            n3 = L.rollout_h(st, z0, a, None, 3)
            l1 = st.rollout(1, z0, a, context=1)
            l3 = st.rollout(1, z0, a, context=3)
        e = dict(real_vs_direct=float((r3 - direct).abs().max()), repeated_vs_direct=float((r3_rep - direct).abs().max()), nohist_c1_vs_library=float((n1 - l1).abs().max()),
                 nohist_c3_vs_library=float((n3 - l3).abs().max()), real_vs_repeated_first_state=float((r3 - r3_rep).abs().max()))
        e["ok"] = bool(e["real_vs_direct"] <= TOL_HIST and e["repeated_vs_direct"] > TOL_REG_RED and e["nohist_c1_vs_library"] == 0.0 and e["nohist_c3_vs_library"] == 0.0)
        out[name] = e
        ok &= e["ok"]
    return dict(passed=bool(ok), detail=out, regression_goes_red=bool(all(v["repeated_vs_direct"] > TOL_REG_RED for v in out.values())))


def _grad_cost(st, z0, goal, a, hist, ctx_n):
    """gradient of the level-1 flat goal cost (last state) with respect to the plan, through the same rollout functions the solver uses."""
    a = a.clone().requires_grad_(True)
    z0e = z0.unsqueeze(1).expand(a.shape[0], a.shape[1], -1)
    roll = L.rollout_h(st, z0e, a, hist, ctx_n) if ctx_n > 1 else st.rollout(1, z0e, a)
    cost = st.goal_cost(roll, goal.unsqueeze(1), 1, 1)
    (gr,) = torch.autograd.grad(cost.sum(), a)
    return gr


def c_reach():
    out, ok = {}, True
    for name in ARMS:
        c = arm(name)
        st = c.st
        ii = np.nonzero(c.hist_ok)[0][::230][:8]
        g = torch.Generator().manual_seed(5)
        a = (st.prior_mean_1 + st.prior_std_1 * torch.randn(len(ii), 4, 12, 2, generator=g)).detach()
        goal = c.z_fut[ii]
        for form in ("c1", "c3"):
            hist, cx = L.hist_of(c, ii, form)
            gr = _grad_cost(st, c.z_now[ii], goal, a, hist, cx)
            m = gr.abs().mean(dim=(0, 1, 2)).tolist()
            out[f"{name}|{form}"] = dict(mean_abs_grad_acc=m[0], mean_abs_grad_yaw=m[1], ok=bool(min(m) > TOL_GRAD))
            ok &= out[f"{name}|{form}"]["ok"]
        e1 = out[f"{name}|c1"]
        # the history changes the gradient (the context-3 path is live): c3 gradient differs from c1
    # zero control: the planted world with the actions disconnected from the predictor, same cost / rollout functions
    st_bad, _ = P.planted_stack(disconnect=True)
    P.planted_fit_priors(st_bad)
    st_bad.predictors[0].context = 3
    D = st_bad.cfg[0].latent_dim
    g = torch.Generator().manual_seed(9)
    z0, goal = torch.randn(8, D, generator=g), torch.randn(8, D, generator=g)
    a = torch.randn(8, 4, 12, 2, generator=g) * 0.3
    hist = dict(z=torch.randn(8, 2, D, generator=g), a=torch.randn(8, 2, 2, generator=g))
    for form, hh, cx in (("c1", None, 1), ("c3", hist, 3)):
        gr = _grad_cost(st_bad, z0, goal, a, hh, cx)
        out[f"planted_disconnected|{form}"] = dict(max_abs_grad=float(gr.abs().max()), ok=bool(float(gr.abs().max()) == 0.0))
        ok &= out[f"planted_disconnected|{form}"]["ok"]
    st_ok, _ = P.planted_stack()
    P.planted_fit_priors(st_ok)
    st_ok.predictors[0].context = 3
    for form, hh, cx in (("c1", None, 1), ("c3", hist, 3)):
        gr = _grad_cost(st_ok, z0, goal, a, hh, cx)
        out[f"planted_connected|{form}"] = dict(mean_abs_grad=float(gr.abs().mean()), ok=bool(float(gr.abs().mean()) > TOL_GRAD))
        ok &= out[f"planted_connected|{form}"]["ok"]
    return dict(passed=bool(ok), detail=out)


# ------------------------------------------------------------------------------------------- 3. planted dynamics, limit, plumbing, leak, split
def c_planted():
    out, ok = {}, True
    for variant, disc in (("connected", False), ("disconnected", True)):
        st, _ = P.planted_stack(disconnect=disc)
        P.planted_fit_priors(st)
        st.predictors[0].context = 3
        D = st.cfg[0].latent_dim

        def fn(form):
            def f(st_, kind, z0, goal, gen, last_k=None):
                if form == "c1":
                    return L.run_planner3(st_, kind, z0, goal, gen, seed_mode="prior", last_k=last_k)
                g = torch.Generator().manual_seed(11)
                h = dict(z=torch.randn(z0.shape[0], 2, D, generator=g), a=torch.randn(z0.shape[0], 2, 2, generator=g) * 0.5)
                return L.run_planner3(st_, kind, z0, goal, gen, seed_mode="prior", last_k=last_k, hist=h, context=3)
            return f
        for form in ("c1", "c3"):
            for kind, lk in (("td3", None), ("flat", 1)):
                mae, base, _, _ = P.planted_recovery(st, kind, n=64, plan_fn=fn(form), last_k=lk)
                row = dict(mae=float(mae.mean()), baseline=float(base.mean()))
                if variant == "connected":
                    row["ok"] = bool(row["mae"] <= TOL_PLANT and row["mae"] <= FACTOR * row["baseline"])
                else:
                    row["ok"] = bool(not (row["mae"] <= TOL_PLANT))           # the regression arm must FAIL the recovery tolerance
                out[f"{variant}|{form}|{kind}"] = row
                ok &= row["ok"]
    return dict(passed=bool(ok), detail=out, regression_goes_red=bool(all(v["ok"] for k, v in out.items() if k.startswith("disconnected"))))


def c_limit():
    out, ok = {}, True
    for name, form in (("R35_s0", "c1"), ("R35_s0", "c3"), ("C0_s0", "c1")):
        c = arm(name)
        idx = IDX16 if form == "c1" else np.array([i for i in IDX16 if c.hist_ok[i]])
        r = L.plan_all3(c, "flat1", "true", form, idx=idx, lam=Q.LAMBDA_LIMIT, seed_mode="hold", collect=False)
        hc = L.hold_c_of(c)[idx].numpy().astype(np.float64)
        s = Q.prior_std_orig(c.st).numpy().astype(np.float64)
        d = float((np.abs(r["actions"] - hc) / s).max())
        out[f"{name}|{form}"] = dict(max_dev_prior_std=d, ok=bool(d <= TOL_LIMIT))
        ok &= out[f"{name}|{form}"]["ok"]
    # mutant: REG off must NOT reproduce HOLD
    c = arm("R35_s0")
    r = L.plan_all3(c, "flat1", "true", "c1", idx=IDX16, lam=0.0, seed_mode="hold", collect=False)
    d = float((np.abs(r["actions"] - L.hold_c_of(c)[IDX16].numpy()) / Q.prior_std_orig(c.st).numpy()).max())
    out["mutant_reg_off"] = dict(max_dev_prior_std=d, goes_red=bool(d > TOL_LIMIT))
    return dict(passed=bool(ok and out["mutant_reg_off"]["goes_red"]), detail=out, regression_goes_red=out["mutant_reg_off"]["goes_red"])


def c_plumbing():
    out, ok = {}, True
    for name, form in (("R35_s0", "c1"), ("R35_s0", "c3"), ("C0_s0", "c3")):
        c = arm(name)
        idx = IDX16 if form == "c1" else np.array([i for i in IDX16 if c.hist_ok[i]])
        res = {}
        for gname in ("true", "rand"):
            a = L.plan_all3(c, "flat1", gname, form, idx=idx, lam=0.1)["actions"]
            b = L.plan_all3(c, "flat1", gname, form, idx=idx, lam=0.1)["actions"]
            hist, cx = L.hist_of(c, idx, form)
            src = {"true": c.z_fut[idx], "rand": c.z_fut[torch.as_tensor(c.donor[idx])]}[gname]
            with torch.no_grad():
                g = c.st.initial_latents(src, 1)[0]
            d = L.run_planner3(c.st, "flat1", c.z_now[idx], g, P.make_gen(0, 0), hold_c=L.hold_c_of(c)[idx], lam=0.1, hist=hist, context=cx)
            res[gname] = (a, np.array_equal(a, b), np.array_equal(a, d.actions.detach().numpy().astype(np.float64)))
        e = dict(bit_repeat=res["true"][1] and res["rand"][1], direct_call_equal=res["true"][2] and res["rand"][2], true_ne_rand=not np.array_equal(res["true"][0], res["rand"][0]))
        # seed mutant
        s1 = L.plan_all3(c, "flat1", "true", form, idx=idx, lam=0.1, seed=1)["actions"]
        e["seed_mutant_differs"] = not np.array_equal(s1, res["true"][0])
        # no-leak: garbage FUTURE logged actions (knots 1..15) on every window change nothing (only a0 enters, through HOLD_c, and the past actions come from the cache, not from A)
        saved = c.A.copy()
        c.A[:, 1:] = np.random.default_rng(0).normal(size=c.A[:, 1:].shape) * 5.0
        g2 = L.plan_all3(c, "flat1", "true", form, idx=idx, lam=0.1)["actions"]
        c.A[:] = saved
        e["no_leak_future_actions_garbage"] = bool(np.array_equal(g2, res["true"][0]))
        # leak mutant: garbage a0 (the one logged action that does enter, as HOLD) DOES change the plan
        c.A[:, 0] = np.random.default_rng(1).normal(size=c.A[:, 0].shape) * 5.0
        g3 = L.plan_all3(c, "flat1", "true", form, idx=idx, lam=0.1)["actions"]
        c.A[:] = saved
        e["leak_mutant_a0_changes_plan"] = bool(not np.array_equal(g3, res["true"][0]))
        e["ok"] = all(e.values())
        out[f"{name}|{form}"] = e
        ok &= e["ok"]
    return dict(passed=bool(ok), detail=out)


def c_split():
    c = arm("R35_s0")
    tune, score = set(c.tune_idx.tolist()), set(c.score_idx.tolist())
    ok = bool(not (tune & score) and set(c.eid[c.tune_idx]).isdisjoint(set(c.eid[c.score_idx])))
    # the PLANTEST2 split of the banked run: same index sets
    import plantest2_real as R2  # noqa: F401  (imports plantest_real, which needs its own tree: skipped if it refuses)
    return dict(passed=ok, detail=dict(n_tune=len(tune), n_score=len(score), episodes_tune=len(set(c.eid[c.tune_idx])), episodes_score=len(set(c.eid[c.score_idx]))))


def c_split_banked():
    c = arm("R35_s0")
    b = np.load(f"{TH}/plantest2/runs/score_M2b_s0_flat1_true_L1_p0.npz")["idx"]
    return dict(passed=bool(np.array_equal(b, c.score_idx)), detail=dict(n=int(len(b))))


def c_macro_action_only():
    """The TRAIN-prior shortcut: the macro-action embeddings do not depend on the level-1 latents."""
    c = arm("C0_s0")
    st = c.st
    tr = P.load_latents(L.TRAIN, "M2b_s0")
    A = c.wm.scale_actions(tr["A"][:256])[:, :-1]
    with torch.no_grad():
        e0 = st.encode(torch.zeros(256, 16, c.D), A)
        e1 = st.encode(torch.randn(256, 16, c.D), A)
        e2 = st.encode(tr["Z1"][:256], A)
    d = max(float((e0.act[l] - e1.act[l]).abs().max()) for l in (1, 2)), max(float((e0.act[l] - e2.act[l]).abs().max()) for l in (1, 2))
    return dict(passed=bool(max(d) == 0.0), detail=dict(zeros_vs_random=d[0], zeros_vs_real=d[1]))


def c_stride():
    """The wrong-stride regressions: argv strides replaced -> LadderMismatch; td3_defined flips when fed the other arm's built strides; TD3 on a (3, 5) model does not run."""
    out = {}
    r, cz = arm("R35_s0"), arm("C0_s0")
    out["td3_defined_R35"], out["td3_defined_C0"] = L.td3_defined(r.built), L.td3_defined(cz.built)
    argv_bad = dict(L.LR.parse_argv(open(f"{L.LR.PULLS}/ladder_R35_s0/argv.txt", encoding="utf-8").read()))
    argv_bad["wm-strides"] = ["2", "2"]
    try:
        L.LR.check_built_vs_argv(r.built, argv_bad)
        out["wrong_argv_stride_raises"] = False
    except L.LR.LadderMismatch:
        out["wrong_argv_stride_raises"] = True
    swapped = dict(r.built)
    swapped["cum_stride"] = cz.built["cum_stride"]
    out["td3_defined_flips_on_wrong_stride"] = bool(L.td3_defined(swapped) != L.td3_defined(r.built))
    idx = IDX16[:4]
    try:
        L.plan_all3(r, "td3", "true", "c1", idx=idx, lam=0.0, seed_mode="prior", collect=False)
        out["td3_on_35_runs_anyway"] = True
    except Exception as e:                                                    # noqa: BLE001
        out["td3_on_35_runs_anyway"] = False
        out["td3_on_35_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    out["reason_R35"] = L.td3_reason(r.built)
    ok = (not out["td3_defined_R35"]) and out["td3_defined_C0"] and out["wrong_argv_stride_raises"] and out["td3_defined_flips_on_wrong_stride"]
    return dict(passed=bool(ok), detail=out, regression_goes_red=bool(out["wrong_argv_stride_raises"] and out["td3_defined_flips_on_wrong_stride"]))


CHECKS = dict(repro=c_repro, latent=c_latent_route, macro=c_macro_action_only, reduction=c_reduction, history=c_history, reach=c_reach, planted=c_planted, limit=c_limit, plumbing=c_plumbing,
              split=c_split_banked, stride=c_stride)


def main(names):
    res = {}
    for n in names or CHECKS:
        t = time.time()
        try:
            res[n] = CHECKS[n]()
        except Exception:                                                     # noqa: BLE001
            res[n] = dict(passed=False, error=traceback.format_exc()[-1500:])
        res[n]["wall_s"] = round(time.time() - t, 1)
        print(n, "PASS" if res[n]["passed"] else "FAIL", res[n]["wall_s"], "s", flush=True)
        if not res[n]["passed"]:
            print(json.dumps(res[n], indent=1, default=str)[:2500], flush=True)
    part = "" if not names else "_" + "_".join(names)
    json.dump(res, open(f"{HERE}/CONTROLS_LADDER_PLANTEST{part}.json", "w"), indent=1, default=str)
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
