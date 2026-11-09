"""LADDER PLANTEST library: PLANTEST (PREREG_V8F_PLANTEST) and PLANTEST2 (PREREG_V8F_PLANTEST2) on the banked ladder world models. Tier T0 (oracle-goal diagnostic), zero training, CPU.
Imports the frozen gates' code UNCHANGED (plantest/plantest_lib.py, plantest2/plantest2_lib.py); adds only (1) a loader that builds a ladder arm's latents from the BUILT model,
(2) plan_topdown3 = a copy of plantest2_lib.plan_topdown2 whose LEVEL-1 rollout can be conditioned on REAL history states (context 3), and (3) the TD3-defined test.
Code root: the ladder tree (commit 1faa1a06), LADDER_READ_CODE_ROOT. See IMPL_CHOICES_LADDER_PLANTEST.md."""
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.normpath(os.path.join(HERE, ".."))
CODE_ROOT = os.environ.get("LADDER_READ_CODE_ROOT", "F:/TanitAD-scratch/ladder_read/tree")
os.environ.setdefault("WMACT_CODE_ROOT", CODE_ROOT)           # wmact_lib only asserts that tanitad was imported from here
for _d in ("ladder_read", "plantest", "plantest2", "wmact"):
    sys.path.insert(0, os.path.join(THEORY, _d))
from types import SimpleNamespace
import numpy as np
import torch
import lr_lib as LR                      # sets sys.path to the ladder tree and asserts tanitad.__file__ is under it
import plantest_lib as P
import plantest2_lib as Q
from tanitad.models.hjepa import PlanResult, donor_indices

ART = LR.ART
EVAL = f"{THEORY}/wmact/latents.pt"
TRAIN = f"{THEORY}/wmact/latents_train.pt"
TRACKS = f"{ART}/sfresh_inputs/inputs/native_tracks.pt"
FRAMES = f"{LR.SCR}/cache"
NH = 2                                    # real history states of the context-3 form (the encoded latents at t-2 and t-1)
CTX_FORMS = ("c1", "c3")
M2B_CK = {"M2b_s0": f"{ART}/sfresh_inputs/inputs/M2b_s0/ckpt_last.pt", "M2b_s1": f"{ART}/sfresh_inputs/inputs/M2b_s1/ckpt_last.pt"}
TD3_CUM = [1, 2, 4]                       # TD3 as the card defines it needs level steps of 0.2 / 0.4 / 0.8 s: cumulative strides (in 0.2 s frames) 1, 2, 4


def seed_of(name):
    return int(name.rsplit("_s", 1)[1])


# ----------------------------------------------------------------------------------------------- TD3 defined on this built model?
def td3_defined(built):
    """Read from the BUILT cumulative strides, never from M2b's numbers. The card's TD3: goal in level 3, top horizon 3 steps = 2.4 s, level 2 matches 3 level-3 states, level 1 plans 12 actions (H_2 = 3 * 2, H_1 = 6 * 2).
    That needs cumulative strides (1, 2, 4) frames. On (3, 5) the level-3 step is 15 frames = 3.0 s: 2.4 s is not a level-3 horizon."""
    return list(built["cum_stride"]) == TD3_CUM


def td3_reason(built):
    cs = list(built["cum_stride"])
    return (f"built cumulative strides {cs} frames: the level-3 step is {cs[2]} frames = {cs[2] * 0.2:.1f} s, so the card's 2.4 s goal (12 frames) is not on level 3's grid and its top horizon of 3 steps "
            f"would be {3 * cs[2] * 0.2:.1f} s; the level-2 step is {cs[1] * 0.2:.1f} s so the card's 6-state sub-plan (H_2 = 6) is {6 * cs[1] * 0.2:.1f} s, not 1.2 s")


# ----------------------------------------------------------------------------------------------- the real-history rollout
def rollout_h(st, z0, a, hist=None, context=1):
    """Level-1 rollout of plan ``a`` (b, S, H, 2) from ``z0`` (b, S, D) conditioned on the last ``context`` states.
    hist = None: identical to ``st.rollout(1, z0, a, context=context)`` (history grown from the rollout's own states, the library form).
    hist = dict(z (b, NH, D), a (b, NH, 2)) with NH = context - 1: the first context states are the REAL encoded latents at t-NH .. t-1 and the logged (measured) actions taken at them; the plan's
    actions follow. Past data only: nothing at or after NOW except the plan reaches the predictor."""
    pred = st.predictors[0]
    cap = getattr(pred, "context", None)
    if cap is not None and context > cap:
        raise ValueError(f"context {context} exceeds the level-1 predictor context {cap}")
    b, S, H, _ = a.shape
    n = b * S
    e = st.act_embed(a.reshape(n, H, -1))
    zs, es = [], []
    if hist is not None:
        nh = hist["z"].shape[1]
        if nh != context - 1:
            raise ValueError(f"history has {nh} states, context {context} needs {context - 1}")
        zh = hist["z"].unsqueeze(1).expand(b, S, nh, -1).reshape(n, nh, -1)
        eh = st.act_embed(hist["a"]).unsqueeze(1).expand(b, S, nh, -1).reshape(n, nh, -1)
        zs += [zh[:, i] for i in range(nh)]
        es += [eh[:, i] for i in range(nh)]
    else:
        nh = 0
    zs.append(z0.reshape(n, -1))
    for h in range(H):
        es.append(e[:, h])
        zc = torch.stack(zs[-context:], dim=1)
        ec = torch.stack(es[-context:], dim=1)
        zs.append(pred(zc, ec)[:, -1])
    out = torch.stack(zs[nh + 1:], dim=1)
    return out.reshape(b, S, H, -1)


# ----------------------------------------------------------------------------------------------- the solver (copy of Q.plan_topdown2 with the level-1 hook)
def plan_topdown3(st, z1_0, goal, *, top_horizon, goal_level, num_candidates, n_iters, lr, lam, subgoal_K, patience, rel_tol, last_k, generator, seeds=None, reg=None, hist=None, context=1):
    """As Q.plan_topdown2 (seeds, reg), plus hist / context for the LEVEL-1 rollout only (every other level keeps context 1 as in the cards). hist None and context 1 consume the generator identically and
    return bit-identical plans to plan_topdown2 (control G)."""
    top = goal_level or st.n_levels
    b = z1_0.shape[0]
    was_training = st.training
    st.eval()

    def roll_level(level, d_lv, a):
        z0e = d_lv.unsqueeze(1).expand(b, a.shape[1], -1)
        if level == 1 and context > 1:
            return z0e, rollout_h(st, z0e, a, hist, context)
        return z0e, st.rollout(level, z0e, a)

    try:
        with torch.enable_grad():
            z0 = [z.detach() for z in st.initial_latents(z1_0.detach(), top)]
            goal = goal.detach()
            res = PlanResult(actions=z1_0)
            sub_k = dict(subgoal_K or {})

            def _opt(level, horizon, cost_fn, n_cand):
                c = st.cfg[level - 1]
                dev, dt = z1_0.device, z1_0.dtype
                mean = getattr(st, f"prior_mean_{level}").to(dev, dt)
                std = getattr(st, f"prior_std_{level}").to(dev, dt)
                lo = getattr(st, f"prior_lo_{level}").to(dev, dt)
                hi = getattr(st, f"prior_hi_{level}").to(dev, dt)
                noise = torch.randn(b, n_cand, horizon, c.action_dim, device=dev, dtype=dt, generator=generator)
                if seeds is not None and level in seeds:
                    sd = seeds[level].to(dev, dt)
                    if tuple(sd.shape) != (b, horizon, c.action_dim):
                        raise ValueError(f"seed level {level}: {tuple(sd.shape)} != {(b, horizon, c.action_dim)}")
                    init = sd.unsqueeze(1) + std * noise
                    init[:, 0] = sd
                else:
                    init = mean + std * noise
                    init[:, 0] = mean.expand(b, horizon, c.action_dim)
                init = torch.max(torch.min(init, hi), lo)
                lr_l = lr[level] if isinstance(lr, dict) else lr
                return st._optimize(cost_fn, init, lr=float(lr_l), n_iters=n_iters, patience=patience, rel_tol=rel_tol, weight_decay=0.0, lo=lo, hi=hi, project=None)

            for level in range(top, 0, -1):
                n_cand = num_candidates[level] if isinstance(num_candidates, dict) else int(num_candidates)
                d_lv = z0[level - 1]
                if level == top:
                    horizon = int(top_horizon)

                    def cost_fn(a, _lv=level, _d=d_lv):
                        _, roll = roll_level(_lv, _d, a)
                        return st.goal_cost(roll, goal.unsqueeze(1), _lv, last_k)
                else:
                    up = st.cfg[level]
                    K = int(sub_k.get(level, 1))
                    if K < 1 or K > res.level_rollouts[level + 1].shape[1]:
                        raise ValueError(f"subgoal_K[{level}]={K} outside 1..H_{level + 1}")
                    horizon = K * up.stride + up.window - 1
                    sub = res.level_rollouts[level + 1][:, :K].detach()

                    def cost_fn(a, _lv=level, _d=d_lv, _sub=sub, _K=K):
                        z0e, roll = roll_level(_lv, _d, a)
                        full = torch.cat([z0e.unsqueeze(2), roll], dim=2)
                        lifted = st._apply_encoder(_lv + 1, full, strict=True)[:, :, 1:]
                        dd = (lifted - _sub.unsqueeze(1)).pow(2).sum(-1)
                        return dd[..., -1] + lam * dd[..., :-1].sum(-1)

                if level == 1 and reg is not None and reg["lam"] != 0.0:
                    if horizon != reg["ref"].shape[1]:
                        raise ValueError("REG reference horizon != level-1 horizon")
                    cost_base = cost_fn

                    def cost_fn(a, _base=cost_base):
                        return _base(a) + Q.reg_term(a, reg["ref"], reg["std"], reg["lam"])

                best_a, best_cost, init_cost, hist_c, nit = _opt(level, horizon, cost_fn, n_cand)
                pick = best_cost.argmin(dim=1)
                ar = torch.arange(b, device=pick.device)
                chosen = best_a[ar, pick]
                with torch.no_grad():
                    if level == 1 and context > 1:
                        roll = rollout_h(st, d_lv.unsqueeze(1), chosen.unsqueeze(1), hist, context)[:, 0]
                    else:
                        roll = st.rollout(level, d_lv, chosen)
                res.level_actions[level] = chosen.detach()
                res.level_rollouts[level] = roll.detach()
                res.costs[level] = best_cost[ar, pick].detach()
                res.init_costs[level] = init_cost.detach()
                res.history[level] = hist_c
                res.iters[level] = nit
            res.actions = res.level_actions[1]
            return res
    finally:
        st.train(was_training)


def run_planner3(st, kind, z1_0, goal, gen, hold_c=None, lam=0.0, seed_mode="hold", last_k=None, hist=None, context=1):
    """kind td3 | flat | flat1. seed_mode 'prior' = PLANTEST's initialisation (lam must be 0); 'hold' = PLANTEST2's SEED (+ REG when lam > 0). hist / context: the level-1 context form."""
    if kind == "flat1":
        kind, last_k = "flat", 1
    if seed_mode == "prior":
        assert lam == 0.0, "REG needs the HOLD seed"
        seeds, reg = None, None
    else:
        seeds = Q.hold_seeds(st, hold_c, z1_0, kind)
        reg = dict(lam=float(lam), ref=hold_c, std=Q.prior_std_orig(st)) if lam != 0.0 else None
    kw = dict(patience=P.PATIENCE, rel_tol=P.REL_TOL, last_k=last_k, generator=gen, seeds=seeds, reg=reg, hist=hist, context=context)
    if kind == "td3":
        return plan_topdown3(st, z1_0, goal, top_horizon=P.TOP_H, goal_level=3, num_candidates=P.S_CAND, n_iters=P.N_ITERS, lr=P.LR, lam=P.LAM, subgoal_K=P.SUBGOAL_K, **kw)
    if kind == "flat":
        return plan_topdown3(st, z1_0, goal, top_horizon=P.H1, goal_level=1, num_candidates=P.S_CAND, n_iters=P.N_ITERS, lr=P.LR, lam=0.0, subgoal_K=None, **kw)
    raise ValueError(kind)


def hist_of(ctx, ii, form):
    if form == "c1":
        return None, 1
    return dict(z=ctx.z_hist[ii], a=ctx.a_hist[ii]), NH + 1


def plan_all3(ctx, kind, gname, form="c1", idx=None, seed=P.PLAN_SEED, batch=P.BATCH, collect=True, lam=0.0, seed_mode="hold"):
    """As Q.plan_all2 (fixed consecutive batches of idx, batch j's generator = make_gen(seed, j), identical across goals and lambdas) with the context form. For form c3, idx must lie in ctx.hist_ok."""
    st = ctx.st
    k = "flat" if kind == "flat1" else kind
    n = ctx.z_now.shape[0]
    idx = np.arange(n) if idx is None else np.asarray(idx)
    if form == "c3":
        assert bool(ctx.hist_ok[idx].all()), "context 3 needs the two past frames"
    hold_all = hold_c_of(ctx) if seed_mode == "hold" else None
    out = dict(actions=[], cost={}, init={}, iters={}, mono={})
    goal_lv = 3 if k == "td3" else 1
    for j, s0 in enumerate(range(0, len(idx), batch)):
        ii = idx[s0:s0 + batch]
        zn = ctx.z_now[ii]
        g = P.goal_for(st, k, gname, ctx.z_fut[ii], zn, ctx.z_fut[torch.as_tensor(np.asarray(ctx.donor)[ii])])
        hist, cx = hist_of(ctx, ii, form)
        hc = None if hold_all is None else hold_all[ii]
        res = run_planner3(st, kind, zn, g, P.make_gen(seed, j), hold_c=hc, lam=lam, seed_mode=seed_mode, hist=hist, context=cx)
        out["actions"].append(res.actions.detach())
        for lv in res.costs:
            out["cost"].setdefault(lv, []).append(res.costs[lv]); out["init"].setdefault(lv, []).append(res.init_costs[lv])
            out["iters"].setdefault(lv, []).append(torch.full((len(ii),), float(res.iters[lv])))
        if collect:
            with torch.no_grad():
                z0s = st.initial_latents(zn, goal_lv)
                for lv, roll in res.level_rollouts.items():
                    full = torch.cat([z0s[lv - 1].unsqueeze(1), roll], 1)
                    lifted = st.lift(full, lv, goal_lv, strict=True) if lv < goal_lv else full
                    d = (lifted - g.unsqueeze(1)).pow(2).sum(-1)
                    out["mono"].setdefault(lv, []).append(P.monotonicity(d))
    r = dict(actions=torch.cat(out["actions"]).numpy().astype(np.float64))
    for key in ("cost", "init", "iters", "mono"):
        for lv, v in out[key].items():
            if v:
                r[f"{key}{lv}"] = torch.cat(v).numpy().astype(np.float64)
    return r


def hold_c_of(ctx):
    """HOLD_c from the first logged action of every window (scaled units), clipped to the level-1 box (plantest2_real.hold_c_of)."""
    return Q.hold_scaled(ctx.wm, ctx.A[:, :1], ctx.st.prior_lo_1, ctx.st.prior_hi_1)


def to_raw(ctx, a):
    out = np.array(a, np.float64)
    out[..., 0] *= 2.0                    # WM_ACC_SCALE
    out[..., 1] *= ctx.ch1
    return out


def unit3(ctx, kind, gname, form, idx, lam, seed=0, seed_mode="hold", collect=True):
    import time
    t = time.time()
    r = plan_all3(ctx, kind, gname, form, idx=idx, seed=seed, collect=collect, lam=lam, seed_mode=seed_mode)
    r["plan_raw"] = to_raw(ctx, r.pop("actions"))
    r["idx"] = np.asarray(idx)
    r["wall_s"] = np.array(time.time() - t)
    return r


# ----------------------------------------------------------------------------------------------- data: an arm's latents from the BUILT model
def fit_priors3(st, wm, tr_A, upper, tr_Z1=None):
    """Level-1 prior from the scaled logged TRAIN actions (the 2,397 TRAIN windows of wmact/latents_train.pt, 15 per window, as plantest_real.fit_priors). Levels 2 and 3 (macro-action embeddings of the SAME
    TRAIN windows) only where TD3 is defined; ``tr_Z1`` is the level-1 sequence passed to ``encode`` (the macro-action path reads the actions only, checked in controls_lp.py)."""
    A_sc = wm.scale_actions(tr_A)[:, :-1]
    P.apply_prior(st, 1, A_sc)
    if upper:
        with torch.no_grad():
            enc = st.encode(tr_Z1, A_sc)
        P.apply_prior(st, 2, enc.act[1])
        P.apply_prior(st, 3, enc.act[2])


def clip_z1(wm, fr):
    """{clip: [T, D]} level-1 latents of the WHOLE clip through the built model: wm.z1_from_mean (proj, enc1 when K = 3) of the frozen-adapter mean tokens (every frame has its real history)."""
    out = {}
    with torch.no_grad():
        for c, zm in zip(fr["clips"], fr["zmean"]):
            out[c] = wm.z1_from_mean(torch.as_tensor(zm)[None])[0].float()
    return out


def build_ctx(wm, fr, tag, ev, tr, upper, z_override=None):
    """Window arrays from the whole-clip latents. Knot k of a window = frame g + k, g = t - 15 (PLANTEST: NOW = knot 0, the goal = knot 12). z_hist = latents at g-2, g-1; a_hist = the logged measured
    (a, yaw rate) at g-2, g-1 scaled by the WM; hist_ok = g >= 2 (windows without both past frames are not scored under context 3)."""
    st = wm.stack.eval()
    n = len(ev["eid"])
    z1c = clip_z1(wm, fr)
    act = dict(zip(fr["clips"], fr["act"]))
    D = next(iter(z1c.values())).shape[-1]
    Zw = torch.zeros(n, 16, D)
    Zh = torch.zeros(n, NH, D)
    Ah = np.zeros((n, NH, 2), np.float32)
    hist_ok = np.zeros(n, bool)
    a_mis = 0.0
    for i, (e, t) in enumerate(zip(ev["eid"], ev["t"])):
        g = int(t) - 15
        z = z1c[str(e)]
        Zw[i] = z[g:g + 16]
        a_mis = max(a_mis, float(np.abs(act[str(e)][g:g + 16] - ev["A"][i].numpy()).max()))
        if g >= NH:
            hist_ok[i] = True
            Zh[i] = z[g - NH:g]
            Ah[i] = act[str(e)][g - NH:g]
    if z_override is not None:
        Zw = z_override
    TRK = torch.load(TRACKS, weights_only=False)["tracks"]
    v0 = np.full(n, np.nan)
    for i, (e, t) in enumerate(zip(ev["eid"], ev["t"])):
        g = int(t) - 15
        r = TRK.get(str(e))
        if r is not None and g >= 0 and 2 * g < len(r):
            v0[i] = float(np.asarray(r)[2 * g, 3])
    codes = {e: i for i, e in enumerate(sorted(set(ev["eid"])))}
    ep_int = torch.tensor([codes[e] for e in ev["eid"]])
    donor = donor_indices(n, episode_ids=ep_int, seed=0).numpy()
    ctx = SimpleNamespace(tag=tag, wm=wm, st=st, ch1=float(wm.ch1_scale), z_now=Zw[:, 0], z_fut=Zw[:, 12], Zw=Zw, z_hist=Zh, a_hist=wm.scale_actions(torch.as_tensor(Ah)).float(), hist_ok=hist_ok,
                          A=ev["A"].numpy().astype(np.float64), eid=ev["eid"], t=ev["t"], v0=v0, donor=donor, n=n, v0_nan=int(np.isnan(v0).sum()), action_max_abs_diff_vs_cache=a_mis, D=D)
    fit_priors3(st, wm, tr["A"], upper, tr["Z1"] if upper else None)
    ctx.tune_idx, ctx.score_idx = Q.split_indices(ctx.eid)
    ctx.std_raw = Q.std_raw_of(st, ctx.ch1)
    return ctx


def prepare_arm(name):
    """A ladder arm (``C0_s0``, ``R35_s1`` ...): load, the BUILT-equals-argv check (raises LadderMismatch), the adapter hash against the cached frames, then the windows."""
    model, cfg, argv, ck = LR.load_arm(f"{LR.PULLS}/ladder_{name}")
    built = LR.built_values(model)
    checked = LR.check_built_vs_argv(built, argv)
    fr = LR.load_eval_frames(f"{FRAMES}/frames_s{seed_of(name)}.npz")
    sha = LR.adapter_sha(model)
    if sha != fr["adapter_sha"]:
        raise LR.LadderMismatch(f"{name}: adapter hash {sha[:12]} != cached frames {fr['adapter_sha'][:12]}")
    if int(argv["seed"][0]) != seed_of(name):
        raise LR.LadderMismatch(f"{name}: argv --seed {argv['seed']}")
    ev, tr = P.load_latents(EVAL, "M2b_s0"), P.load_latents(TRAIN, "M2b_s0")
    ctx = build_ctx(model.wm, fr, name, ev, tr, td3_defined(built))
    ctx.built, ctx.checked, ctx.adapter_sha, ctx.td3_defined = built, checked, sha, td3_defined(built)
    return ctx


def prepare_m2b(tag, whole_clip=False):
    """M2b through THIS code path for the reproduction control. whole_clip False: the level-1 latents of wmact/latents.pt (the ones the banked runs used); True: the whole-clip route above."""
    import wmact_lib as W
    m, cfg, ck = W.load_model(M2B_CK[tag])
    wm = m.wm
    ev, tr = P.load_latents(EVAL, tag), P.load_latents(TRAIN, tag)
    seed = seed_of(tag)
    fr = LR.load_eval_frames(f"{FRAMES}/frames_s{seed}.npz")
    ctx = build_ctx(wm, fr, tag, ev, tr, True, z_override=None if whole_clip else ev["Z1"].clone())     # priors: levels 2 / 3 on the REAL TRAIN latents, as the banked PLANTEST
    ctx.whole_clip = whole_clip
    ctx.Z1_window = ev["Z1"]
    return ctx


def sel_arrays(ctx, form):
    """Window index sets of the three reads: ALL (the cards' set), TUNE and SCORE; context 3 keeps those with both past frames."""
    ok = np.ones(ctx.n, bool) if form == "c1" else ctx.hist_ok
    return dict(all=np.nonzero(ok)[0], tune=np.array([i for i in ctx.tune_idx if ok[i]]), score=np.array([i for i in ctx.score_idx if ok[i]]))
