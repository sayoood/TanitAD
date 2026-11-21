"""PREREG_V8F_PLANTEST2 library: PLANTEST's planner (plantest/plantest_lib.py, imported unchanged) plus SEED and REG(lambda).
Tier T0 (oracle-goal diagnostic). Zero training. CPU. The HJEPAStack in the read-only export is not touched: plan_topdown2 below is a copy of HJEPAStack.plan_topdown with two hooks
(seeds, reg); with seeds=None and reg=None it consumes the generator identically and returns bit-identical plans (control 1)."""
import hashlib, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plantest"))
import numpy as np
import torch
import plantest_lib as P
from tanitad.models.hjepa import PlanResult

SALT = "plantest2-v1"
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
LAMBDA_LIMIT = 1.0e6                  # control 2
COLLAPSE_FRAC = 0.10
FAMS = {"LONGITUDINAL": "acc_mae", "LATERAL": "yaw_mae"}
WIDEN = P.WIDEN


# ----------------------------------------------------------------------------------------------- the split (IMPL section 5)
def split_episodes(eids, salt=SALT):
    """Stable hash split of the distinct clip ids: rank by sha256(salt|id), the first floor(n/2) are TUNE, the rest SCORE. Independent of window order."""
    uniq = sorted(set(str(e) for e in eids))
    key = sorted(uniq, key=lambda e: (hashlib.sha256(f"{salt}|{e}".encode()).hexdigest(), e))
    n_tune = len(key) // 2
    tune, score = set(key[:n_tune]), set(key[n_tune:])
    assert not (tune & score) and len(tune) + len(score) == len(uniq)
    return tune, score


def split_indices(eids, salt=SALT):
    tune, score = split_episodes(eids, salt)
    e = np.asarray([str(x) for x in eids])
    it = np.nonzero(np.isin(e, list(tune)))[0]
    isc = np.nonzero(np.isin(e, list(score)))[0]
    assert len(np.intersect1d(it, isc)) == 0 and set(e[it]).isdisjoint(set(e[isc]))
    return it, isc


# ----------------------------------------------------------------------------------------------- HOLD in the planner's units
def hold_scaled(wm, A_raw, lo=None, hi=None):
    """HOLD (a0 held 12 steps) in the level-1 scaled planning units (a / 2, yaw rate / ch1_scale = wm.scale_actions); only the FIRST logged action of a window is read here.
    lo, hi: the level-1 box of the prior; HOLD_c = HOLD clipped to it (the feasible point the planner can reach)."""
    a0 = torch.as_tensor(np.asarray(A_raw)[:, :1]).float()
    h = wm.scale_actions(a0).repeat(1, P.H1, 1)
    if lo is not None:
        h = torch.max(torch.min(h, hi), lo)
    return h


def prior_std_orig(st):
    return (st.prior_std_1.detach() / WIDEN)


def hold_seeds(st, hold_c, z_now, kind):
    """Level-1 seed = HOLD_c; the upper levels of TD3 seed from the macro encoding of the HOLD sequence (st.encode on 13 copies of the start latent: the macro-action path reads the actions only)."""
    seeds = {1: hold_c}
    if kind == "td3":
        with torch.no_grad():
            z_seq = z_now.unsqueeze(1).expand(-1, P.H1 + 1, -1)
            enc = st.encode(z_seq, hold_c)
        seeds[2], seeds[3] = enc.act[1].detach(), enc.act[2].detach()
    return seeds


# ----------------------------------------------------------------------------------------------- the solver (copy of HJEPAStack.plan_topdown + 2 hooks)
def plan_topdown2(st, z1_0, goal, *, top_horizon, goal_level, num_candidates, n_iters, lr, lam, subgoal_K, patience, rel_tol, last_k, generator, seeds=None, reg=None):
    """seeds: {level: (B, H, A)} replaces the prior mean as the centre of the candidates (candidate 0 = the seed, the others seed + prior_std * N(0,1), clipped to the box); None = the prior mean (PLANTEST).
    reg: None or dict(lam, ref (B,12,2) scaled level-1 reference, std (2,)): adds lam * mean(((a - ref) / std)^2) over steps and channels to the LEVEL-1 cost of every candidate."""
    top = goal_level or st.n_levels
    b = z1_0.shape[0]
    was_training = st.training
    st.eval()
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
                        z0e = _d.unsqueeze(1).expand(b, a.shape[1], -1)
                        roll = st.rollout(_lv, z0e, a)
                        return st.goal_cost(roll, goal.unsqueeze(1), _lv, last_k)
                else:
                    up = st.cfg[level]
                    K = int(sub_k.get(level, 1))
                    if K < 1 or K > res.level_rollouts[level + 1].shape[1]:
                        raise ValueError(f"subgoal_K[{level}]={K} outside 1..H_{level + 1}")
                    horizon = K * up.stride + up.window - 1
                    sub = res.level_rollouts[level + 1][:, :K].detach()

                    def cost_fn(a, _lv=level, _d=d_lv, _sub=sub, _K=K):
                        z0e = _d.unsqueeze(1).expand(b, a.shape[1], -1)
                        roll = st.rollout(_lv, z0e, a)
                        full = torch.cat([z0e.unsqueeze(2), roll], dim=2)
                        lifted = st._apply_encoder(_lv + 1, full, strict=True)[:, :, 1:]
                        dd = (lifted - _sub.unsqueeze(1)).pow(2).sum(-1)
                        return dd[..., -1] + lam * dd[..., :-1].sum(-1)

                if level == 1 and reg is not None and reg["lam"] != 0.0:
                    if horizon != reg["ref"].shape[1]:
                        raise ValueError("REG reference horizon != level-1 horizon")
                    cost_base = cost_fn

                    def cost_fn(a, _base=cost_base):
                        return _base(a) + reg_term(a, reg["ref"], reg["std"], reg["lam"])

                best_a, best_cost, init_cost, hist, nit = _opt(level, horizon, cost_fn, n_cand)
                pick = best_cost.argmin(dim=1)
                ar = torch.arange(b, device=pick.device)
                chosen = best_a[ar, pick]
                with torch.no_grad():
                    roll = st.rollout(level, d_lv, chosen)
                res.level_actions[level] = chosen.detach()
                res.level_rollouts[level] = roll.detach()
                res.costs[level] = best_cost[ar, pick].detach()
                res.init_costs[level] = init_cost.detach()
                res.history[level] = hist
                res.iters[level] = nit
            res.actions = res.level_actions[1]
            return res
    finally:
        st.train(was_training)


def reg_term(a, ref, std, lam):
    """lam * mean over (steps, channels) of ((a - HOLD_c) / prior std)^2, per window and candidate. a (B,S,12,2), ref (B,12,2), std (2,)."""
    d = (a - ref.unsqueeze(1)) / std
    return lam * d.pow(2).mean(dim=(-1, -2))


def run_planner2(st, kind, z1_0, goal, gen, hold_c=None, lam=0.0, seed_mode="hold", last_k=None):
    """kind td3 | flat | flat1 (flat1 = flat with last_k 1). hold_c (B,12,2): HOLD_c in scaled units. seed_mode 'prior' = PLANTEST's initialisation (hold_c unused, REG must be off)."""
    if kind == "flat1":
        kind, last_k = "flat", 1
    if seed_mode == "prior":
        assert lam == 0.0, "REG needs the HOLD seed"
        seeds, reg = None, None
    else:
        seeds = hold_seeds(st, hold_c, z1_0, kind)
        reg = dict(lam=float(lam), ref=hold_c, std=prior_std_orig(st)) if lam != 0.0 else None
    if kind == "td3":
        return plan_topdown2(st, z1_0, goal, top_horizon=P.TOP_H, goal_level=3, num_candidates=P.S_CAND, n_iters=P.N_ITERS, lr=P.LR, lam=P.LAM, subgoal_K=P.SUBGOAL_K,
                             patience=P.PATIENCE, rel_tol=P.REL_TOL, last_k=last_k, generator=gen, seeds=seeds, reg=reg)
    if kind == "flat":
        return plan_topdown2(st, z1_0, goal, top_horizon=P.H1, goal_level=1, num_candidates=P.S_CAND, n_iters=P.N_ITERS, lr=P.LR, lam=0.0, subgoal_K=None,
                             patience=P.PATIENCE, rel_tol=P.REL_TOL, last_k=last_k, generator=gen, seeds=seeds, reg=reg)
    raise ValueError(kind)


def plan_all2(st, kind, gname, z1_now, z1_future, donor, hold_c_all, idx=None, seed=P.PLAN_SEED, batch=P.BATCH, collect=True, lam=0.0, seed_mode="hold"):
    """As plantest_lib.plan_all (fixed consecutive batches of idx, generator of batch j = make_gen(seed, j), identical across goals and lambdas), with SEED / REG."""
    k = "flat" if kind == "flat1" else kind
    n = z1_now.shape[0]
    idx = np.arange(n) if idx is None else np.asarray(idx)
    out = dict(actions=[], cost={}, init={}, iters={}, mono={})
    goal_lv = 3 if k == "td3" else 1
    for j, s0 in enumerate(range(0, len(idx), batch)):
        ii = idx[s0:s0 + batch]
        zn = z1_now[ii]
        g = P.goal_for(st, k, gname, z1_future[ii], zn, z1_future[torch.as_tensor(np.asarray(donor)[ii])])
        hc = None if hold_c_all is None else hold_c_all[ii]
        res = run_planner2(st, kind, zn, g, P.make_gen(seed, j), hold_c=hc, lam=lam, seed_mode=seed_mode)
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


# ----------------------------------------------------------------------------------------------- selection, collapse
def choose_lambda(mae_by_lambda):
    """mae_by_lambda {lambda: mean TUNE MAE under G-TRUE}. Lowest MAE; an exact tie goes to the smaller lambda."""
    return min(mae_by_lambda, key=lambda l: (mae_by_lambda[l], l))


def hold_distance(plan_raw, hold_raw, std_raw):
    """Mean absolute distance of the plan from HOLD over windows and the 12 steps, per channel, in units of the prior std (the TRAIN std of the fitted level-1 prior, raw units).
    Returns (2,) [acc, yaw]."""
    return np.abs(np.asarray(plan_raw) - np.asarray(hold_raw)).mean(axis=(0, 1)) / np.asarray(std_raw)


def collapse_flags(dist):
    """HOLD-COLLAPSE per channel: the mean distance is below 10 % of the prior std."""
    return [bool(d < COLLAPSE_FRAC) for d in dist]


def std_raw_of(st, ch1):
    s = prior_std_orig(st).numpy().astype(np.float64)
    return np.array([s[0] * 2.0, s[1] * ch1])
