"""PREREG_V8F_PLANTEST3 library: PLANTEST2's planner (plantest2/plantest2_lib.py, imported unchanged) plus a decoded cost, a SHRINK(c) rollout and the KIN-CEILING.
Tier T0 (oracle-goal diagnostic). Zero training (closed-form ridge decoders). CPU. plan_topdown3 is a copy of plan_topdown2 with one new argument, `cost`; with cost=None it executes the
same operations in the same order and returns bit-identical plans (control 1)."""
import hashlib, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plantest2"))
sys.path.insert(0, os.path.join(HERE, "..", "plantest"))
import numpy as np
import torch
import plantest_lib as P
import plantest2_lib as Q
from tanitad.models.hjepa import PlanResult

SALT_DEC = "plantest3-dec-v1"
RIDGE = 1.0e-3
LAMBDAS3 = (0.0, 0.1, 1.0)
HOLD_FRAC_INNER = 4                 # the first floor(n/4) TUNE episodes (by hash rank) are held out for the gate
R2_SPEED_MIN, R2_YAW_MIN = 0.70, 0.50
FAMS = Q.FAMS
ARMS = ("end", "path")
FORMS = ("plain", "shrink")
DT = P.DT


# ----------------------------------------------------------------------------------------------- ridge decoders
def r2_cols(y, p):
    """held-out R2 per column: 1 - SSE / SST about the mean of y. y, p (n, m)."""
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    sst = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ((y - p) ** 2).sum(0) / np.maximum(sst, 1e-12)


def ridge_fit(X, y, ridge=RIDGE):
    """Closed form, float64, intercept unpenalised: W = (Xc'Xc + ridge n I)^-1 Xc'yc, b = ym - mu W. X (n, D), y (n,). Returns W (D,), b."""
    X = torch.as_tensor(np.asarray(X), dtype=torch.double) if not torch.is_tensor(X) else X.double()
    y = torch.as_tensor(np.asarray(y), dtype=torch.double) if not torch.is_tensor(y) else y.double()
    n = X.shape[0]
    mu, ym = X.mean(0), y.mean()
    Xc, yc = X - mu, y - ym
    G = Xc.T @ Xc + ridge * n * torch.eye(X.shape[1], dtype=torch.double)
    W = torch.linalg.solve(G, Xc.T @ yc)
    return W, ym - mu @ W


class Decoder:
    """D_1 (level-1 latent -> 2) and D_2 (level-2 latent -> 2), outputs in TUNE-standard units: (speed, yaw rate) / ystd. W (D, 2) float32, b (2,), ystd (2,) numpy."""

    def __init__(self, W1, b1, W2, b2, ystd, n_samples=None):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
        self.ystd = np.asarray(ystd, np.float64)
        self.n_samples = n_samples

    def W(self, level):
        return self.W1 if level == 1 else self.W2

    def decode(self, level, X):
        """standardised outputs (..., 2)."""
        W = self.W1 if level == 1 else self.W2
        b = self.b1 if level == 1 else self.b2
        return X @ W + b

    def decode_raw(self, level, X):
        return self.decode(level, X).detach().numpy().astype(np.float64) * self.ystd

    def digest(self):
        h = hashlib.sha256()
        for t in (self.W1, self.b1, self.W2, self.b2):
            h.update(t.detach().numpy().tobytes())
        h.update(self.ystd.tobytes())
        return h.hexdigest()


def enc2(st, Z):
    """the pointwise level-2 encoding (w = 1) of level-1 latents Z (..., D1)."""
    with torch.no_grad():
        return st.encoders[0](Z)


def build_samples(Z, speed, yawr):
    """Z (m, T, D) latents, speed (m, T) and yawr (m, T) targets (NaN = missing). Returns X (m*T, D), Y (m*T, 2), valid (m*T, 2)."""
    m, T, D = Z.shape
    X = Z.reshape(m * T, D)
    Y = np.stack([np.asarray(speed, np.float64).reshape(-1), np.asarray(yawr, np.float64).reshape(-1)], 1)
    ok_z = torch.isfinite(X).all(1).numpy()
    valid = np.isfinite(Y) & ok_z[:, None]
    return X, Y, valid


def fit_decoder(st, Z, speed, yawr, ridge=RIDGE, with_l2=True):
    """Fit D_1 (and D_2) on the given rows ONLY. Z (m, T, D1) float tensor, speed / yawr (m, T). ystd = std over the valid samples of each target. Each target is fitted on its own valid samples."""
    X, Y, valid = build_samples(Z, speed, yawr)
    ystd = np.array([Y[valid[:, j], j].std() for j in range(2)])
    out = {}
    levels = (1, 2) if with_l2 else (1,)
    for lv in levels:
        Xl = X if lv == 1 else enc2(st, X)
        Ws, bs = [], []
        for j in range(2):
            W, b = ridge_fit(Xl[torch.as_tensor(valid[:, j])], Y[valid[:, j], j] / ystd[j], ridge)
            Ws.append(W.float()); bs.append(b.float())
        out[lv] = (torch.stack(Ws, 1), torch.stack(bs))
    W2, b2 = out.get(2, (torch.zeros(1, 2), torch.zeros(2)))
    return Decoder(out[1][0], out[1][1], W2, b2, ystd, n_samples=int(valid.sum(0).min()))


def heldout_r2(st, dec, level, Z, speed, yawr):
    """R2 of speed and yaw rate on the given rows' encoded latents, pooled over valid (window, knot) samples. Returns (r2 speed, r2 yaw, n speed, n yaw)."""
    X, Y, valid = build_samples(Z, speed, yawr)
    Xl = X if level == 1 else enc2(st, X)
    with torch.no_grad():
        pr = dec.decode(level, Xl).numpy().astype(np.float64) * dec.ystd
    out, ns = [], []
    for j in range(2):
        v = valid[:, j]
        out.append(float(r2_cols(Y[v, j:j + 1], pr[v, j:j + 1])[0])); ns.append(int(v.sum()))
    return out[0], out[1], ns[0], ns[1]


def inner_split(eids, salt=SALT_DEC, frac=HOLD_FRAC_INNER):
    """episode-disjoint split of the given (TUNE) clip ids: ranked by sha256(salt|id); the first floor(n/frac) are held out. Returns (fit ids, held ids)."""
    uniq = sorted(set(str(e) for e in eids))
    key = sorted(uniq, key=lambda e: (hashlib.sha256(f"{salt}|{e}".encode()).hexdigest(), e))
    nh = len(key) // frac
    held, fit = set(key[:nh]), set(key[nh:])
    assert not (held & fit) and len(held) + len(fit) == len(uniq)
    return fit, held


# ----------------------------------------------------------------------------------------------- the cost spec and the SHRINK rollout
def shrunk_rollout(st, level, z0, actions, c):
    """HJEPAStack.rollout with context 1 and the step z' = z + c (pred(z, a) - z). For c = 0 the result is exactly z0 at every step and the gradient with respect to the actions is exactly 0.0."""
    pred = st.predictors[level - 1]
    lead, (h_, a_dim) = actions.shape[:-2], actions.shape[-2:]
    a = actions.reshape(-1, h_, a_dim)
    n = a.shape[0]
    z = z0.expand(n, -1) if z0.dim() == 1 else z0.reshape(n, -1)
    e = st.act_embed(a) if level == 1 else a
    zs = [z]
    for h in range(h_):
        zc = torch.stack(zs[h:h + 1], dim=1)
        nxt = pred(zc, e[:, h:h + 1])[:, -1]
        zs.append(zs[h] + c * (nxt - zs[h]))
    out = torch.stack(zs[1:], dim=1)
    return out.reshape(*lead, h_, out.shape[-1])


class CostSpec:
    """mode: latent | end | path. c: None or 1.0 = the stack's own rollout (unchanged call), else the SHRINK factor. dec: Decoder (end / path)."""

    def __init__(self, mode="latent", c=None, dec=None):
        assert mode in ("latent", "end", "path")
        assert mode == "latent" or dec is not None
        self.mode, self.c, self.dec = mode, (None if c is None else float(c)), dec

    def roll(self, st, level, z0e, a):
        if level != 1 or self.c is None or self.c == 1.0:
            return st.rollout(level, z0e, a)
        return shrunk_rollout(st, level, z0e, a, self.c)

    def top_cost(self, st, roll, goal, level, last_k):
        """FLAT1 (level 1 is the top). roll (B,S,12,D); goal (B,D) for latent / end, (B,12,D) for path."""
        if self.mode == "latent":
            return st.goal_cost(roll, goal.unsqueeze(1), level, last_k)
        W = self.dec.W1
        if self.mode == "end":
            return ((roll[..., -1, :] - goal.unsqueeze(1)) @ W).pow(2).sum(-1)
        return ((roll - goal.unsqueeze(1)) @ W).pow(2).sum(dim=(-1, -2))

    def sub_cost(self, st, lifted, sub, lam):
        """TD3 level 1: lifted (B,S,K,D2), sub (B,K,D2) -> (B,S)."""
        if self.mode == "latent":
            dd = (lifted - sub.unsqueeze(1)).pow(2).sum(-1)
        else:
            assert self.mode == "end", "TD3 is DEC-END only"
            dd = ((lifted - sub.unsqueeze(1)) @ self.dec.W2).pow(2).sum(-1)
        return dd[..., -1] + lam * dd[..., :-1].sum(-1)


# ----------------------------------------------------------------------------------------------- the solver (copy of plan_topdown2 + `cost`)
def plan_topdown3(st, z1_0, goal, *, top_horizon, goal_level, num_candidates, n_iters, lr, lam, subgoal_K, patience, rel_tol, last_k, generator, seeds=None, reg=None, cost=None):
    """seeds, reg: as plan_topdown2. cost: None (the unchanged cost) or a CostSpec (level-1 rollout and level-1 cost; levels 2 and 3 are never touched)."""
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

            def _roll(level, z0e, a):
                if cost is not None and level == 1:
                    return cost.roll(st, level, z0e, a)
                return st.rollout(level, z0e, a)

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

                    if cost is None:
                        def cost_fn(a, _lv=level, _d=d_lv):
                            z0e = _d.unsqueeze(1).expand(b, a.shape[1], -1)
                            roll = st.rollout(_lv, z0e, a)
                            return st.goal_cost(roll, goal.unsqueeze(1), _lv, last_k)
                    else:
                        def cost_fn(a, _lv=level, _d=d_lv):
                            z0e = _d.unsqueeze(1).expand(b, a.shape[1], -1)
                            roll = _roll(_lv, z0e, a)
                            if _lv == 1:
                                return cost.top_cost(st, roll, goal, _lv, last_k)
                            return st.goal_cost(roll, goal.unsqueeze(1), _lv, last_k)
                else:
                    up = st.cfg[level]
                    K = int(sub_k.get(level, 1))
                    if K < 1 or K > res.level_rollouts[level + 1].shape[1]:
                        raise ValueError(f"subgoal_K[{level}]={K} outside 1..H_{level + 1}")
                    horizon = K * up.stride + up.window - 1
                    sub = res.level_rollouts[level + 1][:, :K].detach()

                    if cost is None or level != 1:
                        def cost_fn(a, _lv=level, _d=d_lv, _sub=sub, _K=K):
                            z0e = _d.unsqueeze(1).expand(b, a.shape[1], -1)
                            roll = st.rollout(_lv, z0e, a)
                            full = torch.cat([z0e.unsqueeze(2), roll], dim=2)
                            lifted = st._apply_encoder(_lv + 1, full, strict=True)[:, :, 1:]
                            dd = (lifted - _sub.unsqueeze(1)).pow(2).sum(-1)
                            return dd[..., -1] + lam * dd[..., :-1].sum(-1)
                    else:
                        def cost_fn(a, _lv=level, _d=d_lv, _sub=sub, _K=K):
                            z0e = _d.unsqueeze(1).expand(b, a.shape[1], -1)
                            roll = _roll(_lv, z0e, a)
                            full = torch.cat([z0e.unsqueeze(2), roll], dim=2)
                            lifted = st._apply_encoder(_lv + 1, full, strict=True)[:, :, 1:]
                            return cost.sub_cost(st, lifted, _sub, lam)

                if level == 1 and reg is not None and reg["lam"] != 0.0:
                    if horizon != reg["ref"].shape[1]:
                        raise ValueError("REG reference horizon != level-1 horizon")
                    cost_base = cost_fn

                    def cost_fn(a, _base=cost_base):
                        return _base(a) + Q.reg_term(a, reg["ref"], reg["std"], reg["lam"])

                best_a, best_cost, init_cost, hist, nit = _opt(level, horizon, cost_fn, n_cand)
                pick = best_cost.argmin(dim=1)
                ar = torch.arange(b, device=pick.device)
                chosen = best_a[ar, pick]
                with torch.no_grad():
                    roll = _roll(level, d_lv, chosen)
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


def run_planner3(st, kind, z1_0, goal, gen, hold_c=None, lam=0.0, seed_mode="hold", cost=None):
    """kind td3 | flat1. goal: level 3 latent for td3; level-1 latent (end / latent) or (B,12,D) path for flat1."""
    last_k = None
    if kind == "flat1":
        kind, last_k = "flat", 1
    if seed_mode == "prior":
        assert lam == 0.0, "REG needs the HOLD seed"
        seeds, reg = None, None
    else:
        seeds = Q.hold_seeds(st, hold_c, z1_0, kind)
        reg = dict(lam=float(lam), ref=hold_c, std=Q.prior_std_orig(st)) if lam != 0.0 else None
    if kind == "td3":
        return plan_topdown3(st, z1_0, goal, top_horizon=P.TOP_H, goal_level=3, num_candidates=P.S_CAND, n_iters=P.N_ITERS, lr=P.LR, lam=P.LAM, subgoal_K=P.SUBGOAL_K,
                             patience=P.PATIENCE, rel_tol=P.REL_TOL, last_k=last_k, generator=gen, seeds=seeds, reg=reg, cost=cost)
    if kind == "flat":
        return plan_topdown3(st, z1_0, goal, top_horizon=P.H1, goal_level=1, num_candidates=P.S_CAND, n_iters=P.N_ITERS, lr=P.LR, lam=0.0, subgoal_K=None,
                             patience=P.PATIENCE, rel_tol=P.REL_TOL, last_k=last_k, generator=gen, seeds=seeds, reg=reg, cost=cost)
    raise ValueError(kind)


def goal3(st, kind, gname, cost, z1_future, z1_now, z1_donor_future, z_path, z_path_donor):
    """The planner's goal. td3 / latent / end: PLANTEST's goal (encoded frame-12 latent, level 3 for td3). path: the 12-knot encoded true-future path (G-TRUE) or the donor's (G-RAND)."""
    k = "flat" if kind == "flat1" else kind
    if k == "flat" and cost is not None and cost.mode == "path":
        assert gname in ("true", "rand"), gname
        return (z_path if gname == "true" else z_path_donor).detach()
    return P.goal_for(st, k, gname, z1_future, z1_now, z1_donor_future)


def plan_all3(st, kind, gname, z1_now, z1_future, z_path, donor, hold_c_all, idx=None, seed=P.PLAN_SEED, batch=P.BATCH, collect=True, lam=0.0, seed_mode="hold", cost=None):
    """As plan_all2 (fixed consecutive batches of idx, generator of batch j = make_gen(seed, j)), with the cost spec. z_path (N, 12, D): encoded true-future latents at knots 1..12 (needed for path)."""
    k = "flat" if kind == "flat1" else kind
    n = z1_now.shape[0]
    idx = np.arange(n) if idx is None else np.asarray(idx)
    out = dict(actions=[], cost={}, init={}, iters={}, mono={}, dmono=[])
    goal_lv = 3 if k == "td3" else 1
    dn = np.asarray(donor)
    for j, s0 in enumerate(range(0, len(idx), batch)):
        ii = idx[s0:s0 + batch]
        zn = z1_now[ii]
        dd = torch.as_tensor(dn[ii])
        zp = None if z_path is None else z_path[ii]
        zpd = None if z_path is None else z_path[dd]
        g = goal3(st, kind, gname, cost, z1_future[ii], zn, z1_future[dd], zp, zpd)
        gl = g if (cost is None or cost.mode != "path" or k != "flat") else P.goal_for(st, "flat", gname, z1_future[ii], zn, z1_future[dd])
        hc = None if hold_c_all is None else hold_c_all[ii]
        res = run_planner3(st, kind, zn, g, P.make_gen(seed, j), hold_c=hc, lam=lam, seed_mode=seed_mode, cost=cost)
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
                    d = (lifted - gl.unsqueeze(1)).pow(2).sum(-1)
                    out["mono"].setdefault(lv, []).append(P.monotonicity(d))
                if cost is not None and cost.dec is not None:
                    gend = P.goal_for(st, "flat", gname, z1_future[ii], zn, z1_future[dd])
                    full1 = torch.cat([zn.unsqueeze(1), res.level_rollouts[1]], 1)
                    dm = ((full1 - gend.unsqueeze(1)) @ cost.dec.W1).pow(2).sum(-1)
                    out["dmono"].append(P.monotonicity(dm))
    r = dict(actions=torch.cat(out["actions"]).numpy().astype(np.float64))
    for key in ("cost", "init", "iters", "mono"):
        for lv, v in out[key].items():
            if v:
                r[f"{key}{lv}"] = torch.cat(v).numpy().astype(np.float64)
    if out["dmono"]:
        r["dmono1"] = torch.cat(out["dmono"]).numpy().astype(np.float64)
    return r


# ----------------------------------------------------------------------------------------------- KIN-CEILING
def kin_plan(dec, Zk):
    """Zk (n, 13, D1) encoded true-future latents at knots 0..12 -> raw plan (n, 12, 2): acc_k = (s_{k+1} - s_k) / dt, yaw_k = w_k (decoded, de-standardised)."""
    Y = dec.decode_raw(1, Zk)
    acc = (Y[:, 1:13, 0] - Y[:, 0:12, 0]) / DT
    return np.stack([acc, Y[:, 0:12, 1]], -1)


# ----------------------------------------------------------------------------------------------- planted decoder (control 2)
def planted_decoder(B1):
    """control 2: a planted LINEAR decoder aligned with the action input matrix of the planted world, W = B1 (B1'B1)^-1 (so W'B1 = I). A random 32 -> 2 decoder reached DEC-PATH action MAE 0.083 in the same
    solver budget (scale-invariant: 0.0831 at scales 1 to 30), above the 0.05 bar, because the 24 actions are identified only through W'A^m B1 chains; this one reads the action-driven coordinates."""
    W = B1 @ torch.linalg.inv(B1.T @ B1)
    return Decoder(W, torch.zeros(2), W.clone(), torch.zeros(2), np.ones(2))


def shuffled_r2(st, Z, speed, yawr, seed=0):
    """control 3 helper: fit on half the rows with targets permuted, return held-out R2 on the other half."""
    rng = np.random.default_rng(seed)
    m = Z.shape[0]
    p = rng.permutation(m); a, b = p[: m // 2], p[m // 2:]
    sp = np.asarray(speed)[a].copy(); yw = np.asarray(yawr)[a].copy()
    perm = rng.permutation(sp.size)
    sp = sp.reshape(-1)[perm].reshape(sp.shape); yw = yw.reshape(-1)[perm].reshape(yw.shape)
    dec = fit_decoder(st, Z[a], sp, yw, with_l2=False)
    return dec, heldout_r2(st, dec, 1, Z[b], np.asarray(speed)[b], np.asarray(yawr)[b])
