"""PREREG_V8F_PLANTEST library: the top-down planner (HJEPAStack.plan_topdown) on M2b's levels, goals, references, metrics, bootstrap, synthetic planted world.
Tier T0 (world-model diagnostic; the true-future goal is an oracle). Zero training. CPU. Code root: WMACT_CODE_ROOT (a read-only export of the tip 26fb0bc1)."""
import hashlib, math, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "wmact"))
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------------------------- fixed solver values (IMPL_CHOICES_PLANTEST.md section 1)
S_CAND = 16                       # H-JEPA Tab. 8 (DROID): S = (16, 16)
N_ITERS = 90                      # App. E: at most 90 iterations, early stopping when the best cost stalls
LR = {1: 0.1, 2: 0.3, 3: 0.3}      # level 1: 0.1 (Tab. 7 range, required by the planted control); upper 0.3 (Tab. 8)
LAM = 0.5                         # Tab. 8: subgoal weight beta of Eq. (6)
PATIENCE, REL_TOL = 10, 1e-4      # hjepa.plan_topdown defaults
SUBGOAL_K = {2: 3, 1: 6}          # level 3 plans 3 steps (2.4 s); level 2 matches the 3 level-3 states (H_2 = 3*2 = 6); level 1 matches the 6 level-2 states (H_1 = 6*2 = 12)
TOP_H, H1 = 3, 12
WIDEN, BOX_SIGMA = 1.5, 2.0       # DROID: initial Gaussians 1.5x wider, candidates clipped to prior mean +- 2 std (every level, level 1 included)
BATCH = 64
PLAN_SEED = 0
DT = 0.2
B_BOOT = 2000
GOALS = ("true", "rand", "stay")
PLANNERS = ("td3", "flat")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- the prior (App. E)
def apply_prior(st, level, actions):
    """fit_action_prior on ALL given training actions of the level (max_keep = everything: the stream order of the sample is by episode, so 'the last 4000' would be a few clips), then
    the DROID widening: std x WIDEN for the initial Gaussian, box = original mean +- BOX_SIGMA * original std."""
    a = actions.detach().reshape(-1, actions.shape[-1]).float()
    st.fit_action_prior(level, a, max_keep=10 ** 9)
    mean, std = a.mean(0), a.std(0).clamp_min(1e-6)
    st.set_action_prior(level, mean=mean, std=std * WIDEN, lo=mean - BOX_SIGMA * std, hi=mean + BOX_SIGMA * std)


# ----------------------------------------------------------------------------------------------- the planners (the ONLY place a plan is made)
def make_gen(seed, batch_idx):
    return torch.Generator().manual_seed(int(seed) * 100003 + int(batch_idx))


def run_planner(st, kind, z1_0, goal, gen, last_k=None):
    """z1_0 (B, D1); goal (B, D_goal): level 3 for 'td3', level 1 for 'flat'. Inputs: state + goal ONLY (no logged action reaches this function)."""
    if kind == "td3":
        return st.plan_topdown(z1_0, goal, top_horizon=TOP_H, goal_level=3, num_candidates=S_CAND, n_iters=N_ITERS, lr=LR, lam=LAM, subgoal_K=SUBGOAL_K,
                               patience=PATIENCE, rel_tol=REL_TOL, last_k=last_k, generator=gen)
    if kind == "flat":
        return st.plan_topdown(z1_0, goal, top_horizon=H1, goal_level=1, num_candidates=S_CAND, n_iters=N_ITERS, lr=LR, lam=0.0,
                               patience=PATIENCE, rel_tol=REL_TOL, last_k=last_k, generator=gen)
    raise ValueError(kind)


def goal_for(st, kind, gname, z1_future, z1_now, z1_future_donor):
    """The goal latent in the planner's goal level. true: encoded frame-12 latent; rand: the same quantity of a window of a DIFFERENT episode (donor); stay: the current latent."""
    lv = 3 if kind == "td3" else 1
    src = {"true": z1_future, "rand": z1_future_donor, "stay": z1_now}[gname]
    with torch.no_grad():
        return st.initial_latents(src, lv)[lv - 1]


def plan_all(st, kind, gname, z1_now, z1_future, donor, idx=None, seed=PLAN_SEED, batch=BATCH, collect=True, last_k=None):
    """Plan every window of idx (default all) in fixed consecutive batches; the generator of batch j is the same for every goal kind (paired initial candidates)."""
    n = z1_now.shape[0]
    idx = np.arange(n) if idx is None else np.asarray(idx)
    out = dict(actions=[], cost={}, init={}, iters={}, mono={}, lower=[])
    goal_lv = 3 if kind == "td3" else 1
    for j, s0 in enumerate(range(0, len(idx), batch)):
        ii = idx[s0:s0 + batch]
        zn = z1_now[ii]
        g = goal_for(st, kind, gname, z1_future[ii], zn, z1_future[torch.as_tensor(np.asarray(donor)[ii])])
        res = run_planner(st, kind, zn, g, make_gen(seed, j), last_k=last_k)
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
                    out["mono"].setdefault(lv, []).append(monotonicity(d))
    r = dict(actions=torch.cat(out["actions"]).numpy().astype(np.float64))
    for key in ("cost", "init", "iters", "mono"):
        for lv, v in out[key].items():
            if v:
                r[f"{key}{lv}"] = torch.cat(v).numpy().astype(np.float64)
    return r


def monotonicity(d):
    """H-JEPA Tab. 10 statistic (hjepa.monotonicity): the negative Spearman correlation between the goal cost and time along the chosen plan's trajectory."""
    from tanitad.models.hjepa import monotonicity as mono
    return mono(d)


# ----------------------------------------------------------------------------------------------- references and metrics
def hold_plan(A_raw):
    """HOLD: the current measured action (a0, yaw rate0 = v0 * kappa0) held for 12 steps (RAW units)."""
    return np.repeat(A_raw[:, :1], H1, axis=1)


def prior_plan(st, n, ch1_scale):
    """PRIOR: the fitted level-1 prior mean (scaled units) in RAW units, held for 12 steps."""
    m = st.prior_mean_1.detach().numpy().astype(np.float64)
    raw = np.array([m[0] * 2.0, m[1] * ch1_scale])
    return np.tile(raw, (n, H1, 1))


def unicycle(v0, plan):
    """Integrate (a, yaw rate) raw plan: v' = max(v + a dt, 0), psi' = psi + w dt, position by the speed at the step start. Returns v, psi at 2.4 s and xy path [n, 12, 2]."""
    n = plan.shape[0]
    v, psi, x, y = np.array(v0, np.float64), np.zeros(n), np.zeros(n), np.zeros(n)
    xy = np.zeros((n, H1, 2))
    for t in range(H1):
        x = x + v * np.cos(psi) * DT
        y = y + v * np.sin(psi) * DT
        v = np.maximum(v + plan[:, t, 0] * DT, 0.0)
        psi = psi + plan[:, t, 1] * DT
        xy[:, t] = np.stack([x, y], 1)
    return v, psi, xy


def window_metrics(plan, A_raw, v0):
    """Per-window rows (all vs the logged 12 actions): acc_mae [m/s^2], yaw_mae [rad/s] (deciding); speed_err, heading_err, ade of the unicycle integration of the plan vs of the logged actions (beside)."""
    L = A_raw[:, :H1].astype(np.float64)
    vp, pp, xyp = unicycle(v0, plan)
    vl, pl, xyl = unicycle(v0, L)
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    return dict(acc_mae=np.abs(plan[..., 0] - L[..., 0]).mean(1), yaw_mae=np.abs(plan[..., 1] - L[..., 1]).mean(1),
                speed_err=np.abs(vp - vl), heading_err=np.abs(wrap(pp - pl)), ade=np.linalg.norm(xyp - xyl, axis=-1).mean(1))


# ----------------------------------------------------------------------------------------------- bootstrap (episode-cluster, paired)
def boot_setup(eids, n_boot=B_BOOT, seed=0):
    import wmact_lib as W
    uniq, Wt = W.boot_weights(np.asarray(eids), n_boot, seed)
    pos = np.searchsorted(uniq, np.asarray([str(e) for e in eids]))
    return uniq, Wt, pos


def ep_sum(x, pos, n_ep):
    return np.bincount(pos, weights=np.asarray(x, np.float64), minlength=n_ep)


def boot_mean_diff(a, b, pos, Wt):
    """Paired episode-cluster bootstrap of mean(a) - mean(b) over windows (ratio of per-episode sums). Returns point, lo, hi (95 % percentile), and the bootstrap draws."""
    n_ep = Wt.shape[1]
    d = ep_sum(np.asarray(a) - np.asarray(b), pos, n_ep)
    c = ep_sum(np.ones(len(pos)), pos, n_ep)
    bd = (Wt @ d) / np.maximum(Wt @ c, 1e-12)
    pt = float(d.sum() / c.sum())
    return pt, float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))


def boot_mean(a, pos, Wt):
    n_ep = Wt.shape[1]
    s = ep_sum(a, pos, n_ep); c = ep_sum(np.ones(len(pos)), pos, n_ep)
    bd = (Wt @ s) / np.maximum(Wt @ c, 1e-12)
    return float(s.sum() / c.sum()), float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))


def sep_label(pt, lo, hi):
    """lower metric = better (MAE): 'better' = interval wholly below 0; 'worse' = wholly above 0."""
    return "better" if hi < 0 else ("worse" if lo > 0 else "none")


def family_label(sep_hold, sep_rand):
    """The card's committed outcomes for one (planner, family, seed). sep_* is the label of (G-TRUE minus reference)."""
    if sep_rand == "better" and sep_hold == "better":
        return "PLANNABLE"
    if sep_rand == "none":
        return "GOAL-BLIND"
    if sep_rand == "better":
        return "NO-GAIN"
    return "WORSE-THAN-RAND"          # G-TRUE separated worse than G-RAND: outside the card's list, satisfies none of its outcomes


# ----------------------------------------------------------------------------------------------- data
def load_latents(path, tag):
    L = torch.load(path, weights_only=False)
    ok = L["ok"].numpy().all(1)
    sel = np.nonzero(ok)[0]
    return dict(Z1=L["Z1"][tag][sel].float(), A=L["A_raw"][sel].float(), eid=np.array(L["eid"])[sel].astype("U36"), t=np.asarray(L["t"])[sel], sel=sel,
                n_total=len(ok), ok12=L["ok"].numpy()[:, [0, 12]].all(1).sum())


# ----------------------------------------------------------------------------------------------- the synthetic planted world (control 1)
class LinPred(nn.Module):
    """z' = A z + B a, applied at every position of the context (rollout takes [:, -1]). disconnect=True ignores the action (the deliberate-regression arm)."""
    context = 1

    def __init__(self, A, B, disconnect=False):
        super().__init__()
        self.register_buffer("A", A); self.register_buffer("B", B); self.disconnect = disconnect

    def forward(self, z, a):
        out = z @ self.A.T
        eff = a @ self.B.T
        return out + (0.0 * eff if self.disconnect else eff)       # x 0.0 keeps the graph (the planner needs a grad_fn) and removes every effect of the action


def planted_stack(D=32, seed=0, disconnect=False, theta=3.0, decay=0.95):
    """A 3-level stack with identity encoders and a known linear system: z1' = A z1 + B a (a: 2-d); level 2 acts on the concatenation of two level-1 actions (4-d), level 3 on two of those (8-d), so
    all three levels describe the SAME physics and 12 level-1 actions (24 numbers) are identifiable from the 2.4 s state (D = 32 >= 24)."""
    from tanitad.models.hjepa import HJEPAStack, LevelConfig
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(D, D, generator=g); S = S - S.T
    S = S / torch.linalg.matrix_norm(S, 2)                         # a random skew matrix of spectral norm 1: A1 rotates each latent by at most theta rad per step (slow, like consecutive real latents)
    A1 = decay * torch.matrix_exp(theta * S)
    B1 = torch.randn(D, 2, generator=g) / math.sqrt(D) * 2.0
    A2 = A1 @ A1
    B2 = torch.cat([A1 @ B1, B1], 1)
    A3 = A2 @ A2
    B3 = torch.cat([A2 @ B2, B2], 1)
    lv = [LevelConfig(latent_dim=D, action_dim=2), LevelConfig(latent_dim=D, action_dim=4, stride=2), LevelConfig(latent_dim=D, action_dim=8, stride=2)]
    st = HJEPAStack(lv, sigreg_projections=8, predictor_overrides={1: LinPred(A1, B1, disconnect), 2: LinPred(A2, B2, disconnect), 3: LinPred(A3, B3, disconnect)})
    st.encoders[0] = nn.Identity(); st.encoders[1] = nn.Identity()
    return st.eval(), (A1, B1)


def planted_actions(n, seed, std=0.5):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, H1, 2, generator=g) * std).clamp(-1.5 * std, 1.5 * std)


def planted_fit_priors(st, std=0.5, n=600, seed=12345):
    a = planted_actions(n, seed, std)
    apply_prior(st, 1, a)
    apply_prior(st, 2, a.reshape(n, 6, 4))
    apply_prior(st, 3, a.reshape(n, 3, 8))


@torch.no_grad()
def planted_roll(st, z0, a):
    return st.rollout(1, z0, a)[:, -1]


def planted_recovery(st, kind, n=64, seed=777, std=0.5, plan_fn=None, last_k=None):
    """Plan toward the state reached by planted actions; return per-window MAE of the recovered level-1 actions and the MAE of the prior mean."""
    plan_fn = plan_fn or run_planner
    g = torch.Generator().manual_seed(seed)
    D = st.cfg[0].latent_dim
    z0 = torch.randn(n, D, generator=g)
    a = planted_actions(n, seed + 1, std)
    zT = planted_roll(st, z0, a)
    if kind == "td3":
        goal = st.initial_latents(zT, 3)[2]
    else:
        goal = zT
    res = plan_fn(st, kind, z0, goal, make_gen(0, 0), last_k=last_k)
    rec = res.actions.detach()
    mae = (rec - a).abs().mean((1, 2))
    base = (st.prior_mean_1.expand(n, H1, 2) - a).abs().mean((1, 2))
    return mae.numpy(), base.numpy(), rec, a
