#!/usr/bin/env python3
"""HiCAP stream R-C toy: cheap latent imagination inside a contrastive selector.  numpy only.

Run:  python3 rc_latent_imagination_toy.py            (writes rc_latent_imagination_toy_result.json next to it)
      python3 rc_latent_imagination_toy.py --calibrate  (world diagnostics only, no learner)
Wall time MEASURED 86-103 s on a shared 4-core box (3 seeds x 6 configs = 18 jobs in a Pool(4), 1 BLAS thread each).

QUESTION.  A CLM-style selector scores  tau*cos(u(s), v(a))  over a candidate vocabulary.  The best action depends
on a HIDDEN FUTURE CONSEQUENCE (does the lead vehicle brake, how hard, when) that is only weakly and non-linearly
present in the current state embedding s.  Does adding CHEAP imagination on cached embeddings help, at what
multiply-adds, and can a gate spend it on few ticks?

WORLD (exact simulator, dt = 0.5 s, T = 6 steps = 3 s).  Ego speed v0, lead gap d0, lead speed vl0; hidden mode
z in {steady, brake 3.0 m/s^2, brake 6.5 m/s^2} with a hidden onset time.  Observation at every step
o_t = [v/20, gap/50, (vl-v)/5, cue(6)]; the cue is a noisy 'brake-light' channel (half strength before the onset, full
after; amplitude x SNR).  A FIXED random tanh-MLP 'frozen encoder' maps o_t to a unit 32-d embedding s_t, so every
learner sees the same s.  Candidate = (a1, a2) piecewise-constant accel (1.5 s + 1.5 s); vocabulary K = 32.
Cost J = sum_t [soft/hard gap penalty - progress] + comfort + envelope penalty.  The expert takes
argmin_k (J_k + N(0, 0.6^2)) -- it USES z (labels may use privileged signals); no learner ever sees z as an input.
Metric = REGRET = J(selected) - J(best) on the true simulator (heavy-tailed: a collision costs 40 per step);
also top-1, within-0.5, and 'unsafe' (selected collides while a collision-free best exists).

ARMS
  A0   myopic contrastive selector, hard one-hot expert label (CLM-style; the registry's worst target shape)
  A0s  same tower, SOFT targets from the TRUE per-candidate cost (privileged Hydra-style teacher; zero inference cost)
  QM   model-free Q head q(s,a) ~ sum of state costs, same rows as the imagination heads (confound control)
  A1   A0 + direct multi-horizon outcome head g(s0,a)->s_hat_{2,4,6} on the top-Ki (=4) only; cost head c(s)
  A1s  the same re-rank on top of the stronger proposer A0s
  A2   iterated one-step latent rollouts for ALL K;   A2_pure / *_c = variants (see below)
  A4   A0 retrained on soft targets from the LEARNED imagination (distillation, zero inference cost)
  A5   FACTORISED imagination: learn only the exogenous hidden consequence from LOGS ALONE (readout of visible state +
       lead-mode posterior), evaluate every candidate ANALYTICALLY; A5m also marginalises the onset time;
       A5cal = temperature-scaled mode posterior; DIAG_A5m_true_visible_state = same with the TRUE visible state
  REF  z-blind risk-sensitive lookup on the TRUE visible state (no cue, no learning of z): the floor that separates
       'risk-sensitive expected-cost selection' from 'access to the hidden consequence'
  Two heads (bootstrap + init) per predictor; ensemble mean = imagined value, |A-B| = disagreement.  Counterfactual
  rows ('aug') = 3 extra random candidates per tick whose exact futures are embedded (assumes a simulator/renderer).

PRE-REGISTERED READINGS (written BEFORE the first sweep; verdicts are in RC_imagination.md section 5, not here)
  H1 (sample efficiency)  A1 and A2 beat A0 at small n and the gap shrinks with n.  If it does NOT shrink, the
     advantage is representational, not data-efficiency.
  H2 (negative control)   at cue SNR 0 no arm beats A0 beyond noise: imagination cannot conjure information.
  H3 (support)            outcome heads trained on expert-only actions hallucinate benign futures for out-of-envelope
     candidates; counterfactual augmentation + an analytic feasibility mask fix it.
  H4 (gate)               imagining on <= 30 % of ticks (need x trust gate) keeps >= 80 % of A1's gain.  A gate that
     imagines where heads DISAGREE MOST must be worse.
  H5 (calibration)        head disagreement tracks the true imagination error for the DIRECT head; for iterated
     rollouts it decouples at long horizon ('attractor').

DISCLOSED FORKING PATHS (added AFTER the first sweep, because the first sweep showed A0 was worse than a trivial
z-blind lookup and that A2's win was confounded with risk-sensitivity): A0s, QM, A1s, REF, A1c/A2c (cost head refit
on imagined latents), A5 family, the SNR 3 config, the learned/oracle gates.  The world was re-calibrated twice
BEFORE any arm ran (first version: z did not change the best action) using only two criteria (z-blind regret
materially > 0; oracle regret 0).  Nothing else was tuned toward an arm.

Evidence class of every number printed here: MEASURED (toy, this script).  Not a driving result.
Interval estimator (named, per the programme rule): paired bootstrap over pooled test ticks (3 seeds x 900 ticks);
ticks are i.i.d. draws by construction, there are no episode clusters in the toy.  Seeds: 3 (sd across seeds shown).
"""
import json
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")     # one BLAS thread per process; seeds run in parallel processes
import numpy as np

DT, T, C = 0.5, 6, 6
DEC = np.array([0.0, 3.0, 6.5])
STRENGTH = np.array([0.0, 0.6, 1.0])
PZ = np.array([0.55, 0.25, 0.20])
A1S = np.array([-4.0, -2.0, 0.0, 1.5])
A2S = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0])
GRID = np.array([(a, b) for a in A1S for b in A2S])           # (32, 2)
WILD = np.array([(4.0, 4.0), (6.0, 6.0), (-7.0, -7.0), (-9.0, -9.0)])
K = len(GRID)
CUE_SIGMA = 0.6
EXPERT_SIGMA = 0.6
EMB = 32
KI = 4                                                        # top-Ki re-ranked by imagination


# ----------------------------------------------------------------------------- world
def sample_world(rng, n):
    z = rng.choice(3, size=n, p=PZ)
    return dict(v0=rng.uniform(8, 20, n), d0=rng.uniform(8, 30, n), vl0=None, z=z,
                tb=rng.uniform(0.0, 1.0, n), noise=rng.normal(0, 1, (n, T + 1, C)))


def finish_world(W, rng):
    W["vl0"] = W["v0"] + rng.uniform(-3, 1, len(W["v0"]))
    return W


def simulate(W, acc):
    """W: dict of (n,) arrays; acc: (m,2).  Returns ego speed v, gap g (n,m,T+1) and lead speed vl (n,T+1)."""
    n, m = len(W["v0"]), len(acc)
    tt = np.arange(T + 1) * DT
    dec = DEC[W["z"]][:, None]
    vl = np.maximum(0.0, W["vl0"][:, None] - dec * np.maximum(0.0, tt[None, :] - W["tb"][:, None]))
    v = np.empty((n, m, T + 1)); g = np.empty((n, m, T + 1))
    v[..., 0] = W["v0"][:, None]; g[..., 0] = W["d0"][:, None]
    for t in range(T):
        a = acc[:, 0] if t < 3 else acc[:, 1]
        vn = np.maximum(0.0, v[..., t] + a[None, :] * DT)
        g[..., t + 1] = g[..., t] + (0.5 * (vl[:, t] + vl[:, t + 1])[:, None] - 0.5 * (v[..., t] + vn)) * DT
        v[..., t + 1] = vn
    return v, np.maximum(g, -2.0), vl


def state_cost(v, g):
    """per-step state cost c_t, t = 1..T  (n,m,T): safety - progress."""
    gg, vv = g[..., 1:], v[..., 1:]
    return 30.0 * np.maximum(0.0, 12.0 - gg) / 12.0 + 40.0 * (gg <= 0.5) - 0.06 * vv * DT


def act_cost(acc):
    """analytic terms: comfort + envelope penalty (the programme's kinematic barriers, reused as-is)."""
    comfort = 0.03 * (3 * acc[:, 0] ** 2 + 3 * acc[:, 1] ** 2)
    env = 60.0 * ((acc.max(1) > 2.6) | (acc.min(1) < -5.1))
    return comfort, env


def total_cost(v, g, acc, with_env=True):
    comfort, env = act_cost(acc)
    J = state_cost(v, g).sum(-1) + comfort[None, :]
    return J + env[None, :] if with_env else J


def cue_level(W):
    tt = np.arange(T + 1) * DT
    early = tt[None, :] < W["tb"][:, None]
    return STRENGTH[W["z"]][:, None] * np.where(early, 0.5, 1.0)          # (n, T+1)


class Encoder:
    """fixed random tanh-MLP = the 'frozen foundation encoder'."""
    def __init__(self, seed=1234):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 1.2 / 3.0, (9, 48)); self.b1 = r.normal(0, 0.3, 48)
        self.W2 = r.normal(0, 1.2 / np.sqrt(48), (48, EMB)); self.b2 = r.normal(0, 0.3, EMB)

    def __call__(self, o):
        h = np.tanh(o @ self.W1 + self.b1)
        e = np.tanh(h @ self.W2 + self.b2)
        return e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-9)


def observe(enc, W, v, g, vl, snr, cue_override=None):
    """embeddings s_t for every (tick, candidate, step): (n, m, T+1, EMB)."""
    n, m, _ = v.shape
    cue = (snr * cue_level(W)[:, :, None] * np.ones(C)[None, None, :] + CUE_SIGMA * W["noise"]) \
        if cue_override is None else cue_override
    o = np.empty((n, m, T + 1, 9))
    o[..., 0] = v / 20.0
    o[..., 1] = np.clip(g, -2, 60) / 50.0
    o[..., 2] = (vl[:, None, :] - v) / 5.0
    o[..., 3:] = cue[:, None, :, :]
    return enc(o)


def afeat(acc):
    a = np.asarray(acc, float) / 5.0
    return np.stack([a[:, 0], a[:, 1], a[:, 0] * a[:, 1], np.ones(len(a))], 1)


# ----------------------------------------------------------------------------- tiny MLP + Adam
class MLP:
    def __init__(self, sizes, rng, lr=3e-3):
        self.W = [rng.normal(0, 1 / np.sqrt(a), (a, b)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t, self.lr = 0, lr
        self.macs = sum(a * b for a, b in zip(sizes[:-1], sizes[1:]))

    def fwd(self, x):
        x = x.astype(np.float32, copy=False)
        self.h = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.tanh(x)
            self.h.append(x)
        return x

    def bwd(self, g):
        g = g.astype(np.float32, copy=False)
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:
                g = g * (1 - self.h[i + 1] ** 2)
            gW.insert(0, self.h[i].T @ g); gb.insert(0, g.sum(0))
            g = g @ self.W[i].T
        self.g = gW + gb
        return g

    def step(self, lr_scale=1.0):
        self.t += 1
        for p, gr, m, v in zip(self.W + self.b, self.g, self.m, self.v):
            m *= 0.9; m += 0.1 * gr
            v *= 0.999; v += 0.001 * gr * gr
            p -= self.lr * lr_scale * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)

    def predict(self, x, chunk=60000):
        out = [self.fwd(x[i:i + chunk]) for i in range(0, len(x), chunk)]
        return np.concatenate(out, 0)


def fit_regress(net, X, Y, rng, steps=900, bs=256, boot=False):
    N = len(X)
    pool = rng.integers(0, N, N) if boot else np.arange(N)
    for s in range(steps):
        idx = pool[rng.integers(0, len(pool), bs)]
        pred = net.fwd(X[idx])
        net.bwd(2.0 * (pred - Y[idx]) / bs)
        net.step(1.0 - 0.8 * s / steps)
    return net


def fit_classifier(net, X, y, rng, steps=800, bs=256):
    N = len(X); oh = np.eye(3)[y]
    for st in range(steps):
        idx = rng.integers(0, N, bs)
        lg = net.fwd(X[idx]).astype(np.float64)
        P = np.exp(lg - lg.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        net.bwd((P - oh[idx]) / bs)
        net.step(1.0 - 0.8 * st / steps)
    return net


TAU = 10.0


def fit_selector(S0, target, rng, steps=1500, bs=256, hid=48, dim=16):
    """dual encoder: cosine(u(s), v(a)) * TAU, listwise softmax over the K candidates, soft or one-hot target."""
    u = MLP([EMB, hid, dim], rng); v = MLP([4, 32, dim], rng)
    af = afeat(GRID)
    N = len(S0)
    for s in range(steps):
        idx = rng.integers(0, N, bs)
        U = u.fwd(S0[idx]); V = v.fwd(af)
        Un = U / np.linalg.norm(U, axis=1, keepdims=True); Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
        L = TAU * Un @ Vn.T
        P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        dL = (P - target[idx]) / bs
        dUn = TAU * dL @ Vn; dVn = TAU * dL.T @ Un
        dU = (dUn - Un * (Un * dUn).sum(1, keepdims=True)) / np.linalg.norm(U, axis=1, keepdims=True)
        dV = (dVn - Vn * (Vn * dVn).sum(1, keepdims=True)) / np.linalg.norm(V, axis=1, keepdims=True)
        u.bwd(dU); v.bwd(dV)
        sc = 1.0 - 0.8 * s / steps
        u.step(sc); v.step(sc)
    return u, v


def sel_logits(sel, S0, acc):
    u, v = sel
    U = u.predict(S0); V = v.predict(afeat(acc))
    Un = U / np.linalg.norm(U, axis=1, keepdims=True); Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    return TAU * Un @ Vn.T


# ----------------------------------------------------------------------------- metrics
def rank(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def regret_stats(Jgrid, Jcol, sel):
    """Jgrid (n,K) true feasible cost; sel (n,) chosen index; Jcol = min gap<=0.5 flag (n,K)."""
    n = len(sel); best = Jgrid.min(1); jb = Jgrid[np.arange(n), sel]
    unsafe = Jcol[np.arange(n), sel] & ~Jcol[np.arange(n), Jgrid.argmin(1)]
    return dict(regret=float((jb - best).mean()), top1=float((sel == Jgrid.argmin(1)).mean()),
                within05=float(((jb - best) <= 0.5).mean()), unsafe=float(unsafe.mean()))


# ----------------------------------------------------------------------------- pipeline
def build_data(enc, rng, n, snr, aug):
    W = finish_world(sample_world(rng, n), rng)
    v, g, vl = simulate(W, GRID)
    J = total_cost(v, g, GRID)
    kE = (J + rng.normal(0, EXPERT_SIGMA, J.shape)).argmin(1)
    cand = [kE[:, None]]
    if aug:
        cand.append(rng.integers(0, K, (n, 3)))
    cand = np.concatenate(cand, 1)                                     # (n, R) candidate index per row
    R = cand.shape[1]
    ii = np.repeat(np.arange(n), R); kk = cand.reshape(-1)
    Wsub = {k: val[ii] for k, val in W.items()}
    vs, gs = v[ii, kk][:, None, :], g[ii, kk][:, None, :]
    S = observe(enc, Wsub, vs, gs, vl[ii], snr)[:, 0]                  # (n*R, T+1, EMB)
    cs = state_cost(vs, gs)[:, 0]                                      # (n*R, T)
    s0 = observe(enc, W, v[:, :1], g[:, :1], vl, snr)[:, 0, 0]         # (n, EMB)  state at t=0 (same for all k)
    desc6 = np.stack([v[ii, kk, 6], g[ii, kk, 6], vl[ii, 6]], 1)
    return dict(W=W, v=v, g=g, vl=vl, J=J, kE=kE, S=S, cs=cs, ii=ii, kk=kk, s0=s0, n=n, desc6=desc6)


def fit_imagination(D, rng):
    """direct heads (A,B), one-step heads (A,B), cost head."""
    S, cs, kk = D["S"], D["cs"], D["kk"]
    acc = GRID[kk]; af = afeat(acc)
    Xd = np.concatenate([S[:, 0], af], 1)
    Yd = np.concatenate([S[:, 2] - S[:, 0], S[:, 4] - S[:, 0], S[:, 6] - S[:, 0]], 1)   # residual to s0
    heads = {}
    for name in "AB":
        heads["d" + name] = fit_regress(MLP([EMB + 4, 128, 3 * EMB], rng), Xd, Yd, rng, steps=1000, boot=True)
    steps_a = np.where(np.arange(T)[None, :] < 3, acc[:, :1], acc[:, 1:2])          # (rows, T)
    Xs = np.concatenate([S[:, :-1].reshape(-1, EMB), steps_a.reshape(-1, 1) / 5.0], 1)
    Ys = (S[:, 1:] - S[:, :-1]).reshape(-1, EMB)
    for name in "AB":
        heads["s" + name] = fit_regress(MLP([EMB + 1, 64, EMB], rng), Xs, Ys, rng, steps=1100, boot=True)
    heads["c"] = fit_regress(MLP([EMB, 48, 1], rng), S[:, 1:].reshape(-1, EMB), cs.reshape(-1, 1) / 10.0, rng)
    # Jensen-gap guard: refit the cost head on IMAGINED latents (target = the real per-step cost), so that
    # c'(s_hat) estimates E[c | s_hat] instead of c(E[s']) -- the cost of the mean future is not the mean cost.
    s0rows = S[:, 0]
    Yim = 0.5 * (direct_states(heads["dA"], s0rows, Xd) + direct_states(heads["dB"], s0rows, Xd))
    heads["c_dir"] = fit_regress(MLP([EMB, 48, 1], rng), Yim.reshape(-1, EMB),
                                 cs[:, [1, 3, 5]].reshape(-1, 1) / 10.0, rng, steps=700)
    idx = np.arange(len(S))
    _, _, trj = imagine_rollout(heads, s0rows, acc, idx, want_traj=True)
    Xr_ = np.concatenate([trj[0].reshape(-1, EMB), trj[1].reshape(-1, EMB)], 0)
    Yr_ = np.concatenate([cs.reshape(-1, 1), cs.reshape(-1, 1)], 0) / 10.0
    heads["c_rol"] = fit_regress(MLP([EMB, 48, 1], rng), Xr_, Yr_, rng, steps=700)
    return heads


def direct_states(net, s0rows, X):
    """direct multi-horizon head: s_hat_h = normalize(s0 + delta_h), h in {2, 4, 6} (1, 2, 3 s)."""
    d = net.predict(X).reshape(-1, 3, EMB) + s0rows[:, None, :]
    return d / (np.linalg.norm(d, axis=2, keepdims=True) + 1e-9)


def imagine_direct(heads, s0, acc_rows, s0_idx, ckey="c"):
    """returns J_hat mean over the 2 heads, and |A-B| disagreement, for rows (tick s0_idx, candidate acc_rows)."""
    X = np.concatenate([s0[s0_idx], afeat(acc_rows)], 1)
    outs = []
    for name in "AB":
        Y = direct_states(heads["d" + name], s0[s0_idx], X)
        c = heads[ckey].predict(Y.reshape(-1, EMB)).reshape(-1, 3) * 10.0
        outs.append(2.0 * c.sum(1))                                      # each horizon stands for 2 steps
    comfort, _ = act_cost(acc_rows)
    return 0.5 * (outs[0] + outs[1]) + comfort, np.abs(outs[0] - outs[1])


def imagine_rollout(heads, s0, acc_rows, s0_idx, want_traj=False, ckey="c"):
    """iterated one-step rollouts; returns J_hat, disagreement (|JA-JB|) and optionally the per-step states."""
    res, traj = [], []
    for name in "AB":
        s = s0[s0_idx].copy(); c = np.zeros(len(s)); tr = []
        for t in range(T):
            a = (acc_rows[:, 0] if t < 3 else acc_rows[:, 1])[:, None] / 5.0
            d = heads["s" + name].predict(np.concatenate([s, a], 1))
            s = s + d
            s = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-9)
            c += heads[ckey].predict(s)[:, 0] * 10.0
            tr.append(s)
        res.append(c); traj.append(np.stack(tr, 1))
    comfort, _ = act_cost(acc_rows)
    out = (0.5 * (res[0] + res[1]) + comfort, np.abs(res[0] - res[1]))
    return out + (traj,) if want_traj else out


def pick_beta(logit, Jhat_top, top_idx, Jgrid, Jcol, grid=(0.0, 0.25, 0.5, 1, 2, 4, 1e6)):
    best = None
    for beta in grid:
        sel = combine(logit, Jhat_top, top_idx, beta)
        r = regret_stats(Jgrid, Jcol, sel)["regret"]
        if best is None or r < best[0] - 1e-9:
            best = (r, beta)
    return best[1]


def combine(logit, Jhat_top, top_idx, beta):
    n = len(logit)
    lt = np.take_along_axis(logit, top_idx, 1)
    sc = lt - beta * (Jhat_top - Jhat_top.min(1, keepdims=True)) if beta < 1e5 else -Jhat_top
    return top_idx[np.arange(n), sc.argmax(1)]


def run_one(seed, n, snr, enc, do_extra=False, aug=True, n_test=900, n_val=400):
    rng = np.random.default_rng(seed)
    D = build_data(enc, rng, n, snr, aug)
    onehot = np.eye(K)[D["kE"]]
    sel0 = fit_selector(D["s0"], onehot, rng)                              # A0: hard-label imitation (CLM-style)
    heads = fit_imagination(D, rng)
    # model-free Q baseline on the SAME rows the imagination heads see (expert [+ counterfactual-augmented])
    accq = GRID[D["kk"]]
    qm = fit_regress(MLP([EMB + 4, 128, 1], rng), np.concatenate([D["S"][:, 0], afeat(accq)], 1),
                     D["cs"].sum(1, keepdims=True) / 30.0, rng, steps=1000)
    out = {"n": n, "snr": snr, "aug": aug, "seed": seed}

    def evaluate(nn, seed_off):
        r2 = np.random.default_rng(seed * 1000 + seed_off)
        W = finish_world(sample_world(r2, nn), r2)
        allacc = np.concatenate([GRID, WILD], 0)
        v, g, vl = simulate(W, allacc)
        Jall_noenv = total_cost(v, g, allacc, with_env=False)
        Jall = total_cost(v, g, allacc, with_env=True)
        col = (g[..., 1:].min(-1) <= 0.5)
        s0 = observe(enc, W, v[:, :1], g[:, :1], vl, snr)[:, 0, 0]
        return dict(W=W, v=v, g=g, vl=vl, Jgrid=Jall[:, :K], Jcol=col[:, :K], Jall=Jall, Jall_noenv=Jall_noenv,
                    colall=col, s0=s0, n=nn)

    V = evaluate(n_val, 1); Te = evaluate(n_test, 2)

    def arms(E, sel, mask_env, use_wild):
        acc = np.concatenate([GRID, WILD], 0) if use_wild else GRID
        Kc = len(acc)
        Jtrue = E["Jall"][:, :Kc]
        logit = sel_logits(sel, E["s0"], acc)
        if mask_env:                                                         # analytic feasibility mask
            _, env = act_cost(acc); logit = logit - 1e3 * env[None, :]
        o = {"logit": logit, "Jtrue": Jtrue, "acc": acc}
        o["A0"] = logit.argmax(1)
        top = np.argsort(-logit, 1)[:, :KI]
        rows_i = np.repeat(np.arange(E["n"]), KI); rows_a = acc[top.reshape(-1)]
        Jh, dis = imagine_direct(heads, E["s0"], rows_a, rows_i)
        if mask_env:
            _, env = act_cost(rows_a); Jh = Jh + 1e3 * env
        o["top"], o["Jh_top"], o["dis_top"] = top, Jh.reshape(-1, KI), dis.reshape(-1, KI)
        return o

    # ---- A0 / A1 (beta on validation)
    Vo = arms(V, sel0, True, False)
    beta = pick_beta(Vo["logit"], Vo["Jh_top"], Vo["top"], V["Jgrid"], V["Jcol"])
    To = arms(Te, sel0, True, False)
    res = {"A0": regret_stats(Te["Jgrid"], Te["Jcol"], To["A0"])}
    sel1 = combine(To["logit"], To["Jh_top"], To["top"], beta)
    res["A1"] = regret_stats(Te["Jgrid"], Te["Jcol"], sel1)
    out["beta_A1"] = float(beta)
    # ---- A2 full rollouts over all K (beta tuned on validation too)
    def roll_all(E):
        rows_i = np.repeat(np.arange(E["n"]), K); rows_a = np.tile(GRID, (E["n"], 1))
        Jh, dis = imagine_rollout(heads, E["s0"], rows_a, rows_i)
        return Jh.reshape(-1, K), dis.reshape(-1, K)
    Jr_v, _ = roll_all(V); Jr_t, dis_r = roll_all(Te)
    allidx_v = np.tile(np.arange(K), (V["n"], 1)); allidx_t = np.tile(np.arange(K), (Te["n"], 1))
    beta2 = pick_beta(Vo["logit"], Jr_v, allidx_v, V["Jgrid"], V["Jcol"])
    res["A2"] = regret_stats(Te["Jgrid"], Te["Jcol"], combine(To["logit"], Jr_t, allidx_t, beta2))
    res["A2_pure"] = regret_stats(Te["Jgrid"], Te["Jcol"], combine(To["logit"], Jr_t, allidx_t, 1e6))
    out["beta_A2"] = float(beta2)
    # ---- Jensen-gap guard arms: same predictors, cost head refit on imagined latents
    def topk_J(E, o, ckey):
        ri_ = np.repeat(np.arange(E["n"]), KI); ra_ = o["acc"][o["top"].reshape(-1)]
        return imagine_direct(heads, E["s0"], ra_, ri_, ckey)[0].reshape(-1, KI)
    Jc_v, Jc_t = topk_J(V, Vo, "c_dir"), topk_J(Te, To, "c_dir")
    beta1c = pick_beta(Vo["logit"], Jc_v, Vo["top"], V["Jgrid"], V["Jcol"])
    res["A1c_cost_on_imagined"] = regret_stats(Te["Jgrid"], Te["Jcol"], combine(To["logit"], Jc_t, To["top"], beta1c))
    def roll_all_c(E):
        ri_ = np.repeat(np.arange(E["n"]), K); ra_ = np.tile(GRID, (E["n"], 1))
        return imagine_rollout(heads, E["s0"], ra_, ri_, ckey="c_rol")[0].reshape(-1, K)
    Jrc_v, Jrc_t = roll_all_c(V), roll_all_c(Te)
    beta2c = pick_beta(Vo["logit"], Jrc_v, allidx_v, V["Jgrid"], V["Jcol"])
    res["A2c_cost_on_imagined"] = regret_stats(Te["Jgrid"], Te["Jcol"], combine(To["logit"], Jrc_t, allidx_t, beta2c))
    res["A2c_pure"] = regret_stats(Te["Jgrid"], Te["Jcol"], combine(To["logit"], Jrc_t, allidx_t, 1e6))
    out["beta_A1c"], out["beta_A2c"] = float(beta1c), float(beta2c)
    # ---- QM: model-free Q head, pure argmin of q(s,a) + analytic comfort (same rows / same analytic terms)
    def q_all(E):
        rows_i = np.repeat(np.arange(E["n"]), K); rows_a = np.tile(GRID, (E["n"], 1))
        q = qm.predict(np.concatenate([E["s0"][rows_i], afeat(rows_a)], 1))[:, 0] * 30.0
        return q.reshape(-1, K) + act_cost(GRID)[0][None, :]
    res["QM_modelfree_Q"] = regret_stats(Te["Jgrid"], Te["Jcol"], q_all(Te).argmin(1))
    # ---- A5: FACTORISED imagination.  Learn only the exogenous hidden consequence from LOGS ALONE (no counterfactual
    #      rows, no augmentation): a readout of the visible state and the lead-mode posterior p(z|s); evaluate every
    #      candidate ANALYTICALLY (ego integration + mode-conditioned lead law) and take the expected cost.
    Wd = D["W"]
    ro = fit_regress(MLP([EMB, 64, 3], rng), D["s0"], np.stack([Wd["v0"] / 20, Wd["d0"] / 50, Wd["vl0"] / 20], 1), rng, steps=800)
    mode = fit_classifier(MLP([EMB, 64, 3], rng), D["s0"], Wd["z"], rng, steps=800)

    # validation temperature scaling of the mode posterior (calibration guard; NLL on validation ticks)
    lgv = mode.predict(V["s0"]).astype(float); zv = V["W"]["z"]; bestT = None
    for Tm in (1.0, 1.5, 2.0, 3.0, 4.0):
        pv = np.exp(lgv / Tm - (lgv / Tm).max(1, keepdims=True)); pv /= pv.sum(1, keepdims=True)
        nll = -np.log(pv[np.arange(len(zv)), zv] + 1e-9).mean()
        if bestT is None or nll < bestT[0]:
            bestT = (nll, Tm)
    Tcal = bestT[1]; out["a5_T"] = float(Tcal)

    def a5_expected_cost(E, acc=GRID, T_=1.0, onsets=(0.5,)):
        r = ro.predict(E["s0"]).astype(float); lg = mode.predict(E["s0"]).astype(float) / T_
        pz = np.exp(lg - lg.max(1, keepdims=True)); pz /= pz.sum(1, keepdims=True)
        Jk = 0.0
        for k in range(3):
            for tb_ in onsets:                                        # marginalise the second hidden factor (onset time)
                Wk = dict(v0=np.clip(20 * r[:, 0], 1, 30), d0=np.clip(50 * r[:, 1], 2, 80), vl0=np.clip(20 * r[:, 2], 0, 40),
                          z=np.full(E["n"], k), tb=np.full(E["n"], tb_))
                vk, gk, _ = simulate(Wk, acc)
                Jk = Jk + pz[:, k:k + 1] * total_cost(vk, gk, acc) / len(onsets)
        return Jk, pz
    Ja5, pz5 = a5_expected_cost(Te)
    res["A5_factorised_mode"] = regret_stats(Te["Jgrid"], Te["Jcol"], Ja5[:, :K].argmin(1))
    Ja5m, _ = a5_expected_cost(Te, onsets=(0.15, 0.5, 0.85))
    res["A5m_onset_marginalised"] = regret_stats(Te["Jgrid"], Te["Jcol"], Ja5m[:, :K].argmin(1))
    # diagnostic: same analytic arm but with the TRUE visible state instead of the readout r(s) (isolates readout error)
    pzt = np.exp(mode.predict(Te["s0"]).astype(float)); pzt /= pzt.sum(1, keepdims=True)
    Jtv = 0.0
    for k in range(3):
        for tb_ in (0.15, 0.5, 0.85):
            Wk = dict(v0=Te["W"]["v0"], d0=Te["W"]["d0"], vl0=Te["W"]["vl0"], z=np.full(Te["n"], k), tb=np.full(Te["n"], tb_))
            vk, gk, _ = simulate(Wk, GRID)
            Jtv = Jtv + pzt[:, k:k + 1] * total_cost(vk, gk, GRID) / 3.0
    res["DIAG_A5m_true_visible_state"] = regret_stats(Te["Jgrid"], Te["Jcol"], Jtv.argmin(1))
    Ja5c, _ = a5_expected_cost(Te, T_=Tcal)
    res["A5cal_temp_scaled"] = regret_stats(Te["Jgrid"], Te["Jcol"], Ja5c[:, :K].argmin(1))
    out["a5_mode_acc"] = float((pz5.argmax(1) == Te["W"]["z"]).mean())
    # ---- REF: z-blind, risk-sensitive lookup on the TRUE visible state (no cue, no learning of z): the floor that
    #      separates 'risk-sensitive expected-cost selection' from 'access to the hidden consequence'
    def vbins(Wx):
        return (np.digitize(Wx["d0"], [20, 28, 36]) * 25 + np.digitize(Wx["v0"], [11, 14, 17]) * 5
                + np.digitize(Wx["vl0"] - Wx["v0"], [-1, 0, 1]))
    btr, bte = vbins(D["W"]), vbins(Te["W"])
    Jm = D["J"].mean(0).argmin()
    pol = {b: D["J"][btr == b].mean(0).argmin() for b in np.unique(btr)}
    res["REF_zblind_visible_bins"] = regret_stats(Te["Jgrid"], Te["Jcol"], np.array([pol.get(b, Jm) for b in bte]))
    res["ORACLE"] = regret_stats(Te["Jgrid"], Te["Jcol"], Te["Jgrid"].argmin(1))
    res["CONST_BEST"] = regret_stats(Te["Jgrid"], Te["Jcol"], np.full(Te["n"], np.bincount(
        Te["Jgrid"].argmin(1), minlength=K).argmax()))
    # ---- A4 distilled (soft targets from the learned direct head over ALL K on train ticks);
    #      A0s = same recipe with TRUE per-candidate cost (privileged Hydra-style teacher) for reference
    rows_i = np.repeat(np.arange(D["n"]), K); rows_a = np.tile(GRID, (D["n"], 1))
    Jh_tr, _ = imagine_direct(heads, D["s0"], rows_a, rows_i); Jh_tr = Jh_tr.reshape(-1, K)

    def soft_selector(Jsrc, tag):
        best = None
        for temp in (1.5, 4.0):
            tgt = np.exp(-(Jsrc - Jsrc.min(1, keepdims=True)) / temp); tgt /= tgt.sum(1, keepdims=True)
            sel_ = fit_selector(D["s0"], 0.7 * tgt + 0.3 * onehot, np.random.default_rng(seed + 7), steps=1200)
            lv = sel_logits(sel_, V["s0"], GRID) - 1e3 * act_cost(GRID)[1][None, :]
            rv = regret_stats(V["Jgrid"], V["Jcol"], lv.argmax(1))["regret"]
            if best is None or rv < best[0]:
                best = (rv, temp, sel_)
        out[tag + "_temp"] = best[1]
        return best[2]
    sel4 = soft_selector(Jh_tr, "A4")
    sel0s = soft_selector(D["J"], "A0s")
    lt4 = sel_logits(sel4, Te["s0"], GRID) - 1e3 * act_cost(GRID)[1][None, :]
    res["A4_distilled"] = regret_stats(Te["Jgrid"], Te["Jcol"], lt4.argmax(1))
    lts = sel_logits(sel0s, Te["s0"], GRID) - 1e3 * act_cost(GRID)[1][None, :]
    res["A0s_true_teacher"] = regret_stats(Te["Jgrid"], Te["Jcol"], lts.argmax(1))
    # A1s: imagination re-rank on top of the stronger proposer A0s (beta on validation)
    Vs = arms(V, sel0s, True, False); Ts = arms(Te, sel0s, True, False)
    beta_s = pick_beta(Vs["logit"], Vs["Jh_top"], Vs["top"], V["Jgrid"], V["Jcol"])
    res["A1s_on_A0s"] = regret_stats(Te["Jgrid"], Te["Jcol"], combine(Ts["logit"], Ts["Jh_top"], Ts["top"], beta_s))
    out["beta_A1s"] = float(beta_s)
    out["oracle_rerank_topKi_regret_A0s"] = float((np.take_along_axis(Te["Jgrid"], Ts["top"], 1).min(1) - Te["Jgrid"].min(1)).mean())
    out["res"] = res
    Jg_t = Te["Jgrid"]; nT = Te["n"]; ar = np.arange(nT)
    sel2 = combine(To["logit"], Jr_t, allidx_t, beta2)
    out["ticks"] = {"A0": (Jg_t[ar, To["A0"]] - Jg_t.min(1)).tolist(), "A1": (Jg_t[ar, sel1] - Jg_t.min(1)).tolist(),
                    "A2": (Jg_t[ar, sel2] - Jg_t.min(1)).tolist(),
                    "A4_distilled": (Jg_t[ar, lt4.argmax(1)] - Jg_t.min(1)).tolist(),
                    "QM_modelfree_Q": (Jg_t[ar, q_all(Te).argmin(1)] - Jg_t.min(1)).tolist(),
                    "A5": (Jg_t[ar, Ja5[:, :K].argmin(1)] - Jg_t.min(1)).tolist(),
                    "A5m": (Jg_t[ar, Ja5m[:, :K].argmin(1)] - Jg_t.min(1)).tolist(),
                    "A5cal": (Jg_t[ar, Ja5c[:, :K].argmin(1)] - Jg_t.min(1)).tolist(),
                    "A1s": (Jg_t[ar, combine(Ts["logit"], Ts["Jh_top"], Ts["top"], beta_s)] - Jg_t.min(1)).tolist(),
                    "A2c": (Jg_t[ar, combine(To["logit"], Jrc_t, allidx_t, beta2c)] - Jg_t.min(1)).tolist(),
                    "A1c": (Jg_t[ar, combine(To["logit"], Jc_t, To["top"], beta1c)] - Jg_t.min(1)).tolist(),
                    "A0s_true_teacher": (Jg_t[ar, lts.argmax(1)] - Jg_t.min(1)).tolist()}
    # recall bound: what a perfect re-ranker of A0's top-Ki could achieve
    Jtop = np.take_along_axis(Jg_t, To["top"], 1)
    out["oracle_rerank_topKi_regret"] = float((Jtop.min(1) - Jg_t.min(1)).mean())
    # ---- MACs per tick (toy units; multiply-adds of the learned parts only)
    umac = sel0[0].macs; kdot = K * 16
    dmac = (heads["dA"].macs + 3 * heads["c"].macs) * 2                  # two heads
    smac = (heads["sA"].macs + heads["c"].macs) * 2
    out["macs"] = dict(A0=umac + kdot, A1=umac + kdot + KI * dmac, A2=umac + kdot + K * T * smac, QM=K * qm.macs, A5=ro.macs + mode.macs + 3 * K * T * 14, A5m=ro.macs + mode.macs + 9 * K * T * 14,
                       A4=umac + kdot, per_cand_direct=dmac, per_cand_roll=T * smac)
    if not do_extra:
        return out
    # ================================================================= extras (single configuration)
    ex = {}
    # --- gates on A1: fixed policies, learned gate, oracle gate
    def gate_inputs(E, o, sel_final):
        lg = np.sort(o["logit"], 1)[:, ::-1]; margin = lg[:, 0] - lg[:, 1]
        pr = np.exp(o["logit"] - o["logit"].max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        ent = -(pr * np.log(pr + 1e-12)).sum(1)
        dis = o["dis_top"].mean(1); spread = o["Jh_top"].max(1) - o["Jh_top"].min(1)
        F = np.stack([margin, ent, dis, spread, lg[:, 0], np.log1p(dis), np.log1p(spread)], 1)
        ar_ = np.arange(E["n"])
        reg0 = E["Jgrid"][ar_, o["A0"]] - E["Jgrid"].min(1)
        reg1 = E["Jgrid"][ar_, sel_final] - E["Jgrid"].min(1)
        return F, margin, dis, reg0, reg1
    sel1_v = combine(Vo["logit"], Vo["Jh_top"], Vo["top"], beta)
    Fv, _, _, r0v, r1v = gate_inputs(V, Vo, sel1_v)
    F_t, margin, dis_i, regret_A0, regret_A1 = gate_inputs(Te, To, sel1)
    mu, sd = Fv.mean(0), Fv.std(0) + 1e-9
    Xg = np.concatenate([(Fv - mu) / sd, np.ones((len(Fv), 1))], 1)
    wg = np.linalg.solve(Xg.T @ Xg + 3.0 * np.eye(Xg.shape[1]), Xg.T @ (r0v - r1v))      # ridge: predicted benefit
    pred_benefit = np.concatenate([(F_t - mu) / sd, np.ones((len(F_t), 1))], 1) @ wg
    n_t = Te["n"]; gate = {}
    r_m, r_d = rank(margin) / n_t, rank(dis_i) / n_t
    scores = dict(random=np.random.default_rng(5).random(n_t), margin_low=r_m, trust_low_disagree=r_d,
                  need_and_trust=r_m + r_d, learned_benefit=-pred_benefit,
                  oracle_benefit=-(regret_A0 - regret_A1), anti_high_disagree=-r_d)
    for f in (0.1, 0.2, 0.3, 0.5, 1.0):
        row = {}
        for nm, sc in scores.items():
            k_on = int(round(f * n_t)); on = np.zeros(n_t, bool); on[np.argsort(sc)[:k_on]] = True
            row[nm] = float(np.where(on, regret_A1, regret_A0).mean())
        gate[str(f)] = row
    ex["gate_regret"] = gate
    ex["gate_regret_A0"] = float(regret_A0.mean()); ex["gate_regret_A1"] = float(regret_A1.mean())
    ex["gate_benefit_spearman_learned"] = spearman(pred_benefit, regret_A0 - regret_A1)
    ex["gate_frac_ticks_where_imagination_helps"] = float(((regret_A0 - regret_A1) > 0.25).mean())
    ex["gate_frac_ticks_where_imagination_hurts"] = float(((regret_A0 - regret_A1) < -0.25).mean())
    # --- calibration diagnostics
    Jt_top = np.take_along_axis(To["Jtrue"], To["top"], 1)
    err_top = np.abs(To["Jh_top"] - Jt_top)
    ex["spearman_dis_vs_err_direct"] = spearman(To["dis_top"].reshape(-1), err_top.reshape(-1))
    P = np.exp(To["logit"] - To["logit"].max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    ex["spearman_conf_vs_regret_A0"] = spearman(P.max(1), regret_A0)
    lt = np.take_along_axis(To["logit"], To["top"], 1)
    sc1 = lt - beta * (To["Jh_top"] - To["Jh_top"].min(1, keepdims=True)) if beta < 1e5 else -To["Jh_top"]
    P1 = np.exp(sc1 - sc1.max(1, keepdims=True)); P1 /= P1.sum(1, keepdims=True)
    ex["spearman_conf_vs_regret_A1"] = spearman(P1.max(1), regret_A1)
    # rollout: error vs disagreement per horizon (subsample rows)
    sub = np.random.default_rng(3).choice(Te["n"] * K, 6000, replace=False)
    ri = sub // K; ra = GRID[sub % K]
    _, _, trajs = imagine_rollout(heads, Te["s0"], ra, ri, want_traj=True)
    vv, gg, vll = simulate(Te["W"], GRID)
    Strue = observe(enc, Te["W"], vv, gg, vll, snr)                          # (n,K,T+1,EMB)
    horizon = []
    for t in range(T):
        e = np.linalg.norm(0.5 * (trajs[0][:, t] + trajs[1][:, t]) - Strue[ri, sub % K, t + 1], axis=1)
        d = np.linalg.norm(trajs[0][:, t] - trajs[1][:, t], axis=1)
        horizon.append(dict(t=t + 1, mean_err=float(e.mean()), mean_dis=float(d.mean()), spearman=spearman(d, e)))
    ex["rollout_horizon"] = horizon
    X = np.concatenate([Te["s0"][ri], afeat(ra)], 1)
    YA = direct_states(heads["dA"], Te["s0"][ri], X); YB = direct_states(heads["dB"], Te["s0"][ri], X)
    dh = []
    for j, t in enumerate((2, 4, 6)):
        e = np.linalg.norm(0.5 * (YA[:, j] + YB[:, j]) - Strue[ri, sub % K, t], axis=1)
        d = np.linalg.norm(YA[:, j] - YB[:, j], axis=1)
        dh.append(dict(t=t, mean_err=float(e.mean()), mean_dis=float(d.mean()), spearman=spearman(d, e)))
    ex["direct_horizon"] = dh
    # --- out-of-vocabulary (WILD) candidates.  types: 0,1 = hard accel (collide with the lead);
    #     2,3 = infeasible over-braking (the simulator, like a faithful world model, OBEDIENTLY simulates them)
    wild = {}
    for mask in (False, True):
        Wo = arms(Te, sel0, mask, True)
        sel_w1 = combine(Wo["logit"], Wo["Jh_top"], Wo["top"], beta)
        # imagination-only over K + WILD (direct head on every candidate): pure argmin of imagined cost
        accw = np.concatenate([GRID, WILD], 0); Kc = len(accw)
        ri_a = np.repeat(np.arange(Te["n"]), Kc); ra_a = np.tile(accw, (Te["n"], 1))
        Jh_all, _ = imagine_direct(heads, Te["s0"], ra_a, ri_a); Jh_all = Jh_all.reshape(-1, Kc)
        if mask:
            Jh_all = Jh_all + 1e3 * act_cost(accw)[1][None, :]
        sel_io = Jh_all.argmin(1)
        key = "mask" if mask else "nomask"
        wild[key] = dict(
            A0_selects_wild=float((Wo["A0"] >= K).mean()), A1_selects_wild=float((sel_w1 >= K).mean()),
            IMAG_ONLY_selects_wild=float((sel_io >= K).mean()),
            A0_wild_by_type=[float((Wo["A0"] == K + j).mean()) for j in range(len(WILD))],
            IMAG_ONLY_wild_by_type=[float((sel_io == K + j).mean()) for j in range(len(WILD))])
    rows_i = np.repeat(np.arange(Te["n"]), len(WILD)); rows_a = np.tile(WILD, (Te["n"], 1))
    Jw, dw = imagine_direct(heads, Te["s0"], rows_a, rows_i)
    rows_i2 = np.repeat(np.arange(Te["n"]), K); rows_a2 = np.tile(GRID, (Te["n"], 1))
    Jg, dg = imagine_direct(heads, Te["s0"], rows_a2, rows_i2)
    Jw = Jw.reshape(-1, len(WILD)); dwr = dw.reshape(-1, len(WILD))
    obedient = Te["Jall_noenv"][:, K:]
    wild["imagined_cost_by_type"] = [float(Jw[:, j].mean()) for j in range(len(WILD))]
    wild["obedient_sim_cost_by_type"] = [float(obedient[:, j].mean()) for j in range(len(WILD))]
    wild["envelope_penalty_added_by_mask"] = 60.0
    wild["imagined_cost_grid_mean"] = float(Jg.mean()); wild["obedient_sim_cost_grid_mean"] = float(Te["Jall_noenv"][:, :K].mean())
    wild["disagreement_wild_mean"] = float(dw.mean()); wild["disagreement_grid_mean"] = float(dg.mean())
    lab = np.concatenate([np.zeros(len(dg)), np.ones(len(dw))]); sc = np.concatenate([dg, dw])
    wild["auroc_disagreement_flags_wild"] = float(_auroc(lab, sc))
    thr = np.percentile(dg, 95)
    wild["veto_recall_at_5pct_vocab_fpr"] = float((dw > thr).mean())
    wild["veto_recall_by_type"] = [float((dwr[:, j] > thr).mean()) for j in range(len(WILD))]
    # A5 analytic arm over K + WILD, envelope penalty ON (it is part of total_cost) vs OFF
    accw = np.concatenate([GRID, WILD], 0)
    r_ = ro.predict(Te["s0"]).astype(float); lg_ = mode.predict(Te["s0"]).astype(float)
    pz_ = np.exp(lg_ - lg_.max(1, keepdims=True)); pz_ /= pz_.sum(1, keepdims=True)
    for env_on in (True, False):
        Jk = 0.0
        for k in range(3):
            Wk = dict(v0=np.clip(20 * r_[:, 0], 1, 30), d0=np.clip(50 * r_[:, 1], 2, 80), vl0=np.clip(20 * r_[:, 2], 0, 40),
                      z=np.full(Te["n"], k), tb=np.full(Te["n"], 0.5))
            vk, gk, _ = simulate(Wk, accw)
            Jk = Jk + pz_[:, k:k + 1] * total_cost(vk, gk, accw, with_env=env_on)
        wild["A5_selects_wild_env_" + ("on" if env_on else "off")] = float((Jk.argmin(1) >= K).mean())
    ex["wild"] = wild
    # --- echo / shuffled-cue control on the direct head (ridge probe from s to descriptors at t = 6)
    r = np.random.default_rng(11)
    Wc = dict(Te["W"]); perm = r.permutation(Te["n"])
    cue_sh = (snr * cue_level(Wc)[:, :, None] * np.ones(C)[None, None, :] + CUE_SIGMA * Wc["noise"])[perm]
    s0_sh = observe(enc, Wc, vv[:, :1], gg[:, :1], vll, snr, cue_override=cue_sh)[:, 0, 0]
    Strn = D["S"][:, 6]
    lam = 1e-2
    Xr = np.concatenate([Strn, np.ones((len(Strn), 1))], 1)
    Wr = np.linalg.solve(Xr.T @ Xr + lam * np.eye(Xr.shape[1]), Xr.T @ D["desc6"])
    sub2 = np.random.default_rng(4).choice(Te["n"] * K, 8000, replace=False); ri2 = sub2 // K; ka = sub2 % K
    true_desc = np.stack([vv[ri2, ka, 6], gg[ri2, ka, 6], vll[ri2, 6]], 1)
    echo = {}
    for nm, s0x in (("real_cue", Te["s0"]), ("shuffled_cue", s0_sh)):
        Xh = np.concatenate([s0x[ri2], afeat(GRID[ka])], 1)
        Y6 = 0.5 * (direct_states(heads["dA"], s0x[ri2], Xh) + direct_states(heads["dB"], s0x[ri2], Xh))[:, 2]
        pred = np.concatenate([Y6, np.ones((len(Y6), 1))], 1) @ Wr
        echo[nm] = {k: float(1 - ((pred[:, i] - true_desc[:, i]) ** 2).sum() / ((true_desc[:, i] - true_desc[:, i].mean()) ** 2).sum())
                    for i, k in enumerate(("ego_speed_t6", "gap_t6", "lead_speed_t6"))}
    ex["echo_control_R2"] = echo
    out["extra"] = ex
    return out


def _auroc(lab, sc):
    r = rank(sc) + 1.0
    n1 = lab.sum(); n0 = len(lab) - n1
    return (r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ----------------------------------------------------------------------------- calibration of the WORLD (no arm is run)
def calibrate():
    """World is frozen by two criteria only, decided before any arm ran:
    (1) a z-blind policy (best candidate per visible-state bin) must have materially positive regret;
    (2) oracle-z regret is 0 by construction.  Prints the diagnostics; no learner is involved."""
    rng = np.random.default_rng(99)
    W = finish_world(sample_world(rng, 6000), rng)
    v, g, vl = simulate(W, GRID)
    J = total_cost(v, g, GRID)
    best = J.argmin(1)
    print("best-candidate histogram by z (rows z=0,1,2; cols = K):")
    for z in range(3):
        h = np.bincount(best[W["z"] == z], minlength=K)
        print(z, " top3:", [(int(i), tuple(GRID[i]), int(h[i])) for i in np.argsort(-h)[:3]])
    tr, te = np.arange(3000), np.arange(3000, 6000)
    def bins(idx):
        b = (np.digitize(W["d0"][idx], [20, 28, 36]) * 25 + np.digitize(W["v0"][idx], [11, 14, 17]) * 5
             + np.digitize((W["vl0"] - W["v0"])[idx], [-1, 0, 1]))
        return b
    btr, bte = bins(tr), bins(te)
    pol = {}
    for b in np.unique(btr):
        pol[b] = J[tr][btr == b].mean(0).argmin()
    sel = np.array([pol.get(b, 0) for b in bte])
    n = len(te)
    r_blind = (J[te][np.arange(n), sel] - J[te].min(1)).mean()
    col = (g[..., 1:].min(-1) <= 0.5)
    unsafe = (col[te][np.arange(n), sel] & ~col[te][np.arange(n), J[te].argmin(1)]).mean()
    print(f"z-blind (visible-bin) policy: regret {r_blind:.3f}  unsafe {unsafe:.3f}   oracle-z: regret 0")
    print("J spread: mean(best) %.2f  mean(worst) %.2f  std over K (per tick) %.2f"
          % (J.min(1).mean(), J.max(1).mean(), J.std(1).mean()))
    print("collision-free-best fraction:", float((~col[np.arange(6000), best]).mean()))


# ----------------------------------------------------------------------------- driver
def agg(vals):
    a = np.array(vals, float)
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0, per_seed=[float(x) for x in a])


def agg_tree(trees):
    """mean over seeds of a nested dict/list of floats (sd kept for leaves)."""
    t0 = trees[0]
    if isinstance(t0, dict):
        return {k: agg_tree([t[k] for t in trees]) for k in t0}
    if isinstance(t0, list):
        return [agg_tree([t[i] for t in trees]) for i in range(len(t0))]
    return agg(trees)


def paired_boot(x, y, B=2000, seed=0):
    """paired bootstrap over pooled test ticks (i.i.d. in the toy): [mean(x - y), 2.5 %, 97.5 %]."""
    r = np.random.default_rng(seed); d = np.asarray(x) - np.asarray(y); n = len(d)
    m = np.array([d[r.integers(0, n, n)].mean() for _ in range(B)])
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def _job(a):
    sd, snr, n, aug, extra = a
    return run_one(sd, n, snr, Encoder(), do_extra=extra, aug=aug)


CONFIGS = [(1.0, 300, True, False), (1.0, 1200, True, True), (1.0, 4800, True, False),
           (0.0, 1200, True, False), (3.0, 1200, True, False), (1.0, 1200, False, True)]


def main(argv):
    if "--calibrate" in argv:
        calibrate(); return
    from multiprocessing import Pool
    t0 = time.time()
    seeds = [0, 1, 2]
    out = {"config": dict(K=K, T=T, DT=DT, KI=KI, TAU=TAU, EMB=EMB, CUE_SIGMA=CUE_SIGMA, EXPERT_SIGMA=EXPERT_SIGMA,
                          seeds=seeds, configs=[dict(snr=c[0], n=c[1], aug=c[2]) for c in CONFIGS],
                          note="MEASURED (toy). Not a driving result."), "sweep": {}, "extra": {}, "paired": {}}
    jobs = [(sd,) + c for c in CONFIGS for sd in seeds]
    with Pool(4) as pool:
        results = pool.map(_job, jobs, chunksize=1)
    arms_l = ["A0", "A1", "A2", "A2_pure", "QM_modelfree_Q", "A4_distilled", "A0s_true_teacher", "A1s_on_A0s", "A1c_cost_on_imagined", "A2c_cost_on_imagined", "A2c_pure", "A5_factorised_mode", "A5cal_temp_scaled", "A5m_onset_marginalised", "DIAG_A5m_true_visible_state", "REF_zblind_visible_bins", "ORACLE", "CONST_BEST"]
    for ci, c in enumerate(CONFIGS):
        snr, n, aug, extra = c
        key = f"snr{snr}_n{n}_aug{int(aug)}"
        rs = results[ci * len(seeds):(ci + 1) * len(seeds)]
        out["sweep"][key] = {a: {m: agg([r["res"][a][m] for r in rs]) for m in ("regret", "top1", "within05", "unsafe")}
                             for a in arms_l}
        out["sweep"][key]["beta_A1"] = [r["beta_A1"] for r in rs]; out["sweep"][key]["beta_A2"] = [r["beta_A2"] for r in rs]
        out["sweep"][key]["A4_temp"] = [r["A4_temp"] for r in rs]; out["sweep"][key]["beta_A1s"] = [r["beta_A1s"] for r in rs]
        out["sweep"][key]["oracle_rerank_topKi_regret"] = agg([r["oracle_rerank_topKi_regret"] for r in rs]); out["sweep"][key]["oracle_rerank_topKi_regret_A0s"] = agg([r["oracle_rerank_topKi_regret_A0s"] for r in rs]); out["sweep"][key]["a5_mode_acc"] = [r["a5_mode_acc"] for r in rs]; out["sweep"][key]["a5_T"] = [r["a5_T"] for r in rs]; out["sweep"][key]["beta_A1c"] = [r["beta_A1c"] for r in rs]; out["sweep"][key]["beta_A2c"] = [r["beta_A2c"] for r in rs]
        out["sweep"][key]["macs_per_tick_toy"] = rs[0]["macs"]
        pooled = {a: np.concatenate([r["ticks"][a] for r in rs]) for a in ("A0", "A1", "A2", "A4_distilled", "QM_modelfree_Q", "A0s_true_teacher", "A2c", "A1c", "A5", "A5cal", "A5m", "A1s")}
        out["paired"][key] = {f"A0_minus_{a}": paired_boot(pooled["A0"], pooled[a]) for a in ("A1", "A2", "A4_distilled", "QM_modelfree_Q", "A0s_true_teacher")}
        out["paired"][key]["QM_minus_A2"] = paired_boot(pooled["QM_modelfree_Q"], pooled["A2"])
        out["paired"][key]["A0s_minus_A1s"] = paired_boot(pooled["A0s_true_teacher"], pooled["A1s"])
        out["paired"][key]["A0_minus_A5"] = paired_boot(pooled["A0"], pooled["A5"])
        out["paired"][key]["A5_minus_A5m"] = paired_boot(pooled["A5"], pooled["A5m"])
        out["paired"][key]["QM_minus_A5m"] = paired_boot(pooled["QM_modelfree_Q"], pooled["A5m"])
        out["paired"][key]["A2_minus_A5m"] = paired_boot(pooled["A2"], pooled["A5m"])
        out["paired"][key]["A0s_minus_A5m"] = paired_boot(pooled["A0s_true_teacher"], pooled["A5m"])
        out["paired"][key]["A5_minus_A5cal"] = paired_boot(pooled["A5"], pooled["A5cal"])
        out["paired"][key]["A0_minus_A5cal"] = paired_boot(pooled["A0"], pooled["A5cal"])
        out["paired"][key]["QM_minus_A5"] = paired_boot(pooled["QM_modelfree_Q"], pooled["A5"])
        out["paired"][key]["A2_minus_A5"] = paired_boot(pooled["A2"], pooled["A5"])
        out["paired"][key]["QM_minus_A2c"] = paired_boot(pooled["QM_modelfree_Q"], pooled["A2c"])
        out["paired"][key]["A2_minus_A2c"] = paired_boot(pooled["A2"], pooled["A2c"])
        out["paired"][key]["A1_minus_A1c"] = paired_boot(pooled["A1"], pooled["A1c"])
        out["paired"][key]["A0s_minus_A4"] = paired_boot(pooled["A0s_true_teacher"], pooled["A4_distilled"])
        if extra:
            out["extra"][key] = agg_tree([r["extra"] for r in rs])
    out["wall_seconds"] = time.time() - t0
    with open(os.path.splitext(os.path.abspath(sys.argv[0]))[0] + "_result.json", "w") as f:
        json.dump(out, f, indent=1)
    for key, v in out["sweep"].items():
        print(key, "  ".join(f"{a}={v[a]['regret']['mean']:.2f}" for a in ("A0", "A1", "A2", "A4_distilled", "ORACLE")))
    print(f"total {out['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
