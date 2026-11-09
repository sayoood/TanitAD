#!/usr/bin/env python3
"""DESCRIPTIVE feasibility probe, E1 only (C3, 1,058 navtest tokens with the truth of all 64 P6 proposals).
Which selection families have headroom, cross-fitted by log (5 folds: fit on 4, pick on the 5th).
Nothing fitted here is carried forward or reported as a model: navtest is never used for training. Real fits come from navtrain.
Inputs at pick time: P6's six logits, the model's own proposal geometry, measured ego speed. No truth, no map, no agents."""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/p6_read")
import selc6_analyze as SA

OUT = os.path.dirname(os.path.abspath(__file__))
Z = np.load("F:/TanitAD_p6read_work/run/dump/dump_p6.npz", allow_pickle=False)
T = np.load("F:/TanitAD_p6read_work/run/score/table_p6.npz", allow_pickle=False)
dtok = [str(t) for t in Z["token"]]
assert [str(t) for t in T["token"]] == dtok
keys = ("logits", "pick", "traj", "props8", "ego", "gt_valid", "gt_pose20", "gt_t20", "human8", "frame_ctrl_m", "fingerprint")
arrs = {k: Z[k] for k in keys}
arrs.update(pdms=T["pdms"], sub=T["sub"], valid=T["valid"])
D, A = SA.align_to_c3(arrs, dtok)
zero = np.asarray(D["stratum"]) == "zero"
logs = np.asarray(D["log"]).astype(str)
N = len(zero)
w = np.where(zero, SA.W_ZERO / zero.sum(), SA.W_RANDOM / (~zero).sum())
L = A["logits"].astype(np.float64)
P = A["pdms"]
S = A["sub"]
V = A["valid"]
X8 = A["props8"].astype(np.float64)
ego = A["ego"].astype(np.float64)
assert V.all()
v0 = ego[:, 6]
ix = np.arange(N)
ulog = np.unique(logs)
perm = np.random.default_rng(20261109).permutation(len(ulog))
fold_of = {ulog[perm[i]]: i % 5 for i in range(len(ulog))}
fold = np.array([fold_of[l] for l in logs])


def agg(p, a=1.0, b=1.0, we=5.0, wt=5.0, wc=2.0):
    return p[..., 0] ** a * p[..., 1] ** b * (we * p[..., 2] + wt * p[..., 3] + wc * p[..., 4]) / (we + wt + wc)


def sig(x):
    return 1 / (1 + np.exp(-x))


p0 = sig(L)


def value(k, mask=None):
    m = np.ones(N, bool) if mask is None else mask
    return float((w[m] * P[ix, k][m]).sum() / w[m].sum())


RES, OOF = {}, {}


def record(name, k, t0):
    OOF[name] = k
    RES[name] = dict(pdms=100 * value(k), secs=round(time.time() - t0, 1))
    print(name, round(RES[name]["pdms"], 2), flush=True)


t = time.time()
base_pick = agg(p0).argmax(1)
record("00_P6_rule", base_pick, t)
record("01_oracle", P.argmax(1), t)


def geom():
    x, y, h = X8[..., 0], X8[..., 1], X8[..., 2]
    dx = np.diff(np.concatenate([np.zeros_like(x[..., :1]), x], -1), axis=-1)
    dy = np.diff(np.concatenate([np.zeros_like(y[..., :1]), y], -1), axis=-1)
    sp = np.hypot(dx, dy) / 0.5
    v4, v2 = sp[..., -1], sp[..., 3]
    r_v = np.broadcast_to(v0[:, None], v4.shape)
    return np.stack([x[..., -1], y[..., -1], np.abs(y).max(-1), h[..., -1], np.abs(h[..., -1]), v4, v2, sp.mean(-1),
                     (sp[..., -1] - sp[..., 0]) / 3.5, sp.sum(-1) * 0.5, np.abs(np.diff(sp, n=2, axis=-1)).max(-1),
                     v4 / np.maximum(r_v, 1), r_v], -1)


G = geom()
GNAME = ["x_end", "y_end", "abs_y_max", "h_end", "abs_h_end", "v4", "v2", "v_mean", "acc", "dist", "jerk", "v4_over_v0", "v0"]


def rank01(a):
    return a.argsort(1).argsort(1) / 63.0


def feats():
    s0 = agg(p0)
    st = np.concatenate([p0.mean(1), p0.std(1), p0.max(1), p0.min(1)], 1)
    rep = np.broadcast_to(st[:, None, :], (N, 64, st.shape[1]))
    dv = G[..., 5] - np.median(G[..., 5], 1, keepdims=True)
    dy = G[..., 1] - np.median(G[..., 1], 1, keepdims=True)
    ranks = np.stack([rank01(s0)] + [rank01(p0[..., i]) for i in (0, 1, 2, 3)] + [rank01(G[..., 9])], -1)
    return np.concatenate([L[..., :6], p0[..., :6], s0[..., None], ranks, G, dv[..., None], dy[..., None], rep], -1)


F = feats()
print("features", F.shape, flush=True)


def crossfit(fn):
    k = np.zeros(N, int)
    for f in range(5):
        tr, te = fold != f, fold == f
        k[te] = fn(tr, te)
    return k


def rule_shape(tr, te):
    r = np.random.default_rng(1)
    opts = [0.25, 0.5, 1, 2, 4]
    wts = [1, 2, 5, 8, 12]
    cands = [(1.0, 1.0, 5.0, 5.0, 2.0)] + [(r.choice(opts), r.choice(opts), r.choice(wts), r.choice(wts), r.choice([0, 1, 2])) for _ in range(400)]
    best = max(cands, key=lambda c: value(agg(p0, *c).argmax(1), tr))
    return agg(p0, *best)[te].argmax(1)


t = time.time()
record("10_rule_shape_exponents_weights", crossfit(rule_shape), t)


def veto_pick(c, idx):
    tn, td, tt = c
    pp = p0[idx]
    s = (5 * pp[..., 2] + 5 * pp[..., 3] + 2 * pp[..., 4]) / 12
    ok = (pp[..., 0] >= tn) & (pp[..., 1] >= td) & (pp[..., 3] >= tt)
    return (s - 10.0 * (~ok) + 1e-3 * agg(pp)).argmax(1)


def veto(tr, te):
    grid = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    tri = np.where(tr)[0]
    best, bv = None, -1.0
    for c in [(a, b, d) for a in grid for b in grid for d in grid]:
        kk = np.zeros(N, int)
        kk[tri] = veto_pick(c, tri)
        v = value(kk, tr)
        if v > bv:
            bv, best = v, c
    return veto_pick(best, np.where(te)[0])


t = time.time()
record("11_rule_hard_veto", crossfit(veto), t)


def speed_term(tr, te):
    sc = np.minimum(G[..., 11], 2.5) - 1
    betas = [-0.5, -0.2, 0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2]
    b = max(betas, key=lambda x: value((agg(p0) * np.exp(x * sc)).argmax(1), tr))
    return (agg(p0) * np.exp(b * sc))[te].argmax(1)


t = time.time()
record("12_rule_speed_term", crossfit(speed_term), t)

from scipy.optimize import minimize


def platt_fit(y, x, wt):
    def nll(th):
        z = th[0] * x + th[1]
        return float((wt * (np.logaddexp(0, z) - y * z)).sum() / wt.sum())
    return minimize(nll, [1.0, 0.0], method="L-BFGS-B").x


def platt(tr, te):
    p = p0.copy()
    wt = np.broadcast_to(w[:, None], P.shape)
    for i in range(5):
        a, b = platt_fit(S[tr, :, i].ravel(), L[tr, :, i].ravel(), wt[tr].ravel())
        p[..., i] = sig(a * L[..., i] + b)
    return agg(p)[te].argmax(1)


t = time.time()
record("20_calibration_platt_per_head", crossfit(platt), t)

from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR


def fl(a):
    return a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a.ravel()


KW = dict(learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=64, random_state=0)


def gbdt_pdms(tr, te):
    wt = np.broadcast_to(w[:, None], P.shape)
    g = HGR(max_iter=200, l2_regularization=1.0, **KW).fit(fl(F[tr]), fl(P[tr]), sample_weight=fl(wt[tr]))
    return g.predict(fl(F[te])).reshape(te.sum(), 64).argmax(1)


t = time.time()
record("30_ml_gbdt_true_pdms", crossfit(gbdt_pdms), t)


def gbdt_heads(tr, te):
    wt = np.broadcast_to(w[:, None], P.shape)
    q = p0[te].copy()
    for i in (0, 1, 3):
        y = (S[tr, :, i] > 0.5).astype(int)
        q[..., i] = HGC(max_iter=120, **KW).fit(fl(F[tr]), fl(y), sample_weight=fl(wt[tr])).predict_proba(fl(F[te]))[:, 1].reshape(te.sum(), 64)
    q[..., 2] = np.clip(HGR(max_iter=120, **KW).fit(fl(F[tr]), fl(S[tr, :, 2]), sample_weight=fl(wt[tr])).predict(fl(F[te])).reshape(te.sum(), 64), 0, 1)
    return agg(q).argmax(1)


t = time.time()
record("31_ml_gbdt_per_head_stack", crossfit(gbdt_heads), t)

import torch

torch.set_num_threads(3)
mu, sd = F.mean((0, 1)), F.std((0, 1)) + 1e-6
Fz = np.clip((F - mu) / sd, -6, 6)
LOGRULE = np.log(agg(p0) + 1e-6)


def policy_gradient(tr, te, epochs=60, tau=0.05, seed=0):
    """Full-information policy gradient: the exact expected true PDMS under a softmax over the 64 arms, residual on P6's rule."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(F.shape[-1], 128), torch.nn.GELU(), torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Linear(128, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    xtr = torch.tensor(Fz[tr], dtype=torch.float32)
    ytr = torch.tensor(P[tr], dtype=torch.float32)
    wtr = torch.tensor(w[tr] / w[tr].mean(), dtype=torch.float32)
    btr = torch.tensor(LOGRULE[tr] / tau, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for b in torch.randperm(xtr.shape[0], generator=g).split(32):
            pi = (net(xtr[b]).squeeze(-1) + btr[b]).softmax(1)
            loss = -((pi * ytr[b]).sum(1) * wtr[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        s = net(torch.tensor(Fz[te], dtype=torch.float32)).squeeze(-1) + torch.tensor(LOGRULE[te] / tau, dtype=torch.float32)
    return s.argmax(1).numpy()


t = time.time()
record("40_rl_full_information_policy_gradient", crossfit(policy_gradient), t)

names = sorted(OOF)
ul = {l: np.where(logs == l)[0] for l in ulog}
draws = np.random.default_rng(20260927).integers(0, len(ulog), (2000, len(ulog)))
base = P[ix, OOF["00_P6_rule"]]
gap = float(100 * (w * (P.max(1) - base)).sum())
for n in names:
    d = P[ix, OOF[n]] - base
    bs = []
    for dr in draws:
        idx = np.concatenate([ul[ulog[j]] for j in dr])
        bs.append(100 * (w[idx] * d[idx]).sum() / w[idx].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    dm = float(100 * (w * d).sum())
    RES[n].update(delta=dm, lo=float(lo), hi=float(hi), share_of_gap_closed=dm / gap)
json.dump(dict(note="DESCRIPTIVE, E1 only, cross-fitted by log, nothing carried forward; unadjusted for the number of methods", n_tokens=N, n_logs=len(ulog), geometry_features=GNAME, results=RES), open(OUT + "/probe_families.json", "w"), indent=1)
print("\nname  PDMS  delta [95% CI]  share of the gap closed")
for n in names:
    r = RES[n]
    print(n, round(r["pdms"], 2), round(r["delta"], 2), [round(r["lo"], 2), round(r["hi"], 2)], round(r["share_of_gap_closed"], 3))
print("ZZPROBE_END")
