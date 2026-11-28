#!/usr/bin/env python3
"""selcore.py -- shared core of the P6 selector experiment (PREREG_P6_SELECTORS, frozen 2026-11-09). CPU only; no E:/F: dependency, so it runs on the pod.

Holds: constants, the fit_pack loader, the log split, the features (probe_families.py), every candidate variant (fit on FIT-A rows only, apply on any rows),
the paired log-cluster bootstrap (the recipe of sel_common.log_boot_idx / selc6_analyze.Boot, re-stated here so the pod needs neither), the persistent ledger K
and the Bonferroni percentiles.

A selector is a pair (fit, apply). `fit(A)` receives ONLY a FitData built from FIT-A rows. `apply(params, ctx)` sees only inference inputs:
P6's six logits [n,64,6], the model's own proposal geometry [n,64,8,3] (x, y, heading at 0.5 s steps) and ego speed v0 [n]. No truth, no map, no agents.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("SEL_THREADS", "4"))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

# ------------------------------------------------------------------------------------------------ frozen constants
SPLIT_SEED = 20261109
FRAC_A = 0.70
B_DRAWS, BOOT_SEED = 10000, 20260927
R_DRAWS, R_SEED = 400, 1
R_OPTS, R_WTS, R_COMF = [0.25, 0.5, 1, 2, 4], [1, 2, 5, 8, 12], [0, 1, 2]
P6_RULE = (1.0, 1.0, 5.0, 5.0, 2.0)
V1_W = (5.0, 5.0, 2.0)
KW = dict(learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=64, random_state=0)   # probe_families.py KW
MLP_HIDDEN, MLP_EPOCHS, PG_TAU, PG_EPOCHS = 128, 60, 0.05, 60
OVERSPEED, SLOW = 1.15, 0.85
M = 64
PDM_NAMES = ["nc", "dac", "ep", "ttc", "c", "ddc", "pdms"]
FAMILY_OF = {"R": "R rule shape", "C": "C calibration", "S": "S stackers", "P": "P policy gradient"}
VARIANTS = ["R1", "R2", "C_platt", "C_isotonic", "C_platt_binned", "S_gbdt_pdms", "S_gbdt_heads", "S_mlp_heads", "P_pg"]


def family_of(vid):
    return vid.split("_")[0][0]


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def atomic_write(path, data, mode="wb"):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, mode, **({} if "b" in mode else {"encoding": "utf-8"})) as f:
        f.write(data)
    os.replace(tmp, path)


def wait_for_ram(min_gb=6.0, poll_s=30, max_wait_s=None, log=print):
    """Dev box: block while free RAM < min_gb. Skipped on non-Windows (pod). Returns seconds waited."""
    if os.name != "nt":
        return 0.0
    import psutil
    t0 = time.time()
    while psutil.virtual_memory().available / 2 ** 30 < min_gb:
        if max_wait_s is not None and time.time() - t0 > max_wait_s:
            raise RuntimeError(f"free RAM stayed under {min_gb} GB for {max_wait_s} s")
        log(f"waiting: free RAM {psutil.virtual_memory().available / 2 ** 30:.1f} GB < {min_gb} GB")
        time.sleep(poll_s)
    return time.time() - t0


# ------------------------------------------------------------------------------------------------ pack
class Pack:
    """fit_pack.npz of p6_labels_tool.py cmd_pack, restricted to sets that carry finite PDM labels."""
    REQUIRED = ("log", "token", "traj_xy", "yaw", "logits", "pdm", "has_pdm")

    def __init__(self, path):
        Z = np.load(path, allow_pickle=False)
        miss = [k for k in self.REQUIRED if k not in Z.files]
        if miss:
            raise ValueError(f"pack lacks {miss}")
        self.path, self.sha256 = str(path), sha256_file(path)
        has = Z["has_pdm"].astype(bool)
        pdm = Z["pdm"].astype(np.float64)
        n0 = len(has)
        fin = np.isfinite(pdm).reshape(n0, -1).all(1)
        keep = has & fin
        self.n_total, self.n_no_label, self.n_nonfinite = int(n0), int((~has).sum()), int((has & ~fin).sum())
        if keep.sum() < 0.98 * has.sum():
            raise ValueError(f"{self.n_nonfinite} of {int(has.sum())} labelled sets carry non-finite labels: refusing")
        sel = np.where(keep)[0]
        self.log = np.array([str(x) for x in Z["log"][sel]])
        self.token = np.array([str(x) for x in Z["token"][sel]])
        xy, yaw = Z["traj_xy"][sel], Z["yaw"][sel]
        if xy.ndim != 4 or xy.shape[1] != M or xy.shape[2] != 8 or xy.shape[3] != 2:
            raise ValueError(f"traj_xy shape {xy.shape}: need [n,64,8,2] (8 poses at 0.5 s)")
        self.X8 = np.concatenate([xy.astype(np.float32), yaw.astype(np.float32)[..., None]], -1)
        self.L = Z["logits"][sel].astype(np.float32)
        if self.L.shape[1:] != (M, 6):
            raise ValueError(f"logits shape {self.L.shape}: need [n,64,6]")
        self.pdm = pdm[sel]
        if self.pdm.shape[1:] != (M, 7):
            raise ValueError(f"pdm shape {self.pdm.shape}: need [n,64,7] ({PDM_NAMES})")
        self.P, self.S = self.pdm[..., 6], self.pdm[..., :6]
        self.v0 = Z["v0"][sel].astype(np.float64) if "v0" in Z.files else None
        self.n = len(sel)


# ------------------------------------------------------------------------------------------------ split
def make_split(log, seed=SPLIT_SEED, frac_a=FRAC_A):
    """by whole log: permute the sorted unique logs with default_rng(seed), fill FIT-A until it holds >= frac_a of the sets."""
    ul, cnt = np.unique(log, return_counts=True)
    if len(ul) < 2:
        raise ValueError("a log split needs at least 2 logs")
    perm = np.random.default_rng(seed).permutation(len(ul))
    tot, cum, a = cnt.sum(), 0, []
    for j in perm:
        if cum >= frac_a * tot and a:
            break
        a.append(j)
        cum += cnt[j]
    A = sorted(ul[a].tolist())
    Bl = sorted(set(ul.tolist()) - set(A))
    if not Bl:
        raise ValueError("FIT-B would be empty")
    return {"seed": seed, "frac_a_target": frac_a, "n_sets": int(tot), "n_logs": int(len(ul)), "logs_A": A, "logs_B": Bl,
            "n_sets_A": int(sum(cnt[np.searchsorted(ul, A)])), "n_sets_B": int(sum(cnt[np.searchsorted(ul, Bl)]))}


def split_rows(log, split):
    inA = np.isin(log, split["logs_A"])
    inB = np.isin(log, split["logs_B"])
    if (inA & inB).any() or not (inA | inB).all():
        raise ValueError("split does not partition the sets")
    return np.where(inA)[0], np.where(inB)[0]


# ------------------------------------------------------------------------------------------------ inference context, features
def plan_v4(x8):
    p = np.asarray(x8, np.float64)[..., :2]
    return np.linalg.norm(p[..., 7, :] - p[..., 6, :], axis=-1) / 0.5


def v0_proxy(x8):
    """median over the 64 proposals of the first-step chord speed (0.5 s from the origin): used only when the pack carries no measured ego speed"""
    p = np.asarray(x8, np.float64)[..., 0, :2]
    return np.median(np.linalg.norm(p, axis=-1) / 0.5, axis=1)


class Ctx:
    """inference-time inputs only"""

    def __init__(self, L, X8, v0=None):
        self.L = np.asarray(L, np.float32)
        self.X8 = np.asarray(X8, np.float32)
        if v0 is None:
            self.v0, self.v0_source = v0_proxy(self.X8), "proxy_median_first_chord"
        else:
            self.v0, self.v0_source = np.asarray(v0, np.float64), "measured"
        self._F = None

    def take(self, idx):
        c = Ctx.__new__(Ctx)
        c.L, c.X8, c.v0, c.v0_source, c._F = self.L[idx], self.X8[idx], self.v0[idx], self.v0_source, None
        return c


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def agg64(p, a=1.0, b=1.0, we=5.0, wt=5.0, wc=2.0):
    return p[..., 0] ** a * p[..., 1] ** b * (we * p[..., 2] + wt * p[..., 3] + wc * p[..., 4]) / (we + wt + wc)


def p6_scores(L):
    """P6's benchmark rule, torch float32, the same ops as sel_common.agg_torch(lane=False); first index on ties through np.argmax"""
    import torch
    p = torch.as_tensor(np.asarray(L, np.float32)).sigmoid()
    w1 = torch.tensor(V1_W, dtype=p.dtype)
    return (p[..., 0] * p[..., 1] * (p[..., 2] * w1[0] + p[..., 3] * w1[1] + p[..., 4] * w1[2]) / w1.sum()).numpy()


def p6_pick(L):
    return np.argmax(p6_scores(L), axis=-1)


def rank01(a):
    """tie-correct average ranks in [0,1] along the proposal axis (the probe's argsort-of-argsort breaks ties arbitrarily)"""
    from scipy.stats import rankdata
    return (rankdata(a, axis=1, method="average") - 1.0) / 63.0


def geom(X8, v0):
    x, y, h = X8[..., 0].astype(np.float64), X8[..., 1].astype(np.float64), X8[..., 2].astype(np.float64)
    dx = np.diff(np.concatenate([np.zeros_like(x[..., :1]), x], -1), axis=-1)
    dy = np.diff(np.concatenate([np.zeros_like(y[..., :1]), y], -1), axis=-1)
    sp = np.hypot(dx, dy) / 0.5
    v4, v2 = sp[..., -1], sp[..., 3]
    r_v = np.broadcast_to(np.asarray(v0, np.float64)[:, None], v4.shape)
    return np.stack([x[..., -1], y[..., -1], np.abs(y).max(-1), h[..., -1], np.abs(h[..., -1]), v4, v2, sp.mean(-1),
                     (sp[..., -1] - sp[..., 0]) / 3.5, sp.sum(-1) * 0.5, np.abs(np.diff(sp, n=2, axis=-1)).max(-1),
                     v4 / np.maximum(r_v, 1), r_v], -1)


def features(ctx):
    if ctx._F is not None:
        return ctx._F
    L = ctx.L.astype(np.float64)
    p0 = sig(L)
    G = geom(ctx.X8, ctx.v0)
    s0 = agg64(p0)
    st = np.concatenate([p0.mean(1), p0.std(1), p0.max(1), p0.min(1)], 1)
    rep = np.broadcast_to(st[:, None, :], (L.shape[0], M, st.shape[1]))
    dv = G[..., 5] - np.median(G[..., 5], 1, keepdims=True)
    dy = G[..., 1] - np.median(G[..., 1], 1, keepdims=True)
    ranks = np.stack([rank01(s0)] + [rank01(p0[..., i]) for i in (0, 1, 2, 3)] + [rank01(G[..., 9])], -1)
    ctx._F = np.concatenate([L[..., :6], p0[..., :6], s0[..., None], ranks, G, dv[..., None], dy[..., None], rep], -1)
    return ctx._F


class FitData:
    """what a fit may read: FIT-A rows only (context, truth, logs)"""

    def __init__(self, ctx, P, S, log):
        self.ctx, self.P, self.S, self.log = ctx, np.asarray(P, np.float64), np.asarray(S, np.float64), np.asarray(log)

    @property
    def n(self):
        return len(self.P)


def pick_value(P, k):
    return float(P[np.arange(len(P)), k].mean())


# ------------------------------------------------------------------------------------------------ the bootstrap and the ledger
def log_boot(log, B=B_DRAWS, seed=BOOT_SEED):
    """sel_common.log_boot_idx recipe: np.unique logs, default_rng(seed).choice(n_logs, (B, n_logs))"""
    ul, li = np.unique(log, return_inverse=True)
    draws = np.random.default_rng(seed).choice(len(ul), size=(B, len(ul)), replace=True)
    return ul, li, draws


def boot_stat(vals, li, nl, draws):
    v = np.asarray(vals, np.float64)
    s = np.bincount(li, weights=v, minlength=nl)
    c = np.bincount(li, minlength=nl).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = s[draws].sum(1) / c[draws].sum(1)
    return float(s.sum() / c.sum()), r[np.isfinite(r)]


def pct_pair(K=1, alpha=5.0):
    """Bonferroni over K reads: the two-sided alpha/K interval as (lo_percentile, hi_percentile); K=1 is the plain 2.5/97.5"""
    K = max(1, int(K))
    return alpha / 2.0 / K, 100.0 - alpha / 2.0 / K


def interval(r, K=1):
    lo, hi = pct_pair(K)
    return float(np.percentile(r, lo)), float(np.percentile(r, hi))


class Ledger:
    """persistent count K of every navtest read of a selector across this card. A read is keyed (nominee, params sha256, set); the same read repeated is not a new read."""

    def __init__(self, path):
        self.path = str(path)
        self.d = {"card": "PREREG_P6_SELECTORS", "reads": []}
        if os.path.exists(self.path):
            self.d = json.load(open(self.path, encoding="utf-8"))

    @property
    def K(self):
        return len(self.d["reads"])

    def has(self, key):
        return any(r["key"] == key for r in self.d["reads"])

    def add(self, key, **info):
        if self.has(key):
            return False
        self.d["reads"].append(dict(key=key, n=self.K + 1, time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **info))
        atomic_write(self.path, json.dumps(self.d, indent=1).encode("utf-8"))
        return True


# ------------------------------------------------------------------------------------------------ variants
def mean_pdms_x100(P, k):
    return 100.0 * pick_value(P, k)


def r_candidates():
    r = np.random.default_rng(R_SEED)
    out = [P6_RULE]
    for _ in range(R_DRAWS):
        out.append((float(r.choice(R_OPTS)), float(r.choice(R_OPTS)), float(r.choice(R_WTS)), float(r.choice(R_WTS)), float(r.choice(R_COMF))))
    return out


def rule_scores(L, cand):
    cand = tuple(float(c) for c in cand)
    if cand == P6_RULE:
        return p6_scores(L)
    return agg64(sig(np.asarray(L, np.float64)), *cand)


def fit_R(A):
    """all 401 candidates scored on FIT-A (the objective: mean true PDMS at the pick); R1, R2 = the two best on FIT-A. Reads A only."""
    cands = r_candidates()
    vals = [pick_value(A.P, np.argmax(rule_scores(A.ctx.L, c), axis=1)) for c in cands]
    order = np.argsort(-np.asarray(vals), kind="stable")
    table = [dict(cand=list(cands[i]), fit_a_mean_pdms=float(vals[i]), is_p6_rule=bool(tuple(cands[i]) == P6_RULE)) for i in range(len(cands))]
    return {"R1": dict(kind="rule", cand=list(cands[order[0]]), fit_a_rank=1), "R2": dict(kind="rule", cand=list(cands[order[1]]), fit_a_rank=2)}, table


def apply_rule(params, ctx):
    return rule_scores(ctx.L, params["cand"])


def platt_fit(y, x):
    from scipy.optimize import minimize

    def nll(th):
        z = th[0] * x + th[1]
        return float((np.logaddexp(0, z) - y * z).mean())
    return [float(v) for v in minimize(nll, [1.0, 0.0], method="L-BFGS-B").x]


HEADS5 = range(5)


def fit_platt(A):
    L, S = A.ctx.L.astype(np.float64), A.S
    return dict(kind="platt", ab=[platt_fit(S[..., i].ravel(), L[..., i].ravel()) for i in HEADS5])


def apply_platt(params, ctx):
    L = ctx.L.astype(np.float64)
    p = sig(L)
    for i in HEADS5:
        a, b = params["ab"][i]
        p[..., i] = sig(a * L[..., i] + b)
    return agg64(p)


def fit_isotonic(A):
    from sklearn.isotonic import IsotonicRegression
    L, S = A.ctx.L.astype(np.float64), A.S
    out = []
    for i in HEADS5:
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True).fit(L[..., i].ravel(), S[..., i].ravel())
        out.append((ir.X_thresholds_.astype(np.float64), ir.y_thresholds_.astype(np.float64)))
    return dict(kind="isotonic", thr=out)


def apply_isotonic(params, ctx):
    L = ctx.L.astype(np.float64)
    p = sig(L)
    for i in HEADS5:
        xt, yt = params["thr"][i]
        p[..., i] = np.interp(L[..., i], xt, yt)
    return agg64(p)


def speed_bin(ctx):
    ratio = plan_v4(ctx.X8) / np.maximum(ctx.v0, 1e-6)[:, None]
    return np.where(ratio < SLOW, 0, np.where(ratio > OVERSPEED, 2, 1)), ratio


def fit_platt_binned(A, min_pts=200):
    L, S = A.ctx.L.astype(np.float64), A.S
    bins, _ = speed_bin(A.ctx)
    glob = [platt_fit(S[..., i].ravel(), L[..., i].ravel()) for i in HEADS5]
    ab, counts = [], np.array([(bins == b).sum() for b in range(3)]).tolist()
    for b in range(3):
        row = []
        for i in HEADS5:
            m = bins == b
            row.append(platt_fit(S[..., i][m], L[..., i][m]) if m.sum() >= min_pts else glob[i])
        ab.append(row)
    return dict(kind="platt_binned", ab=ab, bin_counts=counts, edges=[SLOW, OVERSPEED])


def apply_platt_binned(params, ctx):
    L = ctx.L.astype(np.float64)
    bins, _ = speed_bin(ctx)
    p = sig(L)
    for i in HEADS5:
        a = np.array([params["ab"][b][i][0] for b in range(3)])[bins]
        c = np.array([params["ab"][b][i][1] for b in range(3)])[bins]
        p[..., i] = sig(a * L[..., i] + c)
    return agg64(p)


def _fl(a):
    return a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a.ravel()


def fit_gbdt_pdms(A, cfg):
    from sklearn.ensemble import HistGradientBoostingRegressor as HGR
    g = HGR(max_iter=cfg.get("gbdt_pdms_iter", 200), l2_regularization=1.0, **KW).fit(_fl(features(A.ctx)), _fl(A.P))
    return dict(kind="gbdt_pdms", model=g)


def apply_gbdt_pdms(params, ctx):
    F = features(ctx)
    return params["model"].predict(_fl(F)).reshape(F.shape[0], M)


class _Const:
    def __init__(self, c):
        self.c = float(c)

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.c), np.full(len(X), self.c)])


def fit_gbdt_heads(A, cfg):
    from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR
    F, it = _fl(features(A.ctx)), cfg.get("gbdt_heads_iter", 120)
    models = {}
    for i in (0, 1, 3):
        y = (A.S[..., i] > 0.5).astype(int).ravel()
        models[i] = HGC(max_iter=it, **KW).fit(F, y) if 0 < y.sum() < len(y) else _Const(y.mean())
    models[2] = HGR(max_iter=it, **KW).fit(F, A.S[..., 2].ravel())
    return dict(kind="gbdt_heads", models=models)


def apply_gbdt_heads(params, ctx):
    F = features(ctx)
    n, X = F.shape[0], _fl(F)
    q = sig(ctx.L.astype(np.float64))
    for i in (0, 1, 3):
        q[..., i] = params["models"][i].predict_proba(X)[:, 1].reshape(n, M)
    q[..., 2] = np.clip(params["models"][2].predict(X).reshape(n, M), 0, 1)
    return agg64(q)


def _norm(F):
    mu, sd = F.mean((0, 1)), F.std((0, 1)) + 1e-6
    return mu, sd


def _net(d_in):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(d_in, MLP_HIDDEN), torch.nn.GELU(), torch.nn.Linear(MLP_HIDDEN, MLP_HIDDEN), torch.nn.GELU(), torch.nn.Linear(MLP_HIDDEN, 1))


def _state(net):
    return {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}


def _load_net(d_in, st):
    import torch
    net = _net(d_in)
    net.load_state_dict({k: torch.as_tensor(v) for k, v in st.items()})
    net.eval()
    return net


def fit_mlp_heads(A, cfg):
    """one MLP per head (NC, DAC, EP, TTC), two layers of 128, 60 epochs, AdamW 1e-3 / 1e-3, batches of 32 sets; BCE on the sub-score (binary for NC, DAC, TTC)"""
    import torch
    torch.set_num_threads(int(os.environ.get("SEL_THREADS", "4")))
    F = features(A.ctx)
    mu, sd = _norm(F)
    x = torch.tensor(np.clip((F - mu) / sd, -6, 6), dtype=torch.float32)
    nets, epochs = {}, cfg.get("mlp_epochs", MLP_EPOCHS)
    for i in (0, 1, 2, 3):
        torch.manual_seed(0)
        net = _net(F.shape[-1])
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        y = torch.tensor(A.S[..., i], dtype=torch.float32)
        y = (y > 0.5).float() if i != 2 else y.clamp(0, 1)
        g = torch.Generator().manual_seed(0)
        for _ in range(epochs):
            for b in torch.randperm(x.shape[0], generator=g).split(32):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x[b]).squeeze(-1), y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        nets[i] = _state(net)
    return dict(kind="mlp_heads", nets=nets, mu=mu, sd=sd, d_in=F.shape[-1])


def apply_mlp_heads(params, ctx):
    import torch
    F = features(ctx)
    x = torch.tensor(np.clip((F - params["mu"]) / params["sd"], -6, 6), dtype=torch.float32)
    q = sig(ctx.L.astype(np.float64))
    with torch.no_grad():
        for i, st in params["nets"].items():
            q[..., i] = _load_net(params["d_in"], st)(x).squeeze(-1).sigmoid().double().numpy()
    return agg64(q)


def fit_pg(A, cfg):
    """full-information policy gradient: exact expected true PDMS under a softmax over the 64 proposals, residual on P6's rule, temperature 0.05 (probe_families.policy_gradient)"""
    import torch
    torch.set_num_threads(int(os.environ.get("SEL_THREADS", "4")))
    F = features(A.ctx)
    mu, sd = _norm(F)
    Fz = np.clip((F - mu) / sd, -6, 6)
    logrule = np.log(agg64(sig(A.ctx.L.astype(np.float64))) + 1e-6)
    torch.manual_seed(0)
    net = _net(F.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    xtr, ytr = torch.tensor(Fz, dtype=torch.float32), torch.tensor(A.P, dtype=torch.float32)
    btr = torch.tensor(logrule / PG_TAU, dtype=torch.float32)
    g = torch.Generator().manual_seed(0)
    for _ in range(cfg.get("pg_epochs", PG_EPOCHS)):
        for b in torch.randperm(xtr.shape[0], generator=g).split(32):
            pi = (net(xtr[b]).squeeze(-1) + btr[b]).softmax(1)
            loss = -((pi * ytr[b]).sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return dict(kind="pg", net=_state(net), mu=mu, sd=sd, d_in=F.shape[-1])


def apply_pg(params, ctx):
    import torch
    F = features(ctx)
    x = torch.tensor(np.clip((F - params["mu"]) / params["sd"], -6, 6), dtype=torch.float32)
    logrule = np.log(agg64(sig(ctx.L.astype(np.float64))) + 1e-6)
    with torch.no_grad():
        s = _load_net(params["d_in"], params["net"])(x).squeeze(-1).double().numpy() + logrule / PG_TAU
    return s


APPLY = dict(rule=apply_rule, platt=apply_platt, isotonic=apply_isotonic, platt_binned=apply_platt_binned, gbdt_pdms=apply_gbdt_pdms,
             gbdt_heads=apply_gbdt_heads, mlp_heads=apply_mlp_heads, pg=apply_pg)
FIT = {"C_platt": lambda A, c: fit_platt(A), "C_isotonic": lambda A, c: fit_isotonic(A), "C_platt_binned": lambda A, c: fit_platt_binned(A),
       "S_gbdt_pdms": fit_gbdt_pdms, "S_gbdt_heads": fit_gbdt_heads, "S_mlp_heads": fit_mlp_heads, "P_pg": fit_pg}


def fit_variants(A, ids, cfg=None):
    """-> {vid: params}. R1 and R2 come from one candidate table (extra key '_R_table' carried beside them)."""
    cfg, out = cfg or {}, {}
    if any(v in ids for v in ("R1", "R2")):
        rp, table = fit_R(A)
        for v in ("R1", "R2"):
            if v in ids:
                out[v] = rp[v]
        out["_R_table"] = table
    for v in ids:
        if v in FIT:
            out[v] = FIT[v](A, cfg)
    for v, p in out.items():
        if not v.startswith("_"):
            p["v0_source"] = A.ctx.v0_source
    return out


V0_KINDS = ("platt_binned", "gbdt_pdms", "gbdt_heads", "mlp_heads", "pg")


def apply_params(params, ctx, v0_source_check=True):
    if v0_source_check and params["kind"] in V0_KINDS and params.get("v0_source") != ctx.v0_source:
        raise ValueError(f"v0 source mismatch: fitted with {params['v0_source']}, applied with {ctx.v0_source}")
    return APPLY[params["kind"]](params, ctx)


def pick_of(scores):
    return np.argmax(scores, axis=1)       # first index on ties
