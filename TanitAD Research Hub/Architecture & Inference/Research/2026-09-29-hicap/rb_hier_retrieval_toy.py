#!/usr/bin/env python3
"""RB toy: hierarchical (tree / factorised) retrieval over a >= 65,536-entry product action vocabulary.

Stream R-B of the HiCAP design (2026-09-29).  numpy only, CPU, ~2 min (129-145 s measured; not "seconds").
Run:  python3 rb_hier_retrieval_toy.py   (deterministic given numpy Generator/PCG64; only `timing_*` varies).
Experiments in the JSON: E1 tree/factorised vs flat, E1c noise sweep, E2 masks (+ analytic box check), E3 conformal mask
calibration, E4 prior on/off (+ bounded critic, sibling-only offsets), E5 calibrated adaptive-margin beam, E6 re-rank by K,
E7 node-critic type (marginal / max / mean pool), E8 prior estimation from sparse counts, E9 PMI bias vs vocabulary size.  MEASURED (toy): every number
below is a property of THIS SYNTHETIC WORLD, not of driving data.  What it can and cannot show is written
in RB_candidate_space.md section 5.  Result -> rb_hier_retrieval_toy_result.json (same folder).

World (all synthetic, seeded):
  * vocabulary = product of a lateral axis (curvature kappa, `nk` bins) and a longitudinal axis (speed ratio
    rho = v_end/v0 - 1 over 2 s, `nr` bins) => N = nk*nr leaves, laid out in TREE (depth-first) order
        strategic S (=8, coarse kappa bin)  ->  tactical cell C (= LAT sub-bin x LON coarse bin)  ->  Lf leaves
    config A = 8 x 32 x 256 = 65,536 ; config B (v7-shaped, 8 LAT x 8 LON tactical cells) = 8 x 64 x 256 = 131,072.
  * state s = (v0, up to 3 expert "modes", true per-state limits).  p(a|s) = sum_k w_k bump_k(a) * 1[a feasible
    under the state's TRUE limits] / Z  (exact, computed over all N leaves).  The prior p(a) is the TRAIN-sample
    marginal of p(a|s), so it is cruise-dominated because the state distribution is.
  * critic at tree level l:  f_l(s,n) = log P_l(n|s) - log P_l(n) + eps   (the InfoNCE optimum, PMI, with
    level-wide negatives) + iid Gaussian estimation noise eps ~ N(0, sigma^2).  Prior correction adds beta*log P_l(n).
  * MAC accounting = (#node scores actually evaluated) x D, D = 512.  The scores are synthetic; no embedding is
    multiplied.  A separate CPU wall-clock microbenchmark is reported under `timing_cpu_numpy_nondeterministic`.
"""
import json
import math
import os
import time

import numpy as np

SEED = 20260929
D = 512
T_H = 2.0
NEG = -1e30
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rb_hier_retrieval_toy_result.json")

CFGS = {
    "A_8x32x256": dict(S=8, LS=4, LON=8, FK=16, FV=16,
                       edges=[-0.12, -0.06, -0.025, -0.006, 0.010, 0.030, 0.060, 0.090, 0.120]),
    "B_v7shape_8x64x256": dict(S=8, LS=8, LON=8, FK=16, FV=16,
                               edges=[-0.12, -0.06, -0.025, -0.006, 0.026, 0.050, 0.075, 0.100, 0.120]),
}
# nominal mask tiers (acc, dec, lat) m/s^2 -- the same three tiers the sibling stream measured on real REF-C anchors
TIERS = {"T0_nuplan_comfort": (2.40, 4.05, 4.89), "T1_tight": (3.0, 6.0, 4.0), "T2_comfort_plus": (4.0, 8.0, 6.0), "T3_physical": (6.0, 10.0, 9.0)}


# ----------------------------------------------------------------------------------------------- world
class World:
    def __init__(self, cfg):
        self.S, self.LS, self.LON, self.FK, self.FV = (cfg[k] for k in ("S", "LS", "LON", "FK", "FV"))
        S, LS, LON, FK, FV = self.S, self.LS, self.LON, self.FK, self.FV
        E = np.asarray(cfg["edges"])
        per = LS * FK
        self.nk, self.nr = S * per, LON * FV
        self.C, self.Lf = LS * LON, FK * FV
        self.N = S * self.C * self.Lf
        kap = np.concatenate([E[i] + (E[i + 1] - E[i]) * (np.arange(per) + 0.5) / per for i in range(S)])
        rho = -1.0 + 1.5 * (np.arange(self.nr) + 0.5) / self.nr
        K2, R2 = np.meshgrid(kap, rho, indexing="ij")
        self.kap_g = self.dfs(K2).reshape(-1).astype(np.float32)
        self.rho_g = self.dfs(R2).reshape(-1).astype(np.float32)
        gid2d = self.dfs(np.arange(self.nk * self.nr, dtype=np.int64).reshape(self.nk, self.nr)).reshape(-1)
        self.perm2d = np.empty(self.nk * self.nr, dtype=np.int64)     # perm2d[c*nr+r] = gid
        self.perm2d[gid2d] = np.arange(self.N)
        # per-node boxes (kappa range, rho range) for the analytic hierarchical mask
        eS = np.asarray(cfg["edges"])
        self.box1 = [(eS[i], eS[i + 1], -1.0, 0.5) for i in range(S)]
        w = (eS[1:] - eS[:-1]) / LS
        self.box2 = []
        for i in range(S):
            for j in range(LS):
                for m in range(LON):
                    self.box2.append((eS[i] + j * w[i], eS[i] + (j + 1) * w[i],
                                      -1 + 1.5 * m / LON, -1 + 1.5 * (m + 1) / LON))
        self.cruise_cell = None
        for i in range(S):                                              # cell holding kappa=0, rho=0
            for j in range(LS):
                for m in range(LON):
                    lo, hi, rl, rh = self.box2[(i * LS + j) * LON + m]
                    if lo <= 0 <= hi and rl <= 0 <= rh:
                        self.cruise_cell = (i * LS + j) * LON + m
        self.cruise_flat = self.cruise_cell   # cell id = si*C + ci

    def dfs(self, X):
        lead = X.shape[:-2]
        Y = X.reshape(*lead, self.S, self.LS, self.FK, self.LON, self.FV)
        Y = np.swapaxes(Y, -3, -2)
        return Y.reshape(*lead, self.S, self.C, self.Lf)


def sample_states(rng, n):
    v0 = np.clip(np.exp(rng.normal(np.log(10.3), 0.65, n)), 0.3, 32.0)
    L_lat = np.exp(rng.normal(np.log(3.6), 0.30, n))
    L_dec = np.clip(np.exp(rng.normal(np.log(3.2), 0.35, n)), 1.5, 9.0)
    L_acc = 2.4 * np.maximum(0.1, 1 - v0 / 32) * np.exp(rng.normal(0, 0.2, n))
    types = rng.choice(6, size=(n, 3), p=[.62, .12, .08, .09, .05, .04])
    K = rng.choice([1, 2, 3], size=n, p=[.55, .35, .10])
    g = rng.gamma([3.0, 1.0, 1.0], size=(n, 3))
    g = g * (np.arange(3)[None, :] < K[:, None])
    w = g / g.sum(1, keepdims=True)
    kap = np.zeros((n, 3)); rho = np.zeros((n, 3))
    sgn = rng.choice([-1.0, 1.0], size=(n, 3))
    u = rng.random((n, 3)); z = rng.normal(size=(n, 3))
    v = v0[:, None]
    dec_cap = 0.9 * L_dec[:, None] * T_H / v          # largest feasible |rho| when braking
    acc_cap = 0.9 * L_acc[:, None] * T_H / v
    # 0 cruise, 1 slow, 2 accelerate, 3 turn (slow for the curve), 4 stop, 5 nudge
    kap_t = np.stack([0.0005 * z, 0.0005 * z, 0.0005 * z,
                      sgn * (0.02 + 0.07 * u), 0.0005 * z, sgn * (0.004 + 0.016 * u)], -1)
    vmax = np.sqrt(0.85 * L_lat[:, None] / np.maximum(np.abs(kap_t[..., 3]), 1e-4))
    rho_need = 2 * (vmax / v - 1)
    rho_t = np.stack([0.03 * z, -(0.1 + 0.5 * u), np.minimum(0.08 + 0.27 * u, acc_cap),
                      np.minimum(rho_need, -(0.02 + 0.2 * u)),
                      -np.minimum(1.0, dec_cap) + np.abs(0.02 * z), 0.04 * z], -1)
    idx = types[..., None]
    kap = np.take_along_axis(kap_t, idx, -1)[..., 0]
    rho = np.take_along_axis(rho_t, idx, -1)[..., 0]
    rho = np.clip(np.maximum(rho, -dec_cap), -0.99, 0.49)      # drivers do not ask for the impossible
    v0m = v0 * (1 + 0.03 * rng.normal(size=n))                      # measured speed at inference
    f32 = lambda x: np.asarray(x, dtype=np.float32)
    return dict(v0=f32(v0), v0m=f32(np.maximum(v0m, 0.1)), L_lat=f32(L_lat), L_dec=f32(L_dec), L_acc=f32(L_acc),
                kap=f32(kap), rho=f32(rho), w=f32(w), types=types)


def demands(W, v0):
    vm = v0[:, None] * (1 + W.rho_g[None, :] / 2)
    a_lat = (vm * vm) * np.abs(W.kap_g)[None, :]
    a_lon = W.rho_g[None, :] * v0[:, None] / T_H
    return a_lat.astype(np.float32), a_lon.astype(np.float32)


def true_cond(W, st):
    a_lat, a_lon = demands(W, st["v0"])
    feas = (a_lat <= st["L_lat"][:, None]) & (a_lon <= st["L_acc"][:, None]) & (a_lon >= -st["L_dec"][:, None])
    bump = np.zeros_like(a_lat)
    for k in range(3):
        sk = (0.0007 + 0.05 * np.abs(st["kap"][:, k]))[:, None].astype(np.float32)
        sr = (0.03 + 0.07 * np.abs(st["rho"][:, k]))[:, None].astype(np.float32)
        e = (W.kap_g[None, :] - st["kap"][:, k, None]) / sk
        e *= e
        r_ = (W.rho_g[None, :] - st["rho"][:, k, None]) / sr
        r_ *= r_
        e += r_
        e *= -0.5
        np.exp(e, out=e)
        e *= st["w"][:, k, None]
        bump += e
    P = bump * feas
    Z = P.sum(1, keepdims=True)
    bad = (Z[:, 0] < 1e-12)
    if bad.any():                                                    # degenerate state: uniform on feasible set
        P[bad] = feas[bad].astype(np.float32) + 1e-9
        Z = P.sum(1, keepdims=True)
    return P / Z


def nominal_r_all(W, st, tiers):
    """nonconformity r of every leaf under each nominal tier, using the MEASURED speed v0m (speed-dependent accel).
    r <= 1 + mu  <=>  the leaf survives the mask with limits inflated by mu."""
    a_lat, a_lon = demands(W, st["v0m"])
    out = {}
    for tn, (acc, dec, lat) in tiers.items():
        acc_n = acc * np.maximum(0.1, 1 - st["v0m"] / 32)[:, None]
        out[tn] = np.maximum(a_lat / lat, np.where(a_lon >= 0, a_lon / acc_n, -a_lon / dec)).astype(np.float32)
    return out


def box_lb(W, st, tier, boxes):
    """analytic lower bound of r over each node box (never above the min leaf r in the box)."""
    acc, dec, lat = tier
    v0 = st["v0m"][:, None]
    b = np.asarray(boxes, dtype=np.float64)                          # [n,4] klo,khi,rlo,rhi
    kmin = np.where((b[:, 0] <= 0) & (b[:, 1] >= 0), 0.0, np.minimum(np.abs(b[:, 0]), np.abs(b[:, 1])))[None, :]
    vmin = v0 * (1 + b[None, :, 2] / 2)
    r_lat = vmin ** 2 * kmin / lat
    acc_n = acc * np.maximum(0.1, 1 - v0 / 32)
    r_lon_pos = np.maximum(b[None, :, 2], 0) * v0 / T_H / acc_n
    r_lon_neg = np.maximum(-b[None, :, 3], 0) * v0 / T_H / dec
    return np.maximum(r_lat, np.maximum(r_lon_pos, r_lon_neg))


def sample_expert(P, rng):
    cs = np.cumsum(P.astype(np.float64), axis=1)
    u = rng.random(P.shape[0]) * cs[:, -1]
    return np.minimum((cs < u[:, None]).sum(1), P.shape[1] - 1)


def fde(W, gid_a, gid_b, v0):
    def end(g):
        k = W.kap_g[g].astype(np.float64); r = W.rho_g[g].astype(np.float64)
        s = v0 * T_H * (1 + r / 2); th = k * s
        ks = np.where(np.abs(k) < 1e-7, 1.0, k)
        x = np.where(np.abs(k) < 1e-7, s, np.sin(th) / ks)
        y = np.where(np.abs(k) < 1e-7, 0.5 * k * s * s, (1 - np.cos(th)) / ks)
        return x, y
    xa, ya = end(gid_a); xb, yb = end(gid_b)
    return np.hypot(xa - xb, ya - yb)


# ------------------------------------------------------------------------------------------- selection
def beam_fixed(f1, f2, f3, al1, al2, al3, b1, b2, topk=1):
    B, S, C, Lf = f3.shape
    ar = np.arange(B)[:, None]
    s1 = np.where(al1, f1, NEG)
    b1e = min(b1, S)
    k1 = np.argpartition(-s1, b1e - 1, axis=1)[:, :b1e] if b1e < S else np.tile(np.arange(S), (B, 1))
    ok1 = np.take_along_axis(al1, k1, 1)
    s2 = np.take_along_axis(np.where(al2, f2, NEG), k1[:, :, None], 1)
    s2 = np.where(ok1[:, :, None], s2, NEG).reshape(B, -1)
    n2 = (s2 > NEG / 2).sum(1)
    tot2 = b1e * C
    b2e = min(b2, tot2)
    k2 = np.argpartition(-s2, b2e - 1, axis=1)[:, :b2e] if b2e < tot2 else np.tile(np.arange(tot2), (B, 1))
    si = np.take_along_axis(k1, k2 // C, 1); ci = k2 % C
    ok2 = np.take_along_axis(s2, k2, 1) > NEG / 2
    a3g = al3[ar, si, ci] & ok2[:, :, None]
    s3 = np.where(a3g, f3[ar, si, ci], NEG).reshape(B, -1)
    n3 = a3g.sum((1, 2))
    gid_of = ((si * C + ci)[:, :, None] * Lf + np.arange(Lf)[None, None, :]).reshape(B, -1)
    pick = gid_of[np.arange(B), s3.argmax(1)]
    tk = None
    if topk > 1:
        kk = min(topk, s3.shape[1])
        ii = np.argpartition(-s3, kk - 1, axis=1)[:, :kk]
        tk = np.take_along_axis(gid_of, ii, 1)
    macs = D * (al1.sum(1) + n2 + n3)
    return dict(pick=pick, macs=macs, cells=(si * C + ci), cell_ok=ok2, strat=k1, topk=tk)


def flat_select(f3flat, al3flat):
    s = np.where(al3flat, f3flat, NEG)
    return s.argmax(1), D * al3flat.sum(1)


def factor_select(W, f3flat, fk, fr, alk, alr, al3flat, kP, kV):
    """SparseDriveV2-style: score kappa axis and rho axis separately, compose top-kP x top-kV, score composed set."""
    B = f3flat.shape[0]
    ik = np.argpartition(-np.where(alk, fk, NEG), kP - 1, axis=1)[:, :kP]
    ir = np.argpartition(-np.where(alr, fr, NEG), kV - 1, axis=1)[:, :kV]
    gid = W.perm2d[(ik[:, :, None] * W.nr + ir[:, None, :]).reshape(B, -1)]
    s = np.take_along_axis(np.where(al3flat, f3flat, NEG), gid, 1)
    pick = np.take_along_axis(gid, s.argmax(1)[:, None], 1)[:, 0]
    macs = D * (alk.sum(1) + alr.sum(1) + (s > NEG / 2).sum(1))
    return pick, macs


# ------------------------------------------------------------------------------------- pipeline pieces
CH = 100


def make_prior(W, flat):
    p3 = flat.reshape(W.S, W.C, W.Lf); p2 = p3.sum(-1); p1 = p2.sum(-1)
    return dict(flat=flat, l3=np.log(p3).astype(np.float32), l2=np.log(p2).astype(np.float32), l1=np.log(p1).astype(np.float32))


def prior_from_train(W, rng, n):
    acc = np.zeros(W.N)
    for a in range(0, n, CH):
        acc += true_cond(W, sample_states(rng, min(CH, n - a))).sum(0, dtype=np.float64)
    acc /= n
    return make_prior(W, (1 - 1e-3) * acc + 1e-3 / W.N)


def levels(W, P3):
    B = P3.shape[0]
    P3d = P3.reshape(B, W.S, W.C, W.Lf)
    P2 = P3d.sum(-1)
    return P3d, P2, P2.sum(-1)


def crit(W, P3, pr, sigma, rng, beta=1.0, sat=None, sib=0.0, pool="marginal"):
    """noisy level critics.  pool='marginal': PMI of the level marginal (what level-wise softmax training estimates);
    'max' / 'mean': node score = max / mean of the CHILDREN's clean leaf scores (ideal max-heap vs a mean-pooled centroid index).
    sat: bounded critic c*tanh(PMI/c).  sib: per-(state,parent) offsets that sibling-only InfoNCE leaves unconstrained."""
    B = P3.shape[0]
    P3d, P2, P1 = levels(W, P3)
    nz = lambda shp: (sigma * rng.standard_normal(shp, dtype=np.float32))
    if pool == "marginal":
        out = []
        for P, l in ((P1, pr["l1"]), (P2, pr["l2"]), (P3d, pr["l3"])):
            pmi = np.log(P + 1e-30) - l[None]
            if sat:
                pmi = sat * np.tanh(pmi / sat)
            out.append(pmi + beta * l[None] + nz(P.shape))
    else:
        pmi = np.log(P3d + 1e-30) - pr["l3"][None]
        if sat:
            pmi = sat * np.tanh(pmi / sat)
        L = pmi + beta * pr["l3"][None]
        red = np.max if pool == "max" else np.mean
        s2 = red(L, axis=-1); s1 = red(s2, axis=-1)
        out = [s1 + nz(s1.shape), s2 + nz(s2.shape), L + nz(L.shape)]
    if sib > 0:
        out[1] = out[1] + sib * rng.standard_normal((B, W.S, 1), dtype=np.float32)
        out[2] = out[2] + sib * rng.standard_normal((B, W.S, W.C, 1), dtype=np.float32)
    return out


def metrics(W, P3, pick, macs, v0, exp_gid):
    B = P3.shape[0]
    pp = P3[np.arange(B), pick].astype(np.float64)
    pm = P3.max(1).astype(np.float64)
    cell_pick = pick // W.Lf
    P3d, P2, P1 = levels(W, P3)
    return dict(p_pick=pp, p_map=pm, regret=np.log(pm) - np.log(pp + 1e-30),
                tact_agree=(cell_pick == P2.reshape(B, -1).argmax(1)).astype(float),
                strat_agree=((cell_pick // W.C) == P1.argmax(1)).astype(float),
                cruise=(cell_pick == W.cruise_flat).astype(float), macs=np.asarray(macs, dtype=np.float64),
                fde=fde(W, pick, exp_gid, v0))


def summ(m, ref=None):
    o = {k: float(np.mean(v)) for k, v in m.items() if k not in ("p_pick", "p_map")}
    o["E_p_pick"] = float(np.mean(m["p_pick"])); o["E_p_map"] = float(np.mean(m["p_map"]))
    o["hit_frac_of_MAP"] = o["E_p_pick"] / o["E_p_map"]
    o["regret_p95"] = float(np.percentile(m["regret"], 95)); o["macs_p95"] = float(np.percentile(m["macs"], 95))
    for k in ("cell_recall", "oracle_survival"):
        if k in m:
            o[k + "_se"] = float(np.std(m[k], ddof=1) / math.sqrt(len(m[k])))
    if ref is not None:
        dlt = m["p_pick"] - ref["p_pick"]
        o["paired_dp_vs_flat_mean_ci95"] = [float(dlt.mean()), float(1.96 * dlt.std(ddof=1) / math.sqrt(len(dlt)))]
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in o.items()}


def r3(x, nd=4):
    return float(np.round(x, nd))


def cq(x, alpha):
    n = len(x); k = math.ceil((n + 1) * (1 - alpha))
    return float(np.sort(x)[k - 1]) if k <= n else float("inf")


def cell_recall(P2, cells, ok):
    B = P2.shape[0]
    return (P2.reshape(B, -1)[np.arange(B)[:, None], cells] * ok).sum(1)


# ---------------------------------------------------------------------------------- theory: beam miss
def beam_miss_exact(N, b, m, sigma, nq=2001):
    """P(true node, m above N-1 zero-mean competitors, all + iid N(0,sigma^2), is ranked below b)."""
    e = np.linspace(-7 * sigma, 7 * sigma, nq)
    pdf = np.exp(-0.5 * (e / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    tail = np.zeros(nq)
    for i, x in enumerate(e):
        q = 0.5 * math.erfc(((m + x) / sigma) / math.sqrt(2))
        if q <= 1e-300:
            continue
        if q >= 1 - 1e-12:
            tail[i] = 1.0
            continue
        pm = sum(math.exp(math.lgamma(N) - math.lgamma(j + 1) - math.lgamma(N - j) + j * math.log(q) + (N - 1 - j) * math.log1p(-q))
                 for j in range(b))
        tail[i] = 1.0 - min(1.0, pm)
    return float(np.trapezoid(tail * pdf, e))


def markov_bound(N, b, m, sigma):
    return min(1.0, (N - 1) * 0.5 * math.erfc((m / sigma) / 2.0) / b)      # q_pair = Phibar(m/(sigma*sqrt2)) = erfc(m/(2 sigma))/2


def design_arithmetic():
    bw = 273e9
    out = {"assumed_thor_spec": {"LPDDR5X_GBps": 273, "fp8_dense_TFLOPS": 1035, "source": "vendor/secondary spec pages (search 2026-09-29); peak, not sustained"},
           "flat_score_traffic": []}
    for N in (65536, 262144, 1048576, 4194304, 16777216):
        for byt, nm in ((2, "fp16"), (1, "int8")):
            mb = N * D * byt / 1e6
            out["flat_score_traffic"].append(dict(N=N, dtype=nm, MB=round(mb, 1), ms_at_spec_bw=round(mb * 1e6 / bw * 1e3, 3), GMAC=round(N * D / 1e9, 3)))
    M, L = 72, 3
    rr = []
    for K in (8, 16, 32):
        per = K * 14 * D * D + 2 * M * D * D + 2 * K * K * D + 2 * K * M * D    # self-attn(4)+cross Q,O(2)+FFN(8) = 14 K d^2 ; scene K,V proj 2 M d^2
        rr.append(dict(K=K, GMAC_3layers=round(L * per / 1e9, 3), GFLOP_3layers=round(2 * L * per / 1e9, 3),
                       scene_KV_proj_share=round(2 * M * D * D / per, 3), params_M=round(L * 16 * D * D / 1e6, 1)))
    out["crossattn_rerank_d512_M72_L3"] = rr
    out["colbert_maxsim_256cand_5wp_72scene_GMAC"] = round(256 * 5 * M * D / 1e9, 4)
    out["factored_coarse_score_1024x256_GMAC"] = round((1024 + 256) * D / 1e9, 5)
    out["weights_read_ms_at_spec_bw"] = {"87M_bf16": round(87e6 * 2 / bw * 1e3, 3), "8B_bf16": round(8e9 * 2 / bw * 1e3, 1),
                                         "8B_fp8": round(8e9 / bw * 1e3, 1), "8B_fp4": round(4e9 / bw * 1e3, 1)}
    return out


# ------------------------------------------------------------------------------------------------ main
def run_main(W, prior, seed, n_test, n_cal, n_conf, full):
    rng = np.random.default_rng(seed)
    sig = 0.5
    acc = {}

    def put(key, m):
        acc.setdefault(key, []).append(m)

    checks = dict(tree_all_cells_equals_flat=True, rows_sum_to_one=True, box_violations=0)
    for ci, a in enumerate(range(0, n_test, CH)):
        st = sample_states(np.random.default_rng(seed * 7919 + ci), min(CH, n_test - a)); B = len(st["v0"])
        P3 = true_cond(W, st)
        checks["rows_sum_to_one"] &= bool(np.allclose(P3.sum(1), 1.0, atol=1e-4))
        P3d, P2, P1 = levels(W, P3)
        exp_gid = sample_expert(P3, rng)
        al_all = np.ones((B, W.N), dtype=bool)
        A1, A2, A3 = np.ones((B, W.S), bool), np.ones((B, W.S, W.C), bool), al_all.reshape(B, W.S, W.C, W.Lf)
        f1, f2, f3 = crit(W, P3, prior, sig, rng)
        f3flat = f3.reshape(B, -1)
        pk, mc = flat_select(f3flat, al_all)
        put("flat_on", metrics(W, P3, pk, mc, st["v0"], exp_gid))
        put("exact_MAP", metrics(W, P3, P3.argmax(1), np.full(B, D * W.N), st["v0"], exp_gid))
        put("prior_only", metrics(W, P3, np.full(B, int(prior["flat"].argmax())), np.zeros(B), st["v0"], exp_gid))
        # ---- E1: tree beam sweep
        for (b1, b2) in ((1, 4), (2, 8), (3, 16), (4, 32), (8, 64), (8, W.S * W.C)):
            r = beam_fixed(f1, f2, f3, A1, A2, A3, b1, b2, topk=32)
            m = metrics(W, P3, r["pick"], r["macs"], st["v0"], exp_gid)
            m["cell_recall"] = cell_recall(P2, r["cells"], r["cell_ok"])
            m["strat_recall"] = P1[np.arange(B)[:, None], r["strat"]].sum(1)
            m["agree_flat"] = (r["pick"] == pk).astype(float)
            put(f"tree_{b1}_{b2}", m)
            if (b1, b2) == (8, W.S * W.C):
                checks["tree_all_cells_equals_flat"] &= bool(np.array_equal(r["pick"], pk))
            if (b1, b2) == (3, 16):
                tk = r["topk"]
                sc = np.take_along_axis(f3flat, tk, 1); order = np.argsort(-sc, 1)
                tk = np.take_along_axis(tk, order, 1)
                for K in (1, 8, 16, 32):
                    kk = tk[:, :K]
                    fd = np.min(np.stack([fde(W, kk[:, i], exp_gid, st["v0"]) for i in range(K)], 1), 1)
                    lp = np.log(P3[np.arange(B)[:, None], kk] + 1e-30) + 0.15 * rng.standard_normal(kk.shape)   # low-noise re-ranker incl. prior
                    rp = np.take_along_axis(kk, lp.argmax(1)[:, None], 1)[:, 0]
                    mm = metrics(W, P3, rp, r["macs"], st["v0"], exp_gid); mm["oracle_fde_K"] = fd
                    put(f"rerank_K{K}", mm)
        # ---- E1b: factorised (SparseDriveV2-style) coarse scoring then compose
        P2d = P3[:, W.perm2d].reshape(B, W.nk, W.nr)
        Pk, Pr = P2d.sum(2), P2d.sum(1)
        fk = (np.log(Pk + 1e-30) + sig * rng.standard_normal(Pk.shape, dtype=np.float32)).astype(np.float32)
        fr = (np.log(Pr + 1e-30) + sig * rng.standard_normal(Pr.shape, dtype=np.float32)).astype(np.float32)
        alk, alr = np.ones(Pk.shape, bool), np.ones(Pr.shape, bool)
        for (kP, kV) in ((8, 4), (20, 10), (32, 16)):
            pf, mcf = factor_select(W, f3flat, fk, fr, alk, alr, al_all, kP, kV)
            m = metrics(W, P3, pf, mcf, st["v0"], exp_gid); m["agree_flat"] = (pf == pk).astype(float)
            put(f"factor_{kP}x{kV}", m)
        # ---- E2: masks
        rr_all = nominal_r_all(W, st, TIERS)
        for tn, tier in TIERS.items():
            r_leaf = rr_all[tn]
            al3 = r_leaf <= 1.0
            a3d = al3.reshape(B, W.S, W.C, W.Lf); a2 = a3d.any(-1); a1 = a2.any(-1)
            lb2 = (box_lb(W, st, tier, W.box2) <= 1.0).reshape(B, W.S, W.C); lb1 = box_lb(W, st, tier, W.box1) <= 1.0
            viol = int((a2 & ~lb2).sum() + (a1 & ~lb1).sum())
            checks["box_violations"] += viol
            pkm, mcm = flat_select(f3flat, al3)
            mm = metrics(W, P3, pkm, mcm, st["v0"], exp_gid)
            mm.update(kept_leaf_frac=al3.mean(1), oracle_survival=(P3 * al3).sum(1), cells_alive_frac=a2.reshape(B, -1).mean(1),
                      box_false_keep=np.full(B, (lb2 & ~a2).sum() / max(lb2.sum(), 1)))
            put(f"mask_flat_{tn}", mm)
            if tn in ("T0_nuplan_comfort", "T1_tight"):
                r = beam_fixed(f1, f2, f3, a1, a2, a3d, 3, 16)
                mt = metrics(W, P3, r["pick"], r["macs"], st["v0"], exp_gid)
                cell_mass = (P3d * a3d).sum(-1)
                mt["oracle_survival"] = (cell_mass.reshape(B, -1)[np.arange(B)[:, None], r["cells"]] * r["cell_ok"]).sum(1)
                put(f"mask_tree_3_16_{tn}", mt)
            if tn in ("T0_nuplan_comfort", "T1_tight"):
                for mu in (0.0, 0.1, 0.25, 0.5, 1.0):
                    al = r_leaf <= 1.0 + mu
                    put(f"mu_{tn}_{mu}", dict(surv=(P3 * al).sum(1), kept=al.mean(1)))
        # ---- E4: prior-correction variants
        variants = {"off_beta0": dict(beta=0.0), "on_beta1": dict(beta=1.0), "sat_c3_beta0": dict(beta=0.0, sat=3.0),
                    "sat_c3_beta0p75": dict(beta=0.75, sat=3.0), "sat_c3_beta1": dict(beta=1.0, sat=3.0), "sib0p5_beta1": dict(beta=1.0, sib=0.5),
                    "sib1_beta1": dict(beta=1.0, sib=1.0), "sib2_beta1": dict(beta=1.0, sib=2.0)}
        for vn, kw in variants.items():
            g1, g2, g3 = crit(W, P3, prior, sig, rng, **kw)
            pkv, mcv = flat_select(g3.reshape(B, -1), al_all)
            put(f"prior_flat_{vn}", metrics(W, P3, pkv, mcv, st["v0"], exp_gid))
            r = beam_fixed(g1, g2, g3, A1, A2, A3, 3, 16)
            mt = metrics(W, P3, r["pick"], r["macs"], st["v0"], exp_gid)
            mt["cell_recall"] = cell_recall(P2, r["cells"], r["cell_ok"]); mt["agree_flat"] = (r["pick"] == pkv).astype(float)
            put(f"prior_tree_{vn}", mt)
        # ---- E1c: critic-noise sweep
        for sg in (0.25, 1.0, 2.0):
            g1, g2, g3 = crit(W, P3, prior, sg, rng)
            pkv, mcv = flat_select(g3.reshape(B, -1), al_all)
            r = beam_fixed(g1, g2, g3, A1, A2, A3, 3, 16)
            mm = metrics(W, P3, r["pick"], r["macs"], st["v0"], exp_gid); mm["agree_flat"] = (r["pick"] == pkv).astype(float)
            mm["cell_recall"] = cell_recall(P2, r["cells"], r["cell_ok"])
            put(f"sigma_{sg}_tree_3_16", mm); put(f"sigma_{sg}_flat", metrics(W, P3, pkv, mcv, st["v0"], exp_gid))
        # ---- E7: node-critic type (marginal / max-pool / mean-pool), bounded critic
        for pool in ("marginal", "max", "mean"):
            g1, g2, g3 = crit(W, P3, prior, sig, rng, beta=0.75, sat=3.0, pool=pool)
            pkv, mcv = flat_select(g3.reshape(B, -1), al_all)
            put(f"pool_{pool}_flat", metrics(W, P3, pkv, mcv, st["v0"], exp_gid))
            for (b1, b2) in ((1, 4), (2, 8), (3, 16), (4, 32)):
                r = beam_fixed(g1, g2, g3, A1, A2, A3, b1, b2)
                mm = metrics(W, P3, r["pick"], r["macs"], st["v0"], exp_gid)
                mm["cell_recall"] = cell_recall(P2, r["cells"], r["cell_ok"]); mm["agree_flat"] = (r["pick"] == pkv).astype(float)
                put(f"pool_{pool}_tree_{b1}_{b2}", mm)
    T = {k: {kk: np.concatenate([m[kk] for m in ms]) for kk in ms[0]} for k, ms in acc.items()}
    flat = T["flat_on"]
    out = dict(meta=dict(N=W.N, tree=f"{W.S} x {W.C} x {W.Lf}", n_test=n_test, sigma=sig, prior_cruise_cell_mass=r3(prior["flat"].reshape(W.S * W.C, W.Lf)[W.cruise_flat].sum(), 4),
                         prior_top_leaf_mass=r3(prior["flat"].max(), 5), invariants_checked=checks))
    out["reference"] = {k: summ(T[k]) for k in ("exact_MAP", "prior_only", "flat_on")}
    out["E1_tree_vs_flat"] = {k: summ(v, flat) for k, v in T.items() if k.startswith("tree_")}
    out["E1_factorised"] = {k: summ(v, flat) for k, v in T.items() if k.startswith("factor_")}
    out["E1_sigma_sweep"] = {k: summ(v) for k, v in T.items() if k.startswith("sigma_")}
    out["E2_masks"] = {k: summ(v, flat) for k, v in T.items() if k.startswith("mask_")}
    out["E2_mu_sweep_exact_mass"] = {k: dict(oracle_survival=r3(v["surv"].mean()), kept_leaf_frac=r3(v["kept"].mean())) for k, v in T.items() if k.startswith("mu_")}
    out["E4_prior"] = {k: summ(v) for k, v in T.items() if k.startswith("prior_")}
    out["E6_rerank_by_K"] = {k: dict(summ(v), oracle_fde_K=r3(v["oracle_fde_K"].mean(), 3)) for k, v in T.items() if k.startswith("rerank_")}
    out["E7_node_critic_type_sat3"] = {k: summ(v) for k, v in T.items() if k.startswith("pool_")}
    if not full:
        return out, T
    # ---------------- calibration (conformal): mask inflation on T0, per-level beam margins, beta fit
    cal_r, cal_v0, nc1, nc2 = [], [], [], []
    fitp = {b: [] for b in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)}
    for a in range(0, n_cal, CH):
        st = sample_states(rng, min(CH, n_cal - a)); B = len(st["v0"])
        P3 = true_cond(W, st); g = sample_expert(P3, rng)
        rl = nominal_r_all(W, st, {"T0_nuplan_comfort": TIERS["T0_nuplan_comfort"]})["T0_nuplan_comfort"]
        cal_r.append(rl[np.arange(B), g]); cal_v0.append(st["v0m"])
        f1, f2, f3 = crit(W, P3, prior, sig, rng)
        cell = g // W.Lf; s2 = f2.reshape(B, -1)
        nc1.append(np.maximum(f1.max(1) - f1[np.arange(B), cell // W.C], 0)); nc2.append(np.maximum(s2.max(1) - s2[np.arange(B), cell], 0))
        if a < 600:                                       # beta fit on the first 600 calibration states only (cost)
            for b in fitp:
                _, _, g3 = crit(W, P3, prior, sig, rng, beta=b, sat=3.0)
                pkb, _ = flat_select(g3.reshape(B, -1), np.ones((B, W.N), bool))
                fitp[b].append(P3[np.arange(B), pkb])
    cal_r, cal_v0, nc1, nc2 = (np.concatenate(x) for x in (cal_r, cal_v0, nc1, nc2))
    out["E4_beta_fit_bounded_critic_E_p_pick"] = {str(b): r3(np.concatenate(v).mean(), 5) for b, v in fitp.items()}
    mus = {a_: max(cq(cal_r, a_) - 1.0, 0.0) for a_ in (0.10, 0.05, 0.01)}
    kept = {a_: [] for a_ in mus}; r_test, v_test, dom = [], [], []
    for a in range(0, n_conf, CH):
        st = sample_states(rng, min(CH, n_conf - a)); B = len(st["v0"])
        P3 = true_cond(W, st); g = sample_expert(P3, rng)
        rl = nominal_r_all(W, st, {"T0_nuplan_comfort": TIERS["T0_nuplan_comfort"]})["T0_nuplan_comfort"]
        r_test.append(rl[np.arange(B), g]); v_test.append(st["v0m"])
        dom.append(np.take_along_axis(st["types"], st["w"].argmax(1)[:, None], 1)[:, 0])
        for a_, mu in mus.items():
            kept[a_].append((rl <= 1 + mu).mean(1))
    r_test, v_test, dom = (np.concatenate(x) for x in (r_test, v_test, dom))
    conf = {str(a_): dict(target=r3(1 - a_, 3), q_hat=r3(mu + 1, 4), mu_hat=r3(mu, 4), test_coverage=r3((r_test <= 1 + mu).mean(), 4),
                          kept_leaf_frac=r3(np.concatenate(kept[a_]).mean(), 4)) for a_, mu in mus.items()}
    conf["uncalibrated_mu0"] = dict(test_coverage=r3((r_test <= 1.0).mean(), 4), n_cal=len(cal_r), n_test=len(r_test))
    names = ["cruise", "slow", "accelerate", "turn", "stop", "nudge"]
    mu5 = mus[0.05]
    conf["coverage_by_dominant_mode_at_alpha0.05_global_mu"] = {names[t]: dict(n=int((dom == t).sum()), coverage=r3((r_test[dom == t] <= 1 + mu5).mean(), 3))
                                                              for t in range(6) if (dom == t).sum() > 10}
    ed = np.quantile(cal_v0, [1 / 3, 2 / 3]); bc, bt = np.digitize(cal_v0, ed), np.digitize(v_test, ed)
    conf["mondrian_v0_terciles_alpha0.05"] = {f"bin{bi}": dict(v0_range=[r3(v_test[bt == bi].min(), 1), r3(v_test[bt == bi].max(), 1)], n_cal=int((bc == bi).sum()), n_test=int((bt == bi).sum()),
                                                            cov_global_mu=r3((r_test[bt == bi] <= 1 + mu5).mean(), 3), mu_bin=r3(max(cq(cal_r[bc == bi], 0.05) - 1, 0), 3),
                                                            cov_bin_mu=r3((r_test[bt == bi] <= 1 + max(cq(cal_r[bc == bi], 0.05) - 1, 0)).mean(), 3)) for bi in range(3)}
    out["E3_conformal_mask_T0"] = conf
    # ---------------- E5: adaptive margin beam with calibrated per-level margins (same test states as above)
    ad = {}
    for al_ in (0.05, 0.02, 0.01):
        d1, d2 = cq(nc1, al_), cq(nc2, al_)
        rows = []
        for ci, a in enumerate(range(0, n_test, CH)):
            st = sample_states(np.random.default_rng(seed * 7919 + ci), min(CH, n_test - a)); B = len(st["v0"])
            rs = np.random.default_rng(seed + 31 * ci)
            P3 = true_cond(W, st); P3d, P2, P1 = levels(W, P3)
            f1, f2, f3 = crit(W, P3, prior, sig, rs)
            keep1 = f1 >= (f1.max(1, keepdims=True) - d1)
            s2 = np.where(keep1[:, :, None], f2, NEG).reshape(B, -1)
            keep2 = ((s2 >= s2.max(1, keepdims=True) - d2) & (s2 > NEG / 2)).reshape(B, W.S, W.C)
            s3 = np.where(keep2[..., None], f3, NEG).reshape(B, -1)
            pick = s3.argmax(1)
            macs = D * (W.S + keep1.sum(1) * W.C + keep2.sum((1, 2)) * W.Lf)
            m = metrics(W, P3, pick, macs, st["v0"], sample_expert(P3, rs))
            m["cell_recall"] = (P2 * keep2).sum((1, 2)); m["cells_kept"] = keep2.sum((1, 2)).astype(float)
            rows.append(m)
        mm = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        ad[str(al_)] = dict(delta1=r3(d1, 3), delta2=r3(d2, 3), union_bound_floor=r3(1 - 2 * al_, 3), **summ(mm),
                            cells_kept_p95=r3(np.percentile(mm["cells_kept"], 95), 1))
    out["E5_adaptive_margin_beam"] = ad
    return out, T


def prior_estimation_experiment(W, seed):
    """critic built with the TRUE marginal prior; correction uses a prior ESTIMATED from n hard-count expert samples."""
    rng = np.random.default_rng(seed)
    truth = prior_from_train(W, rng, 1000)
    cnt = np.zeros(W.N)
    n_max = 1500; sizes = (100, 300, 1000, 1500); snaps = {}
    done = 0
    for a in range(0, n_max, CH):
        st = sample_states(rng, CH); g = sample_expert(true_cond(W, st), rng)
        np.add.at(cnt, g, 1.0); done += CH
        if done in sizes:
            snaps[done] = cnt.copy()

    def est_flat(c, n):
        return make_prior(W, (1 - 1e-3) * c / n + 1e-3 / W.N)

    def est_hier(c, n):
        c3 = c.reshape(W.S, W.C, W.Lf); c2 = c3.sum(-1); c1 = c2.sum(-1)
        p1 = (c1 + 0.5) / (n + 0.5 * W.S); p2c = (c2 + 0.5) / (c1[:, None] + 0.5 * W.C)
        p3c = (c3 + 0.5 / W.Lf) / (c2[..., None] + 0.5)
        return make_prior(W, (p1[:, None, None] * p2c[..., None] * p3c).reshape(-1))

    res = {n: {} for n in sizes}
    tests = []
    for ci in range(6):
        st = sample_states(np.random.default_rng(seed + 100 + ci), CH)
        P3 = true_cond(W, st); tests.append((P3, 0.5 * rng.standard_normal(P3.shape, dtype=np.float32)))   # same noise for every estimator (paired)
    pm_all = np.concatenate([P3.max(1) for P3, _ in tests])
    brng = np.random.default_rng(seed + 5)
    bidx = brng.integers(0, len(pm_all), size=(300, len(pm_all)))

    def hit(pe_l3):
        pp = []
        for P3, nz in tests:
            B = P3.shape[0]
            score = np.log(P3 + 1e-30) - truth["l3"].reshape(1, -1) + pe_l3 + nz
            pp.append(P3[np.arange(B), score.argmax(1)])
        pp = np.concatenate(pp)
        se = float(np.std(pp[bidx].mean(1) / pm_all[bidx].mean(1), ddof=1))
        return r3(pp.mean() / pm_all.mean(), 4), r3(se, 4)

    for n in sizes:
        ests = {"flat_eps1e-3": est_flat(snaps[n], n), "hierarchical_backoff": est_hier(snaps[n], n)}
        for en, pe in ests.items():
            res[n][en] = hit(pe["l3"].reshape(1, -1))
    ref = {"true_prior_beta1": hit(truth["l3"].reshape(1, -1)), "no_correction_beta0": hit(np.float32(0.0))}
    return dict(hit_frac_of_MAP_by_n_expert_samples=res, references=ref, n_test=6 * CH, format='(hit_frac_of_MAP, bootstrap SE)',
                mean_expert_samples_per_leaf={str(n): r3(n / W.N, 5) for n in sizes})


def size_sweep(seed):
    rows = {}
    for nm, fk, fv, cfgn in (("N=4096 (8x32x16)", 4, 4, "A_8x32x256"), ("N=16384 (8x32x64)", 8, 8, "A_8x32x256"),
                             ("N=65536 (8x32x256)", 16, 16, "A_8x32x256"), ("N=131072 (8x64x256)", 16, 16, "B_v7shape_8x64x256")):
        cfg = dict(CFGS[cfgn]); cfg["FK"], cfg["FV"] = fk, fv
        W = World(cfg); rng = np.random.default_rng(seed)
        pr = prior_from_train(W, rng, 800)
        acc = {"off": [], "on": [], "map": [], "cr_off": [], "cr_on": [], "cr_map": []}
        for ci in range(3):
            st = sample_states(np.random.default_rng(seed + 50 + ci), CH); B = CH
            P3 = true_cond(W, st)
            for nmv, beta in (("off", 0.0), ("on", 1.0)):
                g1, g2, g3 = crit(W, P3, pr, 0.5, rng, beta=beta)
                pk = g3.reshape(B, -1).argmax(1)
                acc[nmv].append(P3[np.arange(B), pk]); acc["cr_" + nmv].append((pk // W.Lf == W.cruise_flat).astype(float))
            acc["map"].append(P3.max(1)); acc["cr_map"].append((P3.argmax(1) // W.Lf == W.cruise_flat).astype(float))
        c = {k: np.concatenate(v).mean() for k, v in acc.items()}
        pm_, po_, pn_ = (np.concatenate(acc[k]) for k in ("map", "off", "on"))
        bi = np.random.default_rng(seed + 9).integers(0, len(pm_), size=(300, len(pm_)))
        se0 = float(np.std(po_[bi].mean(1) / pm_[bi].mean(1), ddof=1)); se1 = float(np.std(pn_[bi].mean(1) / pm_[bi].mean(1), ddof=1))
        rows[nm] = dict(hit_frac_MAP_beta0=r3(c["off"] / c["map"], 4), se_beta0=r3(se0, 4), hit_frac_MAP_beta1=r3(c["on"] / c["map"], 4), se_beta1=r3(se1, 4),
                        cruise_share_beta0=r3(c["cr_off"], 3), cruise_share_beta1=r3(c["cr_on"], 3), cruise_share_exactMAP=r3(c["cr_map"], 3))
    return rows


def timing():
    r = np.random.default_rng(0)
    tab = r.standard_normal((65536, D)).astype(np.float32); T2 = r.standard_normal((256, D)).astype(np.float32)
    T1 = r.standard_normal((8, D)).astype(np.float32); q = r.standard_normal(D).astype(np.float32)

    def flat():
        return (tab @ q).argmax()

    def tree():
        k1 = np.argpartition(-(T1 @ q), 2)[:3]
        s2 = np.concatenate([T2[k * 32:(k + 1) * 32] @ q for k in k1])
        k2 = np.argpartition(-s2, 15)[:16]
        cells = k1[k2 // 32] * 32 + k2 % 32
        return max(float((tab[c * 256:(c + 1) * 256] @ q).max()) for c in cells)

    def tm(fn, n=30):
        fn(); t = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t) / n * 1e3
    return dict(flat_65536x512_ms=round(tm(flat), 3), tree_3_16_ms=round(tm(tree), 3),
                note="this sandbox CPU, float32 numpy, contiguous per-cell slabs; NOT Thor, NOT deterministic; illustrates trend only")


def main():
    t0 = time.time()
    res = {"seed": SEED, "D": D, "n_chunk": CH}
    WA = World(CFGS["A_8x32x256"]); rng = np.random.default_rng(SEED)
    prA = prior_from_train(WA, rng, 1000)
    outA, _ = run_main(WA, prA, SEED, 400, 1200, 1500, True)
    res["A_8x32x256"] = dict(outA, n_train=1000)
    WB = World(CFGS["B_v7shape_8x64x256"]); prB = prior_from_train(WB, np.random.default_rng(SEED + 1), 800)
    outB, _ = run_main(WB, prB, SEED + 1, 200, 0, 0, False)
    res["B_v7shape_8x64x256"] = dict(outB, n_train=800)
    res["E8_prior_estimation_config_A"] = prior_estimation_experiment(WA, SEED + 2)
    res["E9_pmi_bias_vs_vocabulary_size"] = size_sweep(SEED + 3)
    th = []
    for (N, b) in ((32, 4), (32, 8), (256, 16)):
        for m in (0.0, 1.0, 2.0, 3.0, 4.0):
            th.append(dict(children=N, beam=b, margin_in_sigma=m, miss_exact=r3(beam_miss_exact(N, b, m, 1.0), 5), markov_upper_bound=r3(markov_bound(N, b, m, 1.0), 5)))
    res["theory_beam_miss_iid_gaussian"] = th
    res["design_arithmetic"] = design_arithmetic()
    res["timing_cpu_numpy_nondeterministic"] = timing()
    res["total_runtime_s_nondeterministic"] = round(time.time() - t0, 1)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("runtime_s", res["total_runtime_s_nondeterministic"], "->", OUT)


if __name__ == "__main__":
    main()
