"""LAT-TOK stage 2 (ZERO training, CPU): the 2 s yaw-rate readout on three feature sets, both R35 chassis, clip-grouped 5-fold + swap slope.

FROZEN PRE-REGISTRATION: AMENDMENT_1_BCAL_DRAFT.md sha256 4e122aa5cea41d728951046d9b7579a84237700422a1b0a5f0dd2e33c6f53834, section 2.2 item 1.
Windows, K = 10 pairs, targets (abar = mean WM-scaled action over the 10 transitions; ch0 acceleration, ch1 yaw rate), the clip fold map, the swap partner `part`,
the ridge (standardised, lambda grid, inner clip-grouped folds), the MLP pair (256-256 GELU, 12 epochs, AdamW 1e-3, wd 1e-2) and the clip-cluster percentile
bootstrap (B 1000, seed 0) are the code of amend1_kstep.py, copied; the ONLY change is that the ridge is solved in the dual (kernel) form when d > n (the same
solution, exact up to float64), with lambda chosen PER TARGET COLUMN on the same inner folds.
Feature sets (per frame): (a) 40-token mean 256, (b) 10 column means 2560, (c) all 40 tokens 10240.
Readouts per set: LINEAR D = ridge on z_{t+10} - z_t; MLP PAIR = MLP on [z_t, z_{t+10}] (one MLP per target channel, the RY form).
s_swap (yaw) = Cov(R(real) - R(swap), abar - abar') / Var(abar - abar'), swap input = z_t + (z_{t'+10} - z_{t'}), t' = `part`.
THE FROZEN RULE, committed here BEFORE any data is read: a cell (set in {b, c}, readout in {linear D, MLP pair}) PASSES on a chassis iff
  the paired clip-cluster 95 % CI of [R2_yaw(set) - R2_yaw(a)] (same readout, same resamples) is wholly above 0 AND the CI of s_swap_yaw(set) is wholly above 0.
LAT-TOK PASSES iff some cell passes on BOTH chassis (R35_s0 and R35_s1).  All four cells are printed; the multiplicity (4 cells) is part of the report.
Acceleration is reported as context only and enters no rule.
Controls: (1) feature set z1 (the banked 128-d latent) with linear D must read kstep's acc R2 (0.211 / 0.205) and the banked linear-D yaw R2; (2) a permuted yaw
target must read R2 <= ~0; (3) n_pairs and the mean logged action must equal kstep_R35_s{k}.json.
Usage: python lattok_analysis.py <featdir> <outjson> [smoke]
"""
import dataclasses, json, sys, time
import numpy as np
import torch

torch.set_num_threads(8)
sys.path.insert(0, "/root/lattok/stack")
from tanitad.models import v8f_mini as M  # noqa: E402

FEAT, OUT = sys.argv[1], sys.argv[2]
SMOKE = len(sys.argv) > 3
T0 = time.time(); KS = 10
SL = {"a": slice(0, 256), "b": slice(256, 2816), "c": slice(2816, 13056)}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- kstep code (copied)
LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def mlp_fit(X, Yy, seed=0, ep=12):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = torch.tensor((X - mu) / sd, dtype=torch.float32); Yt = torch.tensor(Yy, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, Yy.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    idx = np.arange(len(X))
    for e in range(ep):
        np.random.RandomState(e).shuffle(idx)
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]
            loss = ((net(Xs[b]) - Yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, mu, sd


def mlp_pred(m, X):
    net, mu, sd = m
    with torch.no_grad():
        return net(torch.tensor((X - mu) / sd, dtype=torch.float32)).numpy().astype(np.float64)


def r2(p, y):
    return 1 - ((p - y) ** 2).mean() / ((y - y.mean()) ** 2).mean()


def cv(x, y):
    return ((x - x.mean()) * (y - y.mean())).mean()


# ---------------------------------------------------------------- ridge: kstep's solution; primal when d <= n, dual otherwise
class Ridge:
    def __init__(self, X, Y):
        self.n, self.d = X.shape
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xs = (X - self.mu) / self.sd
        self.ym = Y.mean(0); Yc = Y - self.ym
        self.primal = self.d <= self.n
        if self.primal:
            s, V = np.linalg.eigh(Xs.T @ Xs)
            self.s, self.V, self.B = s, V, V.T @ (Xs.T @ Yc)
        else:
            s, U = np.linalg.eigh(Xs @ Xs.T)
            self.s, self.U, self.B, self.Xs = s, U, U.T @ Yc, Xs

    def predict(self, X, lam_cols):
        """X [m, d] -> [m, ncol], column c with its own lambda lam_cols[c]."""
        Xt = (X - self.mu) / self.sd
        out = np.empty((len(X), self.B.shape[1]))
        P = Xt @ self.V if self.primal else Xt @ self.Xs.T
        for c, lam in enumerate(lam_cols):
            coef = self.B[:, c] / (self.s + lam * self.n)
            out[:, c] = P @ coef if self.primal else P @ (self.U @ coef)
            out[:, c] += self.ym[c]
        return out


def ridge_choose(X, Y, fo):
    """per-column lambda on the inner clip-grouped folds (kstep's ridge_lam, column by column)."""
    err = np.zeros((len(LAMS), Y.shape[1]))
    for g in np.unique(fo):
        m = Ridge(X[fo != g], Y[fo != g])
        for li, lam in enumerate(LAMS):
            p = m.predict(X[fo == g], [lam] * Y.shape[1])
            err[li] += ((p - Y[fo == g]) ** 2).sum(0)
    return [LAMS[i] for i in err.argmin(0)]


def load_chassis(k):
    TAG = f"R35_s{k}"
    ck = torch.load(f"/root/lattok/data/ckpt_{TAG}.pt", map_location="cpu", weights_only=False)
    names = {f.name for f in dataclasses.fields(M.V8FMiniConfig)}
    cfg = M.V8FMiniConfig(**{kk: (tuple(v) if isinstance(v, list) else v) for kk, v in ck["cfg"].items() if kk in names})
    torch.manual_seed(0)
    wm = M.WMAux(cfg)
    wm.load_state_dict({kk[3:]: v for kk, v in ck["model"].items() if kk.startswith("wm.")}, strict=True)
    wm.eval()
    L = torch.load(f"/root/lattok/data/latents_train_{TAG}.pt", map_location="cpu", weights_only=False)
    return TAG, wm, L


def run_chassis(k):
    TAG, wm, L = load_chassis(k)
    Aw, okw, tw, eidw = wm.scale_actions(L["A_raw"].float()), L["ok"].bool(), np.asarray(L["t"]), list(L["eid"])
    Zb = L["Z1"][TAG].float()
    W = Zb.shape[1]
    clips = sorted(set(eidw)); cix = {c: i for i, c in enumerate(clips)}
    rows = np.load(f"{FEAT}/rows.npy"); row_of = {(int(a), int(b)): i for i, (a, b) in enumerate(rows)}
    feat = np.load(f"{FEAT}/feat_s{k}.npy", mmap_mode="r")
    tab = {}
    for w in range(Zb.shape[0]):
        for j in range(W):
            if bool(okw[w, j]):
                key = (cix[eidw[w]], int(tw[w]) - (W - 1) + j)
                if key in row_of:
                    tab.setdefault(key, (row_of[key], Zb[w, j], Aw[w, j]))
    seq = {}
    for ci in sorted({c for c, _ in tab}):
        fr = sorted(f for (c, f) in tab if c == ci)
        seq[ci] = (np.array(fr), np.array([tab[(ci, f)][0] for f in fr]), torch.stack([tab[(ci, f)][1] for f in fr]), torch.stack([tab[(ci, f)][2] for f in fr]))
    EP, IT, IK, ZB1, ZB2, AB, FR, ANC = [], [], [], [], [], [], [], []
    for ci, (fr, rw, Z, A) in seq.items():
        pos = {int(f): i for i, f in enumerate(fr)}
        for i, f in enumerate(fr):
            f = int(f)
            if all((f + m) in pos for m in range(KS + 1)):
                js = [pos[f + m] for m in range(KS)]
                if torch.isfinite(A[js]).all():
                    EP.append(ci); IT.append(rw[i]); IK.append(rw[pos[f + KS]]); ZB1.append(Z[i]); ZB2.append(Z[pos[f + KS]]); AB.append(A[js].mean(0)); FR.append(f); ANC.append(A[js])
    EP, FR, IT, IK = np.array(EP), np.array(FR), np.array(IT), np.array(IK)
    n = len(EP)
    Y = torch.stack(AB).numpy().astype(np.float64)
    kst = json.load(open(f"/root/lattok/data/kstep_R35_s{k}.json"))
    amean = torch.stack(ANC).reshape(-1, 2).mean(0).tolist()
    ctrl = {"n_pairs": n, "kstep_n_pairs": kst["n_pairs"], "a_mean_wm": amean, "kstep_a_mean_wm": kst["a_mean_wm"]}
    log(TAG, "pairs", n, "kstep", kst["n_pairs"], "clips", len(set(EP.tolist())))
    rng = np.random.RandomState(0)
    part = np.empty(n, dtype=int); groups = {}
    for i, e in enumerate(EP):
        groups.setdefault(int(e), []).append(i)
    for e, g in groups.items():
        g = np.array(g)
        for i in g:
            cand = g[np.abs(FR[g] - FR[i]) >= 5]
            part[i] = rng.choice(cand) if len(cand) else i
    fold = (np.random.RandomState(0).permutation(len(clips)) % 5)[EP]
    grp = [np.array(v) for v in groups.values()]
    Ysh = Y[np.random.RandomState(1).permutation(n), 1]            # permuted yaw target (control: must read R2 <= ~0)
    Yl = np.stack([Y[:, 0], Y[:, 1], Ysh], 1)
    sets = {"z1": (torch.stack(ZB1).numpy().astype(np.float64), torch.stack(ZB2).numpy().astype(np.float64))}
    for s, sl in SL.items():
        sets[s] = (np.asarray(feat[IT][:, sl], dtype=np.float32), np.asarray(feat[IK][:, sl], dtype=np.float32))
    H = {}
    for s, (Zt, Zk) in sets.items():
        if SMOKE and s in ("b", "c"):
            pass
        t1 = time.time()
        Dr, Dsw = Zk - Zt, Zk[part] - Zt[part]
        h = {key: np.zeros(n) for key in ("lin_real_acc", "lin_swap_acc", "lin_real_yaw", "lin_swap_yaw", "lin_real_shuf", "mlp_real_acc", "mlp_swap_acc", "mlp_real_yaw", "mlp_swap_yaw")}
        lams = []
        for f in range(5):
            tr, te = fold != f, fold == f
            if not te.any() or not tr.any():
                continue
            Xtr = Dr[tr].astype(np.float64)
            lam = ridge_choose(Xtr, Yl[tr], fold[tr]); lams.append(lam)
            m = Ridge(Xtr, Yl[tr])
            pr, ps = m.predict(Dr[te].astype(np.float64), lam), m.predict(Dsw[te].astype(np.float64), lam)
            h["lin_real_acc"][te], h["lin_swap_acc"][te] = pr[:, 0], ps[:, 0]
            h["lin_real_yaw"][te], h["lin_swap_yaw"][te] = pr[:, 1], ps[:, 1]
            h["lin_real_shuf"][te] = pr[:, 2]
            del m, Xtr
            if s != "z1":
                Xp = np.concatenate([Zt, Zk], 1); Xps = np.concatenate([Zt, Zt + Dsw], 1)
                for ch, nm in ((1, "yaw"), (0, "acc")):
                    mm = mlp_fit(Xp[tr], Y[tr][:, ch:ch + 1], seed=f)
                    h[f"mlp_real_{nm}"][te] = mlp_pred(mm, Xp[te])[:, 0]; h[f"mlp_swap_{nm}"][te] = mlp_pred(mm, Xps[te])[:, 0]
                del Xp, Xps
            log(TAG, "set", s, "fold", f, "lam(acc,yaw,shuf)", lam, f"{time.time() - t1:.0f}s")
        H[s] = h; ctrl.setdefault("lambdas", {})[s] = lams

    def stats_for(ii):
        out = {}
        for s, h in H.items():
            for r in ("lin", "mlp"):
                if s == "z1" and r == "mlp":
                    continue
                for ch, nm in ((1, "yaw"), (0, "acc")):
                    y, d = Y[ii, ch], Y[ii, ch] - Y[part[ii], ch]
                    real, sw = h[f"{r}_real_{nm}"][ii], h[f"{r}_swap_{nm}"][ii]
                    out[f"{r}.{s}.{nm}.r2"] = r2(real, y)
                    out[f"{r}.{s}.{nm}.s_swap"] = cv(real - sw, d) / d.var()
            out[f"lin.{s}.shuf.r2"] = r2(H[s]["lin_real_shuf"][ii], Y[ii, 1])
        for r in ("lin", "mlp"):
            for s in ("b", "c"):
                for nm in ("yaw", "acc"):
                    out[f"{r}.{s}-a.{nm}.dR2"] = out[f"{r}.{s}.{nm}.r2"] - out[f"{r}.a.{nm}.r2"]
        return out

    pt = stats_for(np.arange(n)); keys = list(pt)
    r = np.random.RandomState(0); bs = np.empty((1000, len(keys)))
    for b in range(1000):
        d = stats_for(np.concatenate([grp[g] for g in r.randint(0, len(grp), len(grp))]))
        bs[b] = [d[kk] for kk in keys]
    lo, hi = np.percentile(bs, [2.5, 97.5], 0)
    res = {kk: {"point": float(pt[kk]), "ci95": [float(lo[i]), float(hi[i])]} for i, kk in enumerate(keys)}
    res["_ctrl"] = ctrl
    log(TAG, "done", {kk: round(v["point"], 4) for kk, v in res.items() if kk != "_ctrl" and (kk.endswith("yaw.r2") or kk.endswith("yaw.s_swap") or "dR2" in kk and "yaw" in kk)})
    return res


R = {}
for k in (0, 1):
    R[f"R35_s{k}"] = run_chassis(k)
rule = {}
for r in ("lin", "mlp"):
    for s in ("b", "c"):
        per = {}
        for tag, res in R.items():
            dr, ss = res[f"{r}.{s}-a.yaw.dR2"]["ci95"], res[f"{r}.{s}.yaw.s_swap"]["ci95"]
            per[tag] = {"dR2_lo": dr[0], "s_swap_lo": ss[0], "pass": bool(dr[0] > 0 and ss[0] > 0)}
        rule[f"{r}.{s}"] = {"per_chassis": per, "pass_both": all(v["pass"] for v in per.values())}
verdict = "PASS" if any(v["pass_both"] for v in rule.values()) else "FAIL"
json.dump({"prereg_sha256": "4e122aa5cea41d728951046d9b7579a84237700422a1b0a5f0dd2e33c6f53834", "estimator": "clip-cluster percentile bootstrap, B 1000, seed 0, 141 TRAIN clips (paired: all sets on the same resamples); answers another draw of clips, adapter and fitted readouts held fixed",
           "smoke": SMOKE, "rule_cells": rule, "verdict": verdict, "results": R, "wall_s": time.time() - T0}, open(OUT, "w"), indent=1)
log("VERDICT", verdict, json.dumps({k: v["pass_both"] for k, v in rule.items()}))
