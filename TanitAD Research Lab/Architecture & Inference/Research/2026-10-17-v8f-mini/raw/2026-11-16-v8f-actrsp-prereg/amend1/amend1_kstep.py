"""ACTRSP amendment 1, Part E: the 2 s (K = 10) FIXED readouts that see the transition, measured on R35 TRAIN latents (141 TRAIN clips; no eval window).

Readouts (each cross-fitted with folds grouped by CLIP, the Part D fold map, for every skill and slope number; then fit once on all 141 clips = the frozen readout):
  RA (acceleration): ridge on D = z_t+10 - z_t -> mean acceleration over the 10 transitions (WM units); lambda chosen on inner clip-grouped folds.
  RY (yaw rate):     MLP 256-256 GELU on [z_t, z_t+10] -> mean yaw rate over the 10 transitions (the IDM head's shape), 12 epochs, AdamW 1e-3, wd 1e-2.
  State-only baselines: ridge on z_t and MLP on z_t (same folds); the reference is the BETTER of the two per channel.
Transition-only slope (the calibration a readout must carry; z_t held fixed exactly):
  SWAP:    y = z_t + D_t', t' another start of the SAME clip >= 5 frames away (fixed RNG); s_swap = Cov(R(real) - R(swap), abar - abar') / Var(abar - abar').
  WM_MEAN: y = F^10(z_t, a_mean) (R35's own 10-step rollout, context 1); s_wm = Cov(R(real) - R(wm), abar) / Var(abar).
Term preview on R35 at its end (the fine-tune's step 0): anchors every 10 frames, the four +/- delta contrasts over K = 10 steps at context 1, read at the
  ENDPOINT by the frozen readouts, G[c', c] = (R_c'(+) - R_c'(-)) / (2 delta_c s_c'), with s = the cross-fitted s_swap.
Also: R(z_t, F^10(z_t, a_true)) against abar (does R35's own rollout under the logged actions carry the action through the readout).
Estimator: clip-cluster percentile bootstrap, B 1000, seed 0, 141 TRAIN clips (another draw of clips; trained WM and fitted readouts held fixed).
Usage: python amend1_kstep.py <k> <out.json> <out_readout.pt>
"""
import dataclasses, json, sys, time
import numpy as np
import torch

torch.set_num_threads(8)
sys.path.insert(0, "/root/actrsp_amend/stack")
from tanitad.models import v8f_mini as M  # noqa: E402

K_CH, OUT, OUT_PT = int(sys.argv[1]), sys.argv[2], sys.argv[3]
TAG = f"R35_s{K_CH}"; KS = 10; T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


ck = torch.load(f"/root/actrsp_amend/data/ckpt_{TAG}.pt", map_location="cpu", weights_only=False)
names = {f.name for f in dataclasses.fields(M.V8FMiniConfig)}
cfg = M.V8FMiniConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items() if k in names})
torch.manual_seed(0)
wm = M.WMAux(cfg)
wm.load_state_dict({k[3:]: v for k, v in ck["model"].items() if k.startswith("wm.")}, strict=True)
wm.eval(); st = wm.stack
for p in wm.parameters():
    p.requires_grad_(False)
del ck
L = torch.load(f"/root/actrsp_amend/data/latents_train_{TAG}.pt", map_location="cpu", weights_only=False)
Zw, Aw, okw = L["Z1"][TAG].float(), wm.scale_actions(L["A_raw"].float()), L["ok"].bool()
tw, eidw = np.asarray(L["t"]), list(L["eid"])
clips = sorted(set(eidw)); cix = {c: i for i, c in enumerate(clips)}
tab = {}
for w in range(Zw.shape[0]):
    for j in range(Zw.shape[1]):
        if bool(okw[w, j]):
            tab.setdefault((cix[eidw[w]], int(tw[w]) - (Zw.shape[1] - 1) + j), (Zw[w, j], Aw[w, j]))
seq = {}
for ci in range(len(clips)):
    fr = sorted(f for (c, f) in tab if c == ci)
    seq[ci] = (np.array(fr), torch.stack([tab[(ci, f)][0] for f in fr]), torch.stack([tab[(ci, f)][1] for f in fr]))
EP, ZT, ZK, AB, FR, ANC = [], [], [], [], [], []
for ci, (fr, Z, A) in seq.items():
    pos = {int(f): i for i, f in enumerate(fr)}
    for i, f in enumerate(fr):
        f = int(f)
        if all((f + m) in pos for m in range(KS + 1)):
            js = [pos[f + m] for m in range(KS)]
            if torch.isfinite(A[js]).all():
                EP.append(ci); ZT.append(Z[i]); ZK.append(Z[pos[f + KS]]); AB.append(A[js].mean(0)); FR.append(f); ANC.append(A[js])
EP, FR = np.array(EP), np.array(FR)
zt, zk, abar, aseq = torch.stack(ZT), torch.stack(ZK), torch.stack(AB), torch.stack(ANC)   # aseq (n, K, 2)
n = len(EP)
amean = aseq.reshape(-1, 2).mean(0)                                            # the mean logged action over every K-step stretch (all finite by construction)
with torch.no_grad():
    z_wm = st.rollout(1, zt, amean.expand(n, KS, 2).contiguous(), context=1)[:, -1]
    z_true = st.rollout(1, zt, aseq, context=1)[:, -1]
rng = np.random.RandomState(0)
part = np.empty(n, dtype=int)
groups = {}
for i, e in enumerate(EP):
    groups.setdefault(int(e), []).append(i)
for e, g in groups.items():
    g = np.array(g)
    for i in g:
        cand = g[np.abs(FR[g] - FR[i]) >= 5]
        part[i] = rng.choice(cand) if len(cand) else i
Zt, Zk, Zwm, Ztr = (x.numpy().astype(np.float64) for x in (zt, zk, z_wm, z_true))
Y = abar.numpy().astype(np.float64)
Dr, Dsw, Dwm, Dtr = Zk - Zt, Zt + (Zk[part] - Zt[part]), Zwm, Ztr       # second inputs: real, swap (grafted), WM mean-action, WM true-action endpoints
fold = (np.random.RandomState(0).permutation(len(clips)) % 5)[EP]
grp = [np.array(v) for v in groups.values()]
res = {"tag": TAG, "K": KS, "n_pairs": int(n), "n_clips": len(grp), "a_mean_wm": amean.tolist(), "code": "a13f6f16 (hjepa 0cfd7081, v8f_mini f15c28c9)",
       "estimator": "clip-cluster percentile bootstrap, B 1000, seed 0, 141 TRAIN clips; answers another draw of clips, WM and fitted readouts held fixed"}

LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def ridge_fit(X, Yy, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd; ym = Yy.mean(0)
    W = np.linalg.solve(Xs.T @ Xs + lam * len(X) * np.eye(X.shape[1]), Xs.T @ (Yy - ym))
    return mu, sd, W, ym


def ridge_pred(m, X):
    return ((X - m[0]) / m[1]) @ m[2] + m[3]


def ridge_lam(X, Yy, fo):
    best, bl = np.inf, None
    for lam in LAMS:
        err = sum(((ridge_pred(ridge_fit(X[fo != g], Yy[fo != g], lam), X[fo == g]) - Yy[fo == g]) ** 2).sum() for g in np.unique(fo))
        if err < best:
            best, bl = err, lam
    return bl


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


# cross-fitted predictions on real and counterfactual inputs, per readout
H = {k: np.zeros(n) for k in ("RA_real", "RA_swap", "RA_wm", "RA_true", "RY_real", "RY_swap", "RY_wm", "RY_true", "SA_ridge", "SY_ridge", "SA_mlp", "SY_mlp")}
lam_A, lam_S = [], []
for f in range(5):
    tr, te = fold != f, fold == f
    la = ridge_lam(Dr[tr], Y[tr, :1], fold[tr]); lam_A.append(la)
    mA = ridge_fit(Dr[tr], Y[tr, :1], la)
    H["RA_real"][te] = ridge_pred(mA, Dr[te])[:, 0]; H["RA_swap"][te] = ridge_pred(mA, (Dsw - Zt)[te])[:, 0]
    H["RA_wm"][te] = ridge_pred(mA, (Dwm - Zt)[te])[:, 0]; H["RA_true"][te] = ridge_pred(mA, (Dtr - Zt)[te])[:, 0]
    mY = mlp_fit(np.concatenate([Zt, Zk], 1)[tr], Y[tr, 1:], seed=f)
    H["RY_real"][te] = mlp_pred(mY, np.concatenate([Zt, Zk], 1)[te])[:, 0]; H["RY_swap"][te] = mlp_pred(mY, np.concatenate([Zt, Dsw], 1)[te])[:, 0]
    H["RY_wm"][te] = mlp_pred(mY, np.concatenate([Zt, Dwm], 1)[te])[:, 0]; H["RY_true"][te] = mlp_pred(mY, np.concatenate([Zt, Dtr], 1)[te])[:, 0]
    ls = ridge_lam(Zt[tr], Y[tr], fold[tr]); lam_S.append(ls)
    mS = ridge_fit(Zt[tr], Y[tr], ls); pS = ridge_pred(mS, Zt[te]); H["SA_ridge"][te], H["SY_ridge"][te] = pS[:, 0], pS[:, 1]
    pM = mlp_pred(mlp_fit(Zt[tr], Y[tr], seed=10 + f), Zt[te]); H["SA_mlp"][te], H["SY_mlp"][te] = pM[:, 0], pM[:, 1]
    log("fold", f, "lam_A", la, "lam_S", ls)
res["lambda_RA"], res["lambda_S"] = lam_A, lam_S


def r2(p, y):
    return 1 - ((p - y) ** 2).mean() / ((y - y.mean()) ** 2).mean()


def cv(x, y):
    return ((x - x.mean()) * (y - y.mean())).mean()


def boot(fn, B=1000):
    pt = np.asarray(fn(np.arange(n)), float)
    r = np.random.RandomState(0); bs = np.empty((B, pt.size))
    for b in range(B):
        bs[b] = fn(np.concatenate([grp[g] for g in r.randint(0, len(grp), len(grp))]))
    lo, hi = np.percentile(bs, [2.5, 97.5], 0)
    return pt, lo, hi


def stats(ii):
    out = []
    for c, R in ((0, "RA"), (1, "RY")):
        y, yp = Y[ii, c], Y[part[ii], c]
        real = H[f"{R}_real"][ii]
        sA = "SA" if c == 0 else "SY"
        best_s = max(r2(H[f"{sA}_ridge"][ii], y), r2(H[f"{sA}_mlp"][ii], y))
        d = y - yp
        out += [r2(real, y), r2(H[f"{sA}_ridge"][ii], y), r2(H[f"{sA}_mlp"][ii], y), r2(real, y) - best_s,
                cv(real, y) / y.var(), cv(real - H[f"{R}_swap"][ii], d) / d.var(), cv(real - H[f"{R}_wm"][ii], y) / y.var(),
                r2(H[f"{R}_true"][ii], y), cv(H[f"{R}_true"][ii] - H[f"{R}_wm"][ii], y) / y.var()]
    return np.array(out)


nm = [f"{m}_{c}" for c in ("acc", "yaw") for m in ("r2_readout", "r2_state_ridge", "r2_state_mlp", "dR2_vs_best_state", "s_full", "s_swap", "s_wm", "r2_wm_true_rollout", "s_wm_rollout_true_minus_mean")]
pt, lo, hi = boot(stats)
res["heldout"] = {k: {"point": float(pt[i]), "ci95": [float(lo[i]), float(hi[i])]} for i, k in enumerate(nm)}
log("heldout", {k: (round(v["point"], 4), [round(x, 4) for x in v["ci95"]]) for k, v in res["heldout"].items()})

# frozen readouts (fit on all 141 TRAIN clips) + the term preview at R35's end
laF = ridge_lam(Dr, Y[:, :1], fold); mA = ridge_fit(Dr, Y[:, :1], laF)
mY = mlp_fit(np.concatenate([Zt, Zk], 1), Y[:, 1:], seed=0)
s_cal = np.array([res["heldout"]["s_swap_acc"]["point"], res["heldout"]["s_swap_yaw"]["point"]])
torch.save({"tag": TAG, "K": KS, "lambda": laF, "RA": {"mu": mA[0], "sd": mA[1], "W": mA[2], "ym": mA[3]}, "RY_state_dict": mY[0].state_dict(), "RY_mu": mY[1], "RY_sd": mY[2],
            "s_cal_swap_heldout": s_cal.tolist()}, OUT_PT)
anc = [i for i in range(n) if (FR[i] - seq[EP[i]][0][0]) % KS == 0]
anc = np.array(anc)
with torch.no_grad():
    dl = torch.tensor(M.RSP_DELTA, dtype=torch.float32)
    e = torch.zeros(4, 2); e[0, 0], e[1, 0], e[2, 1], e[3, 1] = 1, -1, 1, -1
    a4 = aseq[anc][None] + (e * dl)[:, None, None, :]
    z0 = zt[anc]
    zend = st.rollout(1, z0[None].expand(4, -1, -1), a4, context=1)[:, :, -1].numpy().astype(np.float64)   # (4, na, D)
Z0 = Zt[anc]
RA = np.stack([ridge_pred(mA, zend[c] - Z0)[:, 0] for c in range(4)])
RY = np.stack([mlp_pred(mY, np.concatenate([Z0, zend[c]], 1))[:, 0] for c in range(4)])
d = np.array(M.RSP_DELTA, dtype=np.float64)
gl = np.stack([(RA[0] - RA[1]) / (2 * d[0]), (RY[0] - RY[1]) / (2 * d[0])], 1)      # (na, c'): read change per unit acc command
gt = np.stack([(RA[2] - RA[3]) / (2 * d[1]), (RY[2] - RY[3]) / (2 * d[1])], 1)      # per unit yaw command
ga = {}
agrp = {}
for j, i in enumerate(anc):
    agrp.setdefault(int(EP[i]), []).append(j)
agl = [np.array(v) for v in agrp.values()]


def gfn(jj):
    G = np.stack([gl[jj] / s_cal, gt[jj] / s_cal], -1)      # (na, c', c)
    return np.array([G[:, 0, 0].mean(), G[:, 1, 1].mean(), G[:, 0, 1].mean(), G[:, 1, 0].mean(), ((G - np.eye(2)) ** 2).mean(), ((G[:, 0, 0] - 1) ** 2).mean()])


ptg = np.asarray(gfn(np.arange(len(anc))))
r = np.random.RandomState(0); bs = np.array([gfn(np.concatenate([agl[g] for g in r.randint(0, len(agl), len(agl))])) for _ in range(1000)])
lo, hi = np.percentile(bs, [2.5, 97.5], 0)
res["term_preview_R35_end"] = {"n_anchors": int(len(anc)), "s_cal_used": s_cal.tolist(), "lambda_frozen_RA": laF,
                               **{k: {"point": float(ptg[i]), "ci95": [float(lo[i]), float(hi[i])]} for i, k in enumerate(("G_aa", "G_ww", "G_aw_accread_yawcmd", "G_wa_yawread_acccmd", "rsp_loss_2ch", "rsp_loss_acc_only"))}}
log("term", json.dumps(res["term_preview_R35_end"]))
res["wall_s"] = time.time() - T0
json.dump(res, open(OUT, "w"), indent=1)
log("ZZ KSTEP DONE ZZ")
