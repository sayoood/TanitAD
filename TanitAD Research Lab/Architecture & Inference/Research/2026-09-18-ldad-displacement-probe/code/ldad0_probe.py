"""E-AI-LDAD-0 -- does the FROZEN trunk's latent DISPLACEMENT carry the ego action beyond pixels?

Transfer test of Delta-JEPA (arXiv 2606.31232, LDAD: decode a_t from dz = z_{t+k} - z_t).
Our trunk is frozen (AI13-2), so encoder-side LDAD cannot act; this measures whether the frozen
displacement geometry is ALREADY action-organised (=> any LDAD lever is predictor-side) or not.
See ../SPEC.md (pinned before run). CPU only. argv: OUT.

Inputs are VISION ONLY (tokens / pixels). Targets use ego (label derivation; admissible).
Targets over a K-frame gap (10 Hz): LON a = (v[t+K]-v[t])/(K*0.1) m/s^2 ; LAT = mean yaw rate on [t, t+K) rad/s.
Ridge, lambda selected by clip-grouped 5-fold CV on the FIT split only; scored split never tuned on.
Score: R2 = 1 - sum SSE / sum SST, SST about the FIT-split target mean (so C-const reads exactly 0).
Clip-cluster bootstrap over scored clips, paired for differences.
"""
import json, os, sys, time
import numpy as np

ROOT = "C:/Users/Admin/tanitad-caches/bevhead-20260913"
BANK, GT = ROOT + "/tokens", ROOT + "/bev_gt"
OUT = sys.argv[1] if len(sys.argv) > 1 else "raw/ldad0.json"
STEP, K, SEED, NBOOT = 4, 8, 20260913, 2000
LAMBDAS = [10.0 ** e for e in range(-2, 7)]
try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception as e:
    print("priority not lowered:", e)
t0 = time.time()
idx = np.load(os.path.join(BANK, "index.npz"), allow_pickle=True)
clip, rawf, sha = idx["clip_ordinal"], idx["raw_frame"], idx["clip_sha12"]
tok = np.load(os.path.join(BANK, "tokens_s32_fp16.npy"), mmap_mode="r")
pix = np.load(os.path.join(BANK, "pix64_u8.npy"), mmap_mode="r")

V, Y = {}, {}
for c in np.unique(clip):
    f = os.path.join(GT, str(sha[c]) + ".bevgt.npz")
    if not os.path.exists(f):
        continue
    g = np.load(f, allow_pickle=True)
    if json.loads(str(g["meta_json"]))["stats"].get("deskew_source") != "egomotion_alpamayo":
        continue
    rf, ok = g["raw_frame"].astype(int), g["label_valid"].astype(bool)
    v, w = g["ego_v_ms"].astype(np.float64), g["ego_yaw_rate_rps"].astype(np.float64)
    if (~ok).any():
        v[~ok] = np.interp(rf[~ok], rf[ok], v[ok]); w[~ok] = np.interp(rf[~ok], rf[ok], w[ok])
    fv, fw = np.full(rf.max() + 1, np.nan), np.full(rf.max() + 1, np.nan)
    fv[rf], fw[rf] = v, w
    V[int(c)], Y[int(c)] = fv, fw
keep = np.array(sorted(V))
rng = np.random.default_rng(SEED)
perm = rng.permutation(keep)            # identical split to VGEO-3/4/5
nfit = int(round(0.7 * len(perm)))
fit_c, sc_c = np.sort(perm[:nfit]), np.sort(perm[nfit:])
assert len(np.intersect1d(fit_c, sc_c)) == 0

# (start row, end row) pairs at gap K within clip, on the STEP grid
S, E, C, A = [], [], [], []
for c in keep:
    r = np.where(clip == c)[0]; r = r[np.argsort(rawf[r])][::STEP]
    fr = rawf[r].astype(int); pos = {int(f): i for i, f in enumerate(fr)}
    for i, f in enumerate(fr):
        f = int(f); j = pos.get(f + K)
        if j is None or f + K >= len(V[c]):
            continue
        v0, v1 = V[c][f], V[c][f + K]; ww = Y[c][f:f + K]
        if not (np.isfinite(v0) and np.isfinite(v1) and np.isfinite(ww).all()):
            continue
        S.append(r[i]); E.append(r[j]); C.append(c); A.append([(v1 - v0) / (K * 0.1), ww.mean()])
S, E, C, A = map(np.array, (S, E, C, A))
print("pairs", len(S), "fit/scored clips", len(fit_c), len(sc_c), round(time.time() - t0, 1), flush=True)

rows = np.unique(np.concatenate([S, E])); where = {int(r): i for i, r in enumerate(rows)}
Z = np.empty((len(rows), 704), np.float32); P = np.empty((len(rows), 1440), np.float32)
for i in range(0, len(rows), 256):
    rr = rows[i:i + 256]; n = len(rr)
    Z[i:i + n] = np.asarray(tok[rr], dtype=np.float32).mean(axis=(2, 3))
    p = np.asarray(pix[rr], dtype=np.float32) / 255.0
    P[i:i + n] = p.reshape(n, 9, 8, 8, 20, 8).mean(axis=(3, 5)).reshape(n, -1)
si, ei = np.array([where[int(s)] for s in S]), np.array([where[int(e)] for e in E])
content = {"Z_std": float(Z.std()), "P_std": float(P.std()), "finite": bool(np.isfinite(Z).all() and np.isfinite(P).all()),
           "LON_std": float(A[:, 0].std()), "LAT_std": float(A[:, 1].std())}
assert content["Z_std"] > 0 and content["P_std"] > 0 and content["finite"], content

gen = np.random.default_rng(SEED + 11)
feats = {
    "latent_dz": Z[ei] - Z[si],
    "latent_concat": np.hstack([Z[si], Z[ei]]),
    "latent_start_only": Z[si],                     # no displacement: what the scene alone predicts
    "pixel_dp": P[ei] - P[si],
    "pixel_concat": np.hstack([P[si], P[ei]]),
}
fitm, scm = np.isin(C, fit_c), np.isin(C, sc_c)


def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd; ym = y.mean()
    n, d = Xs.shape
    if d <= n:
        w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(d), Xs.T @ (y - ym))
    else:
        w = Xs.T @ np.linalg.solve(Xs @ Xs.T + lam * np.eye(n), y - ym)
    return lambda Xn: ((Xn - mu) / sd) @ w + ym


def select_lam(X, y, groups):
    ug = np.unique(groups); folds = np.array_split(np.random.default_rng(SEED + 3).permutation(ug), 5)
    best = None
    for lam in LAMBDAS:
        sse = 0.0
        for fo in folds:
            te = np.isin(groups, fo)
            sse += float(((ridge_fit(X[~te], y[~te], lam)(X[te]) - y[te]) ** 2).sum())
        if best is None or sse < best[1]:
            best = (lam, sse)
    return best[0]


def r2_boot(err, sst):
    cl = C[scm]; uc = np.unique(cl)
    sse_c = np.array([err[cl == c].sum() for c in uc]); sst_c = np.array([sst[cl == c].sum() for c in uc])
    pt = 1 - sse_c.sum() / sst_c.sum()
    b = []
    for _ in range(NBOOT):
        k = gen.integers(0, len(uc), len(uc)); b.append(1 - sse_c[k].sum() / sst_c[k].sum())
    return pt, np.percentile(b, [2.5, 97.5]), sse_c, sst_c


def rec(pt, ci):
    return {"R2": round(float(pt), 4), "ci95": [round(float(ci[0]), 4), round(float(ci[1]), 4)]}


res = {"spec": "SPEC.md E-AI-LDAD-0", "K_frames": K, "step": STEP, "pairs": int(len(S)),
       "fit_pairs": int(fitm.sum()), "scored_pairs": int(scm.sum()), "fit_clips": int(len(fit_c)),
       "scored_clips": int(len(sc_c)), "content": content, "targets": {}}
for ti, tname in enumerate(["LON_accel_ms2", "LAT_yawrate_rps"]):
    y = A[:, ti]; mu_fit = y[fitm].mean(); ys = y[scm]
    sst = (ys - mu_fit) ** 2
    arms, percl = {}, {}
    for name, X in feats.items():
        lam = select_lam(X[fitm], y[fitm], C[fitm])
        e = (ridge_fit(X[fitm], y[fitm], lam)(X[scm]) - ys) ** 2
        pt, ci, sse_c, sst_c = r2_boot(e, sst)
        arms[name] = dict(rec(pt, ci), **{"lambda": lam, "d": int(X.shape[1]), "n_fit": int(fitm.sum())})
        percl[name] = (sse_c, sst_c)
    e0 = (np.full_like(ys, mu_fit) - ys) ** 2
    pt, ci, _, _ = r2_boot(e0, sst); arms["C_const"] = rec(pt, ci)
    Xo = (y + gen.normal(0, 0.01 * y.std(), len(y)))[:, None]
    lam = select_lam(Xo[fitm], y[fitm], C[fitm])
    pt, ci, _, _ = r2_boot((ridge_fit(Xo[fitm], y[fitm], lam)(Xo[scm]) - ys) ** 2, sst); arms["C_oracle"] = rec(pt, ci)
    Eshuf = E.copy()
    for c in np.unique(C):
        m = np.where(C == c)[0]; Eshuf[m] = E[m][gen.permutation(len(m))]
    eis = np.array([where[int(e)] for e in Eshuf]); Xs_ = Z[eis] - Z[si]
    lam = select_lam(Xs_[fitm], y[fitm], C[fitm])
    pt, ci, _, _ = r2_boot((ridge_fit(Xs_[fitm], y[fitm], lam)(Xs_[scm]) - ys) ** 2, sst); arms["C_shuffle_latent_dz"] = rec(pt, ci)
    paired = {}
    for a, b in (("latent_dz", "pixel_dp"), ("latent_concat", "pixel_concat"), ("latent_dz", "latent_start_only")):
        sa, ta = percl[a]; sb, tb = percl[b]; d = []
        for _ in range(NBOOT):
            k = gen.integers(0, len(sa), len(sa))
            d.append((1 - sa[k].sum() / ta[k].sum()) - (1 - sb[k].sum() / tb[k].sum()))
        paired[a + "_minus_" + b] = {"point": round(arms[a]["R2"] - arms[b]["R2"], 4),
                                     "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
    res["targets"][tname] = {"y_std_fit": round(float(y[fitm].std()), 5), "arms": arms, "paired": paired}
    print(tname, json.dumps(res["targets"][tname]), flush=True)
res["seconds"] = round(time.time() - t0, 1)
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print("done", res["seconds"])
