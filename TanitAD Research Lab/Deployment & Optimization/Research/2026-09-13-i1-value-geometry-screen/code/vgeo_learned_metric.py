"""E-DEP-VGEO-2 — learned metric over frozen tokens (see ../PREREG_ADDENDUM_L2_LEARNED_METRIC.md)."""
import json, os, sys, time
import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

BANK = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens"
OUT = sys.argv[2] if len(sys.argv) > 2 else "raw/vgeo_learned_metric.json"
STEP, DMIN, DMAX, SHORT, NBOOT, SEED, K = 4, 4, 160, 60, 2000, 20260913, 64
try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception as e:
    print("priority not lowered:", e)
t0 = time.time()
idx = np.load(os.path.join(BANK, "index.npz"), allow_pickle=True)
clip, rawf = idx["clip_ordinal"], idx["raw_frame"]
tok = np.load(os.path.join(BANK, "tokens_s32_fp16.npy"), mmap_mode="r")
pix = np.load(os.path.join(BANK, "pix64_u8.npy"), mmap_mode="r")
rows = []
for c in np.unique(clip):
    r = np.where(clip == c)[0]; r = r[np.argsort(rawf[r])]; rows.extend(r[::STEP].tolist())
rows = np.array(sorted(rows))
feats = {"latent_704": np.empty((len(rows), 704), np.float32), "latent_7040": np.empty((len(rows), 7040), np.float32),
         "pixel_1440": np.empty((len(rows), 1440), np.float32)}
for i in range(0, len(rows), 256):
    rr = rows[i:i + 256]; n = len(rr)
    t = np.asarray(tok[rr], dtype=np.float32)
    feats["latent_704"][i:i + n] = t.mean(axis=(2, 3))
    feats["latent_7040"][i:i + n] = t.reshape(n, 704, 2, 4, 5, 4).mean(axis=(3, 5)).reshape(n, -1)
    p = np.asarray(pix[rr], dtype=np.float32) / 255.0
    feats["pixel_1440"][i:i + n] = p.reshape(n, 9, 8, 8, 20, 8).mean(axis=(3, 5)).reshape(n, -1)
rclip, rframe = clip[rows], rawf[rows]
clips = np.unique(rclip)
rng = np.random.default_rng(SEED)
perm = rng.permutation(clips)
fit_clips, sc_clips = np.sort(perm[:97]), np.sort(perm[97:])
assert len(np.intersect1d(fit_clips, sc_clips)) == 0 and len(sc_clips) == 42


def pairs(c):
    m = np.where(rclip == c)[0]; fr = rframe[m]
    i, j = np.triu_indices(len(m), 1); d = fr[j] - fr[i]
    k = (d >= DMIN) & (d <= DMAX)
    return m[i[k]], m[j[k]], d[k]


def fit_diag(F):
    I, J, D = [], [], []
    for c in fit_clips:
        i, j, d = pairs(c); I.append(i); J.append(j); D.append(d)
    I, J, D = np.concatenate(I), np.concatenate(J), np.concatenate(D)
    cap = min(200_000, 40_000_000 // F.shape[1])   # host-RAM cap (live MM arm): <=~320 MB float64 design matrix
    if len(D) > cap:   # subsample pair INDICES first so the full design matrix is never materialised
        s = rng.choice(len(D), cap, replace=False); I, J, D = I[s], J[s], D[s]
    A, y = ((F[I] - F[J]) ** 2).astype(np.float64), np.log(D)
    mu, sd = A.mean(0), A.std(0) + 1e-8          # column scaling fitted on FIT only
    w, _ = nnls(np.column_stack([(A - mu) / sd, np.ones(len(y))]), y)   # intercept unconstrained-ish (>=0 ok: log d>0)
    return w[:-1] / sd, int(len(y))


def fit_pca(F):
    fm = np.isin(rclip, fit_clips)
    mu = F[fm].mean(0); U, S, Vt = np.linalg.svd(F[fm] - mu, full_matrices=False)
    P = Vt[:K] / (S[:K, None] / np.sqrt(fm.sum()) + 1e-8)
    return mu, P


def rho_clips(dist_fn, mode="real", which=sc_clips):
    out = {"short": [], "long": [], "all": []}
    for c in which:
        i, j, d = pairs(c)
        if mode == "shuf":
            m = np.where(rclip == c)[0]; j = rng.choice(m, size=len(j))
        if mode == "mut":
            m = np.where(rclip == c)[0]; mp = dict(zip(m, rng.permutation(m))); i = np.array([mp[x] for x in i]); j = np.array([mp[x] for x in j])
        dist = np.zeros(len(i)) if mode == "const" else dist_fn(i, j)
        for b, s in (("short", d <= SHORT), ("long", d > SHORT), ("all", np.ones_like(d, bool))):
            if s.sum() < 5 or np.all(dist[s] == dist[s][0]):
                out[b].append(0.0 if mode == "const" else np.nan)
            else:
                out[b].append(spearmanr(dist[s], d[s]).correlation)
    return {k: np.array(v) for k, v in out.items()}


def boot(a, b=None):
    n = len(a); pt = float(np.nanmean(a if b is None else a - b)); bs = []
    for _ in range(NBOOT):
        s = rng.integers(0, n, n); bs.append(np.nanmean(a[s] if b is None else a[s] - b[s]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"point": round(pt, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n_clips": int(n)}


arms, meta = {}, {}
for name, F in feats.items():
    arms[name + "|raw"] = rho_clips(lambda i, j, F=F: np.linalg.norm(F[i] - F[j], axis=1))
    w, npairs = fit_diag(F)
    meta[name + "|diag"] = {"fit_pairs": npairs, "nonzero_w": int((w > 0).sum()), "dims": int(F.shape[1])}
    arms[name + "|diag"] = rho_clips(lambda i, j, F=F, w=w: np.sqrt(((F[i] - F[j]) ** 2 * w).sum(1)))
    mu, P = fit_pca(F)
    G = (F - mu) @ P.T
    arms[name + "|pca64"] = rho_clips(lambda i, j, G=G: np.linalg.norm(G[i] - G[j], axis=1))
    if name == "latent_704":
        for mode in ("const", "shuf", "mut"):
            arms["C_" + mode + "|latent_704_diag"] = rho_clips(lambda i, j, F=F, w=w: np.sqrt(((F[i] - F[j]) ** 2 * w).sum(1)), mode)
    print(name, "done", round(time.time() - t0, 1))

res = {"prereg": "PREREG_ADDENDUM_L2_LEARNED_METRIC.md", "rows": int(len(rows)), "fit_clips": int(len(fit_clips)),
       "scored_clips": int(len(sc_clips)), "scored_clip_ordinals": sc_clips.tolist(), "k_pca": K, "fit_meta": meta,
       "estimator": "clip-cluster bootstrap over SCORED clips, mean of per-clip Spearman rho, 95% percentile",
       "arms": {k: {b: boot(v[b]) for b in v} for k, v in arms.items()}, "paired": {}}
for lat in ("latent_704", "latent_7040"):
    for m in ("raw", "diag", "pca64"):
        res["paired"][f"{lat}|{m}_minus_pixel_1440|{m}"] = {b: boot(arms[f"{lat}|{m}"][b], arms[f"pixel_1440|{m}"][b]) for b in ("short", "long", "all")}
res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(OUT, "w"), indent=1)
for k, v in res["arms"].items():
    print(k, v["short"]["point"], v["short"]["ci95"])
for k, v in res["paired"].items():
    print(k, v["short"]["point"], v["short"]["ci95"])
