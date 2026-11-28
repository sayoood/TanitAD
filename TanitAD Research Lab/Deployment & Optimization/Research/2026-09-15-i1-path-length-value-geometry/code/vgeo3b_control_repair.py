"""E-DEP-VGEO-3b — control repair of VGEO-3 (see ../PREREG_ADDENDUM_CONTROL_REPAIR.md; SPEC.md otherwise unchanged).

CPU only, below-normal priority. argv: OUT STEP.
"""
import json, os, sys, time
import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

ROOT = "C:/Users/Admin/tanitad-caches/bevhead-20260913"
BANK, GT = ROOT + "/tokens", ROOT + "/bev_gt"
OUT = sys.argv[1] if len(sys.argv) > 1 else "raw/vgeo3_pathlen.json"
STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NBOOT, SEED, P2TOL, P2MIN = 2000, 20260913, 0.02, 20
DELTAS = list(range(12, 61, 4))
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

# ---- speed join -------------------------------------------------------------------------------
speed, excluded = {}, {"no_bev_gt": [], "no_egomotion": []}
n_interp = 0
for c in np.unique(clip):
    f = os.path.join(GT, f"{sha[c]}.bevgt.npz")
    if not os.path.exists(f):
        excluded["no_bev_gt"].append(str(sha[c])); continue
    g = np.load(f, allow_pickle=True)
    meta = json.loads(str(g["meta_json"]))
    if meta["stats"].get("deskew_source") != "egomotion_alpamayo":
        excluded["no_egomotion"].append(str(sha[c])); continue
    rf, v, ok = g["raw_frame"].astype(int), g["ego_v_ms"].astype(np.float64), g["label_valid"].astype(bool)
    vv = v.copy()
    if (~ok).any():
        vv[~ok] = np.interp(rf[~ok], rf[ok], v[ok]); n_interp += int((~ok).sum())
    full = np.zeros(rf.max() + 1); full[rf] = vv
    speed[int(c)] = full
keep_clips = np.array(sorted(speed))
print("clips kept", len(keep_clips), "excluded", {k: len(v) for k, v in excluded.items()}, "interp frames", n_interp)

rows = []
for c in keep_clips:
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
content = {k: {"std": float(v.std()), "finite": bool(np.isfinite(v).all())} for k, v in feats.items()}
assert all(c["std"] > 0 and c["finite"] for c in content.values()), content
rclip, rframe = clip[rows], rawf[rows]
rng = np.random.default_rng(SEED)
perm = rng.permutation(keep_clips)
nfit = int(round(0.7 * len(perm)))
fit_clips, sc_clips = np.sort(perm[:nfit]), np.sort(perm[nfit:])
assert len(np.intersect1d(fit_clips, sc_clips)) == 0

cum = {c: np.concatenate([[0.0], np.cumsum(speed[c] * 0.1)]) for c in keep_clips}   # cum[c][k] = path to frame k
vshift = {}   # C-vswap: speed series of a DIFFERENT scored clip (seeded derangement), truncated / edge-padded
while True:
    der = rng.permutation(sc_clips)
    if not np.any(der == sc_clips):
        break
for c, o in zip(sc_clips, der):
    T, so = len(speed[c]), speed[o]
    vs = so[:T] if len(so) >= T else np.concatenate([so, np.full(T - len(so), so[-1])])
    vshift[c] = np.concatenate([[0.0], np.cumsum(vs * 0.1)])


def pathlen(c, fs, fg, table=cum):
    return table[c][fg] - table[c][fs]


def pairs(c, lo, hi):
    m = np.where(rclip == c)[0]; fr = rframe[m]
    i, j = np.triu_indices(len(m), 1); d = fr[j] - fr[i]
    k = (d >= lo) & (d <= hi)
    return m[i[k]], m[j[k]], d[k]


def fit_diag(F):
    I, J, L = [], [], []
    for c in fit_clips:
        i, j, d = pairs(c, 4, 60); l = pathlen(c, rframe[i], rframe[j]); k = l >= 0.5
        I.append(i[k]); J.append(j[k]); L.append(l[k])
    I, J, L = np.concatenate(I), np.concatenate(J), np.concatenate(L)
    cap = min(200_000, 40_000_000 // F.shape[1])
    if len(L) > cap:
        s = rng.choice(len(L), cap, replace=False); I, J, L = I[s], J[s], L[s]
    A, y = ((F[I] - F[J]) ** 2).astype(np.float64), np.log(L)
    mu, sd = A.mean(0), A.std(0) + 1e-8
    w, _ = nnls(np.column_stack([(A - mu) / sd, np.ones(len(y))]), y)
    return w[:-1] / sd, int(len(y))


def p1(dist_fn, mode="real"):
    """within-Delta Spearman(dist, pathlen), nanmean over Delta per clip."""
    out, ndef = [], 0
    for c in sc_clips:
        m = np.where(rclip == c)[0]
        mp = dict(zip(m, rng.permutation(m))) if mode == "mut" else None
        vals = []
        for D in DELTAS:
            i, j, d = pairs(c, D, D)
            if len(i) < 8:
                continue
            l = pathlen(c, rframe[i], rframe[j], vshift if mode == "vshift" else cum)
            if l.std() / max(l.mean(), 1e-9) < 0.05:
                continue
            if mode == "mut":
                i = np.array([mp[x] for x in i]); j = np.array([mp[x] for x in j])
            dist = np.zeros(len(i)) if mode == "const" else dist_fn(i, j)
            vals.append(0.0 if np.all(dist == dist[0]) else spearmanr(dist, l).correlation)
        out.append(np.nanmean(vals) if vals else np.nan)
        ndef += bool(vals)
    return np.array(out), ndef


def p3(dist_fn):
    out = []
    for c in sc_clips:
        i, j, d = pairs(c, 4, 60); l = pathlen(c, rframe[i], rframe[j])
        dist = dist_fn(i, j)
        out.append(np.nan if l.std() == 0 else spearmanr(dist, l).correlation)
    return np.array(out)


# P2 triplets, built once from raw pixels (matched distance), shared by every arm
Xp = feats["pixel_1440"]
sc_rows = np.where(np.isin(rclip, sc_clips))[0]
trip, p2kept = {}, {}
for c in sc_clips:
    i, j, d = pairs(c, 12, 60)
    if len(i) == 0:
        continue
    s = rng.choice(len(i), min(200, len(i)), replace=False); i, j = i[s], j[s]
    pool = sc_rows[rclip[sc_rows] != c]
    dtrue = np.linalg.norm(Xp[i] - Xp[j], axis=1)
    dd = np.empty(len(i), int); resid = np.empty(len(i))
    for b in range(0, len(i), 25):
        dp = np.linalg.norm(Xp[i[b:b + 25]][:, None, :] - Xp[pool][None, :, :], axis=2)
        k = np.argmin(np.abs(dp - dtrue[b:b + 25, None]), axis=1)
        dd[b:b + 25] = pool[k]; resid[b:b + 25] = np.abs(dp[np.arange(len(k)), k] - dtrue[b:b + 25]) / np.maximum(dtrue[b:b + 25], 1e-9)
    ok = resid <= P2TOL
    p2kept[int(c)] = int(ok.sum())
    if ok.sum() >= P2MIN:
        trip[int(c)] = (i[ok], j[ok], dd[ok])


def p2(dist_fn, mode="real"):
    out = []
    for c in sc_clips:
        if int(c) not in trip:
            out.append(np.nan); continue
        i, j, dd = trip[int(c)]
        if mode == "const":
            out.append(0.5); continue
        a, b = dist_fn(i, j), dist_fn(i, dd)
        out.append(float(np.mean((a < b) + 0.5 * (a == b))))
    return np.array(out)


def boot(a, b=None):
    x = a if b is None else a - b
    ok = ~np.isnan(x); x = x[ok]; n = len(x)
    bs = [np.mean(x[rng.integers(0, n, n)]) for _ in range(NBOOT)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"point": round(float(np.mean(x)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n_clips": int(n)}


arms, meta = {}, {}
for name, F in feats.items():
    raw = lambda i, j, F=F: np.linalg.norm(F[i] - F[j], axis=1)
    w, npairs = fit_diag(F)
    diag = lambda i, j, F=F, w=w: np.sqrt(((F[i] - F[j]) ** 2 * w).sum(1))
    meta[name] = {"fit_pairs": npairs, "nonzero_w": int((w > 0).sum()), "dims": int(F.shape[1])}
    for mname, fn in (("raw", raw), ("diag", diag)):
        a1, nd = p1(fn)
        arms[f"{name}|{mname}"] = {"P1": a1, "P2": p2(fn), "P3": p3(fn)}
        meta[f"{name}|{mname}"] = {"P1_defined_clips": nd}
    if name == "latent_7040":
        arms["C_const"] = {"P1": p1(diag, "const")[0], "P2": p2(diag, "const")}
        arms["C_vswap|latent_7040_diag"] = {"P1": p1(diag, "vshift")[0]}
        arms["C_mut|latent_7040_diag"] = {"P1": p1(diag, "mut")[0]}
    print(name, "done", round(time.time() - t0, 1), flush=True)

res = {"spec": "SPEC.md E-DEP-VGEO-3 + PREREG_ADDENDUM_CONTROL_REPAIR.md (3b)", "bank": BANK, "speed_source": GT, "clips_kept": int(len(keep_clips)),
       "excluded": excluded, "interp_frames": n_interp, "rows": int(len(rows)), "fit_clips": int(len(fit_clips)),
       "scored_clips": int(len(sc_clips)), "scored_clip_ordinals": sc_clips.tolist(), "deltas_rows": DELTAS,
       "fit_meta": meta, "content_checks": content, "n_triplet_clips": len(trip), "p2_kept_per_clip_of_200": p2kept, "p2_tol": P2TOL, "step": STEP,
       "estimator": "clip-cluster bootstrap over SCORED clips (NaN clips dropped per arm), 95% percentile",
       "arms": {k: {s: boot(v) for s, v in d.items()} for k, d in arms.items()}, "paired": {}}
for lat in ("latent_704", "latent_7040"):
    for m in ("raw", "diag"):
        res["paired"][f"{lat}|{m}_minus_pixel_1440|{m}"] = {
            s: boot(arms[f"{lat}|{m}"][s], arms[f"pixel_1440|{m}"][s]) for s in ("P1", "P2", "P3")}
# path-length vs time sanity (descriptive): how much does L deviate from mean-speed * Delta
cv = [np.nanstd(np.diff(cum[c][::40])) / max(np.nanmean(np.diff(cum[c][::40])), 1e-9) for c in sc_clips]
res["descriptive"] = {"scored_clip_pathlen_per4s_cv_median": round(float(np.median(cv)), 4),
                      "scored_clip_pathlen_per4s_cv_p90": round(float(np.percentile(cv, 90)), 4)}
res["seconds"] = round(time.time() - t0, 1)
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
for k, v in res["arms"].items():
    print(k, {s: (x["point"], x["ci95"], x["n_clips"]) for s, x in v.items()})
for k, v in res["paired"].items():
    print(k, {s: (x["point"], x["ci95"]) for s, x in v.items()})
