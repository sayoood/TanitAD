"""E-DEP-VGEO-1 — value-geometry screen on frozen refcv5-v2 trunk tokens (see ../SPEC.md).

CPU only, below-normal priority, every STEP-th row. Writes raw/vgeo_screen.json.
"""
import hashlib, json, os, sys, time
import numpy as np
from scipy.stats import spearmanr

BANK = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens"
OUT = sys.argv[2] if len(sys.argv) > 2 else "raw/vgeo_screen.json"
STEP, DMIN, DMAX, SHORT = 4, 4, 160, 60
NBOOT, SEED = 2000, 20260913

try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception as e:  # priority is a courtesy, not a correctness condition
    print("priority not lowered:", e)

t0 = time.time()
idx = np.load(os.path.join(BANK, "index.npz"), allow_pickle=True)
clip, rawf = idx["clip_ordinal"], idx["raw_frame"]
tok = np.load(os.path.join(BANK, "tokens_s32_fp16.npy"), mmap_mode="r")
pix = np.load(os.path.join(BANK, "pix64_u8.npy"), mmap_mode="r")
assert tok.shape[0] == pix.shape[0] == clip.shape[0], (tok.shape, pix.shape, clip.shape)

rows = []
for c in np.unique(clip):
    r = np.where(clip == c)[0]
    r = r[np.argsort(rawf[r])]
    rows.extend(r[::STEP].tolist())
rows = np.array(sorted(rows))
print("rows selected", len(rows), "clips", len(np.unique(clip)))

Z1 = np.empty((len(rows), 704), np.float32)
Z2 = np.empty((len(rows), 7040), np.float32)
X = np.empty((len(rows), 1440), np.float32)
for i in range(0, len(rows), 256):
    rr = rows[i:i + 256]
    t = np.asarray(tok[rr], dtype=np.float32)                       # [b,704,8,20]
    Z1[i:i + len(rr)] = t.mean(axis=(2, 3))
    Z2[i:i + len(rr)] = t.reshape(len(rr), 704, 2, 4, 5, 4).mean(axis=(3, 5)).reshape(len(rr), -1)
    p = np.asarray(pix[rr], dtype=np.float32) / 255.0                # [b,9,64,160]
    X[i:i + len(rr)] = p.reshape(len(rr), 9, 8, 8, 20, 8).mean(axis=(3, 5)).reshape(len(rr), -1)
content = {"Z1_std": float(Z1.std()), "Z1_nonzero": float((Z1 != 0).mean()), "X_mean": float(X.mean()),
           "Z1_finite": bool(np.isfinite(Z1).all())}
print("content", content, "read s", round(time.time() - t0, 1))
assert content["Z1_std"] > 0 and content["Z1_finite"] and content["X_mean"] > 0, "poisoned bank"

rng = np.random.default_rng(SEED)
rclip, rframe = clip[rows], rawf[rows]
clips = np.unique(rclip)


def per_clip(feat, mode):
    """dict band -> array of per-clip rho (nan where undefined)."""
    out = {"short": [], "long": [], "all": []}
    for c in clips:
        m = np.where(rclip == c)[0]
        F, fr = feat[m], rframe[m]
        if mode == "mut":
            F = F[rng.permutation(len(m))]
        i, j = np.triu_indices(len(m), 1)
        d = fr[j] - fr[i]
        keep = (d >= DMIN) & (d <= DMAX)
        i, j, d = i[keep], j[keep], d[keep]
        if mode == "shuf":
            j = rng.integers(0, len(m), size=len(j))
        dist = np.zeros(len(i)) if mode == "const" else np.linalg.norm(F[i] - F[j], axis=1)
        for band, sel in (("short", d <= SHORT), ("long", d > SHORT), ("all", np.ones_like(d, bool))):
            if sel.sum() < 5 or np.all(dist[sel] == dist[sel][0]):
                out[band].append(0.0 if mode == "const" else np.nan)
            else:
                out[band].append(spearmanr(dist[sel], d[sel]).correlation)
    return {k: np.array(v, float) for k, v in out.items()}


def boot(a, b=None):
    n = len(a)
    pt = float(np.nanmean(a if b is None else a - b))
    bs = []
    for _ in range(NBOOT):
        s = rng.integers(0, n, n)
        bs.append(np.nanmean(a[s] if b is None else a[s] - b[s]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"point": round(pt, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n_clips": int(n)}


arms = {"latent_704": per_clip(Z1, "real"), "latent_7040": per_clip(Z2, "real"), "pixel_1440": per_clip(X, "real"),
        "C_const": per_clip(Z1, "const"), "C_shuf": per_clip(Z1, "shuf"), "C_mut": per_clip(Z1, "mut")}
res = {"spec": "SPEC.md E-DEP-VGEO-1", "bank": BANK, "rows": int(len(rows)), "clips": int(len(clips)),
       "step": STEP, "delta_rows": [DMIN, DMAX], "short_le": SHORT, "row_dt_s": 0.1, "nboot": NBOOT, "seed": SEED,
       "estimator": "clip-cluster bootstrap over per-clip Spearman rho (mean of clips), 95% percentile",
       "content_checks": content, "arms": {}, "paired": {}}
for k, v in arms.items():
    res["arms"][k] = {b: boot(v[b]) for b in v}
for lat in ("latent_704", "latent_7040"):
    res["paired"][lat + "_minus_pixel"] = {b: boot(arms[lat][b], arms["pixel_1440"][b]) for b in ("short", "long", "all")}
res["seconds"] = round(time.time() - t0, 1)
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v["short"] for k, v in res["arms"].items()}, indent=1))
print(json.dumps({k: v["short"] for k, v in res["paired"].items()}, indent=1))
