"""H5 — has refcv5-v2's trunk collapsed? Participation ratio (sigma^2), never effective rank.

participation_ratio = (sum lambda)^2 / sum lambda^2 of the feature covariance.
It reads 1.0 for a rank-1 (collapsed) field and d for an isotropic one.

Surfaces (same 139 eval clips, every 40th frame, frozen refcv5-v2 stride-32 map):
  * token-level: all 160 tokens of a frame as separate samples [n*160, 704]
  * frame-pooled: mean over the 8x20 grid [n, 704] (what LAW and the strategic GRU read)
Controls that must read known values:
  * isotropic Gaussian of the same shape  -> PR close to d (finite-sample below d)
  * rank-1 field (one direction + tiny noise) -> PR ~ 1.0
Reference (programme, MEASURED): frozen DINOv3 on our frames PR 8.56 (G-RANK gate).
⚠️ PR is not dimension-normalised; the DINOv3 reference is a different model and d.
"""
import json
import sys
import numpy as np

C = r"C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens"
ix = np.load(C + "/index.npz", allow_pickle=True)
clip_ord = ix["clip_ordinal"]
tok = np.load(C + "/tokens_s32_fp16.npy", mmap_mode="r")
sel = np.concatenate([np.flatnonzero(clip_ord == c)[::40] for c in np.unique(clip_ord)])
x = np.asarray(tok[np.sort(sel)], dtype=np.float32)          # [n, 704, 8, 20]
n, d = x.shape[0], x.shape[1]


def pr(f):
    f = f.astype(np.float64)
    f = f - f.mean(0)
    lam = np.clip(np.linalg.eigvalsh(f.T @ f / (f.shape[0] - 1)), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum()), lam[::-1]


tokens = x.transpose(0, 2, 3, 1).reshape(-1, d)
pooled = x.mean(axis=(2, 3))
rng = np.random.default_rng(0)
out = {"n_frames": int(n), "n_tokens": int(tokens.shape[0]), "d": int(d)}
for name, f in (("token_level", tokens), ("frame_pooled", pooled)):
    p, lam = pr(f)
    out[name] = {"participation_ratio": round(p, 3),
                 "top1_energy_share": round(float(lam[0] / lam.sum()), 4),
                 "top10_energy_share": round(float(lam[:10].sum() / lam.sum()), 4)}
iso_tok, _ = pr(rng.standard_normal(tokens.shape).astype(np.float32))
iso_pool, _ = pr(rng.standard_normal(pooled.shape).astype(np.float32))
u = rng.standard_normal((1, d))
rank1 = rng.standard_normal((pooled.shape[0], 1)) @ u + 1e-3 * rng.standard_normal(pooled.shape)
r1, _ = pr(rank1)
out["ctrl_isotropic_token_shape"] = round(iso_tok, 2)
out["ctrl_isotropic_pooled_shape"] = round(iso_pool, 2)
out["ctrl_rank1"] = round(r1, 4)
json.dump(out, open(sys.argv[1], "w"), indent=2)
print(json.dumps(out, indent=2))
