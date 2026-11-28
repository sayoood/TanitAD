"""E-DEP-VGEO-4 DIAGNOSTIC -- why did C-oracle read 0.8983 instead of ~1.0?

⛔ This does NOT change the VOID-POWER verdict of the run (SPEC.md §5). It is the Rule-Zero
diagnosis: it measures whether the oracle's deficit is concentrated in cells where path length is
near-collinear with frame position, which is where a PARTIAL statistic must attenuate by
construction. Its output is an input to the NEXT pre-registration, not a re-read of this one.

Emits, per (clip, Delta) cell: n pairs, r_lp (Spearman pathlen vs start frame), and the oracle's
partial value in that cell. CPU only. argv: OUT STEP.
"""
import json, os, sys, time
import numpy as np
from scipy.stats import spearmanr

ROOT = "C:/Users/Admin/tanitad-caches/bevhead-20260913"
BANK, GT = ROOT + "/tokens", ROOT + "/bev_gt"
OUT = sys.argv[1] if len(sys.argv) > 1 else "raw/vgeo4_oracle_diag.json"
STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SEED = 20260913
DELTAS = list(range(12, 61, 4))
MIN_PAIRS = 8
try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception as e:
    print("priority not lowered:", e)
t0 = time.time()
idx = np.load(os.path.join(BANK, "index.npz"), allow_pickle=True)
clip, rawf, sha = idx["clip_ordinal"], idx["raw_frame"], idx["clip_sha12"]

speed = {}
for c in np.unique(clip):
    f = os.path.join(GT, str(sha[c]) + ".bevgt.npz")
    if not os.path.exists(f):
        continue
    g = np.load(f, allow_pickle=True)
    meta = json.loads(str(g["meta_json"]))
    if meta["stats"].get("deskew_source") != "egomotion_alpamayo":
        continue
    rf, v, ok = g["raw_frame"].astype(int), g["ego_v_ms"].astype(np.float64), g["label_valid"].astype(bool)
    vv = v.copy()
    if (~ok).any():
        vv[~ok] = np.interp(rf[~ok], rf[ok], v[ok])
    full = np.zeros(rf.max() + 1); full[rf] = vv
    speed[int(c)] = full
keep_clips = np.array(sorted(speed))

rows = []
for c in keep_clips:
    r = np.where(clip == c)[0]; r = r[np.argsort(rawf[r])]; rows.extend(r[::STEP].tolist())
rows = np.array(sorted(rows))
rclip, rframe = clip[rows], rawf[rows]

# identical seeded split to the main run
rng = np.random.default_rng(SEED)
perm = rng.permutation(keep_clips)
nfit = int(round(0.7 * len(perm)))
fit_clips, sc_clips = np.sort(perm[:nfit]), np.sort(perm[nfit:])
cum = {c: np.concatenate([[0.0], np.cumsum(speed[c] * 0.1)]) for c in keep_clips}


def pairs(c, lo, hi):
    m = np.where(rclip == c)[0]; fr = rframe[m]
    i, j = np.triu_indices(len(m), 1); d = fr[j] - fr[i]
    k = (d >= lo) & (d <= hi)
    return m[i[k]], m[j[k]]


def _sp(a, b):
    if np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    return spearmanr(a, b).correlation


gen = np.random.default_rng(SEED + 7)
cells = []
for c in sc_clips:
    for D in DELTAS:
        i, j = pairs(c, D, D)
        if len(i) < MIN_PAIRS:
            continue
        l = cum[c][rframe[j]] - cum[c][rframe[i]]
        pos = rframe[i].astype(float)
        if l.std() / max(l.mean(), 1e-9) < 0.05:
            continue
        dist = l + gen.normal(0.0, 0.01 * max(l.std(), 1e-9), size=len(l))
        r_dl, r_dp, r_lp = _sp(dist, l), _sp(dist, pos), _sp(l, pos)
        if not np.isfinite([r_dl, r_dp, r_lp]).all():
            continue
        den = np.sqrt(max(1.0 - r_dp ** 2, 0.0) * max(1.0 - r_lp ** 2, 0.0))
        pv = np.nan if den < 1e-6 else (r_dl - r_dp * r_lp) / den
        cells.append({"clip": int(c), "delta": int(D), "n": int(len(i)),
                      "r_lp": round(float(r_lp), 4), "r_dl": round(float(r_dl), 4),
                      "oracle_partial": None if not np.isfinite(pv) else round(float(pv), 4)})

ok = [x for x in cells if x["oracle_partial"] is not None]
a = np.array([x["oracle_partial"] for x in ok])
rlp = np.abs(np.array([x["r_lp"] for x in ok]))
npair = np.array([x["n"] for x in ok])


def band(mask, label):
    if mask.sum() == 0:
        return {"band": label, "cells": 0}
    return {"band": label, "cells": int(mask.sum()), "oracle_mean": round(float(a[mask].mean()), 4),
            "oracle_p10": round(float(np.percentile(a[mask], 10)), 4),
            "frac_below_0.9": round(float((a[mask] < 0.9).mean()), 4),
            "median_n_pairs": int(np.median(npair[mask]))}


res = {"note": "DIAGNOSTIC ONLY -- does not alter the VOID-POWER verdict of SPEC.md",
       "step": STEP, "cells_total": len(cells), "cells_scored": len(ok),
       "cells_degenerate_dropped": len(cells) - len(ok),
       "oracle_cell_mean": round(float(a.mean()), 4),
       "oracle_cell_median": round(float(np.median(a)), 4),
       "oracle_cell_min": round(float(a.min()), 4),
       "collinearity_bands": [band(rlp < 0.5, "|r_lp| < 0.5"), band((rlp >= 0.5) & (rlp < 0.8), "0.5 <= |r_lp| < 0.8"),
                              band((rlp >= 0.8) & (rlp < 0.95), "0.8 <= |r_lp| < 0.95"), band(rlp >= 0.95, "|r_lp| >= 0.95")],
       "npair_bands": [band(npair < 20, "n < 20"), band((npair >= 20) & (npair < 60), "20 <= n < 60"), band(npair >= 60, "n >= 60")],
       "spearman_oracle_vs_abs_r_lp": round(float(_sp(a, rlp)), 4),
       "spearman_oracle_vs_npairs": round(float(_sp(a, npair.astype(float))), 4),
       "seconds": round(time.time() - t0, 1)}
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
json.dump({"summary": res, "cells": cells}, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
