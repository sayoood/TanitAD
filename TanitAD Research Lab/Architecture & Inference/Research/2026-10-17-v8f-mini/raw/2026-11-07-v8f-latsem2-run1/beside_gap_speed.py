"""PREREG_V8F_LATSEM2 beside (labels only, decides nothing): how the lead gap relates to the ego speed in the bank, to read S3's sign.
S3 reads the imagined lead gap LARGER under accelerate than under brake; LATSEM's CF-LON reads the imagined SPEED larger under accelerate. If gap and speed
move together in the labels, a gap probe on a latent that encodes speed returns the sign of that correlation. Run from theory/latsem2 with WMACT_CODE_ROOT set."""
import json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.abspath("../wmact"))
import wmact_lib as W  # noqa: E402,F401
import labels_latsem2 as LB  # noqa: E402

ref = np.load("labels_latsem2.npz")
beid, bg, gap = ref["beid"], ref["bg"], ref["lead_gap"]
TR = {k: np.asarray(v, np.float64) for k, v in torch.load(LB.TRACKS, weights_only=False)["tracks"].items()}
v = np.array([TR[str(e)][2 * int(g), 3] for e, g in zip(beid, bg)])
f = np.isfinite(gap) & np.isfinite(v)
r = float(np.corrcoef(gap[f], v[f])[0, 1])
slope = float(np.polyfit(v[f], gap[f], 1)[0])
fold = np.array([int(__import__("hashlib").sha256(str(e).encode()).hexdigest()[:8], 16) % 2 for e in beid])
out = {"n": int(f.sum()), "pearson_gap_speed": r, "ols_slope_m_per_ms": slope,
       "by_fold_pearson": [float(np.corrcoef(gap[f & (fold == k)], v[f & (fold == k)])[0, 1]) for k in (0, 1)],
       "speed_quartiles_ms": [float(x) for x in np.percentile(v[f], [25, 50, 75])],
       "gap_median_by_speed_tercile_m": [float(np.median(gap[f & (v >= lo) & (v < hi)])) for lo, hi in
                                         zip(np.percentile(v[f], [0, 33.3, 66.7]), list(np.percentile(v[f], [33.3, 66.7])) + [1e9])]}
json.dump(out, open("BESIDE_GAP_SPEED.json", "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps(out, indent=1))
