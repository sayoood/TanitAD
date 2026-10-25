"""Independent re-derivation of the fitted weights from the saved fit-window arrays (a second implementation: np.linalg.lstsq per step/axis, sklearn
IsotonicRegression if installed). Writes fit_verify.json. CPU, numpy."""
import json
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
res = {}
try:
    from sklearn.isotonic import IsotonicRegression
except Exception:                                       # noqa: BLE001
    IsotonicRegression = None
for s in ("M2b_s0", "M2b_s1"):
    z = np.load(f"{OUT}/fit_windows_{s}.npz")
    W = json.load(open(f"{OUT}/weights_{s}.json"))
    tau, P, y = (z[k].astype(np.float64) for k in ("traj30", "p30", "gt"))
    r, d = tau - P, y - P
    raw = np.zeros((30, 2))
    for a in range(2):
        for t in range(30):
            raw[t, a] = np.linalg.lstsq(r[:, t, a:a + 1], d[:, t, a], rcond=None)[0][0]
    clipped = np.clip(raw, 0, 1)
    rec = {"n_windows": int(len(r)), "max_abs_raw_lstsq_vs_json": float(max(np.abs(raw[:, 0] - W["raw_along"]).max(), np.abs(raw[:, 1] - W["raw_cross"]).max()))}
    if IsotonicRegression is not None:
        iso = np.stack([IsotonicRegression(increasing=True).fit_transform(np.arange(30), clipped[:, a]) for a in range(2)], 1)
        rec["max_abs_sklearn_isotonic_vs_json"] = float(max(np.abs(iso[:, 0] - W["w_along_by_step"]).max(), np.abs(iso[:, 1] - W["w_cross_by_step"]).max()))
    else:
        rec["sklearn"] = "not installed: isotonic verified by test_v8f_hblend.py brute force only"
    res[s] = rec
json.dump(res, open(f"{OUT}/fit_verify.json", "w"), indent=1)
print(json.dumps(res, indent=1))
