"""PROBE 12b (TRAIN ONLY): the attribution arm MC-H of PREREG_DRAFT: w_along[k] = HACC's own banked w_along at knot 10 (the replan knot) for k = 1..12,
HACC's banked w beyond and on the cross axis. Train open-loop ADE2/ADE6 and the replan-state a0 / v0 vs P10's state (statekin), as in probe 12.
python probe12b_mch.py > probe12b.log"""
import json, sys
import numpy as np
sys.path.insert(0, "C:/Users/Admin/lcdiag")
import statekin as SK  # noqa: E402
WJ = "C:/Users/Admin/t1k/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-hacc/run/weights_HACC_M2b_s{s}.json"
st = lambda A: SK.state_from_knots(np.vstack([np.zeros((1, 2)), A]), 0)[0]
out = {}
for s in (0, 1):
    wj = json.load(open(WJ.format(s=s)))
    wh = np.stack([wj["w_along_by_step"], wj["w_cross_by_step"]], 1)
    w = wh.copy(); w[:12, 0] = wh[9, 0]
    z = np.load(f"C:/Users/Admin/hlat/dumps_train/ticks_HACC_s{s}_m10_train.npz")
    m0 = (z["tick"] == 0) & z["gt_ok"]
    P, T, Y = z["P"][m0], z["tau"][m0], z["gt"][m0]
    e = np.linalg.norm(P + w[None] * (T - P) - Y, axis=-1)
    r = {"c": float(wh[9, 0]), "ade2": float(e[:, :10].mean()), "ade6": float(e.mean())}
    rows = []
    for i in np.nonzero(z["tick"] <= 1)[0][::16]:
        a = st(z["P"][i] + w * (z["tau"][i] - z["P"][i])); p = st(z["P"][i])
        rows.append((z["tick"][i], z["v0"][i], a[1] - p[1], a[0] - p[0]))
    A = np.array(rows)
    for t in (0, 1):
        for lo, hi in ((0, 5.2), (5.2, 99)):
            m = (A[:, 0] == t) & (A[:, 1] >= lo) & (A[:, 1] < hi)
            r[f"plan@tick{t} v0 {lo}-{hi}"] = {"a0_minus_P10_med": round(float(np.median(A[m, 2])), 4), "v0_minus_P10_med": round(float(np.median(A[m, 3])), 4), "n": int(m.sum())}
    out[f"s{s}"] = r
    print(f"s{s}", r)
json.dump(out, open("C:/Users/Admin/lcdiag/probe12b.json", "w"), indent=1)
