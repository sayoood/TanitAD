"""label_level_diag.py -- DESCRIPTIVE, post-hoc, 0 GPU (NOT a bar; not pre-registered): why did a better-RANKING head (decel AUC 0.7305 vs 0.685)
not move the factored composition?  The composition consumes the head's ARGMAX class, not its probability.  On the K-HEAD union windows (eval139 S6
compliant, GT-decel + GT-cruise, inference seed 0) this reads the LABEL-level decel decisions of each label source: share of GT-decel windows forced to a
decel class (label TPR) and share of GT-cruise windows forced to one (label FPR), for native argmax (= SELFSHARP's labels), LH argmax, DER argmax, and LH's
renormalised P(decel) thresholded at the native argmax's decel RATE (rate-matched; a candidate next arm, which would need its own prereg).  T0 diagnostic.
Episode-resampling bootstrap (B 2000, seed 0) for the rate-matched minus native TPR / FPR differences.  NEW file."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import k_head as K
import l2_common as C
K.assert_registered()
E = C.load_eval_pf(0); comp = np.asarray(E["compliant"], bool)
fd, md, _g, _ = C.lon_indicators(E["os"], E, comp)
pc, pd_ = np.isfinite(fd), np.isfinite(md); ui = np.flatnonzero(pc | pd_); y = pd_[ui]
keys = list(zip(E["ex_clip"][E["rows"]].tolist(), E["ex_ws"][E["rows"]].tolist()))
def lab_of(f):
    z = np.load(f); m = {(int(c), int(w)): int(v) for c, w, v in zip(z["clip_index"], z["ws"], z["lon_v7"])}
    return np.array([m[k] for k in keys])[ui]
DEC = (3, 4, 5)
src = {"native_argmax(SELFSHARP)": lab_of("C:/Users/Admin/refcv8_pf/refcv9_l2/selfsharp/labels_selfsharp_s0.npz"),
       "LH_argmax(LHS)": lab_of(ROOT / "labels/labels_LHS.npz"), "DER_argmax": lab_of(ROOT / "labels/labels_DER.npz")}
dec = {k: np.isin(v, DEC) for k, v in src.items()}
zb = K.load_bank(K.L1_BANK / "ztac_eval_s0.npz"); jz = np.array([zb["idx"][k] for k in keys])[ui]
LH = K.LF.load_frozen(ROOT / "LH_FROZEN.pt", K.REG["LH_FROZEN.pt"])
s_lh, _ = K.renorm_decel(K.softmax(K.LF.predict_logits6(LH, zb["z"][jz], np.asarray(zb["v0_in"], np.float64)[jz])))
s_nat, _ = K.renorm_decel(K.softmax(C.lon6(E["logits"])[ui]))
rate = dec["native_argmax(SELFSHARP)"].mean()
tau = np.quantile(s_lh, 1 - rate); dec["LH_ratematched_P(decel)"] = s_lh >= tau
tau_n = np.quantile(s_nat, 1 - rate); dec["native_ratematched_P(decel)_control"] = s_nat >= tau_n
out = {"tier": "T0 diagnostic, label level (no planner)", "n_union": int(len(ui)), "n_gt_decel": int(y.sum()), "n_gt_cruise": int((~y).sum()),
       "decel_classes_v7": DEC, "native_argmax_decel_rate": float(rate), "LH_rate_matched_tau": float(tau), "rows": {}}
for k, d in dec.items():
    out["rows"][k] = {"label_TPR_on_GT_decel": float(d[y].mean()), "label_FPR_on_GT_cruise": float(d[~y].mean()), "decel_rate": float(d.mean())}
ep = np.asarray(E["eid"])[ui]; ue, inv = np.unique(ep, return_inverse=True); rng = np.random.default_rng(0)
B = 2000; bt = np.empty((B, 2))
a, b = dec["LH_ratematched_P(decel)"].astype(float), dec["native_argmax(SELFSHARP)"].astype(float)
for i in range(B):
    w = np.bincount(rng.integers(0, len(ue), len(ue)), minlength=len(ue)).astype(float)[inv]
    bt[i, 0] = (w * (a - b) * y).sum() / (w * y).sum(); bt[i, 1] = (w * (a - b) * ~y).sum() / (w * ~y).sum()
out["LH_ratematched_minus_native_argmax"] = {"dTPR": float(a[y].mean() - b[y].mean()), "dTPR_ci95": [float(x) for x in np.percentile(bt[:, 0], [2.5, 97.5])],
                                             "dFPR": float(a[~y].mean() - b[~y].mean()), "dFPR_ci95": [float(x) for x in np.percentile(bt[:, 1], [2.5, 97.5])],
                                             "estimator": "episode-resampling bootstrap B 2000 seed 0, decisions held fixed"}
(ROOT / "result" / "LABEL_LEVEL_DIAG.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for k, v in out["rows"].items(): print(f"{k:40s} TPR {v['label_TPR_on_GT_decel']:.4f} FPR {v['label_FPR_on_GT_cruise']:.4f} rate {v['decel_rate']:.4f}")
print(json.dumps(out["LH_ratematched_minus_native_argmax"]))
