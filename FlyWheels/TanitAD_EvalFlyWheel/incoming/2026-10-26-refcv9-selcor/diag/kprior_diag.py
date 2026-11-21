"""Diagnose K-PRIOR (PREREG_SELCOR s4): FIT-side only (fresh split); reads no eval139 window."""
import sys, json
import numpy as np
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2c/code")
import selcor_score as SC
ctl = {}
fr, trf, j, _z = SC.load_fit_rows(ctl, "F:/refcv9_selcor/locus_fresh_s0.npz")
HP = np.asarray(fr["HP"], np.float64)
pr = trf["prior_path"][j].astype(np.float64)
pl = trf["plan8"][j].astype(np.float64)
S6 = list(SC.C.S6_SLOTS)
out = {"K-PRIOR": ctl["K-PRIOR"], "S6_SLOTS": S6, "prior_dtype": str(trf["prior_path"].dtype), "HP_dtype": str(np.asarray(fr["HP"]).dtype)}
d = np.abs(pr[:, S6] - HP).max(axis=(1, 2))
out["per_row_maxabs"] = {"q50": float(np.median(d)), "q90": float(np.quantile(d, .9)), "q99": float(np.quantile(d, .99)), "max": float(d.max()),
                         "frac_le_1e-5": float((d <= 1e-5).mean()), "frac_le_1e-3": float((d <= 1e-3).mean()), "frac_le_1e-2": float((d <= 1e-2).mean())}
out["per_slot_maxabs"] = [float(np.abs(pr[:, s] - HP[:, i]).max()) for i, s in enumerate(S6)]
for name, A in (("prior_path", pr), ("plan8", pl)):
    best = {}
    for off in range(-2, 3):
        idx = [s + off for s in S6]
        if min(idx) < 0 or max(idx) > 7: continue
        best[off] = float(np.abs(A[:, idx] - HP).max())
    out[f"slot_offset_maxabs_{name}"] = best
# relative scale: typical |HP|
out["HP_abs_q50"] = float(np.median(np.abs(HP)))
# arc-length agreement (what the label actually uses)
def arc(P):
    P0 = np.concatenate([np.zeros_like(P[:, :1]), P], 1)
    return np.linalg.norm(np.diff(P0, axis=1), axis=-1).cumsum(1)
out["arc_S6_maxabs_vs_prior"] = float(np.abs(arc(pr[:, S6]) - arc(HP)).max())
# rows that fail: do they cluster by clip?
bad = d > 1e-5
out["n_bad_rows"] = int(bad.sum())
out["n_bad_clips"] = int(len(np.unique(np.asarray(fr["clip"])[bad]))) if bad.any() else 0
out["n_clips"] = int(len(np.unique(np.asarray(fr["clip"]))))
print(json.dumps(out, indent=1))
