"""Step 3c: N1 (past-only, no urban floor) 4-way; posted-limit proxy (CoT sign) compliance; C7 per-class compliance."""
import json, sys, numpy as np
sys.path.insert(0, "F:/refcv9_ceiling/code")
import l4_common as C
from l4_candidates import train_windows, eval_windows, oof_recovered, onehot_of
from l4_decomp import cot_ms

out = {}
for tag, sc_path, v9_path, w in (("eval139", C.SIDECAR_EVAL, C.V9_EVAL, eval_windows()), ("train", C.SIDECAR_TRAIN, C.V9_TRAIN, train_windows())):
    sc = C.load_sidecar(sc_path); v9 = C.V9(v9_path)
    p20 = np.full(len(w["kn"]), np.nan); cache = {}
    for j, (c, k) in enumerate(zip(w["cid"], w["kn"])):
        if c not in cache:
            cache[c] = v9.clip(sc[c]["sid"])["V"]
        s = cache[c][max(0, k - 200):k + 1]
        p20[j] = np.nanmax(s) if np.isfinite(s).any() else np.nan
    gt = w["gt"]
    c7b = C.snap_up_kmh(np.nan_to_num(p20), C.L4_KMH, clamp_value=True)        # N1 past-20 s max -> 4-way containing, >120 keeps value
    n2 = w["n2"]; ok = np.isfinite(n2) & (n2 > 0)
    c7 = np.where(ok, C.snap_up_kmh(np.where(ok, n2, 0) / 3.6, C.L4_KMH, clamp_value=True), np.inf)
    m = np.isfinite(w["y26"]) & np.isfinite(w["v0"])
    folds = np.array([int(C.sha12(c), 16) % 5 for c in w["cid"][m]]); base = np.stack([np.ones(m.sum()), w["v0"][m]], 1)
    r = {}
    e = gt > c7b / 3.6
    r["C7b_past_only_N1_no_floor_4way"] = {"compliance": round(float(1 - e.mean()), 4), "n_exceed": int(e.sum()),
        "class_hist_kmh": {str(int(k)): int(v) for k, v in zip(*np.unique(np.minimum(np.round(c7b), 999), return_counts=True)) if k <= 120},
        "n_above_120_kept": int((c7b > 120).sum()),
        "leak_recovered": round(oof_recovered(w["y26"][m], base, onehot_of(np.round(np.minimum(c7b[m], 121), 3)), folds)[2], 4)}
    c0 = C.snap_up_kmh(w["vhi"])
    r["C7_per_class_compliance"] = {str(int(k)): {"n": int((np.round(c7) == k).sum()), "compliance": round(float(1 - (gt > c7 / 3.6)[np.round(c7) == k].mean()), 4)} for k in (50, 100, 120, 130)}
    r["C7_compliance_on_C0_30_class_windows"] = round(float(1 - (gt > c7 / 3.6)[np.abs(c0 - 30) < 1e-6].mean()), 4)
    # posted-limit proxy: CoT unit-stated sign reading on the clip (ungrounded, ego-coupled; SL-SRC-1)
    cot = np.array([cot_ms(c) for c in w["cid"]]); h = np.isfinite(cot)
    r["posted_limit_proxy_CoT"] = {"n_windows": int(h.sum()), "n_clips": int(len(set(w["cid"][h]))),
        "compliance_vs_raw_sign": round(float((gt[h] <= cot[h] + 1e-6).mean()), 4) if h.any() else None,
        "compliance_vs_sign_snapped_up_4way": round(float((gt[h] <= C.snap_up_kmh(cot[h]) / 3.6 + 1e-6).mean()), 4) if h.any() else None,
        "C0_compliance_same_windows": round(float((gt[h] <= c0[h] / 3.6).mean()), 4) if h.any() else None,
        "C7_compliance_same_windows": round(float((gt[h] <= c7[h] / 3.6).mean()), 4) if h.any() else None}
    out[tag] = r
json.dump(out, open("F:/refcv9_ceiling/raw/l4_extra.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
