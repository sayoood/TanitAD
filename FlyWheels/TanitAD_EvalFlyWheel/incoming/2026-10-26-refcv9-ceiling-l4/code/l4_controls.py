"""Step 1: reproduce the fed ceiling, the human max speed and the 12.52 % from INDEPENDENT artifacts (controls)."""
import json, sys, numpy as np
sys.path.insert(0, "F:/refcv9_ceiling/code")
import l4_common as C

out = {}
z0 = np.load(f"{C.ARCH}/{C.EVAL_TREE}/dump_s0/refcv6_extras.npz")
man = json.load(open(f"{C.ARCH}/{C.EVAL_TREE}/dump_s0/manifest.json", encoding="utf-8"))
ep = {e["episode_index"]: e["clip_id"] for e in man["episodes"]}
ci, ws = z0["clip_index"].astype(int), z0["ws"].astype(int)
fed = z0["v_max_ms"].astype(np.float64); val = z0["v_max_valid"]; gt = z0["gt_max_speed_6s"].astype(np.float64)
lim = np.where(val > 0.5, fed, np.inf)
exc = gt > lim
out["dump"] = {"tree": C.EVAL_TREE, "n_windows": int(len(gt)), "n_clips": int(len(set(ci))), "n_exceed": int(exc.sum()),
               "exceed_frac": round(float(exc.mean()), 6), "valid_frac": float(val.mean())}
# same label / same human on other trees and the other inference seed (the mask is model-free)
for t, s in ((C.EVAL_TREE, 1), ("step30000ema_unc", 0), ("step48852ema_unc", 0)):
    z = np.load(f"{C.ARCH}/{t}/dump_s{s}/refcv6_extras.npz")
    same = (np.array_equal(z["clip_index"], z0["clip_index"]) and np.array_equal(z["ws"], z0["ws"])
            and np.array_equal(z["v_max_ms"], z0["v_max_ms"]) and np.array_equal(z["gt_max_speed_6s"], z0["gt_max_speed_6s"]))
    out.setdefault("mask_identical_across", {})[f"{t}/s{s}"] = bool(same)
# C-FED: sidecar v_hi snapped up == fed value
sc = C.load_sidecar(C.SIDECAR_EVAL)
vhi = np.array([sc[ep[i]]["v_hi_ms"] for i in ci], np.float64)
fed_re = C.snap_up_kmh(vhi)
out["C_FED_snapup_of_sidecar_equals_fed"] = {"n_equal": int((np.abs(fed_re - fed * 3.6) < 1e-4).sum()), "n": int(len(fed))}
# C-GT: the human's [NOW, NOW+6 s] max from the v9 release v_now (a separate artifact) == dump gt_max_speed_6s
v9 = C.V9(C.V9_EVAL)
gt9 = np.full(len(gt), np.nan); lab9 = {}
for j, (c, w) in enumerate(zip(ci, ws)):
    sid = sc[ep[c]]["sid"]
    d = v9.clip(sid)
    if d is None:
        continue
    kn = w + 2
    seg = d["V"][kn:kn + 61]
    if np.isfinite(seg).any():
        gt9[j] = np.nanmax(seg)
    if c not in lab9:
        lb = d["V"][C.ANCHOR_RAW + 20:C.ANCHOR_RAW + 61]
        lab9[c] = (np.nanmax(lb) if np.isfinite(lb).any() else np.nan, sc[ep[c]]["v_hi_ms"])
dg = np.abs(gt9 - gt)
out["C_GT_v9_vnow_reproduces_gt_max_speed_6s"] = {"n_finite": int(np.isfinite(gt9).sum()), "n": int(len(gt)),
    "max_abs_ms": round(float(np.nanmax(dg)), 4), "p99_abs_ms": round(float(np.nanpercentile(dg, 99)), 4),
    "frac_within_0.05ms": round(float((dg < 0.05).mean()), 4),
    "exceed_recomputed_from_v9": int((gt9 > lim).sum())}
L = np.array(list(lab9.values()))
dl = np.abs(L[:, 0] - L[:, 1])
out["C_LABEL_anchor_band_reproduces_sidecar_v_hi"] = {"n_clips": int(len(L)), "rule": "max v_now over raw rows [80+20, 80+60] (anchor 8.0 s, band 2-6 s)",
    "frac_within_0.02ms": round(float((dl < 0.02).mean()), 4), "frac_within_0.2ms": round(float((dl < 0.2).mean()), 4),
    "median_abs_ms": round(float(np.nanmedian(dl)), 4), "max_abs_ms": round(float(np.nanmax(dl)), 4)}
json.dump(out, open("F:/refcv9_ceiling/raw/l4_controls.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
