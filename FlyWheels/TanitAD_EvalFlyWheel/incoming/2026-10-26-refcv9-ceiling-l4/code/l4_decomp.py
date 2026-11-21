"""Step 2: cause decomposition of the windows where the logged human exceeds the fed ceiling (strict), eval139 + train."""
import json, sys, numpy as np
sys.path.insert(0, "F:/refcv9_ceiling/code")
import l4_common as C

CLS = [30, 50, 100, 120]
cot = json.load(open(C.COT_LIMITS, encoding="utf-8"))["read"]
def cot_ms(cid):
    r = cot.get(cid)
    if not r or r.get("unit") not in ("km/h", "mph"):
        return np.nan
    return r["value"] / 3.6 if r["unit"] == "km/h" else r["value"] * 0.44704


def windows_eval():
    z = np.load(f"{C.ARCH}/{C.EVAL_TREE}/dump_s0/refcv6_extras.npz")
    man = json.load(open(f"{C.ARCH}/{C.EVAL_TREE}/dump_s0/manifest.json", encoding="utf-8"))
    ep = {e["episode_index"]: e["clip_id"] for e in man["episodes"]}
    sc = C.load_sidecar(C.SIDECAR_EVAL)
    ci = z["clip_index"].astype(int)
    cid = [ep[i] for i in ci]
    return dict(cid=np.array(cid), kn=z["ws"].astype(int) + 2, gt=z["gt_max_speed_6s"].astype(np.float64),
                vhi=np.array([sc[c]["v_hi_ms"] for c in cid], np.float64),
                valid=np.array([int(sc[c].get("valid", 1)) for c in cid]), fed_kmh=np.round(z["v_max_ms"].astype(np.float64) * 3.6, 3))   # float32 dust: 29.99998 -> 30.0


def windows_train(stride=1):
    sc = C.load_sidecar(C.SIDECAR_TRAIN)
    v9 = C.V9(C.V9_TRAIN)
    cid, kn, gt, vhi, valid = [], [], [], [], []
    n_join = 0
    for c, r in sc.items():
        d = v9.clip(r["sid"])
        if d is None:
            continue
        n_join += 1
        V = d["V"]; kmax = len(V) - 1
        for k in range(9, kmax - 60 + 1, stride):          # NOW rows with a full 6-s future in the release
            seg = V[k:k + 61]
            if not np.isfinite(seg[0]) or np.isnan(seg).mean() > 0.5:
                continue
            cid.append(c); kn.append(k); gt.append(np.nanmax(seg))
            vhi.append(np.nan if r.get("v_hi_ms") is None else r["v_hi_ms"]); valid.append(int(r.get("valid", 1)))
    w = dict(cid=np.array(cid), kn=np.array(kn), gt=np.array(gt), vhi=np.array(vhi, np.float64), valid=np.array(valid))
    w["fed_kmh"] = np.where(w["valid"] > 0, C.snap_up_kmh(np.nan_to_num(w["vhi"])), np.nan)
    return w, {"n_sidecar_clips": len(sc), "n_joined_v9": n_join}


def decompose(w):
    lim = np.where(w["valid"] > 0, w["fed_kmh"] / 3.6, np.inf)
    exc = w["gt"] > lim
    n = len(exc)
    vk = w["vhi"] * 3.6
    clamp_clip = (vk > 120.0 + 1e-6) & exc                          # exceedances on clamped (>120 km/h) clips
    clamp = exc & (w["gt"] <= w["vhi"])                              # (i) CAUSED by the clamp: compliant against the raw v_hi, not against the fed 120
    below_raw = exc & (w["gt"] <= w["vhi"])                          # exceed the fed value while <= the raw band edge (must equal clamp set)
    unknown = exc & (w["valid"] < 1)                                 # (iv)
    t_now = w["kn"] / C.HZ                                           # raw seconds
    band_lo, band_hi = (C.ANCHOR_RAW + 20) / C.HZ, (C.ANCHOR_RAW + 60) / C.HZ
    before = (t_now + 6.0) < band_lo
    after = t_now > band_hi
    overlap = ~before & ~after
    excess = (w["gt"] - lim) * 3.6                                    # km/h above the fed ceiling
    gt_bin = C.snap_up_kmh(w["gt"])
    steps_up = np.searchsorted(np.array(C.L4_KMH), gt_bin) - np.searchsorted(np.array(C.L4_KMH), np.nan_to_num(w["fed_kmh"]))
    cotv = np.array([cot_ms(c) for c in w["cid"]])
    has_cot = np.isfinite(cotv)
    res = {"n_windows": int(n), "n_clips": int(len(set(w["cid"]))), "n_exceed": int(exc.sum()), "exceed_frac": round(float(exc.mean()), 6),
           "causes": {
               "i_snap_clamp_gt120_caused": int(clamp.sum()),
               "i_exceed_on_clamped_clips_any_cause": int(clamp_clip.sum()),
               "i_check_exceed_but_le_raw_v_hi": int(below_raw.sum()),
               "iv_unknown_band": int(unknown.sum()),
               "ii_temporal_label_from_other_time": int((exc & ~clamp & ~unknown).sum()),
               "ii_by_window_position": {"future_before_label_band": int((exc & before).sum()), "future_overlaps_label_band": int((exc & overlap).sum()),
                                         "now_after_label_band": int((exc & after).sum())},
               "ii_steps_above_fed_class": {str(s): int((exc & (steps_up == s)).sum()) for s in (1, 2, 3)},
               "excess_kmh_bins": {"(0,5]": int((exc & (excess <= 5)).sum()), "(5,10]": int((exc & (excess > 5) & (excess <= 10)).sum()),
                                   "(10,20]": int((exc & (excess > 10) & (excess <= 20)).sum()), ">20": int((exc & (excess > 20)).sum())},
               "excess_kmh_quantiles": {q: round(float(np.percentile(excess[exc], p)), 2) for q, p in (("p50", 50), ("p90", 90), ("max", 100))} if exc.any() else {},
               "iii_independent_ref_CoT_unit_stated": {
                   "n_windows_with_ref": int(has_cot.sum()), "n_clips_with_ref": int(len(set(w["cid"][has_cot]))),
                   "n_exceed_with_ref": int((exc & has_cot).sum()),
                   "exceed_and_human_above_CoT_limit": int((exc & has_cot & (w["gt"] > cotv + 1e-6)).sum()),
                   "exceed_and_CoT_limit_ge_human (fed class too low)": int((exc & has_cot & (w["gt"] <= cotv + 1e-6)).sum()),
                   "fed_class_vs_CoT_limit_on_ref_windows": {"fed_lt_cot": int((has_cot & (lim < cotv - 1e-6)).sum()),
                                                             "fed_eq_cot": int((has_cot & (np.abs(lim - cotv) <= 1e-6)).sum()),
                                                             "fed_gt_cot": int((has_cot & (lim > cotv + 1e-6)).sum())}}},
           "per_class": {}}
    for k in CLS:
        m = np.abs(np.nan_to_num(w["fed_kmh"]) - k) < 1e-6
        res["per_class"][str(k)] = {"n_windows": int(m.sum()), "n_clips": int(len(set(w["cid"][m]))), "n_exceed": int((exc & m).sum()),
                                    "exceed_frac": round(float((exc & m).sum() / max(m.sum(), 1)), 4),
                                    "clamp": int((clamp & m).sum()), "temporal": int((exc & m & ~clamp).sum()),
                                    "excess_kmh_p50": round(float(np.median(excess[exc & m])), 2) if (exc & m).any() else None,
                                    "exceed_clips": int(len(set(w["cid"][exc & m])))}
    # clip concentration: how many clips carry the exceedance
    cl, cnt = np.unique(w["cid"][exc], return_counts=True)
    res["clip_concentration"] = {"n_clips_with_any_exceed": int(len(cl)), "top10pct_clips_share_of_exceed_windows":
                                 round(float(np.sort(cnt)[::-1][:max(1, len(cl) // 10)].sum() / max(exc.sum(), 1)), 4)}
    return res, exc


if __name__ == "__main__":
    out = {"_evidence": "MEASURED. eval: dump extras of tree %s (fed ceiling + gt_max_speed_6s as the battery reads them); "
                        "train: sidecar %s x v9 release v_now (all NOW rows with a full 6-s future, stride 1)" % (C.EVAL_TREE, C.SIDECAR_TRAIN),
           "strict": "exceed iff gt_max_speed_6s > fed ceiling (A-R8-CLAMP-2 2.1)"}
    we = windows_eval()
    out["eval139"], _ = decompose(we)
    wt, jt = windows_train()
    out["train"], _ = decompose(wt)
    out["train"]["join"] = jt
    json.dump(out, open("F:/refcv9_ceiling/raw/l4_decomp.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
