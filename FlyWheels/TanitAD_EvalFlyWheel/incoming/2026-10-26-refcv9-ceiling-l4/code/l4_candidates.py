"""Step 3: candidate relabels -- compliance (eval + train), class shift, D4-style LEAK ('recovered'), compliant-set growth."""
import json, sys, numpy as np
sys.path.insert(0, "F:/refcv9_ceiling/code")
import l4_common as C
from l4_decomp import windows_eval

L4 = np.array(C.L4_KMH); L8 = np.array(C.L8_KMH)


def feats_from_v9(cids, kns, sc, v9):
    """per window: v0, y26 (max [NOW+2, NOW+6], D4 target), clip-span max, N2 (past-only), v9 per-frame v_hi [2,8]."""
    v0, y26, cmax, n2, own06 = (np.full(len(kns), np.nan) for _ in range(5))
    cache = {}
    for j, (c, k) in enumerate(zip(cids, kns)):
        if c not in cache:
            cache[c] = v9.clip(sc[c]["sid"])
        d = cache[c]
        V = d["V"]
        if k < len(V):
            v0[j] = V[k]
            s = V[k + 20:k + 61]; y26[j] = np.nanmax(s) if np.isfinite(s).any() else np.nan
            s6 = V[k:k + 61]; own06[j] = np.nanmax(s6) if np.isfinite(s6).any() else np.nan
            n2[j] = d["N2"][k]
        cmax[j] = np.nanmax(V)
    return dict(v0=v0, y26=y26, cmax=cmax, n2=n2, own06=own06)


def train_windows():
    sc = C.load_sidecar(C.SIDECAR_TRAIN); v9 = C.V9(C.V9_TRAIN)
    cid, kn = [], []
    for c, r in sc.items():
        d = v9.clip(r["sid"])
        if d is None:
            continue
        V = d["V"]
        for k in range(9, len(V) - 1 - 60 + 1):
            seg = V[k:k + 61]
            if np.isfinite(seg[0]) and np.isnan(seg).mean() <= 0.5:
                cid.append(c); kn.append(k)
    cid = np.array(cid); kn = np.array(kn)
    f = feats_from_v9(cid, kn, sc, v9)
    vhi = np.array([sc[c]["v_hi_ms"] for c in cid], np.float64)
    return dict(cid=cid, kn=kn, gt=f["own06"], vhi=vhi, fed_kmh=C.snap_up_kmh(vhi), **{k: f[k] for k in ("v0", "y26", "cmax", "n2")})


def eval_windows():
    w = windows_eval()
    sc = C.load_sidecar(C.SIDECAR_EVAL); v9 = C.V9(C.V9_EVAL)
    f = feats_from_v9(w["cid"], w["kn"], sc, v9)
    w.update({k: f[k] for k in ("v0", "y26", "cmax", "n2")})
    return w


def candidates(w):
    """name -> (ceiling km/h per window (inf = unknown), one-hot class index for the leak fit or None, n_classes, admissibility tag)."""
    vhi = w["vhi"]; n2 = w["n2"]
    cands = {}
    cands["C0_current_4way_snapup_clip_anchor"] = C.snap_up_kmh(vhi)
    cands["C1_no_snap_raw_v_hi"] = vhi * 3.6
    cands["C2_8step_roadlaw_snapup"] = C.snap_up_kmh(vhi, C.L8_KMH)
    cands["C3_drop_30_step_{50,100,120}"] = C.snap_up_kmh(vhi, (50.0, 100.0, 120.0))
    cands["C4_5step_add_140_no_clamp"] = C.snap_up_kmh(vhi, (30.0, 50.0, 100.0, 120.0, 140.0))
    cands["C5_per_window_own_[2,6]s_snapup"] = C.snap_up_kmh(w["y26"])
    cands["C6_clip_span_max_snapup"] = C.snap_up_kmh(w["cmax"])
    # N2 (past-only, v9 speed_n2_kmh, already snapped to the 8-step road-law ladder with urban floor 50) fed through the
    # 4-way containing window; refcv8's 'never below the fed value' rule keeps 130 as 130. -1 = not computable -> unknown.
    n2ok = np.isfinite(n2) & (n2 > 0)
    c7 = np.where(n2ok, C.snap_up_kmh(np.where(n2ok, n2, 0) / 3.6, C.L4_KMH, clamp_value=True), np.inf)
    cands["C7_past_only_N2_4way"] = c7
    cands["C8_max(C0,C7)_hybrid"] = np.maximum(C.snap_up_kmh(vhi), np.where(np.isfinite(c7), c7, 0))
    cands["C5b_ECHO_per_window_[0,6]s_snapup (target identity)"] = C.snap_up_kmh(w["gt"])
    return cands


def oof_recovered(y, base, X, folds):
    def oof(M):
        yh = np.zeros_like(y)
        for f in range(5):
            tr = folds != f
            b, *_ = np.linalg.lstsq(M[tr], y[tr], rcond=None)
            yh[~tr] = M[~tr] @ b
        return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    r0 = oof(base); r = oof(np.hstack([base, X]))
    return float(r0), float(r), float((r - r0) / (1 - r0))


def onehot_of(vals):
    u = np.unique(vals[np.isfinite(vals)])
    X = np.zeros((len(vals), len(u)))
    for i, x in enumerate(u):
        X[:, i] = (vals == x)
    return X


def run(w, tag, leak=True):
    gt = w["gt"]
    base_c = C.snap_up_kmh(w["vhi"])
    res = {"n_windows": int(len(gt)), "n_clips": int(len(set(w["cid"]))), "rows": {}}
    m = np.isfinite(w["y26"]) & np.isfinite(w["v0"])
    if leak:
        y = w["y26"][m]; folds = np.array([int(C.sha12(c), 16) % 5 for c in w["cid"][m]])
        base = np.stack([np.ones(m.sum()), w["v0"][m]], 1)
        res["leak_fit"] = {"n": int(m.sum()), "target": "max v over [NOW+2, NOW+6] s (D4)", "folds": "5, clip-grouped by sha12 % 5", "base": "[1, v0]"}
    for name, ceil_kmh in candidates(w).items():
        lim = ceil_kmh / 3.6
        exc = gt > lim
        row = {"compliance": round(float(1 - exc.mean()), 4), "n_exceed": int(exc.sum()),
               "n_compliant": int((~exc).sum()),
               "compliant_episodes": int(len(set(w["cid"][~exc]))),
               "class_hist_kmh": {str(int(k)) if np.isfinite(k) else "unknown": int(v) for k, v in zip(*np.unique(np.round(ceil_kmh, 0), return_counts=True))} if len(np.unique(np.round(ceil_kmh))) <= 12 else "continuous",
               "windows_changed_vs_C0": int((np.abs(ceil_kmh - base_c) > 1e-6).sum())}
        if leak:
            X = onehot_of(np.round(ceil_kmh[m], 3)) if row["class_hist_kmh"] != "continuous" else np.stack([ceil_kmh[m] / 3.6], 1)
            r0, r, rec = oof_recovered(y, base, X, folds)
            row["leak_recovered"] = round(rec, 4); row["r2_v0_only"] = round(r0, 4)
        res["rows"][name] = row
    # tolerance scoring on C0 (label unchanged; only the mask moves)
    lim0 = base_c / 3.6
    res["tolerance_on_C0"] = {}
    for tname, lim_t in (("+3 km/h", lim0 + 3 / 3.6), ("+5 km/h", lim0 + 5 / 3.6), ("+10 km/h", lim0 + 10 / 3.6),
                         ("+10 %", lim0 * 1.10), ("+20 %", lim0 * 1.20),
                         ("next ladder step", np.array(C.snap_up_kmh(lim0 + 1e-3)) / 3.6)):
        e = gt > lim_t
        res["tolerance_on_C0"][tname] = {"compliance": round(float(1 - e.mean()), 4), "n_exceed": int(e.sum()),
                                         "per_class_compliance": {str(int(k)): round(float(1 - e[np.abs(base_c - k) < 1e-6].mean()), 4) for k in C.L4_KMH}}
    if leak:  # controls for the leak instrument
        n2v = w["n2"][m]; ok = np.isfinite(n2v) & (n2v > 0)
        X8 = onehot_of(np.where(ok, n2v, -1.0))
        res["leak_controls"] = {"N0_constant (must be 0)": round(oof_recovered(y, base, np.ones((m.sum(), 1)), folds)[2], 6),
                                "N2_8step_onehot (D4 banked 0.0346, INHERITED)": round(oof_recovered(y, base, X8, folds)[2], 4),
                                "shuffled_C0 (clip-mispaired, must be ~0)": round(oof_recovered(y, base, onehot_of(np.random.default_rng(0).permutation(np.round(base_c[m], 3))), folds)[2], 4)}
    return res


if __name__ == "__main__":
    out = {"_evidence": "MEASURED. compliance = gt_max_speed_6s <= candidate ceiling (strict). eval gt = dump extras (battery's own); train gt = v9 v_now max [NOW, NOW+6]. leak = D4 'recovered' (5-fold clip-grouped OOF OLS, target max v [NOW+2, NOW+6], base [1, v0])."}
    out["eval139"] = run(eval_windows(), "eval")
    json.dump(out, open("F:/refcv9_ceiling/raw/l4_candidates.json", "w", encoding="utf-8"), indent=1)
    out["train"] = run(train_windows(), "train")
    json.dump(out, open("F:/refcv9_ceiling/raw/l4_candidates.json", "w", encoding="utf-8"), indent=1)
    print("done")
