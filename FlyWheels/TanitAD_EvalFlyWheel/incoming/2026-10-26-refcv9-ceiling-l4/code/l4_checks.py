"""Step 3b: (1) C0 leak by window position (why eval 0.03 vs train 0.22); (2) N2 re-derived independently from v_now, past-only."""
import json, sys, numpy as np
sys.path.insert(0, "F:/refcv9_ceiling/code")
import l4_common as C
from l4_candidates import train_windows, eval_windows, oof_recovered, onehot_of

out = {}
for tag, w in (("eval139", eval_windows()), ("train", train_windows())):
    m = np.isfinite(w["y26"]) & np.isfinite(w["v0"])
    rel = (w["kn"] - C.ANCHOR_RAW) / C.HZ                              # NOW minus anchor, s
    out[tag] = {"now_minus_anchor_s_quantiles": [round(float(np.percentile(rel, p)), 1) for p in (0, 25, 50, 75, 100)], "leak_C0_by_position": {}}
    c0 = C.snap_up_kmh(w["vhi"])
    for lo, hi in ((-8, -4), (-4, 0), (0, 4), (4, 8), (8, 12)):
        s = m & (rel >= lo) & (rel < hi)
        if s.sum() < 500:
            continue
        y = w["y26"][s]; folds = np.array([int(C.sha12(c), 16) % 5 for c in w["cid"][s]])
        base = np.stack([np.ones(s.sum()), w["v0"][s]], 1)
        r0, r, rec = oof_recovered(y, base, onehot_of(np.round(c0[s], 3)), folds)
        out[tag]["leak_C0_by_position"][f"[{lo},{hi}) s"] = {"n": int(s.sum()), "recovered": round(rec, 4),
                                                             "target_equals_label_band": bool(lo == 0)}
# (2) N2 independent re-derivation: snap-up-8 of max(v_now over [NOW-20 s, NOW]) with urban floor 50, vs the release column
sc = C.load_sidecar(C.SIDECAR_EVAL); v9 = C.V9(C.V9_EVAL)
agree = tot = 0; fut_inv = 0
rng = np.random.default_rng(0)
for c, r in sc.items():
    d = v9.clip(r["sid"])
    if d is None:
        continue
    V = d["V"]
    for k in d["k"]:
        rel = d["N2"][k]
        if not np.isfinite(rel) or rel <= 0:
            continue
        past = V[max(0, k - 200):k + 1]
        mine = C.snap_up_kmh(np.array([max(np.nanmax(past), 50 / 3.6)]), C.L8_KMH)[0]
        tot += 1; agree += int(abs(mine - rel) < 1e-6)
        # future invariance: scramble every sample after NOW, recompute -> must be identical
        V2 = V.copy(); V2[k + 1:] = rng.uniform(0, 40, len(V2) - k - 1)
        mine2 = C.snap_up_kmh(np.array([max(np.nanmax(V2[max(0, k - 200):k + 1]), 50 / 3.6)]), C.L8_KMH)[0]
        fut_inv += int(mine2 == mine)
out["N2_independent_rederivation_eval139"] = {"n_rows": tot, "agree_with_release_speed_n2_kmh": agree, "agree_frac": round(agree / max(tot, 1), 4),
                                               "future_scramble_invariant": fut_inv,
                                               "rule": "snap-up 8-step road-law of max(v_now[NOW-200 : NOW]) floored at 50 km/h"}
json.dump(out, open("F:/refcv9_ceiling/raw/l4_checks.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
