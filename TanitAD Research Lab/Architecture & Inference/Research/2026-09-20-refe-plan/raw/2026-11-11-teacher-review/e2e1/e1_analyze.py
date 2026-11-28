#!/usr/bin/env python3
"""E1 analysis. Paired log-cluster bootstrap (10,000 draws, seed 20260927, resample logs; ratio estimators) of the 10 Hz vs 20 Hz
sim8 DAC failure rate and of the class-C margin (class C set per rate; also the 10 Hz-class-C set evaluated at both rates).
python e1_analyze.py --rows <e1_s*.jsonl ...> --d1 <d1_navtrain.json> --out <json>"""
import argparse, json
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--rows", nargs="+", required=True); ap.add_argument("--d1", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
rows = [json.loads(l) for p in a.rows for l in open(p, encoding="utf-8")]
err = [r for r in rows if "error" in r]; rows = sorted([r for r in rows if "error" not in r], key=lambda r: (r["log"], r["token"]))
d1 = {(r["log"], r["token"]): r for r in json.load(open(a.d1, encoding="utf-8"))["rows"]}
n = len(rows); logs = np.array([r["log"] for r in rows]); ul, inv = np.unique(logs, return_inverse=True); L = len(ul)
rng = np.random.default_rng(20260927); draws = rng.integers(0, L, size=(10000, L))
def ratio_ci(num, den):
    sn = np.bincount(inv, weights=num, minlength=L); sd = np.bincount(inv, weights=den, minlength=L)
    est = sn[draws].sum(1) / np.maximum(sd[draws].sum(1), 1e-12)
    ok = sd[draws].sum(1) > 0
    return float(num.sum() / max(den.sum(), 1e-12)), float(np.percentile(est[ok], 2.5)), float(np.percentile(est[ok], 97.5)), float(den.sum())
def cls(h):  # D1 classes
    return "A" if h["raw20"] else "B" if h["interp8"] else "C" if h["sim8"] else "pass"
H = {hz: [r[hz] for r in rows] for hz in ("hz10", "hz20")}
f = {hz: np.array([h["sim8"] for h in H[hz]], float) for hz in H}
mg = {hz: np.array([h["margin_m"] for h in H[hz]]) for hz in H}
c = {hz: np.array([cls(h) for h in H[hz]]) for hz in H}
one = np.ones(n)
res = {"n_overlap": n, "n_logs": int(L), "n_errors": len(err), "logs": {str(l): int((logs == l).sum()) for l in ul},
       "estimator": "log-cluster bootstrap over the overlap's logs, 10,000 draws, seed 20260927, 95% percentile; ratio estimators",
       "control_hz10_sim8_vs_banked_d1": {"agree": int(sum(bool(d1[(r['log'], r['token'])]['sim8']) == r["hz10"]["sim8"] for r in rows)), "n": n},
       "sim8_fail_pct": {}, "classes": {hz: {k: int((c[hz] == k).sum()) for k in ("A", "B", "C", "pass")} for hz in H}}
for hz in H:
    p = ratio_ci(f[hz], one); res["sim8_fail_pct"][hz] = [round(100 * x, 3) for x in p[:3]]
dd = ratio_ci(f["hz20"] - f["hz10"], one)
res["sim8_fail_pct"]["hz20_minus_hz10_paired_pts"] = [round(100 * x, 3) for x in dd[:3]]
res["sim8_fail_discordant"] = {"hz10_only": int(((f["hz10"] == 1) & (f["hz20"] == 0)).sum()), "hz20_only": int(((f["hz10"] == 0) & (f["hz20"] == 1)).sum()), "both": int(((f["hz10"] == 1) & (f["hz20"] == 1)).sum())}
def med_ci(vals, strata_mask):
    """median over a masked set with a log-cluster bootstrap of the median."""
    idx = np.where(strata_mask)[0]
    if len(idx) == 0: return None
    est = []
    byl = {l: idx[inv[idx] == l] for l in range(L)}
    for d in draws:
        sel = np.concatenate([byl[l] for l in d]) if any(len(byl[l]) for l in d) else None
        if sel is not None and len(sel): est.append(np.median(vals[sel]))
    return [round(float(np.median(vals[idx])), 3), round(float(np.percentile(est, 2.5)), 3), round(float(np.percentile(est, 97.5)), 3), int(len(idx))]
m = {}
for hz in H:
    m[f"classC_own_set_{hz}"] = med_ci(mg[hz], c[hz] == "C")
    m[f"all_samples_{hz}"] = med_ci(mg[hz], np.ones(n, bool))
for hz in H:
    m[f"classC_of_hz10_set_evaluated_at_{hz}"] = med_ci(mg[hz], c["hz10"] == "C")
    m[f"classC_of_hz20_set_evaluated_at_{hz}"] = med_ci(mg[hz], c["hz20"] == "C")
m["all_samples_margin_hz20_minus_hz10_mean_m"] = [round(x, 3) for x in ratio_ci(mg["hz20"] - mg["hz10"], one)[:3]]
res["margin_m_median_with_ci_and_n"] = m
res["plan_difference_end_pose_m"] = {"median": round(float(np.median([np.hypot(r["hz20"]["end"][0]-r["hz10"]["end"][0], r["hz20"]["end"][1]-r["hz10"]["end"][1]) for r in rows])), 3)}
tr = {"hz10_sim8_fail_le_3pct": res["sim8_fail_pct"]["hz10"][0] <= 3, "hz20_sim8_fail_le_3pct": res["sim8_fail_pct"]["hz20"][0] <= 3,
      "hz20_classC_margin_ge_0.55": (m["classC_own_set_hz20"] or [0])[0] >= 0.55}
res["outcome_inputs"] = tr
json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))
