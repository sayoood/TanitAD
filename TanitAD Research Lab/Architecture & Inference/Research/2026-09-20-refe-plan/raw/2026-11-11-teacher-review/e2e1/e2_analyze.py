#!/usr/bin/env python3
"""E2 analysis: aggregates, controls, log-cluster bootstrap (10,000 draws, seed 20260927, resampling the held-out LOGS).
python e2_analyze.py --rows <e2_s*.jsonl ...> --d1 <d1_navtrain.json> --out <json>"""
import argparse, json, glob
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--rows", nargs="+", required=True); ap.add_argument("--d1", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = []
for p in a.rows:
    for ln in open(p, encoding="utf-8"):
        rows.append(json.loads(ln))
err = [r for r in rows if "error" in r]
rows = sorted([r for r in rows if "error" not in r], key=lambda r: (r["log"], r["token"]))
d1 = {(r["log"], r["token"]): r for r in json.load(open(a.d1, encoding="utf-8"))["rows"]}
ARMS = ["teacher_tracked", "teacher_transparent", "human_tracked", "human_transparent"]
MET = ["nc", "dac", "ttc", "ep", "c", "pdms"]
X = {arm: {m: np.array([r[arm][m] for r in rows]) for m in MET} for arm in ARMS}
logs = np.array([r["log"] for r in rows]); ul, inv = np.unique(logs, return_inverse=True)
n = len(rows); L = len(ul)
rng = np.random.default_rng(20260927)
B = 10000
cnt = np.bincount(inv, minlength=L).astype(float)
draws = rng.integers(0, L, size=(B, L))
def cluster_ci(v):
    """ratio estimator: per-log sums / per-log counts, resampling logs; returns point, lo, hi (2.5/97.5 %)."""
    s = np.bincount(inv, weights=v, minlength=L)
    est = (s[draws].sum(1)) / (cnt[draws].sum(1))
    return float(v.mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))
def fmt(t): return [round(100 * t[0], 3), round(100 * t[1], 3), round(100 * t[2], 3)]
for arm in ARMS:                      # POST-HOC sensitivity (not pre-registered): comfort neutralised (C = 1) in every arm
    X[arm]["pdms_c1"] = np.array([r[arm]["mult"] * (5 * r[arm]["ep"] + 5 * r[arm]["ttc"] + 2.0) / 12.0 for r in rows])
MET2 = MET + ["pdms_c1"]
res = {"n_samples": n, "n_logs": int(L), "n_errors": len(err), "estimator": "log-cluster bootstrap (resample held-out logs with replacement, ratio mean), 10,000 draws, seed 20260927, 95% percentile interval",
       "arms": {}, "contrasts": {}, "controls": {}}
for arm in ARMS:
    res["arms"][arm] = {m: fmt(cluster_ci(X[arm][m])) for m in MET2}
def contrast(a1, a2, m):
    return fmt(cluster_ci(X[a1][m] - X[a2][m]))
for nm, (a1, a2) in {"teacher_transparent_minus_tracked": ("teacher_transparent", "teacher_tracked"),
                     "teacher_minus_human_transparent": ("teacher_transparent", "human_transparent"),
                     "teacher_minus_human_tracked": ("teacher_tracked", "human_tracked"),
                     "teacher_transparent_minus_human_tracked": ("teacher_transparent", "human_tracked"),
                     "human_transparent_minus_tracked": ("human_transparent", "human_tracked")}.items():
    res["contrasts"][nm] = {m: contrast(a1, a2, m) for m in MET2}
# controls
sim8 = np.array([bool(d1[(r["log"], r["token"])]["sim8"]) for r in rows])
raw20 = np.array([bool(d1[(r["log"], r["token"])]["raw20"]) for r in rows])
tdac_fail = X["teacher_tracked"]["dac"] == 0
res["controls"]["identity_tracked_dac_vs_banked_sim8"] = {
    "banked_sim8_fail": int(sim8.sum()), "rerun_tracked_dac0": int(tdac_fail.sum()),
    "per_sample_agreement": int((sim8 == tdac_fail).sum()), "n": n, "disagree": int((sim8 != tdac_fail).sum()),
    "note": "banked sim8 = d1_navtrain.json (navsim_dac.py simulated footprint, 54.7 s run); rerun = NAVSIM PDMScorer DAC on the vendored-verbatim simulator via the metric-cache path"}
tr_fail = X["teacher_transparent"]["dac"] == 0
res["controls"]["liveness_transparent_vs_tracked"] = {
    "dac_differs_on_n": int((tr_fail != tdac_fail).sum()),
    "pdms_differs_on_n": int((np.abs(X["teacher_transparent"]["pdms"] - X["teacher_tracked"]["pdms"]) > 1e-9).sum()),
    "transparent_dac0": int(tr_fail.sum()), "tracked_dac0": int(tdac_fail.sum()),
    "tracked_fail_transparent_ok": int((tdac_fail & ~tr_fail).sum()), "tracked_ok_transparent_fail": int((~tdac_fail & tr_fail).sum())}
res["controls"]["transparent_dac_vs_banked_raw20"] = {
    "banked_raw20_fail": int(raw20.sum()), "transparent_dac0": int(tr_fail.sum()),
    "overlap_fail": int((raw20 & tr_fail).sum()),
    "note": "raw20 = footprint at the 20 poses (0.2 s) in d1; transparent = scorer on 0.1 s cubic-spline states; same poses, finer grid, so counts need not be equal"}
res["controls"]["human_row_same_path"] = {"human_transparent_dac_pct": res["arms"]["human_transparent"]["dac"], "human_tracked_dac_pct": res["arms"]["human_tracked"]["dac"]}
# outcome evaluation (rules as written in TEACHER_REVIEW.md s6 E2, no threshold moved)
tdac = res["arms"]["teacher_transparent"]["dac"][0]
dh = res["contrasts"]["teacher_minus_human_transparent"]["pdms"]
dh_t = res["contrasts"]["teacher_transparent_minus_human_tracked"]["pdms"]
res["outcome_inputs"] = {"transparent_dac_pct": tdac, "A_dac_ge_99.3": tdac >= 99.3, "B_dac_lt_98.5": tdac < 98.5,
                         "teacher_minus_human_pdms_same_path_transparent_pts": dh, "A_within_1pt_same_path": abs(dh[0]) <= 1.0,
                         "teacher_transparent_minus_human_tracked_pdms_pts": dh_t}
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("n_samples", "n_logs", "n_errors", "arms", "controls", "outcome_inputs")}, indent=1))
print(json.dumps(res["contrasts"], indent=1))
