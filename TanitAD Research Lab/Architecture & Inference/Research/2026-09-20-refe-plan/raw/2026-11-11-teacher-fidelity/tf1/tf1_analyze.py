#!/usr/bin/env python3
"""TF-1 analysis: aggregates, controls, log-cluster bootstrap (10,000 draws, seed 20260927, resampling the 11 held-out
LOGS, ratio mean; E2's estimator verbatim). Prints ZZTF1_CONTROLS PASS|FAIL.
python tf1_analyze.py --rows <tf1_s*.jsonl ...> --e2 <e2_rows_1018.jsonl> --p5s <p5s_cubic.json> --out <json>"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--rows", nargs="+", required=True)
ap.add_argument("--e2", required=True)
ap.add_argument("--p5s", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = [json.loads(ln) for p in a.rows for ln in open(p, encoding="utf-8")]
err = [r for r in rows if "error" in r]
rows = sorted([r for r in rows if "error" not in r], key=lambda r: (r["log"], r["token"]))
keys = [(r["log"], r["token"]) for r in rows]
e2 = {(r["log"], r["token"]): r for r in (json.loads(ln) for ln in open(a.e2, encoding="utf-8"))}
p5 = {(r["log"], r["token"]): r for r in json.load(open(a.p5s, encoding="utf-8"))["rows"]}
ARMS = ["T_tracked", "Tc20s_tracked", "Tc20_tracked", "Tw_tracked", "H_tracked", "T_transparent", "Tc20s_transparent"]
MET = ["nc", "dac", "ddc", "ttc", "ep", "c", "pdms"]
X = {arm: {m: np.array([r[arm][m] for r in rows]) for m in MET} for arm in ARMS}
X["T_tracked"]["zero"] = (X["T_tracked"]["pdms"] == 0).astype(float)
t_sim8 = np.array([bool(p5[k]["T"]["sim8"]) for k in keys])        # P5-S: True = the tracked T plan violates DAC
# DERIVED, post hoc (P5 s5 / P5-S s3 "Tsel-s"): Tc20s only where T fails sim8, else T. Descriptive, no verdict.
X["Tsel_s_tracked_DERIVED"] = {m: np.where(t_sim8, X["Tc20s_tracked"][m], X["T_tracked"][m]) for m in MET}
for arm in list(X):
    X[arm]["zero"] = (X[arm]["pdms"] == 0).astype(float)
MET2 = MET + ["zero"]
logs = np.array([k[0] for k in keys])
ul, inv = np.unique(logs, return_inverse=True)
n, L = len(rows), len(ul)
rng = np.random.default_rng(20260927)
B = 10000
cnt = np.bincount(inv, minlength=L).astype(float)
draws = rng.integers(0, L, size=(B, L))
den = cnt[draws].sum(1)


def boot(v):
    s = np.bincount(inv, weights=v, minlength=L)
    return s[draws].sum(1) / den


def ci(v):
    e = boot(v)
    return [round(100 * float(v.mean()), 3), round(100 * float(np.percentile(e, 2.5)), 3),
            round(100 * float(np.percentile(e, 97.5)), 3)]


res = {"n_samples": n, "n_logs": int(L), "n_errors": len(err), "errors": err[:5],
       "estimator": "log-cluster bootstrap (resample the 11 held-out navtrain logs with replacement, ratio mean), "
                    "10,000 draws, seed 20260927, 95% percentile interval; answers 'another draw of LOGS'; labels, "
                    "tracker and scorer are deterministic (no training or inference variance on this arm)",
       "tier": "NAVSIM v1.1 PDMS, non-reactive pseudo-simulation (LQR + kinematic bicycle) on navtrain HELD-OUT logs; NOT navtest",
       "arms": {arm: {m: ci(X[arm][m]) for m in MET2} for arm in X}, "contrasts": {}, "controls": {}}
CON = {"Tc20s_minus_T_tracked": ("Tc20s_tracked", "T_tracked"),
       "Tc20s_tracked_minus_H_tracked": ("Tc20s_tracked", "H_tracked"),
       "T_tracked_minus_H_tracked": ("T_tracked", "H_tracked"),
       "Tc20s_tracked_minus_T_transparent": ("Tc20s_tracked", "T_transparent"),
       "T_transparent_minus_T_tracked": ("T_transparent", "T_tracked"),
       "Tc20_linear_minus_Tc20s_tracked": ("Tc20_tracked", "Tc20s_tracked"),
       "Tw_minus_T_tracked_REGRESSION_ARM": ("Tw_tracked", "T_tracked"),
       "Tc20s_transparent_minus_T_transparent": ("Tc20s_transparent", "T_transparent"),
       "Tsel_s_DERIVED_minus_T_tracked": ("Tsel_s_tracked_DERIVED", "T_tracked"),
       "Tsel_s_DERIVED_minus_Tc20s_tracked": ("Tsel_s_tracked_DERIVED", "Tc20s_tracked")}
for nm, (a1, a2) in CON.items():
    res["contrasts"][nm] = {m: ci(X[a1][m] - X[a2][m]) for m in MET2}
# share of the tracked-to-transparent PDMS gap that the tracker-aware conversion closes, per bootstrap draw
num, dnm = boot(X["Tc20s_tracked"]["pdms"] - X["T_tracked"]["pdms"]), boot(X["T_transparent"]["pdms"] - X["T_tracked"]["pdms"])
ok = np.abs(dnm) > 1e-9
pt = float((X["Tc20s_tracked"]["pdms"] - X["T_tracked"]["pdms"]).mean() / (X["T_transparent"]["pdms"] - X["T_tracked"]["pdms"]).mean())
res["gap_closed_share_pdms"] = [round(pt, 4), round(float(np.percentile(num[ok] / dnm[ok], 2.5)), 4),
                                round(float(np.percentile(num[ok] / dnm[ok], 97.5)), 4)]
# ---- controls
C = res["controls"]
for arm, e2arm in (("T_tracked", "teacher_tracked"), ("H_tracked", "human_tracked"), ("T_transparent", "teacher_transparent")):
    md = max(abs(r[arm][m] - e2[k][e2arm][m]) for r, k in zip(rows, keys) for m in ("nc", "dac", "ttc", "ep", "c", "pdms"))
    miss = sum(1 for k in keys if k not in e2)
    C[f"identity_{arm}_vs_E2_{e2arm}"] = {"max_abs_diff": md, "missing_in_E2": miss, "PASS": md == 0.0 and miss == 0}
for arm, p5ref in (("T_tracked", "T"), ("Tc20s_tracked", "Tc20"), ("Tw_tracked", "Tw")):
    f_sc = X[arm]["dac"] == 0
    f_p5 = np.array([bool(p5[k][p5ref]["sim8"]) for k in keys])
    C[f"join_{arm}_dac0_vs_P5S_{p5ref}_sim8"] = {"scorer_dac0": int(f_sc.sum()), "p5s_sim8_fail": int(f_p5.sum()),
                                               "agree": int((f_sc == f_p5).sum()), "n": n,
                                               "PASS": bool((f_sc == f_p5).sum() >= n - 2)}
reg = res["contrasts"]["Tw_minus_T_tracked_REGRESSION_ARM"]["pdms"]
C["deliberate_regression_Tw_scores_below_T"] = {"delta_pdms_pts": reg, "PASS": reg[2] < 0}
C["liveness_Tc20s_differs_from_T"] = {
    "pdms_differs_on_n": int((np.abs(X["Tc20s_tracked"]["pdms"] - X["T_tracked"]["pdms"]) > 1e-9).sum()),
    "dac_differs_on_n": int((X["Tc20s_tracked"]["dac"] != X["T_tracked"]["dac"]).sum()),
    "T_fail_Tc20s_ok": int(((X["T_tracked"]["dac"] == 0) & (X["Tc20s_tracked"]["dac"] > 0)).sum()),
    "T_ok_Tc20s_fail": int(((X["T_tracked"]["dac"] > 0) & (X["Tc20s_tracked"]["dac"] == 0)).sum())}
C["liveness_Tc20s_differs_from_T"]["PASS"] = C["liveness_Tc20s_differs_from_T"]["pdms_differs_on_n"] > 0
C["coverage"] = {"n": n, "errors": len(err), "PASS": n == 1018 and not err}
# EP anatomy: does compensation change progress on rows where both plans are DAC/NC-alive?
mT =np.array([r["T_tracked"]["mult"] for r in rows]); mC = np.array([r["Tc20s_tracked"]["mult"] for r in rows])
both = (mT > 0) & (mC > 0)
res["ep_on_rows_alive_in_both"] = {"n": int(both.sum()),
                                   "T_tracked_ep": round(100 * float(X["T_tracked"]["ep"][both].mean()), 3),
                                   "Tc20s_tracked_ep": round(100 * float(X["Tc20s_tracked"]["ep"][both].mean()), 3),
                                   "T_transparent_ep": round(100 * float(X["T_transparent"]["ep"][both].mean()), 3),
                                   "H_tracked_ep": round(100 * float(X["H_tracked"]["ep"][both].mean()), 3)}
res["zero_counts"] = {arm: int((X[arm]["pdms"] == 0).sum()) for arm in X}
res["dac0_counts"] = {arm: int((X[arm]["dac"] == 0).sum()) for arm in X}
res["nc0_counts"] = {arm: int((X[arm]["nc"] == 0).sum()) for arm in X}
passed = all(v["PASS"] for v in C.values())
res["controls_pass"] = passed
json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
print(json.dumps({k: res[k] for k in ("n_samples", "n_logs", "n_errors", "arms", "gap_closed_share_pdms",
                                      "ep_on_rows_alive_in_both", "zero_counts", "dac0_counts", "nc0_counts")}, indent=1))
print(json.dumps(res["contrasts"], indent=1))
print(json.dumps(C, indent=1))
print("ZZTF1_CONTROLS", "PASS" if passed else "FAIL")
