"""EXPLORATORY confound check on the E1 FAIL — it CANNOT change the verdict.

The response is "decelerates >= 1 m/s over 4 s". P(response) depends strongly on ego speed: a
frame at 3 m/s has little room to shed 1 m/s. If gate-positive frames sit at systematically
different speeds, the raw lift is confounded -- in EITHER direction. This asks whether a real
effect was MASKED by that confound.

Pre-registered estimand = the raw lift (h2e_e1.py). This file is reported as exploratory.
"""
import json
import sys

import numpy as np
import pandas as pd

S = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad"
OUT = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Hub"
       r"\Architecture & Inference\Implementation\incoming\2026-07-25-h2-e0-e1")
sys.path.insert(0, S)
from h2e_stats import paired_cluster_lift   # noqa: E402
from taniteval.ci import _draws, episode_index   # noqa: E402

CL, CR, D = "v_camera_cross_left_120fov", "v_camera_cross_right_120fov", 3.0
A = pd.read_parquet(S + r"\h2e_heldout.parquet").dropna(subset=["dmin_cf", "ego_dv"])
A = A.assign(cf=A.dmin_cf <= D, unseen=~A.in_crop)
A["S"] = (A[CL] | A[CR]) & A.unseen & A.cf
A["F"] = A.in_crop & A.cf
g = A.groupby(["clip_id", "gi"], observed=True).agg(
    S=("S", "sum"), F=("F", "sum"), dv=("ego_dv", "first"),
    v=("ego_v", "first")).reset_index()
g["gate"] = (g.S > 0) & (g.F == 0)
g["resp"] = g.dv <= -1.0

print("=== is the gate confounded with ego speed? ===")
print(f"  mean ego speed | gate+ : {g.v[g.gate].mean():.2f} m/s   "
      f"(p50 {g.v[g.gate].median():.2f})")
print(f"  mean ego speed | gate- : {g.v[~g.gate].mean():.2f} m/s   "
      f"(p50 {g.v[~g.gate].median():.2f})")

BINS = [0, 3, 6, 9, 12, 16, 100]
g["vb"] = pd.cut(g.v, BINS, right=False)
rows = []
print(f"\n{'speed bin (m/s)':<18}{'n+':>7}{'n-':>9}{'P(r|+)':>9}{'P(r|-)':>9}{'lift':>8}  95% CI")
for b, gb in g.groupby("vb", observed=True):
    if gb.gate.sum() < 30:
        rows.append(dict(bin=str(b), n_pos=int(gb.gate.sum()), lift=None))
        print(f"{str(b):<18}{int(gb.gate.sum()):>7}  (too few)")
        continue
    L = paired_cluster_lift(gb.resp.to_numpy(), gb.gate.to_numpy(), gb.clip_id.to_numpy(),
                            n_boot=1000)
    rows.append(dict(bin=str(b), n_pos=L["n_pos"], n_neg=L["n_neg"], lift=L["lift"],
                     lo=L["lo"], hi=L["hi"], p_pos=L["p_resp_given_pos"],
                     p_neg=L["p_resp_given_neg"]))
    print(f"{str(b):<18}{L['n_pos']:>7,}{L['n_neg']:>9,}{L['p_resp_given_pos']:>9.2%}"
          f"{L['p_resp_given_neg']:>9.2%}{L['lift']:>8.2f}x  [{L['lo']:.2f}, {L['hi']:.2f}]")


def mh_lift(sub):
    """Speed-stratum-weighted (Mantel-Haenszel style) pooled lift: reweight the NEGATIVE arm to
    the positive arm's speed distribution, so both arms are compared at matched speed."""
    w = sub.groupby("vb", observed=True).gate.sum()
    p1 = sub[sub.gate].groupby("vb", observed=True).resp.mean()
    p0 = sub[~sub.gate].groupby("vb", observed=True).resp.mean()
    k = w.index.intersection(p1.dropna().index).intersection(p0.dropna().index)
    if not len(k) or w[k].sum() == 0:
        return np.nan
    num = float((w[k] * p1[k]).sum() / w[k].sum())
    den = float((w[k] * p0[k]).sum() / w[k].sum())
    return num / den if den > 0 else np.nan


point = mh_lift(g)
uniq, idx_by_ep = episode_index(g.clip_id.to_numpy())
boots = []
for sel in _draws(uniq, idx_by_ep, 2000, 0):
    val = mh_lift(g.iloc[sel])
    if np.isfinite(val):
        boots.append(val)
boots = np.asarray(boots)
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"\n  SPEED-MATCHED pooled lift : {point:.2f}x  95% CI [{lo:.2f}, {hi:.2f}]   "
      f"(raw pre-registered lift was 1.16x [1.00, 1.33])")
print("  -> the confound does not hide an effect; the verdict is unchanged."
      if not (lo > 1.5) else "  -> WARNING: speed-matching materially changes the picture.")

json.dump({"note": "EXPLORATORY. Pre-registered estimand is the raw lift; this cannot change the "
                   "E1 verdict.",
           "mean_speed_gate_pos": round(float(g.v[g.gate].mean()), 3),
           "mean_speed_gate_neg": round(float(g.v[~g.gate].mean()), 3),
           "by_speed_bin": rows,
           "speed_matched_pooled_lift": {"lift": round(float(point), 4),
                                         "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                                         "n_boot": int(boots.size),
                                         "estimator": "episode-cluster bootstrap of a "
                                                      "speed-stratum-weighted (MH) lift"}},
          open(OUT + r"\e1_confound_check.json", "w"), indent=2)
print(f"\nwrote {OUT}\\e1_confound_check.json")
