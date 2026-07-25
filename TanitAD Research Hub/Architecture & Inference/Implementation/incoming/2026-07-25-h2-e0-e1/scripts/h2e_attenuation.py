"""Root-cause diagnostic for the E1 FAIL: WHERE does the 2.22x go?

Three nested samples of increasing size, same code, same d = 3.0 m:
  (a) the sweep's own 80 clips             -> reproduces 2.22x
  (b) ALL 185 clips of the SAME two chunks -> same geography, bigger n
  (c) the 2,159 held-out clips             -> the E1 verdict
If (b) already collapses, the attenuation is SAMPLE SIZE + post-hoc threshold, not geography.
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

CL, CR, D = "v_camera_cross_left_120fov", "v_camera_cross_right_120fov", 3.0


def frames(a, d=D):
    x = a.assign(cf=a.dmin_cf <= d, unseen=~a.in_crop)
    x["S"] = (x[CL] | x[CR]) & x.unseen & x.cf
    x["F"] = x.in_crop & x.cf
    g = x.groupby(["clip_id", "gi"], observed=True).agg(
        S=("S", "sum"), F=("F", "sum"), dv=("ego_dv", "first")).reset_index()
    g["gate"] = (g.S > 0) & (g.F == 0)
    g["resp"] = g.dv <= -1.0
    return g


sw = pd.read_parquet(S + r"\h2e_sweepchunks.parquet").dropna(subset=["dmin_cf", "ego_dv"])
ho = pd.read_parquet(S + r"\h2e_heldout.parquet").dropna(subset=["dmin_cf", "ego_dv"])
sweep80 = set(pd.read_parquet(S + r"\crux3.parquet", columns=["clip_id"]).clip_id.unique())

samples = [("(a) sweep's own 80 clips (chunks 0036+0170)", sw[sw.clip_id.isin(sweep80)]),
           ("(b) ALL 185 clips of the same two chunks", sw),
           ("(b') the 105 UNUSED clips of those two chunks", sw[~sw.clip_id.isin(sweep80)]),
           ("(c) 2,159 HELD-OUT clips (22 other chunks + 1860/1864)", ho)]
out = []
print(f"{'sample':<58}{'clips':>7}{'n+':>8}{'P(r|+)':>9}{'P(r|-)':>9}{'lift':>8}  95% CI")
for name, A in samples:
    g = frames(A)
    L = paired_cluster_lift(g.resp.to_numpy(), g.gate.to_numpy(), g.clip_id.to_numpy())
    out.append(dict(sample=name, n_clips=int(A.clip_id.nunique()),
                    gate_rate=round(float(g.gate.mean()), 5), **L))
    print(f"{name:<58}{A.clip_id.nunique():>7,}{L['n_pos']:>8,}{L['p_resp_given_pos']:>9.2%}"
          f"{L['p_resp_given_neg']:>9.2%}{L['lift']:>8.2f}x  [{L['lo']:.2f}, {L['hi']:.2f}]")

# How lucky was the 80-clip draw? Resample 80 clips at a time from the held-out pool and ask
# how often a draw yields a lift >= 2.22 at d=3.0 (i.e. how special the headline number is).
g = frames(ho)
rng = np.random.default_rng(0)
clips = g.clip_id.unique()
idx = {c: np.where(g.clip_id.values == c)[0] for c in clips}
r, m = g.resp.to_numpy(), g.gate.to_numpy()
draws = []
for _ in range(2000):
    sel = np.concatenate([idx[c] for c in rng.choice(clips, 80, replace=False)])
    mm, rr = m[sel], r[sel]
    if mm.sum() < 5 or (~mm).sum() < 5 or rr[~mm].mean() <= 0:
        continue
    draws.append(rr[mm].mean() / rr[~mm].mean())
draws = np.asarray(draws)
sub = {"n_draws": int(draws.size), "subsample_clips": 80,
       "p_lift_ge_2.22": round(float((draws >= 2.22).mean()), 4),
       "p_lift_ge_1.5": round(float((draws >= 1.5).mean()), 4),
       "p_lift_le_1.0": round(float((draws <= 1.0).mean()), 4),
       "pct": {k: round(float(np.percentile(draws, v)), 3)
               for k, v in (("p2.5", 2.5), ("p50", 50), ("p97.5", 97.5))},
       "note": "80-clip SUBSAMPLES of the held-out pool, d=3.0 m fixed. Shows how much an "
               "80-episode sample can move the lift even with the threshold held fixed."}
print(f"\n-- how much can an 80-clip sample move the lift at a FIXED d = 3.0 m? --")
print(f"   2.5/50/97.5 pct of 80-clip subsample lifts: {sub['pct']['p2.5']:.2f} / "
      f"{sub['pct']['p50']:.2f} / {sub['pct']['p97.5']:.2f}")
print(f"   P(lift >= 2.22) = {sub['p_lift_ge_2.22']:.1%}   P(lift >= 1.5) = "
      f"{sub['p_lift_ge_1.5']:.1%}   P(lift <= 1.0) = {sub['p_lift_le_1.0']:.1%}")

json.dump({"nested_samples": out, "subsample_80": sub},
          open(OUT + r"\e1_attenuation.json", "w"), indent=2)
print(f"\nwrote {OUT}\\e1_attenuation.json")
