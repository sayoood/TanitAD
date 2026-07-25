"""FIDELITY CHECK — does the vectorised rebuild reproduce the substrate audit's own numbers?

Runs the SWEEP definition on the SWEEP's own 80 clips and compares against
H2_SUBSTRATE_AND_LABELING.md Sec 6.3/6.4:  gate rate 1.83 %, label rate 0.91 %,
lift 2.22x [1.30, 3.14] @3 m, 0.43x [0.24, 0.71] @6 m.
A rewrite that cannot reproduce the number it is auditing is not admissible.
"""
import json
import sys

import numpy as np
import pandas as pd

S = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad"
sys.path.insert(0, S)
from h2e_stats import paired_cluster_lift   # noqa: E402

CL = "v_camera_cross_left_120fov"
CR = "v_camera_cross_right_120fov"

sweep_clips = set(pd.read_parquet(S + r"\crux3.parquet", columns=["clip_id"]).clip_id.unique())
A = pd.read_parquet(S + r"\h2e_sweepchunks.parquet")
A = A[A.clip_id.isin(sweep_clips)].dropna(subset=["dmin_cf", "ego_dv"])
print(f"sweep clips matched: {A.clip_id.nunique()} (expect 80)   agent-frames {len(A):,}")


def frames(A, d):
    a = A.assign(cf=A.dmin_cf <= d, unseen=~A.in_crop)
    a["S"] = (a[CL] | a[CR]) & a.unseen & a.cf
    a["F"] = a.in_crop & a.cf
    g = a.groupby(["clip_id", "gi"], observed=True).agg(
        S=("S", "sum"), F=("F", "sum"), dv=("ego_dv", "first")).reset_index()
    g["gate"] = (g.S > 0) & (g.F == 0)
    g["resp"] = g.dv <= -1.0
    return g


out = {"purpose": "reproduce H2_SUBSTRATE_AND_LABELING Sec6.3/6.4 on its own 80 clips",
       "n_clips": int(A.clip_id.nunique()), "sweep": []}
print(f"\n{'d':>5} {'pos%':>7} {'n+':>6} {'P(r|+)':>8} {'P(r|-)':>8} {'lift':>7}  CI")
for d in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
    g = frames(A, d)
    r = paired_cluster_lift(g.resp.to_numpy(), g.gate.to_numpy(), g.clip_id.to_numpy())
    rec = dict(d_m=d, gate_rate=round(float(g.gate.mean()), 5), **r)
    out["sweep"].append(rec)
    print(f"{d:5.1f} {g.gate.mean():7.2%} {int(g.gate.sum()):6,} "
          f"{r['p_resp_given_pos']:8.2%} {r['p_resp_given_neg']:8.2%} "
          f"{r['lift']:7.2f}x  [{r['lo']:.2f}, {r['hi']:.2f}]")

g3 = frames(A, 3.0)
lab = g3.gate & g3.resp
out["label_rate_at_3m"] = round(float(lab.mean()), 5)
out["gate_rate_at_3m"] = round(float(g3.gate.mean()), 5)
out["published_gate_rate"] = 0.0183
out["published_label_rate"] = 0.0091
out["published_lift_3m"] = [2.22, 1.30, 3.14]
out["published_lift_6m"] = [0.43, 0.24, 0.71]
print(f"\ngate rate @3 m {g3.gate.mean():.2%}  (published 1.83 %)")
print(f"label rate @3 m {lab.mean():.2%}  (published 0.91 %)")
json.dump(out, open(S + r"\h2e_fidelity.json", "w"), indent=2)
print("\nwrote h2e_fidelity.json")
