"""Decompose the 2 s min-TTC deficit of the learned v8F-mini arms (vs P10 and vs the human) by closing class.
min-TTC is CAPPED at 30 s when a path never closes (lead_metrics), so a mean delta mixes (A) both-close windows,
(B) windows only the arm closes on (a censoring flip: 30 -> finite) and (C) the reverse.
  python ttc_decomp.py <dk_perwindow.npz> <out.json>"""
import json, sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=False)
CAP = 30.0
out = {"source": sys.argv[1].replace("\\", "/").split("/")[-1], "cap_s": CAP, "reads": {}}
for k in ("k10", "k30"):
    lead = z[f"{k}__is_lead"]
    for a, b in [("M2b_s0", "P10"), ("M2b_s1", "P10"), ("M1_s0", "P10"), ("M1_s1", "P10"), ("M2b_s0", "human"), ("M2b_s1", "human"), ("P10", "human")]:
        ta, tb = z[f"{k}__{a}__min_ttc_s"], z[f"{k}__{b}__min_ttc_s"]
        m = lead & np.isfinite(ta) & np.isfinite(tb)
        ta, tb = ta[m], tb[m]
        n = int(m.sum())
        ca, cb = ta < CAP - 1e-9, tb < CAP - 1e-9
        cls = {"A_both_close": ca & cb, "B_only_a_closes": ca & ~cb, "C_only_b_closes": ~ca & cb, "D_neither": ~ca & ~cb}
        d = ta - tb
        row = {"n_used": n, "mean_delta_s": float(d.mean())}
        for name, s in cls.items():
            row[name] = {"n": int(s.sum()), "contrib_s": float(d[s].sum() / n),
                         "mean_delta_in_class_s": float(d[s].mean()) if s.any() else None,
                         "median_a_s": float(np.median(ta[s])) if s.any() else None, "median_b_s": float(np.median(tb[s])) if s.any() else None}
        # TTC below 10 s: the safety-relevant tail
        row["n_a_below_10s"], row["n_b_below_10s"] = int((ta < 10).sum()), int((tb < 10).sum())
        row["n_a_below_5s"], row["n_b_below_5s"] = int((ta < 5).sum()), int((tb < 5).sum())
        out["reads"][f"{k}:{a}-{b}"] = row
json.dump(out, open(sys.argv[2], "w"), indent=1)
for key, r in out["reads"].items():
    s = " | ".join(f"{c[:1]} n={r[c]['n']:3d} {r[c]['contrib_s']:+.2f}" for c in ("A_both_close", "B_only_a_closes", "C_only_b_closes"))
    print(f"{key:22s} n={r['n_used']} delta {r['mean_delta_s']:+.3f} | {s} | <10s {r['n_a_below_10s']}/{r['n_b_below_10s']} <5s {r['n_a_below_5s']}/{r['n_b_below_5s']}")
