"""Paired episode-cluster bootstrap on the LOW-TTC TAIL: share of LEAD windows whose min-TTC is < 5 s (and < 10 s), arm minus comparator.
Answers only "would another draw of EPISODES say this?" (H-ESTIM-SEED-1: the seed floor F rides beside it).
  python ttc_tail_ci.py <dk_perwindow.npz> <out.json>   (PYTHONPATH must reach taniteval)"""
import json, sys
import numpy as np
from taniteval import ci
z = np.load(sys.argv[1], allow_pickle=False)
out = {"estimator": "taniteval.ci.paired_episode_cluster_bootstrap", "n_boot": 2000, "seed": 0, "ci_module": ci.__file__.replace("\\", "/"), "reads": {}}
for k in ("k10", "k30"):
    lead, eid = z[f"{k}__is_lead"], z[f"{k}__eid"]
    for thr in (5.0, 10.0):
        share = {}
        for a in ("M1_s0", "M1_s1", "M2b_s0", "M2b_s1", "P10", "human"):
            t = z[f"{k}__{a}__min_ttc_s"]
            m = lead & np.isfinite(t)
            share[a] = float((t[m] < thr).mean())
        for a, b in [("M2b_s0", "P10"), ("M2b_s1", "P10"), ("M1_s0", "P10"), ("M1_s1", "P10"), ("M2b_s0", "human"), ("M2b_s1", "human"), ("P10", "human"), ("M2b_s0", "M1_s0"), ("M2b_s1", "M1_s1")]:
            ta, tb = z[f"{k}__{a}__min_ttc_s"], z[f"{k}__{b}__min_ttc_s"]
            m = lead & np.isfinite(ta) & np.isfinite(tb)
            r = ci.paired_episode_cluster_bootstrap((ta[m] < thr).astype(float), (tb[m] < thr).astype(float), eid[m], n_boot=2000, seed=0)
            out["reads"][f"{k}:ttc<{thr:g}s:{a}-{b}"] = {"n_used": int(m.sum()), "n_episodes": int(len(np.unique(eid[m]))), "share_a": share[a], "share_b": share[b], "ci": r}
json.dump(out, open(sys.argv[2], "w"), indent=1, default=float)
for key, r in out["reads"].items():
    c = r["ci"]
    lo, hi = c.get("lo", c.get("ci_lo")), c.get("hi", c.get("ci_hi"))
    print(f"{key:30s} n={r['n_used']} ep={r['n_episodes']} share {r['share_a']:.3f} vs {r['share_b']:.3f}  delta {c.get('delta', c.get('point', c.get('mean'))):+.4f} [{lo:+.4f}, {hi:+.4f}] {'sep' if (lo > 0 or hi < 0) else 'ns'}")
