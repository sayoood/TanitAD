"""H2 L2 — post-verdict ROBUSTNESS. Descriptive only; nothing here can move the CONFIRM verdict.

Two questions the CONFIRM run raised and did not settle:

  Q1  The pooled lift (2.41x) is NOT reproduced inside either braking-state stratum
      (free-flow 1.16x, already-braking 1.39x, neither separated). How much of the association
      survives joint adjustment for ego SPEED and BRAKING STATE?
  Q2  Two CONFIRM chunks report a lift of exactly 0.00x (1860 n+=9, 2500 n+=160). Is chunk 2500 a
      real null or a degenerate response channel?

usage:  python l2_robust.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
E01 = os.path.abspath(os.path.join(HERE, "..", "..", "2026-07-25-h2-e0-e1", "scripts"))
sys.path.insert(0, HERE)
sys.path.insert(0, E01)
from h2e_stats import paired_cluster_lift                              # noqa: E402
from taniteval.ci import _draws, episode_index                         # noqa: E402
from l2_label import CONFIRM_CHUNKS, response_r2, trigger_l2           # noqa: E402

TAB = (r"C:\Users\Admin\AppData\Local\Temp\claude"
       r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
       r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\l2tab")
OUT = os.path.abspath(os.path.join(HERE, ".."))
SPEED_BINS = [3, 6, 9, 12, 15]

dev = json.load(open(os.path.join(OUT, "l2_dev.json")))
TAU, RESOLVABLE = dev["tau_star"], dev["A2_decision_exclude_unresolvable"]

D = pd.concat([pd.read_parquet(os.path.join(TAB, f"l2_{c}.parquet")) for c in CONFIRM_CHUNKS],
              ignore_index=True)
eid = D.clip_id.to_numpy()
R = response_r2(D, freeflow=None)
g = trigger_l2(D, TAU, resolvable=RESOLVABLE)
v = D.ego_v.to_numpy()
brk = D.alon_pre.to_numpy() < -0.5                      # "already braking at t"
res = {"tau": TAU, "note": "descriptive robustness, run AFTER the verdict; cannot move it"}


def mh(resp, gate, strat):
    num = den = 0.0
    for s in np.unique(strat):
        m = strat == s
        n1, n0, N = gate[m].sum(), (~gate[m]).sum(), m.sum()
        if n1 == 0 or n0 == 0:
            continue
        num += resp[m & gate].sum() * n0 / N
        den += resp[m & ~gate].sum() * n1 / N
    return num / den if den > 0 else np.nan


def boot_mh(strat, nb=2000):
    uniq, idx = episode_index(eid)
    pt = mh(R, g, strat)
    b = [x for x in (mh(R[s], g[s], strat[s]) for s in _draws(uniq, idx, nb, 0)) if np.isfinite(x)]
    lo, hi = np.percentile(b, [2.5, 97.5]) if len(b) > 50 else (np.nan, np.nan)
    return {"lift": round(float(pt), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "n_strata": int(len(np.unique(strat))), "n_draws": len(b),
            "estimator": "Mantel-Haenszel pooled risk ratio, episode-cluster bootstrap B=2000"}


sb = np.digitize(v, SPEED_BINS)
print("== Q1: joint adjustment ==")
res["MH_speed_only"] = boot_mh(sb)
res["MH_brakingstate_only"] = boot_mh(brk.astype(int))
res["MH_speed_x_brakingstate"] = boot_mh(sb * 2 + brk.astype(int))
for k in ("MH_speed_only", "MH_brakingstate_only", "MH_speed_x_brakingstate"):
    r = res[k]
    print(f"   {k:26s} {r['lift']:6.2f}x [{r['lo']:.2f}, {r['hi']:.2f}]  ({r['n_strata']} strata)")
res["unadjusted_for_reference"] = {"lift": 2.4128, "lo": 1.3998, "hi": 3.7041}
print(f"   {'UNADJUSTED (the verdict)':26s} {2.41:6.2f}x [1.40, 3.70]")

# exposure balance across the adjustment variables
res["confound_balance"] = {
    "P(already_braking | trigger+)": round(float(brk[g].mean()), 4),
    "P(already_braking | trigger-)": round(float(brk[~g].mean()), 4),
    "mean_v_trigger+": round(float(v[g].mean()), 3),
    "mean_v_trigger-": round(float(v[~g].mean()), 3),
    "P(R2 | already_braking)": round(float(R[brk].mean()), 4),
    "P(R2 | free_flow)": round(float(R[~brk].mean()), 4)}
print("\n   balance:", json.dumps(res["confound_balance"], indent=6))

# ---------------------------------------------------------------- Q2: the two 0.00x chunks
print("\n== Q2: the zero-lift chunks ==")
q2 = {}
for ch in ("2500", "1860", "1900", "0906"):
    m = (D.chunk == ch).to_numpy()
    q2[ch] = {"n_frames": int(m.sum()), "n_trigger": int(g[m].sum()),
              "response_base_rate": round(float(R[m].mean()), 5),
              "n_response_frames": int(R[m].sum()),
              "p05_alon_fut_min": round(float(np.percentile(D.alon_fut_min.to_numpy()[m], 5)), 3),
              "mean_ego_v": round(float(v[m].mean()), 2),
              "trigger_rate": round(float(g[m].mean()), 5)}
    print(f"   chunk {ch}: n+={q2[ch]['n_trigger']:4d}  R2 base {q2[ch]['response_base_rate']:.3%} "
          f"({q2[ch]['n_response_frames']} frames)  p05 alon_fut {q2[ch]['p05_alon_fut_min']:+.2f}  "
          f"mean v {q2[ch]['mean_ego_v']:.1f}")
res["Q2_chunk_diagnostics"] = q2

# ------------------------------------- leave-one-chunk-out: is the verdict driven by any one chunk?
loo = []
for ch in sorted(D.chunk.unique()):
    m = (D.chunk != ch).to_numpy()
    L = paired_cluster_lift(R[m], g[m], eid[m], n_boot=800)
    loo.append(dict(dropped=str(ch), lift=L["lift"], lo=L["lo"], hi=L["hi"],
                    excludes_1=bool(L["excludes_1_above"])))
res["leave_one_chunk_out"] = loo
n_ok = sum(x["excludes_1"] for x in loo)
res["leave_one_chunk_out_summary"] = {
    "n_chunks": len(loo), "n_still_excluding_1": n_ok,
    "min_lift": min(x["lift"] for x in loo), "max_lift": max(x["lift"] for x in loo)}
print(f"\n== leave-one-chunk-out: {n_ok}/{len(loo)} still exclude 1.0 "
      f"(lift range {res['leave_one_chunk_out_summary']['min_lift']:.2f}-"
      f"{res['leave_one_chunk_out_summary']['max_lift']:.2f}) ==")
for x in loo:
    print(f"   -{x['dropped']}: {x['lift']:5.2f}x [{x['lo']:.2f}, {x['hi']:.2f}]"
          f"  {'excl 1' if x['excludes_1'] else 'INCLUDES 1'}")

json.dump(res, open(os.path.join(OUT, "l2_robustness.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'l2_robustness.json')}")
