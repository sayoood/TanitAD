#!/usr/bin/env python3
"""The ATTRIBUTION contrast the retraction requires: `rl` - `ctrl_const`.

Both arms start from the SAME frozen base and their BEFORE readouts are
bit-identical (asserted here), so the difference of their AFTER readouts on the
same windows isolates what the REWARD + the >=GT bar did, with the VETO -- present
and identical in both -- differenced out.

  ctrl_const - base   = the VETO alone
  rl - ctrl_const     = the REWARD + the >=GT bar
  rl - base           = both together, and attributes nothing

Same estimator and same separation rule as fan_safety_verdict.py.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fan_safety_verdict import FLAGS, LEAD_ONLY, EXTRAS, MIN_EFFECT, N_BOOT, SEED

RUN = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Admin\rl_rescope\run"
A, B = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("rl", "ctrl_const")
OUT = sys.argv[4] if len(sys.argv) > 4 else os.path.join(RUN, f"contrast_{A}_vs_{B}.json")

def rows(arm, which):
    with open(os.path.join(RUN, arm, f"readout_{which}.json"), encoding="utf-8") as fh:
        return json.load(fh)["per_window"]

ab, aa = rows(A, "before"), rows(A, "after")
bb, ba = rows(B, "before"), rows(B, "after")

# the contrast is only valid if both arms started from the same readout
ib = {r["wi"]: r for r in ab}; jb = {r["wi"]: r for r in bb}
common = sorted(set(ib) & set(jb))
keys = [k for k in ab[0] if any(k.endswith("_" + f) for f in FLAGS) or k in EXTRAS]
maxdiff = max(abs(ib[w][k] - jb[w][k]) for w in common for k in keys)
print(f"[contrast] BEFORE readouts identical across arms to {maxdiff:.3e} "
      f"({len(common)} windows) -- {'OK' if maxdiff < 1e-9 else 'NOT IDENTICAL, contrast is invalid'}")

ia = {r["wi"]: r for r in aa}; ja = {r["wi"]: r for r in ba}
rec = {"_what": f"attribution contrast {A} - {B} on the AFTER readouts (same windows)",
       "_evidence_class": "MEASURED (ours)", "_tier": "T0 readout, 120 fixed EVAL windows",
       "_before_readouts_max_abs_diff": maxdiff,
       "_rule": f"separated = CI excludes 0 AND not all-zero AND |delta| >= {MIN_EFFECT}",
       "_estimator": "paired episode-cluster bootstrap", "metrics": {}}
for k in keys:
    lead_only = any(f in k for f in LEAD_ONLY)
    per_ep = {}
    for w in common:
        if lead_only and not ib[w].get("has_lead"): continue
        per_ep.setdefault(ib[w]["eid"], []).append(ia[w][k] - ja[w][k])
    if not per_ep: continue
    d = np.array([float(np.mean(v)) for v in per_ep.values()])
    rng = np.random.default_rng(SEED)
    bs = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(N_BOOT)])
    lo, hi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
    delta = float(d.mean()); allz = bool(np.all(d == 0.0))
    rec["metrics"][k] = {"delta": delta, "lo": lo, "hi": hi, "n_episodes": int(d.size),
        "all_zero": allz,
        "separated": bool((lo > 0.0 or hi < 0.0)
                          and not allz and abs(delta) >= MIN_EFFECT),
        "population": "lead windows" if lead_only else "all windows"}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(rec, fh, indent=1, default=str)
sep = [k for k, v in rec["metrics"].items() if v["separated"]]
print(f"[contrast] {len(rec['metrics'])} metrics, SEPARATED {len(sep)} -> {OUT}")
