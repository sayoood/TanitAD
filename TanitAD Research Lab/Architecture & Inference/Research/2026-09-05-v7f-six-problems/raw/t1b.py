#!/usr/bin/env python3
"""Closed-loop vs hold-action comparison + echo controls, all banked T1 arms."""
import json, os

ROOT = "/home/nvidia/t1dumps"
ARMS = ["emao14_30k", "emao14_30k_tauramp", "o14fut30k"]

METRICS = ["ade_dense_m", "fde_last_m", "LAT_cross_mae_m", "LAT_heading_mae_deg",
           "LON_speed_mae_mps", "LON_along_mae_m"]

for a in ARMS:
    p = os.path.join(ROOT, a, "t1.json")
    if not os.path.exists(p):
        print("MISSING", a); continue
    j = json.load(open(p))
    print("=" * 80)
    print("ARM:", a, " ckpt:", str(j.get("ckpt"))[:70])
    print("  n_episodes:", j.get("n_episodes"), " n_windows:", j.get("n_windows"),
          " mode:", j.get("mode"), " corpus_key:", str(j.get("corpus_key"))[:40])
    print("  arm_keys:", j.get("arm_keys"))
    arms = j.get("arms", {})
    print("  ARMS PRESENT:", list(arms))
    # cl vs ha table
    print("  %-24s %12s %12s %10s" % ("metric", "cl", "ha(hold)", "cl-ha"))
    for m in METRICS:
        row = []
        for k in ("cl", "ha"):
            v = arms.get(k, {}).get("intervals", {}).get("metrics", {}).get(m, {})
            row.append(v.get("mean"))
        if row[0] is None and row[1] is None:
            continue
        d = (row[0] - row[1]) if (row[0] is not None and row[1] is not None) else None
        print("  %-24s %12s %12s %10s" % (m,
              round(row[0], 4) if row[0] is not None else "-",
              round(row[1], 4) if row[1] is not None else "-",
              round(d, 4) if d is not None else "-"))
    # paired decision-grade
    pdg = j.get("paired_decision_grade")
    if pdg:
        print("  PAIRED DECISION-GRADE:")
        print("   ", json.dumps(pdg)[:1400])
    # echo / copy detector anywhere
    def find(o, pre="", hits=None, d=0):
        if hits is None: hits = {}
        if d > 7: return hits
        if isinstance(o, dict):
            for k, v in o.items():
                if any(t in str(k).lower() for t in ("echo", "copy", "holdv0", "verdict")):
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        hits[pre + "/" + k] = v
                    else:
                        hits[pre + "/" + k] = json.dumps(v)[:300]
                find(v, pre + "/" + str(k), hits, d + 1)
        return hits
    h = find(j)
    print("  ECHO / HOLDV0 / VERDICT FIELDS:")
    for k in sorted(h):
        print("    %-58s %s" % (k[:58], h[k]))
