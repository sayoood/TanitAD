"""P6 — re-fit `n_slot_queries` on THIS sample (prereg §2 forbids inheriting it).

The denominator is the IN-GRID population (0 < cx <= 60, |cy| <= 16), because
`SlotDecodeRanges` decodes into exactly those extents — an agent 200 m behind the
ego is not representable, so counting it would size the head for a target the
decode cannot express.
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
GRID_X, GRID_Y = 60.0, 16.0
hist = Counter(); hist_vis = Counter(); hist_all = Counter()
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line); ag = r["agents"]
        n = sum(1 for d in ag if 0.0 < d["cx"] <= GRID_X and abs(d["cy"]) <= GRID_Y)
        nv = sum(1 for d in ag if 0.0 < d["cx"] <= GRID_X and abs(d["cy"]) <= GRID_Y
                 and int(d.get("occ", 0)) == 0)
        hist[n] += 1; hist_vis[nv] += 1; hist_all[len(ag)] += 1
def stats(h):
    tot = sum(h.values()); xs = np.repeat(list(h), list(h.values()))
    return {"mean": round(float(xs.mean()), 2), "median": int(np.median(xs)),
            "p90": int(np.percentile(xs, 90)), "p95": int(np.percentile(xs, 95)),
            "p99": int(np.percentile(xs, 99)), "max": int(xs.max()), "n_frames": tot}
s_ing = stats(hist)
NQ = int(s_ing["p99"]) + (1 if s_ing["p99"] % 2 else 0)   # p99, rounded up to even
over = {q: round(float(sum(v for k, v in hist.items() if k > q)) / s_ing["n_frames"], 4)
        for q in (16, 32, NQ, 48, 64)}
rec = {"_evidence_class": "MEASURED (ours; the 130 declared clips' own join)",
       "population_all_agents": stats(hist_all),
       "population_in_grid": s_ing,
       "population_in_grid_visible": stats(hist_vis),
       "n_slot_queries_FITTED": NQ,
       "rule": "the in-grid p99, rounded up to an even number",
       "overflow_frac_by_n_queries": over,
       "prior_run_2026-08-16": {"in_grid_p99": 31, "n_slot_queries": 32,
                                "overflow_at_16": 0.0904, "overflow_at_32": 0.0075},
       "note": ("this sample is LEAD-ENRICHED and denser than the corpus "
                "(corpus-wide in-grid p99 = 33 over all 2,308 joined clips), so "
                "the fit is made on the frames the head will actually see")}
Path(sys.argv[2]).write_text(json.dumps(rec, indent=1), encoding="utf-8")
print(json.dumps(rec, indent=1))
