#!/usr/bin/env python3
"""Scale check for the matcher (CPU only): `agent_slots.hungarian` vs scipy's
linear_sum_assignment on SYNTHETIC [N_queries, A] cost matrices built like `_match_cost`
(centre L1 + size L1 + class + presence), N in {100, 300}, A in {1..60}. Real matrices come
from the GPU run's --lever capture_costs; this only prices the shape."""
import json, statistics, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1] + "/stack")
from tanitad.models import agent_slots as AS
from scipy.optimize import linear_sum_assignment as lsa
rng = np.random.default_rng(0)
def mat(n, a):
    bp = np.c_[rng.uniform(0, 60, n), rng.uniform(-16, 16, n), rng.uniform(1, 5, (n, 2))]
    bt = np.c_[rng.uniform(0, 60, a), rng.uniform(-16, 16, a), rng.uniform(1, 5, (a, 2))]
    c = 5 * np.abs(bp[:, None, :2] - bt[None, :, :2]).sum(-1) + 2 * np.abs(bp[:, None, 2:] - bt[None, :, 2:]).sum(-1)
    c += -rng.uniform(0, 1, (n, a)) + 2.0 * rng.uniform(0, 3, (n, 1))
    return c
def t(fn, c, k=5):
    b = 1e9
    for _ in range(k):
        s = time.perf_counter(); fn(c); b = min(b, time.perf_counter() - s)
    return b
out = {}
for n in (100, 300):
    for a in (1, 2, 4, 8, 16, 30, 60):
        ms = [mat(n, a) for _ in range(20)]
        th = [t(AS.hungarian, m) for m in ms]
        ts = [t(lambda m: lsa(m.T), m) for m in ms]
        same = sum(np.array_equal(AS.hungarian(m)[0], lsa(m.T)[1]) for m in ms)
        out[f"N{n}_A{a}"] = {"hungarian_ms": round(1e3 * statistics.median(th), 4),
                             "scipy_ms": round(1e3 * statistics.median(ts), 4), "same_of_20": int(same)}
print(json.dumps(out, indent=1))
Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[2]).write_text(json.dumps({"host": "dev box i9-12900F, 1 thread", "synthetic": True, "rows": out}, indent=1), encoding="utf-8")
