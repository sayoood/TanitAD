"""kprog_walker.py -- PREREG_MAPPATH AMENDMENT 1 K-PROG: an independent loop walker checks the progress-matched fan.  CPU."""
import json, sys
import numpy as np
z0 = np.load("F:/refcv9_mapsel/trace_s0.npz", allow_pickle=False)
z1 = np.load("F:/refcv9_mapsel/trace_s0_mappath.npz", allow_pickle=False)
fan0, fan1, sel = z0["fan"].astype(np.float64), z1["fan"].astype(np.float64), z0["sel"].astype(np.int64)
n, N, S, _ = fan0.shape

def walk(poly, s):
    V = [(0.0, 0.0)] + [tuple(p) for p in poly]
    acc, last = 0.0, None
    for a, b in zip(V[:-1], V[1:]):
        L = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        if L > 0:
            last = (a, b, L)
            if acc + L >= s:
                t = (s - acc) / L
                return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        acc += L
    a, b, L = last
    u = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
    return (V[-1][0] + (s - acc) * u[0], V[-1][1] + (s - acc) * u[1])

def arc_nat(p):
    q = np.vstack([[0.0, 0.0], p])
    return np.cumsum(np.hypot(*np.diff(q, axis=0).T))
rng = np.random.default_rng(0)
lens = np.hypot(*np.diff(np.concatenate([np.zeros((n, N, 1, 2)), fan0], 2), axis=2).transpose(3, 0, 1, 2)).sum(-1)
ok = np.argwhere(lens > 1e-6)
pick = ok[rng.choice(len(ok), 5000, replace=False)]
res = {}
for tag, eps in (("K-PROG", 0.0), ("MUTATION_s_plus_0.05", 0.05)):
    mx = 0.0
    for (w, c) in pick:
        k = int(rng.integers(0, S)) if tag == "K-PROG" else int(rng.integers(0, S))
        s = arc_nat(fan0[w, sel[w]])[k] + eps
        p = walk(fan0[w, c], s)
        mx = max(mx, abs(p[0] - fan1[w, c, k, 0]), abs(p[1] - fan1[w, c, k, 1]))
    res[tag] = {"max_abs_m": mx, "n": 5000}
res["K-PROG"]["pass"] = bool(res["K-PROG"]["max_abs_m"] <= 1e-6)
res["MUTATION_s_plus_0.05"]["red_as_required"] = bool(res["MUTATION_s_plus_0.05"]["max_abs_m"] > 1e-6)
json.dump(res, open("result/MAPPATH_KPROG.json", "w"), indent=1)
print(json.dumps(res))
