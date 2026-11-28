"""T0 twin of the sim session's t2_heads.py: Shapley split of agg(pick) - agg(best-agg holder), real frames, u400.

Population: the 154 CRUISE tokens of refe_ego_probe_20261106. Decaying pick: hold_d < 0.95, hold_d = end distance /
(v0 x 4 s). Holder: best-agg proposal with hold_d >= 0.95. Rule: navsim_v1_lane = NC x DAC x (5 EP + 5 TTC + 2 C)/12 x lane.
Descriptive, no bootstrap. Run with a numpy-2 interpreter, `python -I t0_margin_shapley.py <out.json>`.
"""
import itertools
import json
import math
import sys

import numpy as np

ROOT = "E:/Projects/TanitAD-artifacts/refe_ego_probe_20261106/"
d = np.load(ROOT + "full/results.npz", allow_pickle=True)
L = d["logits_all_A0"].astype(np.float64)
T = d["traj_all_A0"].astype(np.float64)
tok = [str(t) for t in d["token"]]
st = np.array([str(s) for s in d["stratum"]])
v0 = d["ego_real"][:, 6].astype(np.float64)
pk = d["pick"][:, 0]
c = json.load(open(ROOT + "tokens_cruise.json", encoding="utf-8"))
cs = set(c if isinstance(c, list) and isinstance(c[0], str) else (c.get("tokens") if isinstance(c, dict) else [x["token"] for x in c]))
cru = np.array([t in cs for t in tok])
P = 1 / (1 + np.exp(-L))
HEADS = [0, 1, 2, 3, 4, 6]
NAMES = ["NC", "DAC", "EP", "TTC", "C", "lane"]
n = len(HEADS)


def agg(p):
    return p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12 * p[..., 6]


A = agg(P)
assert (A.argmax(1) == pk).all(), "the rule does not reproduce the banked picks"
hd = np.hypot(T[:, :, 19, 0], T[:, :, 19, 1]) / (np.maximum(v0, 1e-6)[:, None] * 4.0)


def shapley(a, b):
    out = np.zeros(n)

    def f(S):
        p = b.copy()
        for i in S:
            p[HEADS[i]] = a[HEADS[i]]
        return agg(p)

    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(n):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in itertools.combinations(others, r):
                out[i] += w * (f(S + (i,)) - f(S))
    return out


rows, no_holder = [], 0
for i in np.where(cru)[0]:
    k = pk[i]
    if hd[i, k] >= 0.95:
        continue
    hold = np.where(hd[i] >= 0.95)[0]
    if len(hold) == 0:
        no_holder += 1
        continue
    b = hold[A[i, hold].argmax()]
    s = shapley(P[i, k], P[i, b])
    m = A[i, k] - A[i, b]
    assert abs(s.sum() - m) < 1e-9, "shares do not sum to the margin"
    rows.append((int(i), s, float(m)))


def summary(sel):
    S = np.array([r[1] for r in sel])
    M = np.array([r[2] for r in sel])
    top = S.argmax(1)
    pos = M > 0
    return {
        "n": len(sel),
        "margin_positive": int(pos.sum()),
        "largest_term_count": {NAMES[j]: int((top == j).sum()) for j in range(n)},
        "median_share_of_margin": {NAMES[j]: round(float(np.median(S[pos, j] / M[pos])), 3) for j in range(n)},
    }


res = {
    "what": "T0, real frames, u400, 154 CRUISE tokens; descriptive, no bootstrap, no training or inference replicate",
    "n_cruise": int(cru.sum()),
    "picks_reproduced": "1058/1058",
    "decaying_picks_with_no_holder": no_holder,
    "all": summary(rows),
    "by_stratum": {s: summary([r for r in rows if st[r[0]] == s]) for s in sorted(set(st[cru]))},
}
json.dump(res, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print(json.dumps(res))
