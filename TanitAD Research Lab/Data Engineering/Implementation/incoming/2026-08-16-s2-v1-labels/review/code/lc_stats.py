"""P1 statistics: exact permutation tests over the n=18 adjudication.

n=4 CORRECT vs n=14 WRONG -> C(18,4)=3060 assignments, so the two-sided
Mann-Whitney p is computed EXACTLY by enumeration. Minimum attainable
two-sided p = 2/3060 = 6.54e-4 (perfect separation). Multiplicity is handled
by (a) Bonferroni over the tested battery and (b) a max-statistic permutation
null over the SAME 3060 assignments, which is exact and not conservative.
"""
import json
import math
import os
from itertools import combinations

SP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")

rows = json.load(open(os.path.join(SP, "lc_feat.json"), encoding="utf-8"))
lab = [r for r in rows if r["gated"] and r.get("pi") in ("wrong", "correct")]
lab.sort(key=lambda r: r["clip_id"])
N = len(lab)
y = [1 if r["pi"] == "correct" else 0 for r in lab]      # 1 = REAL lane change
nc = sum(y)
FEATS = sorted(lab[0]["feat"].keys())


def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    rk = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rk[order[k]] = avg
        i = j + 1
    return rk


IDX = list(range(N))
ALL = list(combinations(IDX, nc))                      # 3060 assignments
TRUE = tuple(i for i in IDX if y[i] == 1)

res = {}
rank_cache = {}
for f in FEATS:
    v = [r["feat"][f] for r in lab]
    if any(not math.isfinite(x) for x in v):
        v = [x if math.isfinite(x) else 1e12 for x in v]
    rk = ranks(v)
    rank_cache[f] = rk
    # rank-sum of the CORRECT group, centred -> |U| statistic
    tot = sum(rk)

    def stat(sel):
        rs = sum(rk[i] for i in sel)
        return abs(rs - nc * (N + 1) / 2.0)

    obs = stat(TRUE)
    null = [stat(s) for s in ALL]
    p = sum(1 for x in null if x >= obs - 1e-12) / len(null)
    a = [v[i] for i in TRUE]
    b = [v[i] for i in IDX if y[i] == 0]
    # Cliff's delta (CORRECT vs WRONG)
    gt = sum(1 for x in a for z in b if x > z)
    lt = sum(1 for x in a for z in b if x < z)
    delta = (gt - lt) / (len(a) * len(b))
    a_s, b_s = sorted(a), sorted(b)

    def _median(v):
        # a TRUE median: the mean of the two middle order statistics for even
        # n. The first version took the upper-middle element, which reported
        # 0.986 where the median is 0.9805 — small, but a median is a named
        # estimator and must be the one it is named after.
        n = len(v)
        return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
    # containment / overlap of ranges
    contained = (min(a) <= min(b) and max(a) >= max(b)) or \
                (min(b) <= min(a) and max(b) >= max(a))
    sep = (min(a) > max(b)) or (max(a) < min(b))
    res[f] = dict(p_exact=p, cliffs_delta=delta,
                  med_correct=_median(a_s), med_wrong=_median(b_s),
                  min_correct=min(a), max_correct=max(a),
                  min_wrong=min(b), max_wrong=max(b),
                  perfect_separation=sep, range_contained=contained,
                  obs_stat=obs)

# ---- exact MAX-STATISTIC null over the battery (family-wise, not conservative)
zs = {}
for f in FEATS:
    rk = rank_cache[f]
    zs[f] = [abs(sum(rk[i] for i in s) - nc * (N + 1) / 2.0) for s in ALL]
maxnull = [max(zs[f][k] for f in FEATS) for k in range(len(ALL))]
for f in FEATS:
    obs = res[f]["obs_stat"]
    res[f]["p_maxstat_fwer"] = sum(
        1 for x in maxnull if x >= obs - 1e-12) / len(maxnull)
    res[f]["p_bonferroni"] = min(1.0, res[f]["p_exact"] * len(FEATS))

json.dump({"n": N, "n_correct": nc, "n_wrong": N - nc,
           "n_features": len(FEATS),
           "min_attainable_two_sided_p": 2.0 / len(ALL),
           "results": res},
          open(os.path.join(SP, "lc_stats.json"), "w", encoding="utf-8"),
          indent=1)

print(f"n={N} (correct={nc}, wrong={N-nc}); features={len(FEATS)}; "
      f"assignments={len(ALL)}; min attainable p={2/len(ALL):.2e}\n")
hdr = (f"{'feature':<22}{'p_exact':>9}{'p_FWER':>9}{'Cliff d':>9}"
       f"{'med_C':>11}{'med_W':>11}{'range_C':>22}{'range_W':>22}  sep")
print(hdr)
print("-" * len(hdr))
for f, d in sorted(res.items(), key=lambda kv: kv[1]["p_exact"]):
    print(f"{f:<22}{d['p_exact']:>9.4f}{d['p_maxstat_fwer']:>9.4f}"
          f"{d['cliffs_delta']:>9.2f}{d['med_correct']:>11.4g}"
          f"{d['med_wrong']:>11.4g}"
          f"{'[%.3g, %.3g]' % (d['min_correct'], d['max_correct']):>22}"
          f"{'[%.3g, %.3g]' % (d['min_wrong'], d['max_wrong']):>22}"
          f"  {'YES' if d['perfect_separation'] else ''}")
