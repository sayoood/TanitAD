"""MM 2026-11-26: split an arm's selection regret (oracle-of-64 minus pick, tracked frame) by the first sub-score in which the pick is
worse than the oracle proposal, and split the safety misses into TRACKER-INDUCED (the pick passes that term as planned, i.e. in the
transparent frame, and fails only after NAVSIM's LQR re-simulation) vs UNSAFE-AS-PLANNED. DESCRIPTIVE, MEASURED, T1 NAVSIM v1.1;
navtest is a SNOOPED population here: no rule may be fitted on it.
usage: safety_miss_split.py <table shard> [<shard> ...]"""
import json, sys
rows = [json.loads(l) for p in sys.argv[1:] for l in open(p, encoding="utf-8")]
n = len(rows)
TERMS = ("nc", "dac", "ddc", "ttc", "ep", "c")
reg_by, n_by, trk_n, trk_reg = {}, {}, {}, {}
for r in rows:
    tr, tp = r["tr"], r["tp"]; pd = tr["pdms"]; i = r["pick"]; m = max(pd); j = pd.index(m)
    reg = m - pd[i]
    if reg <= 1e-9:
        continue
    k = next((t for t in TERMS if tr[t][i] < tr[t][j] - 1e-9), "other")
    reg_by[k] = reg_by.get(k, 0) + reg; n_by[k] = n_by.get(k, 0) + 1
    if k in ("nc", "dac", "ddc", "ttc") and tp[k][i] >= tr[k][j] - 1e-9:
        trk_n[k] = trk_n.get(k, 0) + 1; trk_reg[k] = trk_reg.get(k, 0) + reg
tot = sum(reg_by.values())
print(f"n={n} tokens; total regret {100*tot/n:.3f} PDMS pts")
for k in sorted(reg_by, key=lambda x: -reg_by[x]):
    extra = f"; tracker-induced {trk_n.get(k,0)} tokens, {100*trk_reg.get(k,0)/n:.3f} pts" if k in ("nc", "dac", "ddc", "ttc") else ""
    print(f"  {k:5s} {n_by[k]:5d} tokens  {100*reg_by[k]/n:6.3f} pts ({100*reg_by[k]/tot:5.1f} %){extra}")
