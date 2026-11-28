"""MM 2026-11-26: decompose the navtest gap from a 64-proposal score table (tracked frame): what the scorer picks vs what the generator offers.
usage: oracle_decomp.py <arm label> <table shard> [<shard> ...]   (DESCRIPTIVE, MEASURED, T1 NAVSIM v1.1 tracked; no CI here)"""
import json, sys, random, statistics as st
arm, paths = sys.argv[1], sys.argv[2:]
rows = [json.loads(l) for p in paths for l in open(p, encoding="utf-8")]
n = len(rows)
pick, orac, regret, opt, rank = [], [], [], 0, []
sub = ("nc", "dac", "ddc", "ttc", "ep", "c")
pick_sub = {k: 0.0 for k in sub}; orac_sub = {k: 0.0 for k in sub}
fail_cause = {k: 0 for k in sub}
cov = {1.0: 0, 0.95: 0, 0.9: 0, 0.8: 0}
n_safe_ok = 0
rand = []
rng = random.Random(0)
for r in rows:
    t = r["tr"]; pd = t["pdms"]; i = r["pick"]
    m = max(pd); j = pd.index(m)
    pick.append(pd[i]); orac.append(m); regret.append(m - pd[i]); rand.append(pd[rng.randrange(64)])
    if pd[i] >= m - 1e-9: opt += 1
    rank.append(sorted(pd, reverse=True).index(pd[i]))
    for k in sub:
        pick_sub[k] += t[k][i]; orac_sub[k] += t[k][j]
    if m - pd[i] > 1e-9:
        # which sub-score is worse in the pick than in the oracle proposal (multiplicative terms first)
        for k in ("nc", "dac", "ddc", "ttc", "ep", "c"):
            if t[k][i] < t[k][j] - 1e-9:
                fail_cause[k] += 1; break
    for thr in cov:
        if m >= thr - 1e-9: cov[thr] += 1
    if any(t["nc"][q] == 1 and t["dac"][q] == 1 and t["ttc"][q] == 1 and t["ddc"][q] == 1 for q in range(64)): n_safe_ok += 1
mean = lambda x: 100 * sum(x) / len(x)
print(f"== {arm}: n={n} tokens (T1 tracked, MEASURED, descriptive)")
print(f"  scorer pick PDMS   {mean(pick):7.3f}")
print(f"  random proposal    {mean(rand):7.3f}")
print(f"  ORACLE best-of-64  {mean(orac):7.3f}   (the selection ceiling of this generator)")
print(f"  mean regret        {mean(regret):7.3f}   pick optimal on {100*opt/n:.1f} % of tokens; median pick rank {st.median(rank)}, mean {st.mean(rank):.1f} (0 = best)")
print("  sub-scores x100     pick    oracle-proposal")
for k in sub:
    print(f"    {k:4s}            {100*pick_sub[k]/n:7.3f}  {100*orac_sub[k]/n:7.3f}")
print("  first worse term in the pick vs the oracle proposal (tokens with regret): " + ", ".join(f"{k} {v}" for k, v in fail_cause.items()))
print("  generator coverage: tokens whose best proposal reaches PDMS >= " + ", ".join(f"{t}: {100*c/n:.1f} %" for t, c in cov.items()))
print(f"  tokens with at least one proposal passing NC, DAC, TTC and DDC: {100*n_safe_ok/n:.1f} %")
# regret concentration
rs = sorted(regret, reverse=True)
for q in (0.01, 0.05, 0.10, 0.20):
    k = int(q * n)
    print(f"  top {int(q*100):2d} % of tokens by regret carry {100*sum(rs[:k])/max(sum(rs),1e-12):.1f} % of the total regret")
