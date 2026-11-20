"""Regret attribution (descriptive, T1 tracked tables, EVAL+TUNE all tokens): oracle-of-64 minus pick PDMS x100, split into
zeroed-pick tokens (gate) and non-zeroed tokens; on non-zeroed tokens where the oracle also has NC = DAC = 1, the exact split
(5 dEP + 5 dTTC + 2 dC)/12 (both gates 1, so PDMS is linear there); the remainder is reported as 'other'."""
import json, sys
import numpy as np
import s_rerank as A
out = {}
for spec in sys.argv[1:-1]:
    d = A.load(spec); n = len(d["tok"]); ar = np.arange(n); pk = d["pick"]; tr = d["tr"]
    o = tr["pdms"].argmax(1); pdm = tr["pdms"]
    reg = (pdm.max(1) - pdm[ar, pk]) * 100
    z = (tr["nc"][ar, pk] < .5) | (tr["dac"][ar, pk] < .5)
    lin = ~z & (tr["nc"][ar, o] > .5) & (tr["dac"][ar, o] > .5) & (tr["nc"][ar, pk] > .5) & (tr["dac"][ar, pk] > .5)
    comp = {k: ((w * (tr[k][ar, o] - tr[k][ar, pk])) / 12 * 100) for k, w in (("ep", 5), ("ttc", 5), ("c", 2))}
    r = {"n": n, "regret_total_x100_mean": float(reg.mean()),
         "from_zeroed_picks": float(reg[z].sum() / n), "n_zeroed": int(z.sum()),
         "from_nonzeroed_linear": {k: float(v[lin].sum() / n) for k, v in comp.items()}, "n_linear": int(lin.sum()),
         "from_nonzeroed_other": float(reg[~z & ~lin].sum() / n), "n_other": int((~z & ~lin).sum()),
         "nonzeroed_ttc0_pick_with_ttc1_alt": int((~z & (tr["ttc"][ar, pk] < .5) & (tr["ttc"].max(1) > .5)).sum())}
    out[d["name"]] = r
    print("ZZREGRET", d["name"], json.dumps(r))
json.dump(out, open(sys.argv[-1], "w"), indent=1)
