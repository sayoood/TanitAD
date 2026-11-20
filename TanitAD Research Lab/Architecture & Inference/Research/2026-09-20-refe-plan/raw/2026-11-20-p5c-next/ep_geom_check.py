"""EP ranking check (descriptive; computed on ALL navtest tokens incl. EVAL, so any rule built on it needs a FRESH population):
within-token Spearman of the tracked EP label vs the EP head, path length, endpoint x, over proposals with NC = DAC = 1 (>= 3)."""
import json, numpy as np, s_rerank as A
from head_auc import spearman, bootci
S = "E:/Projects/TanitAD/data/refe_navtest/seams"
names = ["p_nc","p_dac","p_ep","p_ttc","p_c","p_6","agg","rank","gap","end_x","end_y","end_h","p1_x","p1_y","plen","dh","acc","jerk","dcons","dpick","dmed"]
out = {}
for spec in (f"C1=propscore/full_C1_s*of2.jsonl={S}/stage2_p5c_c1.proposals.npz", f"base=propscore/full_base_s*of2.jsonl={S}/stage2_p7_full.proposals.npz"):
    d = A.load(spec); X, tr, n = d["X"], d["tr"], len(d["X"]); ok = (tr["nc"] > .5) & (tr["dac"] > .5); out[d["name"]] = {}
    for f in ("p_ep", "plen", "end_x"):
        k = names.index(f)
        v = np.array([spearman(X[i, ok[i], k], tr["ep"][i, ok[i]]) if ok[i].sum() >= 3 else np.nan for i in range(n)])
        out[d["name"]][f] = bootci(v, d["logs"]); print("ZZEPGEOM", d["name"], f, round(out[d["name"]][f]["mean"], 3))
out["note"] = "T1 tracked per-proposal tables; log-cluster bootstrap 2,000 (another draw of LOGS); descriptive, no bar; snooped population"
json.dump(out, open("propscore/EP_GEOM_CHECK.json", "w"), indent=1)
