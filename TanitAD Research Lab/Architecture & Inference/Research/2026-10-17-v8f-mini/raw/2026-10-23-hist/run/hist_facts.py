"""Zero-GPU facts after HIST: H's four families (from the committed read) and whether GRID's and HIST's ADE6 gains are on the same windows."""
import json, sys
import numpy as np
L = "C:/Users/Admin/box2_ship/logs/"
d = json.load(open(L + "v8f_hist/hist_read_s0.json", encoding="utf-8"))
r = d["rows"]["s0"]; b = r["beside"]
print("PRIMARY", {k: r["primary"][k] for k in ("delta", "lo", "hi", "separated")})
print("MLAT", {k: r["mlat"][k] for k in ("rmse_slot", "fall")}, "mse", {k: r["mlat"]["mse_delta"][k] for k in ("delta", "lo", "hi")})
for k in ("mleak", "mdk"):
    if k in r: x = r[k].get("delta", r[k]); print(k, {q: x.get(q) for q in ("delta", "lo", "hi", "separated")})
f = b["families"]
for fam in ("longitudinal", "lateral"):
    for k, x in f[fam].items():
        if isinstance(x, dict) and "delta" in x:
            print(f"  {fam[:4]} {k:20s} {x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}] {'SEP' if x['separated'] else 'ns'}")
dk = b["distance_keeping_k10"]; dk = dk.get("pair_vs_m2b", next(iter(dk.values())))["dk"]
for k, x in dk.items(): print(f"  dk   {k:20s} {x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}] {'SEP' if x['separated'] else 'ns'}")
print("oracle", b.get("oracle_ade6"), "present_slot", json.dumps(b.get("present_slot", {}).get("rmse_group"))[:300])
for k in b:
    if "slot" in k or "subset" in k or "t11" in k: print("beside key", k, json.dumps(b[k])[:400])
# ---- are the G and H gains on the same windows?
B = np.load(L + "v8f_mini/eval/eval_M2b_s0.npz"); G = np.load(L + "v8f_grid/eval/eval_M2b-8x20_s0.npz"); H = np.load(L + "v8f_hist/eval/eval_M2b-h10_s0.npz")
assert (B["eid"] == G["eid"]).all() and (B["eid"] == H["eid"]).all()
dg, dh = G["ade6"] - B["ade6"], H["ade6"] - B["ade6"]
print("mean dG %.4f dH %.4f | corr(dG,dH) %.3f | mean(dG+dH) %.4f" % (dg.mean(), dh.mean(), np.corrcoef(dg, dh)[0, 1], (dg + dh).mean()))
eid = B["eid"]; ue = np.unique(eid)
pe = np.array([[dg[eid == e].mean(), dh[eid == e].mean()] for e in ue])
print("per-episode corr %.3f; episodes where both improve %d / %d" % (np.corrcoef(pe.T)[0, 1], int(((pe[:, 0] < 0) & (pe[:, 1] < 0)).sum()), len(ue)))
