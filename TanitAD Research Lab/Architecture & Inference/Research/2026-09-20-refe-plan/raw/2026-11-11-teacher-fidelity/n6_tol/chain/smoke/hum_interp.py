import json, os, sys
os.environ.setdefault("N4_RUN", "C:/Users/Admin/n6nt/run")
sys.path.insert(0, "C:/Users/Admin/n6nt/run/n5"); sys.path.insert(0, "C:/Users/Admin/n6nt/run/n6")
import n5_env  # noqa
import numpy as np
import n4_geom as G, transp_score as TS
I = "C:/Users/Admin/n6nt/smoke/in/"
toks = json.load(open(I + "resid.json"))["tokens"]
logs = {t: l for t, l in zip(np.load(I + "n2r_plans.npz")["tokens"], np.load(I + "n2r_plans.npz")["logs"])}
by = {}
for t in toks: by.setdefault(str(logs[t]), []).append(t)
out = {}
for lg, tk, cx in G.contexts(by):
    hum = TS.human_rel(cx.sc, cx.ego)
    p20 = hum[1::2][:20]; h8 = hum[4::5][:8]
    n8 = G.ND.to_navsim(p20)
    d = np.abs(n8 - h8)
    out[tk] = [float(d[:, :2].max()), float(d[:, 2].max())]
    print(tk[:8], "max |to_navsim(plan20) - logged 0.5 s poses|: xy m %.5f  heading rad %.6f" % tuple(out[tk]), flush=True)
json.dump(out, open("C:/Users/Admin/n6nt/smoke/hum_interp.json", "w"))
