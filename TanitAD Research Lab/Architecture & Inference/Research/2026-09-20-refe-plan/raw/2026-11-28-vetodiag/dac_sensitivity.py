"""DAC-SENSITIVITY (MM 2026-11-28): DESCRIPTIVE. How accurate must a vision-predicted-map DAC checker be? On C1's navtest table
(T1 banked), simulate a checker = the true DAC with i.i.d. per-proposal errors: a failing proposal is passed with prob e_miss and a
passing proposal is rejected with prob e_fa. Rule as FILTER-ORACLE: k = argmax C1 agg32 over checker-passed proposals (none -> ship),
scored with tr. Targets: the TRACKED dac (tr; the form the harness executes) and the PLANNED dac (tp). Mean over 5 seeds.
Caveat: real map errors are spatially correlated within a token; i.i.d. errors are a first-order guide only, not a bar."""
import json, sys
import numpy as np

sys.argv = [sys.argv[0]]
src = open("E:/Projects/TanitAD-artifacts/vetodiag/vetodiag.py", encoding="utf-8").read()
exec(compile(src[:src.index("if ok1 != N")], "vetodiag_head", "exec"))
assert ok1 == N and abs(mean_ship - 92.5548) <= 5e-5
TPd = np.array([rows[t]["tp"]["dac"] for t in tok], np.float64)
out = {"tier": "T1 banked table, DESCRIPTIVE sensitivity (i.i.d. per-proposal checker errors)", "grid": []}
for tgt_name, truth in (("tracked", T["dac"] > 0), ("planned", TPd > 0)):
    for e_miss in (0.0, 0.1, 0.25, 0.5):
        for e_fa in (0.0, 0.01, 0.03, 0.1, 0.2):
            dp, de, nsw = [], [], []
            for sd in range(5):
                rng = np.random.default_rng(1000 + sd)
                u = rng.random(truth.shape)
                ok = np.where(truth, u >= e_fa, u < e_miss)
                k = np.where(ok, a, -np.inf).argmax(1)
                none = ~ok.any(1)
                k[none] = kship[none]
                dp.append(float((T["pdms"][ar, k] - pd_ship).mean() * 100))
                de.append(float((T["ep"][ar, k] - T["ep"][ar, kship]).mean() * 100))
                nsw.append(int((k != kship).sum()))
            r = {"target": tgt_name, "e_miss": e_miss, "e_fa": e_fa, "dPDMS": float(np.mean(dp)), "dPDMS_sd": float(np.std(dp)), "dEP": float(np.mean(de)), "n_switched": float(np.mean(nsw))}
            out["grid"].append(r)
            print(f"{tgt_name:8s} miss {e_miss:.2f} fa {e_fa:.2f}  dPDMS {r['dPDMS']:+.3f}  dEP {r['dEP']:+.3f}  switched {r['n_switched']:.0f}")
json.dump(out, open("E:/Projects/TanitAD-artifacts/vetodiag/DAC_SENSITIVITY.json", "w"), indent=1)
