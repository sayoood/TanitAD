import json, os, gzip, numpy as np
c3 = np.load("C:/fp/PK/raw/2026-10-11-c3-selection-headroom/table.npz")
exp = json.load(gzip.open("E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz", "rt", encoding="utf-8"))["tokens"]
cap = "C:/Users/Admin/freept/work/cap"
done = {f[len("inputs_"):-len(".json")] for f in os.listdir(cap) if f.startswith("inputs_") and f.endswith(".json")}
by = {}
for t in c3["token"]:
    t = str(t)
    by.setdefault(exp[t]["log_name"], []).append(t)
cands = sorted(((len(v), l) for l, v in by.items() if l in done), reverse=True)
print(cands[:6])
