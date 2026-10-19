import sys, json, runpy, numpy as np
# re-use nav_natexp's loader by exec with a sentinel: import its functions
src = open(sys.argv[5], encoding="utf-8").read().split("if True:")[0]
g = {"__name__": "x"}; sys.argv = sys.argv[:5]; exec(compile(src, "nav_natexp", "exec"), g)
ROOT = g["ROOT"]; tci = g["tci"]
R, M, _ = g["load"](ROOT / "dump_s0", (("os", "plan_full_nav_true"), ("shuf", "plan_full_nav_shuffled")))
out = {}
for fed in (0, 1, 2):
    for v9 in (0, 1, 2):
        if fed == v9: continue
        m = (R["fed"] == fed) & (R["v9"] == v9) & (R["shuf"] == v9)
        if m.sum() < 10:
            out[f"fed{fed}_v9{v9}"] = {"n": int(m.sum())}; continue
        a, b = M[("shuf", "ade_6s")][m], M[("os", "ade_6s")][m]
        bt = tci.paired_episode_cluster_bootstrap(a, b, R["eid"][m], n_boot=2000, seed=0)
        out[f"fed{fed}_v9{v9}"] = {"n": int(m.sum()), "n_ep": int(len(set(R["eid"][m].tolist()))),
                                   "os_ade6": round(float(b.mean()), 4), "perframe_ade6": round(float(a.mean()), 4),
                                   "delta": round(float(a.mean() - b.mean()), 4),
                                   "ci95": [round(float(bt["lo"]), 4), round(float(bt["hi"]), 4)], "separated": bool(bt["separated"])}
json.dump(out, open(sys.argv[3].replace(".json", "_split.json"), "w"), indent=1); print(json.dumps(out, indent=1))
