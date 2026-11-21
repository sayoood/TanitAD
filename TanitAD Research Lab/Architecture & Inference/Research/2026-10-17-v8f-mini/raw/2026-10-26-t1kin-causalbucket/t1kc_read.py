"""The read of PREREG_V8F_T1KIN_CAUSALBUCKET (d2ed136d), applied verbatim. The row functions ``comps`` and ``paired`` are t1k_read.py's, taken
from its source by AST (not re-typed), so the rows are PREREG_V8F_T1KIN_HACC Amendment 1's by construction.
python t1kc_read.py <t1k export root> <T1KIN run dir> <causal run dir> <t1k_read.py> <pytest log> <out dir>"""
import ast
import json
import math
import os
import sys

import numpy as np

ROOT, RUN0, RUN, READER, PYLOG, OUT = sys.argv[1:7]
sys.path.insert(0, os.path.join(ROOT, "taniteval"))
import torch  # noqa: E402
from taniteval import four_families as FF  # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as PB  # noqa: E402

DT, F, TWO_F = 0.2, 0.1207, 0.2415
src = open(READER, encoding="utf-8").read()
ns = {"np": np, "torch": torch, "FF": FF, "PB": PB, "math": math, "DT": DT}
tree = ast.parse(src)
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("comps", "paired"):
        exec(compile(ast.Module([node], []), READER, "exec"), ns)
    if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "ROWS" for t in node.targets):
        exec(compile(ast.Module([node], []), READER, "exec"), ns)
comps, paired, ROWS = ns["comps"], ns["paired"], ns["ROWS"]
os.makedirs(OUT, exist_ok=True)


def load(run, tag):
    import glob
    fs = sorted(glob.glob(os.path.join(run, f"dump_{tag}", "ep*.npz")))
    assert fs, f"no dumps for {tag} in {run}"
    zs = [np.load(f) for f in fs]
    d = {k: np.concatenate([z[k] for z in zs]) for k in ("g", "cl", "ol", "fl", "eid", "clip_index", "ws")}
    d["json"] = json.load(open(os.path.join(run, f"t1v8f_{tag}.json"), encoding="utf-8"))
    return d


refuse = []
rep = {"prereg": "Project Steering/PREREG_V8F_T1KIN_CAUSALBUCKET.md (d2ed136d)",
       "tier": "T1 (T1-kin: self-action OPEN loop; ego state = own executed motion; cameras logged; never 'closed loop')",
       "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, on episodes; the two seeds are the training replicate",
       "settings_primary": {"m": 10, "nav": "deadreckon", "scene": "logged", "limit": "causal", "schedule": "periodic"},
       "F_m": F, "two_F_m": TWO_F, "controls": {}, "seeds": {}}
chain = open(os.path.join(RUN, "logs", "chain.out"), encoding="utf-8").read()
cl1 = {lim: f"C-L1 {lim}: json and" in chain and "byte-identical" in chain.split(f"C-L1 {lim}: json and")[-1].split("\n")[0] for lim in ("logged", "off")}
rep["controls"]["C-L1"] = {"logged": cl1["logged"], "off": cl1["off"], "PASS": all(cl1.values())}
py = open(PYLOG, encoding="utf-8").read()
c2 = " passed" in py.strip().splitlines()[-1] and "failed" not in py.strip().splitlines()[-1] and "RED 4/4" in py
rep["controls"]["C-L2"] = {"suite_last_line": py.strip().splitlines()[-1], "mutants_red": "RED 4/4" in py, "PASS": c2}
c3 = json.load(open(os.path.join(RUN, "c_l3.json"), encoding="utf-8"))
rep["controls"]["C-L3"] = {"n": c3["n"], "PASS": c3["PASS"]}
for k in ("C-L1", "C-L2", "C-L3"):
    if not rep["controls"][k]["PASS"]:
        refuse.append(k)
reach, floor = {}, {}
D = {}
for s in (0, 1):
    D[s] = {"c": load(RUN, f"HACC_s{s}_m10_causal"), "L": load(RUN0, f"HACC_s{s}_m10")}
    c, L = D[s]["c"], D[s]["L"]
    assert (c["clip_index"] == L["clip_index"]).all() and (c["ws"] == L["ws"]).all() and (c["eid"] == L["eid"]).all()
    st = c["json"]["t1v8f"]["causal_bucket"]["stats_model_ticks"]
    cm = np.asarray(st["confusion_logged_x_causal"])
    off = int(cm.sum() - np.trace(cm))
    dol = float(np.abs(c["ol"] - L["ol"]).max())
    reach[f"s{s}"] = {"ticks": st["ticks"], "no_finite_past": st.get("no_finite_past", 0), "ticks_causal_ne_logged": off,
                      "max_abs_tick0_plan_causal_minus_logged_m": dol, "PASS": off > 0 and dol > 1e-3}
    df = float(np.abs(c["fl"] - L["fl"]).max())
    floor[f"s{s}"] = {"max_abs_floor_causal_minus_t1kin_m": df, "PASS": df <= 1e-9}
rep["controls"]["reachability"] = dict(reach, PASS=any(v["PASS"] for v in reach.values()))
rep["controls"]["floor_untouched"] = dict(floor, PASS=all(v["PASS"] for v in floor.values()))
for k in ("reachability", "floor_untouched"):
    if not rep["controls"][k]["PASS"]:
        refuse.append(k)

prim = {}
for s in (0, 1):
    c, L = D[s]["c"], D[s]["L"]
    G, eid = c["g"], c["eid"]
    Cc, Cf, CL = comps(c["cl"], G), comps(c["fl"], G), comps(L["cl"], G)
    p = paired(Cc["ade6_m"], Cf["ade6_m"], eid)
    p["PASS"] = p["delta"] <= -TWO_F and p["hi"] < 0
    prim[s] = p
    bes = {"rows_causal_minus_p10": {r: paired(Cc[r], Cf[r], eid) for r in ROWS},
           "rows_causal_minus_hacc_logged": {r: paired(Cc[r], CL[r], eid) for r in ROWS},
           "dominance_rows_separated_worse": [r for r in ROWS if paired(Cc[r], Cf[r], eid)["separated"] and paired(Cc[r], Cf[r], eid)["delta"] > 0],
           "mean_ade6_m": {"causal": float(np.nanmean(Cc["ade6_m"])), "p10": float(np.nanmean(Cf["ade6_m"])), "hacc_logged": float(np.nanmean(CL["ade6_m"]))},
           "bucket_stats_model_ticks": c["json"]["t1v8f"]["causal_bucket"]["stats_model_ticks"]}
    tag_all = f"HACC_s{s}_m10_causalall"
    if os.path.exists(os.path.join(RUN, f"t1v8f_{tag_all}.json")):
        A = load(RUN, tag_all)
        assert (A["clip_index"] == c["clip_index"]).all() and (A["ws"] == c["ws"]).all()
        CA, CAf = comps(A["cl"], G), comps(A["fl"], G)
        bes["causal_all_minus_p10_ade6"] = paired(CA["ade6_m"], CAf["ade6_m"], eid)
        bes["causal_all_bucket_stats"] = A["json"]["t1v8f"]["causal_bucket"]["stats_model_ticks"]
    rep["seeds"][f"s{s}"] = {"PRIMARY_causal_minus_p10_ade6": p, "beside": bes}
better = [prim[s]["separated"] and prim[s]["delta"] < 0 for s in (0, 1)]
if refuse:
    out = "REFUSED"
elif all(prim[s]["PASS"] for s in (0, 1)):
    out = "CAUSAL KEEPS THE MARGIN"
elif all(better):
    out = "CAUSAL PARTIAL"
elif not any(better):
    out = "CAUSAL LOSES IT"
else:
    out = "MIXED"
rep["OUTCOME"] = {"outcome": out, "refused_on": refuse}
json.dump(rep, open(os.path.join(OUT, "T1KC_READ.json"), "w"), indent=1)
print("OUTCOME", out, "refused:", refuse)
for s in (0, 1):
    print(f"s{s} PRIMARY", {k: prim[s][k] for k in ("delta", "lo", "hi", "separated", "PASS")})
