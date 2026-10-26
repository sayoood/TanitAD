"""The read of PREREG_V8F_T1KIN_STRICT (963906e8), applied verbatim. Row functions ``comps`` / ``paired`` / ``ROWS`` are t1k_read.py's, taken
from its source by AST.  python t1ks_read.py <t1k export root> <T1KIN run dir> <causal run dir> <strict run dir> <t1k_read.py> <out dir>"""
import ast
import glob
import json
import math
import os
import sys

import numpy as np

ROOT, RUN0, RUNC, RUN, READER, OUT = sys.argv[1:7]
sys.path.insert(0, os.path.join(ROOT, "taniteval"))
import torch  # noqa: E402
from taniteval import four_families as FF  # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as PB  # noqa: E402

DT, F, TWO_F = 0.2, 0.1207, 0.2415
ns = {"np": np, "torch": torch, "FF": FF, "PB": PB, "math": math, "DT": DT}
for node in ast.parse(open(READER, encoding="utf-8").read()).body:
    if (isinstance(node, ast.FunctionDef) and node.name in ("comps", "paired")) or \
       (isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "ROWS" for t in node.targets)):
        exec(compile(ast.Module([node], []), READER, "exec"), ns)
comps, paired, ROWS = ns["comps"], ns["paired"], ns["ROWS"]
os.makedirs(OUT, exist_ok=True)


def load(run, tag):
    fs = sorted(glob.glob(os.path.join(run, f"dump_{tag}", "ep*.npz")))
    assert fs, f"no dumps for {tag} in {run}"
    zs = [np.load(f) for f in fs]
    d = {k: np.concatenate([z[k] for z in zs]) for k in ("g", "cl", "ol", "fl", "eid", "clip_index", "ws")}
    d["json"] = json.load(open(os.path.join(run, f"t1v8f_{tag}.json"), encoding="utf-8"))
    return d


refuse, rep = [], {"prereg": "Project Steering/PREREG_V8F_T1KIN_STRICT.md (963906e8)",
                   "tier": "T1 (T1-kin: self-action OPEN loop; cameras FROZEN at t0; causal speed bucket; never 'closed loop')",
                   "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0; the two seeds are the training replicate",
                   "settings_primary": {"m": 10, "nav": "deadreckon", "scene": "frozen", "limit": "causal", "schedule": "periodic"},
                   "F_m": F, "two_F_m": TWO_F, "controls": {}, "seeds": {}}
chain = open(os.path.join(RUN, "logs", "chain.out"), encoding="utf-8").read()
for s in (0, 1):
    assert f"end HACC_s{s}_m10_strict rc=0" in chain, f"s{s} strict run did not end rc=0"
prim, D = {}, {}
for s in (0, 1):
    X, C, L = load(RUN, f"HACC_s{s}_m10_strict"), load(RUNC, f"HACC_s{s}_m10_causal"), load(RUN0, f"HACC_s{s}_m10")
    for o in (C, L):
        assert (X["clip_index"] == o["clip_index"]).all() and (X["ws"] == o["ws"]).all() and (X["eid"] == o["eid"]).all()
    h = X["json"]["t1v8f"]
    assert h["scene"] == "frozen" and h["limit"] == "causal" and h["replan_steps"] == 10 and h["nav"] == "deadreckon"
    D[s] = (X, C, L)
    rep["controls"][f"floor_untouched_s{s}"] = {"max_abs_m": float(np.abs(X["fl"] - L["fl"]).max())}
    rep["controls"][f"floor_untouched_s{s}"]["PASS"] = rep["controls"][f"floor_untouched_s{s}"]["max_abs_m"] <= 1e-9
    t0 = float(np.abs(X["ol"] - C["ol"]).max())
    ex = float(np.abs(X["cl"] - C["cl"]).max())
    rep["controls"][f"channels_closed_s{s}"] = {"max_abs_tick0_plan_strict_minus_causal_m": t0, "max_abs_executed_strict_minus_causal_m": ex,
                                                "PASS": t0 <= 1e-6 and ex > 1e-3}
for k, v in rep["controls"].items():
    if not v["PASS"]:
        refuse.append(k)
for s in (0, 1):
    X, C, L = D[s]
    G, eid = X["g"], X["eid"]
    Cx, Cf, Cc = comps(X["cl"], G), comps(X["fl"], G), comps(C["cl"], G)
    p = paired(Cx["ade6_m"], Cf["ade6_m"], eid)
    p["PASS"] = p["delta"] <= -TWO_F and p["hi"] < 0
    prim[s] = p
    vs = {r: paired(Cx[r], Cf[r], eid) for r in ROWS}
    rep["seeds"][f"s{s}"] = {"PRIMARY_strict_minus_p10_ade6": p, "beside": {
        "rows_strict_minus_p10": vs,
        "rows_strict_minus_causal": {r: paired(Cx[r], Cc[r], eid) for r in ROWS},
        "dominance_rows_separated_worse": [r for r in ROWS if vs[r]["separated"] and vs[r]["delta"] > 0],
        "mean_ade6_m": {"strict": float(np.nanmean(Cx["ade6_m"])), "p10": float(np.nanmean(Cf["ade6_m"])), "causal": float(np.nanmean(Cc["ade6_m"]))},
        "bucket_stats_model_ticks": X["json"]["t1v8f"]["causal_bucket"]["stats_model_ticks"],
        "envelope_clipped_windows": X["json"]["t1v8f"]["windows_with_envelope_clip"]}}
better = [prim[s]["separated"] and prim[s]["delta"] < 0 for s in (0, 1)]
out = ("REFUSED" if refuse else "STRICT MARGIN" if all(prim[s]["PASS"] for s in (0, 1)) else "STRICT PARTIAL" if all(better)
       else "STRICT LOST" if not any(better) else "MIXED")
rep["OUTCOME"] = {"outcome": out, "refused_on": refuse}
json.dump(rep, open(os.path.join(OUT, "T1KS_READ.json"), "w"), indent=1)
print("OUTCOME", out, "refused:", refuse)
for s in (0, 1):
    print(f"s{s} PRIMARY", {k: prim[s][k] for k in ("delta", "lo", "hi", "separated", "PASS")})
