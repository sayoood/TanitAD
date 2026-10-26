"""The read of PREREG_V8F_HACC_LAT (96604184), applied verbatim: LAT-DOM and KEEP per arm and seed, then the outcome table.
Row functions ``comps`` / ``paired`` / ``ROWS`` are t1k_read.py's, taken by AST. The pipeline's own controls (C-T, C-T2, C-A1, C-A2, C-B1, C-B2,
reachability) are read from the chain's RUN_SUMMARY.json and log; the reader's controls (R-1, R-2 on every arm run; C-1, C-2 on the m = 30 runs;
the floor against T1KIN's) are recomputed here. Any failure => REFUSED, no outcome row.
python t1kl_read.py <t1k export root> <T1KIN run dir> <hlat dir> <t1k_read.py> <out dir>"""
import ast
import glob
import json
import math
import os
import sys

import numpy as np

ROOT, RUN0, HL, READER, OUT = sys.argv[1:6]
RUN = os.path.join(HL, "run")
sys.path.insert(0, os.path.join(ROOT, "taniteval"))
import torch  # noqa: E402
from taniteval import four_families as FF  # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as PB  # noqa: E402

DT, F, HALF_F, TWO_F = 0.2, 0.1207, 0.0604, 0.2415
FLOOR_OL_ADE6 = 2.48699212
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


def r1_r2(run):
    A, B = comps(run["cl"], run["g"]), comps(run["fl"], run["g"])
    ff = run["json"]["arms"]
    lat_cross = [k for k in ff["cl"]["four_families"]["lateral"] if "cross" in k and "mae" in k]
    r1 = {}
    for arm, C in (("cl", A), ("fl", B)):
        lo, la = ff[arm]["four_families"]["longitudinal"], ff[arm]["four_families"]["lateral"]
        r1[arm] = {k: [float(C[k].mean()), lo[k]] for k in ("speed_mae_mps", "along_mae_m", "accel_mae_mps2")}
        if lat_cross:
            r1[arm]["cross_mae_m"] = [float(C["cross_mae_m"].mean()), la[lat_cross[0]]]
        for k in ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm"):
            if isinstance(la.get(k), (int, float)):
                r1[arm][k] = [C["_pooled"][k], la[k]]
    ok1 = all(abs(a - b) <= 6e-5 for arm in r1.values() for a, b in arm.values() if isinstance(b, (int, float)))
    p6 = paired(A["ade6_m"], B["ade6_m"], run["eid"])
    js6 = run["json"]["paired_decision_grade"]["paired_model_minus_floor_t1"]["ade_m"]
    return {"R-1": {"PASS": ok1, "rows": r1}, "R-2": {"reader": p6["delta"], "analyze": js6["delta"], "PASS": abs(p6["delta"] - float(js6["delta"])) <= 1e-6}}


refuse = []
rep = {"prereg": "Project Steering/PREREG_V8F_HACC_LAT.md (96604184)",
       "tier": "T1 (T1-kin: self-action OPEN loop; cameras logged; oracle speed bucket as in PREREG_V8F_T1KIN_HACC; never 'closed loop')",
       "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0; the two seeds are the training replicate",
       "F_m": F, "two_F_m": TWO_F, "half_F_m": HALF_F, "controls": {}, "arms": {}}
summ = json.load(open(os.path.join(HL, "RUN_SUMMARY.json"), encoding="utf-8"))
rep["controls"]["pipeline"] = summ.get("controls", summ)
pipe_fail = [k for k, v in (summ.get("controls") or {}).items() if not (isinstance(v, dict) and (v.get("pass", v.get("PASS")) is True))]
if pipe_fail or summ.get("all_controls_pass") is not True:
    refuse += [f"pipeline:{k}" for k in pipe_fail] or ["pipeline:all_controls_pass"]
for f in sorted(glob.glob(os.path.join(HL, "fit", "hlat_fit_M2b_s*.json"))):
    fc = json.load(open(f, encoding="utf-8"))["controls"]
    rep["controls"][f"fit_{os.path.basename(f)}"] = fc
    for k in ("C-A2_pass", "regression_pass", "kkt_pass"):
        if fc.get(k) is not True:
            refuse.append(f"fit:{os.path.basename(f)}:{k}")
suite = [ln for ln in open(os.path.join(HL, "logs", "chain.out"), encoding="utf-8") if " suite: " in ln]
rep["controls"]["C-4"] = {"line": suite[-1].strip() if suite else None,
                          "PASS": bool(suite) and " passed" in suite[-1] and "failed" not in suite[-1] and "error" not in suite[-1]}
if not rep["controls"]["C-4"]["PASS"]:
    refuse.append("C-4")
chain = open(os.path.join(HL, "logs", "chain.out"), encoding="utf-8").read()
if "ZZHLAT-DONE" not in chain:
    refuse.append("chain not DONE")
REF = {s: load(RUN0, f"HACC_s{s}_m10") for s in (0, 1)}
R = {}
for X in ("A", "B"):
    for s in (0, 1):
        r10, r30 = load(RUN, f"{X}_s{s}_m10"), load(RUN, f"{X}_s{s}_m30")
        H = REF[s]
        for o in (r30, H):
            assert (o["clip_index"] == r10["clip_index"]).all() and (o["ws"] == r10["ws"]).all() and (o["eid"] == r10["eid"]).all()
        h = r10["json"]["t1v8f"]
        assert h["replan_steps"] == 10 and h["scene"] == "logged" and h["limit"] == "logged" and h["nav"] == "deadreckon" and h["schedule"] == "periodic"
        c = r1_r2(r10)
        c["C-1"] = {"max_abs_cl_minus_ol_m30_m": float(np.abs(r30["cl"] - r30["ol"]).max())}
        c["C-1"]["PASS"] = c["C-1"]["max_abs_cl_minus_ol_m30_m"] <= 1e-9
        fl6 = float(np.linalg.norm(r30["fl"] - r30["g"], axis=-1).mean())
        c["C-2"] = {"floor_m30_ade6_m": fl6, "PASS": abs(fl6 - FLOOR_OL_ADE6) <= 1e-5}
        c["floor_vs_T1KIN"] = {"max_abs_m": float(np.abs(r10["fl"] - H["fl"]).max())}
        c["floor_vs_T1KIN"]["PASS"] = c["floor_vs_T1KIN"]["max_abs_m"] <= 1e-9
        c["arm_differs_from_HACC"] = {"max_abs_executed_m": float(np.abs(r10["cl"] - H["cl"]).max())}
        c["arm_differs_from_HACC"]["PASS"] = c["arm_differs_from_HACC"]["max_abs_executed_m"] > 1e-3
        for k, v in c.items():
            if not v["PASS"]:
                refuse.append(f"{X}_s{s}:{k}")
        R[(X, s)] = (r10, r30, H)
        rep["controls"][f"{X}_s{s}"] = c

res = {}
for (X, s), (r10, r30, H) in R.items():
    G, eid = r10["g"], r10["eid"]
    Cx, Cf, Ch = comps(r10["cl"], G), comps(r10["fl"], G), comps(H["cl"], G)
    vs = {r: paired(Cx[r], Cf[r], eid) for r in ROWS}
    vh = {r: paired(Cx[r], Ch[r], eid) for r in ROWS}
    worse = [r for r in ROWS if vs[r]["separated"] and vs[r]["delta"] > 0]
    ade6 = vs["ade6_m"]
    latdom = ade6["delta"] <= -TWO_F and ade6["hi"] < 0 and not worse
    keep = vh["ade6_m"]["hi"] <= HALF_F
    C30x, C30f = comps(r30["cl"], r30["g"]), comps(r30["fl"], r30["g"])
    res[(X, s)] = {"LAT_DOM": latdom, "KEEP": keep, "no_row_worse": not worse, "rows_separated_worse": worse,
                   "rows_arm_minus_p10": vs, "beside_rows_arm_minus_hacc": vh, "mean_ade6_m": float(np.nanmean(Cx["ade6_m"])),
                   "beside_m30_arm_minus_p10_ade6": paired(C30x["ade6_m"], C30f["ade6_m"], r30["eid"]),
                   "beside_envelope_clipped": r10["json"]["t1v8f"]["windows_with_envelope_clip"]}
    rep["arms"][f"{X}_s{s}"] = res[(X, s)]

LAT = ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm", "cross_mae_m")
full = [X for X in ("A", "B") if all(res[(X, s)]["LAT_DOM"] and res[(X, s)]["KEEP"] for s in (0, 1))]
if refuse:
    out, dep = "REFUSED", None
elif full:
    mean = {X: (res[(X, 0)]["mean_ade6_m"] + res[(X, 1)]["mean_ade6_m"]) / 2 for X in full}
    dep = min(full, key=lambda X: (round(mean[X], 12), 0 if X == "A" else 1))
    out = "T1 DOMINANCE REACHED"
elif any(all(res[(X, s)]["no_row_worse"] for s in (0, 1)) for X in ("A", "B")):
    out, dep = "LATERAL FIXED AT A COST", None
elif not any(all(not any(r in LAT for r in res[(X, s)]["rows_separated_worse"]) for s in (0, 1)) for X in ("A", "B")):
    out, dep = "NOT FIXED", None
else:
    out, dep = "MIXED", None
rep["OUTCOME"] = {"outcome": out, "deployed_arm": dep, "refused_on": refuse}
json.dump(rep, open(os.path.join(OUT, "T1KL_READ.json"), "w"), indent=1)
print("OUTCOME", out, "deployed", dep, "refused:", refuse)
for (X, s), v in sorted(res.items()):
    a = v["rows_arm_minus_p10"]["ade6_m"]
    print(f"{X}_s{s} LAT_DOM {v['LAT_DOM']} KEEP {v['KEEP']} ADE6 {a['delta']:+.4f} [{a['lo']:+.4f}, {a['hi']:+.4f}] worse {v['rows_separated_worse']}"
          f" keep_hi {v['beside_rows_arm_minus_hacc']['ade6_m']['hi']:+.4f}")
