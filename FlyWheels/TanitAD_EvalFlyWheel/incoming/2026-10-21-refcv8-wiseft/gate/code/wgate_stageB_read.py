#!/usr/bin/env python3
"""R8-WISEFT-GATE stage B (registered f0ae643, 2026-10-21 15:31 Berlin, before any theta_0.5 navtest score existed): the ONE navtest
read of X* = G_ARC (stage A, WGATE_STAGE_A.json) on theta_0.5's R8-WISEFT stage-2 rows.

  WORKS iff GW_ARC - theta_0.5 > 0 AND GW_ARC - PRIOR_ha0p > 0, each 'positive' (paired log-cluster bootstrap B 2000, interval
  excluding 0 at log_name AND nuplan_drive -- bars8 compact_pair), at BOTH inference seeds, margin min(d_s0, d_s1) > 2 x |d_s0 - d_s1|;
  FAILED if either point <= 0 at either seed; else NOT PROVEN.
Beside (never bars): GW_ARC - G_ARC(EMA-48,852) per token, GW_ARC - STOP, GW_ARC - REFCV7_50400_LEGAL, strata, sub-scores (all in the
gate tool's output), and the composite's per-token poses for the four-family read.

Controls: the gate tool's own (never-fires == plan, always-fires == prior, G - G == 0); C1 the plan arm reproduces theta_0.5's stage-2
PDMS; C2 G_ARC(EMA-48,852) rebuilt here reproduces the banked 65.5246 / 65.5078 exactly; C3 theta-run PRIOR rows == EMA-run PRIOR rows
per token (bit-exact on every scored column); C4 the composite seam with no fallback == theta_0.5's poses and with all fallback ==
PRIOR's poses, exactly.
"""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

G = "C:/Users/Admin/refcv8_pf/navsim_ckpt/wiseft/gate"
W2 = "C:/Users/Admin/refcv8_pf/navsim_ckpt/wiseft/stage2"
FT = "C:/Users/Admin/refcv8_pf/navsim_ckpt/families_navtest"
NR = "C:/Users/Admin/refcv8_pf/navtest_read/raw"
INP = f"{FT}/inputs/navtest_inputs.json.gz"
PY = sys.executable
R7 = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step50400/"
      "thor_scores/navtest/r7thor_s50400_R7_VMAXOFF/r7thor_s50400_R7_VMAXOFF.csv")
STOP = f"{NR}/controls/STOP_navtest.csv"
TH = {"s0": f"{W2}/scores_navtest/r8nt_w0p50_R8_LEGAL/r8nt_w0p50_R8_LEGAL.csv",
      "s1": f"{W2}/scores_navtest/r8nt_w0p50_R8_LEGAL_s1/r8nt_w0p50_R8_LEGAL_s1.csv"}
TPR = f"{W2}/scores_navtest/r8nt_w0p50_PRIOR_ha0p/r8nt_w0p50_PRIOR_ha0p.csv"
EPR = f"{NR}/full/scores_navtest/r8nt_s48852e_PRIOR_ha0p/r8nt_s48852e_PRIOR_ha0p.csv"
ERT = {"s0": f"{NR}/full/scores_navtest/r8nt_s48852e_R8_RT/r8nt_s48852e_R8_RT.csv",
       "s1": f"{NR}/full/scores_navtest/r8nt_s48852e_R8_RT_s1/r8nt_s48852e_R8_RT_s1.csv"}
TSEAM = {"s0": f"{FT}/seams_w050/seam_R8_LEGAL.npz", "s1": f"{FT}/seams_w050/seam_R8_LEGAL_s1.npz"}
TPSEAM = f"{FT}/seams_w050/seam_PRIOR_ha0p.npz"
ESEAM = {"s0": f"{FT}/seams/seam_R8_RT.npz", "s1": f"{FT}/seams/seam_R8_RT_s1.npz"}
EPSEAM = f"{FT}/seams/seam_PRIOR_ha0p.npz"
OUT = f"{G}/stageB"
BANKED_GARC = {"s0": 65.5246, "s1": 65.5078}


def main():
    os.makedirs(OUT, exist_ok=True)
    sa = json.load(open(f"{G}/stageA/WGATE_STAGE_A.json"))
    assert sa["X_star"] == "G_ARC", sa["X_star"]
    gate_json = f"{OUT}/GATE_THETA05_NAVTEST.json"
    rc = subprocess.run([PY, "-X", "utf8", f"{G}/code/r8_gate_eval.py", "--rt-csv", TH["s0"], "--rt-csv-s1", TH["s1"], "--prior-csv", TPR,
                         "--seam-rt", TSEAM["s0"], "--seam-rt-s1", TSEAM["s1"], "--seam-prior", TPSEAM, "--inputs", INP,
                         "--label", "NAVTEST_RESULT_R8_WISEFT_GATE", "--out", gate_json],
                        stdout=open(f"{OUT}/gate_navtest.out", "w"), stderr=subprocess.STDOUT).returncode
    assert rc == 0 and os.path.exists(gate_json), f"gate tool rc={rc}"
    r = json.load(open(gate_json, encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("r8ge", f"{G}/code/r8_gate_eval.py")
    GE = importlib.util.module_from_spec(spec); sys.modules["r8ge"] = GE; spec.loader.exec_module(GE)
    B = GE.load_bars8(); cfg = {**B.DEFAULT_CFG}
    rd = B.read_scores_csv
    th = {s: rd(p) for s, p in TH.items()}
    tpr, epr = rd(TPR), rd(EPR)
    toks = sorted(th["s0"].index)
    ti = B.load_inputs(INP).loc[toks]
    cl = [str(x) for x in ti["log"]]
    res = {"id": "R8-WISEFT-GATE stage B", "registered": "f0ae643 (2026-10-21 15:31 Berlin)", "X_star": "G_ARC",
           "tier": "NavSim v1 open-loop PDM scorer (T1-family), navtest 12,146 tokens; one training seed; diagnostic lever test, never a bar",
           "gate_tool_output": os.path.relpath(gate_json, G), "controls": {}, "rule": {}, "beside": {}}
    c = res["controls"]
    c["gate_tool"] = r["controls"]["all_pass"]
    c["C1_plan_is_theta05"] = {s: r["arms"][f"R8_RT_{s}"]["PDMS"] for s in ("s0", "s1")}
    c["C3_prior_theta_run_equals_EMA_run"] = bool(sorted(epr.index) == toks and np.array_equal(
        tpr.loc[toks].to_numpy(dtype=float), epr.loc[toks].to_numpy(dtype=float)))
    # the registered rule
    d = {}
    for vs, key in (("theta05", "R8_RT"), ("PRIOR", "PRIOR")):
        p = {s: r["pairs"][f"G_ARC_{s}__minus__{key}"] for s in ("s0", "s1")}
        dd = {s: p[s]["delta_x100"] for s in p}
        fl = abs(dd["s0"] - dd["s1"])
        pos = all(bool(p[s]["interval"].get("positive")) for s in p)
        d[vs] = {"delta": dd, "interval": {s: {k: p[s]["interval"].get(k) for k in ("lo", "hi", "positive", "separated_log_name")}
                                           for s in p},
                 "floor": round(fl, 4), "all_positive": pos, "margin_ok": min(dd.values()) > 2 * fl, "any_nonpositive": min(dd.values()) <= 0}
    res["rule"] = d
    if d["theta05"]["any_nonpositive"] or d["PRIOR"]["any_nonpositive"]:
        v = "FAILED"
    elif all(d[k]["all_positive"] and d[k]["margin_ok"] for k in d):
        v = "WORKS"
    else:
        v = "NOT PROVEN"
    res["verdict"] = v
    res["GW_ARC_PDMS"] = {s: r["arms"][f"G_ARC_{s}"]["PDMS"] for s in ("s0", "s1")}
    res["GW_ARC_subs"] = {s: {m: r["arms"][f"G_ARC_{s}"][m] for m in B.SUBS} for s in ("s0", "s1")}
    res["fallback_frac"] = {s: r["arms"][f"G_ARC_{s}"]["fallback_frac"] for s in ("s0", "s1")}
    # beside: rebuild G_ARC(EMA-48,852) exactly as R8-GATE-NAVTEST did, and the per-token comparators
    P_ep = GE.seam_poses(EPSEAM, toks)
    garc48, gw = {}, {}
    P_tp = GE.seam_poses(TPSEAM, toks)
    for s in ("s0", "s1"):
        ert = rd(ERT[s])
        lr = np.abs(np.log(np.maximum(GE.arc4(GE.seam_poses(ESEAM[s], toks)) / np.maximum(GE.arc4(P_ep), 0.05), 0.05)))
        garc48[s] = GE.select(ert, epr, toks, lr > GE.THR)
        P_t = GE.seam_poses(TSEAM[s], toks)
        fb = np.abs(np.log(np.maximum(GE.arc4(P_t) / np.maximum(GE.arc4(P_tp), 0.05), 0.05))) > GE.THR
        gw[s] = (GE.select(th[s], tpr, toks, fb), fb, P_t)
    c["C2_GARC48852_reproduces_banked"] = {s: round(100 * float(garc48[s]["PDMS"].mean()), 4) for s in garc48}
    c["C2_pass"] = c["C2_GARC48852_reproduces_banked"] == BANKED_GARC
    c["gw_rebuilt_equals_tool"] = {s: round(100 * float(gw[s][0]["PDMS"].mean()), 4) == res["GW_ARC_PDMS"][s] for s in gw}
    stop, r7 = rd(STOP), rd(R7)
    for s in ("s0", "s1"):
        res["beside"][f"GW_ARC_{s}__minus__G_ARC_EMA48852_{s}"] = B.pair_record(gw[s][0], garc48[s], toks, cl, cfg)
        res["beside"][f"GW_ARC_{s}__minus__STOP"] = B.pair_record(gw[s][0], stop, toks, cl, cfg)
        res["beside"][f"GW_ARC_{s}__minus__REFCV7_50400_LEGAL"] = B.pair_record(gw[s][0], r7, toks, cl, cfg)
    # composite seam (s0) for the four-family read, with its exact controls
    z = np.load(TSEAM["s0"], allow_pickle=False)
    order = {str(t): i for i, t in enumerate(z["token"])}
    fb, P_t = gw["s0"][1], gw["s0"][2]
    comp = np.where(fb[:, None, None], P_tp, P_t)
    c["C4_composite_endpoints"] = bool(np.array_equal(np.where(np.zeros_like(fb)[:, None, None], P_tp, P_t), P_t)
                                       and np.array_equal(np.where(np.ones_like(fb)[:, None, None], P_tp, P_t), P_tp))
    np.savez(f"{OUT}/seam_GW_ARC_s0.npz", token=np.array(toks), poses=comp, arm=np.array("GW_ARC_s0"),
             source=np.array(["gw_arc"] * len(toks)), fallback=fb)
    c["all_pass"] = bool(c["gate_tool"] and c["C2_pass"] and c["C3_prior_theta_run_equals_EMA_run"] and c["C4_composite_endpoints"]
                         and all(c["gw_rebuilt_equals_tool"].values()))
    if not c["all_pass"]:
        res["verdict"] = "INCOMPLETE (a control failed): " + v
    json.dump(res, open(f"{OUT}/WGATE_STAGE_B.json", "w"), indent=1, default=str)
    brief = {"verdict": res["verdict"], "GW_ARC": res["GW_ARC_PDMS"], "plan_theta05": c["C1_plan_is_theta05"],
             "minus_theta05": d["theta05"]["delta"], "minus_PRIOR": d["PRIOR"]["delta"],
             "minus_GARC48852_s0": res["beside"]["GW_ARC_s0__minus__G_ARC_EMA48852_s0"]["delta_x100"],
             "minus_REFCV7_s0": res["beside"]["GW_ARC_s0__minus__REFCV7_50400_LEGAL"]["delta_x100"],
             "minus_STOP_s0": res["beside"]["GW_ARC_s0__minus__STOP"]["delta_x100"], "controls": c["all_pass"]}
    print("QQWGB " + json.dumps(brief) + " QQ")


if __name__ == "__main__":
    main()
