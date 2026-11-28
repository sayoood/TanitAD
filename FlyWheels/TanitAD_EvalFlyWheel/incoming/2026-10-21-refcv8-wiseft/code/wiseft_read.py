#!/usr/bin/env python3
"""Apply the REGISTERED R8-WISEFT / R8-CKPT-TRAJ reading rule (PREREG_R8_WISEFT.json, landed e8911b1 before any score) to the
pulled Thor scores.  Writes WISEFT_READ.json + .md.  Controls must read their known values or the verdict is INCOMPLETE:
  K1  EMA-48,852 - EMA-48,852 == 0 exactly (pairing machinery)      K2  banked LEGAL s0 reproduces 63.3907 and PRIOR 73.8378
  K3  every scored arm is on the identical 2,089 tokens              K4  wins + ties + losses == n for every pair
"""
import importlib.util
import json
import sys

import numpy as np

W = "C:/Users/Admin/refcv8_pf/navsim_ckpt/wiseft"
R = "C:/Users/Admin/refcv8_pf/navsim_ckpt/run"
BARS8 = ("C:/Users/Admin/refcv8_pf/navsim_ckpt/pkg/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/code/"
         "navsim_bars8.py")
spec = importlib.util.spec_from_file_location("b8w", BARS8)
B = importlib.util.module_from_spec(spec)
sys.modules["b8w"] = B
spec.loader.exec_module(B)
rd = B.read_scores_csv


def sc(lbl):
    return rd(f"{W}/scores/{lbl}/{lbl}.csv")


def try_sc(lbl):
    try:
        return sc(lbl)
    except Exception:
        return None


DIAG = f"{R}/navtrain_sel/diag/scores"
REF = {"s0": rd(f"{DIAG}/r8sel_diag_ema48852_LEGAL_s0/r8sel_diag_ema48852_LEGAL_s0.csv"),
       "s1": rd(f"{DIAG}/r8sel_diag_ema48852_LEGAL_s1/r8sel_diag_ema48852_LEGAL_s1.csv")}
E40 = rd(f"{DIAG}/r8sel_diag_ema40000_LEGAL_s0/r8sel_diag_ema40000_LEGAL_s0.csv")
PR = rd(f"{R}/navtrain_sel/scores/r8sel_ref_PRIOR_ha0p/r8sel_ref_PRIOR_ha0p.csv")
toks = sorted(REF["s0"].index)
ti = B.load_inputs(R + "/inputs/navtrain_sample_inputs.json.gz").loc[toks]
cl = [str(x) for x in ti["log"]]
st = np.array([B.command_key(x) == "forward" for x in ti["command"]])
cfg = {**B.DEFAULT_CFG}


def P(df):
    return round(100 * float(df.loc[toks]["PDMS"].mean()), 4)


LAMS = {"0p00": 0.0, "0p25": 0.25, "0p50": 0.5, "0p75": 0.75}
arms = {(t, s): try_sc(f"r8sel_wise_l{t}_{a}") for t in LAMS for s, a in (("s0", "R8_LEGAL"), ("s1", "R8_LEGAL_s1"))}
traj = {k: try_sc(f"r8sel_traj_ema{k}_R8_LEGAL") for k in (5000, 15000, 30000)}
res = {"id": "R8-WISEFT + R8-CKPT-TRAJ", "registered": "e8911b1 (2026-10-21 13:39 Berlin, before any score)",
       "tier": "NavSim v1 open-loop PDMS, registered navtrain sample (2,089 tokens / 30 logs), Thor CPU scorer",
       "controls": {}, "wise": {}, "traj": {}}
c = res["controls"]
k4 = []


def pr(a, b, tt=None, cc=None):
    r = B.pair_record(a, b, tt or toks, cc or cl, cfg)
    k4.append(r["wins"] + r["ties"] + r["losses"] == r["n"])
    return r


def cells(a, b):
    out = {}
    for nm, m in (("straight", st), ("turn", ~st)):
        tt = [t for t, k in zip(toks, m) if k]
        cc = [x for x, k in zip(cl, m) if k]
        out[nm] = pr(a, b, tt, cc)["delta_x100"]
    return out


c["K1_self_pair_zero"] = B.pair_record(REF["s0"], REF["s0"], toks, cl, cfg)["delta_x100"] == 0.0
c["K2_banked_values"] = {"LEGAL_s0": P(REF["s0"]), "PRIOR": P(PR), "pass": P(REF["s0"]) == 63.3907 and P(PR) == 73.8378}
c["K3_identical_tokens"] = all(sorted(d.index) == toks for d in [*arms.values(), *traj.values()] if d is not None)

adm = {}
for t, lam in LAMS.items():
    rec = {"lambda": lam}
    if arms[(t, "s0")] is None or arms[(t, "s1")] is None:
        rec["status"] = "MISSING"
        res["wise"][t] = rec
        adm[t] = False
        continue
    d = {s: pr(arms[(t, s)], REF[s]) for s in ("s0", "s1")}
    rec["PDMS"] = {s: P(arms[(t, s)]) for s in ("s0", "s1")}
    rec["PDMS_mean"] = round((rec["PDMS"]["s0"] + rec["PDMS"]["s1"]) / 2, 4)
    rec["minus_EMA48852"] = d
    rec["minus_PRIOR_s0"] = pr(arms[(t, "s0")], PR)
    rec["cells_minus_EMA48852_s0"] = cells(arms[(t, "s0")], REF["s0"])
    rec["subs_s0"] = {m: round(100 * float(arms[(t, "s0")].loc[toks][m].mean()), 4) for m in B.SUBS}
    d0, d1 = d["s0"]["delta_x100"], d["s1"]["delta_x100"]
    floor = abs(d0 - d1)
    ok = (d0 > 0 and d1 > 0 and d["s0"]["interval"].get("positive") and d["s1"]["interval"].get("positive")
          and min(d0, d1) > 2 * floor)
    rec["floor"] = round(floor, 4)
    rec["admissible"] = bool(ok)
    adm[t] = bool(ok)
    res["wise"][t] = rec
res["wise_ref_EMA48852"] = {"PDMS": {s: P(REF[s]) for s in ("s0", "s1")}}
cand = [t for t in LAMS if adm[t]]
if cand:
    best = max(res["wise"][t]["PDMS_mean"] for t in cand)
    tied = [t for t in cand if best - res["wise"][t]["PDMS_mean"] <= 0.01]
    lam_star = max(tied, key=lambda t: LAMS[t])
    res["stage1"] = {"verdict": "ADMISSIBLE", "lambda_star": LAMS[lam_star], "tag": lam_star,
                     "next": "stage 2: ONE navtest read of theta_lambda* (registered)"}
else:
    res["stage1"] = {"verdict": "FAILED (no lambda admissible)", "lambda_star": None,
                     "next": "no navtest read; next lever training-side (fine-tune on NavSim; PI decision + compute)"}

for k, df in [(5000, traj[5000]), (15000, traj[15000]), (30000, traj[30000]), (40000, E40), (48852, REF["s0"])]:
    if df is None:
        res["traj"][k] = {"status": "MISSING"}
        continue
    r = {"PDMS_s0": P(df), "minus_PRIOR": pr(df, PR)}
    if k != 48852:
        r["minus_EMA48852"] = pr(df, REF["s0"])
    r["cells_minus_PRIOR"] = cells(df, PR)
    res["traj"][k] = r
below = [k for k in sorted(res["traj"]) if res["traj"][k].get("minus_PRIOR", {}).get("interval", {}).get("negative")]
beat = [k for k in res["traj"] if res["traj"][k].get("minus_EMA48852", {}).get("interval", {}).get("positive")]
res["traj_reading"] = {"earliest_snapshot_below_PRIOR_separated": below[0] if below else None,
                       "snapshots_beating_EMA48852_separated": beat}
c["K4_counts"] = all(k4)
c["all_pass"] = bool(c["K1_self_pair_zero"] and c["K2_banked_values"]["pass"] and c["K3_identical_tokens"] and c["K4_counts"])
if not c["all_pass"]:
    res["stage1"]["verdict"] = "INCOMPLETE (a control failed): " + res["stage1"]["verdict"]
json.dump(res, open(f"{W}/WISEFT_READ.json", "w"), indent=1, default=str)


def iv(r):
    i = r["interval"]
    return f"{r['delta_x100']:+.2f} [{i.get('lo', float('nan')):+.2f}, {i.get('hi', float('nan')):+.2f}]"


L = ["# R8-WISEFT + R8-CKPT-TRAJ read (registered e8911b1)", "",
     f"Tier: {res['tier']}. Controls all pass: **{c['all_pass']}**.", "",
     "| arm | PDMS s0 | PDMS s1 | minus EMA-48,852 s0 [interval] | s1 | floor | admissible |",
     "|---|---|---|---|---|---|---|"]
for t, r in res["wise"].items():
    if r.get("status") == "MISSING":
        L.append(f"| lam {LAMS[t]} | MISSING | | | | | |")
        continue
    L.append(f"| lam {LAMS[t]} | {r['PDMS']['s0']:.2f} | {r['PDMS']['s1']:.2f} | {iv(r['minus_EMA48852']['s0'])} | "
             f"{r['minus_EMA48852']['s1']['delta_x100']:+.2f} | {r['floor']:.2f} | {r['admissible']} |")
ref = res["wise_ref_EMA48852"]["PDMS"]
L.append(f"| lam 1 (EMA-48,852) | {ref['s0']:.2f} | {ref['s1']:.2f} | -- | -- | -- | -- |")
L += ["", f"**Stage 1:** {res['stage1']['verdict']} -- lambda* = {res['stage1']['lambda_star']}; next: {res['stage1']['next']}", "",
      "| EMA snapshot | PDMS s0 | minus PRIOR (73.84) [interval] | minus EMA-48,852 | straight / turn vs PRIOR |",
      "|---|---|---|---|---|"]
for k, r in res["traj"].items():
    if r.get("status") == "MISSING":
        L.append(f"| {k} | MISSING | | | |")
        continue
    e = r.get("minus_EMA48852")
    L.append(f"| {k:,} | {r['PDMS_s0']:.2f} | {iv(r['minus_PRIOR'])} | {'' if e is None else iv(e)} | "
             f"{r['cells_minus_PRIOR']['straight']:+.2f} / {r['cells_minus_PRIOR']['turn']:+.2f} |")
L += ["", "Trajectory reading (diagnostic): earliest snapshot below PRIOR (separated) = "
      f"{res['traj_reading']['earliest_snapshot_below_PRIOR_separated']}; snapshots beating EMA-48,852 (separated) = {beat}."]
open(f"{W}/WISEFT_READ.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("QQWISE " + json.dumps({"stage1": res["stage1"]["verdict"], "lam_star": res["stage1"]["lambda_star"],
                              "wise": {t: r.get("PDMS") for t, r in res["wise"].items()},
                              "traj": {k: r.get("PDMS_s0") for k, r in res["traj"].items()}, "controls": c["all_pass"]}) + " QQ")
