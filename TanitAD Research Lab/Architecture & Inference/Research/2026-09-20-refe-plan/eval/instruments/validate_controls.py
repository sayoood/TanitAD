#!/usr/bin/env python3
"""Controls for W-TRANSP / W-FAM, on NAVTRAIN held-out only (the TF-1 population, 1,018 targets, 11 logs). Never touches navtest.

Reads the rows `transp_score.py` wrote with   --plans T=<T> Tc20s=<Tc20s> Tw=<Tw> --controls --human-arm   and checks:

  C-ID-TRACKED      arm T   .tracked      == TF-1 `T_tracked`            every sub-score, every token, max |diff| = 0.0
  C-ID-TRANSPARENT  arm T   .transparent  == TF-1 `T_transparent`        (E2's builder, bit for bit)
  C-ID-TC20S        arm Tc20s tracked/transparent == TF-1 `Tc20s_tracked` / `Tc20s_transparent`
  C-ID-HUMAN        arm human .tracked    == TF-1 `H_tracked`             (NAVSIM's standard human row)
  C-RED-SIM         ctl_sim_as_executed == tracked                        (LQR states fed to the executed slot reproduce the tracked score)
  C-RED-SHIFT       plans shifted 0.5 m left: DAC changes on >= 1 token, mean PDMS falls, in BOTH scorers
  C-RED-TW          the deliberate-regression label Tw reads WORSE than T on DAC and PDMS (both scorers) and on the LATERAL family
  C-FAM-REF0        arm human: every REFERENCE-side family error reads exactly 0 (the reference IS the logged human future)
  C-FAM-EXEC        arm human: the EXECUTED side reads the tracker's deviation (> 0) and equals P5's banked `exec_H` per token
  C-RERUN           (optional --rerun rows) a second run is bit-identical, rows compared without `t_s`

  python validate_controls.py --rows "full_s*.jsonl" --tf1 "tf1rows/tf1_s[0-3].jsonl" tf1rows/tf1_s4r.jsonl --p5s <p5s_cubic.npz> --out controls.json
Exit code 0 only if every control passes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
KEYS = ("nc", "dac", "ddc", "mult", "ep", "ttc", "c", "pdms")


def rd(patterns):
    out = {}
    for pat in patterns:
        for fp in sorted(glob.glob(pat)):
            for ln in open(fp, encoding="utf-8"):
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if "error" not in r:
                    out[r["token"]] = r
    return out


def maxdiff(rows, arm, scorer, ref, ref_key):
    mx, n, worst = 0.0, 0, None
    for t, r in rows.items():
        if t not in ref or arm not in r["arms"]:
            continue
        for k in KEYS:
            d = abs(r["arms"][arm][scorer][k] - ref[t][ref_key][k])
            n += 1
            if d > mx:
                mx, worst = d, (t, k)
    return {"max_abs_diff": mx, "n_values": n, "n_tokens": n // len(KEYS), "worst": worst, "pass": bool(n > 0 and mx == 0.0)}


def mean(rows, arm, scorer, k):
    return float(np.mean([r["arms"][arm][scorer][k] for r in rows.values() if arm in r["arms"]]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--tf1", nargs="+", required=True)
    ap.add_argument("--p5s", default=None)
    ap.add_argument("--rerun", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()
    rows, tf1 = rd(a.rows), rd(a.tf1)
    res = {"n_rows": len(rows), "n_tf1": len(tf1), "controls": {}}
    C = res["controls"]
    C["C-ID-TRACKED"] = maxdiff(rows, "T", "tracked", tf1, "T_tracked")
    C["C-ID-TRANSPARENT"] = maxdiff(rows, "T", "transparent", tf1, "T_transparent")
    t1 = maxdiff(rows, "Tc20s", "tracked", tf1, "Tc20s_tracked")
    t2 = maxdiff(rows, "Tc20s", "transparent", tf1, "Tc20s_transparent")
    C["C-ID-TC20S"] = {"tracked": t1, "transparent": t2, "pass": t1["pass"] and t2["pass"]}
    C["C-ID-HUMAN"] = maxdiff(rows, "human", "tracked", tf1, "H_tracked")
    # C-RED-SIM
    mx, n = 0.0, 0
    for r in rows.values():
        for arm in ("T", "Tc20s", "Tw"):
            if arm in r["arms"] and "ctl_sim_as_executed" in r["arms"][arm]:
                for k in KEYS:
                    mx = max(mx, abs(r["arms"][arm]["ctl_sim_as_executed"][k] - r["arms"][arm]["tracked"][k]))
                    n += 1
    C["C-RED-SIM"] = {"max_abs_diff": mx, "n_values": n, "pass": bool(n > 0 and mx == 0.0)}
    # C-RED-SHIFT
    sh = {}
    for sc, key in (("tracked", "ctl_shift05_tracked"), ("transparent", "ctl_shift05_transparent")):
        ch = sum(1 for r in rows.values() if "T" in r["arms"] and r["arms"]["T"][key]["dac"] != r["arms"]["T"][sc]["dac"])
        m0 = float(np.mean([r["arms"]["T"][sc]["pdms"] for r in rows.values() if "T" in r["arms"]]))
        m1 = float(np.mean([r["arms"]["T"][key]["pdms"] for r in rows.values() if "T" in r["arms"]]))
        sh[sc] = {"tokens_with_dac_change": ch, "pdms_mean_before": m0, "pdms_mean_after_shift": m1, "pass": bool(ch >= 1 and m1 < m0)}
    C["C-RED-SHIFT"] = {**sh, "pass": all(v["pass"] for v in sh.values())}
    # C-RED-TW (scores)
    tw = {}
    for sc in ("tracked", "transparent"):
        tw[sc] = {"dac_T": mean(rows, "T", sc, "dac"), "dac_Tw": mean(rows, "Tw", sc, "dac"),
                  "pdms_T": mean(rows, "T", sc, "pdms"), "pdms_Tw": mean(rows, "Tw", sc, "pdms")}
        tw[sc]["pass"] = bool(tw[sc]["dac_Tw"] < tw[sc]["dac_T"] and tw[sc]["pdms_Tw"] < tw[sc]["pdms_T"])
    # families (taniteval adapter, reused)
    import families_run as FR
    rl = sorted(rows.values(), key=lambda r: r["token"])
    fam = {}
    for arm in ("T", "Tw", "human"):
        fam[arm] = FR.run([r for r in rl if arm in r["arms"]], arm, f"controls:{arm}", FR.DEFAULT_REPO, a.n_boot)
    LAT = ("heading_mae_deg", "cross_mae_m", "curvature_mae_1pm", "yaw_rate_mae_degps")
    LON = ("speed_mae_mps", "along_mae_m", "accel_mae_mps2")

    def fm(arm, frame, fam_name, key):
        return fam[arm]["frames"][frame]["block"][fam_name][key]

    for sc, frame in (("tracked", "executed_tracked"), ("transparent", "executed_transparent")):
        tw[sc]["lateral_family"] = {k: {"T": fm("T", frame, "lateral", k), "Tw": fm("Tw", frame, "lateral", k)} for k in LAT}
        tw[sc]["lateral_family_worse_on"] = [k for k in LAT if tw[sc]["lateral_family"][k]["Tw"] > tw[sc]["lateral_family"][k]["T"]]
        tw[sc]["pass"] = bool(tw[sc]["pass"] and len(tw[sc]["lateral_family_worse_on"]) >= 1)
    C["C-RED-TW"] = {**tw, "pass": all(tw[s]["pass"] for s in ("tracked", "transparent"))}
    # C-FAM-REF0 / C-FAM-EXEC (human arm)
    ref0 = {k: fm("human", "reference", "lateral", k) for k in LAT}
    ref0.update({k: fm("human", "reference", "longitudinal", k) for k in LON})
    exe = {k: fm("human", "executed_tracked", "lateral", k) for k in LAT}
    exe.update({k: fm("human", "executed_tracked", "longitudinal", k) for k in LON})
    C["C-FAM-REF0"] = {"reference_side": ref0, "pass": all(v == 0.0 for v in ref0.values())}
    ex = {"executed_side": exe, "pass": all(v > 0.0 for v in exe.values())}
    if a.p5s:
        z = np.load(a.p5s)
        pos = {str(t): i for i, t in enumerate(z["tokens"])}
        d = max(float(np.abs(np.asarray(r["arms"]["human"]["fam"]["exec_tracked8"]) - z["exec_H"][pos[r["token"]]]).max())
                for r in rl if r["token"] in pos and "human" in r["arms"])
        dT = max(float(np.abs(np.asarray(r["arms"]["T"]["fam"]["exec_tracked8"]) - z["exec_T"][pos[r["token"]]]).max())
                 for r in rl if r["token"] in pos)
        ex["vs_P5_banked_exec_H_max_abs_m_or_rad"] = d
        ex["vs_P5_banked_exec_T_max_abs_m_or_rad"] = dT
        ex["pass"] = bool(ex["pass"] and d < 1e-4 and dT < 1e-4)   # P5's arrays are float32 (~6e-6 at 50 m)
    C["C-FAM-EXEC"] = ex
    C["lead_states_T"] = fam["T"]["lead_states"]
    if a.rerun:
        r2 = rd(a.rerun)
        bad = [t for t, r in r2.items() if t in rows and json.dumps({k: v for k, v in r.items() if k != "t_s"}, sort_keys=True)
               != json.dumps({k: v for k, v in rows[t].items() if k != "t_s"}, sort_keys=True)]
        C["C-RERUN"] = {"n_compared": len([t for t in r2 if t in rows]), "n_different": len(bad), "pass": bool(r2 and not bad)}
    res["all_pass"] = all(v.get("pass", True) for v in C.values() if isinstance(v, dict))
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    for k, v in C.items():
        if isinstance(v, dict) and "pass" in v:
            print(f"{k:18s} {'PASS' if v['pass'] else 'FAIL'}")
    print("ALL_PASS" if res["all_pass"] else "SOME_FAIL")
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
