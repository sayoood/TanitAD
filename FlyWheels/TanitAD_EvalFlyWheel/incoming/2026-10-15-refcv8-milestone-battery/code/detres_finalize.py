"""detres_finalize.py -- apply the PREREG decision rule MECHANICALLY to `detection_residual_probe.json` and write the `DECISION` block into it.

    python detres_finalize.py --probe <detection_residual_probe.json> --g0 E:/refcv8_eval_kit/battery/step5000/g0.json --pack <detres/pack_analysis.json>

Rule (PREREG_DETECTION_RESIDUAL.md, committed before any arm ran): D* = 0.02382, X = 0.33835, M = 0.31453, T = 0.020619.
  HARNESS_MISMATCH  A0 did not read M to 5 dp (nothing else is concluded)
  R1 NUMERICS_SUFFICIENT  some numerics-only arm (A1, A2:*) with judge.bar_clean has |v - v_A0| >= D*
  R2 NOT_NUMERICS         every numerics-only arm run has |v - v_A0| < D*/2
  R3 PARTIAL              otherwise
A4 (micro 8,8) is a SHAPE arm: reported, never part of the rule.  Needs no GPU.
"""
from __future__ import annotations

import argparse
import json
import statistics as st

KEY = "eval_box3d_det_ap2_automobile_0_20"
X, M, D_STAR, T = 0.33835, 0.31453, 0.02382, max(0.02, 2 / 97)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--g0", required=True)
    ap.add_argument("--pack", default=None)
    a = ap.parse_args()
    rec = json.load(open(a.probe, encoding="utf-8"))
    arms = rec["arms"]
    g = json.load(open(a.g0, encoding="utf-8"))
    T_ = g["verdict_A7"]["terms"]
    cls_of = {k: v.get("cls") for k, v in T_.items()}
    sup = {k: v.get("support_n") for k, v in T_.items()}
    a0 = arms.get("A0") or {}
    d: dict = {"constants": {"D_star": D_STAR, "half": D_STAR / 2, "X_inrun": X, "M_g0": M, "judge_tol": T}}
    if a0.get("status") != "DONE" or a0.get("value") is None or abs(a0["value"] - M) > 1e-5:
        d["verdict"] = "HARNESS_MISMATCH"
        d["why"] = f"A0 status {a0.get('status')} value {a0.get('value')} vs {M}"
        rec["DECISION"] = d
        json.dump(rec, open(a.probe, "w", encoding="utf-8"), indent=1, default=str)
        print("DETRES_DECISION", d["verdict"], d["why"])
        return
    a0j = a0.get("judge") or {}
    d["a0_judge_consistent_with_g0"] = (a0j.get("detection_out") == [KEY] and a0j.get("n_smooth_out") == 0)
    num = {k: v for k, v in arms.items() if v.get("status") == "DONE" and (k == "A1" or k.startswith("A2:"))}
    rows = []
    for k, v in num.items():
        dk = v["value"] - a0["value"]
        rows.append({"arm": k, "value": v["value"], "D_k": round(dk, 5), "abs_D_k": round(abs(dk), 5), "bar_clean": (v.get("judge") or {}).get("bar_clean"),
                     "n_detection_out": (v.get("judge") or {}).get("n_detection_out"), "n_smooth_out": (v.get("judge") or {}).get("n_smooth_out"),
                     "within_judge_tol_of_inrun": abs(v["value"] - X) <= T, "frac_of_D_star": round(abs(dk) / D_STAR, 3)})
    d["numerics_arms"] = rows
    d["not_run"] = {k: v.get("why") or v.get("error") for k, v in arms.items() if v.get("status") in ("NOT_RUN", "SKIPPED", "ERROR")}
    mx = max((r["abs_D_k"] for r in rows), default=0.0)
    d["max_abs_D_k"] = mx
    if any(r["abs_D_k"] >= D_STAR - 1e-9 and r["bar_clean"] for r in rows):
        d["verdict"] = "R1_NUMERICS_SUFFICIENT"
    elif rows and all(r["abs_D_k"] < D_STAR / 2 for r in rows):
        d["verdict"] = "R2_NOT_NUMERICS"
    else:
        d["verdict"] = "R3_PARTIAL"
    big_unclean = [r["arm"] for r in rows if r["abs_D_k"] >= D_STAR - 1e-9 and not r["bar_clean"]]
    if big_unclean:
        d["note_unclean_big_moves"] = f"{big_unclean}: |D_k| >= D* but the arm also pushed another DETECTION/SMOOTH term out of its bar -> not counted"
    a2 = [r["D_k"] for r in rows if r["arm"].startswith("A2:")]
    if len(a2) >= 2:
        d["A2_draws"] = {"n": len(a2), "mean_D": round(st.mean(a2), 5), "sd_D": round(st.stdev(a2), 5), "max_abs": round(max(abs(x) for x in a2), 5)}
    a4 = arms.get("A4") or {}
    d["A4_shape_arm"] = {"status": a4.get("status"), "value": a4.get("value"), "D_vs_A0": a4.get("delta_vs_A0"), "error": a4.get("error")}
    # population view: how large is the arm-vs-A0 spread over the DETECTION AP cells vs the in-run-vs-replay spread (sd 0.00386 on the banked G0)?
    ks = [k for k, c in cls_of.items() if c == "DETECTION" and "_det_ap" in k and (sup.get(k) or 0) >= 30 and k in (a0.get("ap_cells") or {})]
    pop = {"n_cells": len(ks)}
    dev_inrun = [float(T_[k]["inrun"]) - float(T_[k]["mean"]) for k in ks]
    if dev_inrun:
        pop["inrun_minus_replay_sd"] = round(st.pstdev(dev_inrun), 5)
        pop["inrun_minus_replay_sd_without_target"] = round(st.pstdev([x for k, x in zip(ks, dev_inrun) if k != KEY]), 5)
    for k_arm, v in num.items():
        ac = v.get("ap_cells") or {}
        dd = [ac[k] - a0["ap_cells"][k] for k in ks if ac.get(k) is not None and a0["ap_cells"].get(k) is not None]
        if len(dd) > 2:
            pop[k_arm] = {"sd_arm_minus_A0": round(st.pstdev(dd), 5), "max_abs": round(max(abs(x) for x in dd), 5)}
    d["population_detection_ap_cells"] = pop
    if a.pack:
        try:
            pk = json.load(open(a.pack, encoding="utf-8"))
            d["pack_level_dose_response_frac_abs_ge_Dstar"] = {f: v["frac_abs_delta_ge_0.02382"] for f, v in pk["P4"]["dose_response"].items()}
        except (OSError, ValueError, KeyError):
            pass
    rec["DECISION"] = d
    json.dump(rec, open(a.probe, "w", encoding="utf-8"), indent=1, default=str)
    print("DETRES_DECISION", d["verdict"], "max|D|", mx, "n_numerics_arms", len(rows), "A0 consistent", d["a0_judge_consistent_with_g0"])


if __name__ == "__main__":
    main()
