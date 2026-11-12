#!/usr/bin/env python3
"""TF-N3 combo read (navtrain held-out ONLY). RJ5H5 (--pos-jerk-limit 5.0 --policy-interval-s 0.2) paired against the BANKED arms of the TF-N2 reroll,
on the same tokens. Estimator: tfn2_analyze.boot (paired log-cluster bootstrap, B 10,000, seed 20260927, 95 % percentile, ratio mean).
It answers "another draw of LOGS" only (rollout/labels/tracker/scorer deterministic on CPU, policy argmax).
  python tfn3_combo_analyze.py --run <run> --out <json> --fam-dir <run>/fam --banked <run>/banked
BAR (committed in HANDOFF.md BEFORE any RJ5H5 row existed; never changed after):
  carry_to_navtest = [CI lower of dPDMS(RJ5H5.R - RJ5.R) > 0]
                     AND [for each of DAC, NC, C: point delta (RJ5H5.R - RJ5.R) >= 0 AND worse-flips <= better-flips]
  else RJ5 alone stays the lever.
"""
import argparse
import json
import os
import sys

import numpy as np

N2 = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n2"
sys.path.insert(0, N2)
import tfn2_analyze as A  # noqa: E402

SUB = A.SUB
BANKED_ARMS = ("R0", "RJ5", "R5HZ")


def fam_summary(fp):
    d = json.load(open(fp, encoding="utf-8"))
    blk = d["frames"]["executed_tracked"]["block"]
    if "longitudinal" not in blk:
        return {"n": d["n"], "block_status": blk}
    out = {"n": d["n"], "n_logs": d["n_logs"], "strategic": "PAUSED (PI 2026-09-19)"}
    for fam in ("longitudinal", "lateral", "tactical"):
        out[fam] = {k: v for k, v in blk[fam].items() if isinstance(v, (int, float))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fam-dir", default="")
    ap.add_argument("--banked", required=True)
    a = ap.parse_args()
    rows = {}
    for arm in BANKED_ARMS:
        rows[arm] = {r["token"]: r for r in A.load([os.path.join(a.banked, "scores", f"{arm}_s*.jsonl")])}
    rows["RJ5H5"] = {r["token"]: r for r in A.load([os.path.join(a.run, "scores", "RJ5H5_s*.jsonl")])}
    common = sorted(set.intersection(*[set(v) for v in rows.values()]))
    R = {arm: [rows[arm][t] for t in common] for arm in rows}
    logs = [r["log"] for r in R["RJ5H5"]]
    out = {"n_tokens": len(common), "n_logs": len(set(logs)), "dropped_not_in_all_arms": {k: len(v) - len(common) for k, v in rows.items()},
           "estimator": "paired log-cluster bootstrap, B=10000, seed 20260927, 95% percentile (tfn2_analyze.boot); 'another draw of LOGS' only",
           "evidence_class": "MEASURED (ours), navtrain held-out, EXPLORATORY; tier T1 NAVSIM v1.1 PDMS non-reactive LQR-tracked; harness cache not used (nuPlan-DB rebuild), contrasts internally consistent",
           "strategic": "PAUSED (PI 2026-09-19)"}
    out["knobs_in_rows"] = {}
    for r in R["RJ5H5"]:
        k = json.dumps({x: v for x, v in r["knobs"].items() if x != "max_jerk_long"}, sort_keys=True)
        out["knobs_in_rows"][k] = out["knobs_in_rows"].get(k, 0) + 1
    out["levels"] = {}
    for arm in R:
        for f in ("C", "R"):
            lv = {k: 100 * float(np.mean([A.g(r, f, k) for r in R[arm]])) for k in SUB}
            lv["pdms_ci"] = [100 * x for x in A.boot([A.g(r, f, "pdms") for r in R[arm]], logs)]
            out["levels"][f"{arm}.{f}"] = lv

    def pair(arm, f, base, bf):
        return {k: [100 * x for x in A.boot([A.g(r, f, k) for r in R[arm]], logs, [A.g(r, bf, k) for r in R[base]])] for k in SUB}

    out["paired"] = {"RJ5H5.R_vs_R0.R": pair("RJ5H5", "R", "R0", "R"), "RJ5H5.R_vs_RJ5.R": pair("RJ5H5", "R", "RJ5", "R"),
                     "RJ5H5.R_vs_R5HZ.R": pair("RJ5H5", "R", "R5HZ", "R"), "RJ5H5.C_vs_R0.C": pair("RJ5H5", "C", "R0", "C"),
                     "RJ5H5.C_vs_RJ5.C": pair("RJ5H5", "C", "RJ5", "C")}
    flips = {}
    for k in ("nc", "dac", "c"):
        base = np.array([A.g(r, "R", k) for r in R["RJ5"]])
        cur = np.array([A.g(r, "R", k) for r in R["RJ5H5"]])
        wi, bi = np.where(cur < base - 1e-9)[0], np.where(cur > base + 1e-9)[0]
        flips[k] = {"worse": int(len(wi)), "better": int(len(bi)), "worse_tokens": [common[i] for i in wi][:40],
                    "worse_mean_drop": float((base[wi] - cur[wi]).mean()) if len(wi) else 0.0}
    out["token_flips_RJ5H5.R_vs_RJ5.R_tracked"] = flips
    pdm = np.array([r["mech"]["pdm_path4_m"] for r in R["R0"]])
    mech = {}
    for arm in R:
        tp = np.array([r["mech"]["t_path4_m"] for r in R[arm]])
        mech[arm] = {"teacher_path4_over_pdm_path4_median": float(np.median(tp[pdm > 1] / pdm[pdm > 1])),
                     "teacher_v4_minus_pdm_v4_median": float(np.median(np.array([r["mech"]["t_v4"] for r in R[arm]]) - np.array([r["mech"]["pdm_v4"] for r in R[arm]]))),
                     "teacher_amax_median": float(np.median([r["mech"]["t_amax"] for r in R[arm]])),
                     "teacher_path4_minus_RJ5_m": A.boot(tp, logs, np.array([r["mech"]["t_path4_m"] for r in R["RJ5"]]))}
    out["mechanism_rows"] = mech
    if a.fam_dir:
        out["families_executed_tracked"] = {}
        for arm, d in (("RJ5H5", a.fam_dir), ("R0", os.path.join(a.banked, "fam")), ("RJ5", os.path.join(a.banked, "fam")), ("R5HZ", os.path.join(a.banked, "fam"))):
            for f in ("C", "R"):
                fp = os.path.join(d, f"fam_{arm}_{f}.json")
                if os.path.exists(fp):
                    out["families_executed_tracked"][f"{arm}.{f}"] = fam_summary(fp)
        fm = out["families_executed_tracked"]
        if "RJ5H5.R" in fm and "RJ5.R" in fm and "longitudinal" in fm["RJ5H5.R"]:
            out["mechanism_longitudinal_RJ5H5.R_minus_RJ5.R"] = {
                k: fm["RJ5H5.R"]["longitudinal"][k] - fm["RJ5.R"]["longitudinal"][k]
                for k in ("speed_bias_mps", "speed_mae_mps", "accel_mae_mps2")
                if k in fm["RJ5H5.R"]["longitudinal"] and k in fm["RJ5.R"]["longitudinal"]}
    p = out["paired"]["RJ5H5.R_vs_RJ5.R"]
    cond_a = p["pdms"][1] > 0
    cond_b = {k: bool(p[k][0] >= 0 and flips[k]["worse"] <= flips[k]["better"]) for k in ("dac", "nc", "c")}
    out["BAR"] = {"A_dPDMS_ci_lower_gt_0": bool(cond_a), "dPDMS_RJ5H5R_minus_RJ5R": p["pdms"],
                  "B_no_drop_DAC_NC_C": cond_b, "B_detail_point_dsub": {k: p[k] for k in ("dac", "nc", "c")},
                  "WORTH_CARRYING_TO_NAVTEST": bool(cond_a and all(cond_b.values()))}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"n": out["n_tokens"], "logs": out["n_logs"], "BAR": out["BAR"],
                      "levels_pdms": {k: round(v["pdms"], 2) for k, v in out["levels"].items()}}, indent=0))


if __name__ == "__main__":
    main()
