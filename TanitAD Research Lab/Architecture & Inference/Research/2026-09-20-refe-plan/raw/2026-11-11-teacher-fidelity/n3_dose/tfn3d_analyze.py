#!/usr/bin/env python3
"""TF-N3 jerk-limit DOSE read (navtrain held-out ONLY). RJ3 (--pos-jerk-limit 3.0) and RJ8 (8.0) paired against the BANKED RJ5 (5.0), same tokens.
Estimator: tfn2_analyze.boot (paired log-cluster bootstrap, B 10,000, seed 20260927, 95 % percentile, ratio mean). It answers "another draw of LOGS" only
(CPU rollout, labels, tracker, scorer deterministic; argmax policy). Family paired CIs: fam_ci.compare (same estimator) on `families_run.py --per-window` JSONs.
  python tfn3d_analyze.py --run <run_dose> --out <json> --fam-dir <run_dose>/fam --banked <combo run>/banked
BAR (committed in HANDOFF.md and HERE before any RJ3/RJ8 row existed; mechanical; never changed after data). For x in {3, 8}, RJx "beats RJ5" iff ALL:
  A: CI lower bound of dPDMS (RJx.R - RJ5.R) > 0;
  B: for DAC, NC, C (form R, tracked): point delta vs RJ5.R >= 0 AND worse-flips <= better-flips;
  C (longitudinal guard, executed_tracked, form R, point values from the family JSONs): speed-bias diff (RJx - RJ5) <= +0.25 m/s AND speed-MAE diff <= +0.15 m/s.
Otherwise 5.0 stays. Monotone picture (EP, DAC, TTC, speed bias at jerk 1 / 3 / 5 / 8; R0 = 1.0) is DESCRIPTIVE only.
"""
import argparse
import json
import os
import sys

import numpy as np

N2 = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, N2)
sys.path.insert(0, os.path.join(HERE, "raw", "instr"))
import tfn2_analyze as A  # noqa: E402

SUB = A.SUB
JERK = {"R0": 1.0, "RJ3": 3.0, "RJ5": 5.0, "RJ8": 8.0}
NEW = ("RJ3", "RJ8")
BAR_SPEED_BIAS_MAX = 0.25      # m/s, RJx - RJ5, point values
BAR_SPEED_MAE_MAX = 0.15       # m/s, RJx - RJ5, point values
JERK_KEY = "positive_jerk_limit_when_nonnegative_acc"


def fam_summary(fp):
    d = json.load(open(fp, encoding="utf-8"))
    blk = d["frames"]["executed_tracked"]["block"]
    if "longitudinal" not in blk:
        return {"n": d["n"], "block_status": blk}
    out = {"n": d["n"], "n_logs": d["n_logs"], "strategic": "PAUSED (PI 2026-09-19)"}
    for fam in ("longitudinal", "lateral", "tactical"):
        out[fam] = {k: v for k, v in blk[fam].items() if isinstance(v, (int, float))}
    return out


def find_paired(fc, name):
    """paired fam_ci entry for a longitudinal metric (executed_tracked) whose key contains `name`"""
    lon = fc["frames"]["executed_tracked"]["families"]["longitudinal"]
    for k, v in lon.items():
        if name == k and v.get("status") == "OK":
            return {"key": k, **{x: v[x] for x in ("A", "B", "diff_B_minus_A", "n_tokens", "n_logs", "separated_from_0") if x in v}}
    return {"status": "NOT-FOUND", "name": name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fam-dir", required=True)
    ap.add_argument("--banked", required=True)
    a = ap.parse_args()
    import fam_ci as FC
    rows = {}
    for arm in ("R0", "RJ5"):
        rows[arm] = {r["token"]: r for r in A.load([os.path.join(a.banked, "scores", f"{arm}_s*.jsonl")])}
    for arm in NEW:
        rows[arm] = {r["token"]: r for r in A.load([os.path.join(a.run, "scores", f"{arm}_s*.jsonl")])}
    common = sorted(set.intersection(*[set(v) for v in rows.values()]))
    R = {arm: [rows[arm][t] for t in common] for arm in rows}
    logs = [r["log"] for r in R["RJ5"]]
    out = {"n_tokens": len(common), "n_logs": len(set(logs)), "dropped_not_in_all_arms": {k: len(v) - len(common) for k, v in rows.items()},
           "estimator": "paired log-cluster bootstrap, B=10000, seed 20260927, 95% percentile (tfn2_analyze.boot); 'another draw of LOGS' only",
           "evidence_class": "MEASURED (ours), navtrain held-out, EXPLORATORY; tier T1 NAVSIM v1.1 PDMS non-reactive LQR-tracked; harness cache not used (nuPlan-DB rebuild), contrasts internally consistent",
           "strategic": "PAUSED (PI 2026-09-19)",
           "bar": {"A": "CI lower of dPDMS(RJx.R-RJ5.R) > 0", "B": "DAC,NC,C: point delta >= 0 and worse<=better flips",
                   "C": f"speed-bias diff <= +{BAR_SPEED_BIAS_MAX} and speed-MAE diff <= +{BAR_SPEED_MAE_MAX} m/s vs RJ5.R (point values)"}}
    out["knob_reach"] = {}
    for arm in NEW:
        vals = sorted({round(float(r["knobs"][JERK_KEY]), 6) for r in R[arm]})
        out["knob_reach"][arm] = {"jerk_values_in_rows": vals, "requested": JERK[arm], "OK": vals == [JERK[arm]]}
    out["levels"] = {}
    for arm in R:
        for f in ("C", "R"):
            lv = {k: 100 * float(np.mean([A.g(r, f, k) for r in R[arm]])) for k in SUB}
            lv["pdms_ci"] = [100 * x for x in A.boot([A.g(r, f, "pdms") for r in R[arm]], logs)]
            out["levels"][f"{arm}.{f}"] = lv

    def pair(arm, f, base, bf):
        return {k: [100 * x for x in A.boot([A.g(r, f, k) for r in R[arm]], logs, [A.g(r, bf, k) for r in R[base]])] for k in SUB}

    out["paired"], out["token_flips_vs_RJ5.R_tracked"], out["fam_paired_vs_RJ5.R"], out["mechanism_longitudinal_minus_RJ5.R"] = {}, {}, {}, {}
    fam = {}
    for arm in ("R0", "RJ3", "RJ5", "RJ8"):
        d = os.path.join(a.banked, "fam") if arm == "R0" else a.fam_dir
        fp = os.path.join(d, f"fam_{arm}_R.json")
        if os.path.exists(fp):
            fam[arm] = fam_summary(fp)
    out["families_executed_tracked_R"] = fam
    verdict = {}
    for arm in NEW:
        out["paired"][f"{arm}.R_vs_RJ5.R"] = p = pair(arm, "R", "RJ5", "R")
        out["paired"][f"{arm}.R_vs_R0.R"] = pair(arm, "R", "R0", "R")
        out["paired"][f"{arm}.C_vs_RJ5.C"] = pair(arm, "C", "RJ5", "C")
        flips = {}
        for k in ("nc", "dac", "c"):
            base = np.array([A.g(r, "R", k) for r in R["RJ5"]])
            cur = np.array([A.g(r, "R", k) for r in R[arm]])
            wi, bi = np.where(cur < base - 1e-9)[0], np.where(cur > base + 1e-9)[0]
            flips[k] = {"worse": int(len(wi)), "better": int(len(bi)), "worse_tokens": [common[i] for i in wi][:40],
                        "worse_mean_drop": float((base[wi] - cur[wi]).mean()) if len(wi) else 0.0}
        out["token_flips_vs_RJ5.R_tracked"][arm] = flips
        cond_a = bool(p["pdms"][1] > 0)
        cond_b = {k: bool(p[k][0] >= 0 and flips[k]["worse"] <= flips[k]["better"]) for k in ("dac", "nc", "c")}
        cond_c, mech = None, {}
        if arm in fam and "RJ5" in fam and "longitudinal" in fam[arm] and "longitudinal" in fam["RJ5"]:
            for k in ("speed_bias_mps", "speed_mae_mps", "accel_mae_mps2"):
                mech[k] = fam[arm]["longitudinal"][k] - fam["RJ5"]["longitudinal"][k]
            cond_c = bool(mech["speed_bias_mps"] <= BAR_SPEED_BIAS_MAX and mech["speed_mae_mps"] <= BAR_SPEED_MAE_MAX)
        out["mechanism_longitudinal_minus_RJ5.R"][arm] = mech
        fa, fb = os.path.join(a.fam_dir, "fam_RJ5_R.json"), os.path.join(a.fam_dir, f"fam_{arm}_R.json")
        try:
            fc = FC.compare(json.load(open(fa, encoding="utf-8")), json.load(open(fb, encoding="utf-8")))
            out["fam_paired_vs_RJ5.R"][arm] = {"speed_bias": find_paired(fc, "speed_bias_mps"), "speed_mae": find_paired(fc, "speed_mae_mps"),
                                               "accel_mae": find_paired(fc, "accel_mae_mps2"), "progress": find_paired(fc, "ego_progress.progress_ratio_mean")}
            json.dump(fc, open(os.path.join(a.run, "result", f"fam_ci_{arm}R_vs_RJ5R.json"), "w", encoding="utf-8"), indent=1, default=str)
        except Exception as e:                                                  # noqa: BLE001
            out["fam_paired_vs_RJ5.R"][arm] = {"status": f"RAISED {type(e).__name__}: {str(e)[:300]}"}
        reach = out["knob_reach"][arm]["OK"]
        verdict[arm] = {"A_dPDMS_ci_lower_gt_0": cond_a, "dPDMS_RJx_minus_RJ5_R": p["pdms"], "B_no_drop_DAC_NC_C": cond_b,
                        "B_detail_point_dsub": {k: p[k] for k in ("dac", "nc", "c")}, "C_longitudinal_guard": cond_c, "C_detail": mech,
                        "knob_reached": reach,
                        "BEATS_RJ5": bool(cond_a and all(cond_b.values()) and cond_c is True and reach)}
        if cond_c is None:
            verdict[arm]["note"] = "bar C UNVERIFIED (family block missing/raised): verdict fails safe to 5.0 stays"
    out["BAR"] = verdict
    out["BEST"] = [x for x in NEW if verdict[x]["BEATS_RJ5"]]
    out["DECISION"] = ("5.0 stays" if not out["BEST"] else
                       "+".join(out["BEST"]) + " beats RJ5 on the bar" + (
                           "; both: take the larger dPDMS lower-bound arm for the next card" if len(out["BEST"]) == 2 else ""))
    pic = {}
    for arm in ("R0", "RJ3", "RJ5", "RJ8"):
        lv = out["levels"][f"{arm}.R"]
        lon = fam.get(arm, {}).get("longitudinal", {})
        pic[arm] = {"jerk": JERK[arm], "EP": lv["ep"], "DAC": lv["dac"], "TTC": lv["ttc"], "PDMS": lv["pdms"],
                    "speed_bias_mps": lon.get("speed_bias_mps"), "speed_mae_mps": lon.get("speed_mae_mps")}
    out["monotone_picture_R_descriptive_only"] = pic
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"n": out["n_tokens"], "logs": out["n_logs"], "BAR": {k: v["BEATS_RJ5"] for k, v in verdict.items()}, "DECISION": out["DECISION"],
                      "levels_pdms": {k: round(v["pdms"], 2) for k, v in out["levels"].items()}}, indent=0))


if __name__ == "__main__":
    main()
