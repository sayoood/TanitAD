#!/usr/bin/env python3
"""TF-N2 re-roll read (navtrain held-out ONLY): R0 identity control + paired contrasts of the three protocol-knob arms against R0.

Estimator: `tfn2_analyze.boot` (paired log-cluster bootstrap, B = 10,000, seed 20260927, 95 % percentile, ratio mean). It answers
"another draw of LOGS"; labels, tracker, scorer and the rollout are deterministic (CPU, fixed seed), so it does not speak to training or
inference variance (the policy is argmax, so there is no sampling).

  python tfn2_reroll_analyze.py --run <run dir> --out <json> [--fam-dir <dir with fam_<ARM>_<FORM>.json>]
Reads <run>/scores/<ARM>_s*.jsonl for ARM in R0, RJ5, RG8, R5HZ. Rows are intersected over the four arms (a token missing in any arm is dropped and counted).
STRATEGIC is PAUSED (PI 2026-09-19) and is printed as PAUSED, never as a pass.
"""
import argparse
import collections
import glob
import json
import lzma
import os
import sys

import numpy as np

N2 = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n2"
sys.path.insert(0, N2)
import tfn2_analyze as A  # noqa: E402

ARMS = ("R0", "RJ5", "RG8", "R5HZ")
FORMS = ("T", "C", "R")
SUB = A.SUB
# R0 identity bars, committed in the hand-off BEFORE any full-run row existed (the 2-token smoke read 0.0 for both):
ID_ROLL_MAX_M = 1e-6          # R0 roll vs the banked t20, max |dxy| over 20 poses and all tokens (CPU rollouts are deterministic)
ID_SCORE_MAX = 1e-9           # R0 sub-scores vs the banked probe rows, max abs diff over all tokens, forms and sub-scores
C3_BAR = "PREREG_P4 C3: median max|dxy| <= 0.05 m AND >= 90 % of tokens <= 0.25 m"


def probe_rows():
    fp = os.path.join(N2, "raw", "probe_rows.jsonl.xz")
    return {r["token"]: r for r in map(json.loads, lzma.open(fp, "rt", encoding="utf-8")) if "error" not in r}


def find(o, key):
    """first value under `key` anywhere in a nested dict (families block lookups; the block's layout is the instrument's)."""
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = find(v, key)
            if r is not None:
                return r
    return None


FAM_KEYS = {"longitudinal": ("speed_mae_mps", "speed_bias_mps", "accel_mae_mps2", "progress_ratio", "min_headway_m", "min_time_gap_s", "min_ttc_s"),
            "lateral": ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm", "cross_mae_m"),
            "tactical": ("accuracy", "goal_point_error_m")}


def fam_summary(fp):
    d = json.load(open(fp, encoding="utf-8"))
    blk = d["frames"]["executed_tracked"]["block"]
    out = {"n": d["n"], "n_logs": d["n_logs"], "strategic": "PAUSED (PI 2026-09-19)"}
    lon, lat, tac = blk["longitudinal"], blk["lateral"], blk["tactical"]
    out["longitudinal"] = {k: v for k, v in lon.items() if isinstance(v, (int, float))}
    out["lateral"] = {k: v for k, v in lat.items() if isinstance(v, (int, float))}
    out["tactical"] = {k: v for k, v in tac.items() if isinstance(v, (int, float))}
    for k in ("lateral_decision", "longitudinal_decision", "five_way", "goal_point"):
        if isinstance(tac.get(k), dict):
            out["tactical"][k] = {x: y for x, y in tac[k].items() if isinstance(y, (int, float))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fam-dir", default="")
    a = ap.parse_args()
    rows = {}
    for arm in ARMS:
        rows[arm] = {r["token"]: r for r in A.load([os.path.join(a.run, "scores", f"{arm}_s*.jsonl")])}
    common = sorted(set.intersection(*[set(v) for v in rows.values()]))
    dropped = {arm: len(v) - len(common) for arm, v in rows.items()}
    R = {arm: [rows[arm][t] for t in common] for arm in ARMS}
    logs = [r["log"] for r in R["R0"]]
    out = {"n_tokens": len(common), "n_logs": len(set(logs)), "dropped_not_in_all_arms": dropped,
           "estimator": "paired log-cluster bootstrap, B=10000, seed 20260927, 95% percentile, ratio mean (tfn2_analyze.boot); answers 'another draw of LOGS' only",
           "scorer": "W-TRANSP functions on the nuPlan-DB rebuild of the metric-cache parts (navtrain route, NOT the pinned harness cache); levels are not harness-identical, paired contrasts are internally consistent",
           "evidence_class": "MEASURED (ours), navtrain held-out, EXPLORATORY: knob values were fixed in advance, not tuned; tier T1 NAVSIM v1.1 PDMS, non-reactive, LQR-tracked",
           "strategic": "PAUSED (PI 2026-09-19)"}
    # ---- R0 identity controls ----------------------------------------------------------------------------------------------
    d = np.array([r["bank_dxy_m"] for r in R["R0"]])
    pr = probe_rows()
    diff = 0.0
    for r in R["R0"]:
        p = pr.get(r["token"])
        if p is None:
            continue
        for f in FORMS + ("human",):
            for fr in ("tracked", "transparent"):
                for k, v in r["arms"][f][fr].items():
                    diff = max(diff, abs(v - p["arms"][f][fr][k]))
    ident = {"roll_vs_bank_max_dxy_m": {"max": float(d.max()), "median": float(np.median(d)), "share_le_1e-6": float((d <= 1e-6).mean()),
                                        "share_le_0.01": float((d <= 0.01).mean()), "share_le_0.25": float((d <= 0.25).mean())},
             "bar_roll": f"max <= {ID_ROLL_MAX_M} m", "pass_roll": bool(d.max() <= ID_ROLL_MAX_M),
             "C3_bar": C3_BAR, "pass_C3": bool(np.median(d) <= 0.05 and (d <= 0.25).mean() >= 0.90),
             "score_vs_probe_rows_max_abs_diff": diff, "bar_score": f"<= {ID_SCORE_MAX}", "pass_score": bool(diff <= ID_SCORE_MAX)}
    ident["PASS"] = bool(ident["pass_roll"] and ident["pass_score"])
    out["R0_identity"] = ident
    # ---- knob reachability, from the rows themselves ----------------------------------------------------------------------
    out["knobs_in_rows"] = {arm: dict(collections.Counter(json.dumps({k: v for k, v in r["knobs"].items() if k != "max_jerk_long"}, sort_keys=True)
                                                        for r in R[arm])) for arm in ARMS}
    # ---- levels and paired contrasts ---------------------------------------------------------------------------------------
    out["levels"], out["paired_vs_R0_same_form"], out["paired_vs_R0_C"] = {}, {}, {}
    for arm in ARMS:
        for f in FORMS:
            lv = {k: 100 * float(np.mean([A.g(r, f, k) for r in R[arm]])) for k in SUB}
            mult = np.array([A.g(r, f, "mult") for r in R[arm]])
            ep = np.array([A.g(r, f, "ep") for r in R[arm]])
            lv["zero_mult_pct"] = 100 * float((mult == 0).mean())
            lv["ep_alive"] = 100 * float(ep[mult > 0].mean())
            lv["pdms_ci"] = [100 * x for x in A.boot([A.g(r, f, "pdms") for r in R[arm]], logs)]
            out["levels"][f"{arm}.{f}"] = lv
            if arm != "R0":
                out["paired_vs_R0_same_form"][f"{arm}.{f}"] = {
                    k: [100 * x for x in A.boot([A.g(r, f, k) for r in R[arm]], logs, [A.g(r, f, k) for r in R["R0"]])] for k in SUB}
            if f == "R" or arm != "R0":
                out["paired_vs_R0_C"][f"{arm}.{f}"] = {
                    k: [100 * x for x in A.boot([A.g(r, f, k) for r in R[arm]], logs, [A.g(r, "C", k) for r in R["R0"]])] for k in SUB}
    # ---- mechanism: did the knob move the teacher's own motion? ------------------------------------------------------------
    mech = {}
    pdm = np.array([r["mech"]["pdm_path4_m"] for r in R["R0"]])
    for arm in ARMS:
        tp = np.array([r["mech"]["t_path4_m"] for r in R[arm]])
        tv4 = np.array([r["mech"]["t_v4"] for r in R[arm]])
        pv4 = np.array([r["mech"]["pdm_v4"] for r in R[arm]])
        ta = np.array([r["mech"]["t_amax"] for r in R[arm]])
        mech[arm] = {"teacher_path4_over_pdm_path4_median": float(np.median(tp[pdm > 1] / pdm[pdm > 1])),
                     "teacher_v4_minus_pdm_v4_median": float(np.median(tv4 - pv4)), "teacher_amax_median": float(np.median(ta)),
                     "teacher_amax_p95": float(np.percentile(ta, 95)),
                     "teacher_path4_minus_R0_m": A.boot(tp, logs, np.array([r["mech"]["t_path4_m"] for r in R["R0"]]))}
    out["mechanism"] = mech
    # ---- DAC / NC / comfort failures that a lever introduced (token counts, C form, tracked) ---------------------------------
    flips = {}
    for arm in ARMS[1:]:
        for k in ("dac", "nc", "c"):
            base = np.array([A.g(r, "C", k) for r in R["R0"]])
            cur = np.array([A.g(r, "C", k) for r in R[arm]])
            flips[f"{arm}.C.{k}"] = {"worse": int((cur < base - 1e-9).sum()), "better": int((cur > base + 1e-9).sum())}
    out["token_flips_vs_R0_C_tracked"] = flips
    # ---- families (from families_run.py outputs, descriptive: no paired family CI exists, N1 §10 work item) -----------------
    if a.fam_dir:
        out["families_executed_tracked"] = {}
        for arm in ARMS:
            for f in ("C", "R"):
                fp = os.path.join(a.fam_dir, f"fam_{arm}_{f}.json")
                if os.path.exists(fp):
                    out["families_executed_tracked"][f"{arm}.{f}"] = fam_summary(fp)
    # ---- bar reading, mechanical (committed in the hand-off; the harness bars stay 95.8 / DAC 99.9, navtest is N2's, not this run's) ----
    best = max((k for k in out["levels"] if not k.startswith("R0.")), key=lambda k: out["levels"][k]["pdms"])
    out["best_arm_form_by_pdms_EXPLORATORY"] = {"key": best, "pdms": out["levels"][best]["pdms"],
                                                 "delta_vs_R0_same_form": out["paired_vs_R0_same_form"][best]["pdms"],
                                                 "delta_vs_R0_C": out["paired_vs_R0_C"][best]["pdms"]}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"n": out["n_tokens"], "logs": out["n_logs"], "R0_identity_PASS": ident["PASS"],
                      "levels_pdms": {k: round(v["pdms"], 2) for k, v in out["levels"].items()},
                      "paired_pdms_vs_R0_same_form": {k: [round(x, 2) for x in v["pdms"]] for k, v in out["paired_vs_R0_same_form"].items()},
                      "paired_ep_vs_R0_same_form": {k: [round(x, 2) for x in v["ep"]] for k, v in out["paired_vs_R0_same_form"].items()}}, indent=0))


if __name__ == "__main__":
    main()
