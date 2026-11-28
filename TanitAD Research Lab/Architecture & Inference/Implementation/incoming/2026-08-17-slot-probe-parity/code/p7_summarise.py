"""P7 — pull every banked result into ONE table + the trajectory."""
import json, sys
from pathlib import Path
raw = Path(sys.argv[1])
ORDER = [("s02000", 2000), ("s09000", 9000), ("s09250", 9250),
         ("s10000", 10000), ("s11250", 11250)]

def pull(r, arm):
    a = r["per_arm"][arm]; v = r["verdict"][arm]
    g = lambda k: r["paired"].get(f"{arm} vs {k}", {})
    def fmt(d):
        return [d.get("delta"), d.get("lo"), d.get("hi"), d.get("separated")]
    return {
        "n_params": r["arms"][arm].get("n_params"),
        "err_mean": round(a["lead_gap_abs_err_m"]["mean"], 4),
        "err_ci": [a["lead_gap_abs_err_m"]["lo"], a["lead_gap_abs_err_m"]["hi"]],
        "err_median": round(a["median_abs_err_m"], 4),
        "K1_vs_CONST": fmt(g("C-CONST")),
        "K5_vs_EPMEAN": fmt(g("C-EPMEAN")),
        "K2_vs_SHUF": fmt(g(f"{arm}__C-SHUF")),
        "XEP_vs_SHUFXEP": fmt(g(f"{arm}__C-SHUF-XEP")),
        "K3_recall": round(v["K3_recall"], 4),
        "verdict": v["admissible_as"],
        "mean_pred_gap": round(a["mean_pred_gap_m"], 3),
        "mean_gt_gap": round(a["mean_gt_gap_m"], 3),
        "oracle_median": round(a["_diag_oracle_slot_abs_err_m"]["median"], 4),
    }

def common(r):
    return {"stamp": r["run_stamp"], "clusters": r.get("n_bootstrap_clusters"),
            "windows": r["n_scored_windows"],
            "eval_eps_with_lead": r.get("n_eval_episodes_with_gt_lead"),
            "C-CONST": round(r["per_arm"]["C-CONST"]["lead_gap_abs_err_m"]["mean"], 4),
            "C-EPMEAN": (round(r["per_arm"]["C-EPMEAN"]["lead_gap_abs_err_m"]["mean"], 4)
                         if "C-EPMEAN" in r["per_arm"] else None),
            "d_rule": r.get("d_rule"),
            "swap_mae": r.get("c_shuf_discriminability", {}).get("realised_swap_mae_m")}

rows = []
for lbl, step in ORDER:
    p = raw / f"results_{lbl}.json"
    if not p.exists():
        rows.append({"label": lbl, "step": step, "status": "NOT RUN"}); continue
    r = json.loads(p.read_text("utf-8"))
    rows.append({"label": lbl, "step": step, **common(r),
                 **{k: v for k, v in pull(r, "cells").items()}})
extra = {}
for lbl in ("tok11250", "NULLCONTROL", "s11250_nq32",
            "REGRESSION_vs_20260816", "REGRESSION_post_amendment"):
    p = raw / f"results_{lbl}.json"
    if not p.exists():
        extra[lbl] = "NOT RUN"; continue
    r = json.loads(p.read_text("utf-8"))
    extra[lbl] = {**common(r),
                  "arms": {arm: pull(r, arm) for arm in r["verdict"]}}
out = {"trajectory_cells": rows, "other": extra}
(raw / "SUMMARY.json").write_text(json.dumps(out, indent=1, default=str), "utf-8")
print(json.dumps(out, indent=1, default=str))
