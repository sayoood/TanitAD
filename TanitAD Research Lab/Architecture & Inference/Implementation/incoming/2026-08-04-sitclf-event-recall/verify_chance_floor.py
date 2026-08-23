"""INDEPENDENT VERIFIER — the analytic chance floor vs the 200-draw simulation.

`run_event_retune.py` measures the uniform-random floor by SIMULATION (200 random score
columns at the identical budget). The promoted `event_anticipation_report` now computes the
same floor ANALYTICALLY, as an exact hypergeometric expectation. Two independent derivations
agreeing on the same substrate is the check; a disagreement means one of them is wrong and
neither number may be quoted.

It also re-derives every headline of the two candidate cells through the promoted function,
so the report's numbers are not this stream checking its own JSON.

usage:
  python verify_chance_floor.py --out verify_chance_floor.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "2026-08-03-sitclf-horizon"
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "stack"))

from tanitad.eval.sitclf_deploy import event_anticipation_report        # noqa: E402

CELLS = ("CELL_w8_L3.0", "CELL_w8_L1.0", "CELL_w32_L1.0", "CELL_w16_L2.0",
         "CELL_w1_L1.0", "CELL_w4_L1.0")
TOL_FLOOR = 0.05          # simulation SE over 200 draws is ~0.02-0.03 on 37-231 onsets
TOL_HEADLINE = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(PARENT / "results_horizon_ps.scores.npz"))
    ap.add_argument("--results", default="results_event_retune.json")
    ap.add_argument("--out", default="verify_chance_floor.json")
    a = ap.parse_args()

    z = np.load(a.scores)
    R = json.loads(Path(a.results).read_text(encoding="utf-8"))
    sits = [str(s) for s in z["situations"]]
    EVAL = z["eval_rows"].astype(bool)
    cc, onsets, osit = z["clip_cluster"], z["onsets"], z["onset_sit"]

    out = {"_what": ("the ANALYTIC chance floor (promoted stack fn) vs the 200-draw SIMULATION "
                     "(this stream), plus every candidate headline re-derived independently"),
           "_implementation": "tanitad.eval.sitclf_deploy.event_anticipation_report",
           "situations": {}}
    bad = []
    for i, s in enumerate(sits):
        og = onsets[osit == i]
        sim = R["per_situation"][s]["C_CHANCE"]
        row = {"simulated_floor": sim, "cells": {}}
        for cname in CELLS:
            got = event_anticipation_report(
                z[cname][:, i].astype(np.float64), EVAL[:, i], og, cc,
                top_frac=R["top_frac"], h_max_s=R["h_max_s"], deploy_lead_s=3.0)
            want = R["per_situation"][s]["arms"][cname]
            d_r = abs(got["event_recall"] - want["event_recall"]["point"])
            d_p5 = abs(got["alarm_precision_h_max"] - want["alarm_precision_5s"]["point"])
            d_p3 = abs(got["alarm_precision_deploy"] - want["alarm_precision_3s"]["point"])
            d_floor = abs(got["chance_event_recall"] - sim["event_recall"]["mean"])
            d_cp5 = abs(got["chance_alarm_precision_h_max"] - sim["alarm_precision_5s"]["mean"])
            row["cells"][cname] = {
                "event_recall": got["event_recall"],
                "chance_event_recall_ANALYTIC": got["chance_event_recall"],
                "chance_event_recall_SIMULATED": sim["event_recall"]["mean"],
                "abs_diff_floor": round(d_floor, 6),
                "chance_alarm_precision_5s_ANALYTIC": got["chance_alarm_precision_h_max"],
                "chance_alarm_precision_5s_SIMULATED": sim["alarm_precision_5s"]["mean"],
                "abs_diff_chance_prec5": round(d_cp5, 6),
                "alarm_precision_lift_h_max": got["alarm_precision_lift_h_max"],
                "alarm_precision_lift_deploy": got["alarm_precision_lift_deploy"],
                "abs_diff_vs_this_stream": {"event_recall": round(d_r, 9),
                                            "alarm_prec_5s": round(d_p5, 9),
                                            "alarm_prec_3s": round(d_p3, 9),
                                            "n_alarm_match": got["n_alarm"] == want["n_alarm"]}}
            if d_r > TOL_HEADLINE or d_p5 > TOL_HEADLINE or d_p3 > TOL_HEADLINE:
                bad.append((s, cname, "headline", d_r, d_p5, d_p3))
            if d_floor > TOL_FLOOR or d_cp5 > TOL_FLOOR:
                bad.append((s, cname, "floor", d_floor, d_cp5))
            if got["n_alarm"] != want["n_alarm"]:
                bad.append((s, cname, "n_alarm", got["n_alarm"], want["n_alarm"]))
        out["situations"][s] = row
        c0 = row["cells"][CELLS[0]]
        print(f"[{time.strftime('%H:%M:%S')}]  {s:>13}: analytic floor "
              f"{c0['chance_event_recall_ANALYTIC']:.4f} vs simulated "
              f"{c0['chance_event_recall_SIMULATED']:.4f} "
              f"(|d| {c0['abs_diff_floor']:.4f}) | chance prec@5s analytic "
              f"{c0['chance_alarm_precision_5s_ANALYTIC']:.4f} vs "
              f"{c0['chance_alarm_precision_5s_SIMULATED']:.4f}", flush=True)
    out["MISMATCHES"] = bad
    out["VERDICT"] = ("PASS — the analytic floor agrees with the simulation and every headline "
                      "re-derives" if not bad else f"FAIL — {len(bad)} mismatches")
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] {out['VERDICT']}", flush=True)
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
