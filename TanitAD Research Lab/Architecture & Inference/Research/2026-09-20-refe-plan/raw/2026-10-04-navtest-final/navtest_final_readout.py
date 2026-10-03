#!/usr/bin/env python3
"""The FULL navtest readout of REFe's final checkpoint (model_final.pt, md5 b54773f8c302b2ec8ff4c5971d1e1370; Amendment 7
repair ON; rule navsim_v1; Amendment 8 OFF), from the harness's own per-token CSV. CPU, numpy only.

Reads: PDMS (+ log-cluster bootstrap CI of the mean, 10,000 resamples, seed 20260927) and sub-scores on all 12,146 tokens;
the same on the 12,110 tokens that pass the seam's frame/time control (frame_control_navtest_full.json); the off-route
split (goal_census ego_to_route > 20 m); the PDMS = 0 census by failing term; the scorer's PREDICTED score of the pick vs
the harness's TRUE score (proposals.npz logits, the same forward pass). Writes navtest_final_readout.json.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
DRV = os.environ.get("REFE_DRIVE", "D:")
sys.path.insert(0, str(PKG / "eval"))
import proxy_eval as PE  # noqa: E402

DATA = Path(f"{DRV}/Projects/TanitAD/data/refe_navtest")
CSV = DATA / "score" / "refe_navtest_final" / "refe_navtest_final.csv"
PROPS = DATA / "proptable" / "navtest_final" / "proposals.npz"
EXPORT = f"{DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
TERMS = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "EP": "ego_progress",
         "TTC": "time_to_collision_within_bound", "C": "comfort", "DDC": "driving_direction_compliance"}


def agg_v1(logits):
    """navsim_v1 rule over sigmoid probabilities: NC * DAC * (5 EP + 5 TTC + 2 C) / 12 (component order NC DAC EP TTC C DDC)"""
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    return p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12.0


def block(rows, toks, tl):
    s = {t: 100.0 * float(rows[t]["score"]) for t in toks}
    mu, lo, hi = PE.boot(s, tl)
    out = {"n": len(toks), "n_logs": len({tl[t] for t in toks}), "PDMS": mu, "ci95_log_cluster": [lo, hi]}
    for k, c in TERMS.items():
        out[k] = 100.0 * float(np.mean([float(rows[t][c]) for t in toks]))
    out["n_pdms_zero"] = int(sum(s[t] == 0.0 for t in toks))
    return out


def main() -> int:
    rows = PE.read_csv(CSV)
    E = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    tl = {t: E[t]["log_name"] for t in E}
    toks = sorted(rows)
    fc = json.load(open(HERE / "frame_control_navtest_full.json", encoding="utf-8"))
    bad = set(fc["over_bar"])
    gc = json.load(open(PKG / "raw" / "2026-10-01-goal-trigger" / "goal_census_navtest_full.json", encoding="utf-8"))["rows"]
    off = {t for t in toks if gc.get(t, {}).get("ego_to_route_m", 0) > 20.0}
    res = {"csv": str(CSV), "csv_sha256": PE.sha256(CSV), "n_csv_rows": len(rows),
           "estimator": "log-cluster bootstrap of the per-token mean, 10,000 resamples, percentile 95 %, seed 20260927 "
                        "(answers 'another draw of LOGS'; one checkpoint, deterministic inference)",
           "tier": "NAVSIM v1.1 PDMS (navtest) = non-reactive pseudo-simulation of a 4 s open-loop plan",
           "all_12146": block(rows, toks, tl),
           "frame_control_clean_12110": block(rows, [t for t in toks if t not in bad], tl),
           "frame_control_fail_36": block(rows, [t for t in toks if t in bad], tl),
           "on_route": block(rows, [t for t in toks if t not in off], tl),
           "off_route_gt20m": block(rows, [t for t in toks if t in off], tl)}
    res["off_route_cost_to_full_mean"] = (res["on_route"]["PDMS"] - res["off_route_gt20m"]["PDMS"]) * len(off) / len(toks)
    # by driving command
    cmd = {t: int(np.argmax(E[t]["ego_statuses"][-1]["driving_command"])) for t in toks}
    res["by_command"] = {nm: block(rows, [t for t in toks if cmd[t] == i], tl)
                         for i, nm in enumerate(("LEFT", "STRAIGHT", "RIGHT")) if any(cmd[t] == i for t in toks)}
    # PDMS = 0 census
    zero = [t for t in toks if float(rows[t]["score"]) == 0.0]
    why = {}
    for t in zero:
        nc, dac = float(rows[t][TERMS["NC"]]), float(rows[t][TERMS["DAC"]])
        k = "NC+DAC" if nc == 0 and dac == 0 else "NC" if nc == 0 else "DAC" if dac == 0 else "EP=TTC=C=0"
        why[k] = why.get(k, 0) + 1
    res["pdms_zero"] = {"n": len(zero), "share": len(zero) / len(toks), "by_failing_term": why,
                        "in_off_route": int(sum(t in off for t in zero)),
                        "mean_loss_to_full_if_zero_tokens_scored_like_the_rest":
                            len(zero) / len(toks) * res["all_12146"]["PDMS"],
                        "by_command": {nm: int(sum(cmd[t] == i for t in zero)) for i, nm in enumerate(("LEFT", "STRAIGHT", "RIGHT"))}}
    # predicted vs true for the pick (same forward pass)
    if PROPS.exists():
        P = np.load(PROPS)
        idx = {str(t): i for i, t in enumerate(P["token"])}
        pred = {t: 100.0 * float(agg_v1(P["logits"][idx[t]][int(P["pick"][idx[t]])])) for t in toks if t in idx}
        tz = [t for t in zero if t in pred]
        tnz = [t for t in toks if t in pred and t not in set(zero)]
        am = [int(np.argmax(agg_v1(P["logits"][i]))) == int(P["pick"][i]) for i in range(len(P["token"]))]
        res["scorer_on_pick"] = {
            "n": len(pred),
            "CONTROL_argmax_of_this_agg_equals_planner_pick": f"{sum(am)}/{len(am)}",
            "pred_pdms_mean_all": float(np.mean(list(pred.values()))),
            "pred_pdms_of_zero_picks": {"mean": float(np.mean([pred[t] for t in tz])) if tz else None,
                                        "median": float(np.median([pred[t] for t in tz])) if tz else None,
                                        "n_pred_over_80": int(sum(pred[t] > 80 for t in tz))},
            "pred_pdms_of_nonzero_picks_mean": float(np.mean([pred[t] for t in tnz])) if tnz else None,
            "corr_pred_true": float(np.corrcoef([pred[t] for t in pred], [100 * float(rows[t]["score"]) for t in pred])[0, 1])}
    json.dump(res, open(HERE / "navtest_final_readout.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
