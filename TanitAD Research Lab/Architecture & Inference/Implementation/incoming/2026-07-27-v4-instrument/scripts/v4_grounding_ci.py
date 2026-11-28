#!/usr/bin/env python3
"""GAP 1 stage 2 — paired episode-cluster bootstrap over the per-window bands.

ZERO GPU.  Reads the two per-window dumps written by ``v4_grounding_probe.py``
and produces every interval this stream quotes:

  * per-arm band means with an episode-cluster interval;
  * the CROSS-ARM paired contrast v4 - v1 on the IDENTICAL windows (the window
    sets are asserted equal element-by-element first -- if they are not, nothing
    is combined);
  * the WITHIN-ARM paired contrast baseline vs every v0 ablation.

Estimator: ``taniteval.ci.paired_episode_cluster_bootstrap`` /
``episode_cluster_bootstrap``, B = 2000, unit = episode cluster, seed 0.
⛔ ``overlapping_holdout_se`` is NEVER used: it is not a jackknife, and it BIASES
the point estimate (it reports a mean-of-split-means, not the full-set mean).
Every central value here is the ``full_set`` mean.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                    "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "taniteval"))
from taniteval import ci as _ci                                   # noqa: E402

LEVELS = ("op", "tac", "str")
B = 2000


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v4", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A, C = load(a.v1), load(a.v4)
    if A["eid"] != C["eid"] or A["t0"] != C["t0"]:
        raise SystemExit("REFUSING: the two arms' window sets differ. Nothing "
                         "here is paired and no contrast is admissible.")
    eid = A["eid"]
    n_cl = len(set(eid))
    print(f"[ci] {len(eid)} windows / {n_cl} episode clusters, window sets "
          f"asserted IDENTICAL", flush=True)

    out = {
        "experiment": "GAP 1 stage 2 - paired episode-cluster bootstrap",
        "estimator": {"name": "episode-cluster bootstrap (paired where stated)",
                      "module": "taniteval/taniteval/ci.py", "n_boot": B,
                      "seed": 0, "unit": "episode cluster",
                      "REFUSED": "overlapping_holdout_se (biased point "
                                 "estimate + invalid SE)"},
        "n_windows": len(eid), "n_episode_clusters": n_cl,
        "arms": {}, "cross_arm_paired_v4_minus_v1": {},
        "within_arm_paired_ablation_minus_baseline": {},
        "ratio_of_ratios_v4_over_v1": {},
    }

    for name, d in (("v1", A), ("v4fs", C)):
        arm = {}
        for lvl in LEVELS:
            for side, key in (("real_mid_de_m", f"{lvl}_mid"),
                              ("imagined_fwd_ade_m", f"{lvl}_fwd")):
                v = d["per_window"]["baseline"][key].double().numpy()
                r = _ci.episode_cluster_bootstrap(v, eid, n_boot=B, seed=0)
                arm[f"{lvl}__{side}"] = {
                    "full_set_mean": round(float(v.mean()), 6),
                    "ci95": [round(float(r["lo"]), 6), round(float(r["hi"]), 6)]}
            mid = d["per_window"]["baseline"][f"{lvl}_mid"].double().numpy()
            fwd = d["per_window"]["baseline"][f"{lvl}_fwd"].double().numpy()
            arm[f"{lvl}__ratio_real_over_imagined"] = round(
                float(mid.mean() / fwd.mean()), 4)
            arm[f"{lvl}__ratio_agg_corrected_x0.5"] = round(
                float(0.5 * mid.mean() / fwd.mean()), 4)
        out["arms"][name] = arm

    for lvl in LEVELS:
        for side, key in (("real_mid_de_m", f"{lvl}_mid"),
                          ("imagined_fwd_ade_m", f"{lvl}_fwd")):
            x = C["per_window"]["baseline"][key].double().numpy()
            y = A["per_window"]["baseline"][key].double().numpy()
            r = _ci.paired_episode_cluster_bootstrap(x, y, eid, n_boot=B, seed=0)
            out["cross_arm_paired_v4_minus_v1"][f"{lvl}__{side}"] = {
                "delta": round(float(r["delta"]), 6),
                "ci95": [round(float(r["lo"]), 6), round(float(r["hi"]), 6)],
                "separated": bool(r["lo"] > 0 or r["hi"] < 0)}
        rv1 = (A["per_window"]["baseline"][f"{lvl}_mid"].double().mean()
               / A["per_window"]["baseline"][f"{lvl}_fwd"].double().mean())
        rv4 = (C["per_window"]["baseline"][f"{lvl}_mid"].double().mean()
               / C["per_window"]["baseline"][f"{lvl}_fwd"].double().mean())
        out["ratio_of_ratios_v4_over_v1"][lvl] = {
            "v1_ratio": round(float(rv1), 4), "v4_ratio": round(float(rv4), 4),
            "v4_over_v1": round(float(rv4 / rv1), 4),
            "prereg_band_for_OUTCOME_A": "[1/3, 3]",
            "OUTCOME": ("A - comparable, the finding generalises"
                        if 1 / 3 <= float(rv4 / rv1) <= 3 else
                        ("B - v4 materially smaller" if float(rv4 / rv1) < 1 / 3
                         else "C - v4 materially larger"))}

    for name, d in (("v1", A), ("v4fs", C)):
        w = {}
        base = d["per_window"]["baseline"]
        for ab, dd in d["per_window"].items():
            if ab == "baseline":
                continue
            for lvl in LEVELS:
                x = dd[f"{lvl}_fwd"].double().numpy()
                y = base[f"{lvl}_fwd"].double().numpy()
                r = _ci.paired_episode_cluster_bootstrap(x, y, eid,
                                                         n_boot=B, seed=0)
                w[f"{ab}__{lvl}__imagined_fwd_ade_m"] = {
                    "delta": round(float(r["delta"]), 6),
                    "ci95": [round(float(r["lo"]), 6),
                             round(float(r["hi"]), 6)],
                    "separated": bool(r["lo"] > 0 or r["hi"] < 0),
                    "x_vs_baseline": round(float(x.mean() / y.mean()), 4)}
                # the F-A structural control, re-verified from the raw arrays
                xr = dd[f"{lvl}_mid"].double().numpy()
                yr = base[f"{lvl}_mid"].double().numpy()
                w[f"{ab}__{lvl}__real_mid_de_m_max_abs_diff"] = float(
                    np.abs(xr - yr).max())
        out["within_arm_paired_ablation_minus_baseline"][name] = w

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["ratio_of_ratios_v4_over_v1"], indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
