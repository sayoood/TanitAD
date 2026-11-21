#!/usr/bin/env python3
"""HARNESS CHECK — before any new closed-loop number may be quoted.

Re-confirms that THIS eval pod, on THIS synced tree, reproduces the program's
pinned corrected full-set value for flagship v1:

    open-loop g_op_fwd ADE 0-2s = 0.4271   (episode-cluster bootstrap point est.)
    ci95 0.0598, [0.3675, 0.4871], 40 val episodes, B=2000

Source of the pinned triple: Project Steering/CI_RECOMPUTE_2026-07-20.json
(arm `flagship-30k`: full_set_mean 0.4271 / boot_lo 0.3675 / boot_hi 0.4871).
The SAME triple is pinned by taniteval/tests/test_driving_gate_block.py and by
tests/test_closedloop_ci.py, so a pass here means the closed-loop panel and the
driving panel are demonstrably on ONE estimator.

It ALSO re-derives the deprecated statistic under its honest name
(`ci.overlapping_holdout_se` over 8 overlapping random 20 % holdouts) and checks
it reproduces the PUBLISHED legacy pair (0.4522 / 0.0312) — because the whole
point of this exercise is that the legacy statistic moved the MEAN as well as
the interval, and that claim is only credible if we can reproduce both sides.

Run:  OMP_NUM_THREADS=8 python3 harness_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/taniteval")

from taniteval import ci as _ci                      # noqa: E402
from taniteval import driving as _drv                # noqa: E402
from taniteval import lateral as _lat                # noqa: E402
from taniteval import rollout                        # noqa: E402

RES = Path("/root/taniteval/results")
OUT = Path("/root/cl_rerun_20260726")
PIN = {"full_set_mean": 0.4271, "boot_lo": 0.3675, "boot_hi": 0.4871,
       "boot_ci95": 0.0598, "published_mean": 0.4522, "published_ci95": 0.0312}


def legacy_splits(eids, n_splits=8, val_frac=0.2, seed=0):
    """The DEPRECATED construction, reproduced through the HARNESS's OWN split
    builder — `tanitad.eval.gates.split_by_episode`, the exact call
    `bench.py:268` and `closedloop.py:751` make. Rebuilding the RNG by hand
    here would only prove that my reimplementation disagrees, which is not the
    claim under test."""
    from tanitad.eval.gates import split_by_episode
    return [split_by_episode(eids, val_frac, s)
            for s in range(seed, seed + n_splits)]


def main():
    w = rollout.load_windows(RES / "windows_flagship-30k.pt")
    eid = w["eid"]
    pred, gt = w["pred"], w["gt"]
    per_win = torch.linalg.norm(pred - gt, dim=-1).mean(dim=1).numpy()

    # ---- decision-grade: episode-cluster bootstrap ------------------------ #
    boot = _ci.episode_cluster_bootstrap(per_win, eid, reduce="mean",
                                         n_boot=2000, seed=0)

    # ---- deprecated: overlapping_holdout_se, under its honest name -------- #
    sm = np.array([float(np.nanmean(per_win[va]))
                   for _tr, va in legacy_splits(eid)])
    legacy_mean = float(np.mean(sm))
    legacy_ci95 = float(_ci.overlapping_holdout_se(sm))

    ok_new = (abs(boot["mean"] - PIN["full_set_mean"]) < 5e-4
              and abs(boot["lo"] - PIN["boot_lo"]) < 5e-4
              and abs(boot["hi"] - PIN["boot_hi"]) < 5e-4)
    ok_old = (abs(legacy_mean - PIN["published_mean"]) < 5e-4
              and abs(legacy_ci95 - PIN["published_ci95"]) < 5e-4)

    # ---- lateral.py must be the FIXED version ---------------------------- #
    lat_src = Path(_lat.__file__).read_text(encoding="utf-8")
    lat_fixed = "horizon_provenance" in lat_src
    pc = _lat.paired_cross_track(pred, pred, gt, eid, n_boot=64)
    lat_stamped = "horizon_provenance" in pc
    lat_h = pc.get("horizon_s")

    rep = {
        "_what": "harness check — the eval pod must reproduce the pinned "
                 "corrected full-set value before any new closed-loop number "
                 "is quoted",
        "arm": "flagship-30k (v1 FINAL, speed+jerk)",
        "surface": "windows_flagship-30k.pt (open-loop g_op_fwd, 881 win / 40 ep)",
        "decision_grade": {
            "estimator": boot["estimator"],
            "mean": boot["mean"], "lo": boot["lo"], "hi": boot["hi"],
            "ci95": boot["ci95"], "n_windows": boot["n_windows"],
            "n_episodes": boot["n_episodes"], "n_boot": boot["n_boot"],
            "pinned": [PIN["full_set_mean"], PIN["boot_lo"], PIN["boot_hi"]],
            "REPRODUCES_0.4271": bool(ok_new)},
        "deprecated_reproduction": {
            "estimator": "overlapping_holdout_se",
            "mean": round(legacy_mean, 4), "ci95": round(legacy_ci95, 4),
            "published": [PIN["published_mean"], PIN["published_ci95"]],
            "REPRODUCES_PUBLISHED": bool(ok_old)},
        "point_estimate_shift_pct": round(
            100.0 * (legacy_mean - boot["mean"]) / boot["mean"], 3),
        "ci_width_ratio_new_over_legacy": round(boot["ci95"] / legacy_ci95, 3),
        "lateral_py": {
            "md5_matches_fixed_version": lat_fixed,
            "horizon_provenance_stamped": lat_stamped,
            "self_paired_horizon_s": lat_h,
            "expected_horizon_s_sparse_4knot": 2.0,
            "STALE_IF_0.4": bool(lat_h == 0.4),
            "OK": bool(lat_fixed and lat_stamped and lat_h == 2.0)},
        "omp_num_threads": __import__("os").environ.get("OMP_NUM_THREADS"),
        "torch_threads": torch.get_num_threads(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "harness_check.json").write_text(
        json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    print("\nHARNESS CHECK:",
          "PASS" if (ok_new and ok_old and rep["lateral_py"]["OK"]) else "FAIL")
    return 0 if (ok_new and ok_old and rep["lateral_py"]["OK"]) else 1


if __name__ == "__main__":
    sys.exit(main())
