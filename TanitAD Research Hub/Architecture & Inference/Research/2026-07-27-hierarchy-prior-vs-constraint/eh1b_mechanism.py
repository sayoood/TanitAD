#!/usr/bin/env python3
"""E-H1b — POST-HOC mechanism for E-H1.  Not pre-registered; labelled as post-hoc.

E-H1 returned COMMITMENT-INFORMATIVE and threw up one number that demands an
explanation before anything is quoted:  ``R_rand(q=8) = 16.35`` is WORSE than a
single uniform random pick over the whole fan (``R_random = 15.87``, Bar A /
E-V5-1).  Picking the best-of-8 under an informative score should not be worse
than picking one at random.  Per "verify before alarming", measure what the
ranking actually does rather than reasoning about it.

Three measurements, all CPU, all from the same staged artifacts:

  M1  mean candidate error BY DEPLOYED RANK POSITION -- is the deployed score
      informative only at the top, or all the way down?
  M2  for each restriction arm: on what fraction of windows does the pick differ
      from the flat pick, and what is the damage CONDITIONAL on differing?  The
      whole cost of any restriction lives on exactly those windows.
  M3  the rank-position distribution of ``H_graft(q)``'s pick -- how far down the
      deployed ranking does the commitment actually force us?

M3 recovers the pick index by matching the staged per-window ADE against
``fan_err4``; the match uniqueness is measured and reported, not assumed.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
V5 = (REPO / "TanitAD Research Hub" / "Architecture & Inference" / "Implementation"
      / "incoming" / "2026-07-26-v5-imagination-selection" / "raw")
sys.path.insert(0, str(REPO / "taniteval"))
from taniteval.ci import paired_episode_cluster_bootstrap  # noqa: E402


def main() -> None:
    red = torch.load(V5 / "v5_v4_windows_reduced.pt", map_location="cpu",
                     weights_only=False)
    hier = torch.load(V5 / "v5_hier_windows.pt", map_location="cpu",
                      weights_only=False)
    fan_err = red["fan_err4"].numpy().astype(np.float64)
    base_rank = red["base_rank"].numpy().astype(np.int64)
    ep = red["ep"].numpy().astype(np.int64)
    W, N = fan_err.shape
    pos = np.empty_like(base_rank)
    np.put_along_axis(pos, base_rank, np.arange(N)[None, :].repeat(W, 0), axis=1)
    flat_pick = base_rank[:, 0]
    flat = fan_err[np.arange(W), flat_pick]

    out: dict = {
        "_experiment": "E-H1b -- POST-HOC mechanism for E-H1 (NOT pre-registered)",
        "_evidence_class": "MEASURED (ours, CPU, staged artifacts only)",
        "_host": platform.node(), "_python": platform.python_version(),
        "_n_windows": int(W), "_n_candidates": int(N),
    }

    # ---- M1: mean error by deployed rank position -------------------------
    by_rank = np.take_along_axis(fan_err, base_rank, axis=1).mean(axis=0)  # [N]
    out["M1_mean_error_by_rank_position"] = {
        "rank_0_the_flat_pick": round(float(by_rank[0]), 4),
        "rank_1": round(float(by_rank[1]), 4),
        "rank_2": round(float(by_rank[2]), 4),
        "rank_4": round(float(by_rank[4]), 4),
        "rank_8": round(float(by_rank[8]), 4),
        "rank_16": round(float(by_rank[16]), 4),
        "rank_32": round(float(by_rank[32]), 4),
        "rank_64": round(float(by_rank[64]), 4),
        "rank_128": round(float(by_rank[128]), 4),
        "rank_255_worst_ranked": round(float(by_rank[255]), 4),
        "mean_over_all_candidates": round(float(fan_err.mean()), 4),
        "argmin_of_the_curve": int(by_rank.argmin()),
        "min_of_the_curve": round(float(by_rank.min()), 4),
        "max_of_the_curve": round(float(by_rank.max()), 4),
        "spearman_rank_vs_error": round(float(np.corrcoef(
            np.arange(N), by_rank)[0, 1]), 4),
    }

    # ---- M2/M3: per H_graft arm -------------------------------------------
    arms = []
    for q in (8, 16, 32, 64):
        h = hier[f"produced|H_graft_q{q}"].numpy().astype(np.float64)
        # recover the pick index by matching the ADE within the row
        d = np.abs(fan_err - h[:, None])
        pick = d.argmin(axis=1)
        nmatch = (d < 1e-6).sum(axis=1)
        differs = pick != flat_pick
        r = {
            "q": q,
            "ade": round(float(h.mean()), 4),
            "M3_pick_recovery": {
                "windows_with_exactly_one_match": int((nmatch == 1).sum()),
                "windows_with_no_match_below_1e-6": int((nmatch == 0).sum()),
                "windows_with_ties": int((nmatch > 1).sum()),
                "max_abs_residual": round(float(d.min(axis=1).max()), 8),
            },
            "M3_rank_position_of_pick": {
                "median": int(np.median(pos[np.arange(W), pick])),
                "mean": round(float(pos[np.arange(W), pick].mean()), 2),
                "p90": int(np.percentile(pos[np.arange(W), pick], 90)),
                "max": int(pos[np.arange(W), pick].max()),
                "frac_at_rank_0_commitment_not_binding":
                    round(float((pos[np.arange(W), pick] == 0).mean()), 4),
            },
            "M2_conditional_damage": {
                "frac_windows_pick_differs_from_flat": round(float(differs.mean()), 4),
                "n_windows_differing": int(differs.sum()),
                "flat_ade_on_those_windows": round(float(flat[differs].mean()), 4),
                "arm_ade_on_those_windows": round(float(h[differs].mean()), 4),
                "mean_damage_where_it_acts": round(
                    float((h[differs] - flat[differs]).mean()), 4),
                "flat_ade_where_pick_agrees": round(float(flat[~differs].mean()), 4),
            },
        }
        arms.append(r)
        print(f"q={q:3d} ade={r['ade']:.4f} differs={r['M2_conditional_damage']['frac_windows_pick_differs_from_flat']:.3f} "
              f"median_rank={r['M3_rank_position_of_pick']['median']} "
              f"damage_where_acts={r['M2_conditional_damage']['mean_damage_where_it_acts']:.4f}",
              flush=True)
    out["arms"] = arms

    # ---- the matched control the pre-registration could not compute --------
    # R_deep(q): keep the flat pick OUT and take the best-ranked of a random
    # q-subset of the remaining 255. This isolates "the restriction excluded the
    # flat argmax" from "the restriction was random".
    rng_means = []
    for s in range(50):
        rng = np.random.default_rng(7000 + s)
        keys = rng.random((W, N))
        keys[np.arange(W), flat_pick] = np.inf          # force-exclude the flat pick
        sub = np.argsort(keys, axis=1)[:, :64]
        sp = np.take_along_axis(pos, sub, axis=1)
        pick = np.take_along_axis(sub, sp.argmin(axis=1)[:, None], axis=1)[:, 0]
        rng_means.append(float(fan_err[np.arange(W), pick].mean()))
    out["R_rand64_excluding_flat_pick"] = {
        "mean": round(float(np.mean(rng_means)), 4),
        "note": ("random 64-subset that is FORCED to exclude the flat argmax; "
                 "compare to H_graft(64) = 1.0621 and R_rand(64) = 15.7172"),
    }

    # sanity: the paired CI of H_graft(64) vs flat, reproducing the V5 number
    h64 = hier["produced|H_graft_q64"].numpy().astype(np.float64)
    out["reproduce_V5_paired_H64_minus_flat"] = paired_episode_cluster_bootstrap(
        h64, flat, ep, n_boot=2000, seed=0)

    dst = Path(__file__).parent / "eh1b_mechanism.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["M1_mean_error_by_rank_position"], indent=2))
    print("R_rand64 excluding flat pick:", out["R_rand64_excluding_flat_pick"])
    print("reproduce V5 H64-flat:", out["reproduce_V5_paired_H64_minus_flat"])
    print("wrote", dst)


if __name__ == "__main__":
    main()
