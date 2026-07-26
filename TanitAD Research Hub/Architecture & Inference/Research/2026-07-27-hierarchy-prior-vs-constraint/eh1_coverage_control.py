#!/usr/bin/env python3
"""E-H1 — the COVERAGE CONTROL for the E-V5-2 hierarchical-commitment result.

PRE-REGISTERED in ``HIERARCHY_PRIOR_RESEARCH.md`` §6.2 and STAGED BEFORE THIS RAN.

THE QUESTION
------------
E-V5-2 measured that committing to the tactical class and searching only inside
the top-``q`` anchors it favours costs +0.21 .. +5.82 m, monotone in ``q``.  That
arm conflates TWO things:

  (1) COMMITMENT   -- a good candidate outside the class can no longer win;
  (2) COVERAGE LOSS -- the candidate set drops from 256 to ``q``.

Our fan is a SHARED 256-anchor vocabulary, not ``q`` anchors per mode, so a
``q = 8`` gate leaves 8 trajectories to cover a fan whose 2 s along-track span is
108.7 m per window.  If a RANDOM ``q``-subset hurts as much as the GRAFT's
``q``-subset, the headline is a coverage result and "commitment" is the wrong
word for it.  That is pre-registered falsifier **F2**.

CPU ONLY.  NO POD.  NO GPU.  Reads only artifacts already staged in the repo.

INPUTS (staged)
---------------
``…/2026-07-26-v5-imagination-selection/raw/v5_v4_windows_reduced.pt``
    ``fan_err4  [881, 256]``  4-waypoint error of EVERY candidate
    ``base_rank [881, 256]``  per-window permutation: candidates sorted by the
                              DEPLOYED (grafted) selection score, best first.
                              VERIFIED: ``fan_err4.gather(1, base_rank[:, :1]).mean()``
                              = 0.8563 = ``F_flat`` exactly.
    ``ep [881]``              episode cluster id (40 clusters)
``…/2026-07-26-v5-imagination-selection/raw/v5_hier_windows.pt``
    ``produced|H_graft_q{8,16,32,64}``  per-window ADE of the committed arms.

ESTIMATOR
---------
``taniteval/taniteval/ci.py::paired_episode_cluster_bootstrap`` -- B = 2000,
resampling unit = EPISODE CLUSTER.  ``overlapping_holdout_se`` is never used.

THE RANDOM ARM, stated exactly
------------------------------
``R_rand(q)``: draw a uniform random ``q``-subset ``A`` of the 256 candidates per
window; the pick is the member of ``A`` that ranks best under the DEPLOYED score,
i.e. ``argmin_{c in A} pos[w, c]`` where ``pos`` inverts ``base_rank``.  This is
the exact analogue of ``H_graft(q)`` with the graft's admissible set replaced by
a random one of the same size -- information removed, cardinality held fixed.

Reported as the per-window MEAN over ``S`` seeds (the expected ADE of a random
restriction), which is the estimand the comparison needs; the across-seed spread
of the aggregate is reported separately so the choice cannot hide variance.
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

QS = (8, 16, 32, 64, 128, 256)
SEEDS = 200
B = 2000


def main() -> None:
    red = torch.load(V5 / "v5_v4_windows_reduced.pt", map_location="cpu",
                     weights_only=False)
    hier = torch.load(V5 / "v5_hier_windows.pt", map_location="cpu",
                      weights_only=False)

    fan_err = red["fan_err4"].numpy().astype(np.float64)          # [W, 256]
    base_rank = red["base_rank"].numpy().astype(np.int64)          # [W, 256]
    ep = red["ep"].numpy().astype(np.int64)                        # [W]
    W, N = fan_err.shape

    # invert base_rank -> pos[w, c] = the rank POSITION of candidate c (0 = best)
    pos = np.empty_like(base_rank)
    np.put_along_axis(pos, base_rank, np.arange(N)[None, :].repeat(W, 0), axis=1)

    flat = fan_err[np.arange(W), base_rank[:, 0]]                  # F_flat per window
    out: dict = {
        "_experiment": "E-H1 -- coverage control for E-V5-2 hierarchical commitment",
        "_prereg": "HIERARCHY_PRIOR_RESEARCH.md §6.2 (staged before this ran)",
        "_evidence_class": "MEASURED (ours, CPU, from staged artifacts only)",
        "_estimator": ("paired_episode_cluster_bootstrap (B=%d, unit = episode "
                       "cluster). NEVER overlapping_holdout_se." % B),
        "_host": platform.node(), "_python": platform.python_version(),
        "_torch": torch.__version__, "_numpy": np.__version__,
        "_n_windows": int(W), "_n_candidates": int(N),
        "_n_episode_clusters": int(len(set(ep.tolist()))),
        "_seeds_for_random_arm": SEEDS,
    }

    # ---- self-test: the cached path must reproduce the committed numbers ----
    st = {
        "F_flat_from_base_rank": round(float(flat.mean()), 4),
        "F_flat_committed": 0.8563,
        "F_flat_from_hier_dump": round(float(hier["produced|F_flat"].mean()), 4),
        "oracle_in_fan_from_fan_err": round(float(fan_err.min(axis=1).mean()), 4),
        "oracle_in_fan_committed": 0.2505,
    }
    st["S1_pass"] = bool(abs(st["F_flat_from_base_rank"] - 0.8563) < 1e-3
                         and abs(st["oracle_in_fan_from_fan_err"] - 0.2505) < 1e-3)
    out["selftest"] = st
    print("SELFTEST", json.dumps(st))
    if not st["S1_pass"]:
        out["VERDICT"] = "ABORTED -- self-test failed"
        (Path(__file__).parent / "eh1_coverage_control.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8")
        raise SystemExit("S1 FAILED — aborting before any arm is scored")

    rows = []
    for q in QS:
        # ---- R_rand(q): expected ADE / expected oracle under a random q-subset
        acc_pick = np.zeros(W)
        acc_orac = np.zeros(W)
        seed_means = []
        for s in range(SEEDS):
            rng = np.random.default_rng(1000 + s)
            if q >= N:
                sub = np.arange(N)[None, :].repeat(W, 0)
            else:
                sub = np.argsort(rng.random((W, N)), axis=1)[:, :q]      # [W, q]
            sp = np.take_along_axis(pos, sub, axis=1)                     # positions
            pick = np.take_along_axis(sub, sp.argmin(axis=1)[:, None], axis=1)[:, 0]
            e = fan_err[np.arange(W), pick]
            acc_pick += e
            acc_orac += np.take_along_axis(fan_err, sub, axis=1).min(axis=1)
            seed_means.append(float(e.mean()))
        r_rand = acc_pick / SEEDS
        o_rand = acc_orac / SEEDS

        row: dict = {
            "q": q,
            "R_rand_ade": round(float(r_rand.mean()), 4),
            "R_rand_seed_mean_min": round(float(np.min(seed_means)), 4),
            "R_rand_seed_mean_max": round(float(np.max(seed_means)), 4),
            "R_rand_seed_mean_std": round(float(np.std(seed_means)), 4),
            "O_rand_in_subset": round(float(o_rand.mean()), 4),
        }
        key = f"produced|H_graft_q{q}"
        if key in hier:
            h = hier[key].numpy().astype(np.float64)
            row["H_graft_ade"] = round(float(h.mean()), 4)
            # positive => the GRAFT subset is WORSE than a random subset
            row["paired_H_minus_R"] = paired_episode_cluster_bootstrap(
                h, r_rand, ep, n_boot=B, seed=0)
            row["paired_H_minus_flat"] = paired_episode_cluster_bootstrap(
                h, flat, ep, n_boot=B, seed=0)
        row["paired_R_minus_flat"] = paired_episode_cluster_bootstrap(
            r_rand, flat, ep, n_boot=B, seed=0)
        rows.append(row)
        print(f"q={q:4d}  R_rand={row['R_rand_ade']:.4f}  "
              f"H_graft={row.get('H_graft_ade', float('nan')):.4f}  "
              f"O_rand={row['O_rand_in_subset']:.4f}", flush=True)
    out["rows"] = rows

    # ---- adjudicate against the pre-registered outcomes -------------------
    verdicts = []
    for r in rows:
        if "paired_H_minus_R" not in r or r["q"] >= 256:
            continue
        p = r["paired_H_minus_R"]
        if not p["separated"]:
            verdicts.append("COVERAGE")
        elif p["delta"] > 0:
            verdicts.append("COMMITMENT-HARMFUL")
        else:
            verdicts.append("COMMITMENT-INFORMATIVE")
    out["per_q_verdict"] = dict(zip([r["q"] for r in rows if "paired_H_minus_R" in r
                                     and r["q"] < 256], verdicts))
    out["VERDICT"] = (verdicts[0] if len(set(verdicts)) == 1 and verdicts
                      else "MIXED — see per_q_verdict")
    dst = Path(__file__).parent / "eh1_coverage_control.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("VERDICT:", out["VERDICT"])
    print("wrote", dst)


if __name__ == "__main__":
    main()
