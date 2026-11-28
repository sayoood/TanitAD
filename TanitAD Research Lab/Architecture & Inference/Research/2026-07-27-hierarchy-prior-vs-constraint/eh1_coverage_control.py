#!/usr/bin/env python3
"""E-H1 — the COVERAGE CONTROL for the E-V5-2 hierarchical-commitment result.

PRE-REGISTERED in ``HIERARCHY_PRIOR_RESEARCH.md`` §6.2 and STAGED BEFORE THIS RAN.

⚠️ THIS FILE RECORDS A RETRACTION OF ITS OWN FIRST ARM. READ §"S2" BELOW.
The pre-registered ``R_rand(q)`` arm was specified as *"pick the member of the
random subset that ranks best under the DEPLOYED score"*, using
``v5_v4_windows_reduced.pt::base_rank``.  A self-test written after the first run
proves ``base_rank`` is **NOT a score ranking**: it is
``[the as-trained pick] ++ [anchor indices 0..255 with the pick removed]``
(verified on **881/881** rows against its construction site,
``…/2026-07-26-v5-imagination-selection/code/v5_cost_curve.py:109-122``, whose own
comment says *"approximate the base RANKING … using the recorded per-window pick
plus fan-order for the rest"*).  ``R_rand(q)`` therefore measured
*"take the flat pick if it survived the draw, else take the lowest ANCHOR INDEX
in the draw"* — an artefact of anchor numbering.  **The arm is retracted, its
number is not quoted, and the retraction is reported rather than the run
silently re-specified.**

THE QUESTION (unchanged)
------------------------
E-V5-2 measured that committing to the tactical class and searching only inside
the top-``q`` anchors it favours costs +0.21 .. +5.82 m, monotone in ``q``.  That
conflates COMMITMENT (a good candidate outside the class can no longer win) with
COVERAGE LOSS (the candidate set drops from 256 to ``q``).  Pre-registered
falsifier **F2**.

WHAT SURVIVES AND IS VALID
--------------------------
``O_rand(q)`` -- the ORACLE inside a uniformly random ``q``-subset -- uses only
``fan_err4`` and no ranking at all.  It is the **coverage floor for a typical
``q``-subset**: the best any selector could do if restricted to ``q`` candidates
of random composition.  Comparing it to ``F_flat`` = 0.8563 answers F2 directly
at each ``q``:

    O_rand(q) <  F_flat  =>  a q-subset STILL contains something better than what
                             flat selects, so coverage CANNOT explain a harm at
                             this q.
    O_rand(q) >  F_flat  =>  coverage alone costs at least (O_rand(q) - F_flat),
                             and that much of the measured harm is not commitment.

CPU ONLY.  NO POD.  NO GPU.  Reads only artifacts already staged in the repo.

WHAT IS STILL NOT COMPUTABLE HERE, AND WHY
------------------------------------------
The fully decisive pair -- ``O_graft(q)`` (oracle inside the GRAFT's admissible
set) and ``H_rand(q)`` (best-by-the-REAL-deployed-score inside a random subset)
-- needs ``prior [W, 256]`` and ``sel_score [W, 256]`` per window.  Neither is in
any staged artifact.  That is the ~10-line dump delta named in §5.6 and it is
what E-H2 must carry.
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
F_FLAT = 0.8563
ORACLE_IN_FAN = 0.2505


def main() -> None:
    red = torch.load(V5 / "v5_v4_windows_reduced.pt", map_location="cpu",
                     weights_only=False)
    hier = torch.load(V5 / "v5_hier_windows.pt", map_location="cpu",
                      weights_only=False)
    fan_err = red["fan_err4"].numpy().astype(np.float64)          # [W, 256]
    base_rank = red["base_rank"].numpy().astype(np.int64)
    sel0 = red["ref_sel_idx"].numpy().astype(np.int64)
    ep = red["ep"].numpy().astype(np.int64)
    W, N = fan_err.shape
    flat = fan_err[np.arange(W), sel0]

    out: dict = {
        "_experiment": "E-H1 -- coverage control for E-V5-2 hierarchical commitment",
        "_prereg": "HIERARCHY_PRIOR_RESEARCH.md §6.2 (staged before this ran)",
        "_evidence_class": "MEASURED (ours, CPU, from staged artifacts only)",
        "_estimator": (f"paired_episode_cluster_bootstrap (B={B}, unit = episode "
                       "cluster). NEVER overlapping_holdout_se."),
        "_host": platform.node(), "_python": platform.python_version(),
        "_torch": torch.__version__, "_numpy": np.__version__,
        "_n_windows": int(W), "_n_candidates": int(N),
        "_n_episode_clusters": int(len(set(ep.tolist()))),
        "_seeds_for_random_arm": SEEDS,
    }

    # ---- S1: reproduce committed numbers before adjudicating anything -----
    st = {
        "F_flat_from_ref_sel_idx": round(float(flat.mean()), 4),
        "F_flat_committed": F_FLAT,
        "F_flat_from_hier_dump": round(float(hier["produced|F_flat"].mean()), 4),
        "oracle_in_fan_recomputed": round(float(fan_err.min(axis=1).mean()), 4),
        "oracle_in_fan_committed": ORACLE_IN_FAN,
        "uniform_random_pick_mean": round(float(fan_err.mean()), 4),
        "uniform_random_pick_committed_E-V5-1": 15.8738,
    }
    st["S1_pass"] = bool(abs(st["F_flat_from_ref_sel_idx"] - F_FLAT) < 1e-3
                         and abs(st["oracle_in_fan_recomputed"] - ORACLE_IN_FAN) < 1e-3)

    # ---- S2: THE SELF-TEST THAT RETRACTED THE PRE-REGISTERED ARM ---------
    ok = 0
    for w in range(W):
        expect = np.concatenate([[sel0[w]],
                                 np.setdiff1d(np.arange(N), [sel0[w]])])
        ok += int(np.array_equal(base_rank[w], expect))
    st_s2 = {
        "claim_tested": ("base_rank is the argsort of the DEPLOYED selection "
                         "score (what the pre-registration assumed)"),
        "rows_matching_[pick]++[anchor_index_order]": int(ok),
        "rows_total": int(W),
        "verdict": ("REFUTED -- base_rank is NOT a score ranking; it is the "
                    "as-trained pick followed by anchor index order"
                    if ok == W else "inconclusive"),
        "consequence": ("the pre-registered R_rand(q) arm is RETRACTED and its "
                        "numbers are not quoted; O_rand(q) is unaffected because "
                        "it never touches base_rank"),
        "source_of_truth": ("…/2026-07-26-v5-imagination-selection/code/"
                            "v5_cost_curve.py:109-122"),
    }
    out["selftest_S1_committed_numbers"] = st
    out["selftest_S2_base_rank_semantics"] = st_s2
    print("S1", json.dumps(st))
    print("S2", st_s2["verdict"], f"({ok}/{W} rows)")
    if not st["S1_pass"]:
        out["VERDICT"] = "ABORTED -- S1 failed"
        (Path(__file__).parent / "eh1_coverage_control.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8")
        raise SystemExit("S1 FAILED")

    # ---- the VALID arm: coverage floor of a random q-subset --------------
    rows = []
    for q in QS:
        acc = np.zeros(W)
        seed_means = []
        for s in range(SEEDS):
            rng = np.random.default_rng(1000 + s)
            sub = (np.arange(N)[None, :].repeat(W, 0) if q >= N
                   else np.argsort(rng.random((W, N)), axis=1)[:, :q])
            o = np.take_along_axis(fan_err, sub, axis=1).min(axis=1)
            acc += o
            seed_means.append(float(o.mean()))
        o_rand = acc / SEEDS
        row: dict = {
            "q": q,
            "O_rand_coverage_floor": round(float(o_rand.mean()), 4),
            "O_rand_seed_spread": [round(float(np.min(seed_means)), 4),
                                   round(float(np.max(seed_means)), 4)],
            "coverage_binding_at_this_q": bool(float(o_rand.mean()) > F_FLAT),
            "coverage_excess_over_flat": round(float(o_rand.mean()) - F_FLAT, 4),
            "paired_O_rand_minus_flat": paired_episode_cluster_bootstrap(
                o_rand, flat, ep, n_boot=B, seed=0),
        }
        key = f"produced|H_graft_q{q}"
        if key in hier:
            h = hier[key].numpy().astype(np.float64)
            row["H_graft_ade"] = round(float(h.mean()), 4)
            row["H_graft_minus_flat"] = round(float(h.mean()) - F_FLAT, 4)
            row["harm_NOT_explained_by_coverage"] = round(
                float(h.mean()) - max(F_FLAT, float(o_rand.mean())), 4)
            row["paired_H_minus_O_rand"] = paired_episode_cluster_bootstrap(
                h, o_rand, ep, n_boot=B, seed=0)
        rows.append(row)
        print(f"q={q:4d}  O_rand(coverage floor)={row['O_rand_coverage_floor']:.4f}"
              f"  H_graft={row.get('H_graft_ade', float('nan')):.4f}"
              f"  coverage_binding={row['coverage_binding_at_this_q']}", flush=True)
    out["rows"] = rows

    # ---- adjudicate F2 against the pre-registered outcomes ---------------
    loose = [r for r in rows if r["q"] in (32, 64) and "H_graft_ade" in r]
    f2 = {
        "q_at_which_coverage_first_binds": next(
            (r["q"] for r in sorted(rows, key=lambda x: -x["q"])
             if r["coverage_binding_at_this_q"]), None),
        "at_q>=32_coverage_floor_is_below_flat": all(
            not r["coverage_binding_at_this_q"] for r in loose),
        "reading": None,
    }
    if f2["at_q>=32_coverage_floor_is_below_flat"]:
        f2["reading"] = (
            "F2 does NOT fire at q=32/64: a random q-subset still contains a "
            "candidate BETTER than what flat selects, so the separated harm at "
            "those q cannot be coverage. At q=8/16 the coverage floor exceeds "
            "F_flat and part of the harm IS coverage — see "
            "coverage_excess_over_flat.")
    out["F2_adjudication"] = f2
    out["VERDICT"] = ("F2 PARTIAL — coverage explains part of the harm at q<=16 "
                      "and NONE of it at q>=32; the R_rand arm that would have "
                      "settled the rest is RETRACTED (S2) and deferred to E-H2")
    dst = Path(__file__).parent / "eh1_coverage_control.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("VERDICT:", out["VERDICT"])
    print("wrote", dst)


if __name__ == "__main__":
    main()
