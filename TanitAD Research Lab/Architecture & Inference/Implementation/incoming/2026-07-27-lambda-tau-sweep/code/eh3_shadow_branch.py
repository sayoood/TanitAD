#!/usr/bin/env python3
"""E-H3 — GoalFlow's SHADOW BRANCH on our cached fan, measured, not asserted.

THE CLAIM THIS FILE TESTS, AND THE ONE IT REPLACES
--------------------------------------------------
The program currently carries "our produced goal is worse than no goal
(-0.0943 m, separated), so turning it off is free".  That is true and it is
also the WEAKEST available fix, because it throws away the windows on which the
goal is right.  GoalFlow (Xing et al., CVPR 2025) runs a second, UNCONDITIONED
branch and falls back to it *only when the goal looks unreliable*:

    "if the shadow trajectory deviates significantly from the main trajectory,
     we treat the goal point as unreliable and use the shadow as the output."

Their ladder is 85.6 (none) -> 90.3 (predicted) -> 92.1 (oracle) PDMS: their
predicted goal recovers 73% of the oracle headroom.  Ours recovers -147%.  So
the shadow branch is the published fix for exactly our failure.

WHAT IS MEASURED HERE, AND WHAT IS NOT
--------------------------------------
MEASURED, from staged artifacts, CPU only:
  * the ORACLE-SHADOW BOUND -- per-window min(produced, neutral).  Not
    deployable; it is the CEILING every deployable rule is scored against, and
    it is the number that says whether a shadow rule can help AT ALL.
  * four DEPLOYABLE, NON-ORACLE fallback rules, each a threshold on a signal
    available at inference with no ground truth, each evaluated
    LEAVE-ONE-EPISODE-OUT so the reported number is not the in-sample ceiling
    (the failure Bar A's 0.4907 exists to warn about).

NOT MEASURED, and stated rather than fudged: GoalFlow's OWN rule is a distance
between the conditioned and unconditioned TRAJECTORIES.  No staged artifact
holds the neutral branch's trajectory -- only its per-window error.  So the
literal rule is not computable from the repo today; ``eh2_build_cache.py``
dumps ``pick_traj`` for both goal modes, which makes it a one-line addition
here.  The rules below are the best NON-ORACLE reliability signals that DO
exist, and they are labelled as such.

PRE-REGISTERED BAR (HIERARCHY_PRIOR_RESEARCH.md section 6.4):
  CONFIRM  if any deployable rule beats the neutral 0.7620 on the deployable
           surface, paired and separated.
  REFUTE   if the best is "always neutral" with no rule improving on it => "turn
           the goal off and fix the producer" is the whole answer.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[6]
sys.path.insert(0, str(_REPO / "taniteval"))
from taniteval.ci import (episode_cluster_bootstrap,               # noqa: E402
                          paired_episode_cluster_bootstrap)

_V5 = (_REPO / "TanitAD Research Hub" / "Architecture & Inference" /
       "Implementation" / "incoming" / "2026-07-26-v5-imagination-selection")
B = 2000
GOALFLOW = {"none": 85.6, "predicted": 90.3, "oracle": 92.1, "_metric": "PDMS (up)"}


# --------------------------------------------------------------------------- #
def load_staged():
    H = torch.load(_V5 / "raw" / "v5_hier_windows.pt", map_location="cpu",
                   weights_only=False)
    R = torch.load(_V5 / "raw" / "v5_v4_windows_reduced.pt", map_location="cpu",
                   weights_only=False)
    prod = H["produced|F_flat"].double().numpy()
    neut = H["neutral|F_flat"].double().numpy()
    eid = np.asarray([str(int(e)) for e in R["ep"]])
    pick = R["picks"]["A0_as_trained"]
    # NON-ORACLE reliability signals: every one is computable at inference from
    # the model's own outputs. `canary_err` and `fan_err4` are DELIBERATELY
    # excluded -- they are scored against ground truth and would make any rule
    # an oracle wearing a deployable costume.
    sig = {
        "v0_ego_speed": R["v0"].double().numpy(),
        "imag_cost_of_the_deployed_pick":
            R["costs"]["A1_imag_consistency"].gather(
                1, pick[:, None]).squeeze(1).double().numpy(),
        "ctrv_inconsistency_of_the_deployed_pick":
            R["costs"]["C1_ctrv_consistency"].gather(
                1, pick[:, None]).squeeze(1).double().numpy(),
        "imag_cost_spread_over_the_fan":
            R["costs"]["A1_imag_consistency"].double().std(dim=1).numpy(),
    }
    return prod, neut, eid, sig


def apply_rule(prod, neut, s, thr, direction):
    """direction=+1: fall back to NEUTRAL where s > thr; -1: where s < thr."""
    use_neut = (s > thr) if direction > 0 else (s < thr)
    return np.where(use_neut, neut, prod), use_neut


def best_threshold(prod, neut, s, grid):
    """In-fold argmin over (threshold, direction). Returns (thr, dir, value)."""
    best = (np.inf, None, None)
    for d in (+1, -1):
        for t in grid:
            v = float(apply_rule(prod, neut, s, t, d)[0].mean())
            if v < best[0]:
                best = (v, t, d)
    return best[1], best[2], best[0]


def loeo(prod, neut, s, eid, grid):
    """Leave-one-EPISODE-out. The threshold is chosen WITHOUT the window it is
    then scored on, so the number is deployable rather than an in-sample ceiling."""
    out = np.empty_like(prod)
    chosen = []
    for e in np.unique(eid):
        te = eid == e
        thr, d, _ = best_threshold(prod[~te], neut[~te], s[~te], grid)
        out[te] = apply_rule(prod[te], neut[te], s[te], thr, d)[0]
        chosen.append({"held_out_episode_cluster": int(np.flatnonzero(te)[0]),
                       "thr": round(float(thr), 4), "direction": int(d)})
    return out, chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_HERE.parent.parent / "raw"
                                         / "eh3_shadow_branch.json"))
    ap.add_argument("--boot", type=int, default=B)
    ap.add_argument("--n-thr", type=int, default=41)
    a = ap.parse_args()
    t0 = time.time()
    prod, neut, eid, sig = load_staged()

    Rj: dict = {
        "_experiment": "E-H3 — GoalFlow's shadow branch (an unconditioned "
                       "fallback) vs 'turn the produced goal off'",
        "_prereg": "HIERARCHY_PRIOR_RESEARCH.md section 6.4 / section 5.4 (D2)",
        "_evidence_class": "MEASURED (ours)",
        "_estimator": f"paired_episode_cluster_bootstrap (B={a.boot}, unit = "
                      "episode cluster). NEVER overlapping_holdout_se.",
        "_surface": "PRODUCED (deployable). The oracle-goal arm is a bound only.",
        "_host": platform.node(), "_python": platform.python_version(),
        "_torch": torch.__version__, "_device": "cpu — no pod, no GPU",
        "_goalflow_reference_ladder": GOALFLOW,
        "_n_windows": int(prod.size), "_n_episodes": int(len(np.unique(eid))),
    }

    # ---- the three fixed rungs ---------------------------------------------
    Rj["rungs"] = {
        "R_produced_always (shipped)":
            episode_cluster_bootstrap(prod, eid, n_boot=a.boot),
        "R_neutral_always (turn the goal off)":
            episode_cluster_bootstrap(neut, eid, n_boot=a.boot),
        "neutral_minus_produced_paired":
            paired_episode_cluster_bootstrap(neut, prod, eid, n_boot=a.boot),
    }

    # ---- the ORACLE-SHADOW BOUND: the ceiling on ANY fallback rule -----------
    orc = np.minimum(prod, neut)
    Rj["oracle_shadow_bound"] = {
        "ade_0_2s": episode_cluster_bootstrap(orc, eid, n_boot=a.boot),
        "vs_produced_paired": paired_episode_cluster_bootstrap(
            orc, prod, eid, n_boot=a.boot),
        "vs_neutral_paired": paired_episode_cluster_bootstrap(
            orc, neut, eid, n_boot=a.boot),
        "frac_windows_where_neutral_wins": round(float((neut < prod).mean()), 4),
        "_read": "per-window min(produced, neutral). NOT DEPLOYABLE — it needs "
                 "the answer. It is the ceiling every deployable rule below is "
                 "measured against: if this is not far below the better rung, "
                 "no fallback rule can help and the finding is 'fix the "
                 "producer', not 'add a shadow branch'.",
    }

    # ---- the deployable, NON-ORACLE rules -----------------------------------
    rules = {}
    for name, s in sig.items():
        grid = np.unique(np.quantile(s, np.linspace(0.0, 1.0, a.n_thr)))
        thr, d, insample = best_threshold(prod, neut, s, grid)
        cv, chosen = loeo(prod, neut, s, eid, grid)
        rules[name] = {
            "in_sample_ceiling_ade": round(insample, 4),
            "in_sample_threshold": round(float(thr), 4),
            "in_sample_direction": ("fallback where signal > thr" if d > 0
                                    else "fallback where signal < thr"),
            "LOEO_ade": episode_cluster_bootstrap(cv, eid, n_boot=a.boot),
            "LOEO_vs_produced_paired": paired_episode_cluster_bootstrap(
                cv, prod, eid, n_boot=a.boot),
            "LOEO_vs_neutral_paired": paired_episode_cluster_bootstrap(
                cv, neut, eid, n_boot=a.boot),
            "LOEO_frac_windows_falling_back": round(
                float((cv == neut).mean()), 4),
            "_threshold_stability": {
                "n_folds": len(chosen),
                "n_distinct_thresholds": len({c["thr"] for c in chosen}),
                "n_folds_choosing_each_direction": {
                    "+1": sum(1 for c in chosen if c["direction"] > 0),
                    "-1": sum(1 for c in chosen if c["direction"] < 0)},
            },
        }
    Rj["deployable_rules"] = rules

    # ---- the pre-registered verdict -----------------------------------------
    beat = [k for k, v in rules.items()
            if v["LOEO_vs_neutral_paired"]["separated"]
            and v["LOEO_vs_neutral_paired"]["delta"] < 0]
    best = min(rules, key=lambda k: rules[k]["LOEO_ade"]["mean"])
    Rj["verdict"] = {
        "CONFIRM_or_REFUTE": "CONFIRM" if beat else "REFUTE",
        "rules_separated_better_than_neutral": beat,
        "best_deployable_rule": best,
        "best_LOEO_ade": rules[best]["LOEO_ade"]["mean"],
        "neutral_ade": Rj["rungs"]["R_neutral_always (turn the goal off)"]["mean"],
        "oracle_shadow_bound_ade": Rj["oracle_shadow_bound"]["ade_0_2s"]["mean"],
        "_prereg_bar": "CONFIRM if any deployable rule beats the neutral 0.7620, "
                       "paired and separated; REFUTE if the argmin is 'always "
                       "neutral' with no rule improving on it.",
        "_not_computable_here": "GoalFlow's OWN rule — ||traj_conditioned - "
                                "traj_unconditioned|| > d* — needs the NEUTRAL "
                                "branch's trajectory, which no staged artifact "
                                "holds. eh2_build_cache.py dumps `pick_traj` for "
                                "both goal modes; with that cache this becomes a "
                                "fifth signal in `load_staged` and nothing else "
                                "changes.",
    }
    Rj["_wallclock_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(Rj, indent=2), encoding="utf-8")
    print(json.dumps(Rj["verdict"], indent=2))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
