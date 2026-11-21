#!/usr/bin/env python3
"""G-REWARD: does the RL reward prefer the HUMAN to the trivial constant-velocity path?

`D-REFCV5-PLAN-6` makes this a PRECONDITION, not a milestone:

    hold-v0 must beat the human on <= 30 % of lead windows ... A reward that fails
    G-REWARD may not train anything

MEASURED today under `DEFAULT_WEIGHTS`: **72.5 %** (`D-RL-REWARD-FLOOR-2`, 6,089 lead
windows / 73 episodes). So no RL arm on refcv4b or refcv5 may launch until the reward is
repaired. `D-REFCV5-PLAN-6` names the repair -- *"fixed by moving feasibility and comfort
out of the reward into calibrated vetoes"* -- but the repair had never been SCORED.

This scores it, at ZERO GPU, by RE-WEIGHTING the banked per-window component table
(`humanflag_fit120.json`, 6,089 rows x {human, hold_v0, frozen} x {static, track}, every
component stored separately). No model forward, no corpus, no lead join -- the expensive
part was already paid on 2026-09-05.

WHY RE-WEIGHTING IS EXACT AND NOT AN APPROXIMATION. `RewardSpec.__call__` is a LINEAR
combination of per-component values (`rewards.py:645-652`): `sum_n w_n * v_n`. The banked
rows carry every `v_n`. Changing `w` therefore reproduces the composed score EXACTLY --
this is arithmetic on banked measurements, not a re-simulation. Dropping a component to a
VETO is the `w_n = 0` case plus a gate, and the gate is reported separately.

THE CONTROLS ARE THE POINT (CLAUDE.md 2026-08-22: four probe failures in one afternoon,
each caught only by a control that had to read a known value):

  * `CHANNEL`  -- re-weighting under DEFAULT_WEIGHTS must reproduce the BANKED
                 `hold_v0_scores_at_least_human` (0.7249137789456397, track) and the
                 banked composed means to <= 1e-9. If it does not, the arithmetic is
                 wrong and every other number here is void. This is the same-breath
                 control M50 demands: it shares the channel AND the pattern.
  * `frozen`   -- the do-nothing path must stay far below both under every spec; a spec
                 that lets `frozen` win has replaced one degenerate with another.
  * per-component ATTRIBUTION -- the mean of `w_n * (hold_v0_n - human_n)` per component,
                 which must sum to the composed-mean gap. A repair that moves the rate
                 without a term that explains it is a coincidence, not a mechanism.

Tier: T0 instrument probe on banked NON-PARITY RL-fit windows. Evidence class:
MEASURED (ours), arithmetic over `.../2026-09-05-refc-rl-readiness` banked rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

#: The banked default -- `tanitad.rl.rewards.DEFAULT_WEIGHTS`, reproduced here so the
#: gate runs with no torch import. Asserted equal to the banked run's own
#: `reward_weights` block before anything is scored.
DEFAULT_WEIGHTS = {
    "progress": 0.30,
    "collision": 1.00,
    "headway": 0.30,
    "feasibility": 0.50,
    "comfort": 0.20,
}

#: The banked value this gate must reproduce, from
#: `humanflag_fit120.summary.json:summary.track.hold_v0_scores_at_least_human`.
BANKED_TRACK_RATE = 0.7249137789456397
BANKED_STATIC_RATE = 0.7350960748891444

#: `D-REFCV5-PLAN-6`'s precondition.
G_REWARD_CEILING = 0.30

CANDIDATES: dict[str, dict[str, float]] = {
    # the status quo -- FAILS at 0.725
    "default": dict(DEFAULT_WEIGHTS),
    # `D-REFCV5-PLAN-6`'s named repair: feasibility and comfort leave the ranking sum
    # and become vetoes. Nothing else moves -- ONE VARIABLE against `default`.
    "veto_feas_comfort": {"progress": 0.30, "collision": 1.00, "headway": 0.30},
    # the two halves of that repair, so the effect is attributable to a term and not
    # to "we removed two things".
    "veto_comfort_only": {"progress": 0.30, "collision": 1.00, "headway": 0.30,
                          "feasibility": 0.50},
    "veto_feas_only": {"progress": 0.30, "collision": 1.00, "headway": 0.30,
                       "comfort": 0.20},
    # headway is the OTHER term that pays the trivial path (see the attribution table):
    # a constant-velocity follower sits at a steady time gap and scores near the peak.
    "veto_fc_no_headway": {"progress": 0.30, "collision": 1.00},
    # progress alone -- the DELIBERATE-REGRESSION arm (`HACKABLE_WEIGHTS`). Must NOT be
    # adopted; present so the panel contains a spec whose failure mode is known.
    "hackable_progress_only": {"progress": 1.0},
}


def load_rows(path: str) -> tuple[list[dict], dict]:
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    rows = d.get("per_window")
    if not rows:
        raise SystemExit(f"[greward] no per_window rows in {path}")
    return rows, d


def composed(rec: dict, weights: dict[str, float]) -> float:
    """Exactly `RewardSpec.__call__`: the weighted sum of the banked components."""
    return sum(w * rec[name] for name, w in weights.items())


def score_spec(rows: list[dict], weights: dict[str, float], mode: str) -> dict:
    """Rate at which hold-v0 (and frozen) score >= the human, plus attribution."""
    hu, ho, fr, eids = [], [], [], []
    for r in rows:
        hu.append(composed(r[f"human_{mode}"], weights))
        ho.append(composed(r[f"hold_v0_{mode}"], weights))
        fr.append(composed(r[f"frozen_{mode}"], weights))
        eids.append(r["eid"])
    hu, ho, fr = np.array(hu), np.array(ho), np.array(fr)
    ge = (ho >= hu)
    # per-component attribution of the mean (hold_v0 - human) gap
    attrib = {}
    for name, w in weights.items():
        d = np.array([w * (r[f"hold_v0_{mode}"][name] - r[f"human_{mode}"][name])
                      for r in rows])
        attrib[name] = float(d.mean())
    return {
        "weights": dict(weights),
        "n": int(len(rows)),
        "hold_v0_scores_at_least_human": float(ge.mean()),
        "hold_v0_ge_human_ci": episode_cluster_ci(ge.astype(float), np.array(eids)),
        "frozen_scores_at_least_human": float((fr >= hu).mean()),
        "composed_mean": {"human": float(hu.mean()), "hold_v0": float(ho.mean()),
                          "frozen": float(fr.mean())},
        "mean_gap_hold_minus_human": float((ho - hu).mean()),
        "attribution_mean_gap_by_component": attrib,
        "attribution_sum": float(sum(attrib.values())),
        "PASSES_G_REWARD": bool(ge.mean() <= G_REWARD_CEILING),
    }


def episode_cluster_ci(vals: np.ndarray, eids: np.ndarray, *, reps: int = 2000,
                       seed: int = 11) -> list[float]:
    """Episode-cluster bootstrap over the rate -- windows within an episode are not
    independent, so the resampling unit is the EPISODE (`taniteval/ci.py`'s rule)."""
    uniq = np.unique(eids)
    by = {int(e): vals[eids == e] for e in uniq}
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for i in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        out[i] = np.concatenate([by[int(e)] for e in pick]).mean()
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="banked humanflag per-window JSON")
    ap.add_argument("--mode", default="track", choices=("track", "static"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    rows, meta = load_rows(a.rows)
    print(f"[greward] {len(rows)} banked lead windows, "
          f"{len(set(r['eid'] for r in rows))} episodes, mode={a.mode}")

    # ---- CONTROL 1: the banked run's own weights must be our DEFAULT_WEIGHTS -------
    banked_w = meta.get("reward_weights", {})
    if banked_w != DEFAULT_WEIGHTS:
        raise SystemExit(f"[greward] REFUSED: banked weights {banked_w} != "
                         f"DEFAULT_WEIGHTS {DEFAULT_WEIGHTS}; the rows were produced "
                         f"under a different reward and re-weighting is not exact")

    # ---- CONTROL 2: re-weighting must REPRODUCE the banked rate -------------------
    ctrl = score_spec(rows, DEFAULT_WEIGHTS, a.mode)
    banked = BANKED_TRACK_RATE if a.mode == "track" else BANKED_STATIC_RATE
    err = abs(ctrl["hold_v0_scores_at_least_human"] - banked)
    ok = err <= 1e-9
    print(f"[greward] CHANNEL CONTROL: recomputed {ctrl['hold_v0_scores_at_least_human']:.16f}"
          f" vs banked {banked:.16f}  err={err:.3e}  "
          f"{'OK' if ok else 'MISMATCH -- every number below is VOID'}")
    if not ok:
        raise SystemExit("[greward] REFUSED: channel control failed")

    panel = {}
    for name, w in CANDIDATES.items():
        panel[name] = score_spec(rows, w, a.mode)

    print()
    print(f"{'spec':<24} {'hold_v0>=human':>15} {'CI':>22} {'frozen>=human':>14} "
          f"{'gap':>9}  G-REWARD")
    print("-" * 105)
    for name, s in panel.items():
        lo, hi = s["hold_v0_ge_human_ci"]
        print(f"{name:<24} {s['hold_v0_scores_at_least_human']:>15.4f} "
              f"[{lo:>8.4f},{hi:>8.4f}] {s['frozen_scores_at_least_human']:>14.4f} "
              f"{s['mean_gap_hold_minus_human']:>9.4f}  "
              f"{'PASS' if s['PASSES_G_REWARD'] else 'FAIL'}")

    print()
    print("Per-component attribution of the mean (hold_v0 - human) gap, DEFAULT spec:")
    for k, v in sorted(panel["default"]["attribution_mean_gap_by_component"].items(),
                       key=lambda kv: -kv[1]):
        print(f"   {k:<14} {v:+.6f}")
    print(f"   {'SUM':<14} {panel['default']['attribution_sum']:+.6f}  "
          f"(composed-mean gap {panel['default']['mean_gap_hold_minus_human']:+.6f})")

    out = {
        "_what": "G-REWARD precondition: does the reward prefer the human to hold-v0?",
        "_evidence_class": "MEASURED (ours) -- arithmetic re-weighting of banked rows",
        "_tier": "T0 instrument probe, NON-PARITY RL-fit windows",
        "_rule": "D-REFCV5-PLAN-6: hold-v0 must beat the human on <= 30 % of lead "
                 "windows; a reward that fails G-REWARD may not train anything",
        "source_rows": os.path.abspath(a.rows),
        "n_windows": len(rows), "n_episodes": len(set(r["eid"] for r in rows)),
        "lead_mode": a.mode,
        "g_reward_ceiling": G_REWARD_CEILING,
        "channel_control": {"recomputed": ctrl["hold_v0_scores_at_least_human"],
                            "banked": banked, "abs_err": err, "PASS": ok},
        "panel": panel,
    }
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"\n[greward] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
