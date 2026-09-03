#!/usr/bin/env python
"""⭐ THE REFUTATION ARM — 0 GPU, banked JSON only.

THE QUESTION IT SETTLES BEFORE ANY TRAINING ARM IS SPENT. The ep2 checkpoint
ALREADY asks for a turn on 25 of 140 windows, and `plan()` already injects that
turn's canonical control as a named seed. Those 25 windows are a ready-made
natural experiment for the COST, with the decoder held fixed:

    On a window where the goal IS curved, does the cost prefer the turn?

`cost_surface_probe.py` banked, per window, the FULL decomposition of every
named seed's cost into its live terms (`named_<conv>_<seed>_c_{goal,jerk,kappa,
vend,total}`), so the answer needs no forward pass. This tool asks three things
the R10 summary's medians could not:

  1. **Which term actually charges the turn.** The median jerk over the 25 was
     0.0, so the R10 summary read the curvature penalty as the charge — but the
     mean and the tail are dominated by `0.02·jerk²` from the CANONICAL control's
     own longitudinal step (`canonical_controls` sets a first-order approach to
     the token's target speed; the step at k=0 is a jerk). Per window, not median.
  2. **Whether the goal term prefers the turn AT ALL** (sign and magnitude of
     `cv_c_goal - turn_c_goal`). ⛔ If it does NOT, the whole decoder-then-cost
     line is REFUTED: the tactical field cannot tell a turn from a straight
     line, and no decoder or metric repair can help.
  3. **The chord counterfactual (BACKLOG R29).** `chord = sqrt(2*(1-cos))` is a
     strictly monotone transform of the goal term, so it cannot change the goal
     term's own RANKING — but the cost is a SUM, so it silently re-weights goal
     against penalty. This computes the resulting leverage change and the
     penalty/jerk weight that WOULD make each turn win, in both metrics.

⚠️ The banked `named_*` values are float32. Where a difference is smaller than a
few ULPs of the values it is taken between, the tool says so rather than
reporting it — `ulp_of` and `resolvable` are printed per window.

TIER: T0. EVIDENCE CLASS: MEASURED (re-read of banked cost-surface rows).
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

W_JERK = 0.02          #: `refa_v1.py:1982`
W_KAPPA = 0.05         #: `refa_v1.py:1983`


def ulp32(x: float) -> float:
    x = abs(float(x))
    if x == 0.0:
        return float(np.spacing(np.float32(0.0)))
    return float(np.spacing(np.float32(x)))


def analyse(path: str, conv: str, tag: str) -> dict:
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    rows = j["rows"]
    turn = [r for r in rows if float(r.get("goal_kappa_max") or 0.0) > 0.0]
    out: dict = {"arm": tag, "convention": conv, "n_windows": len(rows),
                 "n_goal_carries_curvature": len(turn),
                 "cost_weights": {"goal": 1.0, "jerk": W_JERK, "kappa": W_KAPPA},
                 "windows": []}
    if not turn:
        out["note"] = ("the decoder never asks for a turn on this checkpoint — "
                       "there is no window on which the cost can be asked the "
                       "question")
        return out

    adv, charge_k, charge_j, wins, wins_chord = [], [], [], [], []
    need_k, need_j, need_all = [], [], []
    chord_adv, chord_need_all = [], []
    resolvable = []
    for r in turn:
        g = {k: float(r[f"named_{conv}_goal_canonical_c_{k}"])
             for k in ("goal", "jerk", "kappa", "vend", "total")}
        c = {k: float(r[f"named_{conv}_cv_c_{k}"])
             for k in ("goal", "jerk", "kappa", "vend", "total")}
        a = c["goal"] - g["goal"]                 # what the turn BUYS
        ch_k, ch_j = g["kappa"] - c["kappa"], g["jerk"] - c["jerk"]
        ch = ch_k + ch_j + (g["vend"] - c["vend"])
        # the chord counterfactual on the SAME two goal values
        gc, cc = math.sqrt(2.0 * max(g["goal"], 0.0)), math.sqrt(2.0 * max(c["goal"], 0.0))
        a_ch = cc - gc
        u = max(ulp32(g["goal"]), ulp32(c["goal"]))
        adv.append(a)
        chord_adv.append(a_ch)
        charge_k.append(ch_k)
        charge_j.append(ch_j)
        wins.append(1.0 if a > ch else 0.0)
        wins_chord.append(1.0 if a_ch > ch else 0.0)
        resolvable.append(1.0 if abs(a) > 2 * u else 0.0)
        # the weight that WOULD tip it, holding everything else
        need_k.append((a - ch_j) / (g["kappa"] / W_KAPPA)
                      if g["kappa"] > 0 else float("nan"))
        need_j.append((a - ch_k) / (g["jerk"] / W_JERK)
                      if g["jerk"] > 0 else float("nan"))
        scale = (g["kappa"] + g["jerk"]) / (W_KAPPA + W_JERK) if (g["kappa"] + g["jerk"]) > 0 else float("nan")
        need_all.append(a / (g["kappa"] / W_KAPPA + g["jerk"] / W_JERK)
                        if (g["kappa"] + g["jerk"]) > 0 else float("nan"))
        chord_need_all.append(a_ch / (g["kappa"] / W_KAPPA + g["jerk"] / W_JERK)
                              if (g["kappa"] + g["jerk"]) > 0 else float("nan"))
        out["windows"].append({
            "ep": r.get("ep"), "t": r.get("t"),
            "goal_kappa_max": r.get("goal_kappa_max"),
            "gt_turn_deg": r.get("gt_turn_deg"),
            "turn_c_goal": g["goal"], "cv_c_goal": c["goal"],
            "goal_advantage": a, "goal_advantage_in_ulps": a / u if u else None,
            "resolvable_in_fp32": bool(abs(a) > 2 * u),
            "charge_kappa": ch_k, "charge_jerk": ch_j, "charge_total": ch,
            "turn_total": g["total"], "cv_total": c["total"],
            "turn_wins": bool(a > ch),
            "chord_turn": gc, "chord_cv": cc, "chord_advantage": a_ch,
            "chord_turn_wins_same_weights": bool(a_ch > ch),
            "chord_leverage_gain_x": (a_ch / a) if a not in (0.0,) else None,
        })

    def _q(v):
        v = np.asarray([x for x in v if np.isfinite(x)], float)
        if not v.size:
            return None
        return {"n": int(v.size), "median": float(np.median(v)),
                "mean": float(v.mean()), "min": float(v.min()),
                "max": float(v.max()),
                "p25": float(np.percentile(v, 25)),
                "p75": float(np.percentile(v, 75))}

    out["summary"] = {
        "goal_advantage_cv_minus_turn": _q(adv),
        "n_goal_advantage_positive": int(sum(1 for x in adv if x > 0)),
        "n_goal_advantage_resolvable_fp32": int(sum(resolvable)),
        "charge_kappa": _q(charge_k),
        "charge_jerk": _q(charge_j),
        "n_jerk_charge_exceeds_kappa_charge":
            int(sum(1 for a_, b_ in zip(charge_j, charge_k) if a_ > b_)),
        "n_turn_wins_as_shipped": int(sum(wins)),
        "n_turn_wins_under_chord_same_weights": int(sum(wins_chord)),
        "chord_advantage": _q(chord_adv),
        "chord_leverage_gain_x": _q([w["chord_leverage_gain_x"]
                                     for w in out["windows"]
                                     if w["chord_leverage_gain_x"] is not None]),
        "penalty_scale_that_would_tip_it_1_cos": _q(need_all),
        "penalty_scale_that_would_tip_it_chord": _q(chord_need_all),
        "reading": ("`penalty_scale_that_would_tip_it_*` is the multiplier on "
                    "BOTH explicit weights (0.05 kappa^2, 0.02 jerk^2) at which "
                    "the turn's total cost equals cv's, holding the goal term "
                    "fixed. < 1 means the shipped weights are too heavy by that "
                    "factor; the median is the number to quote."),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", action="append", required=True,
                    help="NAME=<cost_surface_*.json>")
    ap.add_argument("--convention", default="A",
                    help="A|B — the wheelbase/units convention banked by R10")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"tool": "turn_decomposition.py", "tier": "T0",
           "evidence_class": "MEASURED (re-read of banked cost-surface rows)",
           "arms": {}}
    for x in a.cost:
        nm, p = x.split("=", 1)
        res["arms"][nm] = analyse(p, a.convention, nm)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for nm, b in res["arms"].items():
        print(f"=== {nm}: {b['n_goal_carries_curvature']}/{b['n_windows']} "
              f"windows with a CURVED goal")
        if "summary" in b:
            s = b["summary"]
            print(f"  goal advantage (cv - turn): {json.dumps(s['goal_advantage_cv_minus_turn'])}")
            print(f"  positive on {s['n_goal_advantage_positive']}, "
                  f"fp32-resolvable on {s['n_goal_advantage_resolvable_fp32']}")
            print(f"  charge kappa: {json.dumps(s['charge_kappa'])}")
            print(f"  charge jerk : {json.dumps(s['charge_jerk'])}")
            print(f"  jerk > kappa on {s['n_jerk_charge_exceeds_kappa_charge']}")
            print(f"  turn wins as shipped {s['n_turn_wins_as_shipped']}, "
                  f"under chord {s['n_turn_wins_under_chord_same_weights']}")
            print(f"  chord leverage gain x: {json.dumps(s['chord_leverage_gain_x'])}")
            print(f"  penalty scale to tip (1-cos): {json.dumps(s['penalty_scale_that_would_tip_it_1_cos'])}")
            print(f"  penalty scale to tip (chord): {json.dumps(s['penalty_scale_that_would_tip_it_chord'])}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
