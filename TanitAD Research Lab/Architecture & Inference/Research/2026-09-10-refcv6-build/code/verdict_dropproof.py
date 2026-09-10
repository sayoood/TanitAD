#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv6 -- ⛔ PROVE THE NON-REGRESSION CLAUSES CANNOT BE DROPPED.

    python verdict_dropproof.py --json raw/verdict_dropproof.json

`verdict_refcv6.py` CLAIMS that a missing lateral or strategic row blocks SUCCESS.
That claim is worth nothing until a panel that omits each one has been fed to it
and observed NOT to read SUCCESS.

⭐ The reference panel here is a DELIBERATELY GENEROUS one: it clears the ADE bar
against both baselines with room to spare, so every mutant below fails for exactly
one reason -- the dropped clause -- and not because the arm was weak. If the
baseline did not read SUCCESS the mutants would prove nothing.

⛔ A mutant that SURVIVES (still reads SUCCESS) is a hole in the bar and is
reported as such, never quietly dropped.
"""

from __future__ import annotations

import argparse
import copy
import json
import os

from verdict_refcv6 import evaluate


def reference_panel() -> dict:
    """A panel that PASSES every clause. Values are illustrative, not measured."""
    def cell(arm, ref, sep=True, n=4823, **kw):
        d = {"arm": arm, "reference": ref, "separated": sep, "n": n}
        d.update(kw)
        return d
    return {
        "arm": "D", "control": "V0",
        "tier": "T1",
        "estimator": "paired episode-cluster bootstrap (taniteval/ci.py)",
        "replicate_floor": {
            "ade": 0.0050, "heading_err": 0.05, "yaw_rate_err": 0.05,
            "cross_track": 0.005, "curvature_mae_masked": 0.0002,
        },
        "families": {
            "ADE": {
                # arm 0.40 vs ha0_ext 0.50 -> margin 0.20 >= 0.10
                "ade_vs_ha0_ext": cell(0.40, 0.50),
                "ade_vs_ha":      cell(0.40, 0.52),
                # effect 0.05 vs floor 0.005 -> 10x >= 3x
                "ade_vs_control": cell(0.40, 0.45),
            },
            "LATERAL": {
                "heading_err":  cell(1.20, 1.21, sep=False),
                "yaw_rate_err": cell(1.05, 1.05, sep=False),
                "cross_track":  cell(0.098, 0.099, sep=False),
                "curvature_mae_masked": cell(0.00348, 0.00350, sep=False,
                                             straight_line_floor=0.006807),
            },
            "STRATEGIC": {"route_acc": cell(0.7710, 0.7708, sep=False, n=3622)},
            "LONGITUDINAL": {"speed_mae": cell(0.2540, 0.2919),
                             "along_track": cell(0.2348, 0.2655)},
            "TACTICAL": {"tactical_lateral_kappa": cell(0.8200, 0.8193, sep=False)},
        },
    }


def _drop(panel, family, metric):
    p = copy.deepcopy(panel)
    p["families"][family].pop(metric, None)
    return p


def _drop_family(panel, family):
    p = copy.deepcopy(panel)
    p["families"].pop(family, None)
    return p


def mutants(base):
    """(id, description, panel, the real defect it reproduces)."""
    out = []

    # --- the four lateral wins, dropped one at a time ---------------------- #
    for metric in ("heading_err", "yaw_rate_err", "cross_track",
                   "curvature_mae_masked"):
        out.append((
            f"DROP_LATERAL_{metric}",
            f"the results table simply omits LATERAL.{metric}",
            _drop(base, "LATERAL", metric),
            "refcv5-v2 WON this metric; omitting it from the table is how a "
            "non-regression clause gets silently dropped",
        ))

    # --- the whole lateral family ------------------------------------------ #
    out.append(("DROP_LATERAL_FAMILY", "the LATERAL family is absent entirely",
                _drop_family(base, "LATERAL"),
                "ADE-only reporting -- three reports went out this way AFTER "
                "four families were made binding"))

    # --- strategic --------------------------------------------------------- #
    out.append(("DROP_STRATEGIC", "the STRATEGIC family is absent entirely",
                _drop_family(base, "STRATEGIC"),
                "refcv5-v2 is the ONLY arm with a strategic output; refcv4b "
                "reads n = 0"))

    p = copy.deepcopy(base)
    p["families"]["STRATEGIC"]["route_acc"]["n"] = 0
    out.append(("STRATEGIC_N_ZERO", "route_acc present but n = 0", p,
                "refcv4b's exact state -- a row that LOOKS reported and carries "
                "no evidence"))

    # --- curvature without its floor --------------------------------------- #
    p = copy.deepcopy(base)
    p["families"]["LATERAL"]["curvature_mae_masked"].pop("straight_line_floor")
    out.append(("CURVATURE_NO_FLOOR", "masked curvature without its straight-line floor",
                p, "an uninterpretable number: refcv5-v2's value is 0.512x the floor"))

    # --- the ADE margin ----------------------------------------------------- #
    p = copy.deepcopy(base)
    p["families"]["ADE"]["ade_vs_ha0_ext"] = {
        "arm": 0.495, "reference": 0.50, "separated": True, "n": 4823}
    out.append(("ADE_SEPARATED_NO_MARGIN",
                "ADE CI separated but relative margin 0.01 < 0.10", p,
                "⛔ 'Clearing the CI while missing the margin is a FAIL as written'"))

    # --- one of the two baselines missing ----------------------------------- #
    out.append(("DROP_HA_BASELINE", "no comparison against `ha` (only ha0_ext)",
                _drop(base, "ADE", "ade_vs_ha"),
                "the bar names BOTH baselines"))

    # --- the replicate floor ------------------------------------------------ #
    p = copy.deepcopy(base)
    p.pop("replicate_floor")
    out.append(("DROP_REPLICATE_FLOOR", "no replicate floor in the panel", p,
                "H-ESTIM-SEED-1: a separated CI from a one-seed arm is NECESSARY, "
                "NOT SUFFICIENT"))

    # --- the forbidden estimator -------------------------------------------- #
    p = copy.deepcopy(base)
    p["estimator"] = "8-split episode-disjoint jackknife (overlapping_holdout_se)"
    out.append(("FORBIDDEN_ESTIMATOR", "panel scored with overlapping_holdout_se", p,
                "it biases the POINT ESTIMATE as well as the interval "
                "(-6.67 % to +11.69 %, bidirectional, 27 dumps)"))

    # --- the tier stamp ------------------------------------------------------ #
    p = copy.deepcopy(base)
    p["tier"] = "T0"
    out.append(("WRONG_TIER", "panel stamped T0 instead of T1", p,
                "T0 is a WM diagnostic, never driving performance; open-loop "
                "lateral skill was an ACTION ECHO (97.9 % -> ~5 % closed-loop)"))

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    base = reference_panel()
    base_v = evaluate(base)
    baseline_success = base_v["verdict"] == "SUCCESS"

    rows = []
    for mid, desc, panel, defect in mutants(base):
        v = evaluate(panel)
        rows.append({
            "mutant": mid, "description": desc, "real_defect": defect,
            "verdict": v["verdict"],
            "blocked": v["verdict"] != "SUCCESS",
            "n_missing_data": v["n_missing_data"], "n_fail": v["n_fail"],
            "why": v["why"][:220],
        })

    blocked = sum(1 for r in rows if r["blocked"])
    out = {
        "tool": "verdict_dropproof.py",
        "baseline_verdict": base_v["verdict"],
        "baseline_success": baseline_success,
        "n_mutants": len(rows), "n_blocked": blocked,
        "n_survived": len(rows) - blocked,
        "survivors": [r["mutant"] for r in rows if not r["blocked"]],
        "mutants": rows,
        "verdict": "PASS" if (baseline_success and blocked == len(rows)) else "REFUSE",
    }
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
        print(f"[dropproof] wrote {a.json}")

    print(f"[dropproof] baseline (all clauses present) = {base_v['verdict']}")
    for r in rows:
        print(f"  [{'BLOCKED ' if r['blocked'] else 'SURVIVED'}] "
              f"{r['mutant']:<28} -> {r['verdict']}")
    print(f"[dropproof] {blocked}/{len(rows)} mutants blocked | "
          f"verdict = {out['verdict']}")
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
