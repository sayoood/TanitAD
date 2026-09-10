#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv6 -- ⛔ THE COMMITTED BAR, AS CODE. Written BEFORE any data exists.

    python verdict_refcv6.py --panel raw/panel.json --json raw/verdict.json

# Why the bar is a program and not a paragraph

⭐ **THE NON-REGRESSION CLAUSES ARE THE HALF MOST LIKELY TO BE LOST.** refcv5-v2
WON heading (1.2121 vs 1.5489), yaw-rate (1.0534 vs 1.4542), cross-track (0.0994
vs 0.1226) and masked curvature (0.003485 vs 0.004030), and it is the ONLY arm in
the programme with a strategic output at all (route accuracy 0.7708 [0.7146,
0.8254], kappa 0.4614, n 3,622, against a 0.3333 chance, from a 771-parameter
head; refcv4b reads n = 0). The obvious way to "fix" refcv5-v2's ADE loss is to
hand those back.

A clause written only in prose is dropped by being *not mentioned* in the results
table -- which is exactly how ADE-only reports kept going out after four families
were made binding. ⇒ **This tool REFUSES to emit SUCCESS when a required clause's
data is ABSENT.** Silence cannot pass. `MISSING_DATA` is a distinct verdict from
`FAIL`, and neither is `SUCCESS`.

⛔ Absence is not a pass, and it is not a fail either: it is `MISSING_DATA`, which
blocks the verdict. That distinction matters -- reporting a missing family as a
failure would invite deleting the family to "fix" it.

# The bar, verbatim from the pre-registration

**PRIMARY (ADE)** -- `arm` must beat BOTH `ha0_ext` AND `ha`:
    (i)  the paired episode-cluster bootstrap CI on the difference is SEPARATED, and
    (ii) the relative margin is >= 0.10.
    ⛔ Clearing the CI while missing the margin is a **FAIL as written**.

**LATERAL NON-REGRESSION** -- heading, yaw-rate, cross-track, masked curvature.
    None may be separably WORSE than the control by more than the arm's own
    replicate floor on that same metric.

**STRATEGIC NON-REGRESSION** -- route accuracy may not be separably worse than
    the control, and `n` must be > 0. ⛔ `n = 0` is `MISSING_DATA`, never a pass:
    refcv4b reads n = 0 and that is precisely the regression being guarded.

**REPLICATE FLOOR** -- every lever effect is stated as a RATIO to a floor measured
    in the SAME panel. A separated CI from a one-seed arm is NECESSARY, NOT
    SUFFICIENT: the episode-cluster bootstrap resamples EPISODES with the models
    held fixed, so it answers *"would another draw of episodes say this?"* and
    never *"would another training run say this?"*. MEASURED on WP-D: an arm with
    ZERO levers moved read "separably worse" on 5 of 9 family metrics and
    reproduced a headline ADE effect at +0.02460 against the lever's +0.02610.

**ESTIMATOR** -- paired episode-cluster bootstrap (`taniteval/ci.py`) only.
    ⛔ `overlapping_holdout_se` is FORBIDDEN: it biases the POINT ESTIMATE as well
    as the interval (-6.67 % to +11.69 %, bidirectional, over 27 dumps). A panel
    whose estimator field names it is REFUSED outright.

**TIER** -- T1. A panel without a `"tier": "T1"` stamp is REFUSED; comparisons
    across tiers are invalid, and T0 is a WM diagnostic, never driving performance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# --------------------------------------------------------------------------- #
# The clauses. ⛔ REQUIRED means: absent => the verdict cannot be SUCCESS.      #
# --------------------------------------------------------------------------- #

ADE_RELATIVE_MARGIN = 0.10
REPLICATE_FLOOR_RATIO = 1.0     # a lever effect must EXCEED its own noise floor
LEVER_FLOOR_RATIO_STRONG = 3.0  # the ratio a *headline* claim must clear

#: family -> metrics that MUST be present. Each entry says which direction is
#: better, so a "win" cannot be read off a bare sign.
REQUIRED_LATERAL = {
    "heading_err":    {"lower_is_better": True, "refcv5v2": 1.2121, "refcv4b": 1.5489},
    "yaw_rate_err":   {"lower_is_better": True, "refcv5v2": 1.0534, "refcv4b": 1.4542},
    "cross_track":    {"lower_is_better": True, "refcv5v2": 0.0994, "refcv4b": 0.1226},
    "curvature_mae_masked": {"lower_is_better": True, "refcv5v2": 0.003485,
                             "refcv4b": 0.004030,
                             "note": "MASKED, with the straight-line floor beside "
                                     "it: refcv5-v2 reads 0.512x that floor."},
}

REQUIRED_STRATEGIC = {
    "route_acc": {"lower_is_better": False, "refcv5v2": 0.7708, "chance": 0.3333,
                  "refcv4b_n": 0,
                  "note": "771-parameter head; the ONLY strategic output in the "
                          "programme. n must be > 0 -- refcv4b's n = 0 IS the "
                          "regression this clause guards."},
}

REQUIRED_LONGITUDINAL = {
    "speed_mae":   {"lower_is_better": True, "refcv5v2": 0.2919, "refcv4b": 0.2540},
    "along_track": {"lower_is_better": True, "refcv5v2": 0.2655, "refcv4b": 0.2348},
}

REQUIRED_TACTICAL = {
    "tactical_lateral_kappa": {"lower_is_better": False,
                               "refcv5v2": 0.8193, "refcv4b": 0.7374},
}

FORBIDDEN_ESTIMATORS = ("overlapping_holdout_se", "heldout", "jackknife")
REQUIRED_ESTIMATOR_TOKENS = ("paired", "episode", "bootstrap")


class Clause:
    def __init__(self, cid, family, text, required=True):
        self.cid, self.family, self.text, self.required = cid, family, text, required


CLAUSES = [
    Clause("P1", "ADE", "arm beats ha0_ext: paired CI separated AND relative margin >= 0.10"),
    Clause("P2", "ADE", "arm beats ha: paired CI separated AND relative margin >= 0.10"),
    Clause("P3", "ADE", "the ADE effect exceeds the replicate floor measured in THIS panel"),
    Clause("L1", "LATERAL", "heading_err not separably worse than control beyond the replicate floor"),
    Clause("L2", "LATERAL", "yaw_rate_err not separably worse than control beyond the replicate floor"),
    Clause("L3", "LATERAL", "cross_track not separably worse than control beyond the replicate floor"),
    Clause("L4", "LATERAL", "curvature_mae_masked not separably worse (straight-line floor reported beside it)"),
    Clause("S1", "STRATEGIC", "route_acc not separably worse than control, and n > 0"),
    Clause("G1", "LONGITUDINAL", "speed_mae and along_track REPORTED with n (the axis refcv5-v2 lost)"),
    Clause("T1", "TACTICAL", "tactical metrics REPORTED with n"),
]


# --------------------------------------------------------------------------- #

def _get(panel: dict, family: str, metric: str) -> dict | None:
    fam = (panel.get("families") or {}).get(family) or {}
    m = fam.get(metric)
    return m if isinstance(m, dict) else None


def _check_estimator(panel: dict) -> dict:
    est = str(panel.get("estimator", "")).lower()
    tier = str(panel.get("tier", "")).upper()
    problems = []
    for bad in FORBIDDEN_ESTIMATORS:
        if bad in est:
            problems.append(
                f"⛔ estimator names {bad!r}. FORBIDDEN: it biases the POINT "
                f"ESTIMATE as well as the interval (-6.67 % to +11.69 %, "
                f"bidirectional, over 27 dumps)."
            )
    if not est:
        problems.append("⛔ panel declares NO estimator. Never quote an interval "
                        "without its estimator.")
    elif not all(t in est for t in REQUIRED_ESTIMATOR_TOKENS):
        problems.append(f"⛔ estimator {est!r} is not the paired episode-cluster "
                        f"bootstrap required by this pre-registration.")
    if tier != "T1":
        problems.append(f"⛔ tier is {tier or 'ABSENT'!r}, must be T1. T0 is a WM "
                        f"diagnostic and is never a driving result; cross-tier "
                        f"comparison is invalid.")
    return {"estimator": est, "tier": tier, "problems": problems,
            "ok": not problems}


def _margin_and_sep(m: dict, lower_is_better: bool) -> tuple[float | None, bool | None, list]:
    """Return (relative_margin, separated, problems). Missing fields -> None."""
    problems = []
    sep = m.get("separated")
    if sep is None:
        problems.append("no 'separated' field (the CI's decision predicate)")
    arm, ref = m.get("arm"), m.get("reference")
    rel = None
    if arm is None or ref is None:
        problems.append("no 'arm'/'reference' point estimates for a relative margin")
    else:
        try:
            arm, ref = float(arm), float(ref)
            if ref == 0:
                problems.append("reference is 0; a relative margin is undefined")
            else:
                rel = (ref - arm) / abs(ref) if lower_is_better else (arm - ref) / abs(ref)
        except (TypeError, ValueError):
            problems.append("non-numeric point estimates")
    return rel, (bool(sep) if sep is not None else None), problems


def evaluate(panel: dict) -> dict:
    results: list[dict] = []
    est = _check_estimator(panel)

    def add(cid, verdict, detail, why):
        c = next(x for x in CLAUSES if x.cid == cid)
        results.append({"clause": cid, "family": c.family, "text": c.text,
                        "verdict": verdict, "detail": detail, "why": why})

    # ---- PRIMARY: ADE against ha0_ext and ha ------------------------------- #
    for cid, ref_name in (("P1", "ha0_ext"), ("P2", "ha")):
        m = _get(panel, "ADE", f"ade_vs_{ref_name}")
        if m is None:
            add(cid, "MISSING_DATA", {"looked_for": f"families.ADE.ade_vs_{ref_name}"},
                f"⛔ no ADE comparison against {ref_name}. The bar names BOTH "
                f"baselines; a panel missing one cannot clear it.")
            continue
        rel, sep, probs = _margin_and_sep(m, lower_is_better=True)
        if probs:
            add(cid, "MISSING_DATA", {"metric": m, "problems": probs},
                f"⛔ incomplete: {'; '.join(probs)}")
            continue
        ok = bool(sep) and rel is not None and rel >= ADE_RELATIVE_MARGIN
        add(cid, "PASS" if ok else "FAIL",
            {"separated": sep, "relative_margin": round(rel, 6) if rel is not None else None,
             "required_margin": ADE_RELATIVE_MARGIN, "n": m.get("n")},
            (f"separated and margin {rel:.4f} >= {ADE_RELATIVE_MARGIN}" if ok else
             (f"⛔ CI not separated" if not sep else
              f"⛔ margin {rel:.4f} < {ADE_RELATIVE_MARGIN}. ⛔ CLEARING THE CI "
              f"WHILE MISSING THE MARGIN IS A FAIL AS WRITTEN.")))

    # ---- P3: the ADE effect against the replicate floor -------------------- #
    m = _get(panel, "ADE", "ade_vs_control")
    floor = (panel.get("replicate_floor") or {}).get("ade")
    if m is None or floor is None:
        add("P3", "MISSING_DATA",
            {"has_effect": m is not None, "has_floor": floor is not None},
            "⛔ a lever effect without a replicate floor measured in the SAME "
            "panel is not admissible. A separated CI from a one-seed arm is "
            "NECESSARY, NOT SUFFICIENT (H-ESTIM-SEED-1: an arm with zero levers "
            "moved read 'separably worse' on 5 of 9 family cells).")
    else:
        eff = m.get("arm")
        ref = m.get("reference")
        try:
            delta = abs(float(ref) - float(eff))
            fl = abs(float(floor))
            ratio = (delta / fl) if fl > 0 else None
        except (TypeError, ValueError):
            delta = fl = ratio = None
        ok = ratio is not None and ratio >= LEVER_FLOOR_RATIO_STRONG
        add("P3", "PASS" if ok else "FAIL",
            {"delta": delta, "replicate_floor": fl, "ratio": ratio,
             "required_ratio": LEVER_FLOOR_RATIO_STRONG},
            (f"effect {delta:.5f} is {ratio:.2f}x the replicate floor {fl:.5f}" if ok else
             f"⛔ effect is {ratio if ratio is not None else '?'}x the replicate "
             f"floor; a headline claim must clear {LEVER_FLOOR_RATIO_STRONG}x."))

    # ---- LATERAL non-regression -------------------------------------------- #
    for cid, metric in (("L1", "heading_err"), ("L2", "yaw_rate_err"),
                        ("L3", "cross_track"), ("L4", "curvature_mae_masked")):
        spec = REQUIRED_LATERAL[metric]
        m = _get(panel, "LATERAL", metric)
        if m is None:
            add(cid, "MISSING_DATA", {"looked_for": f"families.LATERAL.{metric}",
                                      "refcv5v2_value": spec.get("refcv5v2")},
                f"⛔ {metric} ABSENT. refcv5-v2 WON this metric "
                f"({spec['refcv5v2']} vs refcv4b {spec.get('refcv4b')}); handing "
                f"it back to buy ADE is exactly the trade this clause forbids. "
                f"⛔ SILENCE IS NOT A PASS.")
            continue
        if metric == "curvature_mae_masked" and m.get("straight_line_floor") is None:
            add(cid, "MISSING_DATA", {"metric": m},
                "⛔ masked curvature reported WITHOUT its straight-line floor. "
                "The floor is what makes the number interpretable "
                "(refcv5-v2 reads 0.512x it).")
            continue
        rel, sep, probs = _margin_and_sep(m, lower_is_better=spec["lower_is_better"])
        if probs:
            add(cid, "MISSING_DATA", {"metric": m, "problems": probs},
                f"⛔ incomplete: {'; '.join(probs)}")
            continue
        # A regression is: separated AND in the control's favour.
        regressed = bool(sep) and rel is not None and rel < 0
        fl = (panel.get("replicate_floor") or {}).get(metric)
        within_floor = False
        if regressed and fl not in (None, 0):
            try:
                within_floor = abs(float(m["arm"]) - float(m["reference"])) <= abs(float(fl))
            except (TypeError, ValueError):
                within_floor = False
        ok = (not regressed) or within_floor
        add(cid, "PASS" if ok else "FAIL",
            {"separated": sep, "relative_delta": round(rel, 6) if rel is not None else None,
             "replicate_floor": fl, "within_floor": within_floor, "n": m.get("n")},
            ("no separated regression" if not regressed else
             (f"regression is within the replicate floor {fl}" if within_floor else
              f"⛔ {metric} is separably WORSE than the control by more than its "
              f"replicate floor. refcv6 may not buy ADE with refcv5-v2's lateral "
              f"wins.")))

    # ---- STRATEGIC non-regression ------------------------------------------ #
    m = _get(panel, "STRATEGIC", "route_acc")
    if m is None:
        add("S1", "MISSING_DATA", {"looked_for": "families.STRATEGIC.route_acc"},
            "⛔ route accuracy ABSENT. refcv5-v2 is the ONLY arm in the programme "
            "with a strategic output (0.7708 [0.7146, 0.8254], kappa 0.4614, "
            "n 3,622, chance 0.3333, from 771 parameters); refcv4b reads n = 0. "
            "⛔ AN ABSENT STRATEGIC ROW IS THE REGRESSION, NOT AN OMISSION.")
    else:
        n = m.get("n")
        if not isinstance(n, int) or n <= 0:
            add("S1", "MISSING_DATA", {"n": n},
                "⛔ route accuracy carries n = 0 (or no n). That is refcv4b's "
                "state and it is precisely what this clause guards. n = 0 is "
                "MISSING_DATA, never a pass.")
        else:
            rel, sep, probs = _margin_and_sep(m, lower_is_better=False)
            if probs:
                add("S1", "MISSING_DATA", {"metric": m, "problems": probs},
                    f"⛔ incomplete: {'; '.join(probs)}")
            else:
                regressed = bool(sep) and rel is not None and rel < 0
                add("S1", "FAIL" if regressed else "PASS",
                    {"separated": sep, "relative_delta": round(rel, 6),
                     "n": n, "chance": REQUIRED_STRATEGIC["route_acc"]["chance"]},
                    ("no separated strategic regression" if not regressed else
                     "⛔ route accuracy separably WORSE than the control."))

    # ---- LONGITUDINAL / TACTICAL: reported, with n ------------------------- #
    for cid, family, req in (("G1", "LONGITUDINAL", REQUIRED_LONGITUDINAL),
                             ("T1", "TACTICAL", REQUIRED_TACTICAL)):
        missing = [k for k in req if _get(panel, family, k) is None]
        no_n = [k for k in req
                if _get(panel, family, k) is not None
                and not isinstance((_get(panel, family, k) or {}).get("n"), int)]
        if missing or no_n:
            add(cid, "MISSING_DATA", {"missing": missing, "missing_n": no_n},
                f"⛔ {family}: absent {missing or '-'}, no n for {no_n or '-'}. "
                f"⛔ ADE IS ONE ROW OF FOUR; a family without its n is not "
                f"admissible. Where a family genuinely cannot be computed, say so "
                f"PER FAMILY with the reason and the n.")
        else:
            add(cid, "PASS", {"metrics": sorted(req)},
                f"{family} reported with n")

    # ---- roll up ----------------------------------------------------------- #
    n_missing = sum(1 for r in results if r["verdict"] == "MISSING_DATA")
    n_fail = sum(1 for r in results if r["verdict"] == "FAIL")

    if not est["ok"]:
        verdict = "REFUSED"
        why = ("⛔ the panel's estimator or tier is inadmissible; no verdict may "
               "be issued from it. " + " ".join(est["problems"]))
    elif n_missing:
        verdict = "MISSING_DATA"
        why = (f"⛔ {n_missing} required clause(s) have NO DATA. This is NOT a "
               f"pass and NOT a fail -- the bar cannot be evaluated. The clauses "
               f"were committed before any data existed and cannot be dropped by "
               f"being left out of the results table.")
    elif n_fail:
        verdict = "FAIL"
        why = f"⛔ {n_fail} clause(s) FAILED. Reported as written."
    else:
        verdict = "SUCCESS"
        why = ("every committed clause passed: the ADE bar against BOTH "
               "baselines with the 0.10 margin, the lateral non-regression, the "
               "strategic non-regression, and the replicate floor.")

    return {
        "tool": "verdict_refcv6.py",
        "arm": panel.get("arm"), "control": panel.get("control"),
        "estimator_check": est,
        "clauses": results,
        "n_clauses": len(results), "n_pass": sum(1 for r in results if r["verdict"] == "PASS"),
        "n_fail": n_fail, "n_missing_data": n_missing,
        "verdict": verdict, "why": why,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    try:
        with open(a.panel, "r", encoding="utf-8") as fh:
            panel = json.load(fh)
    except Exception as exc:
        print(f"⛔ could not read panel: {exc}", file=sys.stderr)
        return 3
    out = evaluate(panel)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
        print(f"[verdict] wrote {a.json}")
    for r in out["clauses"]:
        print(f"  [{r['verdict']:>12}] {r['clause']} ({r['family']}): {r['why'][:130]}")
    print(f"[verdict] {out['verdict']} -- {out['why'][:200]}")
    return 0 if out["verdict"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
