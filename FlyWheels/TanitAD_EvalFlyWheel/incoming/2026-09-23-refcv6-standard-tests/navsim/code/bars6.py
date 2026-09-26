#!/usr/bin/env python3
"""SPEC §5's bars, evaluated AS WRITTEN on a milestone's summaries (any python).

    python code/bars6.py --milestone raw/milestones/step5000

Reads ``summary_warmup.json`` / ``summary_navhard.json`` / ``summary_navtest.json`` beside
``MILESTONE_SUMMARY.json`` and writes ``BARS.json``. ⛔ Below step 5,000 (the kit's step-1,000
checkpoint) every bar is ``NOT_EVALUATED — PIPELINE-VALIDATION`` (SPEC §6), whatever the numbers.
A bar whose inputs are missing is ``UNAVAILABLE`` with the reason — never a pass.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


#: every refcv6 number carries this stamp (Master Mind, 2026-09-26)
MODEL_STAMP = ('"F3 detach-only, F4 on the last layer only" — Master Mind audit 2026-09-26 (GOALS_AND_CLAIMS D-REFCV6-F3-WHITELIST, D-REFCV6-LABEL-CLOCK; landed 9d16c441): the F3 per-stage cascade loss never ran (decoder stages 0-2 frozen at init) and tactical labels are read ~0.37 s early; INHERITED')

def _j(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:                                                     # noqa: BLE001
        return None


def warmup_bar(s) -> dict:
    if not s:
        return {"status": "UNAVAILABLE", "reason": "summary_warmup.json absent"}
    arms = s["arms"]
    need = ("R6_A1", "CV_official", "STOP_zero", "ECHO_ha0_ext")
    if any(k not in arms for k in need):
        return {"status": "UNAVAILABLE", "reason": f"arms missing: {[k for k in need if k not in arms]}"}
    u = {k: arms[k]["S2_EPDMS_u"].get("value") for k in need}
    floor = max(u["CV_official"], u["STOP_zero"], u["ECHO_ha0_ext"])
    margin = u["R6_A1"] - floor
    seed = s.get("seed_floor_S2_EPDMS_u")
    ok = margin > 0 and (seed is None or margin > seed)
    return {"bar": "BAR-R6-W1", "rule": "S2-EPDMS-u(R6_A1) > max(CV, STOP, ECHO), margin > seed floor",
            "values": u, "margin": margin, "seed_floor": seed,
            "verdict": "PASS" if ok else ("FAIL — inside the seed floor" if margin > 0 else "FAIL"),
            "stop_fraction_A1": arms["R6_A1"].get("stop_fraction"), "interval": "UNAVAILABLE (7 logs < 8)"}


def navhard_bar(s) -> dict:
    if not s:
        return {"status": "UNAVAILABLE", "reason": "summary_navhard.json absent"}
    arms = s["arms"]
    need = ("R6_A1", "CV_official", "STOP_zero", "ECHO_ha0_ext")
    o = {k: arms.get(k, {}).get("official_two_stage_EPDMS") for k in need}
    if any(not isinstance(v, float) for v in o.values()):
        return {"status": "UNAVAILABLE", "reason": f"official two-stage EPDMS not defined for all of {need}: {o}"}
    floor = max(o["CV_official"], o["STOP_zero"], o["ECHO_ha0_ext"])
    pi = (s.get("paired_intervals") or {}).get("R6_A1__minus__STOP_zero", {})
    sep = pi.get("status") == "OK" and (pi.get("lo") is not None and pi["lo"] > 0)
    sf = (s.get("pairs") or {}).get("R6_A1__minus__R6_A1_s1", {}).get("official_two_stage_EPDMS_delta")
    margin = o["R6_A1"] - floor
    ok = margin > 0 and sep and (sf is None or margin > abs(sf))
    return {"bar": "BAR-R6-H1", "rule": "official two-stage EPDMS(R6_A1) > max(STOP, CV, ECHO); paired "
                                        "log-cluster interval vs STOP excludes 0; margin > seed floor",
            "values": o, "margin": margin, "paired_vs_STOP": pi, "seed_floor_official": sf,
            "verdict": "PASS" if ok else "FAIL"}


def navtest_bar(s) -> dict:
    if not s:
        return {"status": "UNAVAILABLE", "reason": "summary_navtest.json absent"}
    arms = s["arms"]
    p = {k: arms.get(k, {}).get("PDMS") for k in ("R6_A1", "STOP", "CV")}
    if any(v is None for v in p.values()):
        return {"status": "UNAVAILABLE", "reason": f"missing PDMS: {p}"}
    pi = (s.get("pairs") or {}).get("R6_A1__minus__STOP", {}).get("interval", {})
    sep = pi.get("status") == "OK" and pi.get("lo") is not None and pi["lo"] > 0
    ok = p["R6_A1"] > max(p["STOP"], p["CV"]) and sep
    return {"bar": "BAR-R6-T1", "rule": "PDMS(R6_A1) > max(STOP, CV), paired log-cluster interval vs STOP "
                                        "excludes 0", "values_x100": p, "paired_vs_STOP": pi,
            "n_tokens": s.get("n_tokens"), "full_split": s.get("n_tokens") == 12146,
            "stretch_published": {"Ego-Status-MLP": "65.6 / 66.4 (arXiv 2406.15349)",
                                  "DiffusionDrive": ("88.1 PDMS (arXiv 2411.15139, Tab. 1 p. 6; library "
                                                     "sha256 6ad4f8a3...; CAMERA+LiDAR, trained on navtrain "
                                                     "- refcv6 is camera-only and zero-shot)")},
            "verdict": ("PASS" if ok else "FAIL") + ("" if s.get("n_tokens") == 12146
                                                    else " (SUBSET — not the published split)")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", required=True)
    a = ap.parse_args(argv)
    ms = _j(os.path.join(a.milestone, "MILESTONE_SUMMARY.json")) or {}
    step = int(ms.get("step", -1))
    out = {"step": step, "label": ms.get("label"), "spec": "SPEC.md §5 (blob ab7f769a…)",
           "model_as_trained": MODEL_STAMP}
    if step < 5000:
        out["bars"] = "NOT_EVALUATED — PIPELINE-VALIDATION (SPEC §6: step < 5,000)"
    else:
        out["bars"] = {"warmup": warmup_bar(_j(os.path.join(a.milestone, "summary_warmup.json"))),
                       "navhard": navhard_bar(_j(os.path.join(a.milestone, "summary_navhard.json"))),
                       "navtest": navtest_bar(_j(os.path.join(a.milestone, "summary_navtest.json")))}
    with open(os.path.join(a.milestone, "BARS.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
