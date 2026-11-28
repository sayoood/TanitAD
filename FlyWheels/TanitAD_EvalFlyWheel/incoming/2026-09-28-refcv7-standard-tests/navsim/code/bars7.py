#!/usr/bin/env python3
"""SPEC §5's bars, evaluated AS WRITTEN on a milestone's summaries (any python).

    python code/bars7.py --milestone raw/milestones/step5000

Reads ``summary_{warmup,navhard,navtest}.json`` beside ``MILESTONE_SUMMARY.json`` and writes
``BARS.json``. ⛔ Below step 5,000 every bar is ``NOT_EVALUATED -- VALIDATION ONLY`` (SPEC §6),
whatever the numbers. A bar whose inputs are missing is ``UNAVAILABLE`` with the reason -- never a
pass. ⭐ The single-training-seed rule (SPEC_REFCV7 §3, SPEC §4): a margin within 2x the
INFERENCE-seed floor is ``NOT PROVEN``, whatever its interval says.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_stamp7 as M7  # noqa: E402

SPEC_HASH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw",
                         "SPEC_PREREG_HASH.txt")
N_NAVTEST = 12146


def _j(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:                                                     # noqa: BLE001
        return None


def _verdict(margin, sep, seed_floor, extra_ok=True):
    """FAILED unless margin > 0 and sep; NOT PROVEN when inside 2x the seed floor (or the floor
    was not measured); PASS otherwise."""
    if margin is None:
        return "UNAVAILABLE"
    if not (margin > 0 and sep and extra_ok):
        return "FAILED"
    if seed_floor is None:
        return "NOT PROVEN (inference-seed floor not measured on this split)"
    if margin <= 2.0 * abs(seed_floor):
        return "NOT PROVEN (margin inside 2x the inference-seed floor)"
    return "PASS"


def stop_fraction_navtest(bridge: str) -> dict:
    p = os.path.join(bridge, "seam_R7_A1.npz") if bridge else ""
    if not (p and os.path.exists(p)):
        return {"status": "UNAVAILABLE", "reason": "seam_R7_A1.npz absent"}
    z = np.load(p, allow_pickle=False)
    src = [str(s) for s in z["source"]]
    keep = [i for i, s in enumerate(src) if s != "cv_standin"]
    pp = np.asarray(z["poses"], np.float64)[keep]
    end = np.hypot(pp[:, -1, 0], pp[:, -1, 1])
    return {"n": len(keep), "frac_endpoint_lt_1m": float((end < 1.0).mean()),
            "median_4s_distance_m": float(np.median(end))}


def navtest_bar(s, bridge: str) -> dict:
    if not s:
        return {"status": "UNAVAILABLE", "reason": "summary_navtest.json absent"}
    arms, pairs = s.get("arms") or {}, s.get("pairs") or {}
    p = {k: (arms.get(k) or {}).get("PDMS") for k in ("R7_A1", "STOP", "CV", "HUMAN")}
    if p["R7_A1"] is None or p["STOP"] is None:
        return {"status": "UNAVAILABLE", "reason": f"missing PDMS: {p}"}
    full = s.get("n_tokens") == N_NAVTEST
    pi = (pairs.get("R7_A1__minus__STOP") or {}).get("interval") or {}
    sep = pi.get("status") == "OK" and pi.get("lo") is not None and pi["lo"] > 0
    sfp = pairs.get("R7_A1__minus__R7_A1_s1") or {}
    sf = sfp.get("delta_x100")
    margin = p["R7_A1"] - p["STOP"]
    v = _verdict(margin, sep, sf, extra_ok=full)
    beside = {k: (pairs.get(f"R7_A1__minus__{k}") or {}) for k in ("CV", "PRIOR_ha0p", "R7_CEILDECL_d")}
    return {"bar": "BAR-R7-N1 (programme bar, SPEC_REFCV7 §3)",
            "rule": ("navtest PDMS(R7_A1, filter ON) > STOP, paired log-cluster interval excluding 0, "
                     "on the FULL 12,146-token split; a margin inside 2x the full-split inference-seed "
                     "floor |PDMS(A1) - PDMS(A1_s1)| is NOT PROVEN"),
            "values_x100": p, "margin_x100": margin, "paired_vs_STOP": pi,
            "seed_floor_x100": sf, "seed_floor_interval": sfp.get("interval"),
            "n_tokens": s.get("n_tokens"), "full_split": full,
            "stop_fraction_A1": stop_fraction_navtest(bridge),
            "beside_not_instead": {k: {"delta_x100": b.get("delta_x100"),
                                       "interval": b.get("interval")} for k, b in beside.items()},
            "verdict": v + ("" if full else " (SUBSET -- not the bar's split)")}


def warmup_bar(s) -> dict:
    if not s:
        return {"status": "UNAVAILABLE", "reason": "summary_warmup.json absent"}
    arms = s["arms"]
    need = ("R7_A1", "CV_official", "STOP_zero", "ECHO_ha0_ext")
    if any(k not in arms for k in need):
        return {"status": "UNAVAILABLE", "reason": f"arms missing: {[k for k in need if k not in arms]}"}
    u = {k: arms[k]["S2_EPDMS_u"].get("value") for k in need}
    floor = max(u["CV_official"], u["STOP_zero"], u["ECHO_ha0_ext"])
    margin = u["R7_A1"] - floor
    seed = s.get("seed_floor_S2_EPDMS_u")
    return {"bar": "BAR-R7-NW1 (suite bar)",
            "rule": "S2-EPDMS-u(R7_A1) > max(CV, STOP, ECHO); margin above 2x the seed floor",
            "values": u, "margin": margin, "seed_floor": seed,
            "prior_S2_EPDMS_u": (arms.get("PRIOR_ha0p") or {}).get("S2_EPDMS_u", {}).get("value"),
            "verdict": _verdict(margin, True, seed),
            "stop_fraction_A1": arms["R7_A1"].get("stop_fraction"),
            "interval": "UNAVAILABLE (7 logs < 8)"}


def navhard_bar(s) -> dict:
    if not s:
        return {"status": "UNAVAILABLE", "reason": "summary_navhard.json absent"}
    arms = s["arms"]
    need = ("R7_A1", "CV_official", "STOP_zero", "ECHO_ha0_ext")
    o = {k: (arms.get(k) or {}).get("official_two_stage_EPDMS") for k in need}
    if any(not isinstance(v, float) for v in o.values()):
        return {"status": "UNAVAILABLE",
                "reason": f"official two-stage EPDMS not defined for all of {need}: {o}"}
    floor = max(o["CV_official"], o["STOP_zero"], o["ECHO_ha0_ext"])
    pi = (s.get("paired_intervals") or {}).get("R7_A1__minus__STOP_zero", {})
    sep = pi.get("status") == "OK" and (pi.get("lo") is not None and pi["lo"] > 0)
    sf = (s.get("pairs") or {}).get("R7_A1__minus__R7_A1_s1", {}).get("official_two_stage_EPDMS_delta")
    margin = o["R7_A1"] - floor
    return {"bar": "BAR-R7-NH1 (suite bar)",
            "rule": ("official two-stage EPDMS(R7_A1) > max(STOP, CV, ECHO); paired log-cluster "
                     "interval vs STOP excludes 0; margin above 2x the seed floor"),
            "values": o, "margin": margin, "paired_vs_STOP": pi, "seed_floor_official": sf,
            "verdict": _verdict(margin, sep, sf), "stop_fraction_A1": arms["R7_A1"].get("stop_fraction")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", required=True)
    a = ap.parse_args(argv)
    ms = _j(os.path.join(a.milestone, "MILESTONE_SUMMARY.json")) or {}
    step = int(ms.get("step", -1))
    spec = open(SPEC_HASH, encoding="utf-8").read().strip() if os.path.exists(SPEC_HASH) else None
    out = {"step": step, "label": ms.get("label"), "spec": spec, "model_as_trained": M7.stamp(step),
           "tier": ("NavSim open-loop benchmark (T1-family); zero-shot; never closed loop; "
                    "non-parity"),
           "estimator_question": ("the log-cluster interval answers 'another draw of LOGS?' only; "
                                  "inference variance = the R7_A1_s1 seed floor; training variance "
                                  "UNTESTED (one training seed)")}
    if step < 5000:
        out["bars"] = "NOT_EVALUATED -- VALIDATION ONLY (SPEC §6: step < 5,000)"
    else:
        out["bars"] = {"navtest": navtest_bar(_j(os.path.join(a.milestone, "summary_navtest.json")),
                                              os.path.join(a.milestone, "bridge_navtest")),
                       "warmup": warmup_bar(_j(os.path.join(a.milestone, "summary_warmup.json"))),
                       "navhard": navhard_bar(_j(os.path.join(a.milestone, "summary_navhard.json")))}
    with open(os.path.join(a.milestone, "BARS.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(out, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
