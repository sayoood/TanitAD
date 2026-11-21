#!/usr/bin/env python3
"""Read a finished run's ``summary.json`` into the tables `RESULT.md` needs — headline, per stage,
sub-metrics, paired deltas, the interval WITH its estimator, the four families, and the bar verdict.

⛔ EVERY NUMBER COMES FROM ``summary.json``, never from a log line, a progress print or a memory of
what an arm scored. ⛔ The headline is the ``score`` column; ``pdm_score`` is refused on sight.
⛔ An arm whose stage 1 is a CV stand-in has NO two-stage headline and this prints the refusal
verbatim rather than the HYBRID combined row.

⭐ The bar verdict is computed against ``PREREG.md``'s `A1 > max(CV, STOP, ECHO)` on the STAGE-2
statistic, and it prints PASS/FAIL as a literal comparison — not as prose that can drift from the
arithmetic.

    python read_summary.py --run <run dir> --arm A1 --out raw/bar_verdict.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

FORBIDDEN = "pdm_score"


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", default="A1")
    ap.add_argument("--floors", default="CV,STOP,ECHO")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    S = json.load(open(os.path.join(a.run, "summary.json"), encoding="utf-8"))
    hm = S["headline_metric"]
    if hm.get("column") == FORBIDDEN:
        print(f"⛔ REFUSED: the headline column is {FORBIDDEN!r}")
        return 2
    floors = [f.strip() for f in a.floors.split(",") if f.strip()]
    arms = S["arms"]
    out = {"run_id": S["run_id"], "split": S["split"], "protocol": S["protocol"],
           "headline_column": hm["column"], "forbidden_column": hm.get("forbidden_column"),
           "stamps": S["stamps"], "arms": {}}
    print(f"run {S['run_id']}  split {S['split']}  protocol {S['protocol']}")
    print(f"tier {g(S, 'stamps', 'tier')} · loop s1 {g(S, 'stamps', 'loop', 'stage_one')} · "
          f"s2 {g(S, 'stamps', 'loop', 'stage_two')} · closed_loop {g(S, 'stamps', 'closed_loop')}")
    print()
    print("%-6s %-34s %10s %10s %10s" % ("arm", "official two-stage EPDMS", "S2-EPDMS-u", "s1 mean", "s2 mean"))
    for nm, arec in arms.items():
        h = arec.get("headline", {})
        hv = ("UNAVAILABLE" if h.get("status") == "UNAVAILABLE"
              else (f"{h['value']:.6f}" if h.get("value") is not None else "-"))
        st = arec.get("statistics", {})
        u = g(st, "S2_EPDMS_u", "value")
        out["arms"][nm] = {"kind": arec.get("kind"), "status": arec.get("status"),
                           "headline": h, "S2_EPDMS_u": u,
                           "stage_one_scene_mean": st.get("stage_one_scene_mean"),
                           "stage_two_scene_mean": st.get("stage_two_scene_mean"),
                           "interval": arec.get("interval"), "families": arec.get("families"),
                           "declared_inputs": arec.get("declared_inputs"),
                           "modality": arec.get("modality"),
                           "criteria": g(arec, "controls", "criteria_check")}
        print("%-6s %-34s %10s %10s %10s" % (
            nm, hv, f"{u:.6f}" if u is not None else "-",
            f"{st.get('stage_one_scene_mean'):.6f}" if st.get("stage_one_scene_mean") is not None else "-",
            f"{st.get('stage_two_scene_mean'):.6f}" if st.get("stage_two_scene_mean") is not None else "-"))
        if h.get("status") == "UNAVAILABLE":
            print(f"        ⛔ {h.get('reason', '')[:200]}")
    # ---- the pre-registered bar -------------------------------------------------------
    me = g(out, "arms", a.arm, "S2_EPDMS_u")
    others = {f: g(out, "arms", f, "S2_EPDMS_u") for f in floors}
    have = {k: v for k, v in others.items() if v is not None}
    missing = [k for k, v in others.items() if v is None]
    # ⛔ A FLOOR THAT IS MISSING MAKES THE BAR UNDEFINED — it is NEVER dropped from the max. The
    # pre-registered bar is `A1 > max(CV, STOP, ECHO)`; deciding it over the floors that happened to
    # score would be moving the goalpost after seeing which arms survived, and ECHO is exactly the
    # floor a RAM abort is most likely to take out (it is scored in the same window as A1).
    # (W7, 2026-09-21: the first version silently filtered `None` out of `have` and would have done it.)
    if me is None or missing:
        bar = {"status": "UNDEFINED",
               "reason": ("the arm has no S2_EPDMS_u" if me is None else
                          f"pre-registered floor(s) {missing} were not scored; the bar names them and is "
                          "NOT decided over the survivors"),
               "arm": me, "floors": others}
    else:
        worst = max(have, key=lambda k: have[k])
        bar = {"status": "PASS" if me > have[worst] else "FAIL",
               "bar": "BAR-W7-1: A1 > max(CV, STOP, ECHO) on S2-EPDMS-u (PREREG.md)",
               "arm": a.arm, "arm_value": me, "strongest_floor": worst,
               "strongest_floor_value": have[worst],
               "delta_vs_strongest_floor": round(me - have[worst], 6),
               "all_floors": have,
               "deltas": {k: round(me - v, 6) for k, v in have.items()}}
        if bar["status"] == "FAIL":
            beat = [k for k, v in have.items() if me > v]
            bar["sub_case"] = ("FAIL-A (beats CV, loses to a floor)" if "CV" in beat
                               else "FAIL-B (does not even beat CV)")
            bar["floors_beaten"] = beat
    out["bar_verdict"] = bar
    print()
    print("BAR-W7-1 (pre-registered): " + json.dumps(bar, indent=1))
    # ---- paired blocks ---------------------------------------------------------------
    pr = g(arms, a.arm, "paired", default={}) or {}
    out["paired"] = pr
    for f, blk in pr.items():
        print(f"\npaired {a.arm} vs {f}: " + json.dumps(blk, indent=1)[:1400])
    iv = g(arms, a.arm, "interval", default={})
    print("\ninterval: " + json.dumps(iv, indent=1)[:900])
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
