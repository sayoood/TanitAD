#!/usr/bin/env python3
"""Prove the STRATEGIC scorer can DISCRIMINATE — before anyone quotes it.

The programme's rule is that a metric which cannot separate a right answer from
a wrong one certifies nothing, however reasonable its value looks (learned the
hard way when PSNR ranked a WRONG frame first on the night clip). So: four
synthetic arms are scored against the **real** map-derived option sets of every
admissible surveyed scene.

  ORACLE              always takes the branch the ego took        -> must be 1.0
  RANDOM              uniform over the admissible options
  ALWAYS_STRAIGHTEST  the option with the smallest |branch dyaw|  -> the
                      degenerate policy a flat model would learn
  NO_HEAD             emits nothing                               -> must be 0.0

It doubles as the power analysis: the CI width it reports on a handful of
scenes is what tells you how many scenes a real verdict needs.

    python score_demo.py --results results/strategic_gt --out results/score_demo.json
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategic_gt import score_strategic  # noqa: E402


def build_arms(ev, seed=0):
    rng = random.Random(seed)
    arms = {"ORACLE": {}, "RANDOM": {}, "ALWAYS_STRAIGHTEST": {}, "NO_HEAD": {}}
    for evs in ev.values():
        for e in evs:
            if not e["SCOREABLE"]:
                continue
            k = e["event_id"]
            arms["ORACLE"][k] = {"road": e["connecting_road_taken"],
                                 "class": e["route_gt_class"]}
            o = rng.choice(e["options"])
            arms["RANDOM"][k] = {"road": o["road"], "class": o["class"]}
            s = min(e["options"], key=lambda x: abs(x["branch_dyaw_deg"] or 999))
            arms["ALWAYS_STRAIGHTEST"][k] = {"road": s["road"], "class": s["class"]}
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/strategic_gt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    ev, refused = {}, []
    for f in sorted(glob.glob(str(Path(a.results) / "strategic_gt_*.json"))):
        d = json.loads(Path(f).read_text())
        sid = Path(f).stem.replace("strategic_gt_", "")
        if not d["ADMISSIBLE"]:
            refused.append({"scene": sid,
                            "worst_abs_err_deg":
                                d["SELFCONSISTENCY_CONTROL"]["worst_abs_err_deg_untruncated"]})
            continue
        ev[sid] = d["events"]

    rep = {"tool": "score_demo.py", "evidence_class": "MEASURED",
           "n_admissible_scenes": len(ev),
           "n_scenes_refused_by_selfconsistency_control": len(refused),
           "refused": refused, "arms": {}}
    for name, p in build_arms(ev).items():
        r = score_strategic(ev, p, n_boot=a.n_boot)
        r.pop("rows", None)
        rep["arms"][name] = r
        if r["n_events"] == 0:
            print(name, r["reason"])
            continue
        acc = r["route_choice_accuracy"]
        print(f"  {name:20s} choice={acc['mean']:.4f} "
              f"[{acc.get('lo', float('nan')):.4f},{acc.get('hi', float('nan')):.4f}] "
              f"class={r['route_class_accuracy']['mean']:.4f} floor={r['chance_floor']:.4f} "
              f"n_events={r['n_events']} n_scenes={r['n_scenes']}")

    o = rep["arms"].get("ORACLE", {}).get("route_choice_accuracy", {})
    rr = rep["arms"].get("RANDOM", {}).get("route_choice_accuracy", {})
    if "lo" in o and "lo" in rr:
        rep["DISCRIMINATES"] = bool(o["lo"] > rr["hi"])
        rep["ci_width_random"] = round(rr["hi"] - rr["lo"], 4)
        rep["power_note"] = (
            f"RANDOM's 95% CI is {rep['ci_width_random']:.3f} wide on "
            f"{rep['arms']['RANDOM']['n_events']} events from "
            f"{rep['arms']['RANDOM']['n_scenes']} scenes. A cluster bootstrap narrows "
            "roughly as 1/sqrt(n_scenes), so halving that width needs ~4x the scenes and "
            "a +-0.10 verdict needs ~100+ scenes with a scoreable branch. That is exactly "
            "what the 141 T1 scenes in nurec_survey_report.json are for -- a shortlist "
            "cannot carry a strategic verdict.")
        print(f"\nDISCRIMINATES: {rep['DISCRIMINATES']}  (ORACLE lo {o['lo']:.4f} vs "
              f"RANDOM hi {rr['hi']:.4f})")
        print(rep["power_note"])
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
