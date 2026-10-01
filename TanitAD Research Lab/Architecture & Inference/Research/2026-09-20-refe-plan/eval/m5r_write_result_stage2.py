#!/usr/bin/env python3
"""Measure 5r STAGE 2: writes eval/RESULT_M5R_STAGE2.md from eval/raw/m5r/readout_stage2.json (+ the Stage-2 gates) --
no number is typed. Pre-written before the readout exists; every branch of PREREG_MEASURE5R §6 + ADDENDUM 1 (blob
2f32042b; scope correction ff3fd2f3) carries its registered consequence:
  ADOPT V3r / ADOPT V4r -> a DRAFTED pod kit (V3r: --repair-last-heading, no copies; V4r: + --slow-copies 0.75 frac 0.5),
                           shipped ONLY on the PI's go; nothing goes to the pod before it
  NOTHING ADOPTED       -> each arm's verdict as read; next lever: a NAVSIM-faithful label for every component on the
                           simulated states (collision first), then a re-test
The hand-written section below the marker is kept on re-run.

    python eval/m5r_write_result_stage2.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw", "m5r")
OUTP = os.path.join(HERE, "RESULT_M5R_STAGE2.md")
MARK = "<!-- HAND-WRITTEN BELOW THIS LINE (kept on re-run) -->"


def ci(x, nd=4):
    lo, hi = x["ci95"]
    return f"{x['diff']:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]" + (" (separated)" if lo > 0 or hi < 0 else "")


def main() -> int:
    R = json.load(open(os.path.join(RAW, "readout_stage2.json"), encoding="utf-8"))
    d = R["decision"]
    v = d["verdict"]
    cons = ("a DRAFTED pod kit for the adopted arm; it ships ONLY on the PI's go (nothing has been sent to the pod)"
            if v.startswith("ADOPT") else
            "nothing is deployed; next lever (registered): a NAVSIM-faithful label for every component on the "
            "simulated states (collision first), then a re-test")
    L = ["# RESULT -- Measure 5r STAGE 2 (fresh-token confirmation; Amendment 5's 923 tokens, 43 logs)", "",
         "*Pre-registration `eval/PREREG_MEASURE5R.md` §6 + ADDENDUM 1 (blob `2f32042b`; scope-prose correction "
         "`ff3fd2f3`). Applied as written. Numbers are written by `eval/m5r_write_result_stage2.py` from "
         "`eval/raw/m5r/readout_stage2.json`; none is typed.*", "",
         f"## Stage-2 decision (as registered): **{v}**", "", f"Registered consequence: {cons}.", "",
         f"Gates: {json.dumps(R['gates'])} (context admissibility: `stage2_gate_context.json`; eval: "
         f"`stage2_gate_eval.json`). Estimator: {R.get('estimator')}; n = {R['n_tokens']} tokens, {R['n_logs']} logs.", "",
         "| arm vs V3 | E1 (>= +0.03, lo > 0, > floor) | seed floor | E3a (lo > -0.01) | E3b (lo > -2.0) | E2 (lo > -2.0) | verdict |",
         "|---|---|---|---|---|---|---|"]
    for X, a in R["arms"].items():
        L.append(f"| {X} | {ci(a['E1'])} | {a['seed_floor']} | {ci(a['E3a'])} | {ci(a['E3b'], 2)} | {ci(a['E2'], 2)} | "
                 f"**{a['verdict']}** |")
    L += ["", f"Named bar (both pass -> V3r unless): E1 V4r - V3r = {ci(d['named_bar_E1_V4r_minus_V3r'])}, seed floor "
              f"{d['named_bar_seed_floor']} -> V4r beats V3r: **{d['V4r_beats_V3r_on_the_named_bar']}**.", "",
          "## All comparisons (reported)", "", "| metric | pair | diff [95 %] |", "|---|---|---|"]
    for key, pr in R["comparisons"].items():
        for pn, x in pr.items():
            L.append(f"| {key} | {pn} | {ci(x, 2 if key in ('E3b', 'E2') else 4)} |")
    ps = R.get("pick_subscores_x100")
    if ps:
        L += ["", "## EP and NAVSIM sub-scores of the picks (x100; NC, DAC, EP, TTC, C, DDC)", ""]
        for nm, r in ps["arm_means"].items():
            L.append(f"* {nm}: " + "; ".join(f"{route} {vals}" for route, vals in r.items()))
        L += ["", "| route | pair | EP | NC | DAC | TTC | C |", "|---|---|---|---|---|---|---|"]
        for route, pr in ps["pairs"].items():
            for pn, x in pr.items():
                L.append(f"| {route} | {pn} | " + " | ".join(
                    f"{x[h]['diff']:+.2f} [{x[h]['ci95'][0]:+.2f}, {x[h]['ci95'][1]:+.2f}]" for h in ("EP", "NC", "DAC", "TTC", "C")) + " |")
        L += ["", f"E2 selection mix: {json.dumps(R.get('E2_selection_mix'))}"]
    body = "\n".join(L) + "\n"
    old = open(OUTP, encoding="utf-8").read() if os.path.exists(OUTP) else ""
    hand = old.split(MARK, 1)[1] if MARK in old else "\n## Interpretation, four families, declared departures\n\n(to be written)\n"
    open(OUTP, "w", encoding="utf-8", newline="\n").write(body + "\n" + MARK + hand)
    print(f"ZZM5R2_RESULT_WRITTEN {OUTP} verdict {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
