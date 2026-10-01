#!/usr/bin/env python3
"""Measure 5r: writes eval/RESULT_M5R_STAGE1.md from the banked artifacts -- no number is typed. Pre-written BEFORE the
readout exists, so the verdict is read the moment the run ends; every branch of PREREG_MEASURE5R §6 (blob a8241136) has
its registered consequence written here in advance:
  STAGE-1 PASS -> Stage 2 (A5's 923 fresh tokens, 73 new repaired harness runs, M6's eval cache) runs; ADOPT only there
  REFUTED      -> v4r not deployed; next lever: a NAVSIM-faithful label for every component on the simulated states
                  (collision first, PREREG_MEASURE5 §1b), then a re-test
  NOT PROVEN   -> not deployed; Stage 2 not run; the same next lever
  NO READOUT   -> a gate failed (named); G-R3/G-R4 failing = v4r NOT tested (BLOCKED on the label)

    python eval/m5r_write_result.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw", "m5r")
OUTP = os.path.join(HERE, "RESULT_M5R_STAGE1.md")
MARK = "<!-- HAND-WRITTEN BELOW THIS LINE (kept on re-run) -->"
CONSEQUENCE = {
    "STAGE-1 PASS": "Stage 2 runs next: the same four bars on Amendment 5's 923 fresh tokens (43 logs), repaired truth "
                    "(73 new harness runs), M6's eval cache for the scorer context after its admissibility gate. ADOPT "
                    "is decided ONLY there; an ADOPT then leads to a DRAFTED pod kit, shipped only on the PI's go.",
    "REFUTED": "v4r is NOT deployed. Next lever (registered): a NAVSIM-faithful label for every component on the "
               "simulated states -- collision first (PREREG_MEASURE5 §1b) -- then a re-test. Stage 2 is not run.",
    "NOT PROVEN": "v4r is NOT deployed and Stage 2 is not run. Next lever (registered): a NAVSIM-faithful label for "
                  "every component on the simulated states (collision first), then a re-test.",
}


def j(name, default=None):
    try:
        return json.load(open(os.path.join(RAW, name), encoding="utf-8"))
    except (OSError, ValueError):
        return default


def ci(x, nd=4):
    if not x:
        return "n/a"
    lo, hi = x["ci95"]
    return f"{x['diff']:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]" + (" (separated)" if lo > 0 or hi < 0 else "")


def main() -> int:
    R = j("readout_stage1.json", {})
    G = {k: j(f) for k, f in (("G-R1", "selftest_repair.json"), ("G-R2", "gate_gr2.json"), ("G-R3/4", "gate_gr34.json"),
                              ("G-R5", "gate_gr5.json"), ("G-R6", "gate_gr6_eval.json"))}
    F = j("families_stage1.json", {})
    d = R.get("decision", {})
    v = d.get("verdict", "NOT YET READ")
    L = [f"# RESULT -- Measure 5r (v4r: the M5 labels on the Amendment-7-repaired plans), STAGE 1", "",
         "*Pre-registration `eval/PREREG_MEASURE5R.md`, blob `a8241136` (registered by the coordinator 2026-09-28, "
         "before any v4r label). Applied as written. Numbers are written by `eval/m5r_write_result.py` from "
         "`eval/raw/m5r/`; none is typed.*", "",
         f"## Stage-1 verdict (as registered): **{v}**", "",
         f"Registered consequence: {next((c for k, c in CONSEQUENCE.items() if v.startswith(k)), 'a gate failed -- no statistic is read; see the gates below')}",
         ""]
    if d.get("E1"):
        fl = d["seed_floor"]
        L += ["| criterion (§6, stage 1) | bar | measured (V4r - V3) | met |", "|---|---|---|---|",
              f"| E1 slow-plan pair concordance (masked) | >= +0.03, CI lo > 0, > seed floor {fl} | {ci(d['E1'])} | "
              f"{'yes' if d['E1']['diff'] >= 0.03 and d['E1']['ci95'][0] > 0 and d['E1']['diff'] > fl else 'NO'} |",
              f"| E3a pair concordance among the 64 | CI lo > -0.01 | {ci(d['E3a'])} | {'yes' if d['E3a']['ci95'][0] > -0.01 else 'NO'} |",
              f"| E3b PDMS of the pick among the 64 | CI lo > -2.0 | {ci(d['E3b'], 2)} | {'yes' if d['E3b']['ci95'][0] > -2.0 else 'NO'} |",
              f"| E2 PDMS of the pick, 64 + extras | CI lo > -2.0 | {ci(d['E2'], 2)} | {'yes' if d['E2']['ci95'][0] > -2.0 else 'NO'} |",
              f"| REFUTED iff E1 CI hi < 0 | -- | {d['E1']['ci95'][1]:+.4f} | {'REFUTED' if d['E1']['ci95'][1] < 0 else 'no'} |", "",
              f"Estimator: {R.get('estimator')}; answers {R.get('variance_answered')}; n = {R.get('n_tokens')} tokens, "
              f"{R.get('n_logs')} logs; estimator self-test {R.get('estimator_selftest')}.", ""]
    L += ["## Gates", "", "| gate | pass | artifact |", "|---|---|---|"]
    gs = R.get("gates", {})
    for g, f in (("G-R1", "selftest_repair.json"), ("G-R2", "gate_gr2.json"), ("G-R3", "gate_gr34.json"),
                 ("G-R4", "gate_gr34.json"), ("G-R5", "gate_gr5.json"), ("G-R6", "gate_gr6_eval.json")):
        L.append(f"| {g} | {gs.get(g, 'n/a')} | `eval/raw/m5r/{f}` |")
    g34 = G.get("G-R3/4") or {}
    if g34 and (g34.get("G_R3") or {}).get("navsim"):
        L += ["", f"G-R3 NAVSIM agreement on repaired plans: DAC {g34['G_R3']['navsim']['DAC']['agree']}, comfort "
                  f"{g34['G_R3']['navsim']['C']['agree']} over {g34['G_R3']['navsim']['DAC']['n']} slots. G-R4 collision "
                  f"(copies vs originals): {json.dumps(g34['G_R4'])}. Label-ranking ceiling on the extras: "
                  f"{json.dumps(g34.get('label_ranking_ceiling_extras'))}."]
    if R.get("token_means"):
        L += ["", "## Arm token means", "", "| model | E1 masked | E1 plain | E3a | E3b | E2 |", "|---|---|---|---|---|---|"]
        for k, x in R["token_means"].items():
            L.append(f"| {k} | {x['E1_masked']:.4f} | {x['E1_plain']:.4f} | {x['E3a']:.4f} | {x['E3b']:.2f} | {x['E2']:.2f} |")
        L += ["", "## All paired contrasts (reported; only V4r - V3 gates)", "", "| metric | pair | diff [95 %] |", "|---|---|---|"]
        for key, pr in R.get("comparisons", {}).items():
            for pn, x in pr.items():
                L.append(f"| {key} | {pn} | {ci(x, 2 if key in ('E3b', 'E2') else 4)} |")
        L += ["", f"Seed floor (E1): {json.dumps(R.get('seed_floor_E1'))}"]
    ps = R.get("pick_subscores_x100")
    if ps:
        L += ["", "## EP and the NAVSIM sub-scores of the picks (x100; order NC, DAC, EP, TTC, C, DDC)", ""]
        for nm, r in ps["arm_means"].items():
            L.append(f"* {nm}: " + "; ".join(f"{route} {vals}" for route, vals in r.items()))
        L += ["", "| route | pair | EP [95 %] | NC | DAC | TTC | C |", "|---|---|---|---|---|---|---|"]
        for route, pr in ps["pairs"].items():
            for pn, x in pr.items():
                L.append(f"| {route} | {pn} | " + " | ".join(f"{x[h]['diff']:+.2f} [{x[h]['ci95'][0]:+.2f}, {x[h]['ci95'][1]:+.2f}]"
                                                             for h in ("EP", "NC", "DAC", "TTC", "C")) + " |")
        L += ["", f"E2 selection mix: {json.dumps(R.get('E2_selection_mix'))}"]
    if F.get("blocks"):
        L += ["", "## Four families (families6.py on the REPAIRED executed picks; per-model blocks, not paired)", "",
              "| pick | speed MAE | progress ratio | heading MAE | cross MAE | goal-point err | long. kappa | lat. kappa | strategic |",
              "|---|---|---|---|---|---|---|---|---|"]
        for tag, b in F["blocks"].items():
            if "status" in b:
                L.append(f"| {tag} | {b['status']} |||||||| ")
                continue
            lo, la, ta = b["longitudinal"], b["lateral"], b["tactical"]
            L.append(f"| {tag} | {lo.get('speed_mae_mps')} | {lo.get('progress_ratio_mean')} | {la.get('heading_mae_deg')} | "
                     f"{la.get('cross_mae_m')} | {ta.get('goal_point_error_m')} | {ta.get('longitudinal_decision_kappa')} | "
                     f"{ta.get('lateral_decision_kappa')} | {b.get('strategic')} |")
    body = "\n".join(L) + "\n"
    old = open(OUTP, encoding="utf-8").read() if os.path.exists(OUTP) else ""
    hand = old.split(MARK, 1)[1] if MARK in old else "\n## Interpretation, declared departures, what remains\n\n(to be written)\n"
    open(OUTP, "w", encoding="utf-8", newline="\n").write(body + "\n" + MARK + hand)
    print(f"ZZM5R_RESULT_WRITTEN {OUTP} verdict {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
