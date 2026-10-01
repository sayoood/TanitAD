#!/usr/bin/env python3
"""Measure 5: writes the NUMBERS of eval/RESULT_M5_EFFECTIVENESS.md from the banked artifacts -- no number is typed.
Reads eval/raw/m5_effectiveness/{readout,families,a0_reference,gpu_chain_times,repaired_extras}.json and
<m5>/{ft_data.json,label_summary.json,ft/trainlog_s*.json,evals/gates.json}. The interpretation section is written by
hand below the marker line and is preserved when this script re-runs.

    python eval/m5_write_result.py [--m5 <data root>] [--out <md>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw", "m5_effectiveness")
M5 = "D:/Projects/TanitAD/data/refe_m5"
MARK = "<!-- HAND-WRITTEN BELOW THIS LINE (kept on re-run) -->"


def j(p, default=None):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return default


def ci(x, nd=4):
    if not x:
        return "n/a"
    d, (lo, hi) = x["diff"], x["ci95"]
    sep = "yes" if (lo > 0 or hi < 0) else "no"
    return f"{d:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}] | {sep}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m5", default=M5)
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default=os.path.join(HERE, "RESULT_M5_EFFECTIVENESS.md"))
    a = ap.parse_args()
    R = j(os.path.join(a.raw, "readout.json"), {})
    F = j(os.path.join(a.raw, "families.json"), {})
    A0R = j(os.path.join(a.raw, "a0_reference.json"), {})
    TM = j(os.path.join(a.raw, "gpu_chain_times.json"), {})
    FD = j(os.path.join(a.m5, "ft_data.json"), {})
    LS = j(os.path.join(a.m5, "label_summary.json"), {})
    L = []
    w = L.append
    dec = R.get("decision", {})
    w("# RESULT -- Measure 5 effectiveness (label_version 4 slow-plan labels; scorer-only dev-box fine-tune)")
    w("")
    w("*Pre-registration: `eval/PREREG_MEASURE5.md`, blob `c668b5b6` (SPEC_PREREG_HASH, 2026-09-27 13:22 Berlin), "
      "criteria, arms, readout and decision rule applied AS WRITTEN. Numbers in the tables below are written by "
      "`eval/m5_write_result.py` from the banked artifacts; none is typed.*")
    w("")
    w(f"## Verdict (as registered): **{dec.get('verdict', 'NOT YET READ')}**")
    w("")
    if dec:
        w("| criterion (§5) | bar | measured | met |")
        w("|---|---|---|---|")
        e1, e3a, e3b, fl = dec["E1"], dec["E3a"], dec["E3b"], dec["seed_floor"]
        w(f"| E1 V4 - V3 (masked slow-plan pair concordance) | >= +0.03 | {e1['diff']:+.4f} | {'yes' if e1['diff'] >= 0.03 else 'NO'} |")
        w(f"| E1 CI lower bound | > 0 | {e1['ci95'][0]:+.4f} (CI [{e1['ci95'][0]:+.4f}, {e1['ci95'][1]:+.4f}]) | {'yes' if e1['ci95'][0] > 0 else 'NO'} |")
        w(f"| E1 effect > the seed floor | > {fl:.4f} | {e1['diff']:+.4f} | {'yes' if e1['diff'] > fl else 'NO'} |")
        w(f"| E3a V4 - V3 CI lower bound (pair concordance among the 64) | > -0.01 | {e3a['ci95'][0]:+.4f} (diff {e3a['diff']:+.4f}) | {'yes' if e3a['ci95'][0] > -0.01 else 'NO'} |")
        w(f"| E3b V4 - V3 CI lower bound (PDMS of the pick among the 64) | > -2.0 | {e3b['ci95'][0]:+.2f} (diff {e3b['diff']:+.2f}) | {'yes' if e3b['ci95'][0] > -2.0 else 'NO'} |")
        w(f"| REFUTED iff E1 CI upper bound < 0 | -- | {e1['ci95'][1]:+.4f} | {'REFUTED' if e1['ci95'][1] < 0 else 'not refuted'} |")
        w("")
        w(f"Estimator: {R.get('estimator')}. Variance answered: {R.get('variance_answered')}. "
          f"n = {R.get('n_tokens')} tokens over {R.get('n_logs')} logs; seeds {R.get('seeds')}. "
          f"Estimator self-test (reproduces snapshot_pair_under_rule's published ep014 -> ep015 interval): "
          f"{R.get('estimator_selftest')}.")
        w("")
    # gates
    g = R.get("gates", {})
    if g:
        w("## Gates (all must pass before the statistic is read)")
        w("")
        w("| gate | value | bar |")
        w("|---|---|---|")
        w(f"| the 8 copies scored by the model == the harness-scored copies | max abs {g.get('copies_equal_harness_scored_max_abs')} | 0.0 |")
        w(f"| harness rows missing / invalid for the extras | {g.get('harness_missing')} | 0 |")
        w(f"| A0 (re-run on the new cache) vs the E-6 table's logits | max abs {g.get('A0_pure_vs_table_logits_max_abs')} | <= 1e-3 |")
        w(f"| A0's pick == the table's pick | {g.get('A0_pick_equals_table')} / {R.get('n_tokens')} | all |")
        w(f"| masked route leaves the 64's scores unchanged | max abs {g.get('masked_64_vs_pure_max_abs')} | <= 1e-4 |")
        tm = R.get("token_means", {}).get("A0", {})
        if A0R and tm:
            w(f"| ADDED control: A0 vs the INDEPENDENT reference (slow-copy probe GPU dump, loop concordance; "
              f"`a0_reference.json`) | E1 {tm.get('E1_masked')} vs {A0R['E1_masked_token_mean']:.5f}; E3a "
              f"{tm.get('E3a')} vs {A0R['E3a_token_mean']:.5f}; E3b {tm.get('E3b')} vs {A0R['E3b_pdms_x100']:.3f}; E2 "
              f"{tm.get('E2')} vs {A0R['E2_pdms_x100']:.3f} | equal to rounding |")
        w("")
    # arm means
    tm = R.get("token_means", {})
    if tm:
        w("## Arm token means (every model)")
        w("")
        w("| model | E1 masked (PRIMARY) | E1 plain | E3a | E3b PDMS | E2 PDMS |")
        w("|---|---|---|---|---|---|")
        for k, v in tm.items():
            w(f"| {k} | {v['E1_masked']:.4f} | {v['E1_plain']:.4f} | {v['E3a']:.4f} | {v['E3b']:.2f} | {v['E2']:.2f} |")
        w("")
        w("## Paired comparisons (arm = seed mean per token; paired log-cluster bootstrap)")
        w("")
        w("| metric | pair | diff [95 %] | separated |")
        w("|---|---|---|---|")
        for key, pr in R.get("comparisons", {}).items():
            for pair, x in pr.items():
                w(f"| {key} | {pair} | {ci(x, 2 if key in ('E3b', 'E2') else 4)} |")
        w("")
        sf = R.get("seed_floor_E1", {})
        w(f"Seed floor (E1 masked, largest |seed_i - seed_j| of the token mean within an arm): **{sf.get('floor')}** "
          f"-- {json.dumps(sf.get('by_arm'))}.")
        w("")
    # EP prominently
    if F.get("pairs"):
        w("## Ego progress (EP) and the other harness sub-scores of the PICK")
        w("")
        w("Harness sub-scores x100 of each arm's pick (arm = seed mean per token). E3b = the argmax among the 64 (the "
          "shipped route); E2 = the argmax among the 64 + 9 extras (masked route). `eval/raw/m5_effectiveness/families.json`.")
        w("")
        for route in ("E3b", "E2"):
            am = F["arm_means"][route]
            w(f"**{route}** arm means: " + "; ".join(f"{nm} EP {v['EP']:.2f} / PDMS {v['PDMS']:.2f}" for nm, v in am.items()))
            w("")
            w("| pair | " + " | ".join(("NC", "DAC", "EP", "TTC", "C", "DDC", "PDMS")) + " |")
            w("|---|" + "---|" * 7)
            for pair, x in F["pairs"][route].items():
                w(f"| {route} {pair} | " + " | ".join(
                    f"{x[c]['diff']:+.2f} [{x[c]['ci95'][0]:+.2f}, {x[c]['ci95'][1]:+.2f}]" for c in
                    ("NC", "DAC", "EP", "TTC", "C", "DDC", "PDMS")) + " |")
            w("")
            w(f"Seed floor per column ({route}): {json.dumps(F.get('seed_floor', {}).get(route))}")
            w("")
        w(f"E2 selection mix (original / 0.75x copy / STOP) per model: {json.dumps(F.get('E2_selection_mix'))}")
        w("")
    # families
    f6 = F.get("families6", {})
    if f6.get("arm_seed_means"):
        w("## The four metric families (families6.py, per-model blocks vs the logged human future; seed means)")
        w("")
        w("| route | model | long: speed MAE m/s | long: progress ratio | long: along MAE m | lat: heading MAE deg | "
          "lat: cross MAE m | lat: curvature MAE 1/m | tac: goal-point err m | tac: long. kappa | tac: lat. kappa | strategic |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for route, arms in f6["arm_seed_means"].items():
            for nm, b in arms.items():
                lo, la, ta = b["longitudinal"], b["lateral"], b["tactical"]
                w(f"| {route} | {nm} | {lo['speed_mae_mps']} | {lo['progress_ratio_mean']} | {lo['along_mae_m']} | "
                  f"{la['heading_mae_deg']} | {la['cross_mae_m']} | {la['curvature_mae_1pm']} | {ta['goal_point_error_m']} | "
                  f"{ta['longitudinal_decision_kappa']} | {ta['lateral_decision_kappa']} | {b.get('strategic')} |")
        w("")
        w(f"Amendment 6 family guard, each arm vs the shipped pick (A0, E3b): "
          f"{json.dumps(f6.get('family_guard_vs_shipped', {}).get('result'))}. Control: A0's rebuilt pick vs the "
          f"landed seam max |d| = {f6.get('control_A0_E3b_pick_vs_landed_seam_max_abs_m')} m.")
        w("")
    # E4
    e4 = R.get("E4_reported")
    if e4:
        w("## E4 (reported): predicted NC / TTC / EP probability vs the harness means")
        w("")
        w(f"Harness: STOP {e4['harness_stop_NC_TTC_EP']}, copies {e4['harness_copies_NC_TTC_EP']}.")
        w("")
        w("| model | STOP predicted NC / TTC / EP | copies predicted NC / TTC / EP |")
        w("|---|---|---|")
        for k, v in e4["models"].items():
            w(f"| {k} | {v['pred_stop_NC_TTC_EP']} | {v['pred_copies_NC_TTC_EP']} |")
        w("")
    # val BCE
    logs = {int(os.path.basename(p)[10:-5]): j(p) for p in glob.glob(os.path.join(a.m5, "ft", "trainlog_s*.json"))}
    if logs:
        w("## Fine-tune record and navtrain-val BCE per component (reported)")
        w("")
        for s, lg in sorted(logs.items()):
            w(f"* seed {s}: device {lg['device']}, bf16 autocast {lg['amp_bf16']}, lr {lg['lr']}, wd {lg['wd']}, score "
              f"weight {lg['score_w']}, batch {lg['batch']}, epochs {lg['epochs']}, N {lg['N']}, steps {len(lg['steps'])}, "
              f"{lg['seconds']} s; batch-order sha {lg['order_sha']}")
            for vb in lg.get("val_bce", []):
                w(f"  * epoch {vb['epoch']}: " + "; ".join(
                    f"{arm} pure {vb[arm]['pure']} copies {vb[arm]['copies']}" for arm in lg["arms"]))
        w("")
    if FD or LS:
        w("## Data")
        w("")
        w(f"Labels (`{a.m5}/label_summary.json`): {json.dumps(LS)}. Fine-tune data (`ft_data.json`): {json.dumps(FD)}.")
        w("")
    p3 = F.get("posthoc_repaired_truth", {})
    if p3.get("arm_means"):
        w("## POST HOC, NOT GATING: the same metrics against the REPAIRED harness truth (Amendment 7)")
        w("")
        w("| metric | A0 | V3 | V4 | V4all | V4 - V3 [95 %] | V3 - A0 [95 %] |")
        w("|---|---|---|---|---|---|---|")
        for key, am in p3["arm_means"].items():
            pr = p3["pairs"][key]
            w(f"| {key} | {am['A0']} | {am['V3']} | {am['V4']} | {am['V4all']} | {ci(pr['V4_minus_V3'])} | "
              f"{ci(pr['V3_minus_A0'])} |".replace("| yes |", "(separated) |").replace("| no |", "|"))
        w("")
    if TM:
        w("## GPU stage record (`gpu_chain_times.json`)")
        w("")
        w("```")
        w(json.dumps(TM, indent=1))
        w("```")
        w("")
    body = "\n".join(L) + "\n"
    old = open(a.out, encoding="utf-8").read() if os.path.exists(a.out) else ""
    hand = old.split(MARK, 1)[1] if MARK in old else "\n## Interpretation, what remains, next lever\n\n(to be written)\n"
    open(a.out, "w", encoding="utf-8", newline="\n").write(body + "\n" + MARK + hand)
    print(f"ZZM5_RESULT_WRITTEN {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
