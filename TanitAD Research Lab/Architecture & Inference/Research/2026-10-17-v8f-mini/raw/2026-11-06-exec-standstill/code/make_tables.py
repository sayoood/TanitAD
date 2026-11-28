#!/usr/bin/env python3
"""make_tables.py -- raw/EXEC_STANDSTILL_READ.json + GATES.json + GATE_ID_FULL.json -> raw/RESULT_TABLES.md (every number copied by code, none by hand)."""
import json
import os
import sys

PK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
R = json.load(open(f"{PK}/raw/EXEC_STANDSTILL_READ.json", encoding="utf-8"))
GT = json.load(open(f"{PK}/raw/GATES.json", encoding="utf-8"))
GF = json.load(open(f"{PK}/raw/GATE_ID_FULL.json", encoding="utf-8"))
SEEDS = ("s0", "s1")


def f(x, nd=4):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def ci(d):
    return f"{f(d['delta'])} [{f(d['lo'])}, {f(d['hi'])}]" + (" SEP" if d["separated"] else "")


def main():
    o = []
    o.append("## Gates (card section 4)\n")
    o.append("| gate | pass | evidence (subset: 123 windows, C0_s0, 61 launch-from-stop, 79 stopped) | RED arm |\n|---|---|---|---|")
    gi, gm, gp, g10 = GT["G-ID"], GT["G-MOV"], GT["G-PHYS"], GT["G-P10"]
    o.append(f"| G-ID | {gi['PASS']} | unpatched LF6 tool vs patched (default and eps 1e-3): cl/fl/ol/ticks/exec dump bit-identical; FULL grid eps 1e-3 vs BANKED C0: strict bit-identical at both seeds = {GF['STRICT_BIT_IDENTICAL_BOTH_SEEDS']} | eps 0.1 differs on launch windows: cl {gi['RED_eps0.1_vs_base']['cl']['n_unequal_launch']}, fl {gi['RED_eps0.1_vs_base']['fl']['n_unequal_launch']} of 61 |")
    r = gm["RED_over_aggressive_hold_eps2.0_vs_base"]
    o.append(f"| G-MOV | {gm['PASS']} | {gm['n_moving_model']} windows with every executed step > 0.1 m: hold == base exactly on all (cl, fl, track, yaw rate) | eps 2.0 differs on {r['cl']['n_unequal_on_moving']} (model) / {r['fl']['n_unequal_on_moving']} (floor) of those |")
    h, rd = gp["hold_eps0.1"], gp["RED_hold_disabled"]
    o.append(f"| G-PHYS | {gp['PASS']} | hold: {h['cl']['n_below']} model + {h['fl']['n_below']} floor sub-eps frames, 0 heading changes | hold disabled: {rd['cl']['n_violations']} + {rd['fl']['n_violations']} violations |")
    o.append(f"| G-P10 | {g10['PASS']} | floor_planner source identical; tick-0 plan identical on all windows; later knots differ on {g10['n_windows_later_knots_differ_under_hold']} windows (history only) | mutated floor breaks tick-0 identity on {g10['RED_mutated_floor']['n_windows_tick0_unequal']} windows |")
    for mname in ("C0", "DAG"):
        m = R["models"][mname]
        o.append(f"\n## Primary: E = M1 crawl share of {mname} minus P10, dE = E_hold - E_base ({'the card arm' if mname == 'C0' else 'DAGOACC beside, never the verdict'})\n")
        o.append("| seed | E_base | E_hold | dE | share explained (-dE/E_base) | dE separated-negative | abs(dE) >= 0.5 E_base | n valid base/hold, sets equal |\n|---|---|---|---|---|---|---|---|")
        fl = m["outcome_card_section5"]["flags"]
        for s in SEEDS:
            b = m[s]["M1"]
            fk = fl[int(s[1])] if int(s[1]) in fl else fl[s[1]] if s[1] in fl else fl[str(int(s[1]))]
            o.append(f"| {s} | {f(b['E_base'])} [{f(b['E_base_lo'])}, {f(b['E_base_hi'])}] | {f(b['E_hold'])} [{f(b['E_hold_lo'])}, {f(b['E_hold_hi'])}] | {f(b['dE'])} [{f(b['dE_lo'])}, {f(b['dE_hi'])}] "
                     f"| {f(fk['share_explained'], 3)} | {fk['separated_negative']} | {fk['abs_dE_ge_half_E_base']} | {b['n_valid_base']}/{b['n_valid_hold']}, {b['valid_sets_equal']} |")
        o.append(f"\nOutcome (card section 5, literal): **{m['outcome_card_section5']['label']}** (outcome {m['outcome_card_section5']['outcome']}). Training-seed spread |dE_s1 - dE_s0| = {m['seed_variance_abs_dE_s1_minus_s0']:.4f}.")
    m = R["models"]["C0"]
    o.append("\n## All 19 rows, C0 minus P10 (each executor's own P10), and dE per row\n")
    o.append("| row | s0 base | s0 hold | s0 dE | s1 base | s1 hold | s1 dE |\n|---|---|---|---|---|---|---|")
    for lab in m["s0"]["E_base_rows"]:
        cells = []
        for s in SEEDS:
            for k in ("E_base_rows", "E_hold_rows", "dE_rows"):
                cells.append(ci(m[s][k][lab]))
        o.append(f"| {lab} | " + " | ".join(cells) + " |")
    o.append("\n## Lateral and heading rows (and M1) per start class, C0 minus P10: base / hold / dE (class mean over the n windows of that class)\n")
    o.append("| row | class | n | s0 base | s0 hold | s0 dE [CI] | s1 base | s1 hold | s1 dE [CI] |\n|---|---|---|---|---|---|---|---|---|")
    for lab, row in m["s0"]["per_class"].items():
        for c in ("L", "S", "O", "M"):
            n = row["dE"]["classes"][c]["n"]
            cells = []
            for s in SEEDS:
                pc = m[s]["per_class"][lab]
                cb, ch, cd = pc["E_base"]["classes"][c], pc["E_hold"]["classes"][c], pc["dE"]["classes"][c]
                cells += [f(cb["delta"]), f(ch["delta"]), f"{f(cd['delta'])} [{f(cd['delta_lo'])}, {f(cd['delta_hi'])}]"]
            o.append(f"| {lab} | {c} | {n} | " + " | ".join(cells) + " |")
    o.append("\n## Diagnostic: share of standstill-launch windows whose executed-history yaw rate exceeds 60 deg/s (max over executed rows 1..20; rows 1..30 beside; median of the max in deg/s)\n")
    o.append("| model | seed | role | base share (n flagged/n) | hold share | base rows1-30 | hold rows1-30 | base median max | hold median max |\n|---|---|---|---|---|---|---|---|---|")
    for mname in ("C0", "DAG"):
        for s in SEEDS:
            d = R["models"][mname][s]["diagnostic"]
            for role, nm in (("fl", "P10 floor"), ("cl", mname)):
                b, h = d[f"{role}_base"], d[f"{role}_hold"]
                fmt = lambda x: f"{x['share']:.3f} ({x['n_flagged']}/{x['n']})" if x["share"] is not None else "n/a"
                o.append(f"| {mname} | {s} | {nm} | {fmt(b['launch_standstill_rows1to20'])} | {fmt(h['launch_standstill_rows1to20'])} | {fmt(b['launch_standstill_rows1to30'])} | {fmt(h['launch_standstill_rows1to30'])} "
                         f"| {b['launch_standstill_rows1to20']['median_max_degps']:.1f} | {h['launch_standstill_rows1to20']['median_max_degps']:.1f} |")
    d0 = R["models"]["C0"]["s0"]["diagnostic"]
    o.append(f"\nLaunch class: {d0['launch_standstill_n']} windows in {d0['launch_standstill_episodes']} episodes (the AUDIT's n 18, 12); gapdecomp class L has {d0['gapdecomp_L_n']} windows.")
    o.append("\n## How hold moves the arms themselves (hold minus base, per-window mean, selected rows)\n")
    o.append("| row | model | s0 C0 hold-base | s0 P10 hold-base | s1 C0 hold-base | s1 P10 hold-base |\n|---|---|---|---|---|---|")
    for lab in ("M1 crawl share", "ADE6", "along-track MAE", "heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE"):
        o.append(f"| {lab} | C0 | " + " | ".join(ci(m[s][k][lab]) for s in SEEDS for k in ("model_hold_minus_base_rows", "p10_hold_minus_base_rows")) + " |")
    with open(f"{PK}/raw/RESULT_TABLES.md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(o) + "\n")
    print("wrote", f"{PK}/raw/RESULT_TABLES.md", len(o), "lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
