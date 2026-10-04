"""Generate the Stage-2 tables (raw/TABLES_s2_generated.md) from the raw JSONs — no hand transcription."""
import json
import sys

RAW = sys.argv[1]
E = json.load(open(f"{RAW}/s2_echo_leak.json"))
LEV = {f"{p}_{i}": json.load(open(f"{RAW}/s2_levers_{p}_{i}.json")) for p in ("eval139", "trainS") for i in ("ols", "hgb1", "hgb2")}
V = ("A30", "A50", "A80", "B")
L = []


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def ci(c):
    return f"[{f(c[0])}, {f(c[1])}]" if c and c[0] is not None else ""


L.append("## E1 — trivial planner floor (constant v0, arc through the checkpoint)\n")
for pop, title in (("evaldiag", "EVAL-DIAG (route package grid; bank GT; windows where all four variants are valid)"),
                   ("all_windows", "all eval139 windows (log GT at 6 s, all variants valid)")):
    S = E["E1"][pop]
    L.append(f"### {title} — n: {S['n']}\n")
    L.append("| arm | turn ADE m | turn dir-correct | turn heading ≤15° | straight ADE m | straight heading ≤15° | all ADE m |")
    L.append("|---|---|---|---|---|---|---|")
    for a, row in S["arms"].items():
        t, s, al = row.get("turn", {}), row.get("straight", {}), row.get("all_classified", {})
        L.append(f"| {a} | {f(t['ade'][0])} {ci(t['ade'][1])} | {f(t['dir'][0])} {ci(t['dir'][1])} | {f(t['h15'][0])} {ci(t['h15'][1])} | "
                 f"{f(s['ade'][0])} | {f(s['h15'][0])} | {f(al['ade'][0])} |")
    L.append("")
L.append(f"GT cross-check (log vs bank, EVAL-DIAG): {json.dumps(E['E1']['gt_crosscheck_log_vs_bank'])}\n")
L.append("## E2 — speed leak: recovered fraction ρ of the future-speed information v0 lacks (mean over τ = 1..6 s)\n")
L.append("| population · instrument | certified | ρ NAV | " + " | ".join(f"RC {v} / NAV+RC−NAV" for v in V) + " | ρ deranged | ρ O1 oracle | ρ target |")
L.append("|---|---|---|" + "---|" * len(V) + "---|---|---|")
for k, R in LEV.items():
    rho = R["rho"]
    cells = [f"{rho['REG_s8_' + v]:+.3f} / {rho['NAV_REG_s8_' + v] - rho['NAV']:+.3f} {'PASS' if R['bar']['REG_s8_' + v]['pass'] else 'FAIL'}" for v in V]
    L.append(f"| {k} (n {R['n_rows']}, {R['n_clips']} clips) | {R['certified']} | {rho['NAV']:+.3f} | " + " | ".join(cells)
             + f" | {rho['CTRL_RC_A50_deranged']:+.3f} | {rho['CTRL_O1_oracle']:+.3f} | {rho['CTRL_target_itself']:+.3f} |")
L.append("\n### E2 levers (SPEC E4 order) — NAV+RC−NAV, trainS, both tree instruments (the decisive population)\n")
L.append("| arm | " + " | ".join(f"{v} hgb1 / hgb2" for v in V) + " |")
L.append("|---|" + "---|" * len(V))
for arm in ("REG_s8", "L1_s15", "L2_s8_noheading", "L3_s8_noised", "DIAG_L1L2_s15_noheading"):
    cells = []
    for v in V:
        a, b = LEV["trainS_hgb1"]["bar"][f"{arm}_{v}"], LEV["trainS_hgb2"]["bar"][f"{arm}_{v}"]
        cells.append(f"{a['nav_plus_minus_nav']:+.3f} / {b['nav_plus_minus_nav']:+.3f}")
    L.append(f"| {arm} | " + " | ".join(cells) + " |")
a, b = LEV["trainS_hgb1"]["bar"]["DIAG_heavy_s25_A50"], LEV["trainS_hgb2"]["bar"]["DIAG_heavy_s25_A50"]
L.append(f"| DIAG heavy σ25 (A50 only) | — | {a['nav_plus_minus_nav']:+.3f} / {b['nav_plus_minus_nav']:+.3f} | — | — |")
L.append("\n## E3 — lateral leak |δ| of the checkpoint from the σ = 25 m route at the same arc (m)\n")
L.append("| population | variant | all p90 (registered bar ≤ 1.0) | straight-support p90 (A1 bar ≤ 1.0) / share | curved-support p90 | straight: LC-scale vs none, median | dev6 ΔR² (input over heavy route) |")
L.append("|---|---|---|---|---|---|---|")
for pop in ("eval139", "trainS"):
    for v in V:
        r = E["E3"][pop]["variants"][v]
        d6 = r.get("dev6_oof_r2", {})
        L.append(f"| {pop} | {v} | {f(r['all']['p90'])} {'PASS' if r['E3_as_registered_pass'] else 'FAIL'} | "
                 f"{f(r['straight_support']['p90'])} {'PASS' if r['E3_A1_pass'] else 'FAIL'} / {f(r['straight_support']['share_of_valid'], 2)} | "
                 f"{f(r['curved_support']['p90'])} | {f(r['straight_support_lc_scale_excursion']['median'])} vs {f(r['straight_support_no_lc_excursion']['median'])} | "
                 f"{f(d6.get('delta'), 4)} |")
L.append("\n## E4 — decision (rule fixed in SPEC §9.E + addendum S2-A1)\n")
L.append("```\n" + json.dumps(E["E4"], indent=1) + "\n```")
open(f"{RAW}/TABLES_s2_generated.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("written", len(L), "lines")
