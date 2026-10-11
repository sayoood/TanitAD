#!/usr/bin/env python3
"""Render P5's raw JSON as the Markdown tables of eval/RESULT_P5.md, so no number is hand-transcribed.

  python eval/p5_render.py --labels raw/2026-10-11-p5-target-labels/p5_labels.json \
                           --families raw/2026-10-11-p5-target-labels/p5_families.json > tables.md
"""
from __future__ import annotations

import argparse
import json
import sys

REFS = ["T", "Tc20", "Tc8", "Tw", "ident", "H", "Hc"]


def f(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    try:
        return f"{float(x):+.{nd}f}" if nd and float(x) < 0 else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def pct(x):
    return "—" if x is None else f"{100 * float(x):.2f} %"


def ci(c, nd=4, rate=False):
    if rate:
        return f"{100 * c['delta']:+.2f} pp [{100 * c['lo']:+.2f}, {100 * c['hi']:+.2f}]"
    return f"{c['delta']:+.{nd}f} [{c['lo']:+.{nd}f}, {c['hi']:+.{nd}f}]"


def dig(d, path, default=None):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def families_rows(fam):
    out = ["| pair (executed vs intended) | speed MAE m/s | along MAE m | accel MAE m/s² | heading MAE ° | yaw-rate MAE °/s "
           "| curvature MAE 1/m | cross MAE m | lat-decision κ | lon-decision κ | goal-point err m | STRATEGIC |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, b in fam["families"].items():
        if b.get("status") == "RAISED":
            out.append(f"| {name} | RAISED: {b['error']} |" + " |" * 10)
            continue

        def comp(family, key, nd):
            v = dig(b, f"{family}.{key}")
            c = dig(b, f"{family}.ci.components.{key}.ci95")
            return "—" if v is None else (f"{v:.{nd}f} ± {c:.{nd}f}" if c is not None else f"{v:.{nd}f}")

        def tac(dec, key, nd=3):
            c = dig(b, f"tactical.{dec}.ci.{key}")
            return "—" if not c else f"{c['mean']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}]"

        out.append("| " + " | ".join([
            name, comp("longitudinal", "speed_mae_mps", 3), comp("longitudinal", "along_mae_m", 3),
            comp("longitudinal", "accel_mae_mps2", 3), comp("lateral", "heading_mae_deg", 3),
            comp("lateral", "yaw_rate_mae_degps", 3), comp("lateral", "curvature_mae_1pm", 4),
            comp("lateral", "cross_mae_m", 3), tac("lateral_decision", "kappa"), tac("longitudinal_decision", "kappa"),
            tac("goal_setting", "goal_point_error_m"), str(dig(b, "strategic.status"))]) + " |")
    return out


def dual_compliance(rows):
    """sim8 AND interp8, counted from the per-row records (descriptive; True = violation in both flags)."""
    if not rows:
        return []
    n = len(rows)
    out = ["", "**Dual compliance: executed path passes sim8 AND the drawn 8 poses pass interp8** (descriptive, "
           "counted from `rows`)", "", "| ref | n | sim8 fail | interp8 fail | fail both | pass both |",
           "|---|---|---|---|---|---|"]

    def line(name, recs):
        s = [bool(x["sim8"]) for x in recs]
        i = [bool(x["interp8"]) for x in recs]
        ok = sum(not a and not b for a, b in zip(s, i))
        return (f"| {name} | {len(recs)} | {sum(s)} | {sum(i)} | {sum(a and b for a, b in zip(s, i))} | "
                f"{ok} ({pct(ok / len(recs))}) |")

    for r in [x for x in REFS if x in rows[0]]:
        out.append(line(r, [x[r] for x in rows]))
    # DERIVED post hoc, not pre-registered: compensate only the rows whose plain label fails sim8
    sel = [bool(x["T"]["sim8"]) for x in rows]
    out.append(line("Tsel (DERIVED: Tc20 where T fails sim8, else T)",
                    [x["Tc20"] if f else x["T"] for x, f in zip(rows, sel)]))
    ci8 = [bool(x["Tc20"]["interp8"]) for x in rows]
    new = sum(c and not bool(x["T"]["interp8"]) for c, x in zip(ci8, rows))
    on_sel = sum(c and f for c, f in zip(ci8, sel))
    out += ["", f"Tc20 interp8 failures: {sum(ci8)}; on rows where T's interp8 passes: {new}; on rows where T fails "
            f"sim8: {on_sel}."]
    k = sum(sel)
    if k:
        def med(v):
            v = sorted(v)
            return 0.5 * (v[(len(v) - 1) // 2] + v[len(v) // 2])
        la = [med([x["learn"][lab][j] for x, f in zip(rows, sel) if f]) for lab in ("t20", "Tc20") for j in (0, 1)]
        out.append(f"Learnability on the {k} rows Tsel compensates (median per-row max): |acc| t20 {la[0]:.3f} vs "
                   f"Tc20 {la[2]:.3f} m/s² ({la[2] / la[0]:.2f}×); |heading rate| t20 {la[1]:.3f} vs Tc20 "
                   f"{la[3]:.3f} °/s ({la[3] / la[1]:.2f}×).")
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--families", default=None)
    a = ap.parse_args()
    d = json.load(open(a.labels, encoding="utf-8"))
    L = [f"n = {d['n']} rows, {d['n_logs']} logs, {d['n_errors']} rows raised; {d['seconds']:.0f} s.", ""]

    L += ["**Controls (§4)**", "", "| control | result |", "|---|---|"]
    L += [f"| {k} | {'PASS' if v else '**FAIL**'} |" for k, v in d["gates"].items()]
    L += ["", "raw: " + ", ".join(f"{k} {v}" for k, v in d["controls_raw"].items()), ""]

    L += ["**Per reference, scored on its executed trajectory (§3)**", "",
          "| ref | intended | n | sim8 fail | interp8 fail | DDC mean | DDC < 1 | lk10 violation | comfort | path-length ratio (median) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in REFS:
        p = d["per_ref"][r]
        L.append(f"| {r} | {p['intended']} | {p['n']} | {pct(p['sim8_rate'])} | {pct(p['interp8_rate'])} | "
                 f"{p['ddc_mean']:.4f} | {pct(p['ddc_lt1_rate'])} | {pct(p['lk10_rate'])} | {pct(p['comfort_rate'])} | "
                 f"{f(dig(p, 'path_len_ratio.median'))} |")
    keys = list(d["per_ref"]["T"]["exec_vs_intended"].keys())
    stats = list(d["per_ref"]["T"]["exec_vs_intended"][keys[0]].keys())
    L += ["", f"**Executed vs intended** ({' / '.join(stats)})", "",
          "| ref | intended | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for r in REFS:
        e = d["per_ref"][r]["exec_vs_intended"]
        L.append(f"| {r} | {d['per_ref'][r]['intended']} | " +
                 " | ".join(" / ".join(f(e[k][s], 3) for s in stats) for k in keys) + " |")
    L += ["", "**Executed vs H (descriptive)**", "", "| ref | " + " | ".join(d["per_ref"]["T"]["exec_vs_H"]) + " |",
          "|---|" + "---|" * len(d["per_ref"]["T"]["exec_vs_H"])]
    for r in REFS:
        if "exec_vs_H" in d["per_ref"][r]:
            L.append(f"| {r} | " + " | ".join(f(v, 3) for v in d["per_ref"][r]["exec_vs_H"].values()) + " |")

    for title, key in (("Arm (ii): Tc20 − T, paired log-cluster bootstrap (§6)", "arm_ii"),
                       ("Control 3, wrong sign: Tw − T (same estimator)", "wrong_sign_Tw")):
        A = d[key]
        L += ["", f"**{title}** — verdict **{A['verdict']}**; adverse guards: "
              f"{', '.join(A['adverse_guards']) if A['adverse_guards'] else 'none'}", "",
              "| metric | Δ [95 % CI] |", "|---|---|"]
        for m, c in A["cis"].items():
            L.append(f"| {m} | {ci(c, 4, rate=c['kind'] == 'ratio')} |")
    L += ["", f"**Learnability** (flag: **{d['arm_ii']['learnability_flag']}**; RISK iff a Tc20 median > 2× t20's)", "",
          "| label | max acc median | max acc p95 | max heading-rate median °/s | p95 °/s |", "|---|---|---|---|---|"]
    for r, v in d["learnability"].items():
        L.append(f"| {r} | {v['max_acc_mps2']['median']:.3f} | {v['max_acc_mps2']['p95']:.3f} | "
                 f"{v['max_heading_rate_dps']['median']:.3f} | {v['max_heading_rate_dps']['p95']:.3f} |")
    L += ["", "**D1 class breakdown of what Tc20 repairs and breaks**", "",
          "| D1 class | n | T sim8 fail | Tc20 sim8 fail | repaired | broken |", "|---|---|---|---|---|---|"]
    for c, v in d["d1_class_breakdown"].items():
        L.append(f"| {c} | {v['n']} | {v['T_fail']} | {v['Tc20_fail']} | {v['repaired']} | {v['broken']} |")
    ai = d["arm_i"]
    a6 = ai["at_0.6"]
    L += ["", f"**Arm (i): margin < 0.6 m on T's plan footprint** — verdict **{ai['verdict']}**", "",
          f"- at 0.6 m: flagged {a6['flagged']} ({pct(a6['flagged_share'])}), recall {pct(a6['recall'])} "
          f"({a6['caught']} of {ai['n_sim8_fail_T']} sim8 failures), precision {pct(a6['precision'])}",
          f"- drop-if-sim8 comparison point: flagged {pct(ai['drop_if_sim8']['flagged_share'])}, recall "
          f"{pct(ai['drop_if_sim8']['recall'])}",
          f"- margin of T: median {ai['margin_T']['median']:.3f} m, p05 {ai['margin_T']['p05']:.3f} m; "
          f"H median {ai['margin_H']['median']:.3f} m (descriptive)", "",
          "| threshold m (descriptive sweep) | flagged share | recall | precision |", "|---|---|---|---|"]
    for s in ai["sweep_descriptive"]:
        L.append(f"| {s['threshold_m']} | {pct(s['flagged_share'])} | {pct(s['recall'])} | {pct(s['precision'])} |")
    L += ["", "**Iteration traces** (median max-xy deviation, m, k = 0..3)", "", "| ref | " +
          " | ".join(f"k={k}" for k in range(len(d["iteration_trace_median_max_xy"]["Tc20"]))) + " |",
          "|---|" + "---|" * len(d["iteration_trace_median_max_xy"]["Tc20"])]
    for r, v in d["iteration_trace_median_max_xy"].items():
        L.append(f"| {r} | " + " | ".join(f"{x:.4f}" for x in v) + " |")
    t = d["tc20_minus_tc8_max_m"]
    L += ["", f"`max |to_navsim(Tc20) − Tc8|`: median {t['median']:.4f} m, max {t['max']:.4f} m."]
    L += dual_compliance(d.get("rows") or [])
    if a.families:
        fam = json.load(open(a.families, encoding="utf-8"))
        L += ["", f"**Four families** (tier {fam['tier']}, n {fam['n']}, {fam['n_logs']} logs, value ± 95 % half-width; "
              f"descriptive only). STRATEGIC: {fam['strategic']}.", ""]
        L += families_rows(fam)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
