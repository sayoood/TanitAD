"""Render `results_matched_capacity.json` into the markdown tables the report quotes.

Every number in MATCHED_CAPACITY.md comes out of here, so a table cannot drift from the JSON by
being retyped. Run it again after any re-scoring and paste the output.

usage:  python render_tables.py results_matched_capacity.json > tables.md
"""
from __future__ import annotations

import json
import sys


def sep(d):
    return "**\\***" if d.get("separated") else ""


def ci(d, k="delta"):
    return f"{d[k]:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"


def main():
    R = json.load(open(sys.argv[1], encoding="utf-8"))
    sits = list(R["per_situation"])
    order = list(R["per_situation"][sits[0]]["arms"])
    spec = R["rungs"]

    print("## The capacity ladder\n")
    print("| rung | params/head | " + " | ".join(f"{s} AP-lift" for s in sits) + " |")
    print("|---|---:|" + "---:|" * len(sits))
    for n in order:
        cells = []
        for s in sits:
            a = R["per_situation"][s]["arms"][n]
            cells.append(f"{a['point']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]")
        print(f"| `{n}` | {spec[n]['params_per_head']:,} | " + " | ".join(cells) + " |")

    print("\n## Each rung against its OWN permuted-feature null (paired)\n")
    print("| rung | " + " | ".join(sits) + " |")
    print("|---|" + "---|" * len(sits))
    for n in order:
        cells = []
        for s in sits:
            d = R["per_situation"][s]["arms"][n]["paired_vs_own_null"]
            cells.append(ci(d) + " " + sep(d))
        print(f"| `{n}` | " + " | ".join(cells) + " |")

    print("\n## Each rung against the DEPLOYED rung (paired)\n")
    dep = R["protocol"]["deployed_rung"]
    print(f"baseline = `{dep}`\n")
    print("| rung | " + " | ".join(sits) + " |")
    print("|---|" + "---|" * len(sits))
    for n in order:
        if n == dep:
            continue
        cells = []
        for s in sits:
            d = R["per_situation"][s]["paired"].get(f"{n} - {dep}")
            cells.append((ci(d) + " " + sep(d)) if d else "-")
        print(f"| `{n}` | " + " | ".join(cells) + " |")

    print("\n## Operating point at a 5 % alarm budget - PRECISION alongside recall\n")
    print("| rung | " + " | ".join(f"{s} P / R (fires, tp)" for s in sits) + " |")
    print("|---|" + "---|" * len(sits))
    for n in order:
        cells = []
        for s in sits:
            o = R["per_situation"][s]["arms"][n]["op_5pct"]
            cells.append(f"{o['precision']:.4f} / {o['recall']:.4f} "
                         f"({o['n_alarm']:,}, {o['tp']})")
        print(f"| `{n}` | " + " | ".join(cells) + " |")

    print("\n## Controls\n")
    print("### NEG_FEATURE - AP-lift of every rung fitted on features permuted ACROSS clips\n")
    print("| rung | " + " | ".join(sits) + " |")
    print("|---|" + "---:|" * len(sits))
    for n in order:
        cells = [f"{R['controls']['NEG_FEATURE_ap_lift'][s][n]:.4f}" for s in sits]
        print(f"| `{n}` | " + " | ".join(cells) + " |")
    print("\n### NEG_LABEL - labels permuted across whole clusters (AP-lift)\n")
    print("| rung | " + " | ".join(sits) + " |")
    print("|---|" + "---:|" * len(sits))
    for n in R["controls"]["NEG_LABEL_ap_lift"][sits[0]]:
        cells = [f"{R['controls']['NEG_LABEL_ap_lift'][s][n]:.4f}" for s in sits]
        print(f"| `{n}` | " + " | ".join(cells) + " |")
    sc = R["controls"].get("SELF_CONSISTENCY_component_vs_family", {})
    print("\n### SELF-CONSISTENCY - family AP-lift vs an independent recomputation\n")
    for s, v in sc.items():
        print(f"* `{s}`: identical = **{v['identical']}** on {v['n_rows']:,} rows - "
              f"family {v['family_ap_lift']} vs recomputed {v['recomputed_independently']}")

    print("\n## Power\n")
    print("| situation | scorable rows | positives | clusters | clusters WITH a positive | base rate |")
    print("|---|---:|---:|---:|---:|---:|")
    for s in sits:
        r = R["per_situation"][s]
        print(f"| {s} | {r['n_scorable']:,} | {r['n_pos']:,} | {r['n_clusters']:,} | "
              f"{r['n_clusters_with_a_positive']:,} | {r['base_rate']:.5f} |")

    print("\n## Selected epochs (the optimisation read)\n")
    print("| rung | epochs trained | epoch* per fold |")
    print("|---|---:|---|")
    for n in order:
        sp = spec[n]
        if sp["kind"] != "CausalSitHead":
            print(f"| `{n}` | - (closed form) | lambda* {sp['lambda_selected']} |")
        else:
            print(f"| `{n}` | {sp['epochs_trained']} | {sp['epochs_selected']} |")

    print("\n## THE FOUR FAMILIES (peak rung vs deployed rung)\n")
    for s in sits:
        ff = R["four_families"][s]
        fam = ff["families"]
        t = fam["TACTICAL"]
        print(f"\n### {s}  (peak = `{R['per_situation'][s]['peak_rung']}`)\n")
        print(f"**TACTICAL** - n {t['n_rows_scored']:,}, positives {t['n_pos']}, "
              f"base {t['base_rate']:.5f}")
        for k, v in t["ap_lift"].items():
            op = t["operating_point_5pct"][k]
            ld = t["anticipation_lead"][k]
            print(f"* `{k}`: AP-lift **{v:.4f}**, P@5% {op['precision']:.4f} / R@5% "
                  f"{op['recall']:.4f} ({op['n_alarm']:,} fires, {op['tp']} tp), lead "
                  f"{ld['median_lead_s']} s ({ld['n_runs_no_alarm']}/{ld['n_runs']} runs no alarm)")
        print(f"* paired delta AP-lift: {ci(t['paired_delta_ap_lift'])} "
              f"{sep(t['paired_delta_ap_lift'])}")
        for famname in ("LONGITUDINAL", "LATERAL"):
            f = fam[famname]
            print(f"\n**{famname}** - not computable: {f.get('_not_computable')} "
                  f"({f.get('_not_computable_reason')}); reported as regime strata:")
            for nm, st in f.get("strata", {}).items():
                if st.get("_status") == "UNPOWERED":
                    print(f"* `{nm}`: UNPOWERED (n {st['n_rows']:,}, pos {st['n_pos']})")
                else:
                    print(f"* `{nm}`: n {st['n_rows']:,}, pos {st['n_pos']}, "
                          f"delta {ci(st['paired_delta_ap_lift'])} "
                          f"{sep(st['paired_delta_ap_lift'])}")
        st_ = fam["STRATEGIC"]
        print(f"\n**STRATEGIC** - {st_['_status']} (n {st_['n_rows']:,}): {st_['_reason']}")


if __name__ == "__main__":
    main()
