"""Machine-generated tables for RESULT.md from sim_results.json (nothing is typed by hand).   python make_tables_n1.py <sim_results.json> [<s0bank sim_results.json>] > tables.md
Also computes R-DUP (mean distinct logs under generator A, N1 salt) from the outer draws alone (no replicate data)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
S0_RESULTS = "C:/Users/Admin/par18/s0amend/work/sim/sim_results.json"
HEADS = ("NC", "DAC", "TTC")


def pct(r):
    return f"{100 * r['rate']:.1f} % [{100 * r['wilson95'][0]:.1f}, {100 * r['wilson95'][1]:.1f}] ({r['k']}/{r['n']})"


def pw(r):
    return f"{100 * r['rate']:.1f} % [{100 * r['wilson95'][0]:.1f}, {100 * r['wilson95'][1]:.1f}]"


def main(path, s0path=S0_RESULTS):
    res = json.load(open(path, encoding="utf-8"))
    s0 = json.load(open(s0path, encoding="utf-8")) if os.path.exists(s0path) else None
    J = res["jobs"]
    out = []
    geos = sorted({j.split("-")[1] for j in J}, key=lambda g: int(g[2:]))
    out.append("#### N1-T1. THE THREE COMMITTED BARS (SIMULATED; primary NaN convention = real sigma, fail-closed convention in brackets). Wilson 95 %. Generator A.\n")
    out.append("| geometry | base (seeds) | A1-r1 false refusal (bar: upper <= 5.0 %) | A2-N1 (1.05) false refusal (bar: upper <= 5.0 %) | N1 DAC power x1.25, margin 1.05 (bar: point >= 70 %) | every bar met in this cell |")
    out.append("|---|---|---|---|---|---|")
    for g in geos:
        for b, lab in (("C", "C carve-B clones (3)"), ("X", "X S0 stored G25 (2)")):
            e = J.get(f"{b}-{g}-A")
            if not e:
                continue
            p, f = e["bars_primary"], e["bars_failclosed"]
            d = p["decision"]
            out.append(f"| {g} | {lab} | {pct(p['A1_r1_false_refusal'])} | {pct(p['A2_N1_1.05_false_refusal'])} [{100 * f['A2_N1_1.05_false_refusal']['rate']:.1f} %] | {pw(p['N1_DAC_power_x1.25_margin1.05'])} [{100 * f['N1_DAC_power_x1.25_margin1.05']['rate']:.1f} %] | **{d['all_met']}**" + ("" if d["all_met"] else f" (missed: {', '.join(m.split('_')[0] + '_' + m.split('_')[1] for m in d['missed'])})") + " |")
    out.append("")
    out.append("| geometry | A1-r1 FR | A2-N1 FR | N1 DAC power x1.25 | geometry ALL-MET (MET = met on C AND on X) | cells meeting every bar | R-NAN | R-SEED (C vs X, pp: A1 FR / N1 FR / DAC power) | R-GEN (P vs A on C, pp: N1 FR / DAC power; status differs) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for g in geos:
        s = res["geometries"].get(g)
        if not s:
            continue
        st = list(s["status_primary_convention"].values())
        gen = s.get("R-GEN_P_vs_A_pp")
        cells = [k for k, v in s["cells_all_met"].items() if v]
        out.append(f"| {g} | {st[0]} | {st[1]} | {st[2]} | **{s['all_met']}** | {', '.join(cells) if cells else 'none'} | {s['R-NAN']} | {s['R-SEED_C_vs_X_pp']['A1_fr']:.1f} / {s['R-SEED_C_vs_X_pp']['A2_N1_fr']:.1f} / {s['R-SEED_C_vs_X_pp']['DAC_power']:.1f} | "
                   + (f"{gen['A2_N1_fr']:.1f} / {gen['DAC_power']:.1f}; {s['R-GEN_status_differs_from_A']}" if gen else "n/a") + " |")
    out.append("")
    out.append("#### N1-T2. DESCRIPTIVE (not bars): N1 power by head and margin, A1 power, the A2L reference, generator P, and the INHERITED all-picks (uniform plant) A2 of the S0BANK simulation\n")
    out.append("| geometry | base-gen | N1 DAC x1.25 @1.00 / @1.05 / @1.10 | DAC x1.5 @1.05 | TTC x1.25 @1.05 | NC x1.25 @1.05 | NC x1.5 @1.05 | TTC x1.5 @1.05 | A2L (all picks, localised plant) DAC x1.25 @1.05 | A1 power +0.135 / +0.05 | A2-N1 FR (fail only / undefined verdicts) | INHERITED all-picks A2 (S0BANK, uniform plant) DAC x1.25 @1.05 | INHERITED all-picks A2 FR |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g in geos:
        for b in ("C", "X"):
            for gen in ("A", "P"):
                e = J.get(f"{b}-{g}-{gen}")
                if not e:
                    continue
                S = e["summaries"]

                def n1(name, h, m):
                    return pw(S[name]["N1"]["by_margin"][m]["per_head_any_seed"][h]) if name in S else "n/a"
                row = [g, f"{b}-{gen}", " / ".join(n1("PLANT_N1_DAC_1.25", "DAC", m) for m in ("1.00", "1.05", "1.10")) if "PLANT_N1_DAC_1.25" in S else "n/a",
                       n1("PLANT_N1_DAC_1.5", "DAC", "1.05"), n1("PLANT_N1_TTC_1.25", "TTC", "1.05"), n1("PLANT_N1_NC_1.25", "NC", "1.05"), n1("PLANT_N1_NC_1.5", "NC", "1.05"), n1("PLANT_N1_TTC_1.5", "TTC", "1.05")]
                row.append(pw(S["PLANT_A2L_DAC_1.25"]["A2L"]["by_margin"]["1.05"]["per_head_any_seed"]["DAC"]) if "PLANT_A2L_DAC_1.25" in S else "n/a")
                row.append(" / ".join(pw(S[f"PLANT_EP_{d}"]["A1"]["r1_any_seed"]) for d in ("0.135", "0.05")) if "PLANT_EP_0.135" in S else "n/a")
                m = S["CAL"]["N1"]["by_margin"]["1.05"]
                row.append(f"{100 * m['any_head_any_seed_fail_only']['rate']:.1f} % / {m['undefined_verdicts']}")
                inh = s0["jobs"].get(f"{b}-{g}-A") if (s0 and gen == "A") else None
                if inh:
                    row.append(pw(inh["summaries"]["PLANT_A2_DAC_1.25"]["A2"]["by_margin"]["1.05"]["per_head_any_seed"]["DAC"]))
                    row.append(pw(inh["summaries"]["CAL"]["A2"]["by_margin"]["1.05"]["any_head_any_seed"]))
                else:
                    row += ["n/a", "n/a"]
                out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append("#### N1-T3. CONTROLS and flags\n")
    for job, e in J.items():
        if "controls" in e:
            c = e["controls"]
            out.append(f"* `{job}`: controls_ok **{e['controls_ok']}**; " + "; ".join(f"{k.split('_')[0]} {v.get('pass')}" for k, v in c.items()))
    cg = res["controls_global"]
    out.append("* global: " + "; ".join(f"{k} {v.get('pass')}" for k, v in cg.items()) + f"; overall controls_ok **{res['controls_ok']}**")
    pc1 = {job: e["controls"]["PC1b_N1_base_exactness_at_the_changed_picks"] for job, e in J.items() if "controls" in e and "PC1b_N1_base_exactness_at_the_changed_picks" in e["controls"]}
    out.append("* PC1b deviations: " + json.dumps({k: [v["CAL_max_abs_dev_from_1"], v["PLANT_DAC_1.25_max_abs_dev_from_1.25"]] for k, v in pc1.items()}))
    for job, e in J.items():
        S = e["summaries"]
        cal = S["CAL"]["N1"]
        out.append(f"* R-NAN counts `{job}` CAL (replicate-seed-heads with NO observed unsafe mass at the changed picks; n = replicates x seeds): {json.dumps(cal['k0_replicate_seed_heads'])}; undefined replicate-seed-heads {json.dumps(cal['und_replicate_seed_heads'])}")
    out.append("")
    out.append("#### N1-T4. R-EVENTS / R-DOSE (replicate bases): mean changed picks and mean observed unsafe mass AT THE CHANGED PICKS per replicate base (CAL, generator A), by seed\n")
    out.append("| job | changed picks per base (by seed) | observed unsafe mass at the changed picks per base: NC | DAC | TTC |")
    out.append("|---|---|---|---|---|")
    for job, e in J.items():
        cal = e["summaries"]["CAL"]["N1"]
        eo = cal["base_observed_unsafe_mass_at_changed_mean"]
        out.append(f"| {job} | {', '.join('%.0f' % v for v in cal['base_changed_picks_mean_by_seed'])} | " + " | ".join(", ".join("%.1f" % v for v in eo[h]) for h in HEADS) + " |")
    out.append("")
    out.append("#### N1-T5. R-POOLRATIO / R-EVENTS of the banked bases (point estimates only; descriptive; no interval)\n")
    seen = set()
    for job, e in J.items():
        b, g, gen = job.split("-")
        ex = e["run"]["extras"]
        if b == "C" and "C" not in seen:
            seen.add("C")
            out.append("**Base C pool (carve-B, 400 tokens), by search seed:**\n")
            out.append("| seed | changed picks (share) | head | distinct changed picks with unsafe mass | observed / predicted unsafe mass at the CHANGED picks (pooled ratio) | at the UNCHANGED picks (pooled ratio) |")
            out.append("|---|---|---|---|---|---|")
            for s, v in ex["pool_events"].items():
                for h in HEADS:
                    hv = v["heads"][h]
                    r1 = hv["N1_pooled_point_ratio"]
                    r2 = hv["unchanged_pooled_point_ratio"]
                    out.append(f"| {s} | {v['changed']} ({100 * v['changed_share']:.1f} %) | {h} | {hv['distinct_changed_picks_with_unsafe_mass']} | {hv['observed_unsafe_mass_at_changed']:.2f} / {hv['predicted_unsafe_mass_at_changed']:.2f} ({'n/a' if r1 is None else '%.3f' % r1}) | {hv['observed_unsafe_mass_at_unchanged']:.2f} / {hv['predicted_unsafe_mass_at_unchanged']:.2f} ({'n/a' if r2 is None else '%.3f' % r2}) |")
            out.append("")
        if b == "X":
            out.append(f"**Base X on the {g} logs, by seed:**\n")
            out.append("| seed | changed picks of sets (share) | head | observed / predicted unsafe mass at the CHANGED picks (pooled ratio) | at the UNCHANGED picks (pooled ratio) |")
            out.append("|---|---|---|---|---|")
            for s, v in ex["basex_events_geometry_logs"].items():
                for h in HEADS:
                    hv = v["heads"][h]
                    r1 = hv["N1_pooled_point_ratio"]
                    r2 = hv["unchanged_pooled_point_ratio"]
                    out.append(f"| {s} | {v['changed']:.0f} ({100 * v['changed'] / v['n_sets']:.1f} %) | {h} | {hv['observed_unsafe_mass_at_changed']:.2f} / {hv['predicted_unsafe_mass_at_changed']:.2f} ({'n/a' if r1 is None else '%.3f' % r1}) | {hv['observed_unsafe_mass_at_unchanged']:.2f} / {hv['predicted_unsafe_mass_at_unchanged']:.2f} ({'n/a' if r2 is None else '%.3f' % r2}) |")
            out.append("")
    import epadm_sim_n1 as M
    dup = {n: float(np.mean([len(set(M.outer_idx(r, n, "A").tolist())) for r in range(2000)])) for n in (41, 117)}
    out.append(f"* R-DUP (mean distinct logs of N under generator A, 2,000 outer draws, N1 salt): GS41: {dup[41]:.1f}, GS117: {dup[117]:.1f}")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
