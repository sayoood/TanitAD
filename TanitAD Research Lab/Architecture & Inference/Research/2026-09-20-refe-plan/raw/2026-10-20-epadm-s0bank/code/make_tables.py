"""Machine-generated tables for AMENDMENT_S0BANK_DRAFT.md section 3 / 5 from sim_results.json (nothing is typed by hand).   python make_tables.py <sim_results.json> > tables.md
Also computes R-DUP (mean distinct logs under generator A) from the outer draws alone (no replicate data)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def pct(r):
    return f"{100 * r['rate']:.1f} % [{100 * r['wilson95'][0]:.1f}, {100 * r['wilson95'][1]:.1f}] ({r['k']}/{r['n']})"


def pw(r):
    return f"{100 * r['rate']:.1f} % [{100 * r['wilson95'][0]:.1f}, {100 * r['wilson95'][1]:.1f}]"


def main(path):
    res = json.load(open(path, encoding="utf-8"))
    J = res["jobs"]
    out = []
    geos = sorted({j.split("-")[1] for j in J}, key=lambda g: int(g[2:]))
    out.append("#### S1. THE THREE COMMITTED BARS (SIMULATED; primary NaN convention = real sigma; the fail-closed convention in brackets). Wilson 95 %. `MET` needs Base C AND Base X.\n")
    out.append("| geometry | base (seeds) | A1-r1 false refusal (bar: upper <= 5.0 %) | A2 (1.05) false refusal (bar: upper <= 5.0 %) | DAC power x1.25, margin 1.05 (bar: point >= 70 %) |")
    out.append("|---|---|---|---|---|")
    for g in geos:
        for b, lab in (("C", "C carve-B clones (3)"), ("X", "X S0 stored G25 (2)")):
            e = J.get(f"{b}-{g}-A")
            if not e:
                continue
            p, f = e["bars_primary"], e["bars_failclosed"]
            out.append(f"| {g} | {lab} | {pct(p['A1_r1_false_refusal'])} | {pct(p['A2_1.05_false_refusal'])} [{100 * f['A2_1.05_false_refusal']['rate']:.1f} %] | {pw(p['DAC_power_x1.25_margin1.05'])} [{100 * f['DAC_power_x1.25_margin1.05']['rate']:.1f} %] |")
    out.append("")
    out.append("| geometry | A1-r1 FR | A2 (1.05) FR | DAC power x1.25 | all three MET | R-NAN (status differs between conventions) | R-GEN (T vs A, A1 FR, pp) | R-SEED (C vs X, pp: A1 FR / DAC power) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for g in geos:
        s = res["geometries"].get(g)
        if not s:
            continue
        st = list(s["status_primary_convention"].values())
        gen = s["R-GEN_T_vs_A_A1_fr_pp"]
        out.append(f"| {g} | {st[0]} | {st[1]} | {st[2]} | **{s['all_met']}** | {s['R-NAN']} | {('%.1f' % gen) if gen is not None else 'n/a'} | {s['R-SEED_C_vs_X_pp']['A1_fr']:.1f} / {s['R-SEED_C_vs_X_pp']['DAC_power']:.1f} |")
    out.append("")
    out.append("#### S2. DESCRIPTIVE (not bars): A1 power, A2 power by head and margin, per-seed A1 false refusal, generator T cross-check\n")
    out.append("| geometry | base-gen | A1 power +0.135 | A1 power +0.05 | A2 power DAC x1.25 @1.00 / @1.05 / @1.10 | DAC x1.5 @1.05 | TTC x1.25 @1.05 | NC x1.25 @1.05 | NC x1.5 @1.05 | A1 FR per seed |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for g in geos:
        for b in ("C", "X"):
            for gen in ("A", "T"):
                e = J.get(f"{b}-{g}-{gen}")
                if not e:
                    continue
                S = e["summaries"]

                def a2(name, h, m):
                    return pw(S[name]["A2"]["by_margin"][m]["per_head_any_seed"][h]) if name in S else "n/a"
                row = [g, f"{b}-{gen}"]
                row.append(pw(S["PLANT_EP_0.135"]["A1"]["r1_any_seed"]) if "PLANT_EP_0.135" in S else "n/a")
                row.append(pw(S["PLANT_EP_0.05"]["A1"]["r1_any_seed"]) if "PLANT_EP_0.05" in S else "n/a")
                row.append(" / ".join(a2("PLANT_A2_DAC_1.25", "DAC", m) for m in ("1.00", "1.05", "1.10")))
                row.append(a2("PLANT_A2_DAC_1.5", "DAC", "1.05"))
                row.append(a2("PLANT_A2_TTC_1.25", "TTC", "1.05"))
                row.append(a2("PLANT_A2_NC_1.25", "NC", "1.05"))
                row.append(a2("PLANT_A2_NC_1.5", "NC", "1.05"))
                row.append(" / ".join(f"{100 * r['rate']:.1f}" for r in S["CAL"]["A1"]["r1_per_seed"]) + " %")
                out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append("#### S3. CONTROLS and flags\n")
    for job, e in J.items():
        if "controls" not in e:
            continue
        c = e["controls"]
        out.append(f"* `{job}`: controls_ok **{e['controls_ok']}**; PC1a {c['PC1a_calibrated_by_construction']['pass']}, PC1b {c['PC1b_base_exactness']['pass']} (max |obs/pred - 1| at the base = {c['PC1b_base_exactness']['max_abs_dev']}), PC2 {c['PC2_check_has_teeth']['pass']}, "
                   f"PC4 {c['PC4_decision_code_is_the_modules']['pass']}, PC5 {c['PC5_determinism'].get('pass')}, PC6 {c.get('PC6_identity_candidate', {}).get('pass')}; REPORTED A2 ratio means after the redraw {json.dumps(c['PC1b_base_exactness']['REPORTED_A2_ratio_point_mean_after_redraw'])}")
    out.append(f"* PC3 (shortcut = `analyze_candidate`): {res['controls_global']['PC3_shortcut_is_the_estimator'].get('pass')}; PC8 (Base X code = the R1 simulator, bit for bit vs R2-CONFIRM): {res['controls_global']['PC8_code_is_the_R1_simulator'].get('pass')}; overall controls_ok **{res['controls_ok']}**.")
    for job, e in J.items():
        ex = e["run"].get("extras", {})
        if "pool_events" in ex:
            out.append(f"* R-EVENTS / R-DOSE (Base C pool, per seed; `{job}`): " + "; ".join(f"seed {s}: unsafe picks NC {v['unsafe_picks']['NC']} / DAC {v['unsafe_picks']['DAC']} / TTC {v['unsafe_picks']['TTC']}, changed {v['changed']} ({100 * v['changed_share']:.1f} %)" for s, v in ex["pool_events"].items()))
            break
    for job, e in J.items():
        if job.endswith("-A"):
            k0 = e["summaries"]["CAL"]["A2"]["k0_replicate_seed_heads"]
            out.append(f"* R-NAN counts (`{job}`, CAL, replicate-seed pairs whose base has no observed unsafe pick; n = {e['summaries']['CAL']['n']} x {len(e['summaries']['CAL']['means']['M_p'])}): {json.dumps(k0)}")
    import epadm_sim_s0 as S
    geo = {}
    for g in geos:
        n = S.load_geometry(os.environ.get("S0_GEOMETRY", os.path.join(HERE, "..", "geometry_s0.json")), g)["n"]
        geo[g] = float(np.mean([len(set(S.outer_idx(r, n, "A").tolist())) for r in range(2000)]))
    out.append("* R-DUP (mean distinct logs of N under generator A, 2,000 outer draws): " + ", ".join(f"{g}: {v:.1f}" for g, v in geo.items()))
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
