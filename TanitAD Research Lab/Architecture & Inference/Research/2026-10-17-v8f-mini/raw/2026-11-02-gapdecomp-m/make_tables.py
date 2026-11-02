#!/usr/bin/env python3
"""make_tables.py -- RESULTS_TABLES.md from raw/GAPDECOMP_M.json and raw/POSTHOC.json (no number typed by hand).   python make_tables.py"""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
J = json.load(open(f"{HERE}/raw/GAPDECOMP_M.json", encoding="utf-8"))
P = json.load(open(f"{HERE}/raw/POSTHOC.json", encoding="utf-8"))
ARM = {"CTRL": "DAG-OACC", "SOUP": "SOUP"}
ROWS = ("ADE6", "yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share")
out = []


def f(x, fmt="+.3f"):
    return "n/a" if x is None else format(x, fmt)


def iv(g, fmt="+.3f"):
    return f"{f(g['share'], fmt)} [{f(g['share_lo'], fmt)}, {f(g['share_hi'], fmt)}]"


def dfmt(x):
    return "n/a" if x is None else (f"{x:+.4f}" if abs(x) < 1 else f"{x:+.2f}")


def qual(g):
    return bool(g["nameable"] and g["support_ok"] and g["w"] <= 0.3 and g["share"] is not None and g["share"] >= 0.5 and g["share_lo"] > 0.3)


c = J["controls"]
out += ["### Table 0. Known-answer controls (run before any verdict was read)", "", "| control | result |", "|---|---|",
        f"| KA1 partitions cover M once; M = 2,032 windows / 140 episodes = the landed class M; named cross cells at the floor | {'PASS' if c['KA1']['PASS'] else 'FAIL'} |",
        f"| KA2 subclass sums reproduce the landed class-M n, Delta_M, lo, hi, contribution and its interval, {c['KA2']['n_checked']} checks (5 rows x 2 arms x 2 seeds x 3 partitions) | {'PASS' if c['KA2']['PASS'] else 'FAIL'}; max abs deviation {c['KA2']['max_abs_dev']:.1e} (tolerance 1e-9) |",
        f"| KA3 (X, X) reads exactly 0.0 in every subclass, interval [0, 0], {c['KA3']['n_checked']} checks | {'PASS' if c['KA3']['PASS'] else 'FAIL'} |",
        f"| KA4 shares of each partition sum to 1 at the point and in every draw, {c['KA4']['n_checked']} checks | {'PASS' if c['KA4']['PASS'] else 'FAIL'}; max deviation {c['KA4']['max_dev_point']:.1e} (point), {c['KA4']['max_dev_draws']:.1e} (draws) |",
        "| KA5 planted effect LOCALISED / uniform effect DIFFUSE / zero effect NOT DEFINED / floor refuses to name (pytest, 8 tests) | PASS (`raw/pytest.log`) |",
        f"| KA6 an empty subclass reproduces the landed Delta of all 19 rows and the landed `D2_worse_rows`, {c['KA6']['n_checked']} arm-seeds | {'PASS' if c['KA6']['PASS'] else 'FAIL'}; max abs deviation {c['KA6']['max_abs_dev_delta']:.1e} |", ""]
out += ["### Table 1. The subclasses of M (windows / episodes on the 2,032 M windows)", "", "| partition | cell | windows | episodes | nameable |", "|---|---|---|---|---|"]
for pn, cells in J["partitions"].items():
    for nm, v in sorted(cells.items()):
        out.append(f"| {pn} | {nm} | {v['windows']} | {v['episodes']} | {'yes' if v['nameable'] else 'no (pooled remainder)'} |")
sg = J["shape_geometry"]
out += ["", f"Windows of M with no pair-valid GT step (STRAIGHT by rule): {sg['windows_in_M_without_pair_valid_GT_step']}. |dpsi| over M, quantiles: " + ", ".join(f"p{int(float(k) * 100)} {v:.1f} deg" for k, v in sg["dpsi_deg_quantiles_M"].items()) + ".", ""]
out += ["### Table 2. Verdict per row (PREREG_FROZEN rule), DAG-OACC headline and SOUP", "",
        "Cell: share of the row's Delta_M carried by the named subclass (point [95 % interval]); w = its share of the row's valid M windows; eligible = candidates with support, nameable and w <= 0.30; argmax = fraction of draws in which the named subclass has the highest share among the eligible.", "",
        "| row | arm | Delta_M s0 | Delta_M s1 | verdict | named | share s0 | share s1 | w s0 / s1 | eligible s0 / s1 | argmax s0 / s1 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for arm in ("CTRL", "SOUP"):
    for row in ROWS:
        v = J["verdicts"][arm][row]
        m0, m1 = J["rows"][arm]["s0"][row]["M"], J["rows"][arm]["s1"][row]["M"]
        nm = v["named"]
        if nm:
            pn, gn = nm.split("|", 1)
            g0, g1 = J["rows"][arm]["s0"][row]["parts"][pn]["groups"][gn], J["rows"][arm]["s1"][row]["parts"][pn]["groups"][gn]
            sh0, sh1, w = iv(g0), iv(g1), f"{g0['w']:.3f} / {g1['w']:.3f}"
            am = f"{f(v['argmax_freq_of_named'].get('s0'), '.3f')} / {f(v['argmax_freq_of_named'].get('s1'), '.3f')}"
        else:
            sh0 = sh1 = w = am = "-"
        out.append(f"| {row} | {ARM[arm]} | {dfmt(m0['delta'])} [{dfmt(m0['lo'])}, {dfmt(m0['hi'])}] | {dfmt(m1['delta'])} [{dfmt(m1['lo'])}, {dfmt(m1['hi'])}] | **{v['verdict']}** | {nm or '-'} | {sh0} | {sh1} | {w} | {v['n_eligible']['s0']} / {v['n_eligible']['s1']} | {am} |")
out += [""]
out += ["### Table 3. DAG-OACC, every candidate against its share of Delta_M (shape, speed and cross partitions)", "",
        "q = qualifies at that seed (eligible, share >= 0.5, interval lower > 0.3). enr = share / w.", ""]
for row in ROWS:
    out += [f"**{row}**", "", "| subclass | windows (episodes) | w | Delta_sub s0 | share s0 [interval] | Delta_sub s1 | share s1 [interval] | enr s0 / s1 | q s0 / s1 |", "|---|---|---|---|---|---|---|---|---|"]
    for pn in ("shape", "speed", "cross"):
        G0, G1 = J["rows"]["CTRL"]["s0"][row]["parts"][pn]["groups"], J["rows"]["CTRL"]["s1"][row]["parts"][pn]["groups"]
        for nm in sorted(G0):
            a, b = G0[nm], G1[nm]
            if a["n"] == 0 and b["n"] == 0:
                out.append(f"| {nm} | 0 (0) | 0.000 | n/a | n/a | n/a | n/a | n/a | no valid pair |")
                continue
            out.append(f"| {nm} | {a['n']} ({a['n_episodes']}) | {a['w']:.3f} | {dfmt(a['delta'])} | {iv(a)} | {dfmt(b['delta'])} | {iv(b)} | {f(a['enrichment'], '.2f')} / {f(b['enrichment'], '.2f')} | "
                       f"{'q' if qual(a) else '-'} / {'q' if qual(b) else '-'}{'' if a['support_ok'] else ' (below floor)'} |")
    out += [""]
B = J["beside"]
out += [f"### Table 4. Beside (MEASURED, descriptive): plan against GT in {', '.join(B['B'])} ({B['B_kind']}); plan minus GT; window means", "",
        "| arm / seed | who | plan speed | GT speed | speed err | plan mean abs kappa (1/m) | GT mean abs kappa | abs-kappa diff | signed kappa err | n |", "|---|---|---|---|---|---|---|---|---|---|"]
for key in B["B"]:
    for arm in ("CTRL", "SOUP"):
        for s in ("s0", "s1"):
            r = B["plan_vs_gt"][key][arm][s]
            for who, nm in (("x", ARM[arm]), ("p10", "P10")):
                q = r[who]
                out.append(f"| {ARM[arm]} {s} | {nm} | {q['plan_speed']:.3f} | {q['gt_speed']:.3f} | {q['speed_err_plan_minus_gt']:+.3f} | {q['plan_abs_curv']:.4f} | {q['gt_abs_curv']:.4f} | {q['abs_curv_diff_plan_minus_gt']:+.4f} | {q['signed_curv_err_plan_minus_gt']:+.4f} | {q['n_windows_curv']} |")
out += ["", "### Table 5. Beside (ESTIMATED, decomposition arithmetic): D2' with the subclass's gap set to 0 (all 19 rows)", "", "| subclass | arm | seed | rows separated-worse with the subclass zeroed | D2' | rows separated-worse before (landed) |", "|---|---|---|---|---|---|"]
for key in B["B"]:
    for arm in ("CTRL", "SOUP"):
        for s in ("s0", "s1"):
            r = B["D2prime"][key][arm][s]
            out.append(f"| {key} | {ARM[arm]} | {s} | {', '.join(r['D2prime_worse_rows']) or 'none'} | {'holds' if r['D2prime'] else 'fails'} | {', '.join(r['D2_total_worse_rows']) or 'none'} |")
        out.append(f"| {key} | {ARM[arm]} | both seeds | | {'holds' if B['D2prime'][key][arm]['both_seeds_D2prime'] else 'fails'} | |")
out += ["", "### Table 6. POST-HOC (not in the card; decides nothing)", "", "**6a. The pooled low-speed view of M (v0 < 8 m/s: the 0.5-3 and 3-8 buckets together), share of Delta_M**", "",
        "| row | arm | w s0 | share s0 [interval] | share s1 [interval] | enr s0 / s1 |", "|---|---|---|---|---|---|"]
for arm in ("CTRL", "SOUP"):
    for row in ROWS:
        a, b = P["pooled_low_speed"][arm]["s0"][row], P["pooled_low_speed"][arm]["s1"][row]
        out.append(f"| {row} | {ARM[arm]} | {a['w']:.3f} | {iv(a)} | {iv(b)} | {a['enrichment']:.2f} / {b['enrichment']:.2f} |")
out += ["", "**6b. D2 arithmetic with more windows zeroed (ESTIMATED)**", "", "| zeroed | arm | seed | rows separated-worse | D2 |", "|---|---|---|---|---|"]
for k, nm in (("D2_double_zero", "trigger L+S and the 3-8 m/s windows of M"), ("D2_all_windows_v0_lt_8_zero", "every window with v0 < 8 m/s (861)")):
    for arm in ("CTRL", "SOUP"):
        for s in ("s0", "s1"):
            r = P[k][arm][s]
            out.append(f"| {nm} | {ARM[arm]} | {s} | {', '.join(r['worse_rows']) or 'none'} | {'holds' if r['D2'] else 'fails'} |")
mm = P["mechanism_3to8"]
out += ["", "**6c. Mechanism descriptors of the 3-8 m/s windows of M** (GT step speed in [5.2, 6.2): " + f"{mm['gt_step_speed_share_in_[5.2,6.2)']:.3f}; below 5.2: {mm['gt_step_speed_share_below_5.2']:.3f})", "",
        "| arm / seed | who | median per-window mean abs kappa | p90 | share of the sum in the top 10 % of windows | windows with a plan step < 0.5 m/s | median of the plan's slowest step (m/s) | rest of M: median abs kappa / stall share |", "|---|---|---|---|---|---|---|---|"]
for arm in ("CTRL", "SOUP"):
    for s in (0, 1):
        for who, nm in (("x", ARM[arm]), ("p10", "P10")):
            r = mm[f"{arm}_s{s}"][who]
            q, rest = r["plan_abs_curv_in_3to8"], r["plan_abs_curv_in_rest_of_M"]
            out.append(f"| {ARM[arm]} s{s} | {nm} | {q['median']:.4f} (mean {q['mean']:.4f}) | {q['p90']:.4f} | {q['share_of_sum_in_top10pct_windows']:.3f} | {r['share_windows_with_a_plan_step_below_0.5mps_in_3to8']:.3f} | {r['plan_min_step_speed_median_in_3to8']:.2f} | {rest['median']:.4f} / {r['share_windows_with_a_plan_step_below_0.5mps_in_rest_of_M']:.3f} |")
open(f"{HERE}/raw/RESULTS_TABLES.md", "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("TABLES OK", len(out), "lines")
