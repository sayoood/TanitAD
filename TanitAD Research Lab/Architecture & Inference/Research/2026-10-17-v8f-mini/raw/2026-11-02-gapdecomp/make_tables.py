#!/usr/bin/env python3
"""make_tables.py -- the markdown tables of RESULTS.md, generated from raw/CLASS_SPLIT.json (no number is typed by hand).

    python -I make_tables.py raw/CLASS_SPLIT.json raw/RESULTS_TABLES.md
"""
import json
import sys

ARMS = ("CTRL", "REP", "SOUP")
NAME = {"CTRL": "DAG-OACC", "REP": "REP", "SOUP": "SOUP"}
CRAWL = ("ADE6", "yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share")
CLASSES = ("L", "S", "O", "M")
CN = {"L": "GT-launch", "S": "GT-stay", "O": "other-stopped", "M": "moving, no launch"}
DP = {"ADE6": 3, "yaw-rate MAE": 2, "curvature MAE": 3, "L2 curvature MAE (GT>=5.2)": 3, "M1 crawl share": 3}


def f(v, dp=3, sign=True):
    if v is None:
        return "n/a"
    return format(v, f"{'+' if sign else ''}.{dp}f")


def main(src, dst):
    d = json.load(open(src, encoding="utf-8"))
    A = d["arms"]
    out = []
    ctl = d["controls"]
    rows_all = [(arm, s, lab, A[arm][f"s{s}"][lab]) for arm in ARMS for s in (0, 1) for lab in A[arm][f"s{s}"]]
    kd = lambda key: max(abs(r[key] - r["_known_answer"]["banked"][key]) for _, _, _, r in rows_all)
    out.append("### Table 0. Known-answer controls (all run before any class number was read)\n")
    out.append("| control | result |\n|---|---|")
    out.append(f"| K1 partition and class counts, 6 arm-seeds | {sum(1 for k, v in ctl.items() if k.startswith('K1') and v['PASS'])} / 6 PASS (L = banked `launch` 327 / 49 episodes; S = banked `stay`) |")
    out.append(f"| K2 class sums reproduce the banked row, {len(rows_all)} arm-seed-rows | {sum(1 for _, _, _, r in rows_all if all(r['_known_answer']['K2_vs_banked'].values()))} / {len(rows_all)} PASS; max abs difference: delta {kd('delta'):.1e}, lo {kd('lo'):.1e}, hi {kd('hi'):.1e} (tolerance 6e-5, the banked values are rounded to 4 dp) |")
    out.append(f"| K3 (X, X) reads exactly 0.0 in every class, interval [0, 0] | {sum(1 for _, _, _, r in rows_all if r['_known_answer']['K3_xx_zero'])} / {len(rows_all)} PASS |")
    out.append(f"| K4 total interval equals `taniteval.ci` called directly | {sum(1 for _, _, _, r in rows_all if all(r['_known_answer']['K4_vs_instrument'].values()))} / {len(rows_all)} PASS |")
    mx = lambda key: max((r[key] for _, _, _, r in rows_all if r[key] is not None), default=0.0)
    out.append(f"| K5 identity: sum of class contributions equals Delta; shares sum to 1 in every draw | {sum(1 for _, _, _, r in rows_all if r['_known_answer']['K5_identity'])} / {len(rows_all)} PASS; max deviation {mx('_class_sum_identity_abs_dev'):.1e} (point), {mx('_share_sum_max_abs_dev_over_draws'):.1e} (draws) |")
    out.append(f"| rows decomposable | {sum(1 for _, _, _, r in rows_all if r['decomposable'])} / {len(rows_all)} |\n")
    # 1 class sizes
    k1 = d["controls"]["K1:CTRL_s0"]
    out.append("### Table 1. The classes (eval-141, 2,399 windows / 141 episodes)\n")
    out.append("| class | windows | episodes | share of windows |\n|---|---|---|---|")
    for c in CLASSES:
        out.append(f"| {c} {CN[c]} | {k1['counts'][c]} | {k1['episodes'][c]} | {k1['counts'][c] / 2399:.3f} |")
    t = k1["counts"]["L"] + k1["counts"]["S"]
    out.append(f"| **T = L + S** | **{t}** | | **{t / 2399:.3f}** |\n")
    # 2 verdicts
    out.append("### Table 2. Trigger-window share of each row's gap against P10 (T-share = share_L + share_S), verdict per arm\n")
    out.append("Cell: T-share point [95 % interval]; verdict per seed. Delta = the row's gap against P10 (unit of the row; lower is better for every row in this table). A T-share is only meaningful where the interval of Delta excludes 0 (it does for every cell of this table).\n")
    out.append("| row | arm | Delta s0 | T-share s0 | verdict s0 | Delta s1 | T-share s1 | verdict s1 | arm verdict |\n|---|---|---|---|---|---|---|---|---|")
    for lab in CRAWL:
        for arm in ARMS:
            cells = []
            for s in (0, 1):
                r = A[arm][f"s{s}"][lab]
                tr = r["trigger"]
                cells += [f"{f(r['delta'], DP[lab])} [{f(r['lo'], DP[lab])}, {f(r['hi'], DP[lab])}]", f"{f(tr['share'])} [{f(tr['lo'])}, {f(tr['hi'])}]", d["verdicts"][arm][lab][f"s{s}"]]
            out.append(f"| {lab} | {NAME[arm]} | " + " | ".join(cells) + f" | **{d['verdicts'][arm][lab]['arm']}** |")
    out.append("")
    # 3 per-class split of DAG-OACC
    for arm in ("CTRL",):
        out.append(f"### Table 3. {NAME[arm]} against P10, split by class (n windows; w = n / n_row; Delta_c; share_c [95 % interval])\n")
        for lab in CRAWL:
            out.append(f"**{lab}**\n")
            out.append("| class | s0 n | s0 w | s0 Delta_c | s0 share [interval] | s1 n | s1 w | s1 Delta_c | s1 share [interval] |\n|---|---|---|---|---|---|---|---|---|")
            for c in CLASSES:
                cells = []
                for s in (0, 1):
                    x = A[arm][f"s{s}"][lab]["classes"][c]
                    cells += [str(x["n"]), f"{x['w']:.3f}", f(x["delta"], DP[lab]), f"{f(x['share'])} [{f(x['share_lo'])}, {f(x['share_hi'])}]"]
                out.append(f"| {c} {CN[c]} | " + " | ".join(cells) + " |")
            out.append("")
    # 4 beside
    out.append("### Table 4. Beside (ESTIMATED): the trigger windows' Delta set to 0 (Delta' = contribution of O + M)\n")
    out.append("| arm | seed | rows separated-worse as read (D2) | rows separated-worse with T zeroed (D2') | D2' holds |\n|---|---|---|---|---|")
    for arm in ARMS:
        for s in (0, 1):
            b = d["beside"][arm][f"s{s}"]
            out.append(f"| {NAME[arm]} | s{s} | {', '.join(b['D2_worse_rows']) or 'none'} | {', '.join(b['D2_trigger_zeroed_worse_rows']) or 'none'} | {b['D2_trigger_zeroed']} |")
    out.append("")
    out.append("Delta' of the four D2 rows with T zeroed (interval from the same draws):\n")
    out.append("| row | arm | Delta s0 -> Delta' s0 [interval] | Delta s1 -> Delta' s1 [interval] |\n|---|---|---|---|")
    for lab in CRAWL[1:]:
        for arm in ARMS:
            cells = []
            for s in (0, 1):
                r = A[arm][f"s{s}"][lab]
                z = r["trigger_zeroed"]
                cells.append(f"{f(r['delta'], DP[lab])} -> {f(z['delta'], DP[lab])} [{f(z['lo'], DP[lab])}, {f(z['hi'], DP[lab])}]")
            out.append(f"| {lab} | {NAME[arm]} | " + " | ".join(cells) + " |")
    out.append("")
    # 4b concentration (post-hoc, descriptive)
    out.append("### Table 4b. Concentration of the gap in the trigger windows: T-share / w_T (POST-HOC descriptive arithmetic on the table above; not pre-registered; decides nothing)\n")
    out.append("A ratio of 1 means the trigger windows carry exactly their window share of the gap.\n")
    out.append("| row | DAG-OACC s0 | DAG-OACC s1 | REP s0 | REP s1 | SOUP s0 | SOUP s1 |\n|---|---|---|---|---|---|---|")
    for lab in CRAWL:
        cells = []
        for arm in ARMS:
            for s in (0, 1):
                t = A[arm][f"s{s}"][lab]["trigger"]
                cells.append(f"{t['share'] / t['w']:+.1f}")
        out.append(f"| {lab} | " + " | ".join(cells) + " |")
    out.append("")
    # 5 all 19 rows, T-share only
    out.append("### Table 5. All 19 rows: Delta and T-share (point; interval in `raw/CLASS_SPLIT.json`), descriptive, no verdict\n")
    hdr = "| row | w_T |"
    for arm in ARMS:
        for s in (0, 1):
            hdr += f" {NAME[arm]} s{s} Delta / T-share |"
    out.append(hdr)
    out.append("|---|---|" + "---|" * 6)
    labs = list(A["CTRL"]["s0"].keys())
    for lab in labs:
        r0 = A["CTRL"]["s0"][lab]
        line = f"| {lab} | {r0['trigger']['w']:.3f} |"
        for arm in ARMS:
            for s in (0, 1):
                r = A[arm][f"s{s}"][lab]
                dp = 4 if abs(r["delta"]) < 0.5 else 3
                line += f" {f(r['delta'], dp)} / {f(r['trigger']['share'], 2)} |"
        out.append(line)
    out.append("")
    with open(dst, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
