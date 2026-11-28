"""Render the tables of RESULT.md from run/results/census.json (no numbers are typed by hand).  python gen_tables.py > RESULT_tables.md"""
from __future__ import annotations

import json
import sys

RUN = "C:/Users/Admin/par18/fastneg/run"
ROWS = ("nc", "dac")
STR = ("S", "Q1", "Q2", "Q3", "Q4")


def ci(d, p=4):
    return f"{d['point']:.{p}f} [{d['lo']:.{p}f}, {d['hi']:.{p}f}]"


def pct(d, p=2):
    return f"{100 * d['point']:.{p}f} % [{100 * d['lo']:.{p}f}, {100 * d['hi']:.{p}f}]"


def main():
    R = json.load(open(f"{RUN}/results/census.json", encoding="utf-8"))
    o = []
    w = o.append
    st = R["strata"]
    w("## T1. Strata (written by the label-free script before any label was read)\n")
    c = st["cut_points"]
    w(f"cut points of r = v_p / v_b over the {sum(v for k, v in st['rows_per_stratum'].items() if k != 'S'):,} proposals of the non-S sets: **c25 = {c['c25']:.6f}, c50 = {c['c50']:.6f}, c75 = {c['c75']:.6f}**. "
      f"S (v_b < 0.5 m/s) = {st['n_S_sets']} of {st['n_sets']:,} sets.\n")
    w("| stratum | rows |\n|---|---|")
    for k in STR:
        w(f"| {k} | {st['rows_per_stratum'][k]:,} |")
    w("\n## T2. (a) Label support per stratum (paired log-cluster bootstrap, 10,000, seed 20260927; MEASURED, selection tier)\n")
    w("| stratum | rows | NC = 0 | NC = 0.5 | DAC = 0 | NC = 0 or DAC = 0 | unsafe share of rows | share of ALL unsafe rows |\n|---|---|---|---|---|---|---|---|")
    for k in STR:
        a = R["a_support"][k]
        w(f"| {k} | {a['rows']:,} | {a['nc_eq_0']:,} | {a['nc_eq_0.5']:,} | {a['dac_eq_0']:,} | {a['unsafe_any_nc0_or_dac0']:,} | {pct(a['unsafe_share_of_rows'])} | {pct(a['share_of_all_unsafe_rows'])} |")

    def auc_table(title, src, key=None, groups=("S", "Q1", "Q2", "Q3", "Q4", "Q123")):
        w(f"\n{title}\n")
        w("| row | " + " | ".join(groups) + " | pairs in F | F - Q1-3 |\n|---|" + "---|" * (len(groups) + 2))
        for row in ROWS:
            d = src[row] if key is None else src[key][row]
            w(f"| {row.upper()} | " + " | ".join(ci(d["auc"][g]) for g in groups) + f" | {d['auc']['Q4']['pairs']:,.0f} | **{ci(d['F_minus_Q123'])}** |")
    auc_table("## T3. (b)(i) Within-token AUC of P2's FROZEN rows (in-sample for P2; both members of a pair in the stratum; Q1-3 = Q1 u Q2 u Q3)", R["b_i_p2"])
    b2 = R["b_ii_probe"]
    if b2["status"] == "PRESENT":
        w(f"\nHEADROW-PROBE OOF file: sha256 `{b2['file_sha256'][:16]}...`, set keys equal to the bank's; cross-checks {json.dumps(b2['cross_checks'])}")
        for arm in sorted(b2["arms"]):
            auc_table(f"### T4. (b)(ii) HEADROW-PROBE {arm} OOF within-token AUC", b2["arms"][arm])
    else:
        w("\n## T4. (b)(ii) HEADROW-PROBE OOF: **MISSING**\n")
    cr = R["c_refit"]
    if cr["status"] == "PRESENT":
        w("\n## T5. (c) Our L-arm refit, OOF within-token AUC of the fast stratum F = Q4 at F-training-row fractions, per row and seed\n")
        w("| row | seed | F AUC 25 % | F AUC 50 % | F AUC 100 % | Q1-3 AUC 100 % | F - Q1-3 at 100 % | deficit call | F(100) - F(25) | slope call |\n|---|---|---|---|---|---|---|---|---|---|")
        for row in ROWS:
            for s in (0, 1):
                d = cr["rows"][row][f"s{s}"]
                w(f"| {row.upper()} | {s} | {ci(d['f25']['auc']['Q4'])} | {ci(d['f50']['auc']['Q4'])} | {ci(d['f100']['auc']['Q4'])} | {ci(d['f100']['auc']['Q123'])} | **{ci(d['F_minus_Q123_at_100'])}** | {d['deficit_call']} | "
                  f"**{ci(d['slope_F_100_minus_25'])}** | {d['slope_call']} |")
        w("\nSecondary slopes (descriptive): F(50) - F(25), F(100) - F(50), Q1-3 (100) - Q1-3 (25), and the sensitivity F - Q1-3 with unsafe := label < 1.\n")
        w("| row | seed | F(50) - F(25) | F(100) - F(50) | Q1-3(100) - Q1-3(25) | sensitivity: F - Q1-3 (unsafe := label < 1) |\n|---|---|---|---|---|---|")
        for row in ROWS:
            for s in (0, 1):
                d = cr["rows"][row][f"s{s}"]
                w(f"| {row.upper()} | {s} | {ci(d['slope_F_50_minus_25'])} | {ci(d['slope_F_100_minus_50'])} | {ci(d['slope_Q123_100_minus_25'])} | {ci(d['sensitivity_unsafe_lt_1_F_minus_Q123_at_100'])} |")
        w("\n## T6. The registered rule (parent s5) on our 100 % L refit\n")
        w("| row | deficit by seed | slope by seed | row class |\n|---|---|---|---|")
        for row in ROWS:
            r = R["rule"]["rows"][row]
            w(f"| {row.upper()} | {r['deficit_by_seed']} | {r['slope_by_seed']} | **{r['class']}** |")
        if "robustness_probe_L" in R["rule"]:
            w("\nRobustness: the same rule with the DEFICIT test read from HEADROW-PROBE's L OOF (slope from our refits):\n")
            w("| row | deficit by seed | slope by seed | row class |\n|---|---|---|---|")
            for row in ROWS:
                r = R["rule"]["robustness_probe_L"][row]
                w(f"| {row.upper()} | {r['deficit_by_seed']} | {r['slope_by_seed']} | {r['class']} |")
    n = R["c0_null"]
    if n["status"] == "PRESENT":
        w("\n## T7. (c0) Powered null: two independent 100 % L refits differing only in the batch-order seed; slope test AUC_F(A) - AUC_F(B), fires iff lower CI > +0.01 on BOTH seeds\n")
        w("| row | rep | seed 0: A - B | seed 1: A - B | fires |\n|---|---|---|---|---|")
        for row in ROWS:
            for rp in n["rows"][row]["replicates"]:
                c0, c1 = rp["ci_A_minus_B_by_seed"]["s0"], rp["ci_A_minus_B_by_seed"]["s1"]
                w(f"| {row.upper()} | {rp['rep']} | {ci(c0, 5)} | {ci(c1, 5)} | {rp['fires']} |")
        w("\n| row | fires | fire rate | max abs point |\n|---|---|---|---|")
        for row in ROWS:
            r = n["rows"][row]
            w(f"| {row.upper()} | {r['n_fire']} of {len(r['replicates'])} | {r['fire_rate']:.2f} | {r['max_abs_point']:.5f} |")
        w(f"\nCensus VOID by the null: **{n['void']}**.")
    if "outcome" in R:
        o_ = R["outcome"]
        w(f"\n## T8. Outcome (A1.6 precedence)\n\n**{o_['outcome']}** (NC = {o_['nc']}, DAC = {o_['dac']}, void = {o_['void']}; null: {o_['null_status']}). Committed next arm: {o_['next_arm']}")
        if "outcome_robustness_probe_L" in R:
            w(f"\nRobustness (deficit from the probe's L OOF): {R['outcome_robustness_probe_L']['outcome']}; agrees with the primary: {R['outcome_agrees_with_probe_L']}.")
    pl = R["d_placebo"]
    if pl["status"] == "PRESENT":
        w("\n## T9. D-PLACEBO (no authority): Q2 training rows at 25 %, F at 100 %; F's AUC change vs the 100 % refit, next to the F-subsample's own\n")
        w("| row | seed | F AUC (placebo) | F(100) - F(placebo) | F(100) - F(25, F-subsample) |\n|---|---|---|---|---|")
        for row in ROWS:
            for s in (0, 1):
                d = pl["rows"][row][f"s{s}"]
                w(f"| {row.upper()} | {s} | {ci(d['F_auc_placebo'])} | {ci(d['F_100_minus_placebo'])} | {ci(d['F_100_minus_F25_subsample'])} |")
    cd = R["d_cold"]
    if cd["status"] == "PRESENT":
        w("\n## T10. D-COLD (no authority): the same fits from a ZERO start (no stored logit)\n")
        w("| row | seed | F AUC 25 % | F AUC 50 % | F AUC 100 % | F - Q1-3 at 100 % | would read | F(100) - F(25) | would read |\n|---|---|---|---|---|---|---|---|---|")
        for row in ROWS:
            for s in (0, 1):
                d = cd["rows"][row][f"s{s}"]
                w(f"| {row.upper()} | {s} | {ci(d['f25']['Q4'])} | {ci(d['f50']['Q4'])} | {ci(d['f100']['Q4'])} | {ci(d['F_minus_Q123_at_100'])} | {d['would_read_deficit']} | {ci(d['slope_F_100_minus_25'])} | {d['would_read_slope']} |")
    w("\n## T11. Controls\n")
    cc = R["controls"]
    if "C_REPRO_our_L100_vs_probe_L" in cc:
        w("C-REPRO (our 100 % L refit vs HEADROW-PROBE's L OOF logits):\n")
        w("| row_seed | max abs logit diff | mean abs diff | fraction bit-identical |\n|---|---|---|---|")
        for k, v in cc["C_REPRO_our_L100_vs_probe_L"].items():
            w(f"| {k} | {v['max_abs_logit_diff']:.3e} | {v['mean_abs_logit_diff']:.3e} | {v['fraction_bit_identical']:.4f} |")
    w("\nC-AUC instrument (real labels and strata; random scores must bracket 0.5, oracle scores must read 1.0):\n")
    w("| row | random scores F | random scores Q1-3 | oracle F | oracle Q1-3 |\n|---|---|---|---|---|")
    for row in ROWS:
        v = cc["C_AUC_instrument"][row]
        w(f"| {row.upper()} | {ci(v['random_scores_F'])} | {ci(v['random_scores_Q123'])} | {v['oracle_scores_F']:.4f} | {v['oracle_scores_Q123']:.4f} |")
    print("\n".join(o))


if __name__ == "__main__":
    main()
