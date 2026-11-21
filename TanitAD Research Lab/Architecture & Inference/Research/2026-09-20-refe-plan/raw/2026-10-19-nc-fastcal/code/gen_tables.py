"""NC-FASTCAL: render RESULT_tables.md from result.json (+ posthoc_diag.json).  Every number in the tables comes from the JSON; nothing is typed by hand.  LF output."""
from __future__ import annotations

import json
import os
import sys

RUN = "C:/Users/Admin/par18/ncfastcal/run/results"
STRATA = ("S", "Q1", "Q2", "Q3", "Q4")


def ci(c, n=4, sign=True):
    f = f"{{:{'+' if sign else ''}.{n}f}}"
    return f"{f.format(c['point'])} [{f.format(c['lo'])}, {f.format(c['hi'])}]"


def main() -> int:
    r = json.load(open(f"{RUN}/result.json", encoding="utf-8"))
    ph = json.load(open(f"{RUN}/posthoc_diag.json", encoding="utf-8"))
    a = r["arms"]
    L = []
    w = L.append
    p2 = a["P2_unvetoed"]
    w("# NC-FASTCAL: result tables (generated from `raw/results/result.json` and `posthoc_diag.json` by `code/gen_tables.py`)\n")
    w(f"PREREG sha256 `{r['prereg_sha256']}`; {r['n_sets']} sets, {r['n_logs']} logs, {r['n_tokens']} tokens; paired log-cluster bootstrap B = {r['bootstrap']['B']}, seed {r['bootstrap']['seed']}.\n")
    w("## 1. The registered decision (PRIMARY arm C_s: the EP-led composite with the OOF per-stratum Platt NC)\n")
    w("| seed | dNC vs P2 (NC-unsafe rate), point [lo, hi] | (i) upper < +0.001 | dPDMS x100 vs P2, point [lo, hi] | (ii) lower >= 0 | seed meets both |")
    w("|---|---|---|---|---|---|")
    for sd in ("0", "1"):
        d = r["decision"]["per_seed"][sd]
        i_, ii_ = d["cond_i_upper_dNC_lt_0.001"], d["cond_ii_lower_dPDMS_ge_0"]
        w(f"| {sd} | {ci(d['dNC'])} | {i_} | {ci(d['dPDMS_x100'], 3)} | {ii_} | {i_ and ii_} |")
    w(f"\n**RECOVERS = {r['decision']['recovers']}** (both seeds must meet both conditions). C-NULL: RECOVERS fired on {r['null']['fires']} of {r['null']['of']}; VOID = {r['null']['void']}. **Final: {r['final_outcome']}.**\n")
    w("## 2. Every arm against P2's rule (rates over all sets; d = arm minus P2's unvetoed pick, paired, same draws)\n")
    w("| arm | NC-unsafe rate | dNC | DAC-unsafe rate | dDAC (no authority) | pick-level PDMS x100 | dPDMS x100 | picks changed vs P2 |")
    w("|---|---|---|---|---|---|---|---|")
    w(f"| P2 (rule on P2's rows) | {p2['nc_unsafe_rate']['point']:.4f} [{p2['nc_unsafe_rate']['lo']:.4f}, {p2['nc_unsafe_rate']['hi']:.4f}] | | {p2['dac_unsafe_rate']['point']:.4f} | | {p2['pdms_x100']['point']:.3f} | | |")
    names = {"U": "U uncalibrated EP-led composite (beside)", "C": "**C calibrated composite (PRIMARY)**", "G": "G global Platt (descriptive)"}
    for sd in (0, 1):
        for k in ("U", "C", "G"):
            x = a[f"{k}_s{sd}"]
            w(f"| {names[k]}, seed {sd} | {x['nc_unsafe_rate']['point']:.4f} | {ci(x['d_nc_unsafe_rate'])} | {x['dac_unsafe_rate']['point']:.4f} | {ci(x['d_dac_unsafe_rate'])} | {x['pdms_x100']['point']:.3f} | {ci(x['d_pdms_x100'], 3)} | {x['n_changed_vs_p2']} |")
    x = a["PC"]
    w(f"| PC: P2's own EP with the calibrated NC (descriptive) | {x['nc_unsafe_rate']['point']:.4f} | {ci(x['d_nc_unsafe_rate'])} | {x['dac_unsafe_rate']['point']:.4f} | {ci(x['d_dac_unsafe_rate'])} | {x['pdms_x100']['point']:.3f} | {ci(x['d_pdms_x100'], 3)} | {x['n_changed_vs_p2']} |")
    w(f"\nPicks changed by the calibration inside the composite (C vs U): seed 0 {r['diagnostics']['n_picks_changed_C_vs_U']['seed0']}, seed 1 {r['diagnostics']['n_picks_changed_C_vs_U']['seed1']} of {r['n_sets']} sets.\n")
    w("## 3. The calibration: P2's NC head is already near-calibrated inside every speed stratum\n")
    w("Fitted (a, b) by fold (5 folds), `z' = a z + b`; held-out BCE pooled over the five held-out folds, before (raw logit) and after (calibrated):\n")
    w("| stratum | rows | a by fold | b by fold | held-out BCE before | after | improvement (nats) |")
    w("|---|---|---|---|---|---|---|")
    P = r["fit"]["params_a_b_by_fold_stratum"]
    for s, n in enumerate(STRATA):
        h = r["diagnostics"]["heldout_bce_by_stratum"][n]
        w(f"| {n} | {h['rows']} | {', '.join(f'{P[k][s][0]:.3f}' for k in range(5))} | {', '.join(f'{P[k][s][1]:+.3f}' for k in range(5))} | {h['bce_before']:.5f} | {h['bce_after']:.5f} | {h['improvement']:+.5f} |")
    g = r["diagnostics"]["heldout_bce_global_platt"]["all_proposals"]
    w(f"\nGlobal Platt (one stratum), held-out BCE {g['bce_before']:.5f} -> {g['bce_after']:.5f} ({g['improvement']:+.5f}). Fit failures: {r['fit']['n_fit_failed']} of {len(r['fit']['diag'])} (stratum, fold) fits; max Newton iterations {max(d['iters'] for d in r['fit']['diag'])}, max |gradient| {max(d['grad_inf'] for d in r['fit']['diag']):.2e}.\n")
    w("## 4. C-NULL: strata codes permuted within token (5 replicates; a replicate FIRES iff it reads RECOVERS)\n")
    w("| rep | perm seed | RECOVERS | (i) alone on both seeds | seed 0: dNC upper / dPDMS lower | seed 1: dNC upper / dPDMS lower | mean a by stratum S, Q1..Q4 |")
    w("|---|---|---|---|---|---|---|")
    for rp in r["null"]["replicates"]:
        s0, s1 = rp["per_seed"]["0"], rp["per_seed"]["1"]
        w(f"| {rp['rep']} | {rp['perm_seed']} | {rp['recovers']} | {rp['cond_i_only_both_seeds']} | {s0['d_nc']['hi']:+.4f} / {s0['d_pdms_x100']['lo']:+.3f} | {s1['d_nc']['hi']:+.4f} / {s1['d_pdms_x100']['lo']:+.3f} | {', '.join(f'{v:.3f}' for v in rp['params_mean_a_by_stratum'])} |")
    w(f"\nFires: **{r['null']['fires']} of {r['null']['of']}**; condition (i) alone on both seeds: {r['null']['cond_i_only_fires']} of {r['null']['of']}; VOID iff fires >= 2: **{r['null']['void']}**.\n")
    w("## 5. Q1-style numbers for the calibrated composite (descriptive, no authority)\n")
    w("| seed | n sets where C is NC-unsafe and P2 is not | of which own calibrated NC sigmoid < 0.5 | median calibrated sigmoid there (q25, q75) | the uncalibrated composite: n, over, median (q25, q75) |")
    w("|---|---|---|---|---|")
    for sd in ("0", "1"):
        q, qu = r["diagnostics"]["q1_like_C"][f"seed{sd}"], r["diagnostics"]["q1_like_C"][f"seed{sd}_uncalibrated_for_reference"]
        qq, qv = q["sigmoid_nc_at_A_picks_quantiles_min_q25_med_q75_max"], qu["sigmoid_nc_at_A_picks_quantiles_min_q25_med_q75_max"]
        w(f"| {sd} | {q['n_A']} | {q['n_over']} | {qq[2]:.3f} ({qq[1]:.3f}, {qq[3]:.3f}) | {qu['n_A']}, {qu['n_over']}, {qv[2]:.3f} ({qv[1]:.3f}, {qv[3]:.3f}) |")
    w("\n## 6. POST-HOC, DESCRIPTIVE (written after `result.json` existed; `posthoc_diag.py`; no authority): predicted vs observed unsafe mass at the SELECTED picks\n")
    w("Predicted unsafe mass = sum over picks of (1 - NC sigmoid); observed = sum of (1 - NC label), labels 0 / 0.5 / 1 read as probabilities of safe. Ratio observed / predicted with the paired log-cluster bootstrap interval (same draws). Below 1 = the row is not overconfident at its picks.\n")
    w("| picks of | n | label-0 picks | predicted mass | observed mass | observed / predicted |")
    w("|---|---|---|---|---|---|")
    for k, v in ph["arms"].items():
        w(f"| {k} | {v['n']} | {v['nc_unsafe_picks_label0']} | {v['predicted_unsafe_mass']:.1f} | {v['observed_unsafe_mass']:.1f} | {ci(v['ratio_observed_over_predicted'], 3, sign=False)} |")
    w("\nBy fixed bins of the RAW NC sigmoid at the pick (n, observed unsafe mass, predicted mass, label-0 picks):\n")
    bins = list(next(iter(ph["arms"].values()))["bins"].keys())
    w("| picks of | " + " | ".join(bins) + " |")
    w("|---|" + "---|" * len(bins))
    for k, v in ph["arms"].items():
        w(f"| {k} | " + " | ".join(f"{v['bins'][b]['n']}, {v['bins'][b]['observed_unsafe_mass']:.1f}, {v['bins'][b]['predicted_unsafe_mass_raw']:.1f}, {v['bins'][b]['nc_unsafe_picks_label0']}" for b in bins) + " |")
    with open(f"{RUN}/RESULT_tables.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
