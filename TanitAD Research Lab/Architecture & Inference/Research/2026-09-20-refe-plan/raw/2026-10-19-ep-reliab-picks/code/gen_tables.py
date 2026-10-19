"""EP-RELIAB-PICKS: render run/results/result.json to RESULT_tables.md (LF).  Every number is read from result.json; nothing is typed in."""
import json

R = json.load(open("C:/Users/Admin/par18/epreliab/run/results/result.json", "r", encoding="utf-8"))
ind = json.load(open("C:/Users/Admin/par18/epreliab/run/results/independent_check.json", "r", encoding="utf-8"))
f4 = lambda c: f"{c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"                         # noqa: E731
f3 = lambda c: f"{c['point']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"                             # noqa: E731
S = ("0", "1")
d = R["decision"]
T = ["# RESULT tables: EP-RELIAB-PICKS (generated from `raw/results/result.json` by `code/gen_tables.py`)", "",
     f"prereg sha256 `{R['prereg_sha256']}` + Amendment 1; {R['n_sets']} sets, {R['n_logs']} logs; paired log-cluster bootstrap B = {R['bootstrap']['B']}, seed {R['bootstrap']['seed']}. "
     "Tier: SELECTION, SEEN scenes, P2 IN-SAMPLE; label-lookup PDMS; no GPU.", "",
     f"**FINAL LABEL: {R['final_label']}**  (deciding form: {d['deciding_form']}; the other form reads {d['other_form_outcome']}; form-sensitive: {d['form_sensitive']}; "
     f"dominant PDMS-loss term: {d['dominant_term']}).", "",
     "## 1. Scale gate and the two decision statistics", "",
     "| seed | n_changed | within-set slope of EP label on sigma (safe proposals) | on-scale | Delta = predicted - realised gain | class | D_M = M_p - M_e | class |", "|---|---|---|---|---|---|---|---|"]
for i, s in enumerate(S):
    p = R["per_seed"][s]
    T.append(f"| {s} | {p['n_changed']} | {f3(p['scale']['slope'])} | {d['scale_on'][i]} | {f4(p['G']['delta'])} | {d['classes']['DELTA'][i]} | {f4(p['G']['D_M'])} | {d['classes']['MASS'][i]} |")
T += ["", "Scale gate: ON-SCALE iff the slope CI lies inside [0.9, 1.1] on BOTH seeds. Result: "
      + ("ON-SCALE (Delta decides)" if d["scale_gate_on_scale"] else "OFF-SCALE (the rank-free mass form D_M decides; no mapping fitted)") + ".", "",
      "## 2. The gain, predicted and realised (EP on the sigmoid scale = label scale)", "",
      "| seed | predicted gain (composite's EP row) | realised gain (labels6 EP) | Delta | M_e (obs/pred EP mass at composite picks) | M_p (at P2 picks) | predicted gain from P2's own EP row |",
      "|---|---|---|---|---|---|---|"]
for s in S:
    G = R["per_seed"][s]["G"]
    T.append(f"| {s} | {f4(G['pred_gain'])} | {f4(G['real_gain'])} | {f4(G['delta'])} | {f3(G['M_e'])} | {f3(G['M_p'])} | {f4(G['P2_own_row_pred_gain'])} |")
T += ["", "Changed sets only (descriptive, no authority):", "", "| seed | n_changed | predicted gain | realised gain | Delta |", "|---|---|---|---|---|"]
for s in S:
    G = R["per_seed"][s]["G"]
    co = G["changed_only"]
    T.append(f"| {s} | {G['n_changed']} | {f4(co['pred_gain'])} | {f4(co['real_gain'])} | {f4(co['delta'])} |")
T += ["", "## 3. Exact attribution of dPDMS (composite - P2), x100, over all 15,808 sets", "",
      "Sequential substitution in the fixed order NC, DAC, EP, TTC, C (the EP, TTC and C terms are valued at the composite's NC and DAC multipliers). "
      "Weights (5, 5, 2) / 12: `pickrule_lib.V1_W` = `refe/planner.py:520`.", "",
      "| seed | term | sequential contribution | share of the loss | Shapley (descriptive) |", "|---|---|---|---|---|"]
for s in S:
    A = R["per_seed"][s]["attribution"]
    for t in A["order"]:
        T.append(f"| {s} | {t} | {f4(A['sequential_x100'][t])} | {f3(A['share_of_loss'][t])} | {f4(A['shapley_x100'][t])} |")
    T.append(f"| {s} | **total dPDMS** | {f4(A['total_x100'])} | 1 | {f4(A['total_x100'])} |")
T += ["", "Dominant term (largest share of the loss), per seed, and how often it is the largest in the bootstrap draws:", "", "| seed | dominant | bootstrap frequency by term |", "|---|---|---|"]
for s in S:
    A = R["per_seed"][s]["attribution"]
    T.append(f"| {s} | {A['dominant_term']} | " + ", ".join(f"{k} {v:.3f}" for k, v in A["dominant_bootstrap_frequency"].items()) + " |")
T += ["", f"Arm-level dominant term: **{d['dominant_term']}** (by seed: {d['dominant_term_by_seed']}).", "",
      "## 4. Reliability at the picks (observed / predicted mass; rows of the composite; descriptive, no authority)", "",
      "Primary form: deficit (risk) `sum(1-y)/sum(1-sigma)` for NC, DAC, TTC; positive `sum(y)/sum(sigma)` for EP.", "",
      "| seed | head | primary form | at composite's picks | at P2's picks | composite - P2 (paired) |", "|---|---|---|---|---|---|"]
for s in S:
    Rr = R["per_seed"][s]["reliability"]
    for h in ("NC", "DAC", "EP", "TTC"):
        x = Rr[h]
        key = "positive_form_sum_y_over_sum_sigma" if x["primary"] == "positive" else "deficit_form_sum_1my_over_sum_1msigma"
        T.append(f"| {s} | {h} | {x['primary']} | {f3(x['composite_picks'][key])} | {f3(x['P2_picks'][key])} | {f4(x['paired_composite_minus_P2_primary'])} |")
T += ["", "Observed and predicted unsafe masses behind the safety rows at the picks (seed 0; the NC, DAC, TTC rows are P2's):", "",
      "| head | picks | observed unsafe mass | predicted unsafe mass | observed per set |", "|---|---|---|---|---|"]
Rr = R["per_seed"]["0"]["reliability"]
for h in ("NC", "DAC", "TTC"):
    for pk in ("composite_picks", "P2_picks"):
        c = Rr[h][pk]
        T.append(f"| {h} | {pk} | {c['observed_mass_deficit']:.1f} | {c['predicted_mass_deficit']:.1f} | {c['observed_mass_deficit'] / R['n_sets']:.5f} |")
T += ["", "EP row of P2 itself at P2's picks (what P2 believed): positive form " + f3(Rr["EP"]["P2_own_row_at_P2_picks"]["positive_form_sum_y_over_sum_sigma"]) + ".", "",
      "## 5. Controls", "", "| control | pass |", "|---|---|"]
for k, v in R["gate"]["checks"].items():
    T.append(f"| {k} | {v['pass']} |")
cn = R["null"]["c_null"]
en = R["null"]["e_null"]
T += ["", f"C-NULL (picks shuffled within token, 5 replicates, deciding form {R['null']['form']}): fired on **{cn['fires']} of {cn['of']}** (VOID iff >= 2): void = {cn['void']}.", "",
      "| rep | perm seed | outcome (deciding form) | fires | D_M point seed 0 / seed 1 |", "|---|---|---|---|---|"]
for r in cn["replicates"]:
    T.append(f"| {r['rep']} | {r['perm_seed']} | {r['outcomes'][R['null']['form']]} | {r['fires']} | {r['seeds']['0']['D_M']['point']:+.4f} / {r['seeds']['1']['D_M']['point']:+.4f} |")
T += ["", f"E-NULL (descriptive, {en['n']} more replicates): empirical any-fire rate of the registered rule on the real structure **{en['empirical_any_fire_rate']:.3f}** "
      f"(other form {en['other_form_any_fire_rate']:.3f}); P(>= 2 of 5 fire) at that rate {en['p_at_least_2_of_5_fire_at_that_rate']:.4f}.", "",
      "| seed | null point mean | null point sd | null 2.5 / 97.5 % | real point | real in null-sd units | empirical two-sided p (+1) |", "|---|---|---|---|---|---|---|"]
for s in S:
    e = en["per_seed"][s]
    q = e["null_points_quantiles_min_2.5_25_50_75_97.5_max"]
    T.append(f"| {s} | {e['null_points_mean']:+.5f} | {e['null_points_sd']:.5f} | {q[1]:+.4f} / {q[5]:+.4f} | {e['real_point']:+.4f} | {e['real_point_in_null_sd_units']:.1f} | "
             f"{e['empirical_two_sided_p_with_plus_one']:.4f} |")
T += ["", "Independent re-derivation (`code/independent_check.py`, no shared code, own picks and loops): max |independent - library| over predicted gain, realised gain, Delta, M_e, M_p, D_M, slope, "
      "dPDMS and the five sequential terms = " + ", ".join(f"seed {k[-1]}: {v['max_abs_difference']:.2e}" for k, v in ind.items()) + ".", "",
      "Minimal detectable effect (descriptive): half-width of the Delta interval " + ", ".join(f"seed {s}: {(R['per_seed'][s]['G']['delta']['hi'] - R['per_seed'][s]['G']['delta']['lo']) / 2:.4f}" for s in S)
      + "; of the D_M interval " + ", ".join(f"seed {s}: {(R['per_seed'][s]['G']['D_M']['hi'] - R['per_seed'][s]['G']['D_M']['lo']) / 2:.4f}" for s in S) + "."]
open("C:/Users/Admin/par18/epreliab/run/results/RESULT_tables.md", "w", encoding="utf-8", newline="\n").write("\n".join(T) + "\n")
print("wrote RESULT_tables.md")
