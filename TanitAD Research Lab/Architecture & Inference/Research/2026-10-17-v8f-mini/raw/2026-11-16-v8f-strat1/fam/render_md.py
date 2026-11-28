"""render_md.py -- RESULT_FAM_EXTRA.json -> RESULT_FAM_EXTRA.md (tables only; every cell carries its n; families never pooled).  Usage: python render_md.py [json] [md]"""
import json
import sys

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "F:/TanitAD-scratch/strat1/fam/out/RESULT_FAM_EXTRA.json"
DST = sys.argv[2] if len(sys.argv) > 2 else SRC[:-5] + ".md"
r = json.load(open(SRC, encoding="utf-8"))
arms = r["arms_read"]
DK, TC = r["families"]["LONGITUDINAL-DK"], r["families"]["TACTICAL"]
W = []
w = W.append


def f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def lv(c, d=2):
    if "mean" not in c:
        return f"NOT COMPUTABLE ({c.get('reason', '')})"
    return f"{f(c['mean'], d)} [{f(c['lo'], d)}, {f(c['hi'], d)}]"


def pt(c, d=3):
    if not isinstance(c, dict) or "point" not in c:
        return "n/a"
    return f"{f(c['point'], d)} [{f(c['lo'], d)}, {f(c['hi'], d)}]"


def dl(c, d=3):
    if not isinstance(c, dict) or "delta" not in c or c.get("status") not in (None, "OK") or not np.isfinite(c["delta"]):
        return "n/a"
    sep = "*" if c.get("separated") else ""
    fl = {True: " >F", False: " <=F", None: ""}[c.get("above_floor")]
    return f"{f(c['delta'], d)} [{f(c['lo'], d)}, {f(c['hi'], d)}]{sep}{fl}"


w("# RESULT_FAM_EXTRA -- LONGITUDINAL distance keeping and TACTICAL decision quality on the banked STRAT1 T1-kin dumps\n")
w(f"Work item `W-FAM-STRAT1-TTC-TACCONF` (MM ruling 12). Evidence class: **{r['evidence_class']}**. CPU, zero training, nothing re-dumped.")
w(f"\nWindows: {r['n_windows']:,} in {r['n_episodes']} episodes per arm. Lead-window states (B1 lead block k30): {r['lead_window_states']}.")
w(f"\nEstimator: {r['estimator']}. `*` = interval excludes 0 (necessary, not sufficient for a lever effect); `>F` / `<=F` = |delta| above / not above the seed floor F = |C0_rep - C0_s0| on that cell.")
w("\n| tier key | tier |\n|---|---|")
for k, v in r["tiers"].items():
    w(f"| {k} | {v} |")
w("\n## Controls\n")
c = r["controls"]
w(f"* K-GRID (dump windows = eval grid windows): **{c['grid_matches_dump']}**")
w("* K-HUMAN (the human path scored as the plan reads n_viol exactly 0 and progress exactly 1.0): " + ", ".join(f"{b} {'PASS' if v['PASS'] else 'FAIL'} (n_at_risk {v['n_at_risk']})" for b, v in c["K-HUMAN"].items()))
w(f"* K1 (self-pair reads exactly 0.0 [0.0, 0.0] on every DK and TAC cell): **{'PASS' if c['K1_self_pair']['pass'] else 'FAIL'}**")
k = c["K-T0"]
w("* K-T0 (this module's confusion and goal code on the T0 evaluator's own heads reproduces the evaluator's own tactical block exactly): " + ", ".join(f"{a} {'PASS' if v.get('PASS') else v.get('status', 'FAIL')}" for a, v in k.items()))
w(f"* problems: {c['problems'] or 'none'}")
w("\n## LONGITUDINAL distance keeping (DK) -- per arm, full 0-6 s band\n")
w("Headway = min gap (m) from the rig origin to the lead's rear face over the horizon steps at which the lead is ahead and inside the 2 m corridor; time-gap = headway / v0 (undefined below 0.5 m/s); min-TTC capped at 30 s when never closing, so `n_closing` rides with it. Means over LEAD windows only, with episode-cluster 95 % CIs. Human = the logged path against the same lead.\n")
w("| arm | n lead win | headway m | time-gap s | min-TTC s | n_closing / n_headway | n_lead_lost | DK-VIOL n_viol / n_at_risk | share of at-risk [CI] | progress ratio (clipped) | verdict |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
b0 = DK[arms[0]]["full_0_6s"]
w(f"| human (reference) | {b0['n_lead_windows']} | {lv(b0['human_levels']['headway_min_m'])} | {lv(b0['human_levels']['time_gap_min_s'])} | {lv(b0['human_levels']['min_ttc_s'])} | - | - | - | - | - | - |")
for a in arms:
    d = DK[a]["full_0_6s"]
    v = d["DK-VIOL"]
    sh = v.get("share_of_at_risk_ci") or {}
    shs = f"{f(v.get('share_of_at_risk'), 4)} [{f(sh.get('lo'), 4)}, {f(sh.get('hi'), 4)}]" if v.get("status") != "NOT COMPUTABLE" else "NOT COMPUTABLE"
    w(f"| {a} | {d['n_lead_windows']} | {lv(d['levels']['headway_min_m'])} | {lv(d['levels']['time_gap_min_s'])} | {lv(d['levels']['min_ttc_s'], 1)} | {d['n_closing']} / {d['n_headway']} | {d['n_lead_lost']} | "
      f"{v.get('n_viol')} / {v.get('n_at_risk')} | {shs} | {f((v.get('progress') or {}).get('lead_progress_ratio_clipped'))} | {(v.get('verdict') or {}).get('verdict')} |")
for bn in ("operative_0_2s", "tactical_2_6s"):
    w(f"\n### DK, band {bn} (n lead windows {DK[arms[0]][bn]['n_lead_windows']})\n")
    w("| arm | headway m | time-gap s | min-TTC s | n_closing | DK-VIOL n_viol / n_at_risk | share of at-risk |")
    w("|---|---|---|---|---|---|---|")
    for a in arms:
        d = DK[a][bn]
        v = d["DK-VIOL"]
        w(f"| {a} | {lv(d['levels']['headway_min_m'])} | {lv(d['levels']['time_gap_min_s'])} | {lv(d['levels']['min_ttc_s'], 1)} | {d['n_closing']} | {v.get('n_viol')} / {v.get('n_at_risk')} | {f(v.get('share_of_at_risk'), 4)} |")
w("\n## DK arm deltas (paired, same windows; a positive delta is MORE gap / time / TTC than the comparator, not 'better')\n")
w("| pair | band | headway m | time-gap s | min-TTC s | DK-VIOL share of at-risk (delta, at-risk set) | n used (headway) |")
w("|---|---|---|---|---|---|---|")
for pn, pr in r["pairs"].items():
    for bn, blk in pr["DK"].items():
        L_ = blk["levels"]
        av = (blk.get("DK-VIOL") or {}).get("at_risk") or {}
        avs = "n/a" if av.get("delta") is None else f"{f(av['delta'], 4)} [{f(av['lo'], 4)}, {f(av['hi'], 4)}]{'*' if av.get('separated') else ''}{({True: ' >F', False: ' <=F', None: ''})[(blk.get('DK-VIOL') or {}).get('above_floor')]} (n {av.get('n_at_risk')})"
        w(f"| {pn} | {bn.split('_')[0]} | {dl(L_['headway_min_m'], 2)} | {dl(L_['time_gap_min_s'], 2)} | {dl(L_['min_ttc_s'], 2)} | {avs} | {L_['headway_min_m'].get('n_windows', 'n/a')} |")
w("\nFloors F (|C0_rep - C0_s0|): " + json.dumps({b: {k: (None if v is None else round(v, 4)) for k, v in d.items()} for b, d in r["floors"]["F"]["DK"].items()}))

w("\n## TACTICAL -- the head's decision (selected) vs the v9 label, per arm (T1-kin tick 0)\n")
w("Rows = GT label, columns = head argmax. Accuracy, kappa, macro-F1 over classes present, per-class recall; episode-cluster CIs; majority baseline beside.\n")
w("| arm | head | n | accuracy | kappa | macro-F1 | majority acc / macro-F1 | recall per present class |")
w("|---|---|---|---|---|---|---|---|")
for a in arms:
    for hn, key in (("lat (5)", "TAC-SEL-LAT"), ("lon (7)", "TAC-SEL-LON")):
        b = TC[a][key]
        rec = "; ".join(f"{k} {f(v['point'], 2)}" for k, v in b["recall"].items() if v)
        w(f"| {a} | {hn} | {b['n']} | {pt(b['accuracy'])} | {pt(b['kappa'])} | {pt(b['macro_f1'])} | {f(b['majority_baseline']['accuracy'])} / {f(b['majority_baseline']['macro_f1'])} | {rec} |")
w("\n### Confusion matrices (reference arm C0_s0 and S1_s0; the JSON has every arm)\n")
for a in [x for x in ("C0_s0", "S1_s0") if x in TC]:
    for hn, key, names in (("lat", "TAC-SEL-LAT", TC[a]["TAC-SEL-LAT"]["classes"]), ("lon", "TAC-SEL-LON", TC[a]["TAC-SEL-LON"]["classes"])):
        b = TC[a][key]
        w(f"**{a} {hn}** (rows GT, cols predicted; n {b['n']})\n")
        w("| GT \\ pred | " + " | ".join(names) + " | support |")
        w("|---|" + "---|" * (len(names) + 1))
        for i, row in enumerate(b["confusion_rows_GT_cols_pred"]):
            w(f"| {names[i]} | " + " | ".join(str(x) for x in row) + f" | {b['support'][i]} |")
        w("")
w("## TACTICAL -- goal-token selection (22 tokens, `goal_prob > 0.5` vs `goal_y` where `goal_w`)\n")
w("| arm | macro-F1 [CI] | micro token accuracy | top-1 token is a GT positive (n windows) | tokens with weighted positives |")
w("|---|---|---|---|---|")
for a in arms:
    g = TC[a]["TAC-GOAL"]
    t1 = g["top1_in_positives"]
    npos = sum(1 for v in g["n_positive_per_token"].values() if v > 0)
    w(f"| {a} | {pt(g['macro_f1'])} | {pt(g['micro_token_accuracy'])} | {f(t1.get('mean'))} [{f(t1.get('lo'))}, {f(t1.get('hi'))}] (n {t1.get('n_windows')}) | {npos} |")
w("\nPer-token F1 for C0_s0 and S1_s0 (positives n in brackets): ")
for a in [x for x in ("C0_s0", "S1_s0") if x in TC]:
    g = TC[a]["TAC-GOAL"]
    w(f"* {a}: " + "; ".join(f"{k} {f(v, 2)} ({g['n_positive_per_token'][k]})" for k, v in g["f1_per_token"].items() if v is not None))
w("\n## TACTICAL -- executed manoeuvre (read off the rolled plan) vs the human path's manoeuvre (T1-kin)\n")
w("| arm | lat 3-way accuracy | lat kappa | lat macro-F1 | lon 3-way accuracy | lon kappa | lon macro-F1 | n |")
w("|---|---|---|---|---|---|---|---|")
for a in arms:
    la, lo = TC[a]["TAC-EXEC-LAT"], TC[a]["TAC-EXEC-LON"]
    w(f"| {a} | {pt(la['accuracy'])} | {pt(la['kappa'])} | {pt(la['macro_f1'])} | {pt(lo['accuracy'])} | {pt(lo['kappa'])} | {pt(lo['macro_f1'])} | {la['n']} |")
w("\n## TACTICAL -- declared (head, tick 0) vs executed (plan) lateral agreement (T1-kin)\n")
w("Mapped classes only (LANE_KEEP, TURN_L, TURN_R); LANE_CHANGE_L/R have no kinematic-gate class and are excluded and counted. The longitudinal head vocabulary has no one-to-one map onto the 3 kinematic classes: its cross-table is in the JSON (`TAC-SELEXE-LON`), no scalar is formed.\n")
w("| arm | agreement [CI] | n mapped | n excluded (lane change) |")
w("|---|---|---|---|")
for a in arms:
    s = TC[a]["TAC-SELEXE-LAT"]
    ag = s["agreement"]
    w(f"| {a} | {lv(ag, 3)} | {s['n_mapped']} | {s['n_excluded_lane_change']} |")
w("\n## TACTICAL arm deltas (paired; `>F` above the seed floor)\n")
w("| pair | block | accuracy | kappa | macro-F1 | n windows |")
w("|---|---|---|---|---|---|")
for pn, pr in r["pairs"].items():
    for blk in ("TAC-SEL-LAT", "TAC-SEL-LON", "TAC-EXEC-LAT", "TAC-EXEC-LON"):
        c_ = pr["TAC"][blk]
        w(f"| {pn} | {blk} | {dl(c_.get('accuracy'))} | {dl(c_.get('kappa'))} | {dl(c_.get('macro_f1'))} | {(c_.get('accuracy') or {}).get('n_windows', 'n/a')} |")
    g = pr["TAC"]["TAC-GOAL"]
    w(f"| {pn} | TAC-GOAL | micro {dl(g.get('goal_micro_token_accuracy'))} | - | macro-F1 {dl(g.get('goal_macro_f1'))} | {(g.get('goal_macro_f1') or {}).get('n_windows', 'n/a')} |")
    s = pr["TAC"]["TAC-SELEXE-LAT"]
    w(f"| {pn} | TAC-SELEXE-LAT | agreement {dl(s)} | - | - | {s.get('n_windows', 'n/a')} |")
w("\n## NOT COMPUTABLE\n")
w("| item | reason | n |\n|---|---|---|")
a0 = arms[0]
w(f"| TAC-ANCHOR (anchor / candidate selection) | {TC[a0]['TAC-ANCHOR']['reason']} | {TC[a0]['TAC-ANCHOR']['n']} |")
w("| TAC-SELEXE-LON scalar agreement | the head vocabulary (HOLD, CREEP, STOP, FOLLOW, DECELERATE, ACCELERATE, KEEP) has no one-to-one map onto brake_stop / steady / accelerate; counts only | see JSON |")
w("| DK on windows without a lead (NO_LEAD, NOT_STRAIGHT, NO_LABEL) | counted, never scored as free flow: " + str(r["lead_window_states"]) + " | per state |")
w("| DK closed-reaction to the arm | the lead follows its logged future and does not react to the arm | tier caveat |")
w("\n## Beside rows (not controls)\n")
w("| arm | lat acc T1 tick 0 / T0 real nav | lon acc T1 tick 0 / T0 real nav | goal macro-F1 T1 tick 0 / T0 real nav |")
w("|---|---|---|---|")
for a, v in r["beside_rows"]["T1_tick0_vs_T0_real_nav"].items():
    w(f"| {a} | {f(v['lat_accuracy_t1_tick0'])} / {f(v['lat_accuracy_t0_real_nav'])} | {f(v['lon_accuracy_t1_tick0'])} / {f(v['lon_accuracy_t0_real_nav'])} | {f(v['goal_macro_f1_t1_tick0'])} / {f(v['goal_macro_f1_t0_real_nav'])} |")
w("\nThe two columns differ because T1 feeds the dead-reckoned nav and T0 the real nav.")
w("\nTAC floors F: " + json.dumps({b: {k: round(v, 4) for k, v in d.items()} for b, d in r["floors"]["F"]["TAC"].items()}))
w(f"\nProvenance: lead_metrics {r['provenance']['lead_metrics']}; ci {r['provenance']['ci']}; code of record {r['provenance']['code_of_record']}; lead block {r['provenance']['lead_block_sha256']}; reader sha256 {r['provenance']['reader_sha256']}; wall {r['wall_s']} s.")
open(DST, "w", encoding="utf-8").write("\n".join(W) + "\n")
print("WROTE", DST)
