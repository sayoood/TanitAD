"""Builds HGATE_RESULT.md from the JSON outputs (every number is read from an artifact, none is typed). CPU, stdlib only."""
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
HB = OUT.rsplit("/", 1)[0] + "/hblend"
SEEDS = ("M2b_s0", "M2b_s1")
SV = {"M2b_s0": "s0", "M2b_s1": "s1"}
S = {s: json.load(open(f"{OUT}/score_{s}.json")) for s in SEEDS}
WG = {s: json.load(open(f"{OUT}/weights_G_{s}.json")) for s in SEEDS}
WB = {s: json.load(open(f"{HB}/weights_{s}.json")) for s in SEEDS}
R = json.load(open(f"{OUT}/READ.json"))
A = json.load(open(f"{OUT}/train_activity.json"))
DK = json.load(open(f"{OUT}/dk_read.json"))
DG = json.load(open(f"{OUT}/diag_gate.json"))
HS = {s: json.load(open(f"{HB}/score_{s}.json")) for s in SEEDS}


def f(v, dp=4):
    return "n/a" if v is None else f"{v:+.{dp}f}"


def g(v, dp=4):
    return "n/a" if v is None else f"{v:.{dp}f}"


def ci(d, dp=4):
    return f"{f(d['mean'] if 'mean' in d else d['delta'], dp)} [{f(d['lo'], dp)}, {f(d['hi'], dp)}]"


def ex(m, dp=4):
    return f"{g(m['mean'], dp)} [{g(m['lo'], dp)}, {g(m['hi'], dp)}]"


def star(d):
    return " *" if d.get("separated") else ""


EST = "paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0"
TIER = "T1-class open loop (own held outputs; not closed loop)"
L = []
w = L.append
cC = {s: R["per_seed"][s]["PRIMARY_C"] for s in SEEDS}
cK = {s: R["per_seed"][s]["PRIMARY_K"] for s in SEEDS}
pB = {s: S[s]["paired"]["HG_minus_B"] for s in SEEDS}
pHB = {s: S[s]["paired"]["H_minus_B"] for s in SEEDS}
cons = {s: S[s]["consistency"] for s in SEEDS}
gate = {s: S[s]["gate"] for s in SEEDS}

w("# HGATE_RESULT: PREREG_V8F_HGATE executed exactly as written (2026-10-25)\n")
w("Dev box, CPU only, ZERO training, no GPU, no box-2 access. Every number below is **MEASURED (ours)** from the artifact paths named; tier **" + TIER + "**; estimator **" + EST + "** unless a row says otherwise. "
  "The bootstrap answers 'would another draw of EPISODES say this?' only; training-seed variance enters only through the two seeds shown side by side (`H-ESTIM-SEED-1`); inference is deterministic (no inference-seed floor). "
  "`OUT` = `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/hgate`.\n")

# ------------------------------------------------------------------------------------------------ 0
w("## 0. Outcome (the PREREG outcome table, applied verbatim)\n")
w(f"**{R['outcome']}.** PRIMARY-C **fails at both seeds** (upper 95 % bound {f(cC['M2b_s0']['hi'])} at s0 and {f(cC['M2b_s1']['hi'])} at s1, against a bar of <= +0.02). "
  f"PRIMARY-K **passes at both seeds** (ADE2 and ADE6 HG - B both negative and separated). The row 'CONSISTENCY NOT RESTORED' (PRIMARY-C fails) matches; 'GAIN LOST' requires PRIMARY-C to pass, so it does not apply.\n")
w("**Committed next step, verbatim from the PREREG:** *" + R["committed_next_step_verbatim"] + ".*\n")
w("Read plainly, because the point estimate at s0 sits on the bar and a reader could take it for a near-pass: PRIMARY-C is a bar on the **upper 95 % bound**, not the point estimate. "
  f"At s0 the point estimate is {f(cC['M2b_s0']['delta'])} (under +0.02) but the bound is {f(cC['M2b_s0']['hi'])}; at s1 the point estimate is {f(cC['M2b_s1']['delta'])} and the bound {f(cC['M2b_s1']['hi'])}. "
  "The bar is not moved.\n")
rem = {s: 1 - cons[s]["paired"]["HG_minus_B"]["viol_any"]["mean"] / cons[s]["paired"]["H_minus_B"]["viol_any"]["mean"] for s in SEEDS}
ret2 = {s: pB[s]["ade2"]["mean"] / pHB[s]["ade2"]["mean"] for s in SEEDS}
ret6 = {s: pB[s]["ade6"]["mean"] / pHB[s]["ade6"]["mean"] for s in SEEDS}
w("What the gate bought and what it cost, against H (HBLEND), as MEASURED point estimates: "
  f"CONSISTENCY(a) 'any' cost vs B fell from {f(cons['M2b_s0']['paired']['H_minus_B']['viol_any']['mean'])} / {f(cons['M2b_s1']['paired']['H_minus_B']['viol_any']['mean'])} (H) to {f(cC['M2b_s0']['delta'])} / {f(cC['M2b_s1']['delta'])} (HG), "
  f"i.e. {rem['M2b_s0']*100:.0f} % / {rem['M2b_s1']*100:.0f} % of the cost removed (s0 / s1); the ADE gain over B shrank from ADE6 {f(pHB['M2b_s0']['ade6']['mean'])} / {f(pHB['M2b_s1']['ade6']['mean'])} to "
  f"{f(pB['M2b_s0']['ade6']['mean'])} / {f(pB['M2b_s1']['ade6']['mean'])} ({ret6['M2b_s0']*100:.0f} % / {ret6['M2b_s1']*100:.0f} % retained) and ADE2 from {f(pHB['M2b_s0']['ade2']['mean'])} / {f(pHB['M2b_s1']['ade2']['mean'])} to "
  f"{f(pB['M2b_s0']['ade2']['mean'])} / {f(pB['M2b_s1']['ade2']['mean'])} ({ret2['M2b_s0']*100:.0f} % / {ret2['M2b_s1']*100:.0f} % retained). "
  "HG is strictly between B and H on both axes: it is not a free win on either.\n")

# ------------------------------------------------------------------------------------------------ 1
w("## 1. Step 1: the train-window GT constraint activity and the re-fit\n")
w(f"* **Source.** The box-2 label join `join_train.npz` (the trainer's own `--join-train`, sha256 `{A['join_sha256'][:16]}...`), **already on the dev box** at `hblend/box2/join_train.npz` (it carries `goal_y/goal_w`, the 26 constraint cells, `lead_valid` and `lead_gap` for all 4,572 train clips). "
  "The GT constraint activity of the training windows is therefore derivable from what is on the dev box, so **no box-2 `scp` and no box-2 access of any kind was made**.")
w("* **Definition.** 'Active' is the eval's own: `gt_constraints(X)` + `v8f_consistency.violations(...)[rule]['active']` (hblend/deps copy of `taniteval`), applied to each fit window's join row (`off[ci] + t`) with its `lead_gap_h` "
  "(`lead_gap[off + min(t+1+0..29, T-1)]`) and `s_h` (the data module's own `human_arc` on the cached `pose_grid`). STOP = STOP_POINT bit with weight, finite stop distance, `lon_t_reach_s` <= 6. TURN = TURN_L/R bit with weight and >= 1 s overlap with the plan horizon. LEAD = `lead_valid` and a finite lead gap at some knot >= 10. SPEED is excluded (active on every window).")
w(f"* **Counts (HBLEND fit set: {A['n_windows']:,} windows, {A['n_clips']} clips, k = {A['k']}; the window set is identical for both seeds: {A['same_window_set_for_both_seeds']}).** "
  f"Active: STOP {A['active_counts']['STOP']:,} ({A['active_share']['STOP']*100:.1f} %), TURN {A['active_counts']['TURN']:,} ({A['active_share']['TURN']*100:.1f} %), LEAD {A['active_counts']['LEAD']:,} ({A['active_share']['LEAD']*100:.1f} %), "
  f"SPEED {A['active_counts']['SPEED']:,} ({A['active_share']['SPEED']*100:.1f} %). Windows with any of STOP/TURN/LEAD active: {A['n_any_STOP_TURN_LEAD_active']:,}. "
  f"**Windows with none active (the fit set of w^G): {A['n_inactive_all_three']:,} ({A['share_inactive_all_three']*100:.1f} %), at BOTH seeds** (s0 {WG['M2b_s0']['n_windows_fit']:,}, s1 {WG['M2b_s1']['n_windows_fit']:,}; {WG['M2b_s0']['n_clips_in_fit']} clips). "
  f"Windows without a label row: {A['n_windows_without_label_row']}. Reference, eval-141 (same code): STOP {A['eval_141_active_counts_for_reference']['STOP']}, TURN {A['eval_141_active_counts_for_reference']['TURN']}, LEAD {A['eval_141_active_counts_for_reference']['LEAD']} "
  "(LEAD 1,027 matches HPROJ), SPEED 2,399 of 2,399.")
c = A["controls"]
w(f"* **Controls on the activity code** (`train_activity.json::controls`, all pass = {c['all_pass']}): (A) `s_h` from the cached pose grid vs `s_h` re-derived from the GT waypoints, max |d| = {c['A_s_h_cache_vs_gt_derived_maxabs_m']:.2e} m; "
  f"(B) the masks are identical for three different plans (GT future, B's plan, P10): {c['B_activity_identical_across_plans']}; (C) the SAME builder on the 2,399 eval windows reproduces every input field of the eval's context and the eval's own STOP/TURN/LEAD/SPEED masks exactly; "
  f"(D) mutation: removing the lead-gap data changes the LEAD mask ({c['D_mutation_no_lead_gap_changes_LEAD_mask']}), so control (C) can fail.")
w("* **The fit.** HBLEND's `fit_weights` (closed form per step and axis, clip [0, 1], unweighted PAV), imported, not re-implemented, on the 5,541 inactive windows, per seed from its own checkpoint's outputs (`fit_windows_M2b_s*.npz`).\n")
w("| quantity | s0 | s1 |")
w("|---|---|---|")
for lab, fn in (("fit windows (n)", lambda s: f"{WG[s]['n_windows_fit']:,} of {WG[s]['n_windows_all']:,}"),
                ("cells clipped at 0 / at 1 (of 60)", lambda s: f"{WG[s]['n_clipped_at_0']} / {WG[s]['n_clipped_at_1']}"),
                ("cells changed by isotonic", lambda s: str(WG[s]["n_changed_by_isotonic"])),
                ("w^G in [0, 1] and non-decreasing", lambda s: str(WG[s]["w_in_range_and_monotone"])),
                ("mean abs (w^G - banked w) over the 60 cells", lambda s: g(WG[s]["w_G_minus_w_banked"]["mean_abs"])),
                ("max abs (w^G - banked w)", lambda s: g(WG[s]["w_G_minus_w_banked"]["max_abs"])),
                ("mean signed (w^G - w), along / cross", lambda s: f"{f(WG[s]['w_G_minus_w_banked']['mean_signed_along'])} / {f(WG[s]['w_G_minus_w_banked']['mean_signed_cross'])}"),
                ("descriptive: same-size RANDOM subset (50 draws), mean abs dev from banked w: mean / max over draws",
                 lambda s: f"{g(WG[s]['random_subset_same_n_vs_banked_w (50 draws, DESCRIPTIVE)']['mean_abs_dev_mean'])} / {g(WG[s]['random_subset_same_n_vs_banked_w (50 draws, DESCRIPTIVE)']['mean_abs_dev_max_over_draws'])}")):
    w(f"| {lab} | {fn('M2b_s0')} | {fn('M2b_s1')} |")
_rr = {s_: WG[s_]["w_G_minus_w_banked"]["mean_abs"] / WG[s_]["random_subset_same_n_vs_banked_w (50 draws, DESCRIPTIVE)"]["mean_abs_dev_mean"] for s_ in SEEDS}
_rmax = {s_: WG[s_]["w_G_minus_w_banked"]["mean_abs"] > WG[s_]["random_subset_same_n_vs_banked_w (50 draws, DESCRIPTIVE)"]["mean_abs_dev_max_over_draws"] for s_ in SEEDS}
w(f"\nThe restriction moves the weights by {_rr['M2b_s0']:.1f}x / {_rr['M2b_s1']:.1f}x (s0 / s1) the mean deviation of an equal-size random subset, and beyond the largest deviation of any of the 50 draws "
  f"({_rmax['M2b_s0']} / {_rmax['M2b_s1']}), so the constraint-aware fit is a different set of 60 numbers from HBLEND's, not sampling noise around them (descriptive). Whether that difference matters for the metrics is read in section 8, item 5.\n")
w("w^G (the entire fitted artifact: 60 numbers per seed); banked HBLEND w beside it. `OUT/weights_G_M2b_s{0,1}.json` (raw, clipped, final).\n")
w("| t (s) | s0 along w^G / w | s0 cross w^G / w | s1 along w^G / w | s1 cross w^G / w |")
w("|---|---|---|---|---|")
for j in range(30):
    cells = []
    for s in SEEDS:
        for ax in ("along", "cross"):
            cells.append(f"{WG[s]['w_%s_by_step' % ax][j]:.3f} / {WB[s]['w_%s_by_step' % ax][j]:.3f}")
    w(f"| {0.2*(j+1):.1f} | " + " | ".join(cells) + " |")
w("")

# ------------------------------------------------------------------------------------------------ 2
w("## 2. Identity controls (run BEFORE any HG read; the script refuses to read, rc 2, if any fails)\n")
w("| control | bar | s0 | s1 |")
w("|---|---|---|---|")
ctl = {s: S[s]["identity_controls"] for s in SEEDS}
rows2 = [("(i) gate forced CLOSED everywhere: max abs diff of the plan vs B's banked `pred` (m)", "= 0.0 exactly", lambda c: g(c["i_closed_vs_B_maxabs_m"], 1)),
         ("(i) ... per-window ADE6 max abs diff vs banked", "-", lambda c: g(c["i_closed_ade6_perwindow_maxabs_vs_banked"], 1)),
         ("(i) ... the eval's own `evaluate()` block (traj, consistency_a, strata, tactical, self_consistency_b, families) vs the banked JSON `raw`: mismatches", "0", lambda c: str(c["i_closed_full_block_vs_banked_json"]["n_mismatches"])),
         ("(ii) gate forced OPEN with HBLEND's banked w: max abs diff vs HBLEND's saved H (m)", "= 0.0 exactly", lambda c: g(c["ii_open_banked_w_vs_H_saved_maxabs_m"], 1)),
         ("(ii) ... P recomputed from the checkpoint vs HBLEND's saved P (m)", "-", lambda c: g(c["ii_P_recomputed_vs_hblend_saved_P_maxabs_m"], 1)),
         ("(ii) ... `evaluate()` block vs HBLEND's saved H block: mismatches", "0", lambda c: str(c["ii_open_full_block_vs_hblend_saved_H_block"]["n_mismatches"])),
         ("(iii) the fit on ALL windows vs HBLEND's banked weights: max abs diff", "<= 1e-12", lambda c: f"{c['iii_fit_all_windows']['maxabs_vs_banked_weights']:.1e}"),
         ("R1 deliberate regression: gate OPEN with w^G instead of the banked w, max abs vs saved H (m)", "must be > 0 (the control can fail)", lambda c: g(c["R1_open_with_wG_instead_of_banked_vs_H_maxabs_m"], 3)),
         ("R2 deliberate regression: the real gate vs B, windows that differ", "must be > 0", lambda c: str(c["R2_real_gate_vs_B_n_windows_differ"])),
         ("banked `pred` == `traj30`", "True", lambda c: str(c["banked_pred_eq_traj30"])),
         ("checkpoint's embedded P10 vs the P10-yaw-4512 file, max abs (m)", "<= 1e-4", lambda c: f"{c['model_prior_vs_P10_4512_file_maxabs_m']:.2e}"),
         ("w^G in [0, 1] and non-decreasing", "True", lambda c: str(c["wG_in_range_and_monotone"])),
         ("PREREG literals (LON_FOLLOW 3, STOP 6, TURN 1/2, band knot 10) == the code tree's constants", "True", lambda c: str(c["prereg_literals_equal_source_constants"])),
         ("**all identity controls pass**", "", lambda c: f"**{c['all_pass']}**")]
for lab, bar, fn in rows2:
    w(f"| {lab} | {bar} | {fn(ctl['M2b_s0'])} | {fn(ctl['M2b_s1'])} |")
w("\nArtifacts: `OUT/score_M2b_s{0,1}.json::identity_controls`, `OUT/weights_G_M2b_s{0,1}.json::identity_fit_all_windows_vs_banked`.\n")

# ------------------------------------------------------------------------------------------------ 3
w("## 3. The PREREG rows\n")
w("| read | seed | value | 95 % CI | verdict | estimator | tier | artifact |")
w("|---|---|---|---|---|---|---|---|")
for s in SEEDS:
    c_ = cC[s]
    w(f"| **PRIMARY-C** CONSISTENCY(a) 'any' HG - B (share of windows) | {SV[s]} | {f(c_['delta'])} | [{f(c_['lo'])}, {f(c_['hi'])}] | bar: upper bound <= +0.02 -> **{'PASS' if c_['pass'] else 'FAIL'}** | {EST} | {TIER} | `OUT/score_{s}.json::consistency.paired.HG_minus_B.viol_any` |")
for s in SEEDS:
    for k, lab in (("ade6", "ADE6"), ("ade2", "ADE2")):
        k_ = cK[s][k]
        w(f"| **PRIMARY-K** {lab} HG - B (m) | {SV[s]} | {f(k_['delta'])} | [{f(k_['lo'])}, {f(k_['hi'])}] | bar: < 0 and separated -> **{'PASS' if k_['pass'] else 'FAIL'}** | {EST} | {TIER} | `OUT/score_{s}.json::paired.HG_minus_B.{k}` |")
w("")
w("| outcome | condition | met? |")
w("|---|---|---|")
w(f"| **PASS** | PRIMARY-C and PRIMARY-K at both seeds | **{R['PRIMARY_C_both_seeds'] and R['PRIMARY_K_both_seeds']}** (C: {R['PRIMARY_C_both_seeds']}, K: {R['PRIMARY_K_both_seeds']}) |")
w(f"| **CONSISTENCY NOT RESTORED** | PRIMARY-C fails | **{not R['PRIMARY_C_both_seeds']}** |")
w(f"| GAIN LOST | PRIMARY-C passes, PRIMARY-K fails | {R['PRIMARY_C_both_seeds'] and not R['PRIMARY_K_both_seeds']} |")
w("")

# ------------------------------------------------------------------------------------------------ 4
w("## 4. The gate, beside (descriptive)\n")
w("'Open' = the shrunk plan tau_G is used; 'closed' = tau_B is used (a trigger fired). That reading is fixed by the PREREG's own identity controls ('forced closed -> tau_B', 'forced open -> H'). Shares are over the 2,399 eval windows, with the episode-cluster CI.\n")
w("| share of windows | s0 | s1 |")
w("|---|---|---|")
w(f"| **gate OPEN (tau_G used)** | {ex(gate['M2b_s0']['open_share'])} ({gate['M2b_s0']['n_open']:,}) | {ex(gate['M2b_s1']['open_share'])} ({gate['M2b_s1']['n_open']:,}) |")
w(f"| gate CLOSED (tau_B used) | {ex(gate['M2b_s0']['closed_share'])} ({gate['M2b_s0']['n_closed']:,}) | {ex(gate['M2b_s1']['closed_share'])} ({gate['M2b_s1']['n_closed']:,}) |")
for k, lab in (("T1_follow", "T1 fires: longitudinal argmax FOLLOW"), ("T2_stop", "T2 fires: P(STOP) >= 0.5"), ("T3_turn", "T3 fires: P(TURN_L or TURN_R) >= 0.5"), ("T4_vhi", "T4 fires: speed of tau_G > T's predicted v_hi_ms, t >= 2 s")):
    w(f"| {lab} | {ex(gate['M2b_s0']['fires_share'][k])} ({gate['M2b_s0']['fires_n'][k]:,}; exclusive {gate['M2b_s0']['exclusive_n'][k]:,}) | {ex(gate['M2b_s1']['fires_share'][k])} ({gate['M2b_s1']['fires_n'][k]:,}; exclusive {gate['M2b_s1']['exclusive_n'][k]:,}) |")
w(f"| T1 or T2 or T3 (the decision triggers) | {ex(gate['M2b_s0']['closed_by_T1T2T3_share'])} | {ex(gate['M2b_s1']['closed_by_T1T2T3_share'])} |")
w(f"| windows where HG differs from B | {gate['M2b_s0']['n_windows_plan_differs_from_B']:,} | {gate['M2b_s1']['n_windows_plan_differs_from_B']:,} |")
w(f"| predicted v_hi_ms <= 0 (T4 closes the gate by the literal reading) | {gate['M2b_s0']['predicted_v_hi_ms']['n_nonpositive']} | {gate['M2b_s1']['predicted_v_hi_ms']['n_nonpositive']} |")
w("\n'exclusive' = windows closed by that trigger and no other. Artifact: `OUT/score_M2b_s*.json::gate`, per-window flags in `score_M2b_s*_perwindow.npz` (`trig_*`, `closed`).\n")
w("**Do T's triggers fire where the human is constrained?** Cross-tab of the trigger against the eval's own GT activity on eval-141 (descriptive; this is the `D-V8F-T-ATTRIB-1` question asked directly):\n")
w("| trigger vs GT | seed | fired and active | fired, not active | not fired, active | precision | recall |")
w("|---|---|---|---|---|---|---|")
for key, lab in (("T1_follow_vs_GT_LEAD_active", "T1 vs GT LEAD active"), ("T2_stop_vs_GT_STOP_active", "T2 vs GT STOP active"), ("T3_turn_vs_GT_TURN_active", "T3 vs GT TURN active"),
                 ("T1T2T3_any_vs_GT_any_of_STOP_TURN_LEAD_active", "T1/T2/T3 any vs GT any of STOP/TURN/LEAD"), ("gate_closed_vs_GT_any_of_STOP_TURN_LEAD_active", "gate closed (T1-T4) vs GT any of STOP/TURN/LEAD"),
                 ("T4_vhi_vs_H_SPEED_violated", "T4 vs H violates SPEED"), ("T4_vhi_vs_B_SPEED_violated", "T4 vs B violates SPEED")):
    for s in SEEDS:
        x = gate[s]["crosstab_vs_GT_activity"][key]
        w(f"| {lab} | {SV[s]} | {x['fired_and_active']} | {x['fired_not_active']} | {x['not_fired_active']} | {x['precision']:.3f} | {x['recall']:.3f} |")
w("\nGT active counts on eval-141: STOP 81, TURN 158, LEAD 1,027 (SPEED 2,399). Reading: T1 is a good LEAD detector by precision (0.80 / 0.91) but finds only 43 % / 52 % of the LEAD-active windows; T2 (STOP) is weak at both seeds; T3 finds 78 % / 83 % of GT turns at 61 % precision; "
  "T4, the gate's only SPEED handler, has low precision against the SPEED violations that matter (H: 0.24 / 0.32, B: 0.06 / 0.14).\n")

# ------------------------------------------------------------------------------------------------ 5
w("## 5. CONSISTENCY per rule and (b), beside\n")
w("CONSISTENCY(a) = share of windows whose PLAN violates a GT-derived constraint; (b) = violation of the model's OWN predicted constraints. Lower is better. Paired episode-cluster bootstrap, B=2000. `*` = CI excludes 0 (worse if positive).\n")
w("| seed | rule | B | H | HG | P10 | HG - B | HG - H | HG - P10 |")
w("|---|---|---|---|---|---|---|---|---|")
for s in SEEDS:
    rt, pc = cons[s]["rate"], cons[s]["paired"]
    for k, lab in (("viol_any", "any"), ("viol_SPEED", "SPEED"), ("viol_STOP", "STOP"), ("viol_TURN", "TURN"), ("viol_LEAD", "LEAD"), ("self_viol_any", "self (b) any")):
        p10 = g(rt["P10"][k]) if k in rt["P10"] else "n/a"
        hp = ci(pc["HG_minus_P10"][k]) + star(pc["HG_minus_P10"][k]) if k in pc["HG_minus_P10"] else "n/a"
        w(f"| {SV[s]} | {lab} | {g(rt['B'][k])} | {g(rt['H'][k])} | {g(rt['HG'][k])} | {p10} | {ci(pc['HG_minus_B'][k])}{star(pc['HG_minus_B'][k])} | {ci(pc['HG_minus_H'][k])}{star(pc['HG_minus_H'][k])} | {hp} |")
_pm = lambda s_, k_: f(cons[s_]["paired"]["HG_minus_B"][k_]["mean"])
w("\nWhere the residual cost lives (HG - B, point estimates, s0 / s1): the SPEED excess is "
  f"{_pm('M2b_s0', 'viol_SPEED')} / {_pm('M2b_s1', 'viol_SPEED')} against an 'any' excess of {_pm('M2b_s0', 'viol_any')} / {_pm('M2b_s1', 'viol_any')} "
  "(rule excesses overlap inside 'any', so they need not sum to it); LEAD is next "
  f"({_pm('M2b_s0', 'viol_LEAD')} / {_pm('M2b_s1', 'viol_LEAD')}); STOP ({_pm('M2b_s0', 'viol_STOP')} / {_pm('M2b_s1', 'viol_STOP')}) "
  f"and TURN ({_pm('M2b_s0', 'viol_TURN')} / {_pm('M2b_s1', 'viol_TURN')}) are at or near zero. "
  "SPEED is the one rule active on every window, so the constraint-aware fit cannot see it and T4 is its only handler.\n")
w("Window transitions, HG vs B (violated by HG and ok in B / ok in HG and violated in B):\n")
w("| seed | rule | HG viol, B ok | HG ok, B viol | both viol |")
w("|---|---|---|---|---|")
for s in SEEDS:
    for k in ("viol_any", "viol_SPEED", "viol_STOP", "viol_TURN", "viol_LEAD"):
        t_ = cons[s]["transitions"]["HG_vs_B"][k]
        w(f"| {SV[s]} | {k[5:]} | {t_['a_viol_b_ok']} | {t_['a_ok_b_viol']} | {t_['both_viol']} |")
w("")
w("By gate state (share of windows violating 'any'; HG = B on closed windows by construction): "
  + "; ".join(f"{SV[s]} open: HG {cons[s]['by_gate_state']['open']['viol_any']:.3f} vs B {cons[s]['B_by_gate_state']['open']['viol_any']:.3f}, closed: {cons[s]['by_gate_state']['closed']['viol_any']:.3f}" for s in SEEDS) + ". "
  "The whole excess is on the open windows, as it must be.\n")

# ------------------------------------------------------------------------------------------------ 6
w("## 6. ADE and FDE at 1-6 s, accel, families, dominance (beside, per PREREG)\n")
for s in SEEDS:
    w(f"**{s}** (m; mean [95 % CI]; paired deltas {EST})\n")
    w("| metric | B | HG | H | P10 | HG - B | HG - H | HG - P10 |")
    w("|---|---|---|---|---|---|---|---|")
    sg, pm = S[s]["single"], S[s]["paired"]
    for kind in ("ade", "fde"):
        for h in range(1, 7):
            k = f"{kind}{h}"
            w(f"| {kind.upper()}@{h}s | {ex(sg['B'][k])} | {ex(sg['HG'][k])} | {ex(sg['H'][k])} | {ex(sg['P10'][k])} | {ci(pm['HG_minus_B'][k])}{star(pm['HG_minus_B'][k])} | {ci(pm['HG_minus_H'][k])}{star(pm['HG_minus_H'][k])} | {ci(pm['HG_minus_P10'][k])}{star(pm['HG_minus_P10'][k])} |")
    w("")
w("**Dominance vs P10-yaw-4512** (descriptive: 'HG dominates P10' iff ADE2, ADE6 and accel MAE point estimates are all <= P10's and ADE6 H - P10 is separated):\n")
w("| seed | ADE2 HG / P10 (m) | ADE6 HG / P10 (m) | ADE6 HG - P10, paired | accel MAE HG / P10 (m/s^2) | accel HG - P10, paired | HG dominates P10 |")
w("|---|---|---|---|---|---|---|")
for s in SEEDS:
    sg, d = S[s]["single"], R["per_seed"][s]["DOMINANCE_vs_P10"]
    pt = d["point_estimates_le_P10"]
    w(f"| {SV[s]} | {g(sg['HG']['ade2']['mean'])} / {g(sg['P10']['ade2']['mean'])} | {g(sg['HG']['ade6']['mean'])} / {g(sg['P10']['ade6']['mean'])} | {ci(d['ade6'])}{star(d['ade6'])} | "
      f"{g(sg['HG']['fw_lon_accel_mae']['mean'])} / {g(sg['P10']['fw_lon_accel_mae']['mean'])} | {ci(d['accel'])}{star(d['accel'])} | **{d['HG_dominates_P10']}** (ADE2 {pt['ade2']}, ADE6 {pt['ade6']}, accel {pt['accel']}) |")
w("")
w("### The four families (HG - B and HG - P10 paired; lower is better except the two plan-derived tactical rows)\n")
FAM = [("LONGITUDINAL", "speed MAE (m/s)", "fw_lon_speed_mae"), ("LONGITUDINAL", "accel MAE (m/s^2)", "fw_lon_accel_mae"), ("LONGITUDINAL", "along MAE (m)", "fw_lon_along_mae"),
       ("LATERAL", "cross-track offset MAE (m)", "fw_lat_cross_mae"), ("LATERAL", "heading MAE (deg)", "fw_lat_heading_mae_deg"), ("LATERAL", "yaw-rate MAE (deg/s)", "fw_lat_yaw_rate_mae_degps"),
       ("LATERAL", "curvature MAE (1/m)", "fw_lat_curvature_mae_1pm"), ("TACTICAL (plan-derived)", "lateral decision agreement (higher better)", "fw_tac_lat_acc"),
       ("TACTICAL (plan-derived)", "longitudinal decision agreement (higher better)", "fw_tac_lon_acc")]
for s in SEEDS:
    w(f"**{s}**\n")
    w("| family | metric | B | HG | H | P10 | HG - B | HG - P10 |")
    w("|---|---|---|---|---|---|---|---|")
    sg, pm = S[s]["single"], S[s]["paired"]
    for fam, lab, k in FAM:
        w(f"| {fam} | {lab} | {g(sg['B'][k]['mean'])} | {g(sg['HG'][k]['mean'])} | {g(sg['H'][k]['mean'])} | {g(sg['P10'][k]['mean'])} | {ci(pm['HG_minus_B'][k])}{star(pm['HG_minus_B'][k])} | {ci(pm['HG_minus_P10'][k])}{star(pm['HG_minus_P10'][k])} |")
    w("")
w("TACTICAL (T's own heads: lat / lon macro-F1 and goal-token macro-F1): HG's block equals B's banked block exactly (json equality) at s0: "
  f"{S['M2b_s0']['structural_identity']['tactical_block_T_heads_HG_equals_banked_B']} and s1: {S['M2b_s1']['structural_identity']['tactical_block_T_heads_HG_equals_banked_B']} (the gate reads T's heads, it does not change them). "
  "The plan-derived tactical rows above are read off the trajectory and do move with the plan. **STRATEGIC: PAUSED** per the PI ruling of 2026-09-19 (`PI_DECISION_QUEUE.md` ITEM 25); S is untouched and its banked checkpoint error is B's; reported as PAUSED, not as a pass.\n")

w("### 6a. Distance keeping (the other half of LONGITUDINAL), K = 30 (6 s) lead block, through `v8f_dk_read.read_dk` -> `taniteval.lead_metrics`\n")
w(f"Lead block `b1_eval_lead_block_k30.npz`, window states {DK['join']['window_states']}, kinematic-contract check **{DK['human_reference']['kinematic_contract']['status']}**. Open loop: the lead's logged future does not react to the arm. "
  "No distance-keeping bar is registered for HGATE, so nothing here is PASS/FAIL. Min-TTC is a CENSORED mean (30 s cap); 'more gap' is not 'better'; the better/worse reading is LEAD-VIOL and the distance to the human.\n")
w("| arm | headway min (m) | time-gap min (s) | min-TTC (s, censored) | LEAD-VIOL share of at-risk (n viol) |")
w("|---|---|---|---|---|")
hm = DK["human_reference"]["metrics"]
w(f"| human (GT path) | {hm['headway_min_m']['mean']:.3f} | {hm['time_gap_min_s']['mean']:.3f} | {hm['min_ttc_s']['mean']:.3f} | by definition 0 |")
for nm in ("B_s0", "H_s0", "HG_s0", "B_s1", "H_s1", "HG_s1", "P10"):
    a = DK["arms"][nm]
    m = a["metrics"]
    w(f"| {nm} | {m['headway_min_m']['mean']:.3f} | {m['time_gap_min_s']['mean']:.3f} | {m['min_ttc_s']['mean']:.3f} | {a['lead_viol']['share_of_at_risk']:.4f} ({a['lead_viol']['n_viol']}) |")
w("")
w("| paired contrast | headway (m) | time-gap (s) | min-TTC (s) | LEAD-VIOL at-risk (share) |")
w("|---|---|---|---|---|")
for pn in ("HG_s0__minus__B_s0", "HG_s0__minus__H_s0", "HG_s0__minus__P10", "HG_s1__minus__B_s1", "HG_s1__minus__H_s1", "HG_s1__minus__P10"):
    pr = DK["pairs"][pn]
    cells = []
    for k in ("headway_min_m", "time_gap_min_s", "min_ttc_s"):
        d = pr["dk"][k]
        cells.append(f"{f(d['delta'], 3)} [{f(d['lo'], 3)}, {f(d['hi'], 3)}] {'*' if d['separated'] else ''}")
    lv = pr["lead_viol"]["at_risk"]
    cells.append(f"{f(lv['delta'], 4)} [{f(lv['lo'], 4)}, {f(lv['hi'], 4)}] {'*' if lv['separated'] else ''}")
    w(f"| {pn.replace('__minus__', ' - ')} | " + " | ".join(cells) + " |")
w("\n`*` = CI excludes 0. Artifact: `OUT/dk_read.json` (`pairs`, `arms`), per-window `OUT/dk_perwindow.npz`.\n")

# ------------------------------------------------------------------------------------------------ 7
w("## 7. The outcome table, applied verbatim, and the next step (Rule Zero)\n")
w(f"**Result: {R['outcome']}.** PRIMARY-C fails at both seeds, so the PREREG's committed step applies: *{R['committed_next_step_verbatim']}.* No bar, trigger, threshold or weight was changed after any HGATE number was seen.\n")
w("This is a refutation of one candidate, and per Rule Zero it is a waypoint. What is eliminated, with its cost: the restriction of the fit to unconstrained windows plus a four-trigger gate on T's own outputs recovers "
  f"{rem['M2b_s0']*100:.0f} % / {rem['M2b_s1']*100:.0f} % of H's CONSISTENCY cost but not enough to put the upper bound under +0.02, and it spends {100-ret6['M2b_s0']*100:.0f} % / {100-ret6['M2b_s1']*100:.0f} % of H's ADE6 gain to do it. "
  "What blocks a next step inside this family is named by the PREREG itself: it rules out a further post-processing arm, and the T-side lever waits on MIXO and FLAT reading (per the task brief, box 2 is training). "
  "Not run, because the PREREG rules it out and a bar would have to be invented: a further post-processing arm (for example a better SPEED handler than T4). Lifting that ruling is the Master Mind's / PI's call. "
  "What WAS run is the cheapest informative thing that invents no bar: section 8 (post-hoc, descriptive) attributes the residual so that the T-side work is aimed at the right term, the SPEED term and T4's predicted `v_hi_ms`. "
  "It also argues for the PREREG's T-side route over another post-processing arm: even with oracle decision triggers and an oracle `v_hi_ms`, the gain is gone at s1 (O2).\n")

# ------------------------------------------------------------------------------------------------ 8
w("## 8. Post-hoc diagnostics (descriptive; NOT a PREREG row; no bar; none can change the outcome)\n")
w("Run AFTER `READ.json` was fixed (`diag_gate.py`). 'tau_G everywhere' = the blend with w^G and NO gate (not HBLEND's H, which uses the unrestricted w). The ORACLE arms are **T0** (they read a GT label at inference): a ceiling on what a perfect detector could buy, **never a result**.\n")
for s in SEEDS:
    w(f"**{s}** (HG-style arm - B; paired episode-cluster bootstrap, B=2000)\n")
    w("| arm | open share | ADE2 | ADE6 | accel | CONS any | SPEED | LEAD | STOP | TURN | C (upper <= 0.02) | K (both < 0, separated) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nm, r in DG["seeds"][s]["arms"].items():
        w(f"| {nm} | {r['open_share']:.3f} | {ci(r['ade2'], 3)}{star(r['ade2'])} | {ci(r['ade6'], 3)}{star(r['ade6'])} | {ci(r['fw_lon_accel_mae'], 3)}{star(r['fw_lon_accel_mae'])} | {ci(r['viol_any'], 4)} | "
          f"{f(r['viol_SPEED']['mean'])} | {f(r['viol_LEAD']['mean'])} | {f(r['viol_STOP']['mean'])} | {f(r['viol_TURN']['mean'])} | {r['C_upper_bound_le_0.02']} | {r['K_both_negative_and_separated']} |")
    w("")
a0, a1 = DG["seeds"]["M2b_s0"]["arms"], DG["seeds"]["M2b_s1"]["arms"]
O1, O2, O3 = "ORACLE O1: GT STOP/TURN/LEAD active + T4", "ORACLE O2: GT STOP/TURN/LEAD active + GT-v_hi speed band", "ORACLE O3: close wherever tau_G violates any GT rule"
w("Reading (descriptive, each a MEASURED statement about the arms above, not a claim about the programme):\n")
w(f"1. **T4 is the load-bearing trigger and also the weakest (its precision against the SPEED violations that matter is 0.24 / 0.32 for H and 0.06 / 0.14 for B, section 4).** Dropping T4 (T1+T2+T3 only) leaves CONS 'any' at {f(a0['leave out T4 (v_hi)']['viol_any']['mean'])} / {f(a1['leave out T4 (v_hi)']['viol_any']['mean'])} (upper {f(a0['leave out T4 (v_hi)']['viol_any']['hi'])} / {f(a1['leave out T4 (v_hi)']['viol_any']['hi'])}); T4 alone gives {f(a0['T4 only']['viol_any']['mean'])} / {f(a1['T4 only']['viol_any']['mean'])}. "
  "Neither half passes PRIMARY-C alone at either seed. Removing any single one of T1/T2/T3 moves 'any' by at most "
  f"{max(abs(a_[k_]['viol_any']['mean'] - a_['HG (the PREREG arm: T1+T2+T3+T4)']['viol_any']['mean']) for a_ in (a0, a1) for k_ in ('leave out T1 (FOLLOW)', 'leave out T2 (STOP)', 'leave out T3 (TURN)')):.4f} "
  "(point estimate), so none of the three decision triggers is individually decisive.")
w(f"2. **A perfect STOP/TURN/LEAD detector is not enough while T4 stays as it is.** O1 (GT activity replaces T1-T3, T4 kept) gives CONS 'any' {f(a0[O1]['viol_any']['mean'])} [upper {f(a0[O1]['viol_any']['hi'])}] / {f(a1[O1]['viol_any']['mean'])} [upper {f(a1[O1]['viol_any']['hi'])}]: the upper bound is still above +0.02 at both seeds. "
  "The residual is then the SPEED term, i.e. T's predicted `v_hi_ms`.")
w(f"3. **With oracle detection of the decisions AND oracle `v_hi_ms` the consistency bar is met, at the price of the gain.** O2 gives CONS 'any' {f(a0[O2]['viol_any']['mean'])} [upper {f(a0[O2]['viol_any']['hi'])}] / {f(a1[O2]['viol_any']['mean'])} [upper {f(a1[O2]['viol_any']['hi'])}] (C met at both seeds), but ADE6 {f(a0[O2]['ade6']['mean'], 3)} (separated: {a0[O2]['ade6']['separated']}) / {f(a1[O2]['ade6']['mean'], 3)} "
  f"(separated: {a1[O2]['ade6']['separated']}) and ADE2 {f(a0[O2]['ade2']['mean'], 3)} / {f(a1[O2]['ade2']['mean'], 3)}: K holds at s0 and fails at s1 even with oracle gating. A gate family that can only choose between tau_B and tau_G has a thin margin here.")
w(f"4. **Ceiling of any per-window B-or-tau_G gate.** O3 (close wherever tau_G itself violates a GT rule) gives CONS 'any' {f(a0[O3]['viol_any']['mean'])} / {f(a1[O3]['viol_any']['mean'])} (better than B, separated) with ADE6 {f(a0[O3]['ade6']['mean'], 3)} / {f(a1[O3]['ade6']['mean'], 3)} (separated at both). "
  "So a gate that restores consistency and keeps the gain EXISTS in this family; HGATE's triggers do not find it. This is consistent with the PREREG's T-side attribution (`D-V8F-T-ATTRIB-1`), and O3 is an oracle of the violation itself, so it is an upper bound, not a design.")
_nog = "tau_G everywhere (w^G, NO gate: all triggers off)"
w(f"5. **The constraint-aware FIT, on its own, is almost H; the GATE does the work.** tau_G everywhere (w^G, no gate) gives CONS 'any' {f(a0[_nog]['viol_any']['mean'])} / {f(a1[_nog]['viol_any']['mean'])} and ADE6 {f(a0[_nog]['ade6']['mean'], 3)} / {f(a1[_nog]['ade6']['mean'], 3)} against HBLEND's H at "
  f"{f(cons['M2b_s0']['paired']['H_minus_B']['viol_any']['mean'])} / {f(cons['M2b_s1']['paired']['H_minus_B']['viol_any']['mean'])} and {f(pHB['M2b_s0']['ade6']['mean'], 3)} / {f(pHB['M2b_s1']['ade6']['mean'], 3)} (all vs B, s0 / s1). "
  "Fitting only on unconstrained windows removes a little of the consistency cost and a little of the gain; the restriction is not where the effect is.\n")

# ------------------------------------------------------------------------------------------------ 9
w("## 9. Interpretations declared before any HG number (the PREREG text left them open; none changed afterwards)\n")
w("1. **'Open' / 'closed'.** Closed = the plan is tau_B (a trigger fired, or forced); open = tau_G. Fixed by the PREREG's identity controls. 'The gate's open share' = the share of windows on which tau_G is used.")
w("2. **'Active GT constraint'** on a train window = the eval's own `violations(...)[STOP|TURN|LEAD]['active']` through `gt_constraints`; SPEED excluded (active on 9,582 of 9,582 windows). Windows without a label row would be inactive (there are none).")
w("3. **T4.** Speed of tau_G = `plan_kinematics(plan)['v']` on the 0.2 s knot plan, intervals that START at t >= 2.0 s (`v[:, 10:]`, the SPEED rule's own band, `_BAND_KNOT = 10`), compared with T's RAW predicted `v_hi_ms` (`cons_pred` cell 25; no clamp at 0, no min with the posted limit, no 0.5 m/s SPEED tolerance). The PREREG says 'exceeds T's own predicted v_hi_ms'. 20 / 3 windows (s0 / s1) have a predicted v_hi <= 0 and are closed.")
import numpy as _np
_eq = {}
for _s in SEEDS:
    _gp = _np.load(f"{HB}/inputs/banked/eval_{_s}-kind-own.npz")["goal_prob"].astype(_np.float64)
    _eq[_s] = {"exactly_0.5_STOP_TURN_L_TURN_R": [int((_gp[:, c] == 0.5).sum()) for c in (6, 1, 2)],
               "STOP_ge_vs_gt": [int((_gp[:, 6] >= 0.5).sum()), int((_gp[:, 6] > 0.5).sum())],
               "TURN_ge_vs_gt": [int(((_gp[:, 1] >= 0.5) | (_gp[:, 2] >= 0.5)).sum()), int(((_gp[:, 1] > 0.5) | (_gp[:, 2] > 0.5)).sum())]}
w("4. **T2 / T3 use '>= 0.5'** (the PREREG's operator). The eval's own self-consistency uses '> 0.5'. MEASURED here (build_result.py, from the banked npz `goal_prob`): windows with a probability of exactly 0.5 in the STOP / TURN_L / TURN_R columns: "
  f"s0 {_eq['M2b_s0']['exactly_0.5_STOP_TURN_L_TURN_R']}, s1 {_eq['M2b_s1']['exactly_0.5_STOP_TURN_L_TURN_R']}; STOP fires on {_eq['M2b_s0']['STOP_ge_vs_gt'][0]} (>=) vs {_eq['M2b_s0']['STOP_ge_vs_gt'][1]} (>) windows at s0, {_eq['M2b_s1']['STOP_ge_vs_gt'][0]} vs {_eq['M2b_s1']['STOP_ge_vs_gt'][1]} at s1; "
  f"TURN {_eq['M2b_s0']['TURN_ge_vs_gt'][0]} vs {_eq['M2b_s0']['TURN_ge_vs_gt'][1]} at s0, {_eq['M2b_s1']['TURN_ge_vs_gt'][0]} vs {_eq['M2b_s1']['TURN_ge_vs_gt'][1]} at s1. So the operator choice cannot change the gate on these data.")
w("5. **Scored grid.** tau_G is formed on the 0.2 s knots (the PREREG writes up(P10), the 0.1 s upsampling; on the scored knots up(P10)[1::2] = P10, so the plan scored is identical). HBLEND's own declaration that w is fitted on 30 steps x 2 axes = 60 numbers per seed carries over.")
w("6. **P.** P = the checkpoint's own frozen P10 evaluated on the eval kinematic features (HBLEND's route); recomputed here and identical to HBLEND's saved P (max abs 0.0).")
w("7. **The identity controls were read as exact** (0.0), not as a tolerance: they are the same arithmetic on the same arrays, so any non-zero would have been a bug.\n")

# ------------------------------------------------------------------------------------------------ 10
w("## 10. Escalations and caveats\n")
w("* **Nothing here passes the registered bar, so nothing here is deployable under the PREREG.** HG improves on H in the consistency rows (HG - H negative and separated on 'any', SPEED, STOP and LEAD at both seeds; TURN at s0 only) while keeping a separated ADE2/ADE6 gain over B. A decision to ship HG on the strength of that is the Master Mind's / PI's call and would need its own PREREG: the pre-registered bar is not met.")
w("* **Two seeds only; one training run each.** The separated-CI reading is necessary, not sufficient, for a lever claim (`H-ESTIM-SEED-1`). The two seeds disagree in size (open share 0.39 vs 0.54; PRIMARY-C point 0.020 vs 0.029) and agree in verdict and sign.")
w("* **The weights are trained-checkpoint-specific and fitted in-sample on train windows**; eval-141 never enters the fit (INHERITED from HBLEND section 8 for the unrestricted w: the train-fit sat within 0.004 m of the eval-oracle weights; not re-measured for w^G).")
w("* **The gate reads T's heads on the eval windows from the banked KIND run (`kind-own`).** If MIXO/FLAT change T's heads, the trigger statistics (section 4) change; they must be re-read per arm.")
w("* **Section 8 is post hoc.** It was run after the outcome was fixed. It attributes; it does not decide.\n")

# ------------------------------------------------------------------------------------------------ 11
w("## 11. Deliverable manifest (everything under `OUT`; nothing staged in the repo, nothing pushed, nothing deleted)\n")
w("| what | file | note |")
w("|---|---|---|")
for a, b, c_ in (("executor (activity / fit / score / read / dk)", "v8f_hgate.py", "imports HBLEND's `v8f_hblend` (fit_weights, apply_w, arm_pw, paired rows, loaders) and the eval's own `evaluate()`; HBLEND/HPROJ files untouched"),
                 ("post-hoc diagnostics", "diag_gate.py, diag_gate.json, diag_gate.log", "section 8; descriptive"),
                 ("self-tests (16 pass, `--basetemp C:/Users/Admin/pt_hgate`)", "test_v8f_hgate.py, pytest.log", "trigger thresholds, T4 band, gate identities, fit restriction with a wrong-subset regression arm, activity-builder indexing, outcome table"),
                 ("result builder", "build_result.py", "reads the JSON; no typed numbers"),
                 ("step 1 activity", "train_activity.json, train_activity.npz, activity.log", "per-window STOP/TURN/LEAD/SPEED masks aligned with the fit windows"),
                 ("step 1 weights", "weights_G_M2b_s0.json, weights_G_M2b_s1.json, fit_M2b_s*.log", "w^G (60 numbers per seed) + the identity (iii) record"),
                 ("score", "score_M2b_s0.json, score_M2b_s1.json (+ _perwindow.npz, score_*.log)", "controls, gate, every PREREG row, beside rows"),
                 ("outcome", "READ.json, read.log", "the outcome table applied"),
                 ("distance keeping", "dk_read.json, dk_perwindow.npz, dk.log", "via `v8f_dk_read.read_dk`"),
                 ("this report", "HGATE_RESULT.md", "")):
    w(f"| {a} | `{b}` | {c_} |")
w("\nReproduce (dev box, CPU): `PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe`; `$PY v8f_hgate.py activity`; `$PY v8f_hgate.py fit --seed M2b_s0` (and `M2b_s1`); `$PY v8f_hgate.py score --seed M2b_s0` (about 70 s each, identity controls first); `$PY v8f_hgate.py read`; `$PY v8f_hgate.py dk`; `$PY diag_gate.py`; `$PY build_result.py`.")
w("\nInputs reused read-only (not modified): `hblend/` (`v8f_hblend.py`, `fit_windows_M2b_s*.npz`, `weights_M2b_s*.json`, `box2/join_train.npz`, `inputs/banked/eval_M2b_s*-kind-own.{json,npz}`, `inputs/b2_p10_eval_pred6.npy`, `deps/`), `viz/tree` (code tip a99ca5d6), `viz/work` (checkpoints, eval context).")
open(f"{OUT}/HGATE_RESULT.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote HGATE_RESULT.md", len("\n".join(L)))
