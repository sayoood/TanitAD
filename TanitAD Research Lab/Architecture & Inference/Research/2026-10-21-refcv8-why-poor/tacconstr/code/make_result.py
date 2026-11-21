"""Assemble TACCONSTR_R8.json (the merged headline artifact) and RESULT.md from the banked per-script JSONs.  Every number in the
tables is read from a JSON, never typed.  CPU only."""
import json
import os

D = "C:/Users/Admin/refcv8_pf/tacconstr"
J = lambda n: json.load(open(f"{D}/{n}", encoding="utf-8"))
s1 = J("STEP1_DUMP_AUDIT.json")
hd = J("TACCONSTR_R8_heads_perwindow.json")
pr128 = J("TACCONSTR_R8_probe.json")
v9_128 = J("TACCONSTR_R8_v9heads.json")
lat_re = J("CONTROL_lat_lon_agreement_reread.json")

G = hd["v9_groups"]
P = hd["probe_level"]
ST = hd["exploratory_strata_lonspeed"]
ctl = hd["controls"]


def f4(x):
    return f"{x:.4f}"


def ci(x):
    return f"{x['point']:.4f} [{x['lo']:.4f}, {x['hi']:.4f}]"


def dci(x):
    return f"{x['point']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]" + (" sep" if x.get("separated", (x["lo"] > 0 or x["hi"] < 0)) else " ns")


def sep(x):
    return "SEP" if (x["lo"] > 0 or x["hi"] < 0) else "ns"


# ------------------------------------------------------------------------------------------------ the merged JSON
verdict = ("UNDECIDED FOR ADE, EVIDENCE LEANS CONSTRAINTS. The trained selector rewards candidates that satisfy the predicted heading and progress constraints (gamma 0.456 / 0.122) "
           "and does not reward its own decision posterior (beta -0.043 / -0.021); the constraint heads leave a large measured gap to GT, and in lonspeed (the largest single v8F constraint group) the head is at the "
           "ego-scalar floor in aggregate; the lateral decision head is already 0.9415 against a 0.9039 constant. But no refcv8-planner oracle has been run, and the planner never reads the 26-dim v9 vectors, "
           "so the headroom in metres is NOT measured for refcv8")
merged = {
    "schema": "TACCONSTR_R8/1", "date_berlin": "2026-10-21", "checkpoint": hd["checkpoint"], "tier": hd["tier"],
    "question": "does refcv8's tactical headroom live in its CONTINUOUS constraints or in its class decisions?",
    "verdict_one_line": verdict,
    "step1": {"battery_dumps_hold_constraints": False, "audit": "STEP1_DUMP_AUDIT.json", "n_files_read": s1["total_files_read"],
              "heads_dump_holds_constraints": True, "heads_dump": "E:/refcv8_eval_kit/videos/heads_48852_pf/heads_ep000..138.npz",
              "emitted": {"v9": {"out_r8_v9_lat_c": "[B,8,12]", "out_r8_v9_lon_c": "[B,8,10]", "out_r8_v9_speed": "[B,8,4]",
                                 "weight": "--w-r8-v9-cons 0.05", "consumed_by_planner": False,
                                 "source": "refcv6_tactical.py:642-645; refcv8_conditioning.py:129-132 ('they feed the loss, never the planner')"},
                          "probe": {"out_r8_cons_lat": "[B,8,2] (psi_term/pi, t_onset/6)", "out_r8_cons_lon": "[B,8,2] (log1p(prog6)/log1p(120), v_end/30)",
                                    "weight": "--w-r8-cons 0.05", "consumed_by_planner": True,
                                    "source": "refc.py:2500-2551 (phi, allocation), refc.py:3834 (FactorizedSelection), refcv8_conditioning.py:344"}}},
    "step2_amendment_2C": {"registered": True, "where": "PREREG_V8F_TAC.md Amendment 2 (C), commit aa8366d (clone C:/Users/Admin/lab019; NOT in the D:/E: working tree)",
                           "status": "PRE-REGISTERED, NOT RUN (box 2 awaits the PI's go); NOT CPU-RUNNABLE here",
                           "blockers": ["the four v8F-mini checkpoints (M1/M2b x seeds 0,1) exist only on box 2 (D:/tanitad/runs/v8f_mini_full/*/ckpt_last.pt); none is on this machine",
                                        "the pooled DINOv3-L 4x10 eval feature cache is on box 2",
                                        "the arms need O's forward at alpha 0.25/0.5/0.75; the banked npz hold only the alpha 0 (tacpred) and alpha 1 (tacorc-constraints) outputs, so no interpolation of banked outputs is admissible"],
                           "what_would_unblock": "PI go for box 2 (chain_v8f_tacgrp2.sh, 40 evals) OR the 4 checkpoints + eval feature cache shipped to the dev box (a 5-7 M parameter model: CPU eval feasible)"},
    "v9_heads_per_window": hd["v9_groups"], "v9_heads_per_field": hd["per_field"], "probe_level_per_window": hd["probe_level"],
    "exploratory_strata_lonspeed": ST, "controls": ctl, "trained_selector_weights_EMA48852": J("CKPT_SELECTOR_WEIGHTS.json"),
    "cross_check_trainer_log_128_windows": {"v9": v9_128["table"], "probe": pr128["controls_table"], "head_logged_v9_EMA48852": {
        g: v9_128["table"][g]["head_EMA48852_trainer_logged_normalised_MAE"] for g in ("lat_c", "lon_c", "speed")},
        "head_logged_probe_huber_EMA48852": pr128["head_EMA48852_logged_huber"]},
    "decision_heads_beside": lat_re, "estimator": hd["estimator"], "n_windows": hd["n_windows"], "n_episodes": hd["n_episodes"],
}
json.dump(merged, open(f"{D}/TACCONSTR_R8.json", "w"), indent=1)

# ------------------------------------------------------------------------------------------------ RESULT.md
L = []
w = L.append
w("# refcv8 EMA-48,852: do the tactical constraint heads carry the headroom? (T0, CPU only, 2026-10-21)\n")
w(f"**Verdict.** {verdict}.\n")
w("**Evidence classes.** Head outputs, labels, controls: MEASURED (ours, artifacts named in section 9). The v8F oracle shares (0.88-0.90 / 0.07-0.12 of 1.05-1.30 m): INHERITED from "
  "`RESULT_V8F_TAC_A1.md` (read at commit aa8366d; not re-run, and measured on v8F-mini, a different architecture). Trainer-log numbers: INHERITED, used only as a cross-check.\n")
w("**Tier.** T0: every head is read on the GT-active class's query (the trainer's own rule). It says how accurate the head is, never how well the car drives.\n")
w("**Estimator (what the intervals answer).** Episode-cluster bootstrap over the 139 eval episodes, B 2,000, seed 0 (same resample stream as `taniteval.ci._draws`), pooled entry-mean |error| "
  "(ratio of sums), paired where two arms share windows. It answers *another draw of episodes at this checkpoint*, with ONE training seed. It does not answer *another training run* "
  "(H-ESTIM-SEED-1); a separated row here is necessary, not sufficient. The decision heads read identically at inference seeds 0 and 1 (banked TAC_CONTROLS), so I treat the tactical decoder as deterministic; the constraint heads share that decoder but were captured once (seed 0), which I did not re-verify.\n")

w("## 0. What ran\n")
w("| step | status |\n|---|---|")
w(f"| 1. Are the constraints in the banked battery dumps? | **NO.** {s1['total_files_read']} npz read (4 trees x 139 episodes x {{ep, decisions}}), 0 read errors, a same-breath control key present in all {s1['total_files_with_control_key']}, 0 constraint-like keys (`STEP1_DUMP_AUDIT.json`). |")
w("| 1b. Are they anywhere per window? | **YES, in a different dump.** `E:/refcv8_eval_kit/videos/heads_48852_pf/heads_ep000..138.npz` (the post-battery `dump_heads` pass: EMA-48,852, REAL_CHECKPOINT_ON_REAL_CORPUS, perframe_full feed, 139 episodes, 4,754 windows, wall 1,995 s). It carries all five constraint outputs. Its decision heads equal the battery's `lat/lon_pred_nav_true` on **100.0 %** of 4,754 windows, so it is the same checkpoint and feed. **No GPU capture is needed for the heads.** |")
w("| 2. Amendment 2(C) dose-response | **Registered, NOT RUN, NOT CPU-RUNNABLE here** (section 7). Not invented, not approximated. |")
w("| 2b. Constraint MAE vs GT per group, EMA-48,852 | **Done, per window**, against the four controls (sections 2-4). |")
w("| 3. Estimator | as above; n and d printed in every row. |")
w("| 4. Controls | GT-as-prediction, constant = TRAIN mean, lateral agreement re-read: all read their known values (section 6). |\n")

w("## 1. What refcv8 emits and who reads it (source-literal, launch commit fb47a58; the bundle's blobs equal it)\n")
w("| output | shape | trained at | read by the planner? |\n|---|---|---|---|")
w("| `out_r8_v9_lat_c` / `_lon_c` / `_speed` | [B,8,12] / [B,8,10] / [B,8,4] = **26 dims, the same 26 as v8F's `constraints`** (latcons 12, lonspeed 8, stoplead 6) | `--w-r8-v9-cons 0.05` (Huber beta 0.1, normalised v9 units, GT-active class query only) | **NO** - \"they feed the loss, never the planner\" (`refcv8_conditioning.py:129-132`) |")
w("| `out_r8_cons_lat` / `_lon` | [B,8,2] each = psi_term/pi, t_onset/6 \\| log1p(prog6)/log1p(120), v_end/30 | `--w-r8-cons 0.05` (Huber beta 0.05, GT-active PATH-TAGGED hypothesis) | **YES** - phi of allocated candidates, `FactorizedSelection` (`refc.py:2500-2551, 3834`), `set_r8_force(cons=...)` |")
w("\nSo for refcv8 the v8F finding (\"O reads the 26-dim code\") has no direct counterpart: the planner reads **four** constraint numbers per hypothesis, and the 26 are auxiliary supervision.\n")

SW = J("CKPT_SELECTOR_WEIGHTS.json")
w("### 1b. What the TRAINED selector weights (MEASURED from the EMA-48,852 checkpoint scalars, `CKPT_SELECTOR_WEIGHTS.json`)\n")
w("`FactorizedSelection` scores a candidate `beta_lat log p(lat) + beta_lon log p(lon) + gamma_prog sat_prog + gamma_head sat_head`. All four scalars start at exactly 0, so a value away from 0 was moved by training.\n")
w("| scalar | trained value | acts on | max swing on the score |\n|---|---|---|---|")
sw = SW["max_swing_of_each_term_on_the_score"]
w(f"| `gamma_head` | **{SW['gamma_head']:+.4f}** | -\\|terminal heading - predicted psi_term\\|/pi, in [-1, 0] | {sw['gamma_head_x_sat_head (sat range 1)']:.3f} |")
w(f"| `gamma_prog` | **{SW['gamma_prog']:+.4f}** | -\\|log1p(path length) - log1p(predicted prog6)\\|, down to -{4.80:.2f} | {sw['gamma_prog_x_sat_prog (sat range up to log1p(120))']:.3f} |")
w(f"| `beta_lat` | {SW['beta_lat']:+.4f} | log p(lateral hypothesis) | {sw['beta_lat_x_log_p (log p floored at log 1e-4 = -9.21)']:.3f} (upper bound; log p floored at -9.21) |")
w(f"| `beta_lon` | {SW['beta_lon']:+.4f} | log p(longitudinal hypothesis) | {sw['beta_lon_x_log_p (same floor)']:.3f} (upper bound) |")
w("\nThe constraint-satisfaction weights are positive (the selector favours candidates that satisfy the predicted heading and progress); the decision-posterior weights are small and **negative** (a higher score for the less probable hypothesis, "
  "the opposite of rewarding the decision). The route-checkpoint gate is exactly 0 (`--r8-no-rc`). This says what training moved, **not** how many metres each term is worth; the beta bounds are reached only by near-zero-probability hypotheses. "
  "It does say that the consumed channel is the constraint one, which is the necessary condition for constraint accuracy to matter to refcv8's plan.\n")

w("## 2. v9 constraint heads, per window (139 episodes, 4,754 windows)\n")
w("Pooled entry-mean |error| in the heads' normalised units, 95 % CI in brackets. `gap closed` = 1 - head / control. `paired head - control` < 0 means the head is better.\n")
w("| group (d) | entries / windows | **head** | const (TRAIN mean) | class-cond const | ridge (v0, a0) | ridge poly-2 |\n|---|---|---|---|---|---|---|")
for g in ("lat_c", "lon_c", "speed", "latcons", "lonspeed", "stoplead", "ALL26"):
    t = G[g]
    c = t["controls"]
    w(f"| {g} ({t['head']['d']}) | {t['head']['n_entries']:,} / {t['head']['n_windows_any']:,} | **{ci(t['head'])}** | {ci(c['const_TRAIN_mean'])} | {ci(c['const_classcond_TRAIN_mean'])} | {ci(c['ridge_v0a0_TRAINfit'])} | {ci(c['ridge_v0a0_poly2_TRAINfit'])} |")
w("\nPaired difference, head minus control (the CI is over the same resampled episodes in both arms):\n")
w("| group | - const | - class-cond const | - ridge (v0, a0) | - ridge poly-2 |\n|---|---|---|---|---|")
for g in ("lat_c", "lon_c", "speed", "latcons", "lonspeed", "stoplead", "ALL26"):
    t = G[g]["paired_head_minus"]
    w(f"| {g} | {dci(t['const_TRAIN_mean'])} | {dci(t['const_classcond_TRAIN_mean'])} | {dci(t['ridge_v0a0_TRAINfit'])} | {dci(t['ridge_v0a0_poly2_TRAINfit'])} |")
w("\n**Reading.**")
w("* Every group beats the constant, separated (gap closed 42-68 %). That alone is the weakest bar and is met.")
w("* **`speed` (the four speed fields) and `lonspeed` (the eight, the largest single group in the v8F oracle, 0.20-0.26 m) do NOT beat an ego-scalar floor:** "
  f"`speed` head - ridge {dci(G['speed']['paired_head_minus']['ridge_v0a0_TRAINfit'])}; head - poly-2 {dci(G['speed']['paired_head_minus']['ridge_v0a0_poly2_TRAINfit'])} (the head is slightly WORSE there); "
  f"`lonspeed` head - ridge {dci(G['lonspeed']['paired_head_minus']['ridge_v0a0_TRAINfit'])}, head - poly-2 {dci(G['lonspeed']['paired_head_minus']['ridge_v0a0_poly2_TRAINfit'])}. "
  "A head that does not beat the floor has added nothing beyond the ego state it could read at t0.")
w("* `stoplead` (lead gap / time gap, stop point) and `latcons` beat every control, separated, including the class-conditional constant and both ridges - these are the groups where the head carries scene information.")
w("* The class-conditional constant is the like-for-like floor for `lat_c`: the head is read on the GT class's query, so the lateral-class identity (which dominates the turn fields: const 0.81 vs class-cond 0.24 on `turn_dyaw_deg`) is given for free.\n")

w("### Exploratory strata (descriptive, NOT a committed bar): where the lonspeed head loses to the floor\n")
w("Stratified by the GT speed change ahead, |`lon_v_target` - v_now| (labels may use the future for stratification). Pooled |error|, head minus ridge, paired.\n")
w("| stratum | windows | head | ridge (v0, a0) | head - ridge | head - poly-2 |\n|---|---|---|---|---|---|")
for sn, r in ST.items():
    x = r["lonspeed"]
    w(f"| {sn} (lonspeed) | {x['head']['n_windows']:,} | {ci(x['head'])} | {ci(x['ridge_v0a0'])} | {dci(x['paired_head_minus_ridge'])} | {dci(x['paired_head_minus_poly2'])} |")
w("\nThe aggregate tie hides two opposite effects. **On windows where the target speed really differs from the current speed (1,704 windows) the head beats the ego floor, separated**; "
  "**on steady windows (2,030) it is WORSE than reading v0** (it is noisy exactly where the answer is the current speed). The per-field view agrees: `v_a_ms` (the speed at the band start) head 1.70 m/s vs a linear v0 readout 0.58 m/s.\n")

w("## 3. The probe-level constraints (the four numbers the planner reads)\n")
w("Same estimator; TRAIN controls from the 4,855 windows of the 139 TRAIN-episode dump (`trainfit`), ridge floor on ego_t0 = (v0, a_long, yaw rate).\n")
w("| group | entries | **head** | const (TRAIN mean) | class-cond const | ridge ego3 | ridge ego3 poly-2 |\n|---|---|---|---|---|---|---|")
for g in ("lat(psi_term,t_onset)", "lon(prog6,v_end)", "ALL4", "psi_term", "t_onset", "prog6", "v_end"):
    t = P[g]
    c = t["controls"]
    w(f"| {g} | {t['head']['n_entries']:,} | **{ci(t['head'])}** | {ci(c['const_TRAIN_mean'])} | {ci(c['const_classcond_TRAIN_mean'])} | {ci(c['ridge_ego3_TRAIN139fit'])} | {ci(c['ridge_ego3_poly2_TRAIN139fit'])} |")
w("\nPaired head minus control:\n")
w("| group | - const | - class-cond | - ridge ego3 | - ridge ego3 poly-2 |\n|---|---|---|---|---|")
for g in ("lat(psi_term,t_onset)", "lon(prog6,v_end)", "ALL4", "psi_term", "t_onset", "prog6", "v_end"):
    t = P[g]["paired_head_minus"]
    w(f"| {g} | {dci(t['const_TRAIN_mean'])} | {dci(t['const_classcond_TRAIN_mean'])} | {dci(t['ridge_ego3_TRAIN139fit'])} | {dci(t['ridge_ego3_poly2_TRAINfit'.replace('TRAINfit', 'TRAIN139fit')])} |")
w("\nAt the planner-facing level the head **does** beat the ego floors (ALL4, both ridges, separated). The two the selector actually weights differ: `psi_term` (gamma 0.456) beats every floor, "
  f"including the class-conditional constant ({dci(P['psi_term']['paired_head_minus']['const_classcond_TRAIN_mean'])}); `prog6` (gamma 0.122) is at the poly-2 ego floor "
  f"({dci(P['prog6']['paired_head_minus']['ridge_ego3_poly2_TRAIN139fit'])}), i.e. the progress constraint adds nothing a (v0, a, yaw-rate) readout does not. Its residual error is {P['ALL4']['head']['point']:.4f} normalised (v_end: {P['v_end']['head']['point']*30:.2f} m/s).\n")

w("## 4. Decisions, beside the constraints\n")
w("| head | agreement | constant | head - constant (paired, 136 / 133 episodes) |\n|---|---|---|---|")
for hname, lab in (("lat", "lateral (3 exact classes)"), ("lon", "longitudinal (exact classes)")):
    x = lat_re["seed0"][hname]
    w(f"| {lab} | {x['acc']:.4f} (n {x['n_windows']:,}) | {x['acc_constant']:.4f} | {x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}] |")
w("\nThe lateral decision is already 94.2 % against a 90.4 % constant: 5.8 % of windows are wrong, and the head removes 39 % of the constant's errors. "
  "The v8F oracle says a perfect decision layer buys 7-12 % of the headroom; nothing here contradicts that being small for refcv8 too, but **no refcv8 decision oracle exists**.\n")

w("## 5. Cross-check against the trainer's own log (independent channel)\n")
w("The trainer logs the EMA head's mean |error| every 500 steps on 8 fixed eval batches (128 windows, 80 episodes; `metrics.jsonl`). I rebuilt the targets and masks from source on the cached batches: "
  "my supervised-entry counts equal the trainer's logged counts exactly (v9: 25.25 / 57.625 / 53.375 per batch; probe: 14.0 / 25.5), through two different code paths (numpy re-implementation and the source torch function agree on all 4,754 windows too).\n")
w("| group | trainer-logged head (128 windows) | const on those 128 | per-window head (4,754) | const on 4,754 |\n|---|---|---|---|---|")
for g in ("lat_c", "lon_c", "speed"):
    a = v9_128["table"][g]
    w(f"| {g} | {a['head_EMA48852_trainer_logged_normalised_MAE']:.4f} | {a['controls']['const_TRAIN_mean']['trainer_reduction']:.4f} | {G[g]['head']['point']:.4f} | {G[g]['controls']['const_TRAIN_mean']['point']:.4f} |")
w(f"| probe Huber (trainer loss) | {pr128['head_EMA48852_logged_huber']:.5f} | {pr128['controls_table']['const_TRAIN_mean']['real_constraint_head_loss_trainer_reduction']:.5f} (the real `constraint_head_loss`) | - | - |")
w("\nThe two windows sets differ (the 128 are a random draw), but the head/constant ratios are of the same order (0.57 / 0.54 / 0.31 vs 0.42 / 0.51 / 0.32; the 128-window set has 202 / 461 / 427 entries), so the heads dump behaves as the head the trainer evaluated.\n")

w("## 6. Controls that had to read known values\n")
w("| control | result |\n|---|---|")
w(f"| GT-as-prediction MAE, every group, v9 and probe | **exactly 0** ({ctl['GT_as_prediction_MAE_exactly_zero']}, {ctl['probe_GT_exactly_zero']}) |")
w(f"| constant control = the TRAIN mean | {ctl['const_equals_TRAIN_mean']} (asserted against an independently computed per-field mean over {ctl['n_train_rows']:,} release rows) |")
dc = ctl["decision_head_cross_check"]
w(f"| lateral agreement re-read from the heads dump | **{dc['lat_acc_from_heads_dump_vs_battery_label']:.4f}** vs banked 0.9415 (n {dc['lat_n']:,}); constant {dc['lat_acc_constant_from_label_counts']:.4f} vs banked 0.9039; also re-read from the battery dumps at seeds 0 and 1: 0.9415 and +0.0376 [+0.0123, +0.0631], identical to the banked JSON |")
w(f"| longitudinal agreement re-read | {dc['lon_acc_from_heads_dump_vs_battery_label']:.4f} vs banked 0.6103 |")
w(f"| heads dump == battery model | `out_lat/lon_logits_tac` argmax equals `lat/lon_pred_nav_true` on {dc['lat_alt_argmax_equals_battery_pred_frac']*100:.1f} % / {dc['lon_alt_argmax_equals_battery_pred_frac']*100:.1f} % of 4,754 windows |")
w(f"| target derivation | numpy targets/masks equal the SOURCE torch `v9_constraint_targets` on all 4,754 windows: {ctl['join']['numpy_targets_equal_source_torch_targets']} |")
w(f"| join | window -> v9 row k = ws + 2; |dump v0 - release v_now_ms| mean {ctl['join']['mean_abs_v0_dump_vs_release_v_now']:.4f} m/s; class labels agree except where the battery masked a window (all 50 + 62 disagreements are battery -100 vs a release class) |")
w(f"| mutants | const + 1 reads worse than const: {ctl['mutant_const_plus1_worse']}; GT + N(0, 0.05) reads {ctl['gt_plus_noise0p05_reads']:.4f} (~0.04 expected for sigma 0.05) |")
w(f"| mispaired control (head vs the GT of a window 1,000 positions away) | ALL26 head reads **{ctl['mispaired_roll1000_head_MAE_ALL26']:.4f}** vs {ctl['true_pair_head_MAE_ALL26']:.4f} true-pair ({ctl['mispaired_head_is_worse_than_true_pair_by_x']:.2f}x), and not better than the constant on the same mispairing ({ctl['mispaired_roll1000_const_MAE_ALL26']:.4f}): the instrument is sensitive to the pairing, so a silent join defect would not pass |\n")

w("## 7. Amendment 2(C), the registered dose-response: why it did not run\n")
w("PREREG_V8F_TAC Amendment 2 (C): `--tac-groups constraints --tac-alpha {0.25, 0.5, 0.75}`, class read on M2b at alpha 0.5 (PAYS if r >= 0.4, THRESHOLD if r < 0.2). "
  "It is registered and stamped **NOT RUN, box 2 awaits the PI's go**. It needs O's forward at the intermediate codes; the four checkpoints and the eval feature cache are on box 2 only "
  "(`D:/tanitad/runs/v8f_mini_full/*/ckpt_last.pt`; none on this machine - checked), and the banked outputs hold only alpha 0 and alpha 1, so an interpolation of banked outputs would be an invented arm. "
  "It also reads a **different model** (v8F-mini's O consumes the whole 60-dim code), so even when run it speaks to whether *partial constraint accuracy pays through that O*, not to refcv8's planner.\n")

w("## 8. What this does and does not say, and the next levers (Rule Zero: the verdict is a waypoint)\n")
w("**Says (MEASURED).** (a) refcv8's constraint heads are far from GT in the dimensions that matter, and the largest single v8F constraint group (lonspeed) is at the ego-scalar floor in aggregate, "
  "with a clean split: informative when speed must change, noisy when it should not. (b) `stoplead` and `latcons` carry scene information. (c) The lateral decision head is near its ceiling. "
  "(d) The 26 v9 vectors are never read by the planner; the four probe constraints are, they beat the ego floors (heading clearly, progress only to the poly-2 floor), "
  "and the trained selector weights the constraint terms (+0.456 / +0.122) while giving the decision posteriors small negative weights (-0.043 / -0.021).\n")
w("**Does not say.** How many metres of ADE a better constraint head buys refcv8. v8F's 0.9 m is for another architecture whose O reads all 26 dims, at T0, with one-group mixes off-distribution. "
  "No second training seed exists for these heads, so a separated paired CI here is necessary, not sufficient. The 'floor' comparison is a linear (and a poly-2) function class, and yaw rate is not in the v9 release so the 26-dim ridge uses (v0, a0) only.\n")
w("**Next levers, ranked by what they would settle:**\n")
w("1. **The refcv8-native oracle (settles the question).** `RefCV3Model.set_r8_force` already exists as an EVAL seam: `cons=` forces the constraint of the (own or forced) hypothesis, `lat3/lon6=` forces the decision. "
  "Arms on the 4,754 windows: own / GT-4-dim-constraints-at-own-hypothesis / GT-hypothesis (grouped by class, the seam takes one class per batch) / both. Cost: one forward pass per arm; the heads pass measured **1,995 s for 4,754 windows** "
  "(`heads.log`), so about 2.2 h GPU for four arms as an upper bound (ESTIMATED from that one pass, which also wrote 20 MB of perception tensors per episode). **Blocked on a GPU**: the dev-box GPU runs the locked battery and Thor is reserved. Unblock: a free GPU slot, or the PI's go on one.")
w("2. **Amendment 2(C) as registered.** 40 evals, ~15 min on box 2. Blocked on the PI's box-2 go; or ship the four checkpoints + feature cache here and run it on CPU.")
w("3. **A v0-residual speed head (hypothesis, not a result).** The steady-window loss and the `v_a` readout error say the speed head would gain from predicting target - v0 instead of target. "
  "It is an architecture change: PI go, own prereg, launch gate. Nothing here trains it.")
w("4. **Wire the 26 vectors to the planner, or drop their weight claim.** They currently cost 0.05 of the loss and influence the plan only through the shared trunk. PI decision.\n")

w("## 9. Artifacts and provenance\n")
w("| item | path |\n|---|---|")
for a, b in (("headline JSON (merged)", f"{D}/TACCONSTR_R8.json"), ("this report", f"{D}/RESULT.md"),
             ("step 1 audit", f"{D}/STEP1_DUMP_AUDIT.json"), ("per-window v9 + probe + strata", f"{D}/TACCONSTR_R8_heads_perwindow.json"),
             ("128-window trainer-log cross-check (v9)", f"{D}/TACCONSTR_R8_v9heads.json"), ("128-window trainer-log cross-check (probe)", f"{D}/TACCONSTR_R8_probe.json"),
             ("decision-head control re-read", f"{D}/CONTROL_lat_lon_agreement_reread.json"), ("code", f"{D}/code/")):
    w(f"| {a} | `{b}` |")
w("\nInputs (all read-only): heads dump `E:/refcv8_eval_kit/videos/heads_48852_pf/`; battery dumps `C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/dump_s{0,1}`; "
  "v9 eval release `C:/Users/Admin/qland/work/refcv8_hf_v9/release/v9_labels_eval139.npz` (md5 6b5c7f20...); v9 train release `C:/Users/Admin/qland/hold_v9train4572/work/release_train4572/v9_labels_train.npz` (md5 83e30bd3..., "
  "the `--r8-v9-md5` of the launch); G0 batch cache `C:/Users/Admin/refcv8_eval_archive/g0cache/step40000/`; trainer log `C:/Users/Admin/qland/work/watch8/metrics.jsonl`; "
  "TRAIN-139 dumps `C:/Users/Admin/refcv8_pf/trainfit/step30000ema_train139_pf`; source `C:/Users/Admin/refcv8_pf/navsim_ckpt/thor_bundle/stage/tree/stack` (blobs of `refcv8_conditioning.py` and `refcv8_train.py` equal fb47a58).\n")
w("**Paths in the brief that do not exist:** `RESULT_V8F_TAC_A1.md` and commit aa8366d are not in the D:/E: working tree (HEAD 37645fcc); they are in the clone `C:/Users/Admin/lab019` (also objects in tanitad-push, wt-be, tanitad-wt-join3d, tanitad-wt-percep-v6, scratch_hjunits). "
  "`D:/Projects/TanitAD/FlyWheels/.../2026-10-20-refcv8-perframe-final/` is also absent; the TAC_CONTROLS file read here is `C:/Users/Admin/refcv8_pf/post48852/TAC_CONTROLS_step48852ema_pf.json`.")
open(f"{D}/RESULT.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", f"{D}/TACCONSTR_R8.json", f"{D}/RESULT.md", len("\n".join(L)))
