# PREREG_V8F_MIXO: result, stage s0 (executed 2026-10-25 on box 2; X = M2b + `--code-mix kind`, seed 0 only; seed 1 not trained, as committed)

**Outcome, as committed: FAIL, O LEARNED.** The PRIMARY missed its bar at seed 0, and the dose response r(0.5) rose by more than 0.10.

* **Evidence class:** MEASURED (ours). Every number below cites `mixo_read_s0.json`, the per-run eval JSONs, or the hand re-derivation (paths in the manifest, section 10).
* **Tier.** The PREREG commits "T1-class open loop (eval-141: 2,399 windows, 141 episodes)" for the own-code reads, and T0 oracle evals for D-M and HARM. Nothing here is a driving claim. CAVEAT: The eval JSONs stamp themselves `raw.tier = "T0-open-loop-own-held-outputs (trunk-probe grid)"`. Both labels mean open-loop scoring of the logged future. I use the PREREG's label and have not reconciled the two stamps.
* **Estimator of every interval:** paired episode-cluster bootstrap, B = 2,000, over 141 episodes (2,399 windows). The reader uses `taniteval.ci.paired_episode_cluster_bootstrap`. The hand re-derivation uses its own implementation (`numpy.random.default_rng(0)`, 2.5 / 97.5 percentiles). A level shown with its own interval (for example an arm's ADE6) is an UNPAIRED per-arm interval and is marked as such.
* **The question every interval answers:** "would another draw of EPISODES say this?". It does NOT answer "would another TRAINING RUN say this?" (X is one seed; B has two banked seeds, and F = |M2b_s0 − M2b_s1| = 0.1207 m is the only training-variance information on this rig). It does not answer "would another INFERENCE RUN say this?" either: the PGT PREREG records that v8F-mini's planner is deterministic.
* **Sign convention.** X − B (X = `M2b_s0` with `--code-mix kind`, B = banked `M2b_s0`). Errors: negative = X better. D-M uses the TAC_A2 definition. HARM is the GT-code oracle ADE6, X − B, positive = X worse.
* **Pre-registration:** `Project Steering/PREREG_V8F_MIXO.md` at commit `9daaa255` ("written 2026-10-25, before any instrument, any KIND data or any MIXO data"). Code under test: the gate commit `18c62306`. **Nothing was staged, committed or pushed by this write-up.**

---

## 0. The three reads and the outcome (the committed tables, applied verbatim)

F = 0.1207 m (B's own two seeds, `|M2b_s0 − M2b_s1|`); the committed bar 2F = 0.2415 m.

| read | condition as committed | what was measured (seed 0; n = 2,399 windows / 141 episodes) | result |
|---|---|---|---|
| **PRIMARY**: own-code ADE6, X_s − M2b_s | "≤ −2F, separated" at seed 0 (bar −0.2415 m) | **+0.0224 m [−0.0577, +0.1059]**, p(Δ>0) 0.6985, NOT separated. ADE6: X 2.1977, B 2.1753 | **MISSED.** The point sits 0.264 m above the bar, and even the interval's lower end (−0.058) is 0.18 m above it |
| **D-M** (mechanism): the α-dose response r(0.5) of `RESULT_V8F_TAC_A2.md` (C) on X, against B's 0.36 | "r(0.5) rises by ≥ 0.10 over B's" | r_X(0.5) = **0.6440**, against B's committed 0.36 (measured 0.3647): rise **+0.2840** (+0.2793 over the measured B) | **Rise ≥ 0.10: met** (2.8× the bar) |
| **HARM** (beside, "never a bar"): GT-code oracle ADE6, X − B | beside; not deployable | **+0.4002 m [+0.3488, +0.4525]**, separated. Oracle ADE6: X 1.3710, B 0.9709 | X reads the GT code WORSE |
| families, Λ | beside, per family, never pooled | section 5 | reported |

| outcome (PREREG table) | condition | this run |
|---|---|---|
| PASS | PRIMARY met at both seeds | no |
| **FAIL, O LEARNED** | PRIMARY missed at seed 0, r(0.5) rises by ≥ 0.10 over B's | **YES** |
| FAIL, O DID NOT LEARN | PRIMARY missed at seed 0, r(0.5) rises by < 0.10 | no |

* **Seed 1 was not trained.** The PREREG commits "Seed 1 runs only after a seed-0 PASS". The reader's decision is `STOP`, verdict `FAIL-O-LEARNED`, `table_cell: defined`.
* **The committed next step, quoted from the PREREG:** *"O now consumes partial codes, but T is too wrong where O reads it: the T-side arm on the kind KIND ranks first"*.
* The rig's committed run-to-run floor is "the seed-paired swing (0.102 m ADE6, R66). An effect below it is not reported as a lever effect, whatever its interval." The PRIMARY, +0.022 m, is below that floor and below F = 0.1207 m. **It is not a lever effect in either direction.**

NOTE: this is a FAIL of the PRIMARY at its committed bar and it is reported as one. Section 9 says what the next arm is and what blocks it.

---

## 1. The verdict, re-derived by hand beside the reader's numbers

**Why a second derivation.** The MIXO mutation proof (model, reader, queue and chain groups) ended at 148 of 162 mutants RED with **14 survivors** (`raw/2026-10-25-mixo/mutation_proof_rerun.out` at `9daaa255`). They include the exact-boundary case ("a PRIMARY exactly at −2F does not clear") and the verdict aggregation ("a miss is O-LEARNED if ANY missed seed learned"). The other survivors are: five probe-refusal mutants, a next-step text altered (PASS), the dose identity cells not enforced, a non-finite dose ADE6 accepted, oracle-arm window disagreement without a v9 row, K = 10 lead windows not compared, the GT-code check not run, and "B need not be a whole-code run". So the verdict was recomputed independently, from the per-window eval arrays, without importing the reader (`verify_mixo_s0.py`, output `verify_mixo_s0.out`).

| quantity | reader (`mixo_read_s0.json`) | hand re-derivation | difference |
|---|---|---|---|
| PRIMARY point (X − B ADE6) | +0.022355262121917 | +0.022355262121917 | < 1e-15 |
| PRIMARY interval | [−0.0577, +0.1059] | [−0.05854, +0.10111] | 0.0008 / 0.0048 |
| PRIMARY separated / met | no / no | no / no | none |
| ADE6 X, B | 2.19766, 2.17531 | 2.19766, 2.17531 | none |
| r_X(0.5) | 0.6439541671317497 | 0.6439541671317492 | 5e-16 |
| r_B(0.5) | 0.36468449009235 | 0.36468449009235 | none |
| d_full X / B | −0.80096 / −1.07556 | −0.80096 / −1.07556 | none |
| d_half X / B | −0.51578 / −0.39224 | −0.51578 / −0.39224 | none |
| rise over committed 0.36 | +0.283954167 | +0.283954167 | none |
| HARM point | +0.40016006 | +0.40016006 | 3e-16 |
| HARM interval | [+0.3488, +0.4525] | [+0.34748, +0.44947] | 0.0013 / 0.0030 |
| `tacpred` vs standard-eval ADE6, X and B | max dev 0.0 m | max dev 0.0 m | none |
| verdict | `FAIL-O-LEARNED` | `FAIL-O-LEARNED` | **same** |

* All point estimates agree to floating-point precision. The interval endpoints differ in the third decimal. That is bootstrap-draw variation (the same estimator, the same resampling unit and pairing, B = 2,000, a different random stream), not a different estimate. No conclusion changes: not separated in either, bar missed in either.
* The hand script reads `ade6` from the eval `.npz` files (these hold no GT; `pred == traj30` was asserted `true`) and asserts that X's and B's `eid` arrays are equal.
* CAVEAT: *My reading of the survivors, not a committed statement:* neither the exact-boundary survivor nor the "ANY missed seed" survivor can bear on this verdict. The PRIMARY misses by 0.264 m (no boundary case), and only one seed was read (no cross-seed aggregation). The hand derivation makes this moot. The five probe-refusal survivors concern the refusal paths of the training-time `code_mix_probe`; the probe's actual output on the real run is in the selfcheck (section 6: 124 of 128 draws partly GT).

---

## 2. PRIMARY: own-code ADE6, X − B (tier T1-class open loop)

| quantity | X (`M2b-mixo_s0`) | B (`M2b_s0`) | X − B (paired) |
|---|---|---|---|
| **ADE6, m** | 2.1977 [2.0219, 2.3873] (own interval, unpaired) | 2.1753 [2.0071, 2.3568] (own interval, unpaired) | **+0.0224 [−0.0577, +0.1059]**, p(Δ>0) 0.6985, not separated; n 2,399 / 141 |
| FDE6, m (beside) | 6.4306 [5.8841, 7.0182] | 6.2810 [5.7498, 6.8269] | not computed paired by the reader |
| ADE2, m (beside) | 0.2814 [0.2424, 0.3293] | 0.3155 [0.2765, 0.3621] | difference of points −0.0341; intervals overlap; not computed paired |

* CAVEAT: ADE2 is not a committed read. X's point is lower by 0.034 m. No paired interval exists in the reader or the eval JSON, so this is a descriptive observation, not a finding.
* Training pairing (checked from the run records, not assumed): the same 19,164 steps, 306,566 windows, 4,512 clips, `excluded_sha256 f6a1dee0…`, `train_sha256 37f4f9ee…`, 7,214,495 trainable parameters with an identical group split (adapter 275,712 · cond 25,088 · S 1,321,737 · T 1,463,868 · O 2,120,216 · WM 2,007,874). X's argv differs from B's by exactly `--code-mix kind` and `--out`.
* Wall-clock: X 14,375 s, B 15,741 s.

---

## 3. D-M: the α-dose response r(0.5) on X (tier T0 oracle; TAC_A2's own definition)

r(α) = d(α) / d(1), where d(α) = mean ADE6 with T's constraints moved α of the way to GT, minus mean ADE6 with T's own code (`-tacpred`). Paired over 2,399 windows / 141 episodes.

| quantity | X | B (`M2b_s0`) |
|---|---|---|
| ADE6, T's own code (`-tacpred`) | 2.1977 | 2.1753 |
| d(1): all constraints GT, minus own | −0.8010 [−0.9221, −0.6912] | −1.0756 [−1.1946, −0.9656] |
| d(0.5): constraints halfway, minus own | −0.5158 [−0.5903, −0.4455] | −0.3922 [−0.4742, −0.3162] |
| **r(0.5) = d(0.5) / d(1)** | **0.6440** | **0.3647** (TAC_A2's own 0.3646; committed 0.36) |

* **Rise over B's committed 0.36: +0.2840.** The bar is ≥ +0.10. **Met.** Over the measured B: +0.2793.
* r is a ratio of two paired means and the reader attaches no interval to it. The committed bar is on the point value, so none is invented here.
* CAVEAT: *Post hoc arithmetic on the reported numbers (not a committed read, no interval):* both terms of the ratio moved. The numerator |d(0.5)| grew by 1.31× (0.392 → 0.516 m) and the denominator |d(1)| shrank to 0.74× (1.076 → 0.801 m). In ADE6 terms, at α = 0.5 X would sit at 1.682 m against B's 1.783 m (X lower by 0.101 m), and at α = 1 at 1.397 m against B's 1.100 m (X higher by 0.297 m). The last of these is related to, but not the same as, the HARM read below (HARM uses the full GT code, decisions included; d(1) moves only the constraints and keeps T's decisions).

---

## 4. HARM (beside, never a bar): the GT-code oracle (tier T0)

| | X | B | X − B (paired) |
|---|---|---|---|
| oracle ADE6 (`-tacorc`: O reads the full GT code), m | 1.3710 | 0.9709 | **+0.4002 [+0.3488, +0.4525]**, separated; n 2,399 / 141 |

* The PREREG anticipated the direction: "O now sees fewer all-GT codes". It is 1.41× B's oracle ADE6.
* Source files: the HARM value reads `-tacorc` for both arms (B's mean 0.97085 matches `RESULT_V8F_KIND.md`'s "full GT code 0.971 at s0"). The `gt_source` field in the reader JSON ("B: `-tacorc-all-goal`, X: `-tacorc`") names which file supplied the GT constraint **codes** for the dose reads, because B's `-tacorc.npz` has no `tac_code` array. It does not name the HARM file. (Checked: B's `-tacorc` mean is 0.9709, B's `-tacorc-all-goal` mean is 1.0500.)

---

## 5. The four metric families (binding), X − B

Common to every cell unless stated: **tier T1-class open loop; paired episode-cluster bootstrap, B = 2,000; n = 2,399 windows / 141 episodes; one training seed (answers "another draw of episodes" only).** "Separated" = the 95 % interval excludes 0. In the reader's `families` block the comparator key is named `p10`, but its values are B's (checked: `speed_mae` 1.0674 and `yaw_rate_mae` 6.1605 equal `eval_M2b_s0.json`; X's 1.0502 equals `eval_M2b-mixo_s0.json`).

### 5.1 LONGITUDINAL

**CAVEAT: anti-echo is UNDISCHARGED on both arms.** Both eval JSONs report `anti_echo: UNAVAILABLE` ("no ego speed at t0 in the window dict … v0 is NOT imputed") with the consequence quoted verbatim: *"the PI's 2026-08-16 anti-echo condition is UNDISCHARGED on this block. Every LONGITUDINAL number beside it must be read as UNVERIFIED against a hold-v0 copy — it may not be presented as a longitudinal capability result."* Both JSONs also report their own `distance_keeping: UNAVAILABLE` (no lead track supplied to the eval). The distance-keeping numbers below come from the reader's separate join to the banked lead block, not from the eval JSON.

**Speed, along-track, acceleration (n 2,399 / 141):**

| cell | X | B | X − B [95 % CI] | separated |
|---|---|---|---|---|
| speed MAE, m/s | 1.0502 | 1.0674 | −0.0172 [−0.0588, +0.0234] | no |
| along-track MAE, m | 1.8225 | 1.8097 | +0.0128 [−0.0596, +0.0882] | no |
| **acceleration MAE, m/s²** | 1.0900 | 1.2058 | **−0.1158 [−0.1561, −0.0743]** | **yes (X better, −9.6 %)** |

Further levels, **no paired interval in the JSONs** (descriptive): speed bias X −0.4175, B −0.3986 m/s; speed RMSE 1.6578 vs 1.6552; target-speed accuracy within 0.5 / 1.0 / 2.0 m/s, X 0.4547 / 0.6557 / 0.8398 vs B 0.4466 / 0.6446 / 0.8339 (fractions of 71,970 horizon steps, not windows; the bands are "PROPOSED reporting tolerances, not a gate"); along-track bias −0.6162 vs −0.5561 m; along-track final bias −2.4016 vs −2.2475 m; ego-progress ratio mean 0.9300 vs 0.9275 (n 2,367, 32 windows excluded for low GT progress); under-progress rate 0.6907 vs 0.6738.

**Distance keeping, K = 10 (the committed block; 2 s; "the banked K = 10 block", `PREREG_V8F_PGT.md`)**: LEAD windows 743 for both arms.

| metric | X level [own CI] (n / episodes) | B level [own CI] (n / episodes) | **paired X − B [95 % CI]** (n_used / episodes) |
|---|---|---|---|
| min headway, m | 27.661 [23.892, 31.865] (699 / 73) | 27.515 [23.727, 31.623] (705 / 74) | +0.0739 [−0.0099, +0.1591] (694 / 73), not separated |
| min time gap, s | 4.1355 [3.3439, 5.1065] (660 / 73) | 4.1411 [3.3249, 5.1627] (661 / 74) | −0.0019 [−0.0406, +0.0281] (660 / 73), not separated |
| min TTC, s (censored at a 30 s cap; closing windows X 384, B 413) | 20.737 [18.833, 22.608] (699 / 73) | 20.612 [18.668, 22.443] (705 / 74) | +0.2076 [−0.2480, +0.6058] (694 / 73), not separated |
| \|headway error\| vs the human's own value, m | not tabulated | not tabulated | **−0.0752 [−0.1506, −0.0045]** (683 / 72), **separated** (X closer to the human) |
| \|time-gap error\| vs human, s | | | **−0.0277 [−0.0626, −0.0056]** (649 / 72), **separated** |
| \|min-TTC error\| vs human, s | | | −0.3718 [−0.7683, +0.0707] (683 / 72), not separated |

* LEAD-VIOL (plan time gap < 1 s while the human's is ≥ 1 s): **5 of 743 lead windows for both arms**; at-risk share 5 / 605 = 0.0083 [0.0016, 0.0176] for both; paired delta 0.0 [0.0, 0.0] (605 / 70).
* Progress control: lead-progress ratio X 1.0079, B 1.0149; paired −0.0070 [−0.0124, −0.0014] (604 / 70), **separated** (X advances 0.7 pp less); `meets_min_progress` true for both.
* The min-TTC means are over censored data: quote `n_closing` beside them. (The JSON's `n_censored_at_cap` reads 381 for X and 377 for B, larger than "windows that never close" (315, 292); not reconciled here.)

**Reader's extra block, K = 30 (6 s; lead block `b1_eval_lead_block_k30.npz`, sha256 `7a879baa…73d9`, 29,556 rows, join key (clip_id, raw frame 2t), speed proof max 2.6e-05 m/s against a 1e-3 tolerance; LEAD 743 / NO_LEAD 552 / NOT_STRAIGHT 931 / NO_LABEL 173):** this block is not in the PREREG's table, so it is beside.

| metric | paired X − B [95 % CI] (n_used / episodes) | separated |
|---|---|---|
| min headway, m | +0.1831 [−0.0433, +0.3989] (719 / 73) | no |
| min time gap, s | +0.0610 [−0.0047, +0.1456] (663 / 73) | no |
| min TTC, s | +0.3133 [−0.2211, +0.8412] (719 / 73) | no |
| \|headway error\| vs human, m | +0.1538 [−0.0439, +0.3557] (712 / 72) | no |
| \|time-gap error\| vs human, s | +0.0118 [−0.0646, +0.0995] (656 / 72) | no |
| \|min-TTC error\| vs human, s | −0.2019 [−0.7197, +0.3433] (712 / 72) | no |
| LEAD-VIOL at risk (23 vs 21 violations; 595 at risk) | +0.0034 [−0.0038, +0.0123] (595 / 70) | no |
| lead-progress ratio (0.9152 vs 0.9243) | −0.0091 [−0.0299, +0.0116] (595 / 70) | no |

**Per-family gaps (longitudinal):** none missing. The only caveat is the undischarged anti-echo condition above, which is a property of the eval, not of this arm.

### 5.2 LATERAL

n = 2,399 windows / 141 episodes. The heading, curvature and yaw-rate cells are MICRO means over valid steps (steps with displacement below `min_ds` are excluded: 5,014 excluded steps on B). Valid-step counts differ by arm because the mask depends on each arm's own path: heading X 67,026 / B 66,956; curvature and yaw rate X 64,489 / B 64,435.

| cell | X | B | X − B [95 % CI] | separated |
|---|---|---|---|---|
| heading MAE, ° | 3.9843 | 3.9773 | +0.0070 [−0.2002, +0.2235] | no |
| **yaw-rate MAE, °/s** | 6.6868 | 6.1605 | **+0.5263 [+0.1792, +0.9648]** | **yes (X worse, +8.5 %)** |
| curvature MAE, 1/m | 0.048942 | 0.044356 | +0.004586 [−0.001911, +0.012410] | no |
| cross-track MAE (lateral offset at matched time index), m | 0.7799 | 0.7662 | +0.0137 [−0.0307, +0.0625] | no |

**Per-family gaps (lateral):** none missing. Note the eval's own warning that `cross_mae` is an offset at a matched time index, not a distance to the GT path, and is uninterpretable alone when the along-track error is large.

### 5.3 TACTICAL

CAVEAT: **No paired X − B interval exists for any tactical cell** in the reader JSON or the eval JSONs. What exists is each arm's own (unpaired) interval where the eval emits one. This is a **work item**: the per-window arrays that would support a paired bootstrap (`lat_logits`, `lon_logits`, `goal_prob`) are in the eval `.npz` files, but no paired tactical read was computed here.

*(a) From-trajectory decisions* (executed manoeuvres labelled from the plan, by `refc_tactical.factor_from_kinematics`; n 2,399 / 141; per-arm episode-cluster intervals, unpaired):

| cell | X | B | point difference |
|---|---|---|---|
| lateral decision, kappa | 0.6685 [0.5957, 0.7302] | 0.6652 [0.6036, 0.7193] | +0.0033 |
| lateral decision, accuracy | 0.8441 [0.8107, 0.8749] | 0.8458 [0.8173, 0.8745] | −0.0017 |
| longitudinal decision, kappa | 0.3398 [0.2807, 0.3953] | 0.3566 [0.2994, 0.4100] | −0.0168 |
| longitudinal decision, accuracy | 0.5661 [0.5248, 0.6069] | 0.5790 [0.5385, 0.6175] | −0.0129 |
| 5-way collapsed, kappa | 0.4529 [0.3932, 0.5075] | 0.4586 [0.4035, 0.5074] | −0.0057 |
| goal-point error (= FDE6), m | 6.4306 [5.8841, 7.0182] | 6.2810 [5.7498, 6.8269] | +0.1496 |
| goal bearing MAE, ° (n 2,367) | 4.1132 [2.7864, 5.7422] | 4.0999 [2.7704, 5.8253] | +0.0133 |

*(b) The tactical heads' macro-F1* (point values only, no interval in the eval JSON; lateral n = 2,107 (support 1,910 / 47 / 150 over three of five classes), longitudinal n = 2,240 (7 classes), goal head over 22 tokens, 3 of them without support):

| cell | X | B | point difference | majority baseline |
|---|---|---|---|---|
| lateral macro-F1 | 0.8269 | 0.8157 | +0.0112 | 0.3170 |
| longitudinal macro-F1 | 0.5015 | 0.4992 | +0.0024 | 0.0707 |
| goal macro-F1 | 0.2712 | 0.2618 | +0.0095 | 0.0 |

* **Unavailable (both arms, with the eval's reason):** anchor selection ("this dump's arm commits to ONE path per window, so there is no candidate fan and no selection to score"), and "selected vs executed" ("Both label streams are EXECUTED manoeuvres … stays unavailable on a trajectory dump"). Both are work items, not passes.
* Reading, descriptive: every tactical point difference is small (|Δ| ≤ 0.017 in kappa or accuracy, ≤ 0.011 in macro-F1), and the unpaired per-arm intervals overlap almost completely. That is a statement about the intervals shown, not a paired test.

### 5.4 STRATEGIC

The read says: **NOT PAUSED, un-paused for v8F-mini (PI ruling 2026-10-17)**; each eval JSON carries `families.strategic = {"status": "see H-MINI-STRAT-1 (checkpoint error)", "paused": false}`. The numbers are S's checkpoint position and heading error, from the eval JSON. This write-up reports them and does not decide the family's status.

CAVEAT: **No paired X − B interval exists.** The intervals below are each arm's own (unpaired, episode-cluster). This is a **work item** (the per-window `ck_pos_err` array is in the eval `.npz`).

| cell | n (windows / episodes) | X [own CI] | B [own CI] | point difference |
|---|---|---|---|---|
| position error @ 30 m, m | 2,322 / 137 | 7.0141 [6.3807, 7.7034] | 7.0231 [6.4063, 7.7107] | −0.0090 |
| position error @ 50 m, m | 2,319 / 137 | 8.0368 [7.3217, 8.8102] | 8.0122 [7.3116, 8.7866] | +0.0246 |
| position error @ 80 m, m | 2,294 / 136 | 11.3682 [10.0932, 12.7702] | 11.2059 [10.0432, 12.5054] | +0.1623 |
| heading error @ 30 m, ° | 2,322 / 137 | 4.1894 [3.0569, 5.6547] | 4.2002 [3.1335, 5.5686] | −0.0108 |
| heading error @ 50 m, ° | 2,319 / 137 | 6.7680 [5.1761, 8.5357] | 6.6864 [5.0692, 8.4948] | +0.0816 |
| heading error @ 80 m, ° | 2,294 / 136 | 12.9855 [9.8982, 16.5150] | 13.0540 [9.9683, 16.6572] | −0.0685 |

The n is reduced below 2,399 on every strategic row (n_valid 2,322 / 2,319 / 2,294 at 30 / 50 / 80 m); the JSON does not state the reason, so none is claimed here. Every point difference is far inside the intervals.

### 5.5 Beside (named in the PREREG: Λ; reader-added: M-DK share, T's present slot)

| read | X | B | X − B [95 % CI] (n / episodes) |
|---|---|---|---|
| **Λ, the latcons leak** (along-track at 2 s of `-tacorc-all-latcons` minus that of `-tacorc`, on the LEAD windows of the K = 10 block), m | 0.0467 | 0.2678 [0.1578, 0.3888] | **−0.2211 [−0.3406, −0.1120]** (743 / 77), separated (−82.6 %) |
| M-DK share S5 (share of LEAD windows with 2 s min-TTC < 5 s) | 0.1542 | 0.1715 | −0.0173 [−0.0350, −0.0028] (694 / 73), separated; all-LEAD variant −0.0162 [−0.0325, −0.0026] (743 / 77). B's excess over P10 is +0.0442 |
| T's present-slot RMSE (n 2,351 windows / 139 episodes) | 0.5455 | 0.5363 | MSE delta +0.0099 [−0.0010, +0.0219], not separated |
| T present-slot RMSE by group, X / B | latcons 0.6891 / 0.6861; lonspeed 0.4668 / 0.4515; stoplead 0.8208 / 0.8155 | | descriptive |

* The PREREG names Λ as a beside read and commits no bar or meaning for it. The "halves" and "at least half" bars in `PREREG_V8F_PGT.md` belong to that PREREG and are not applied here.
* The reader's `mleak` / `mdk` are the PGT PREREG's M-LEAK and M-DK definitions, re-used.

### 5.6 Reading across the families (descriptive, not a pooled score)

Four cells are separated at this one seed: **acceleration MAE better**, **Λ lower**, **S5 lower** (all three favour X), and **yaw-rate MAE worse**. Two further distance-keeping cells are separated in X's favour (|headway error| and |time-gap error| vs the human), and the progress control is separated with a slightly lower X ratio (1.0079 against 1.0149). The ADE6 PRIMARY, speed, along-track, heading, curvature, cross-track, tactical and strategic cells show no separated difference. Section 8 says why a separated cell at one training seed is necessary and not sufficient.

---

## 6. Controls (every committed control read; the read was not stopped)

| control | committed condition | result |
|---|---|---|
| identity control (the pipeline run on B against B), s0 | exactly +0 [0, 0] | all 15 cells read `0.0 [0.0, 0.0]`: primary, mleak, mdk, mdk_alt, present_mse, oracle_ade6, the seven longitudinal and lateral family cells, dose_half, dose_full. The K = 30 distance-keeping identity (headway, time gap, min-TTC, lead-viol, n_used 722 / 665 / 722) is also exactly 0 |
| known values reproduced (reader `known_values.checked = true`) | within tolerance | Λ(B) 0.26784 / 0.16669 (banked 0.2678 / 0.1667; tol 6e-4); excess over P10 0.04422 / 0.02503 (banked 0.0442 / 0.0250; tol 6e-5); F 0.12074 (banked 0.1207; tol 6e-5) |
| B's r(0.5) reproduced from the banked dose files (`check_inputs.json`) | against the committed 0.36 / 0.21 | s0 0.36468 (TAC_A2's own 0.36463); s1 0.20847 (TAC_A2's own 0.20843) |
| own-code ADE6 equals the standard eval, per window | max deviation 0 | `tacpred` against the standard eval: X 0.0 m, B 0.0 m. (The header difference between the two standard evals is `tac`: X `pred`, B `None`. The control shows ADE6 is unaffected.) |
| same windows and episodes | 2,399 / 141 | yes; the hand script asserts X's and B's `eid` arrays are equal; 743 LEAD windows in both K blocks |
| **checkpoint sha256** | asserted before and after | `M2b_s0` `601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8`; `M2b_s1` `6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6`; **X `c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d` after training AND after the seven evals (identical)**. I re-hashed the local copy `runs/M2b_s0_mixo/ckpt_last.pt` (86,969,237 bytes): the same value |
| train-log p_gt schedule | B's: 1.0 → 0.2 | 384 records; first `p_gt` 0.99795 (step 50), last 0.20004 (step 19,164). **Identical to B's `p_gt` array over all 384 records (max abs difference 0.0; checked here)**. Both 19,164 steps. The lever changes only the granularity of the draw, not the schedule |
| launch gate | PASS token bound to commit, argv and data | `gate_token.json`: `gate PASS`, commit `18c623068626d5f9f57d5aeb09d2fddb6b89e1fb`, argv_sha `9b10652f4eafc33a`, eight key-file sha256s, `train_sha256 37f4f9ee…`, `excluded_sha256 f6a1dee0…`, v9 md5s `83e30bd3…` / `6b5c7f20…` |
| G-DVB / G-LIVE (`gate_M2b-mixo/selfcheck.json`) | declared = built; every term present; frozen-prior control exactly 0 | `declared_vs_built PASS`; built `code_mix = kind`, `p_gt 1.0 → 0.2`; no missing, unexpected or dead parameters; `control_prior_grad_sum = 0.0`; `n_trainable` 252 |
| the mixing reached O (`code_mix_probe`, p_gt = 0.5) | a PARTIAL code | 128 draws: **124 partly-GT, 3 all-GT, 1 all-own**; per-unit GT rate 0.469 to 0.570 (decisions 0.500, speed 0.570, angle 0.484, tte 0.539, timegap 0.555, geom 0.469, leadgap 0.477): expected GT fraction ≈ p_gt per cell, joint changed |
| the run's selfcheck against the gate's | same | identical except the `argv_sha` line (`argv_sha` hashes the whole argv, `--out` included, so it differs between the gate invocation and the run) |

**Pre-run proofs the PREREG lists, and where they live (INHERITED from the committed artifacts; not re-run for this write-up):**

* **Reachability from the caller** (`raw/2026-10-25-mixo/trace_reachability.out`): twelve hops from the argv flag to the code O reads, with file:line at each (the mixing line is `v8f_mini.py:663`; `run_o` reads its output at `:675`). Deleting the mixing line makes the in-process gate refuse (`ZZDVB-REFUSEDZZ`, RED).
* **`whole` = B bit for bit**, **200-step loss trace identity**, **analytic identities at p_gt 1 / 0 / 0.5 (4,096 draws, partial)**, **G-DVB entry**: the tests `test_whole_builds_the_saved_pre_change_model_and_forward_bit_for_bit…`, `test_whole_reproduces_the_saved_pre_change_200_step_loss_trace_exactly_and_kind_does_not`, `test_p_gt_1_kind_equals_whole…`, `test_p_gt_0_both_modes_hand_o_exactly_t_s_own_code`, `test_p_gt_half_the_per_sample_gt_fraction_is_partial_across_4096_draws…`, `test_the_flag_has_its_entry_and_a_new_flag_without_one_is_refused` in `stack/tests/test_v8f_mixo.py`. The suite record (`suites.out`) reads 998 passed, 2 skipped.
* **Mutation proof** of the training-side instrument, and of the reader and chain: 148 of 162 mutants RED, 14 survived (listed in section 1).
* **A provenance note on the run summaries.** B's `summary.json` records `residual_zero_init: false`, X's records `true`, and B's `levers` block carries fewer keys. `v8f_mini_train.py` writes the summary's `levers` from the same dict as the pre-training `selfcheck.json`, and the repo's tests (`test_v8f_grid.py:218`) name this key as "written BEFORE training (True) and AFTER it (False): the only expected difference". I did not inspect the weights. X's and B's argv differ only by `--code-mix kind`.

---

## 7. What it means (inside the PREREG's committed meanings)

**The committed meaning, FAIL, O LEARNED (quoted):** *"O now consumes partial codes, but T is too wrong where O reads it: the T-side arm on the kind KIND ranks first."*

What the measured numbers say in those terms, and no further:

1. **The PRIMARY missed.** Training O on codes mixed per unit did not move own-code ADE6 by 2F at seed 0: it moved by +0.022 m [−0.058, +0.106], which is not separated and is below the committed floor. The lever did not buy ADE6.
2. **O did learn to use a partial code.** r(0.5) rose from 0.36 to 0.64 (rise +0.284 against a bar of +0.10). By the committed table, that is the "O learned" branch.
3. **The committed reading of the pair:** the missing ADE6 gain is attributed to T's error where O reads it ("T is too wrong where O reads it"), not to O's inability to consume a partly right code.

**Post hoc, labelled, committing nothing (arithmetic and descriptive observations on the reported numbers):**

* The rise in r is carried by both terms (section 3): X gains more from a half-correct T, and gains less from a fully correct one.
* HARM: X reads the full GT code 0.400 m worse than B does. The PREREG listed this direction as a possibility and made it "never a bar".
* Λ fell by 82.6 % and the 2 s low-TTC share S5 fell by 1.7 pp (both separated). Acceleration MAE improved by 9.6 %. Yaw-rate MAE worsened by 8.5 %. None of these was committed as a bar for MIXO, and ADE6, the committed bar, did not move.
* These separated cells come from one training seed. Section 8 gives the caution.

---

## 8. What these numbers do not say

* **Training variance.** X is one seed. The paired bootstrap resamples episodes with the models held fixed. Per `H-ESTIM-SEED-1`, a separated interval from a one-seed arm is necessary, not sufficient, for a claim that a LEVER moved a metric. The only rig-level floor is B's two seeds (F = 0.1207 m ADE6; the PREREG also cites the R66 swing of 0.102 m). No replicate of X exists, and seed 1 was not trained because the PRIMARY failed. So the separated cells in section 5 (acceleration MAE, yaw-rate MAE, Λ, S5, the two vs-human distance-keeping errors, the progress ratio) are observations at one seed, not lever effects.
* **Inference variance.** Not applicable here (the planner is deterministic per the PGT PREREG).
* **Tier.** Open-loop scoring of the logged future (own-code reads) and GT-conditioned oracles (D-M, HARM). The oracle reads say how O consumes a code. They are not driving performance.
* **Longitudinal numbers are unverified against a hold-v0 copy** (anti-echo undischarged, section 5.1), for both arms.
* **Tactical and strategic have no paired interval** (section 5.3 and 5.4): the cells there are descriptive.
* **This is the seed-0 stage only.** The PREREG's PASS needs both seeds; the FAIL branch is fully decided by seed 0 and the committed next step does not depend on seed 1.

---

## 9. The next lever (RULE ZERO: a refutation leaves the next arm behind)

* **Committed next arm:** the T-side arm "on the kind KIND ranks first". None was run here. No PREREG exists for it yet (the programme's convention is that an arm is pre-registered before it runs), and box 2 has needed the PI's go for each arm, as for this one.
* CAVEAT: **A gap in the committed wording, flagged and not resolved here.** `RESULT_V8F_KIND.md` read **INTERACTIVE**: no kind clears 2F at both seeds, and it states that single-kind gains "are lower bounds, and they cannot rank T-side levers". Its gain_k table (ADE6 against T's own code, s0 / s1) reads speed +0.270 / +0.185 (Amendment 2: "not a carrier"), angle +0.075 / +0.043, time-to-event −0.237 / −0.315, geometry −0.434 / −0.421, time gap and lead gap about 0. KIND's own write-up offers one lead for later: GT angle cells are worth +0.77 / +0.55 m inside the 140 bend windows (17 episodes, wide intervals). So "the kind KIND ranks first" is not defined by KIND's committed output. Which kind the T-side arm targets, if any, is the Master Mind's and the PI's call and needs its own PREREG. This write-up does not choose.
* **What blocks it:** (1) the kind is undefined by KIND's committed output; (2) the T-side arm needs a PREREG; (3) box 2 needs the PI's go. Nothing else is blocked: this stage's instruments, controls and reads are complete.
* **RESOLVED by the Master Mind, 2026-10-25, before any data: `PREREG_V8F_KINDX` (`eeb2043e`).** KIND's single-kind gains could not rank T-side levers because O could not use a partly right code; this stage shows O on X now can (r(0.5) 0.644). So the ranking KIND could not give on B is re-read on X: KIND's own per-kind oracle, on `M2b-mixo_s0`, a seed-0 screen on the dev-box CPU. CANDIDATE (some kind's gain ≥ 2F, separated) ⇒ train X at s1 and re-read (a carrier needs both seeds), then the T-side arm on that kind under its own PREREG; NO CANDIDATE ⇒ the longitudinal line (KSCORE, OACC) stays the top lever. This is also the re-read `PREREG_V8F_KIND` commits to after its O-side arm.
* **Open work items from this read (not bars):** the anti-echo condition is undischarged in both eval JSONs (no hold-v0 copy checked the longitudinal numbers); paired CIs for the tactical and strategic families are not computed (the per-window arrays exist); the eval JSONs stamp `raw.tier` T0 for the own-code eval where the PREREG and this document say T1-class open loop. The reconciliation is an instrument-labelling item.

---

## 10. Manifest (nothing staged, committed or pushed; box 2's folder was read only)

| artifact | location |
|---|---|
| this result | `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/mixo_hand/RESULT_V8F_MIXO.md` |
| every reader number | `C:/Users/Admin/box2_ship/logs/v8f_mixo/mixo_read_s0.json` (schema `v8f-mixo-read-v1`) |
| the hand re-derivation | `…/scratchpad/mixo_hand/verify_mixo_s0.py`, `verify_mixo_s0.out` |
| per-run evals (JSON and per-window `.npz`) | `C:/Users/Admin/box2_ship/logs/v8f_mixo/eval/eval_M2b-mixo_s0*.{json,npz}` (seven evals), `eval_M2b_s0*.{json,npz}`, `eval_M2b_s1*` (the F floor) |
| controls and inputs | `…/v8f_mixo/check_inputs.json`, `ckpt_sha256.json`, `gate_token.json`, `gate_M2b-mixo/selfcheck.json` |
| the X run | `…/v8f_mixo/runs/M2b_s0_mixo/` (`argv.json`, `summary.json`, `selfcheck.json`, `train.log`, `train_log.jsonl`, `ckpt_last.pt`); on box 2 `D:\tanitad\runs\v8f_mini_full\M2b_s0_mixo\ckpt_last.pt` |
| telemetry | `…/v8f_mixo/telemetry_v8fmixo.csv` |
| PREREG | `Project Steering/PREREG_V8F_MIXO.md` at `9daaa255` (repo mirror `C:/Users/Admin/tanitad-push`) |
| code, tests, pre-run proofs | gate commit `18c62306`: `stack/tests/test_v8f_mixo.py`, `…/2026-10-17-v8f-mini/code/v8f_mixo_read.py`, `…/raw/2026-10-25-mixo/{trace_reachability.out, mutation_proof_rerun.out, suites.out, check_inputs_real.out}` |
