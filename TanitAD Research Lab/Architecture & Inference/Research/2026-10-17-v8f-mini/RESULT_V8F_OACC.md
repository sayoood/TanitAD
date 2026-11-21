# PREREG_V8F_OACC: result, stage s0 (X = M2b + `--w-acc 42.9311`, seed 0 only; trained on box 2 and read 2026-10-26; seed 1 not trained, as committed)

**Outcome, as committed: COSTS POSITION.** PRIMARY-ACC passed at seed 0 (accel MAE X − B −0.5663 m/s², separated). PRIMARY-KEEP failed (ADE6 X − B upper bound +0.1262 m against the bar +0.06037 m). DOMINANCE (HX − P10) also failed, on one row: curvature MAE is separated-worse (+0.0084), where the row needs separated-better.

* **Evidence class:** MEASURED (ours). Every number below is copied from `oacc_read_s0.json` (schema `v8f-oacc-read-v1`) or the run records beside it; JSON key paths are in section 11. Nothing in this document is a new statistic. Where a sentence compares two printed numbers, it says which.
* **Tier, as the reader states it:** `T1-class open loop (eval-141, 2,399 windows over 141 episodes)`. Nothing here is a driving claim. CAVEAT: both eval JSONs (X and B) stamp themselves `raw.tier = "T0-open-loop-own-held-outputs (trunk-probe grid)"`. Both labels mean open-loop scoring of the logged future; I use the reader's label and have not reconciled the two stamps (the same open item as `RESULT_V8F_MIXO.md`).
* **Estimator of every interval:** paired episode-cluster bootstrap, B = 2,000, over 141 episodes (2,399 windows), `taniteval.ci.paired_episode_cluster_bootstrap` through the reader. A level shown with its own interval (an arm's ADE6, a strategic error) is an UNPAIRED per-arm episode-cluster interval and is marked as such.
* **The question every interval answers:** "would another draw of EPISODES say this?". It does NOT answer "would another TRAINING RUN say this?" (X is one seed, with no replicate; the reader's own estimator line: "training-seed variance enters only through F (H-ESTIM-SEED-1)"). The planner is deterministic (the reader says so), so the INFERENCE-run question does not arise.
* **Sign convention.** X − B, HX − B, HX − X, HX − P10 (B = banked `M2b_s0`; X = `M2b-oacc_s0`; HX = X with per-step weights re-fitted on X's own training windows, the deployed form; P10 = the kinematic floor `p10_yaw4512`). Errors: negative = the first arm better.
* **Pre-registration:** `Project Steering/PREREG_V8F_OACC.md` with Amendment 1 (`ea67b7a`, before any OACC data) and Amendment 2 (`A2-2026-10-26`, `04d5667`, POST-DATA and CONTROL ONLY: see section 10). Code under test: tip `04d5667114b82087c29611f0dd9b8043771c6e58`.
* **Status of this file:** a draft for the Master Mind. Nothing was staged, committed or pushed by it.

---

## 0. The three reads and the outcome (the committed tables, applied verbatim)

F = 0.12074012131477874 m (B's own two seeds, `|ADE6(M2b_s0) − ADE6(M2b_s1)|`); the committed KEEP bar F/2 = +0.06037006065738937 m (Amendment 1, I1: "the table's '+0.060' is its rounding").

The PREREG's reading table, quoted:

| read | definition | bar |
|---|---|---|
| **PRIMARY-ACC** | accel MAE X − B | **< 0 and separated, at the seed run** |
| **PRIMARY-KEEP** | ADE6 X − B | **upper bound ≤ +F/2 = +0.060 m** |
| **DOMINANCE** (the programme read) | HX − P10, where HX is X with HBLEND's per-step weights re-fitted on X's own training windows (the deployed form) | HSPEC's PRIMARY-DOM row verbatim: separated-better on ADE2, ADE6, accel MAE and curvature MAE; not separated-worse on any other family metric |

What was measured (seed 0; tier T1-class open loop; paired episode-cluster bootstrap, B = 2,000; n = 2,399 windows / 141 episodes):

| read | point [95 % interval] | separated | bar | result |
|---|---|---|---|---|
| **PRIMARY-ACC**: accel MAE X − B, m/s² | **−0.5663 [−0.6724, −0.4711]** | yes, X better | < 0 and separated | **PASSED** |
| **PRIMARY-KEEP**: ADE6 X − B, m | **+0.0451 [−0.0322, +0.1262]** | no | upper bound ≤ +0.06037006 | **FAILED**: the upper bound 0.1262 is above the bar. The point (+0.0451) is below the bar; the bar is on the upper bound |
| **DOMINANCE**: HX − P10 | section 3 | 14 of 15 rows meet their requirement | all rows | **FAILED** on `curvature MAE` alone: +0.0084 [+0.0039, +0.0139], separated-worse, needed separated-better |

The PREREG's outcome table, quoted, and the reading of it (Amendment 1, I2, "applied in this order"):

| outcome | condition | next (committed) |
|---|---|---|
| **DOMINATES ITS PRIOR** | DOMINANCE at both seeds (s1 runs on s0's PRIMARY-ACC and PRIMARY-KEEP pass) | HX is the deployed v8F output and the register states dominance over the **kinematic floor** on every family; never as state of the art. Next: the same row against FLAT |
| **SMOOTHER, NOT DOMINANT** | PRIMARY-ACC and PRIMARY-KEEP pass, DOMINANCE fails | X becomes the base for later arms. The acceleration gap is quantified per horizon; next, the longitudinal decision (`PREREG_V8F_KHYP`) |
| **COSTS POSITION** | PRIMARY-ACC passes, PRIMARY-KEEP fails | the trade-off is reported; the weight rule is not re-tuned post hoc |
| **NO EFFECT** | PRIMARY-ACC fails | the acceleration error is not reachable through O's loss at this capacity; next, an O-capacity read |

| I2 step (Amendment 1) | this run |
|---|---|
| 1. PRIMARY-ACC fails at any seed that has run ⇒ NO EFFECT | no (ACC passed) |
| 2. Else PRIMARY-KEEP fails at any seed that has run ⇒ **COSTS POSITION** | **YES** (`seed_states.s0`: ACC true, KEEP false, DOM false) |

* **Matched row: COSTS POSITION.** The reader's record: `OUTCOME` "COSTS POSITION"; `s1_required` false; `next` "COSTS POSITION: the trade-off is reported; the weight rule is not re-tuned post hoc."
* **Seed 1 was not trained.** The PREREG commits "s1 runs on s0's PRIMARY-ACC and PRIMARY-KEEP pass". The chain's closing line: "seed 1 NOT run: seed 0 failed PRIMARY-ACC or PRIMARY-KEEP".
* **The committed next step, quoted from the PREREG:** *"the trade-off is reported; the weight rule is not re-tuned post hoc."*

### What failed, stated plainly

1. **PRIMARY-KEEP failed.** The ADE6 interval is [−0.0322, +0.1262]; its upper end is above +0.06037. The data neither show a position cost nor exclude one larger than the bar. The point +0.0451 is not separated from 0 and sits below F = 0.12074 (the only run-to-run information on this rig).
2. **DOMINANCE failed, on curvature MAE only.** HX − P10 curvature MAE +0.0084 [+0.0039, +0.0139] is separated-worse. The other 14 rows meet their requirement (section 3).
3. **Therefore neither "SMOOTHER, NOT DOMINANT" nor "DOMINATES ITS PRIOR" is available.** No dominance statement over P10 is made, at one seed or otherwise.

---

## 1. The weight rule and the arm

* **Rule (Amendment 1, R1 to R3):** `w_acc = 0.5 · L_traj(B_s0) / L_acc(B_s0)` on B's s0 checkpoint over HBLEND's fit windows. Measured (`wacc_record`): `L_traj` 1.2459830045700073 (`w_traj · (ade/3 + 0.5·fde/6)`, with `L_ade` 2.1679351329803467 and `L_fde` 6.280055522918701), `L_acc` 0.01451141107827425, 9,582 windows, `w_acc_raw` 42.93114562909151, **`w_acc` = 42.9311** (the only value in X's argv; `check_w_acc_x_L_acc_over_L_traj` 0.4999994685782217). The ADE-only reading, 74.6976, is shown in the record and was not used. The record agrees with the banked fit windows (`banked_crosscheck`: relative difference 0.0 on all four quantities, max |Δ traj30| 0.0 m, `pass` true); the offline dry check printed in Amendment 1 (1.24598 / 0.0145114 / 42.9311) is the same value.
* **X's argv differs from B's by exactly `--w-acc 42.9311` and `--out`** (checked here: ordered diff of `runs/M2b_s0_oacc/argv.json` against `runs/M2b_s0/argv.json`).
* **Training pairing (from the run records):** both runs 19,164 steps, 306,566 windows, 4,512 clips, `excluded_sha256 f6a1dee0…`, `train_sha256 37f4f9ee…`, 7,214,495 trainable parameters, an identical group split (adapter 275,712 · cond 25,088 · S 1,321,737 · T 1,463,868 · O 2,120,216 · WM 2,007,874). Wall-clock: X 14,442 s (`summary.json` `seconds` 14442.006851911545), B 15,741 s.

---

## 2. PRIMARY-ACC and PRIMARY-KEEP

**Tier T1-class open loop; paired episode-cluster bootstrap, B = 2,000; n 2,399 / 141. One training seed.**

| quantity | X (`M2b-oacc_s0`) | B (`M2b_s0`) | X − B (paired) |
|---|---|---|---|
| **accel MAE, m/s²** (eval level) | 0.6395 [0.5935, 0.6869] (own, unpaired) | 1.2058 [1.0809, 1.341] (own, unpaired) | **−0.5663 [−0.6724, −0.4711]**, separated |
| **ADE6, m** (eval level) | 2.2204 [2.041, 2.4145] (own, unpaired) | 2.1753 [2.0071, 2.3568] (own, unpaired) | **+0.0451 [−0.0322, +0.1262]**, not separated |

* B's second seed, from `check_inputs.json`: accel MAE `B_accel_mae` s0 1.2058080453060727, s1 0.9313306247209101; `B_ade6` s0 2.175306138079546, s1 2.0545660167647672. X's accel MAE (0.6395) is below both B seeds' values. This is the only run-to-run information on this rig for either metric; X has no replicate.
* The reader's own sentence: "PREREG_V8F_OACC: COSTS POSITION; w_acc 42.9311; accel MAE X - B -0.5663 m/s^2 (need < 0 separated); ADE6 X - B upper +0.1262 m (bar +0.0604); DOMINANCE HX - P10 FAIL". The bar it prints (+0.0604) is the rounding of the exact `keep_bar` 0.06037006065738937; `hi_le_printed_0.060` reads false and the exact comparison reads false, so the two spellings agree.

---

## 3. DOMINANCE: HX − P10 (the programme read)

HX = X with HBLEND's rule (closed form, clip to [0, 1], isotonic) re-fitted on X's own held outputs over the TRAIN windows only (9,582 windows, 141 clips; `fit_clips_sha256 458ce297…`). **Tier T1-class open loop; paired episode-cluster bootstrap, B = 2,000.** The DOMINANCE record stores point, interval and flags per row; it does not store n per row. The four paired ADE rows of the same comparison are 2,399 / 141 (section 4); the heading, yaw-rate and curvature rows are over windows with valid steps, so their n is lower and is not stored in this block.

| row (HX − P10) | point [95 % interval] | separated | direction | requirement |
|---|---|---|---|---|
| ADE2, m | −0.0172 [−0.0215, −0.0124] | yes | better | separated-better: **met** |
| ADE6, m | −0.5029 [−0.6390, −0.3723] | yes | better | separated-better: **met** |
| accel MAE, m/s² | −0.0304 [−0.0484, −0.0113] | yes | better | separated-better: **met** |
| **curvature MAE, 1/m** | **+0.0084 [+0.0039, +0.0139]** | yes | **WORSE** | separated-better: **NOT MET** |
| speed MAE, m/s | −0.1130 [−0.1668, −0.0511] | yes | better | not separated-worse: met |
| along-track MAE, m | −0.2097 [−0.3053, −0.1065] | yes | better | not separated-worse: met |
| cross-track MAE, m | −0.4343 [−0.5435, −0.3269] | yes | better | not separated-worse: met |
| heading MAE, ° | −1.7264 [−2.1809, −1.3137] | yes | better | not separated-worse: met |
| yaw-rate MAE, °/s | −0.1664 [−0.4203, +0.0989] | no | neither | not separated-worse: met |
| plan-derived lateral decision agreement | +0.0738 [+0.0521, +0.0971] | yes | better | not separated-worse: met |
| plan-derived longitudinal decision agreement | +0.0492 [+0.0062, +0.0896] | yes | better | not separated-worse: met |
| CONSISTENCY(a) `any` (violation rate) | −0.0950 [−0.1266, −0.0634] | yes | better | not separated-worse: met |
| DK headway min, m (K = 30) | +0.4176 [+0.0201, +0.8343] | yes | better | not separated-worse: met |
| DK time-gap min, s (K = 30) | +0.1307 [+0.0180, +0.2526] | yes | better | not separated-worse: met |
| DK LEAD-VIOL at-risk share (K = 30) | −0.0286 [−0.0572, −0.0044] | yes | better | not separated-worse: met |

`DOMINANCE`: `pass` false; `all_better_ok` false; `all_not_worse_ok` true. The failing requirement is the fourth row alone.

---

## 4. Beside: ADE and FDE at each horizon (tier T1-class open loop; paired, B = 2,000; n 2,399 / 141)

| metric | X − B | HX − B | HX − P10 |
|---|---|---|---|
| ADE2, m | −0.0777 [−0.1003, −0.0574], separated | −0.1107 [−0.1499, −0.0806], separated | −0.0172 [−0.0215, −0.0124], separated |
| ADE4, m | +0.0129 [−0.0273, +0.0533] | −0.1267 [−0.1706, −0.0892], separated | −0.1558 [−0.1982, −0.1132], separated |
| **ADE6, m** | **+0.0451 [−0.0322, +0.1262]** | −0.1912 [−0.2671, −0.1174], separated | −0.5029 [−0.6390, −0.3723], separated |
| FDE6, m | +0.0123 [−0.1923, +0.2259] | −0.4235 [−0.6467, −0.2105], separated | −1.8098 [−2.2969, −1.3323], separated |

Levels from the eval blocks (each with its own unpaired interval): ADE2 X 0.2378 [0.2131, 0.2688], HX 0.2049 [0.1846, 0.2315], B 0.3155 [0.2765, 0.3621]. ADE6 X 2.2204 [2.041, 2.4145], HX 1.9841 [1.8271, 2.1474], B 2.1753 [2.0071, 2.3568]; P10 ADE6 2.4870 (`check_inputs.json` `P10_ade6` 2.4869921202702483). FDE6 X 6.2933 [5.7427, 6.8963], HX 5.8575 [5.365, 6.3712], B 6.281 [5.7498, 6.8269].

* Reading, descriptive: X's ADE2 is separated-better than B (−0.0777) while its ADE4 and ADE6 are not separated; HX's ADE4, ADE6 and FDE6 are separated-better than B. None of these was a committed bar except ADE6 X − B (PRIMARY-KEEP).

---

## 5. The four metric families (binding), per family, never pooled

Common to every cell unless stated: **tier T1-class open loop; paired episode-cluster bootstrap, B = 2,000; n = 2,399 windows / 141 episodes unless a row says otherwise; one training seed (answers "another draw of episodes" only).** "Separated" = the 95 % interval excludes 0.

**How to read the two kinds of lateral and longitudinal numbers.** The eval blocks give MICRO levels (means over valid steps; `lateral.excluded_below_min_ds` 4,952 steps on X, 5,014 on B, 3,945 on HX). The paired rows are means of per-window components (`fw_*`), over the windows that have a valid value. The two are different quantities: do not subtract micro levels to reproduce a paired delta. For example heading MAE levels are X 3.6534, B 3.9773 (micro), while the paired X − B is −0.3855 (per-window, n 2,342).

### 5.1 LONGITUDINAL

**CAVEAT: anti-echo is UNDISCHARGED on all three eval blocks (X, HX, B).** Each carries `anti_echo: UNAVAILABLE` ("no ego speed at t0 in the window dict … v0 is NOT imputed from gt/pred") with the consequence quoted verbatim: *"the PI's 2026-08-16 anti-echo condition is UNDISCHARGED on this block. Every LONGITUDINAL number beside it must be read as UNVERIFIED against a hold-v0 copy — it may not be presented as a longitudinal capability result."* Each eval block also reports its own `distance_keeping: UNAVAILABLE` (no lead track supplied to the eval). The distance-keeping numbers below come from the reader's separate join to the lead block (below), not from the eval JSON.

**Paired per-window rows** (the PRIMARY-ACC row is the second):

| cell | X − B | HX − B | HX − X |
|---|---|---|---|
| speed MAE, m/s | −0.1071 [−0.1542, −0.0641], separated | −0.2002 [−0.2578, −0.1482], separated | −0.0931 [−0.1258, −0.0618], separated |
| **accel MAE, m/s²** | **−0.5663 [−0.6724, −0.4711]**, separated | −0.5986 [−0.7130, −0.4944], separated | −0.0323 [−0.0498, −0.0135], separated |
| along-track MAE, m | +0.0536 [−0.0216, +0.1350] | −0.1977 [−0.2794, −0.1228], separated | −0.2513 [−0.3361, −0.1719], separated |

HX − P10 on the same cells is in section 3 (speed MAE −0.1130, along-track −0.2097, accel −0.0304, all separated-better).

**Levels** (eval blocks; X / HX / B; unpaired; no interval stored in the paired block): speed MAE 0.9603 / 0.8672 / 1.0674 m/s (own intervals X [0.8791, 1.0481], HX [0.7993, 0.9363], B [0.9771, 1.1638]); speed bias −0.4249 / −0.2956 / −0.3986; speed RMSE 1.5485 / 1.4077 / 1.6552; target-speed accuracy within 0.5 / 1.0 / 2.0 m/s: X 0.5077 / 0.6819 / 0.8544, HX 0.5208 / 0.716 / 0.8794, B 0.4466 / 0.6446 / 0.8339 (fractions of 71,970 horizon steps, not windows; the bands are "PROPOSED reporting tolerances, not a gate"); along-track MAE 1.8633 / 1.612 / 1.8097 m; along-track bias −0.7293 / −0.3918 / −0.5561 m; along-track final bias −2.253 / −1.3893 / −2.2475 m; accel MAE 0.6395 / 0.6072 / 1.2058 m/s²; ego-progress ratio mean 0.9185 / 0.9788 / 0.9275 (n 2,367, 32 windows excluded for low GT progress); under-progress rate 0.6743 / 0.6358 / 0.6738.

**Distance keeping, K = 30 (6 s; the HSPEC convention the PREREG inherits).** Lead block `raw/2026-10-22-dk/b1_eval_lead_block_k30.npz`, sha256 `7a879baa2b6f4c140c2b64c9c098a38f84ecb04926da61fa8e96cfffc33073d9`, 29,556 rows, join key (clip_id, raw frame 2t), speed proof max 2.6242784824326293e-05 m/s against a 1e-3 tolerance. Windows: LEAD 743, NO_LEAD 552, NOT_STRAIGHT 931, NO_LABEL 173. This is not the K = 10 block of `RESULT_V8F_MIXO.md`: the two are not comparable cell by cell.

Levels (own unpaired intervals; n windows / episodes):

| arm | min headway, m | min time gap, s | min TTC, s (censored at a 30 s cap; closing windows) | LEAD-VIOL at risk (violations / 595) |
|---|---|---|---|---|
| B | 25.6223 [21.8386, 29.7384] (722 / 74) | 4.0223 [3.1043, 5.269] (665 / 74) | 19.3964 [17.3135, 21.3442] (722 / 74); closing 458 | 0.035294 [0.019, 0.0553] (21) |
| X | 25.751 [22.0335, 29.8949] (725 / 74) | 4.008 [3.1342, 5.1634] (664 / 74) | 20.9303 [18.9177, 22.8388] (725 / 74); closing 403 | 0.036975 [0.0198, 0.0574] (22) |
| HX | 25.6698 [21.648, 30.0266] (731 / 75) | 3.9687 [3.0762, 5.181] (665 / 74) | 20.623 [18.5213, 22.4576] (731 / 75); closing 415 | 0.052101 [0.0294, 0.0781] (31) |
| P10 | 25.1162 [21.2126, 29.5019] (728 / 75) | 3.8316 [2.9634, 4.9659] (663 / 74) | 19.9473 [17.7066, 22.0248] (728 / 75); closing 411 | 0.080672 [0.0504, 0.1147] (48) |
| human | 25.6648 [21.6332, 29.7907] (725 / 73) | 3.9954 [3.1011, 5.0788] (659 / 72) | 19.7898 [17.4084, 21.9416] (725 / 73); closing 447 | — |

The min-TTC means are over censored data: quote the closing count beside them.

Paired (first − second; `n_used` / episodes):

| metric | X − B | HX − B | HX − X | HX − P10 |
|---|---|---|---|---|
| min headway, m | +0.2297 [−0.0428, +0.5339] (718 / 74) | −0.0854 [−0.3219, +0.1761] (720 / 74) | −0.3437 [−0.5622, −0.1293] (725 / 74), separated | +0.4176 [+0.0201, +0.8343] (727 / 75), separated |
| min time gap, s | −0.0076 [−0.1298, +0.0873] (662 / 74) | −0.0540 [−0.1218, +0.0063] (663 / 74) | −0.0477 [−0.1340, +0.0405] (664 / 74) | +0.1307 [+0.0180, +0.2526] (662 / 74), separated |
| min TTC, s | **+1.5046 [+0.7407, +2.3263]** (718 / 74), separated | **+1.1129 [+0.2942, +1.9769]** (720 / 74), separated | −0.3848 [−0.9298, +0.1339] (725 / 74) | +0.6380 [+0.0444, +1.3233] (727 / 75), separated |
| \|headway error\| vs the human, m | +0.0926 [−0.1477, +0.3102] (711 / 72) | +0.0756 [−0.1145, +0.2608] (712 / 72) | −0.0414 [−0.1914, +0.1069] (717 / 72) | −0.3455 [−0.6851, −0.0192] (720 / 73), separated |
| \|time-gap error\| vs human, s | −0.0544 [−0.1837, +0.0331] (655 / 72) | +0.0230 [−0.0377, +0.0852] (655 / 72) | +0.0773 [+0.0115, +0.1643] (656 / 72), separated (HX farther) | −0.0986 [−0.2211, +0.0108] (655 / 72) |
| \|min-TTC error\| vs human, s | −0.7616 [−1.4639, −0.0659] (711 / 72), separated | −0.7583 [−1.5517, −0.0593] (712 / 72), separated | +0.0022 [−0.4756, +0.5185] (717 / 72) | −0.3824 [−0.8438, +0.0559] (720 / 73) |
| LEAD-VIOL, all lead windows | +0.0013 [−0.0064, +0.0090] (743 / 77) | **+0.0135 [+0.0027, +0.0249]** (743 / 77), separated | **+0.0121 [+0.0026, +0.0230]** (743 / 77), separated | −0.0229 [−0.0462, −0.0015] (743 / 77), separated |
| LEAD-VIOL, at risk (595 / 70) | +0.0017 [−0.0080, +0.0114] | **+0.0168 [+0.0036, +0.0306]**, separated | **+0.0151 [+0.0019, +0.0284]**, separated | −0.0286 [−0.0572, −0.0044], separated |
| lead-progress ratio (595 / 70) | −0.0236 [−0.0499, −0.0028], separated (X 0.9007, B 0.9243) | +0.0451 [+0.0157, +0.0763], separated (HX 0.9694) | +0.0687 [+0.0426, +0.0998], separated | −0.1641 [−0.2404, −0.1043], separated (P10 1.1335) |

* The direction of "better" is labelled by the reader only for headway, time gap and LEAD-VIOL in the DOMINANCE block (section 3); the min-TTC cells above are numeric differences and carry no reader label.
* **A cost is visible here and belongs in the report:** HX has more LEAD-VIOL windows than B and X (31 against 21 and 22 of 595 at risk), separated in both pairs. Against P10 (48) HX has fewer, separated, and that is the DOMINANCE row.
* The reader's `x7_direction_met_at_risk` flag reads true for HX − P10, false for X − B, HX − B and HX − X.

**Per-family gaps (longitudinal):** the anti-echo condition (above). `seed_floor_F_of_arms_in_pair` is empty in the reader's DK block (no seed floor attached to the distance-keeping pairs); no longitudinal cell has an interval that speaks about another training run.

### 5.2 LATERAL

Paired per-window rows (n 2,399 / 141 unless stated):

| cell | X − B | HX − B | HX − X | HX − P10 |
|---|---|---|---|---|
| heading MAE, ° | −0.3855 [−0.6737, −0.1196], separated (2,342) | −0.7560 [−1.2622, −0.3637], separated (2,343) | −0.3088 [−0.5848, −0.0963], separated (2,351) | −1.7264 [−2.1809, −1.3137], separated |
| yaw-rate MAE, °/s | −0.9526 [−1.4634, −0.4095], separated (2,325) | −2.0323 [−2.8642, −1.3906], separated (2,328) | −1.0661 [−1.7606, −0.5115], separated (2,327) | −0.1664 [−0.4203, +0.0989] |
| **curvature MAE, 1/m** | −0.0044 [−0.0174, +0.0096] (2,325) | −0.0312 [−0.0536, −0.0160], separated (2,328) | −0.0267 [−0.0459, −0.0122], separated (2,327) | **+0.0084 [+0.0039, +0.0139], separated-WORSE** |
| cross-track MAE, m | −0.0101 [−0.0422, +0.0264] | −0.0147 [−0.0455, +0.0194] | −0.0045 [−0.0089, −0.0012], separated | −0.4343 [−0.5435, −0.3269], separated |

**Levels** (micro, eval blocks; X / HX / B; unpaired; own intervals in the eval JSON): heading MAE 3.6534 / 3.6417 / 3.9773 °; yaw-rate MAE 4.9808 / 4.9019 / 6.1605 °/s; curvature MAE 0.034686 / 0.031654 / 0.044356 1/m; cross-track MAE 0.756 / 0.7515 / 0.7662 m (an offset at a matched time index, not a distance to the GT path: uninterpretable alone when the along-track error is large).

* **The lateral cost is on curvature against the floor, not against B.** HX is separated-better than both B and X on curvature MAE, and separated-worse than P10 (+0.0084). The row sits in the PREREG's must-be-separated-better set, so it decides DOMINANCE.
* **Per-family gaps (lateral):** none missing. P10's own level is not stored in the reader JSON (only the HX − P10 differences are).

### 5.3 TACTICAL

**Paired plan-derived decision agreement** (higher is better; the manoeuvre decisions labelled from the executed plan by `refc_tactical.factor_from_kinematics`):

| cell | X − B | HX − B | HX − X | HX − P10 |
|---|---|---|---|---|
| lateral decision agreement | −0.0042 [−0.0188, +0.0104] | +0.0008 [−0.0138, +0.0158] | +0.0050 [+0.0021, +0.0083], separated | +0.0738 [+0.0521, +0.0971], separated |
| longitudinal decision agreement | +0.0029 [−0.0125, +0.0200] | +0.0038 [−0.0129, +0.0221] | +0.0008 [−0.0058, +0.0063] | +0.0492 [+0.0062, +0.0896], separated |

**From-trajectory decisions, levels with each arm's own unpaired interval (n 2,399 / 141):**

| cell | B | X | HX |
|---|---|---|---|
| lateral decision, kappa | 0.6652 [0.6036, 0.7193] | 0.6478 [0.5902, 0.699] | 0.6596 [0.604, 0.7101] |
| lateral decision, accuracy | 0.8458 [0.8173, 0.8745] | 0.8416 [0.8122, 0.8699] | 0.8466 [0.8178, 0.8737] |
| longitudinal decision, kappa | 0.3566 [0.2994, 0.41] | 0.3607 [0.3023, 0.4172] | 0.362 [0.3026, 0.4188] |
| longitudinal decision, accuracy | 0.579 [0.5385, 0.6175] | 0.5819 [0.5413, 0.6215] | 0.5827 [0.5417, 0.6229] |
| 5-way collapsed, kappa | 0.4586 [0.4035, 0.5074] | 0.458 [0.4056, 0.5043] | 0.465 [0.4134, 0.5123] |
| goal-point error (= FDE6), m | 6.281 [5.7498, 6.8269] | 6.2933 [5.7427, 6.8963] | 5.8575 [5.365, 6.3712] |
| goal bearing MAE, ° (n 2,367) | 4.0999 [2.7704, 5.8253] | 4.5788 [3.0652, 6.5037] | 3.2512 [2.3204, 4.6742] |

**The tactical heads' macro-F1** (point values only; X's and HX's tactical-head blocks are equal in the JSON and `HX_equals_X_by_construction` reads true; lateral n = 2,107, support 1,910 / 47 / 150 over three of five classes; longitudinal n = 2,240 over 7 classes; goal head over 22 tokens, 3 of them without support):

| cell | B | X (= HX) | majority baseline |
|---|---|---|---|
| lateral macro-F1 | 0.8156957183062286 | 0.8067299703990352 | 0.3169861422288607 |
| longitudinal macro-F1 | 0.49915533884637825 | 0.5059046846309375 | 0.07073276068909257 |
| goal macro-F1 | 0.26175532861267575 | 0.29441946085632603 | 0.0 |

* **Unavailable (all arms, with the eval's reason):** anchor selection ("this dump's arm commits to ONE path per window, so there is no candidate fan and no selection to score … A WORK ITEM, not a pass"), and "selected vs executed" (both label streams are executed manoeuvres). Both are work items.
* **WORK ITEM: no paired X − B interval exists for the tactical heads (macro-F1, kappa, per-token F1) or for the goal-setting cells.** The per-window arrays that would support one are in `eval_M2b-oacc_s0.npz` and `eval_M2b_s0.npz`; `stack/scripts/v8f_paired_families.py` computes exactly this for the MIXO pair and was not run on this pair. The unpaired intervals above overlap almost completely for every cell; that is a statement about the intervals shown, not a paired test.

### 5.4 STRATEGIC

The read says: **NOT PAUSED for v8F** (`strategic_family`: "READ for v8F (switched on: PI-V8F-2026-10-17, R72)"; `families.strategic.status` the same; the eval JSONs carry `{"status": "see H-MINI-STRAT-1 (checkpoint error)", "paused": false}`). The numbers are S's checkpoint position and heading errors, from the eval JSONs. This write-up reports them and does not decide the family's status.

**S is not B's S.** `identical_to_B` false; `identical_by_construction` false, with the reader's reason: "X is a NEW training run with an extra loss term; O's gradient reaches S through the checkpoints in the full interface, so S is B's architecture, not B's weights".

Levels, each arm's own unpaired episode-cluster interval:

| cell | n (windows / episodes) | X | B |
|---|---|---|---|
| position error @ 30 m, m | 2,322 / 137 | 7.0379 [6.4204, 7.7286] | 7.0231 [6.4063, 7.7107] |
| position error @ 50 m, m | 2,319 / 137 | 8.0573 [7.3504, 8.8492] | 8.0122 [7.3116, 8.7866] |
| position error @ 80 m, m | 2,294 / 136 | 11.2206 [10.018, 12.538] | 11.2059 [10.0432, 12.5054] |
| heading error @ 30 m, ° | 2,322 / 137 | 4.3174 [3.17, 5.8015] | 4.2002 [3.1335, 5.5686] |
| heading error @ 50 m, ° | 2,319 / 137 | 6.8143 [5.1383, 8.6527] | 6.6864 [5.0692, 8.4948] |
| heading error @ 80 m, ° | 2,294 / 136 | 12.9148 [9.8727, 16.4534] | 13.054 [9.9683, 16.6572] |

**Paired position error, X − B** (`X_minus_B_ck_pos_err`; paired episode-cluster bootstrap, B = 2,000):

| cell | X − B | n (windows / episodes) |
|---|---|---|
| mean over the three checkpoints | +0.0259 [−0.1574, +0.1984], not separated | 2,322 / 137 |
| checkpoint index 0 (n equals the 30 m row's) | +0.0148 [−0.0455, +0.0753], not separated | 2,322 / 137 |
| checkpoint index 1 (n equals the 50 m row's) | +0.0451 [−0.1058, +0.1919], not separated | 2,319 / 137 |
| checkpoint index 2 (n equals the 80 m row's) | +0.0147 [−0.3370, +0.3724], not separated | 2,294 / 136 |

The reader does not label the three checkpoints; the mapping to 30 / 50 / 80 m is by the n of each row (2,322 / 2,319 / 2,294 = `n_valid` of the 30 / 50 / 80 m cells). **No paired heading-error interval exists** in the reader JSON: the heading cells are unpaired levels (work item, as for the tactical heads).

**What the `Mean of empty slice` warning means (chain log, `v8f_oacc_read.py:441`).** The paired block computes `np.nanmean(cx, 1)` over the three checkpoint columns per window. A window whose three checkpoint position errors are all NaN has no value to average, so numpy warns and returns NaN, and the paired bootstrap drops that window. Checked here on the eval `.npz` files: `ck_pos_err` has **77 windows with all three columns NaN** (and 105 with any NaN), the same 77 windows in X and in B (and in `M2b_s1`); the non-NaN counts per column are 2,322 / 2,319 / 2,294. 2,399 − 77 = 2,322 = the `n_windows` of the mean-over-checkpoints row. So the warning is the expected consequence of 77 windows with no valid checkpoint label; it did not change a number. (The same `RuntimeWarning: Mean of empty slice` at that line is also printed by the reader's own test suite in the HACC-MC chain's C-7 run, `hmc/logs/suite.out`, so it is a known property of the reader and not new to this read.) The cause of those 77 is not stated in the JSON (the MIXO addendum attributes them to missing label rows and invalid checkpoints).

**A flag in the JSON that is vacuous:** `families.strategic.HX` is `null` and `HX_equals_X_by_construction` reads `true`. The reader computes the latter as `jround(resHX.get("strategic")) == jround(resX.get("strategic"))` (`v8f_oacc_read.py`, the `strategic` entry of the families block). With HX's value null, the flag can read true only if X's value is null too, so it compares two empty values. Checked here: `X_eval_block.families.strategic` equals `HX_eval_block.families.strategic`, but each holds only the status string above, no numbers. So there is no measured HX strategic number in this read. HX is a post-map on O's plan and the construction gives no route for it to alter S's checkpoints, but that is the construction, not a measurement here.

### 5.5 Beside: CONSISTENCY(a), the HX weights, HS

* **CONSISTENCY(a)** (plan-vs-declared-code violation rate; lower is better), point rates B / X / HX / P10: `viol_any` 0.15964985410587745 / 0.17465610671112963 / 0.21592330137557317 / 0.310962901208837; `viol_SPEED` 0.09545644018340975 / 0.10921217173822426 / 0.14422676115047936 / 0.24176740308461858; `viol_STOP` 0.018757815756565235 / 0.01834097540641934 / 0.028761984160066693 / 0.033764068361817424; `viol_TURN` 0.004168403501458941 / 0.00917048770320967 / 0.006252605252188412 / 0.03418090871196332; `viol_LEAD` 0.07253022092538558 / 0.0712796998749479 / 0.0921217173822426 / 0.13839099624843684; `self_viol_any` (no P10 value) 0.22967903293038766 / 0.23843268028345144 / 0.31513130471029593. **Only HX − P10 is paired** (−0.0950 [−0.1266, −0.0634], separated, n 2,399 / 141): X and HX sit above B on `viol_any` as descriptive point values with no paired interval. That is a visible cost of the lever and of the blend, and a work item (no paired X − B or HX − X consistency read exists).
* **HX weights** (`weights_HX`, 30 steps each, fitted on 9,582 TRAIN windows of 141 clips; `fit_clips_sha256` `458ce2975914b20c7dd91ed4b47032de55d86d9b8fda4df8a0cc0e3ca09921b1`): `w_along_by_step` runs from 0.0 at step 0 to 0.6639919617911029 at step 29; `w_cross_by_step` starts at 0.5659824194805373. The full list is in the JSON and in `w/weights_M2b-oacc_s0.json`.
* **HS on X (descriptive):** the PREREG lists it beside; the reader JSON holds no HS block (the reader's code comment: it lives in the HSPEC package). Not read here.

---

## 6. Controls (every committed control read; the read was not stopped)

| control | committed condition | result |
|---|---|---|
| identity controls (`identity_controls`) | the scoring route reproduces the programme's own numbers | `all_pass` true: P vs the P10 file, max absolute difference 1.52587890625e-05 m; HX at weight 1 vs X's `pred`, max absolute difference 0.0 m; HX at weight 0 gives ADE6 2.4869921192297424 against the PREREG's 2.4870 (absolute difference 7.880770257706615e-06); X's banked ADE6 recomputed, max absolute difference 0.0; HX at weight 1 vs X's own eval block: 0 mismatches (excluding the zero-accel threshold diagnostic, which the reader drops by HSPEC I7) |
| w_acc rule (`wacc_record`) | `w_acc = 0.5 · L_traj / L_acc` on B_s0, banked cross-check | PASS, section 1 |
| one-lever control, B vs X (`check_x_run`) | X's built levers equal B's except `w_acc` | passes with the Amendment 2 fix (section 10); argv differs by `--w-acc` and `--out` only (checked here) |
| same windows and episodes | 2,399 / 141 | yes; 743 LEAD windows in the K = 30 block for every arm |
| checkpoint sha256, asserted before and after | unchanged | `M2b_s0` `601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8`; `M2b_s1` `6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6`; **X `9a33da47972265f0aac3197cf968994e34ace19ced30838402be020f028221dc` after training and after the evals (identical)** |
| launch gate | PASS token bound to commit, argv and data | `gate_token.json`: `gate` PASS; commit `04d5667114b82087c29611f0dd9b8043771c6e58`; `argv_sha` `353f25b44e364cae`; `w_acc` 42.9311; twelve key-file sha256s; `train_sha256 37f4f9ee…`; `excluded_sha256 f6a1dee0…`; v9 md5s `83e30bd3…` / `6b5c7f20…` |
| G-DVB / G-LIVE (`gate_M2b-oacc/selfcheck.json`) | declared = built; every loss term present; frozen-prior control exactly 0 | `declared_vs_built` PASS; `live.pass` true; no missing, unexpected or dead terms or parameters; `control_prior_grad_sum` 0.0; `head_o_grad_step0` 738.9362783432007; `n_trainable` 252 |
| the new term reaches O (`acc_probe`) | non-zero contribution and a non-zero head gradient, with controls reading exactly 0 | PASS: `w_acc_declared` = `w_probe` = 42.9311; `acc` 0.006943174637854099; `contribution` 0.2980785369873047 (`w_x_acc` 0.29807812469517814); `head_o_grad_diff` 210.35409545898438; `read_o_grad_diff` 0.0; `control_repeat_total_diff`, `control_repeat_grad_diff`, `control_prior_grad_diff`, `control_wm_grad_diff` all 0.0 |
| built levers | `w_acc` 42.9311 in the run record | `summary.json` `levers.w_acc` 42.9311; `levers.code_mix` "whole", `o_interface` "full" (B's values) |

**Items I could not reconcile (disclosed, not resolved):**

1. **The gate token and the gate selfcheck in this folder are re-issues.** File creation times: `gate_M2b-oacc/` and `gate_token.json` 2026-10-26 06:13:54 / 06:13:55; `ckpt_last.pt` 10:19:22. Their modification times are 11:53:55 / 11:53:57 / 11:55:39, i.e. rewritten by the later chain run, whose log in this folder starts at 11:44. The token on disk binds commit `04d5667` (committed 11:44:30), which is later than the training start. The re-issued `selfcheck.json` carries four keys the training-time `runs/M2b_s0_oacc/selfcheck.json` lacks (`levers.ego_perturb`, `levers.init_ckpt`, `levers.lr_schedule`, `ego_perturb_probe`); the tip's trainer carries the default-off O-RECOVER flags (commit `9871325`, "ego-state perturbation + --init-ckpt in the v8F-mini trainer"), and the training-time selfcheck has no such keys. Everything else in the two selfchecks is equal except `argv_sha` (gate `353f25b44e364cae`, run `29ff911937e56dec`; no gate argv file is in the folder to say why). X's argv carries none of the new flags. The training-time token itself is not in this folder.
2. **The chain log in this folder is the later chain's.** Its `t_M2b_s0_oacc` entry runs 11:53 to 11:53:59 (`ZZJOB-END … rc=0`), under a minute: it cannot be the 14,442 s training. That is consistent with Amendment 2's statement that X "had trained, been evaluated and been re-fitted on box 2" before the refused read, and with the run files' creation time of 10:19. I have not checked how the chain skips a finished run. The training wall-clock above comes from `summary.json`.
3. **Pre-run proofs (INHERITED from committed artifacts; not re-run here):** `raw/2026-10-25-oacc/mutate_oacc.out` ends "154/154 RED  RED BY EXCEPTION ONLY (no test fails by assertion): [4 mutants]"; run 4 of the exception hardening reads "1/1 RED"; `suites.out` lists `test_v8f_oacc.py` at 132 passed. Amendment 2's evidence on the tip (`raw/2026-10-26-oacc-a2/out/`): `real_records_check.out` (the reader as run refuses on the real records, naming `residual_zero_init` and `w`; the fixed reader returns no defect), `pytest_oacc_ego.out` (211 passed, 1 skipped), `mutate_oacc_check.out` (309 mutants, each anchor found once). The 21-mutant run on the new weight check is NOT on the tip (its `NOTES.md` says it was still running at landing): it is `E:/Projects/TanitAD-artifacts/oaccfix/out/mutate_oacc_sel.out` (modified 2026-10-26 11:59), which ends `21/21 RED`, including `#162 loss weights may differ from B's (Amendment 2)`. That file should be banked beside this result.

---

## 7. What it means (inside the PREREG's committed outcome)

1. **The acceleration term works on the metric it targets.** At seed 0, accel MAE fell from B's 1.2058 to X's 0.6395 m/s² (−0.5663, separated), and X is below both B seeds (1.2058, 0.9313). By the committed bar, PRIMARY-ACC is met.
2. **The position bar was not met.** ADE6 X − B +0.0451 [−0.0322, +0.1262]: the upper bound is above F/2. The committed name for ACC-met and KEEP-failed is COSTS POSITION, and the committed handling is "the trade-off is reported; the weight rule is not re-tuned post hoc".
3. **The deployed form HX does not dominate the floor.** HX − P10 is separated-better on ADE2, ADE6 and accel MAE and fails on curvature MAE (+0.0084, separated-worse). One row decides it.

**Post hoc, labelled, committing nothing (descriptive reads of the printed numbers):**

* X is separated-better than B on ADE2, speed MAE, heading MAE, yaw-rate MAE and the K = 30 min-TTC, and separated-worse on none of the paired per-window longitudinal or lateral cells in section 5. Differences to note that are not in the lever's favour: HX has more LEAD-VIOL windows than B and X (section 5.1), the consistency point rates of X and HX sit above B's (section 5.5, unpaired), and X's lead-progress ratio is separated-lower than B's (0.9007 against 0.9243).
* The curvature gap that fails DOMINANCE is against the floor P10, not against B: HX is separated-better than B on curvature MAE (−0.0312) and still above P10 (+0.0084).
* These cells come from one training seed. Section 8 gives the caution.

---

## 8. What these numbers do not say

* **Training variance.** X is one seed. The paired bootstrap resamples episodes with the models held fixed. Per `H-ESTIM-SEED-1`, a separated interval from a one-seed arm is necessary, not sufficient, for a claim that a LEVER moved a metric. The only run-to-run information is B's two seeds (ADE6 F = 0.12074; accel MAE 1.2058 and 0.9313). No replicate of X exists, and seed 1 was not trained because PRIMARY-KEEP failed. So every separated cell in section 5 is an observation at one seed, not a lever effect.
* **Inference variance.** Not applicable (deterministic planner, per the reader).
* **Tier.** Open-loop scoring of the logged future. Nothing here says the model drives.
* **Longitudinal numbers are UNVERIFIED against a hold-v0 copy** (anti-echo undischarged on X, HX and B).
* **Tactical and strategic have no paired X − B interval** for the heads, goal-setting and heading cells (sections 5.3 and 5.4); the cells there are descriptive.
* **PRIMARY-KEEP's failure is an interval-width failure at the point level of +0.0451.** It is reported as a failure at its committed bar. It is not evidence that X costs position and not evidence that it does not.
* **This is the seed-0 stage only.** The COSTS POSITION branch is fully decided by seed 0.

---

## 9. The next lever (RULE ZERO: a failed bar leaves the next arm behind)

* **Committed next, verbatim:** *"the trade-off is reported; the weight rule is not re-tuned post hoc."* That is done by this file. No second OACC arm exists in the PREREG, and it commits: "No weight rule, mask, bar or seed order changes after any OACC number is seen."
* **What this read leaves for the Master Mind (a description, not a decision):** (a) the lever moved accel MAE a long way at the committed weight while the position bar's interval was too wide to clear; (b) the single DOMINANCE failure is curvature MAE against P10, the family the floor is strongest on; (c) HX adds LEAD-VIOL and consistency cost against B. Any arm built from these needs its own PREREG with both outcomes committed before it runs; the weight rule may not be re-tuned on this read.
* **What blocks anything beyond the committed next:** (1) no PREREG exists for a second arm; (2) box 2 has needed the PI's go for each arm (as `RESULT_V8F_MIXO.md` records); (3) a replicate of X (seed 1, or a re-seeded s0) is the only way to read the ACC effect against the rig's run-to-run floor, and the committed chain does not run seed 1 after a KEEP failure. The queue already holds `PREREG_V8F_ORECOVER` (PI go in chat, 2026-10-26: "queue O-RECOVER on box 2 after OACC and MIXO"), which does not depend on this outcome.
* **Open work items from this read (not bars):** the anti-echo condition (no hold-v0 copy); paired tactical and strategic intervals for the OACC pair (`v8f_paired_families.py` on `eval_M2b-oacc_s0.npz` against `eval_M2b_s0.npz`); a paired consistency read for X − B and HX − X; the `raw.tier` label reconciliation; the vacuous `HX_equals_X_by_construction` strategic flag (reader fix: compare the `families.strategic` blocks, or drop the flag); the gate-token provenance items in section 6.

---

## 10. Amendment 2 (`A2-2026-10-26`): post-data, control only

* **When, and what had been seen** (PREREG text): written 2026-10-26 ~10:40. X_s0 had trained, been evaluated and been re-fitted on box 2. The read of seed 0 was then REFUSED by the B-vs-X built-lever control in `check_x_run` (message in full: `built levers ['residual_zero_init', 'w'] differ from B_s0's`, for `summary.json` and `selfcheck.json`). The reader refuses before it computes any row and wrote no read file. The PREREG states that no outcome row, family row or eval number of X had been read by anyone when the amendment was written.
* **The defect:** the control compared two trainer record formats, not levers. `residual_zero_init` is a readout (True while `head_o.2` is all zero): B's older trainer writes it after training (False) in `summary.json` and before training (True) in `selfcheck.json`, the current trainer writes the before-training value in both; X's head trained (`head_o.2.weight` L1 norm 164.4 against B's 137.4). The current trainer records `w_tac` and `w_cons` inside `w`, B's older record lists only `w_pen`, `w_pen_self`, `w_wm`; B's own checkpoint cfg reads w_tac 1.0 and w_cons 0.5, and so does X's.
* **The fix** (in `code/v8f_oacc_read.py`): `residual_zero_init` joins `lever_skip`; `w` is compared key by key, a weight B's record lacks must equal B's BUILT config (`W_B_BUILT = {w_tac: 1.0, w_cons: 0.5}`), any other difference is refused by name. The control stays strict.
* **Checked here** (the run records, not the PREREG's account): `runs/M2b_s0_oacc/summary.json` `levers.residual_zero_init` true and `levers.w` = {w_pen 0.5, w_pen_self 0.25, w_wm 0.1, w_tac 1.0, w_cons 0.5}; `runs/M2b_s0/summary.json` `levers.residual_zero_init` false and `levers.w` = {w_pen 0.5, w_pen_self 0.25, w_wm 0.1}. The weights in the two `w` blocks that exist in both are equal.
* **Nothing else moved:** the rule's `w_acc`, the rows, the bars, the outcome table, the seed-1 condition and every committed next step. This document applies them as written.
* **Disclosure:** this is a post-data amendment. It is control-only by its text, and the read file is the one produced after it. The root-cause class the PREREG records: a cross-version control (an older banked arm compared field by field with a newer trainer's record).

---

## 11. Provenance and manifest (nothing staged, committed or pushed; box 2's folder was read only)

**JSON key paths** (all in `C:/Users/Admin/box2_ship/logs/v8f_oacc/oacc_read_s0.json`, sha256 `a21303c2f623d02df24a90c4cbee1fdd7c01b0d387a3e669732d1f0d1feb7fb5`, 255,787 B):

| content | key path |
|---|---|
| outcome, next, seed state, reading line | `OUTCOME`, `next`, `s1_required`, `seed_states.s0`, `reading` |
| tier, estimator, strategic status | `tier`, `estimator`, `strategic_family` |
| F, KEEP bar, checkpoint sha256 | `F`, `keep_bar`, `keep_bar_printed`, `checkpoint_sha256` |
| the weight rule | `w_acc`, `wacc_record.*` |
| PRIMARY-ACC, PRIMARY-KEEP | `per_seed.s0.PRIMARY_ACC`, `per_seed.s0.PRIMARY_KEEP` |
| DOMINANCE rows | `per_seed.s0.DOMINANCE.rows.<row>` (`pass`, `all_better_ok`, `all_not_worse_ok`) |
| identity controls | `per_seed.s0.identity_controls` |
| ADE / FDE at each horizon | `per_seed.s0.beside.ADE.<ADE2/ADE4/ADE6/FDE6>.<X_minus_B/HX_minus_B/HX_minus_P10>` |
| paired longitudinal and lateral rows, tactical plan-derived agreement | `per_seed.s0.beside.families.longitudinal_lateral_paired_per_window.<X_minus_B/HX_minus_B/HX_minus_X>.fw_*` |
| levels (longitudinal, lateral, trajectory, tactical-from-trajectory, strategic) | `per_seed.s0.beside.families.<X_eval_block/HX_eval_block/B_banked_block>.{families,traj}` |
| tactical heads | `per_seed.s0.beside.families.tactical.<B/X/HX>`, `.HX_equals_X_by_construction` |
| strategic levels, paired position error, flags | `per_seed.s0.beside.families.strategic.{B,X,HX,X_minus_B_ck_pos_err,identical_to_B,identical_by_construction,HX_equals_X_by_construction}` |
| distance keeping K = 30 | `per_seed.s0.beside.families.distance_keeping_k30.{horizon,lead_block,join,arms,pairs}` |
| consistency | `per_seed.s0.beside.consistency_a.{rate,HX_minus_P10_any}` |
| HX weights | `per_seed.s0.beside.weights_HX` |

**Files read** (all under `C:/Users/Admin/box2_ship/logs/v8f_oacc/` unless stated): `check_inputs.json`, `ckpt_sha256.json`, `gate_token.json`, `gate_M2b-oacc/selfcheck.json`, `w_acc.json`, `w/weights_M2b-oacc_s0.json`, `eval/eval_M2b-oacc_s0.{json,npz,log}`, `eval/eval_M2b_s0.{json,npz}`, `eval/eval_M2b_s1.{json,npz}`, `runs/M2b_s0_oacc/{argv,selfcheck,summary}.json`, `runs/M2b_s0/{argv,selfcheck,summary}.json`, `telemetry_v8foacc.csv`, and `C:/Users/Admin/box2_ship/logs/chain_v8f_oacc.out` (sha256 `7ff60ca164ff1a3b1faf27c9c3c7484b4034bbb5cd43f072f25ec3f3c78a2c00`).
The X checkpoint is `runs/M2b_s0_oacc/ckpt_last.pt` (86,969,301 B); on box 2 `D:\tanitad\runs\v8f_mini_full\M2b_s0_oacc\ckpt_last.pt`. PREREG: `Project Steering/PREREG_V8F_OACC.md` at tip `04d5667` (repo mirror `C:/Users/Admin/tanitad-push`). Reader: `…/2026-10-17-v8f-mini/code/v8f_oacc_read.py`; chain: `code/box2_ops/chain_v8f_oacc.sh`.
**Checks made for this write-up beyond reading the JSON:** the argv diff (section 1), the `ck_pos_err` NaN count (section 5.4), the `strategic` block equality (section 5.4), the creation and modification times of the gate and run files (section 6), and a scan that every `point [low, high]` triple in this file occurs together in one JSON object (output beside this file: `verify_triples.out`).
