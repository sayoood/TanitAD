# FAMILIES_DAG: the pre-registered 'beside' rows of the DAgger arm (DAG vs DAG-0), four metric families at both tiers

*Draft for the Master Mind. Nothing was staged, committed or pushed by it. Zero training, zero new rollouts: every number below is a re-reading of banked artifacts with the programme's own instruments.*

* **Pre-registration:** `Project Steering/PREREG_V8F_DAGGER.md` section 5, *Beside (never decisive)*. The decisive read is `E:/Projects/TanitAD-artifacts/v8f-dagger-t1/read/DAG_READ.json` (OUTCOME **ROBUST**, all 176 of its controls PASS: True); this document decides nothing and changes no outcome. DAG ran at p = 0.1 (`--dagger-p 0.1`, argv.json).
* **Evidence class:** MEASURED (ours). Artifact paths are in the tables' `source` keys (`DAG_FAMILIES_ROWS.json`) and in section 9.
* **Estimator of every interval:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, over 141 episodes, 2,399 windows (eval-141), A minus B per seed on the same windows. It answers only *would another draw of EPISODES say this?* The training-seed question is answered only by the two seeds (H-ESTIM-SEED-1); the planner is deterministic (PREREG_V8F_PGT), so no inference-seed question arises. A separated interval from one training seed is necessary, not sufficient.
* **Tiers.** T0 (teacher-forced own-code eval-141; box 2's standard eval, ages cycle). T1-kin (self-action OPEN loop: plain causal bucket, nav dead-reckoned, scene logged, replan every 10 steps). T1-kin is a **self-action OPEN loop**; it is never 'closed loop' and nothing here is a driving claim. The eval JSON stamps T0 as `T0-open-loop-own-held-outputs (trunk-probe grid)`; the paired-families tool stamps its output `T1-class open loop (own held outputs; no recorded future action)`. These are three labels for one open-loop scoring and I use the PREREG's names (the same unreconciled labelling as RESULT_V8F_ORECOVER section 3).
* **STRATEGIC is NOT paused for v8F** (PI ruling 2026-10-17, R73): it is reported in section 2.4 and 4.
* **Recipe:** `orc_families.py` of O-RECOVER (raw/2026-10-26-orecover-result/) re-parametrised (ORC -> DAG, ORC-0 -> DAG-0), run on the DAG chain's own tool tree `E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a` (the instruments `v8f_beta_eval.py`, `ci.py`, `lead_metrics.py`, `four_families.py`, `v8f_paired_families.py`, `v8f_dk_read.py` are byte-equal to O-RECOVER's 07dcb513 tree modulo CRLF, and to the mirror tip 7d29172b; hashes in section 9).
* **Column key.** A is the first-named arm (for `DAG-DAG0`: DAG; for `DAG-ORC`: DAG; for `DAG-M2b`: DAG). 'A better' / 'A WORSE' is printed only where the metric has a direction (error: lower is better; agreement, F1: higher; LEAD-VIOL share: lower). Headway, time gap, TTC and lead-progress carry no direction label from the instrument (more distance kept is not automatically better).

## 0. Headline, per family (computed from the tables)

Separated rows where DAG is WORSE than the comparator (a direction-labelled metric, interval excluding 0, point on the worse side):

* **DAG-DAG0:** speed MAE [LONGITUDINAL] T0 s1 +0.0204 [+0.0073, +0.0335]; accel MAE [LONGITUDINAL] T0 s1 +0.0338 [+0.0190, +0.0479]; along-track MAE [LONGITUDINAL] T0 s1 +0.0461 [+0.0191, +0.0745]; lateral decision agreement (plan-derived) [TACTICAL] T0 s1 -0.0108 [-0.0200, -0.0025]; absolute time-gap error vs the human [LONGITUDINAL (distance keeping, K=30, lead windows)] T1-kin s0 +0.0835 [+0.0147, +0.1701]; absolute min-TTC error vs the human [LONGITUDINAL (distance keeping, K=30, lead windows)] T1-kin s0 +0.8675 [+0.0571, +1.5978]; LEAD-VIOL, at risk (violating share) [LONGITUDINAL (distance keeping, K=30, lead windows)] T1-kin s0 +0.0420 [+0.0131, +0.0794]; LEAD-VIOL, all lead windows [LONGITUDINAL (distance keeping, K=30, lead windows)] T1-kin s0 +0.0336 [+0.0088, +0.0642]; accel MAE [LONGITUDINAL] T1-kin s1 +0.4490 [+0.1344, +0.8103]; curvature MAE [LATERAL] T1-kin s1 +0.0168 [+0.0052, +0.0303]; absolute min-TTC error vs the human [LONGITUDINAL (distance keeping, K=30, lead windows)] T1-kin s1 +0.5971 [+0.0021, +1.2395]; LEAD-VIOL, at risk (violating share) [LONGITUDINAL (distance keeping, K=30, lead windows)] T1-kin s1 +0.0303 [+0.0118, +0.0551]; LEAD-VIOL, all lead windows [LONGITUDINAL (distance keeping, K=30, lead windows)] T1-kin s1 +0.0242 [+0.0088, +0.0433]; ADE6, 6 s [ADE6 (reference row)] T0 s1 +0.0483 [+0.0178, +0.0807]; lon head accuracy (window argmax vs v9 label) [TACTICAL] T0 s0 -0.0170 [-0.0326, -0.0009]; lateral decision accuracy (plan-derived, tool's own) [TACTICAL] T0 s1 -0.0108 [-0.0200, -0.0025]; goal-point error (FDE6) [TACTICAL] T0 s1 +0.1011 [+0.0071, +0.2008]
* **DAG-M2b:** LEAD-VIOL, at risk (violating share) [LONGITUDINAL (distance keeping, K=30, lead windows)] T0 s0 +0.0218 [+0.0093, +0.0367]; LEAD-VIOL, all lead windows [LONGITUDINAL (distance keeping, K=30, lead windows)] T0 s0 +0.0175 [+0.0070, +0.0292]; speed MAE [LONGITUDINAL] T0 s1 +0.0638 [+0.0273, +0.0985]; accel MAE [LONGITUDINAL] T0 s1 +0.1267 [+0.0985, +0.1541]; along-track MAE [LONGITUDINAL] T0 s1 +0.0780 [+0.0077, +0.1464]; curvature MAE [LATERAL] T0 s1 +0.0097 [+0.0012, +0.0193]; yaw-rate MAE [LATERAL] T0 s1 +0.7814 [+0.4201, +1.1707]; longitudinal decision agreement (plan-derived) [TACTICAL] T0 s1 -0.0313 [-0.0600, -0.0029]; S checkpoint position error at arc 80 m [STRATEGIC] T0 s0 +0.4911 [+0.0093, +1.0719]; longitudinal decision accuracy (plan-derived, tool's own) [TACTICAL] T0 s1 -0.0313 [-0.0600, -0.0029]
* **DAG-ORC:** none

Context for the list: the `ADE6, 6 s` T0 s1 row (+0.0483 m) is the decisive read's R-T0 guard, which that read PASSED (bar: point <= +0.0604 m, F/2); it appears here only because its interval excludes 0. No row in the DAG - DAG-0 list is a guard of the decisive read other than that one, and none changes the OUTCOME (this document decides nothing). The T1-kin LEAD-VIOL and absolute-TTC rows are the ones that cut against the T1-kin gains: the lead follows its LOGGED future and does not react to the ego (an open-loop property of the rig, as RESULT_V8F_ORECOVER section 3 states for these runs), and the anti-echo block (1.1) does not admit either arm's longitudinal numbers as a capability claim.

| contrast | family | cells (tier x seed x metric) | separated | separated, DAG better | separated, DAG WORSE | separated, no direction label |
|---|---|---|---|---|---|---|
| DAG-DAG0 | LONGITUDINAL (incl. distance keeping) | 48 | 22 | 5 | 11 | 6 |
| DAG-DAG0 | LATERAL | 16 | 7 | 6 | 1 | 0 |
| DAG-DAG0 | TACTICAL | 40 | 18 | 14 | 4 | 0 |
| DAG-DAG0 | STRATEGIC | 24 | 12 | 12 | 0 | 0 |
| DAG-M2b | LONGITUDINAL (incl. distance keeping) | 24 | 11 | 3 | 5 | 3 |
| DAG-M2b | LATERAL | 8 | 5 | 3 | 2 | 0 |
| DAG-M2b | TACTICAL | 26 | 3 | 1 | 2 | 0 |
| DAG-M2b | STRATEGIC | 12 | 1 | 0 | 1 | 0 |

## 1. Controls (each must hold; a failed control would make this document unusable)

| control | what it asserts | status |
|---|---|---|
| C1 identity, families | every paired family row run on A against A reads exactly +0 [0, 0] (T0 and T1-kin DAG-DAG0, T0 DAG-M2b, both seeds, 9 rows each) | PASS |
| C2 identity, distance keeping | `read_dk`'s own identity control (headway, time gap, TTC, LEAD-VIOL) reads +0 [0, 0] in all 6 blocks | PASS |
| C3 reproduction of the reader's R-T0 / R-ADE | this driver's ADE6 DAG - DAG0 equals `DAG_READ.json` rows R-T0 (T0) and R-ADE (T1-kin), delta, lo, hi, separated, both seeds: s0 T0 +0.0019, T1-kin -0.5435; s1 T0 +0.0483, T1-kin -0.4046 | PASS |
| C4 T1-kin grid | the dump's windows, episode ids, `2*ws` frames and `v0` equal the T0 grid's for DAG and DAG-0, both seeds; dump GT vs grid GT max difference 7.6e-06 m (float32 round trip) | PASS |
| C5 cross-instrument | the plan-derived lateral / longitudinal decision agreement from `v8f_paired_families.py` equals the `family_windows` rows to delta, lo, hi, separated (4 of 4 for DAG-DAG0, 4 of 4 for DAG-M2b) | PASS |
| C6 the paired-families CLI's own controls | `controls_ok` (A-vs-A identity on 45 cells; each arm's point values reproduce its own eval JSON, 119 values) true for all 4 runs; 2,399 windows, 141 episodes, ages cycle | PASS |
| C7 reproduction of the reader's R-TLAT / R-TLON | head macro-F1 DAG - DAG-0, ticks>=1, matched cells: delta, lo, hi, separated, matched-pair count and both F1 levels equal `DAG_READ.json` (4 rows) | PASS |
| C8 identity, head F1 / class F1 / S-POS | (a, a) reads exactly +0 [0, 0] on the head-F1 rows (DAG-DAG0 and DAG-ORC), the per-class F1 routine, and the paired S-POS rows | PASS |
| C9 class-F1 bootstrap vs the standard estimator | the routine used for per-class F1 reproduces `taniteval.ci` on macro-F1 exactly (delta, lo, hi, separated) for both heads and seeds | PASS |
| C10 per-class levels | the mean of the per-class F1 levels equals the run's macro-F1 (the macro-F1 is the nan-mean of them) | PASS |
| C11 S-POS vs CTRV | the per-run S - CTRV rows (ticks>=1) recomputed here equal `DAG_READ.json` beside.per_run_t1heads for DAG and DAG-0 (delta, lo, hi, separated; 12 rows) | PASS |
| C12 executed-track rebuild | the rebuilt executed track equals the dumped `cl` (max 0.0 m) for DAG, DAG-0 and ORC, both seeds (the T1-HEADS X1 check) | PASS |
| C13 M2b comparator | the banked M2b eval's ADE6 mean equals MODEL_REGISTRY section 17 at 4 dp: s0 2.1753 vs 2.1753, s1 2.0546 vs 2.0546 | PASS |
| C14 inputs | lead block sha256 `7a879baa2b6f4c14...` is the registered one; grid `659c3b65e364...`, kin `be57e283c3f0...`, join `f0ea722f59c1...` | PASS |

**All controls: PASS.**

*Not a control, disclosed:* `DAG_READ.json` itself was produced by `dag_read.py` (sha256 `1cf56c7c30195ac4...`) from the same artifacts; C3 and C7 tie this document to it.

### 1.1 Anti-echo caveat on every LONGITUDINAL number (binding for how to read section 2.1)

| run | tier | anti_echo.status | longitudinal_claim_admissible | flagged | summary | source |
|---|---|---|---|---|---|---|
| DAG s0 | T1-kin | OK | False | True | holdv0=NOT_SEPARATED · copy_detector=CLEAN (echo_index 0.0000 vs GT 0.0992) · shuffle=UNAVAILABLE | `t1v8f_DAG_s0_m10_causal_heads.json` |
| DAG-0 s0 | T1-kin | OK | False | True | holdv0=LOSES_TO_HOLDV0 · copy_detector=CLEAN (echo_index 0.0000 vs GT 0.0992) · shuffle=UNAVAILABLE | `t1v8f_DAG0_s0_m10_causal_heads.json` |
| DAG s1 | T1-kin | OK | False | True | holdv0=LOSES_TO_HOLDV0 · copy_detector=CLEAN (echo_index 0.0000 vs GT 0.0992) · shuffle=UNAVAILABLE | `t1v8f_DAG_s1_m10_causal_heads.json` |
| DAG-0 s1 | T1-kin | OK | False | True | holdv0=LOSES_TO_HOLDV0 · copy_detector=CLEAN (echo_index 0.0000 vs GT 0.0992) · shuffle=UNAVAILABLE | `t1v8f_DAG0_s1_m10_causal_heads.json` |
| DAG s0 | T0 | UNAVAILABLE | n/a | n/a | no ego speed at t0 in the window dict (condition undischarged) | `v8f-dagger/M2b_s0_dag/eval_M2b-dag_s0.json` |
| DAG-0 s0 | T0 | UNAVAILABLE | n/a | n/a | no ego speed at t0 in the window dict (condition undischarged) | `v8f-dagger/M2b_s0_dag0/eval_M2b-dag0_s0.json` |
| DAG s1 | T0 | UNAVAILABLE | n/a | n/a | no ego speed at t0 in the window dict (condition undischarged) | `v8f-dagger/M2b_s1_dag/eval_M2b-dag_s1.json` |
| DAG-0 s1 | T0 | UNAVAILABLE | n/a | n/a | no ego speed at t0 in the window dict (condition undischarged) | `v8f-dagger/M2b_s1_dag0/eval_M2b-dag0_s1.json` |

The M2b banked eval (`dagfam/ref/m2b/eval_M2b_s0.json`, `_s1.json`) carries the T0 anti-echo status UNAVAILABLE / UNAVAILABLE. A LONGITUDINAL contrast here describes what the lever changed on these windows; it is not a statement that either arm sets the right speed.

## 2. DAG - DAG-0 (the decisive pair), per family, both tiers, both seeds

A = DAG, B = DAG-0 (same init M2b, steps, data, LR path and seed; the lever is the route-paired own-state replacement at p = 0.1). 'level' columns are each arm's mean on the paired windows.

### 2.0 ADE6 reference (the guard rows R-T0 and R-ADE of the decisive read, reproduced here: control C3)

| row | tier | seed | DAG level | DAG-0 level | DAG - DAG-0 [95 % interval] | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| R-T0: T0 own-code ADE6, m | T0 | s0 | 1.9736 | 1.9717 | +0.0019 [-0.0230, +0.0266] | not separated | 2,399 / 141 |
| R-T0: T0 own-code ADE6, m | T0 | s1 | 2.0984 | 2.0501 | +0.0483 [+0.0178, +0.0807] | separated, A WORSE | 2,399 / 141 |
| R-ADE: T1-kin plain ADE6, m | T1-kin | s0 | 2.2777 | 2.8212 | -0.5435 [-0.7564, -0.3690] | separated, A better | 2,399 / 141 |
| R-ADE: T1-kin plain ADE6, m | T1-kin | s1 | 2.4565 | 2.8611 | -0.4046 [-0.5496, -0.2782] | separated, A better | 2,399 / 141 |

### 2.1 LONGITUDINAL

| metric (A - B) | unit | tier | seed | A level | B level | A - B [95 % interval] | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|---|
| speed MAE | m/s | T0 | s0 | 0.9208 | 0.9307 | -0.0098 [-0.0243, +0.0041] | not separated | 2,399 / 141 |
| speed MAE | m/s | T0 | s1 | 0.9751 | 0.9546 | +0.0204 [+0.0073, +0.0335] | separated, A WORSE | 2,399 / 141 |
| speed MAE | m/s | T1-kin | s0 | 1.3115 | 1.7773 | -0.4658 [-0.6127, -0.3310] | separated, A better | 2,399 / 141 |
| speed MAE | m/s | T1-kin | s1 | 1.5860 | 1.8676 | -0.2816 [-0.3899, -0.1812] | separated, A better | 2,399 / 141 |
| accel MAE | m/s^2 | T0 | s0 | 0.9992 | 0.9929 | +0.0063 [-0.0104, +0.0225] | not separated | 2,399 / 141 |
| accel MAE | m/s^2 | T0 | s1 | 1.0581 | 1.0243 | +0.0338 [+0.0190, +0.0479] | separated, A WORSE | 2,399 / 141 |
| accel MAE | m/s^2 | T1-kin | s0 | 2.8970 | 2.9163 | -0.0193 [-0.2929, +0.2327] | not separated | 2,399 / 141 |
| accel MAE | m/s^2 | T1-kin | s1 | 3.5335 | 3.0845 | +0.4490 [+0.1344, +0.8103] | separated, A WORSE | 2,399 / 141 |
| along-track MAE | m | T0 | s0 | 1.5813 | 1.5815 | -0.0002 [-0.0242, +0.0231] | not separated | 2,399 / 141 |
| along-track MAE | m | T0 | s1 | 1.6861 | 1.6400 | +0.0461 [+0.0191, +0.0745] | separated, A WORSE | 2,399 / 141 |
| along-track MAE | m | T1-kin | s0 | 1.8299 | 2.2316 | -0.4017 [-0.6146, -0.2297] | separated, A better | 2,399 / 141 |
| along-track MAE | m | T1-kin | s1 | 2.0010 | 2.2500 | -0.2490 [-0.3763, -0.1311] | separated, A better | 2,399 / 141 |
| min headway | m | T0 | s0 | 25.3657 | 25.2804 | -0.1691 [-0.4240, +0.0282] | not separated | 724 / 73 |
| min headway | m | T0 | s1 | 25.9014 | 25.4507 | +0.1922 [-0.1112, +0.5493] | not separated | 723 / 73 |
| min headway | m | T1-kin | s0 | 24.7412 | 25.8763 | -1.0222 [-1.4505, -0.5753] | separated (no direction label) | 710 / 73 |
| min headway | m | T1-kin | s1 | 25.9321 | 26.4430 | -0.5627 [-0.9631, -0.2178] | separated (no direction label) | 713 / 74 |
| min time gap | s | T0 | s0 | 3.9403 | 3.9356 | +0.0034 [-0.0271, +0.0356] | not separated | 664 / 73 |
| min time gap | s | T0 | s1 | 3.9736 | 3.9718 | -0.0047 [-0.0406, +0.0299] | not separated | 661 / 73 |
| min time gap | s | T1-kin | s0 | 3.8318 | 3.8810 | -0.1612 [-0.3006, -0.0597] | separated (no direction label) | 660 / 73 |
| min time gap | s | T1-kin | s1 | 4.0294 | 4.1232 | -0.0939 [-0.1470, -0.0458] | separated (no direction label) | 665 / 74 |
| min TTC (censored at a 30 s cap) | s | T0 | s0 | 19.6959 | 19.6521 | -0.0415 [-0.3734, +0.3456] | not separated | 724 / 73 |
| min TTC (censored at a 30 s cap) | s | T0 | s1 | 19.7162 | 19.5921 | +0.0532 [-0.1888, +0.3172] | not separated | 723 / 73 |
| min TTC (censored at a 30 s cap) | s | T1-kin | s0 | 17.6347 | 19.7000 | -2.2480 [-2.9945, -1.5545] | separated (no direction label) | 710 / 73 |
| min TTC (censored at a 30 s cap) | s | T1-kin | s1 | 18.4832 | 19.5842 | -1.2406 [-1.9889, -0.5407] | separated (no direction label) | 713 / 74 |
| absolute headway error vs the human | m | T0 | s0 | - | - | -0.1179 [-0.3547, +0.0648] | not separated | 717 / 72 |
| absolute headway error vs the human | m | T0 | s1 | - | - | -0.0580 [-0.3540, +0.2330] | not separated | 717 / 72 |
| absolute headway error vs the human | m | T1-kin | s0 | - | - | +0.2280 [-0.0715, +0.5059] | not separated | 703 / 72 |
| absolute headway error vs the human | m | T1-kin | s1 | - | - | +0.0722 [-0.1135, +0.2534] | not separated | 703 / 72 |
| absolute time-gap error vs the human | s | T0 | s0 | - | - | +0.0053 [-0.0175, +0.0353] | not separated | 657 / 72 |
| absolute time-gap error vs the human | s | T0 | s1 | - | - | -0.0301 [-0.0587, -0.0065] | separated, A better | 655 / 72 |
| absolute time-gap error vs the human | s | T1-kin | s0 | - | - | +0.0835 [+0.0147, +0.1701] | separated, A WORSE | 653 / 72 |
| absolute time-gap error vs the human | s | T1-kin | s1 | - | - | +0.0126 [-0.0153, +0.0404] | not separated | 655 / 72 |
| absolute min-TTC error vs the human | s | T0 | s0 | - | - | -0.0328 [-0.4125, +0.3151] | not separated | 717 / 72 |
| absolute min-TTC error vs the human | s | T0 | s1 | - | - | -0.0905 [-0.3596, +0.1484] | not separated | 717 / 72 |
| absolute min-TTC error vs the human | s | T1-kin | s0 | - | - | +0.8675 [+0.0571, +1.5978] | separated, A WORSE | 703 / 72 |
| absolute min-TTC error vs the human | s | T1-kin | s1 | - | - | +0.5971 [+0.0021, +1.2395] | separated, A WORSE | 703 / 72 |
| LEAD-VIOL, at risk (violating share) | share | T0 | s0 | 0.0571 | 0.0605 | -0.0034 [-0.0115, +0.0049] | not separated | 595 / 70 (violations A/B among 595: 34/36) |
| LEAD-VIOL, at risk (violating share) | share | T0 | s1 | 0.0454 | 0.0521 | -0.0067 [-0.0169, +0.0017] | not separated | 595 / 70 (violations A/B among 595: 27/31) |
| LEAD-VIOL, at risk (violating share) | share | T1-kin | s0 | 0.0824 | 0.0403 | +0.0420 [+0.0131, +0.0794] | separated, A WORSE | 595 / 70 (violations A/B among 595: 49/24) |
| LEAD-VIOL, at risk (violating share) | share | T1-kin | s1 | 0.0521 | 0.0218 | +0.0303 [+0.0118, +0.0551] | separated, A WORSE | 595 / 70 (violations A/B among 595: 31/13) |
| LEAD-VIOL, all lead windows | share | T0 | s0 | 0.0458 | 0.0485 | -0.0027 [-0.0092, +0.0037] | not separated | 743 / 77 (violations A/B among 743: 34/36) |
| LEAD-VIOL, all lead windows | share | T0 | s1 | 0.0363 | 0.0417 | -0.0054 [-0.0135, +0.0014] | not separated | 743 / 77 (violations A/B among 743: 27/31) |
| LEAD-VIOL, all lead windows | share | T1-kin | s0 | 0.0659 | 0.0323 | +0.0336 [+0.0088, +0.0642] | separated, A WORSE | 743 / 77 (violations A/B among 743: 49/24) |
| LEAD-VIOL, all lead windows | share | T1-kin | s1 | 0.0417 | 0.0175 | +0.0242 [+0.0088, +0.0433] | separated, A WORSE | 743 / 77 (violations A/B among 743: 31/13) |
| lead-progress ratio | ratio | T0 | s0 | 0.9985 | 0.9973 | +0.0012 [-0.0053, +0.0075] | not separated | 595 / 70 |
| lead-progress ratio | ratio | T0 | s1 | 0.9545 | 0.9649 | -0.0104 [-0.0202, +0.0001] | not separated | 595 / 70 |
| lead-progress ratio | ratio | T1-kin | s0 | 1.3000 | 1.4544 | -0.1544 [-0.5203, +0.0673] | not separated | 595 / 70 |
| lead-progress ratio | ratio | T1-kin | s1 | 1.3814 | 1.3367 | +0.0447 [-0.0596, +0.1313] | not separated | 595 / 70 |

### 2.2 LATERAL

| metric (A - B) | unit | tier | seed | A level | B level | A - B [95 % interval] | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|---|
| heading MAE | deg | T0 | s0 | 4.3289 | 4.5127 | -0.1837 [-0.4869, +0.0188] | not separated | 2,343 / 141 |
| heading MAE | deg | T0 | s1 | 4.4410 | 4.3679 | +0.0731 [-0.0758, +0.2564] | not separated | 2,344 / 141 |
| heading MAE | deg | T1-kin | s0 | 6.1932 | 9.7071 | -3.5139 [-4.7142, -2.4711] | separated, A better | 2,373 / 141 |
| heading MAE | deg | T1-kin | s1 | 7.9534 | 9.7180 | -1.7646 [-2.2837, -1.2020] | separated, A better | 2,373 / 141 |
| curvature MAE | 1/m | T0 | s0 | 0.0602 | 0.0632 | -0.0030 [-0.0075, +0.0007] | not separated | 2,327 / 141 |
| curvature MAE | 1/m | T0 | s1 | 0.0702 | 0.0689 | +0.0013 [-0.0025, +0.0049] | not separated | 2,329 / 141 |
| curvature MAE | 1/m | T1-kin | s0 | 0.1336 | 0.1407 | -0.0071 [-0.0180, +0.0041] | not separated | 2,365 / 141 |
| curvature MAE | 1/m | T1-kin | s1 | 0.1857 | 0.1689 | +0.0168 [+0.0052, +0.0303] | separated, A WORSE | 2,365 / 141 |
| yaw-rate MAE | deg/s | T0 | s0 | 7.0727 | 7.2606 | -0.1878 [-0.4050, -0.0182] | separated, A better | 2,327 / 141 |
| yaw-rate MAE | deg/s | T0 | s1 | 7.7126 | 7.6260 | +0.0867 [-0.0862, +0.2891] | not separated | 2,329 / 141 |
| yaw-rate MAE | deg/s | T1-kin | s0 | 19.8327 | 26.2940 | -6.4613 [-9.6959, -3.7079] | separated, A better | 2,365 / 141 |
| yaw-rate MAE | deg/s | T1-kin | s1 | 28.5328 | 30.2295 | -1.6967 [-4.3367, +0.3956] | not separated | 2,365 / 141 |
| cross-track MAE | m | T0 | s0 | 0.7700 | 0.7636 | +0.0063 [-0.0076, +0.0208] | not separated | 2,399 / 141 |
| cross-track MAE | m | T0 | s1 | 0.8233 | 0.8175 | +0.0058 [-0.0147, +0.0269] | not separated | 2,399 / 141 |
| cross-track MAE | m | T1-kin | s0 | 0.9402 | 1.2357 | -0.2955 [-0.3981, -0.2108] | separated, A better | 2,399 / 141 |
| cross-track MAE | m | T1-kin | s1 | 0.9927 | 1.2996 | -0.3069 [-0.4193, -0.2127] | separated, A better | 2,399 / 141 |

### 2.3 TACTICAL

Plan-derived decisions are scored on both tiers (`family_windows`). Head and goal F1 at T0 come from the unchanged CLI `v8f_paired_families.py`; at T1-kin the head F1 rows are the reader's R-TLAT / R-TLON (re-derived, control C7) and the per-class rows of section 5. The goal-point rows are T0 only.

| metric (A - B) | unit | tier | seed | A level | B level | A - B [95 % interval] | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|---|
| lateral decision agreement (plan-derived) | frac | T0 | s0 | 0.8508 | 0.8508 | +0.0000 [-0.0079, +0.0083] | not separated | 2,399 / 141 |
| lateral decision agreement (plan-derived) | frac | T0 | s1 | 0.8370 | 0.8479 | -0.0108 [-0.0200, -0.0025] | separated, A WORSE | 2,399 / 141 |
| lateral decision agreement (plan-derived) | frac | T1-kin | s0 | 0.7857 | 0.6857 | +0.1000 [+0.0704, +0.1330] | separated, A better | 2,399 / 141 |
| lateral decision agreement (plan-derived) | frac | T1-kin | s1 | 0.7782 | 0.5865 | +0.1917 [+0.1505, +0.2332] | separated, A better | 2,399 / 141 |
| longitudinal decision agreement (plan-derived) | frac | T0 | s0 | 0.5773 | 0.5711 | +0.0063 [-0.0046, +0.0179] | not separated | 2,399 / 141 |
| longitudinal decision agreement (plan-derived) | frac | T0 | s1 | 0.5607 | 0.5673 | -0.0067 [-0.0171, +0.0029] | not separated | 2,399 / 141 |
| longitudinal decision agreement (plan-derived) | frac | T1-kin | s0 | 0.5002 | 0.4460 | +0.0542 [+0.0242, +0.0846] | separated, A better | 2,399 / 141 |
| longitudinal decision agreement (plan-derived) | frac | T1-kin | s1 | 0.4856 | 0.4139 | +0.0717 [+0.0383, +0.1055] | separated, A better | 2,399 / 141 |
| lat head macro-F1 (window argmax vs v9 label) | - | T0 | s0 | 0.8347 | 0.8342 | +0.0005 [-0.0186, +0.0235] | not separated | 2,107 / 139 |
| lat head macro-F1 (window argmax vs v9 label) | - | T0 | s1 | 0.8351 | 0.8442 | -0.0091 [-0.0310, +0.0100] | not separated | 2,107 / 139 |
| lat head accuracy (window argmax vs v9 label) | - | T0 | s0 | 0.9530 | 0.9530 | +0.0000 [-0.0047, +0.0048] | not separated | 2,107 / 139 |
| lat head accuracy (window argmax vs v9 label) | - | T0 | s1 | 0.9535 | 0.9568 | -0.0033 [-0.0096, +0.0019] | not separated | 2,107 / 139 |
| lon head macro-F1 (window argmax vs v9 label) | - | T0 | s0 | 0.4944 | 0.5119 | -0.0175 [-0.0541, +0.0167] | not separated | 2,240 / 136 |
| lon head macro-F1 (window argmax vs v9 label) | - | T0 | s1 | 0.5212 | 0.5527 | -0.0315 [-0.0702, +0.0025] | not separated | 2,240 / 136 |
| lon head accuracy (window argmax vs v9 label) | - | T0 | s0 | 0.6232 | 0.6402 | -0.0170 [-0.0326, -0.0009] | separated, A WORSE | 2,240 / 136 |
| lon head accuracy (window argmax vs v9 label) | - | T0 | s1 | 0.6295 | 0.6384 | -0.0089 [-0.0246, +0.0059] | not separated | 2,240 / 136 |
| goal head macro-F1 (22 tokens) | - | T0 | s0 | 0.2472 | 0.2631 | -0.0159 [-0.0288, +0.0042] | not separated | 2,365 / 139 |
| goal head macro-F1 (22 tokens) | - | T0 | s1 | 0.2747 | 0.2750 | -0.0002 [-0.0486, +0.0211] | not separated | 2,365 / 139 |
| goal head micro token accuracy | - | T0 | s0 | 0.9565 | 0.9574 | -0.0009 [-0.0021, +0.0003] | not separated | 2,365 / 139 |
| goal head micro token accuracy | - | T0 | s1 | 0.9581 | 0.9586 | -0.0005 [-0.0018, +0.0008] | not separated | 2,365 / 139 |
| 5-way collapsed manoeuvre accuracy (plan-derived) | - | T0 | s0 | 0.5802 | 0.5736 | +0.0067 [-0.0058, +0.0196] | not separated | 2,399 / 141 |
| 5-way collapsed manoeuvre accuracy (plan-derived) | - | T0 | s1 | 0.5602 | 0.5652 | -0.0050 [-0.0154, +0.0050] | not separated | 2,399 / 141 |
| lateral decision accuracy (plan-derived, tool's own) | - | T0 | s0 | 0.8508 | 0.8508 | +0.0000 [-0.0079, +0.0083] | not separated | 2,399 / 141 |
| lateral decision accuracy (plan-derived, tool's own) | - | T0 | s1 | 0.8370 | 0.8479 | -0.0108 [-0.0200, -0.0025] | separated, A WORSE | 2,399 / 141 |
| longitudinal decision accuracy (plan-derived, tool's own) | - | T0 | s0 | 0.5773 | 0.5711 | +0.0063 [-0.0046, +0.0179] | not separated | 2,399 / 141 |
| longitudinal decision accuracy (plan-derived, tool's own) | - | T0 | s1 | 0.5607 | 0.5673 | -0.0067 [-0.0171, +0.0029] | not separated | 2,399 / 141 |
| goal-point error (FDE6) | m | T0 | s0 | 5.6500 | 5.7020 | -0.0520 [-0.1318, +0.0252] | not separated | 2,399 / 141 |
| goal-point error (FDE6) | m | T0 | s1 | 6.0567 | 5.9556 | +0.1011 [+0.0071, +0.2008] | separated, A WORSE | 2,399 / 141 |
| goal bearing MAE | deg | T0 | s0 | 3.6282 | 4.0859 | -0.4577 [-0.8901, -0.1414] | separated, A better | 2,367 / 141 |
| goal bearing MAE | deg | T0 | s1 | 3.9731 | 3.8303 | +0.1428 [-0.0376, +0.3917] | not separated | 2,367 / 141 |
| R-TLAT: lat macro-F1, ticks>=1, matched cells | - | T1-kin | s0 | 0.6029 | 0.4947 | +0.1082 [+0.0491, +0.1572] | separated, A better | 2,399 / 141 (matched cell pairs 3863) |
| R-TLAT: lat macro-F1, ticks>=1, matched cells | - | T1-kin | s1 | 0.6385 | 0.5201 | +0.1184 [+0.0755, +0.1585] | separated, A better | 2,399 / 141 (matched cell pairs 3849) |
| R-TLON: lon macro-F1, ticks>=1, matched cells | - | T1-kin | s0 | 0.3213 | 0.2659 | +0.0554 [+0.0300, +0.0847] | separated, A better | 2,399 / 141 (matched cell pairs 3770) |
| R-TLON: lon macro-F1, ticks>=1, matched cells | - | T1-kin | s1 | 0.3311 | 0.2505 | +0.0806 [+0.0560, +0.1043] | separated, A better | 2,399 / 141 (matched cell pairs 3758) |
| lat head F1 of TURN_L, ticks>=1, matched cells | - | T1-kin | s0 | 0.3660 | 0.2327 | +0.1333 [-0.0260, +0.2629] | not separated | 2,399 / 141 (support 92 matched cells in 11 episodes; draws with the class absent 0) |
| lat head F1 of TURN_L, ticks>=1, matched cells | - | T1-kin | s1 | 0.4294 | 0.2812 | +0.1482 [+0.0533, +0.2378] | separated, A better | 2,399 / 141 (support 90 matched cells in 11 episodes; draws with the class absent 0) |
| lat head F1 of TURN_R, ticks>=1, matched cells | - | T1-kin | s0 | 0.4891 | 0.3183 | +0.1708 [+0.1067, +0.2298] | separated, A better | 2,399 / 141 (support 292 matched cells in 27 episodes; draws with the class absent 0) |
| lat head F1 of TURN_R, ticks>=1, matched cells | - | T1-kin | s1 | 0.5292 | 0.3390 | +0.1901 [+0.1174, +0.2599] | separated, A better | 2,399 / 141 (support 295 matched cells in 26 episodes; draws with the class absent 0) |
| lat head F1 of LANE_KEEP, ticks>=1, matched cells | - | T1-kin | s0 | 0.9537 | 0.9332 | +0.0206 [+0.0119, +0.0300] | separated, A better | 2,399 / 141 (support 3479 matched cells in 139 episodes; draws with the class absent 0) |
| lat head F1 of LANE_KEEP, ticks>=1, matched cells | - | T1-kin | s1 | 0.9569 | 0.9400 | +0.0169 [+0.0091, +0.0262] | separated, A better | 2,399 / 141 (support 3464 matched cells in 139 episodes; draws with the class absent 0) |

### 2.4 STRATEGIC (NOT paused)

T0: S's checkpoint position / heading error at 30 / 50 / 80 m against the v9 route checkpoints (`v8f_paired_families.py`). T1-kin: the paired rows are built from the T1-HEADS reader's own S error arrays (control C11 ties them to the reader); the within-run S - CTRV (constant-turn-rate-and-velocity) contrast the reader carries is in section 2.5.

| metric (A - B) | unit | tier | seed | A level | B level | A - B [95 % interval] | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|---|
| S checkpoint position error at arc 30 m | m | T0 | s0 | 7.0048 | 7.0131 | -0.0083 [-0.0354, +0.0187] | not separated | 2,322 / 137 |
| S checkpoint position error at arc 30 m | m | T0 | s1 | 6.9723 | 6.9713 | +0.0010 [-0.0314, +0.0331] | not separated | 2,322 / 137 |
| S checkpoint heading error at arc 30 m | deg | T0 | s0 | 4.2440 | 4.2580 | -0.0140 [-0.1327, +0.1158] | not separated | 2,322 / 137 |
| S checkpoint heading error at arc 30 m | deg | T0 | s1 | 4.1296 | 4.1223 | +0.0073 [-0.1896, +0.1821] | not separated | 2,322 / 137 |
| S checkpoint position error at arc 50 m | m | T0 | s0 | 8.1082 | 8.1026 | +0.0056 [-0.0525, +0.0662] | not separated | 2,319 / 137 |
| S checkpoint position error at arc 50 m | m | T0 | s1 | 7.9593 | 7.9401 | +0.0192 [-0.0622, +0.0988] | not separated | 2,319 / 137 |
| S checkpoint heading error at arc 50 m | deg | T0 | s0 | 7.1077 | 7.1877 | -0.0800 [-0.2549, +0.1043] | not separated | 2,319 / 137 |
| S checkpoint heading error at arc 50 m | deg | T0 | s1 | 6.9929 | 6.8909 | +0.1020 [-0.1231, +0.3227] | not separated | 2,319 / 137 |
| S checkpoint position error at arc 80 m | m | T0 | s0 | 11.6970 | 11.6685 | +0.0284 [-0.1305, +0.2028] | not separated | 2,294 / 136 |
| S checkpoint position error at arc 80 m | m | T0 | s1 | 11.2166 | 11.1435 | +0.0730 [-0.0738, +0.2236] | not separated | 2,294 / 136 |
| S checkpoint heading error at arc 80 m | deg | T0 | s0 | 13.9881 | 14.0434 | -0.0552 [-0.3549, +0.2732] | not separated | 2,294 / 136 |
| S checkpoint heading error at arc 80 m | deg | T0 | s1 | 13.3560 | 13.4232 | -0.0672 [-0.4166, +0.3149] | not separated | 2,294 / 136 |
| S checkpoint position error at arc 30 m (ticks>=1, mean over scored ticks) | m | T1-kin | s0 | 3.3937 | 5.8796 | -2.4858 [-2.9816, -1.9725] | separated, A better | 2,321 / 141 |
| S checkpoint position error at arc 30 m (ticks>=1, mean over scored ticks) | m | T1-kin | s1 | 3.3763 | 6.8148 | -3.4385 [-4.0237, -2.8852] | separated, A better | 2,321 / 141 |
| S checkpoint heading error at arc 30 m (ticks>=1) | deg | T1-kin | s0 | 8.7130 | 15.2808 | -6.5678 [-7.9858, -5.1834] | separated, A better | 2,321 / 141 |
| S checkpoint heading error at arc 30 m (ticks>=1) | deg | T1-kin | s1 | 8.3705 | 18.0349 | -9.6644 [-11.3360, -7.9982] | separated, A better | 2,321 / 141 |
| S checkpoint position error at arc 50 m (ticks>=1, mean over scored ticks) | m | T1-kin | s0 | 6.3385 | 11.0538 | -4.7153 [-5.6453, -3.7692] | separated, A better | 2,313 / 141 |
| S checkpoint position error at arc 50 m (ticks>=1, mean over scored ticks) | m | T1-kin | s1 | 6.2185 | 12.7044 | -6.4860 [-7.5233, -5.4640] | separated, A better | 2,313 / 141 |
| S checkpoint heading error at arc 50 m (ticks>=1) | deg | T1-kin | s0 | 12.4405 | 19.1367 | -6.6962 [-8.2334, -5.2001] | separated, A better | 2,313 / 141 |
| S checkpoint heading error at arc 50 m (ticks>=1) | deg | T1-kin | s1 | 12.2124 | 21.9205 | -9.7080 [-11.4681, -8.0005] | separated, A better | 2,313 / 141 |
| S checkpoint position error at arc 80 m (ticks>=1, mean over scored ticks) | m | T1-kin | s0 | 12.6215 | 20.2218 | -7.6003 [-9.2151, -5.9920] | separated, A better | 2,290 / 141 |
| S checkpoint position error at arc 80 m (ticks>=1, mean over scored ticks) | m | T1-kin | s1 | 12.3760 | 22.7372 | -10.3612 [-12.1269, -8.6323] | separated, A better | 2,290 / 141 |
| S checkpoint heading error at arc 80 m (ticks>=1) | deg | T1-kin | s0 | 19.2006 | 25.3864 | -6.1858 [-7.8455, -4.6291] | separated, A better | 2,290 / 141 |
| S checkpoint heading error at arc 80 m (ticks>=1) | deg | T1-kin | s1 | 18.7095 | 27.4433 | -8.7338 [-10.7107, -6.7780] | separated, A better | 2,290 / 141 |

### 2.5 STRATEGIC at T1-kin against CTRV, per run (the reader's unpaired view, recomputed; negative = S nearer the route checkpoint than CTRV)

| run | arc | S - CTRV, ticks>=1 [95 % interval] | verdict | n windows |
|---|---|---|---|---|
| DAG s0 | 30 m | -1.7175 [-2.2707, -1.2803] | separated, S nearer than CTRV | 2,321 |
| DAG s0 | 50 m | -4.4832 [-5.6271, -3.4998] | separated, S nearer than CTRV | 2,313 |
| DAG s0 | 80 m | -9.6871 [-11.8435, -7.5669] | separated, S nearer than CTRV | 2,290 |
| DAG-0 s0 | 30 m | +0.1178 [-0.2696, +0.4935] | not separated | 2,321 |
| DAG-0 s0 | 50 m | -0.9084 [-1.9668, +0.0696] | not separated | 2,313 |
| DAG-0 s0 | 80 m | -3.8923 [-6.0888, -1.8480] | separated, S nearer than CTRV | 2,290 |
| DAG s1 | 30 m | -1.8734 [-2.3640, -1.4387] | separated, S nearer than CTRV | 2,321 |
| DAG s1 | 50 m | -4.9034 [-5.9596, -3.9235] | separated, S nearer than CTRV | 2,313 |
| DAG s1 | 80 m | -10.3254 [-12.4401, -8.3492] | separated, S nearer than CTRV | 2,290 |
| DAG-0 s1 | 30 m | +0.8148 [+0.3481, +1.2917] | separated, S farther than CTRV | 2,321 |
| DAG-0 s1 | 50 m | +0.3335 [-0.8136, +1.4573] | not separated | 2,313 |
| DAG-0 s1 | 80 m | -1.9495 [-4.3742, +0.4859] | not separated | 2,290 |

## 3. DAG - ORC on R-TLAT / R-TLON (T1-kin): on-policy states against the law

⚠️ **The arms differ in more than the lever.** ORC is the O-RECOVER ego-perturbation arm (perturbation law, p = 0.3, trained on box 2 by another build); DAG is own-state replacement at p = 0.1. The two also differ in dose, in the fine-tune build and in what the replaced states are. This is the PREREG's 'scientific comparison, never decisive'. Same estimator, matched cells (scored in both runs), ticks >= 1.

| row | seed | tier | DAG level | ORC level | DAG - ORC [95 % interval] | verdict | matched cell pairs | own-cells version beside |
|---|---|---|---|---|---|---|---|---|
| R-TLAT (lat macro-F1) | s0 | T1-kin | 0.6071 | 0.5342 | +0.0728 [+0.0233, +0.1138] | separated, DAG better | 3,829 | +0.0689 [+0.0176, +0.1116], separated |
| R-TLAT (lat macro-F1) | s1 | T1-kin | 0.6409 | 0.5526 | +0.0883 [+0.0369, +0.1372] | separated, DAG better | 3,832 | +0.0876 [+0.0343, +0.1390], separated |
| R-TLON (lon macro-F1) | s0 | T1-kin | 0.3201 | 0.2756 | +0.0445 [+0.0205, +0.0674] | separated, DAG better | 3,727 | +0.0478 [+0.0248, +0.0709], separated |
| R-TLON (lon macro-F1) | s1 | T1-kin | 0.3280 | 0.2918 | +0.0362 [+0.0154, +0.0601] | separated, DAG better | 3,726 | +0.0392 [+0.0148, +0.0657], separated |

For reference, the same rows against DAG-0 (the decisive rows, reproduced here by control C7):

| row | seed | DAG level | DAG-0 level | DAG - DAG-0 [95 % interval] | verdict | matched cell pairs |
|---|---|---|---|---|---|---|
| R-TLAT | s0 | 0.6029 | 0.4947 | +0.1082 [+0.0491, +0.1572] | separated, DAG better | 3,863 |
| R-TLAT | s1 | 0.6385 | 0.5201 | +0.1184 [+0.0755, +0.1585] | separated, DAG better | 3,849 |
| R-TLON | s0 | 0.3213 | 0.2659 | +0.0554 [+0.0300, +0.0847] | separated, DAG better | 3,770 |
| R-TLON | s1 | 0.3311 | 0.2505 | +0.0806 [+0.0560, +0.1043] | separated, DAG better | 3,758 |

## 4. DAG - M2b at T0 (what 3,000 more steps plus own-state replacement did to the base)

A = DAG, B = M2b (the init checkpoint; its banked box-2 standard eval `raw/full/eval/eval_M2b_s{0,1}.npz` at the mirror tip 7d29172b, the source of MODEL_REGISTRY section 17: ADE6 2.1753 / 2.0546). ⚠️ DAG is M2b plus a 3,000-step fine-tune at lr 1e-4; **the fine-tune alone is DAG-0's job** (section 2). This contrast mixes the lever with 3,000 more steps and a changed LR schedule, and is T0 only (the M2b T1-kin dumps are not part of this package). Its sign across seeds is the information; each arm is one training seed.

### 4.1 ADE6 (reference)

| tier | seed | DAG level | M2b level | DAG - M2b [95 % interval] | verdict | n |
|---|---|---|---|---|---|---|
| T0 | s0 | 1.9736 | 2.1753 | -0.2018 [-0.2936, -0.1107] | separated, A better | 2,399 / 141 |
| T0 | s1 | 2.0984 | 2.0546 | +0.0438 [-0.0311, +0.1205] | not separated | 2,399 / 141 |

### 4.2 LONGITUDINAL

| metric (A - B) | unit | tier | seed | A level | B level | A - B [95 % interval] | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|---|
| speed MAE | m/s | T0 | s0 | 0.9208 | 1.0674 | -0.1466 [-0.2005, -0.0962] | separated, A better | 2,399 / 141 |
| speed MAE | m/s | T0 | s1 | 0.9751 | 0.9112 | +0.0638 [+0.0273, +0.0985] | separated, A WORSE | 2,399 / 141 |
| accel MAE | m/s^2 | T0 | s0 | 0.9992 | 1.2058 | -0.2066 [-0.2589, -0.1535] | separated, A better | 2,399 / 141 |
| accel MAE | m/s^2 | T0 | s1 | 1.0581 | 0.9313 | +0.1267 [+0.0985, +0.1541] | separated, A WORSE | 2,399 / 141 |
| along-track MAE | m | T0 | s0 | 1.5813 | 1.8097 | -0.2285 [-0.3227, -0.1415] | separated, A better | 2,399 / 141 |
| along-track MAE | m | T0 | s1 | 1.6861 | 1.6081 | +0.0780 [+0.0077, +0.1464] | separated, A WORSE | 2,399 / 141 |
| min headway | m | T0 | s0 | 25.3657 | 25.6223 | -0.3228 [-0.5689, -0.0540] | separated (no direction label) | 721 / 73 |
| min headway | m | T0 | s1 | 25.9014 | 25.5441 | +0.3657 [-0.0145, +0.7911] | not separated | 726 / 74 |
| min time gap | s | T0 | s0 | 3.9403 | 4.0223 | -0.0834 [-0.1954, +0.0160] | not separated | 664 / 73 |
| min time gap | s | T0 | s1 | 3.9736 | 3.9505 | +0.0244 [-0.0273, +0.0809] | not separated | 662 / 73 |
| min TTC (censored at a 30 s cap) | s | T0 | s0 | 19.6959 | 19.3964 | +0.1857 [-0.2870, +0.7231] | not separated | 721 / 73 |
| min TTC (censored at a 30 s cap) | s | T0 | s1 | 19.7162 | 19.5236 | +0.1790 [-0.4058, +0.7273] | not separated | 726 / 74 |
| absolute headway error vs the human | m | T0 | s0 | - | - | -0.0016 [-0.2004, +0.2122] | not separated | 713 / 72 |
| absolute headway error vs the human | m | T0 | s1 | - | - | +0.2244 [-0.1007, +0.6091] | not separated | 719 / 73 |
| absolute time-gap error vs the human | s | T0 | s0 | - | - | -0.0366 [-0.1595, +0.0496] | not separated | 656 / 72 |
| absolute time-gap error vs the human | s | T0 | s1 | - | - | -0.0252 [-0.0599, +0.0076] | not separated | 655 / 72 |
| absolute min-TTC error vs the human | s | T0 | s0 | - | - | -0.2829 [-0.7688, +0.1243] | not separated | 713 / 72 |
| absolute min-TTC error vs the human | s | T0 | s1 | - | - | -0.2826 [-0.7826, +0.1881] | not separated | 719 / 73 |
| LEAD-VIOL, at risk (violating share) | share | T0 | s0 | 0.0571 | 0.0353 | +0.0218 [+0.0093, +0.0367] | separated, A WORSE | 595 / 70 (violations A/B among 595: 34/21) |
| LEAD-VIOL, at risk (violating share) | share | T0 | s1 | 0.0454 | 0.0605 | -0.0151 [-0.0313, +0.0016] | not separated | 595 / 70 (violations A/B among 595: 27/36) |
| LEAD-VIOL, all lead windows | share | T0 | s0 | 0.0458 | 0.0283 | +0.0175 [+0.0070, +0.0292] | separated, A WORSE | 743 / 77 (violations A/B among 743: 34/21) |
| LEAD-VIOL, all lead windows | share | T0 | s1 | 0.0363 | 0.0485 | -0.0121 [-0.0257, +0.0013] | not separated | 743 / 77 (violations A/B among 743: 27/36) |
| lead-progress ratio | ratio | T0 | s0 | 0.9985 | 0.9243 | +0.0741 [+0.0507, +0.1016] | separated (no direction label) | 595 / 70 |
| lead-progress ratio | ratio | T0 | s1 | 0.9545 | 1.0063 | -0.0518 [-0.0887, -0.0154] | separated (no direction label) | 595 / 70 |

### 4.3 LATERAL

| metric (A - B) | unit | tier | seed | A level | B level | A - B [95 % interval] | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|---|
| heading MAE | deg | T0 | s0 | 4.3322 | 4.7771 | -0.4449 [-0.8354, -0.1312] | separated, A better | 2,341 / 141 |
| heading MAE | deg | T0 | s1 | 4.4410 | 4.3732 | +0.0678 [-0.1050, +0.2533] | not separated | 2,344 / 141 |
| curvature MAE | 1/m | T0 | s0 | 0.0602 | 0.0812 | -0.0209 [-0.0390, -0.0087] | separated, A better | 2,327 / 141 |
| curvature MAE | 1/m | T0 | s1 | 0.0702 | 0.0605 | +0.0097 [+0.0012, +0.0193] | separated, A WORSE | 2,329 / 141 |
| yaw-rate MAE | deg/s | T0 | s0 | 7.0727 | 8.1014 | -1.0287 [-1.7266, -0.4851] | separated, A better | 2,327 / 141 |
| yaw-rate MAE | deg/s | T0 | s1 | 7.7126 | 6.9312 | +0.7814 [+0.4201, +1.1707] | separated, A WORSE | 2,329 / 141 |
| cross-track MAE | m | T0 | s0 | 0.7700 | 0.7662 | +0.0038 [-0.0268, +0.0336] | not separated | 2,399 / 141 |
| cross-track MAE | m | T0 | s1 | 0.8233 | 0.8644 | -0.0411 [-0.0834, +0.0034] | not separated | 2,399 / 141 |

### 4.4 TACTICAL

| metric (A - B) | unit | tier | seed | A level | B level | A - B [95 % interval] | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|---|
| lateral decision agreement (plan-derived) | frac | T0 | s0 | 0.8508 | 0.8458 | +0.0050 [-0.0096, +0.0196] | not separated | 2,399 / 141 |
| lateral decision agreement (plan-derived) | frac | T0 | s1 | 0.8370 | 0.8287 | +0.0083 [-0.0079, +0.0279] | not separated | 2,399 / 141 |
| longitudinal decision agreement (plan-derived) | frac | T0 | s0 | 0.5773 | 0.5790 | -0.0017 [-0.0242, +0.0192] | not separated | 2,399 / 141 |
| longitudinal decision agreement (plan-derived) | frac | T0 | s1 | 0.5607 | 0.5919 | -0.0313 [-0.0600, -0.0029] | separated, A WORSE | 2,399 / 141 |
| lat head macro-F1 (window argmax vs v9 label) | - | T0 | s0 | 0.8347 | 0.8157 | +0.0190 [-0.0322, +0.0770] | not separated | 2,107 / 139 |
| lat head macro-F1 (window argmax vs v9 label) | - | T0 | s1 | 0.8351 | 0.8292 | +0.0059 [-0.0188, +0.0374] | not separated | 2,107 / 139 |
| lat head accuracy (window argmax vs v9 label) | - | T0 | s0 | 0.9530 | 0.9549 | -0.0019 [-0.0105, +0.0063] | not separated | 2,107 / 139 |
| lat head accuracy (window argmax vs v9 label) | - | T0 | s1 | 0.9535 | 0.9544 | -0.0009 [-0.0079, +0.0051] | not separated | 2,107 / 139 |
| lon head macro-F1 (window argmax vs v9 label) | - | T0 | s0 | 0.4944 | 0.4992 | -0.0047 [-0.0541, +0.0381] | not separated | 2,240 / 136 |
| lon head macro-F1 (window argmax vs v9 label) | - | T0 | s1 | 0.5212 | 0.5095 | +0.0117 [-0.0368, +0.0600] | not separated | 2,240 / 136 |
| lon head accuracy (window argmax vs v9 label) | - | T0 | s0 | 0.6232 | 0.6085 | +0.0147 [-0.0123, +0.0438] | not separated | 2,240 / 136 |
| lon head accuracy (window argmax vs v9 label) | - | T0 | s1 | 0.6295 | 0.6339 | -0.0045 [-0.0347, +0.0238] | not separated | 2,240 / 136 |
| goal head macro-F1 (22 tokens) | - | T0 | s0 | 0.2472 | 0.2618 | -0.0145 [-0.0402, +0.0170] | not separated | 2,365 / 139 |
| goal head macro-F1 (22 tokens) | - | T0 | s1 | 0.2747 | 0.2621 | +0.0126 [-0.0187, +0.0457] | not separated | 2,365 / 139 |
| goal head micro token accuracy | - | T0 | s0 | 0.9565 | 0.9571 | -0.0006 [-0.0039, +0.0024] | not separated | 2,365 / 139 |
| goal head micro token accuracy | - | T0 | s1 | 0.9581 | 0.9582 | -0.0001 [-0.0020, +0.0019] | not separated | 2,365 / 139 |
| 5-way collapsed manoeuvre accuracy (plan-derived) | - | T0 | s0 | 0.5802 | 0.5811 | -0.0008 [-0.0254, +0.0229] | not separated | 2,399 / 141 |
| 5-way collapsed manoeuvre accuracy (plan-derived) | - | T0 | s1 | 0.5602 | 0.5782 | -0.0179 [-0.0429, +0.0088] | not separated | 2,399 / 141 |
| lateral decision accuracy (plan-derived, tool's own) | - | T0 | s0 | 0.8508 | 0.8458 | +0.0050 [-0.0096, +0.0196] | not separated | 2,399 / 141 |
| lateral decision accuracy (plan-derived, tool's own) | - | T0 | s1 | 0.8370 | 0.8287 | +0.0083 [-0.0079, +0.0279] | not separated | 2,399 / 141 |
| longitudinal decision accuracy (plan-derived, tool's own) | - | T0 | s0 | 0.5773 | 0.5790 | -0.0017 [-0.0242, +0.0192] | not separated | 2,399 / 141 |
| longitudinal decision accuracy (plan-derived, tool's own) | - | T0 | s1 | 0.5607 | 0.5919 | -0.0313 [-0.0600, -0.0029] | separated, A WORSE | 2,399 / 141 |
| goal-point error (FDE6) | m | T0 | s0 | 5.6500 | 6.2810 | -0.6310 [-0.9125, -0.3577] | separated, A better | 2,399 / 141 |
| goal-point error (FDE6) | m | T0 | s1 | 6.0567 | 5.8582 | +0.1985 [-0.0381, +0.4432] | not separated | 2,399 / 141 |
| goal bearing MAE | deg | T0 | s0 | 3.6282 | 4.0999 | -0.4717 [-1.4533, +0.1530] | not separated | 2,367 / 141 |
| goal bearing MAE | deg | T0 | s1 | 3.9731 | 3.5884 | +0.3847 [-0.1485, +1.2492] | not separated | 2,367 / 141 |

### 4.5 STRATEGIC (NOT paused)

| metric (A - B) | unit | tier | seed | A level | B level | A - B [95 % interval] | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|---|
| S checkpoint position error at arc 30 m | m | T0 | s0 | 7.0048 | 7.0231 | -0.0183 [-0.0745, +0.0373] | not separated | 2,322 / 137 |
| S checkpoint position error at arc 30 m | m | T0 | s1 | 6.9723 | 6.9802 | -0.0079 [-0.0748, +0.0673] | not separated | 2,322 / 137 |
| S checkpoint heading error at arc 30 m | deg | T0 | s0 | 4.2440 | 4.2002 | +0.0438 [-0.2037, +0.3003] | not separated | 2,322 / 137 |
| S checkpoint heading error at arc 30 m | deg | T0 | s1 | 4.1296 | 4.0437 | +0.0859 [-0.0974, +0.2608] | not separated | 2,322 / 137 |
| S checkpoint position error at arc 50 m | m | T0 | s0 | 8.1082 | 8.0122 | +0.0960 [-0.0644, +0.2674] | not separated | 2,319 / 137 |
| S checkpoint position error at arc 50 m | m | T0 | s1 | 7.9593 | 7.9384 | +0.0209 [-0.1190, +0.2089] | not separated | 2,319 / 137 |
| S checkpoint heading error at arc 50 m | deg | T0 | s0 | 7.1077 | 6.6864 | +0.4213 [-0.1062, +1.0461] | not separated | 2,319 / 137 |
| S checkpoint heading error at arc 50 m | deg | T0 | s1 | 6.9929 | 6.7187 | +0.2742 [-0.1693, +0.8685] | not separated | 2,319 / 137 |
| S checkpoint position error at arc 80 m | m | T0 | s0 | 11.6970 | 11.2059 | +0.4911 [+0.0093, +1.0719] | separated, A WORSE | 2,294 / 136 |
| S checkpoint position error at arc 80 m | m | T0 | s1 | 11.2166 | 11.2141 | +0.0025 [-0.2667, +0.2615] | not separated | 2,294 / 136 |
| S checkpoint heading error at arc 80 m | deg | T0 | s0 | 13.9881 | 13.0540 | +0.9341 [-0.1986, +2.4094] | not separated | 2,294 / 136 |
| S checkpoint heading error at arc 80 m | deg | T0 | s1 | 13.3560 | 13.6014 | -0.2453 [-0.7404, +0.2492] | not separated | 2,294 / 136 |

## 5. Per-class lat-head F1 (TURN_L / TURN_R), T1-kin ticks >= 1

The T1-HEADS drop sat in the lateral head. Per-class F1 = `taniteval.v8f_consistency.macro_f1(...)['per_class_f1']`, the quantity the macro-F1 is the nan-mean of, on the SAME encoded cells, the same matched-cell rule and slot layout as R-TLAT, and the same bootstrap draws (`ci._draws`, seed 0, B = 2,000; a draw in which the class has no labelled cell is excluded and counted). Class ids are v9 ids (`build_v9_labels.py:461`, LAT9): 0 LANE_KEEP, 1 TURN_L, 2 TURN_R, 3 LANE_CHANGE_L, 4 LANE_CHANGE_R. **Support is small for the turn classes, so read the n.**

### 5.1 DAG - DAG-0 (the decisive pair)

| class | seed | DAG F1 | DAG-0 F1 | DAG - DAG-0 [95 % interval] | verdict | support (matched cells) / episodes | draws with the class absent |
|---|---|---|---|---|---|---|---|
| TURN_L | s0 | 0.3660 | 0.2327 | +0.1333 [-0.0260, +0.2629] | not separated | 92 / 11 | 0 |
| TURN_L | s1 | 0.4294 | 0.2812 | +0.1482 [+0.0533, +0.2378] | separated, DAG better | 90 / 11 | 0 |
| TURN_R | s0 | 0.4891 | 0.3183 | +0.1708 [+0.1067, +0.2298] | separated, DAG better | 292 / 27 | 0 |
| TURN_R | s1 | 0.5292 | 0.3390 | +0.1901 [+0.1174, +0.2599] | separated, DAG better | 295 / 26 | 0 |
| LANE_KEEP | s0 | 0.9537 | 0.9332 | +0.0206 [+0.0119, +0.0300] | separated, DAG better | 3,479 / 139 | 0 |
| LANE_KEEP | s1 | 0.9569 | 0.9400 | +0.0169 [+0.0091, +0.0262] | separated, DAG better | 3,464 / 139 | 0 |
| LANE_CHANGE_L | s0 | - | - | not computed | not computed: class has no labelled cell among the matched ticks>=1 cells | 0 | - |
| LANE_CHANGE_L | s1 | - | - | not computed | not computed: class has no labelled cell among the matched ticks>=1 cells | 0 | - |
| LANE_CHANGE_R | s0 | - | - | not computed | not computed: class has no labelled cell among the matched ticks>=1 cells | 0 | - |
| LANE_CHANGE_R | s1 | - | - | not computed | not computed: class has no labelled cell among the matched ticks>=1 cells | 0 | - |

### 5.2 DAG - ORC (arms differ in more than the lever)

| class | seed | DAG F1 | ORC F1 | DAG - ORC [95 % interval] | verdict | support (matched cells) / episodes | draws with the class absent |
|---|---|---|---|---|---|---|---|
| TURN_L | s0 | 0.3709 | 0.2741 | +0.0967 [-0.0261, +0.1912] | not separated | 91 / 11 | 0 |
| TURN_L | s1 | 0.4348 | 0.3314 | +0.1034 [-0.0258, +0.2203] | not separated | 90 / 11 | 0 |
| TURN_R | s0 | 0.4954 | 0.3867 | +0.1086 [+0.0391, +0.1767] | separated, DAG better | 290 / 26 | 0 |
| TURN_R | s1 | 0.5304 | 0.3827 | +0.1476 [+0.0737, +0.2256] | separated, DAG better | 292 / 26 | 0 |
| LANE_KEEP | s0 | 0.9549 | 0.9419 | +0.0130 [+0.0062, +0.0209] | separated, DAG better | 3,448 / 139 | 0 |
| LANE_KEEP | s1 | 0.9575 | 0.9437 | +0.0138 [+0.0069, +0.0221] | separated, DAG better | 3,450 / 139 | 0 |
| LANE_CHANGE_L | s0 | - | - | not computed | not computed: class has no labelled cell among the matched ticks>=1 cells | 0 | - |
| LANE_CHANGE_L | s1 | - | - | not computed | not computed: class has no labelled cell among the matched ticks>=1 cells | 0 | - |
| LANE_CHANGE_R | s0 | - | - | not computed | not computed: class has no labelled cell among the matched ticks>=1 cells | 0 | - |
| LANE_CHANGE_R | s1 | - | - | not computed | not computed: class has no labelled cell among the matched ticks>=1 cells | 0 | - |

### 5.3 Per-run per-class F1 levels (unpaired, each run's own scored cells): tick 0 (logged-state rollout step) against ticks >= 1 (the model's own executed states)

| run | class | F1 tick 0 | support tick 0 | F1 ticks>=1 | support ticks>=1 |
|---|---|---|---|---|---|
| DAG s0 | LANE_KEEP | 0.9744 | 1910 | 0.9534 | 3527 |
| DAG s0 | TURN_L | 0.8037 | 47 | 0.3694 | 94 |
| DAG s0 | TURN_R | 0.7625 | 150 | 0.4829 | 294 |
| DAG-0 s0 | LANE_KEEP | 0.9753 | 1910 | 0.9321 | 3535 |
| DAG-0 s0 | TURN_L | 0.8155 | 47 | 0.2254 | 96 |
| DAG-0 s0 | TURN_R | 0.7604 | 150 | 0.3166 | 290 |
| ORC s0 | LANE_KEEP | 0.9725 | 1910 | 0.9415 | 3495 |
| ORC s0 | TURN_L | 0.8148 | 47 | 0.2741 | 89 |
| ORC s0 | TURN_R | 0.7423 | 150 | 0.3834 | 295 |
| DAG s1 | LANE_KEEP | 0.9760 | 1910 | 0.9570 | 3540 |
| DAG s1 | TURN_L | 0.8317 | 47 | 0.4268 | 91 |
| DAG s1 | TURN_R | 0.7643 | 150 | 0.5244 | 295 |
| DAG-0 s1 | LANE_KEEP | 0.9768 | 1910 | 0.9398 | 3505 |
| DAG-0 s1 | TURN_L | 0.8269 | 47 | 0.2769 | 92 |
| DAG-0 s1 | TURN_R | 0.7826 | 150 | 0.3371 | 296 |
| ORC s1 | LANE_KEEP | 0.9741 | 1910 | 0.9429 | 3501 |
| ORC s1 | TURN_L | 0.7921 | 47 | 0.3218 | 89 |
| ORC s1 | TURN_R | 0.7674 | 150 | 0.3808 | 298 |

(LANE_CHANGE_L / LANE_CHANGE_R have no labelled cell at either tick in this window set: the labelled lateral cells fall in 3 of the 5 classes. They are listed as 'not computed' in 5.1 and 5.2 and carry the same status in `DAG_T1CELLS.json`. Tick-0 supports 1,910 / 47 / 150 agree with the T0 paired-families tool's lateral-head `support` [1910, 47, 150, 0, 0] in `PAIRFAM_T0_DAG-DAG0_s0.json`.)

## 6. Not computed, with the reason and n

| family | cell | tier | reason | n |
|---|---|---|---|---|
| TACTICAL | paired lat / lon head accuracy, goal-head F1, 5-way collapsed accuracy, goal-point error / bearing at T1-kin | T1-kin | `v8f_paired_families.py` needs an eval `run` dump (`pred`, `lat_logits`, `lon_logits`, `goal_prob`, `ckpt`, `ck_pos_err`, `ade6` per window) and cannot read the tick dumps; the head macro-F1 (R-TLAT / R-TLON) and per-class F1 are computed here from the ticks instead. The goal-head F1 exists UNPAIRED per run in `DAG_READ.json` beside.goal_macro_f1 (ticks>=1 and tick 0) | 0 (not attempted) |
| TACTICAL | anchor selection; selected-vs-executed manoeuvre decisions | both | UNAVAILABLE BY DESIGN: a trajectory dump carries one path per window and no tactical selection (the tool's own statement) | 0 |
| LONGITUDINAL | target-speed accuracy bands; ego-progress ratio | both | are in each eval JSON's `raw.families.longitudinal` as unpaired levels and were not paired here (nothing to pair at T1-kin: `arms.cl.four_families` carries them per run) | 0 (not attempted) |
| LONGITUDINAL | anti-echo discharge | T0 | UNAVAILABLE in the eval JSON itself (no ego speed at t0 in the window dict); see 1.1 | 2,399 windows each |
| LONGITUDINAL | distance keeping where no lead exists | both | only LEAD windows are scored (743 of 2,399); NO_LEAD 552, NOT_STRAIGHT 931, NO_LABEL 173 are counted and never scored as free flow | see section 2.1 n |
| STRATEGIC | S checkpoint at T1-kin against the v9 route label on windows with no valid checkpoint or off-route ticks | T1-kin | windows or ticks with no label, or where S's pose is more than 3 m from the logged route, are not scored (the reader's rule); n per cell is in section 2.4 | see n |
| per-class | LANE_CHANGE_L / LANE_CHANGE_R | T1-kin | no labelled cell in the matched ticks>=1 cells | 0 |
| DAG - ORC | the four families other than R-TLAT / R-TLON | both | not asked by the PREREG (ORC is compared on the two head rows only) | 0 (not attempted) |
| DAG - M2b | T1-kin | T1-kin | M2b has no T1-kin dump in this package; the comparison is T0 | 0 |
| tick-0 F1, R76 crawl rows M1 / L2, T1 off-route fraction at tick 2, R-TLAT / R-TLON at section 2's tie rule | - | T1-kin | not requested of this package. The first three are carried per run (unpaired) in `DAG_READ.json` beside; the tie-rule re-score needs the bank matcher | 0 |

## 7. What the families say, descriptively (computed from the tables; no interpretation)

### DAG - DAG-0

Digest by (metric, tier): the two seeds read together. A star marks the seed whose interval excludes 0. Cells with no separated seed are not listed. Counts are cells = metrics x tiers (two seeds each).

* **LONGITUDINAL** (24 metric-tier cells)
  * *both seeds separated, same sign, DAG better* (2): speed MAE (T1-kin; s0 -0.4658*, s1 -0.2816*); along-track MAE (T1-kin; s0 -0.4017*, s1 -0.2490*)
  * *both seeds separated, same sign, DAG WORSE* (3): absolute min-TTC error vs the human (T1-kin; s0 +0.8675*, s1 +0.5971*); LEAD-VIOL, at risk (violating share) (T1-kin; s0 +0.0420*, s1 +0.0303*); LEAD-VIOL, all lead windows (T1-kin; s0 +0.0336*, s1 +0.0242*)
  * *both seeds separated, same sign, no direction label* (3): min headway (T1-kin; s0 -1.0222*, s1 -0.5627*); min time gap (T1-kin; s0 -0.1612*, s1 -0.0939*); min TTC (censored at a 30 s cap) (T1-kin; s0 -2.2480*, s1 -1.2406*)
  * *one seed separated, DAG better* (1): absolute time-gap error vs the human (T0; s0 +0.0053, s1 -0.0301*)
  * *one seed separated, DAG WORSE* (5): speed MAE (T0; s0 -0.0098, s1 +0.0204*); accel MAE (T0; s0 +0.0063, s1 +0.0338*); along-track MAE (T0; s0 -0.0002, s1 +0.0461*); accel MAE (T1-kin; s0 -0.0193, s1 +0.4490*); absolute time-gap error vs the human (T1-kin; s0 +0.0835*, s1 +0.0126)
* **LATERAL** (8 metric-tier cells)
  * *both seeds separated, same sign, DAG better* (2): heading MAE (T1-kin; s0 -3.5139*, s1 -1.7646*); cross-track MAE (T1-kin; s0 -0.2955*, s1 -0.3069*)
  * *one seed separated, DAG better* (2): yaw-rate MAE (T0; s0 -0.1878*, s1 +0.0867); yaw-rate MAE (T1-kin; s0 -6.4613*, s1 -1.6967)
  * *one seed separated, DAG WORSE* (1): curvature MAE (T1-kin; s0 -0.0071, s1 +0.0168*)
* **TACTICAL** (20 metric-tier cells)
  * *both seeds separated, same sign, DAG better* (6): lateral decision agreement (plan-derived) (T1-kin; s0 +0.1000*, s1 +0.1917*); longitudinal decision agreement (plan-derived) (T1-kin; s0 +0.0542*, s1 +0.0717*); R-TLAT: lat macro-F1, ticks>=1, matched cells (T1-kin; s0 +0.1082*, s1 +0.1184*); R-TLON: lon macro-F1, ticks>=1, matched cells (T1-kin; s0 +0.0554*, s1 +0.0806*); lat head F1 of TURN_R, ticks>=1, matched cells (T1-kin; s0 +0.1708*, s1 +0.1901*); lat head F1 of LANE_KEEP, ticks>=1, matched cells (T1-kin; s0 +0.0206*, s1 +0.0169*)
  * *one seed separated, DAG better* (2): goal bearing MAE (T0; s0 -0.4577*, s1 +0.1428); lat head F1 of TURN_L, ticks>=1, matched cells (T1-kin; s0 +0.1333, s1 +0.1482*)
  * *one seed separated, DAG WORSE* (4): lateral decision agreement (plan-derived) (T0; s0 +0.0000, s1 -0.0108*); lon head accuracy (window argmax vs v9 label) (T0; s0 -0.0170*, s1 -0.0089); lateral decision accuracy (plan-derived, tool's own) (T0; s0 +0.0000, s1 -0.0108*); goal-point error (FDE6) (T0; s0 -0.0520, s1 +0.1011*)
* **STRATEGIC** (12 metric-tier cells)
  * *both seeds separated, same sign, DAG better* (6): S checkpoint position error at arc 30 m (ticks>=1, mean over scored ticks) (T1-kin; s0 -2.4858*, s1 -3.4385*); S checkpoint heading error at arc 30 m (ticks>=1) (T1-kin; s0 -6.5678*, s1 -9.6644*); S checkpoint position error at arc 50 m (ticks>=1, mean over scored ticks) (T1-kin; s0 -4.7153*, s1 -6.4860*); S checkpoint heading error at arc 50 m (ticks>=1) (T1-kin; s0 -6.6962*, s1 -9.7080*); S checkpoint position error at arc 80 m (ticks>=1, mean over scored ticks) (T1-kin; s0 -7.6003*, s1 -10.3612*); S checkpoint heading error at arc 80 m (ticks>=1) (T1-kin; s0 -6.1858*, s1 -8.7338*)

### DAG - M2b

Digest by (metric, tier): the two seeds read together. A star marks the seed whose interval excludes 0. Cells with no separated seed are not listed. Counts are cells = metrics x tiers (two seeds each).

* **LONGITUDINAL** (12 metric-tier cells)
  * *SEEDS DISAGREE (both separated, opposite signs)* (4): speed MAE (T0; s0 -0.1466*, s1 +0.0638*); accel MAE (T0; s0 -0.2066*, s1 +0.1267*); along-track MAE (T0; s0 -0.2285*, s1 +0.0780*); lead-progress ratio (T0; s0 +0.0741*, s1 -0.0518*)
  * *one seed separated, DAG WORSE* (2): LEAD-VIOL, at risk (violating share) (T0; s0 +0.0218*, s1 -0.0151); LEAD-VIOL, all lead windows (T0; s0 +0.0175*, s1 -0.0121)
  * *one seed separated, no direction label* (1): min headway (T0; s0 -0.3228*, s1 +0.3657)
* **LATERAL** (4 metric-tier cells)
  * *SEEDS DISAGREE (both separated, opposite signs)* (2): curvature MAE (T0; s0 -0.0209*, s1 +0.0097*); yaw-rate MAE (T0; s0 -1.0287*, s1 +0.7814*)
  * *one seed separated, DAG better* (1): heading MAE (T0; s0 -0.4449*, s1 +0.0678)
* **TACTICAL** (13 metric-tier cells)
  * *one seed separated, DAG better* (1): goal-point error (FDE6) (T0; s0 -0.6310*, s1 +0.1985)
  * *one seed separated, DAG WORSE* (2): longitudinal decision agreement (plan-derived) (T0; s0 -0.0017, s1 -0.0313*); longitudinal decision accuracy (plan-derived, tool's own) (T0; s0 -0.0017, s1 -0.0313*)
* **STRATEGIC** (6 metric-tier cells)
  * *one seed separated, DAG WORSE* (1): S checkpoint position error at arc 80 m (T0; s0 +0.4911*, s1 +0.0025)

### DAG - ORC

* **R-TLAT.** s0 +0.0728 [+0.0233, +0.1138] (separated); s1 +0.0883 [+0.0369, +0.1372] (separated)
* **R-TLON.** s0 +0.0445 [+0.0205, +0.0674] (separated); s1 +0.0362 [+0.0154, +0.0601] (separated)

## 8. Which variance each interval answers

1. **Another draw of EPISODES** (models held fixed): every interval here. That is all the bootstrap answers.
2. **Another TRAINING run:** answered only by the two seeds. DAG-0 is the control for '3,000 more steps' and ORC-0 (banked by O-RECOVER, same argv as DAG-0, another build) is the replicate floor `RF` that the decisive bars used; there is no replicate of DAG itself at a given seed. A separated interval from one training seed is necessary, not sufficient (H-ESTIM-SEED-1). Where the two seeds disagree in sign on a cell, the table shows it.
3. **Another INFERENCE run:** does not arise; the planner is deterministic (PREREG_V8F_PGT; DAG_READ.json estimator line).

## 9. Artifacts, provenance, estimator line

```
estimator : taniteval.ci.paired_episode_cluster_bootstrap (via v8f_beta_eval.paired_row / lead_metrics), B=2000, seed 0, over episodes; A - B per seed on the same windows
tier      : T0 = box 2 standard eval (ages cycle); T1-kin = dev-box CPU --limit causal --dump-ticks --dump-heads, replan-steps 10 (self-action OPEN loop)
windows   : 2399 windows, 141 episodes (eval-141); B = 2000; seed 0
tool tree : E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a  (instruments byte-equal to the 07dcb513 tree used by O-RECOVER modulo CRLF)
lead_block_sha256         : 7a879baa2b6f4c140c2b64c9c098a38f84ecb04926da61fa8e96cfffc33073d9
grid_sha256               : 659c3b65e364ea5de87a4e17d076eba5645df95cdcf39d8abc08caab1b2d594c
kin_sha256                : be57e283c3f037abd60d49814e84b05124a9667586894dfc6d1c3ecd191ba203
join_sha256               : f0ea722f59c1a4d8ba81ce7f9b204dd3a28950fc1c16778f63c2962786e5206a
read_json_sha256          : 3393fdd56bd561fd5181a8219d2c75d700036b9373da6ec12a8bcba51decde3d
runs_json_sha256          : e5cf61bc6198a2d756a56afd8420b672d1ae5171acaaf2aa8ef4d3c0319301d7
```

Run artifacts consumed (paths from `E:/Projects/TanitAD-artifacts/v8f-dagger-t1/runs.json`, sha256 of the T0 npz in `DAG_FAMILIES.json` `*/ids`):

* T0 s0: DAG `E:/Projects/TanitAD-artifacts/v8f-dagger/M2b_s0_dag/eval_M2b-dag_s0.npz` (sha256 `b147dc1bed03a89f...`); DAG-0 `E:/Projects/TanitAD-artifacts/v8f-dagger/M2b_s0_dag0/eval_M2b-dag0_s0.npz` (sha256 `4c0084c594b54124...`); M2b `E:/Projects/TanitAD-artifacts/dagfam/ref/m2b/eval_M2b_s0.npz` (sha256 `8ddcd42c119f7ca9...`)
* T1-kin s0: DAG `E:/Projects/TanitAD-artifacts/v8f-dagger-t1/dump_DAG_s0_m10_causal_heads`; DAG-0 `E:/Projects/TanitAD-artifacts/v8f-dagger-t1/dump_DAG0_s0_m10_causal_heads`; ticks sha256 in `DAG_T1CELLS.json` inputs.ticks
* T0 s1: DAG `E:/Projects/TanitAD-artifacts/v8f-dagger/M2b_s1_dag/eval_M2b-dag_s1.npz` (sha256 `8fc07dc18c4e11f8...`); DAG-0 `E:/Projects/TanitAD-artifacts/v8f-dagger/M2b_s1_dag0/eval_M2b-dag0_s1.npz` (sha256 `50569468526753f3...`); M2b `E:/Projects/TanitAD-artifacts/dagfam/ref/m2b/eval_M2b_s1.npz` (sha256 `0b9e40ff71fd3636...`)
* T1-kin s1: DAG `E:/Projects/TanitAD-artifacts/v8f-dagger-t1/dump_DAG_s1_m10_causal_heads`; DAG-0 `E:/Projects/TanitAD-artifacts/v8f-dagger-t1/dump_DAG0_s1_m10_causal_heads`; ticks sha256 in `DAG_T1CELLS.json` inputs.ticks

The package files and their roles are listed in `MANIFEST.txt` in this directory.
