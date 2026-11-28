<title>P5 — tracker-aware targets</title>

# RESULT P5: tracker-aware and margin-aware targets, held-out label test (2026-10-11)

**Pre-registration:** `eval/PREREG_P5_TRACKER_AWARE_TARGETS.md`, landed in `ac9e8ab4` before the full run. Its §9 logs
every later decision with its date. **Tier:** T1 label test on NAVSIM v1.1's non-reactive pseudo-simulation: NAVSIM's
own LQR tracker and kinematic bicycle execute each 8-pose plan. **No model is trained or scored.** **Population:** D1's
1,018 held-out rank-0 navtrain teacher targets over 11 logs, nothing re-selected, 0 rows raised. **Estimator:** paired
log-cluster bootstrap over the 11 logs, 10,000 resamples, seed 20260927. It answers *another draw of LOGS*. Labels and
tracker are deterministic (control C7 reads 0.0 m), so there is no training-seed or inference-seed variance to
replicate. With only 11 logs, the log count is the binding limit on every interval. **Evidence classes:** MEASURED
(artifact path) · INHERITED · ESTIMATED · HYPOTHESIS. Every number below is MEASURED from
`raw/2026-10-11-p5-target-labels/p5_labels.json` (labels, controls, verdicts) or `p5_families.json` (families),
and the tables in the appendix are rendered from them by `eval/p5_render.py`, not transcribed.

## 0 · Verdict

**Arm (ii), tracker-compensated labels: CONFIRMED, with LEARNABILITY RISK.**
- Δ sim8 fail (Tc20 − T) **−7.07 pp [−13.26, −1.83]**: 8.06 % → 0.98 % of rows. No guard is adverse.
- Tc20 repairs **76 of the 78** tracker-made failures (D1 class C). The failure count falls from 82 to 10.
- The executed trajectory lands far closer to the teacher's intent:
  - max-xy deviation −0.398 m [−0.458, −0.337] (median 0.414 → 0.060 m);
  - heading MAE −0.98° [−1.37, −0.50].
- **But the label is harder to learn.** Tc20's median per-row max |acceleration| is **2.824 m/s² against t20's 1.371,
  2.06×**, above the registered 2× bar. Heading rate is 1.68×, under the bar.
  - Under §6, a CONFIRMED arm with this flag goes to GPU only after a pre-registered smoothed-label amendment.
  - That amendment, **P5-S**, is registered in the prereg's §9 in the same commit as this result, before any P5-S code
    or number exists.

**Arm (i), a 0.6 m margin flag: NOT VIABLE.**
- It flags **46.07 %** of rows (bar ≤ 25 %), with recall 85.37 % (70 of 82) and precision 14.93 %.
- T's median margin is **0.632 m**, essentially at the threshold, so a 0.6 m criterion splits the data near its
  middle.
- No threshold in the descriptive sweep reaches both bars: 0.3 m flags 12.08 % but catches only 29.27 %.

**Control 3, wrong sign: REFUTED, as it must be.** Tw raises sim8 failure by +26.03 pp [+15.30, +35.28], and six
guards are adverse. The iteration measures tracking error, not noise.

**Two caveats, stated plainly. Neither changes the registered verdict.**
1. **Comfort's upper bound is exactly +0.00.** Δ comfort is −0.10 pp [−0.41, +0.00]: **one** row of 1,018 loses
   comfort. The rule's "adverse iff the upper bound < 0" is strict, so this is not adverse. It is reported, not
   rounded away.
2. **Tc20 is compliant only through NAVSIM's tracker.** Its own drawn 8-pose footprint fails interp8 on **12.87 %**
   of rows (131), against **0.59 %** (6) for T. So the share of rows that pass **both** checks *falls*: **86.84 %**
   for Tc20 against **91.75 %** for T (descriptive, §5). The label is right for the scorer NAVSIM runs and wrong as
   a drawn path. This is the "softer form of test-time compensation" the prereg's §8 names, and it is direct evidence
   for **PI decision (a)**.

**Stopping state (RULE ZERO).** Not done. The next lever, P5-S, is 0-GPU and unblocked, and it runs now (§7).

## 1 · Controls (all passed; each proven able to fail)

| control | known value | read | artifact |
|---|---|---|---|
| C1 identity: `ident`'s label vs t20, its 8 poses vs T, its sim8 vs T | bit-exact, every row | label bit-exact **True**, p8 bit-exact **True**, sim8 equal **True** | `p5_labels.json` `controls_raw` |
| C2 knot exactness: `e20[[4,9,14,19]] == e8[[1,3,5,7]]` | exact, every check | **True** over **9,162** checks | same |
| C3 wrong sign: Tw's median max-xy > T's, and Tw not CONFIRMED | both | 2.676 m vs 0.414 m; Tw **REFUTED** | same, `wrong_sign_Tw` |
| C4 D1 join: T's sim8 vs D1's, same rows | 100 %, 0 missing | **0** mismatches, **0** missing | same |
| C5 REVIEW_7: DAC 100 %, DDC and lk10 ≥ 99.5 % for T and H | as stated | **1,018 / 1,018** on all six counts | same |
| C6 margin direction: margin < 0 ⇒ interp8 violation | 0 exceptions | **0** | same |
| C7 batch independence: T alone vs T in batch | ≤ 1e-6 m | **0.0 m** | same |
| controls-only smoke, 40 rows | all PASS | **all PASS** | `smoke40_controls.{json,log}` |

**Deliberate-regression arms**, 6 rows each, run before landing. Each must turn exactly its own control FALSE:

| `--mutate` | breaks | control FALSE | all others | artifact |
|---|---|---|---|---|
| `knot` | knots shifted by 0.05 s | **C2** only | TRUE | `mutate_knot.{json,log}` |
| `ident` | `ident`'s sign 0 → 1e-9 | **C1** only | TRUE | `mutate_ident.*` |
| `d1` | T's sim8 flipped on row 0 | **C4** only | TRUE | `mutate_d1.*` |
| `r7` | T's DAC flipped on row 0 | **C5** only | TRUE | `mutate_r7.*` |
| `margin` | T's margin forced negative | **C6** only | TRUE | `mutate_margin.*` |
| `batch` | "alone" compared against the wrong row | **C7** only | TRUE | `mutate_batch.*` |

## 2 · Arm (ii): what the compensated label repairs and breaks

| D1 class | n | T sim8 fail | Tc20 sim8 fail | repaired | broken |
|---|---|---|---|---|---|
| pass | 934 | 0 | 2 | 0 | 2 |
| C, made by NAVSIM's tracker | 78 | 78 | 2 | **76** | 0 |
| A, on the teacher's own path | 6 | 4 | 6 | 0 | 2 |

- The repair is exactly where D1 said the failures were: the tracker-made class.
- Class A rows (the teacher's own 20-pose path leaves the drivable area) are not repaired. They cannot be: Tc20 lands
  *on* the teacher's intent, and that intent is the failing path. That is a target-quality problem, not a tracking
  problem.
- The four new failures (2 in pass, 2 in A) are the cost.
- Iteration converges: the median max-xy deviation goes **0.4136 → 0.1225 → 0.0764 → 0.0606 m** over k = 0..3. P4's
  8-pose compensation (Tc8) ends at 0.0573 m. The two agree to a median of 0.0195 m (max 0.2623 m) on the poses
  NAVSIM receives.

## 3 · Learnability: the flag, and what drives it

| label | max \|acc\| median | p95 | max \|heading rate\| median °/s | p95 |
|---|---|---|---|---|
| t20 | 1.371 | 3.136 | 5.110 | 32.742 |
| Tc20 | **2.824 (2.06×)** | 6.309 | 8.590 (1.68×) | 44.022 |
| Tw (wrong sign) | 3.319 | 15.939 | 9.929 | 79.018 |

- **HYPOTHESIS** for the mechanism, to be tested by P5-S: the correction `e20` is a **piecewise-linear**
  interpolation of the 8 knot errors. Its slope jumps at every knot. On a 0.2 s grid, a slope jump appears as a
  second-difference spike, and acceleration is that second difference over 0.04 s². Three iterations add three such
  kinked corrections.
- If this is right, an interpolant with a continuous second derivative (natural cubic spline) should remove most of
  the excess while keeping the same knot values, and therefore the same tracker repair.

## 4 · Arm (i): why a margin flag does not work here

- Margin and tracker failure are related: recall rises with the threshold.
- But T's margin distribution sits on top of the threshold. Median **0.632 m**, p05 0.234 m; H's median is 0.685 m.
- Any threshold that catches most failures also flags most good targets. 0.45 m: 28.00 % flagged, 69.51 % recall.
  0.75 m: 61.89 % / 97.56 %.
- So a margin training term on its own would push on half the dataset to fix 8 % of it. Drop-if-sim8 flags 8.06 %
  with 100 % recall, but needs the tracker at label time.
- ⇒ **P5-M is not pursued.** The margin signal is not discarded; it may still be useful as a *weighted* term. That
  would need its own prereg and is not on the path.

## 5 · The interp8 caveat, quantified (descriptive, no verdict)

Counts over the 1,018 rows (MEASURED, counted from the per-row records in `p5_labels.json` `rows` by
`eval/p5_render.py`; the full table, with Tw and `ident`, is in the appendix):

| ref | sim8 fail | interp8 fail | fail both | pass both |
|---|---|---|---|---|
| T | 82 | 6 | 4 | **934 (91.75 %)** |
| Tc20 | 10 | 131 | 7 | **884 (86.84 %)** |
| Tc8 | 10 | 134 | 7 | 881 (86.54 %) |
| H | 0 | 8 | 0 | 1,010 (99.21 %) |
| Hc | 6 | 147 | 3 | 868 (85.27 %) |
| *Tsel, DERIVED post hoc* | 6 | 52 | 3 | *963 (94.60 %)* |

- 128 of Tc20's 131 interp8 failures are on rows where T's own footprint passes. 50 are on rows where T fails sim8.
- ⚠️ **Tsel is DERIVED after seeing the data. It is not a pre-registered arm and carries no verdict.** It is "use Tc20
  only on the 82 rows where T fails sim8, else T", computed by recombining the rows already scored.
  - Like drop-if-sim8, it needs the tracker at label time.
  - It is the only variant here that beats T on *both* checks (94.60 % against 91.75 %).
  - But on the 82 rows it changes, the compensated label's median max |acceleration| is **5.019 against t20's 2.456
    m/s², 2.04×**. Selection does not remove the learnability cost; it concentrates it on the rows that are hardest
    already.
  - It is registered in the prereg's §9 as a descriptive read of P5-S and as an input to the P5-C design. It is
    nothing more until a pre-registration tests it.
- The pre-distortion pushes the *drawn* path outward so that the *tracked* path lands inside. Both compensations (Tc20,
  Tc8, Hc) show the same pattern, on teacher and human plans alike, so it is a property of compensation, not of Tc20.
- **What this means for PI decision (a)** (families on the reference vs on the executed trajectory). A student trained
  on Tc20 learns to draw a path that is wrong as drawn. Under NAVSIM's scoring, which executes the plan, that is
  correct. Under any evaluation or deployment that uses the drawn path directly, it is a regression on about 1 in 8
  rows. **The PI's decision (a) is therefore also a decision about which labels are admissible.** This result
  supplies the trade-off; it does not make the decision.

## 6 · Four families (T1, descriptive only)

Executed against intended, n 1,018, 11 logs, cluster-bootstrap half-width (2,000 resamples):

| family | metric | exec T vs T | exec Tc20 vs T |
|---|---|---|---|
| LONGITUDINAL | speed MAE m/s | 0.105 ± 0.013 | **0.027 ± 0.003** |
| | along MAE m | 0.167 ± 0.027 | **0.025 ± 0.004** |
| LATERAL | heading MAE ° | 1.530 ± 0.544 | **0.582 ± 0.156** |
| | yaw-rate MAE °/s | 2.128 ± 0.714 | 1.160 ± 0.297 |
| | curvature MAE 1/m | 0.0101 ± 0.0040 | 0.0061 ± 0.0024 |
| | cross MAE m | 0.120 ± 0.042 | **0.027 ± 0.008** |
| TACTICAL | lat-decision κ | 0.971 [0.959, 0.980] | 0.967 [0.942, 0.977] |
| | lon-decision κ | 0.971 [0.952, 0.992] | 0.992 [0.988, 0.998] |
| | goal-point error m | 0.429 [0.362, 0.497] | **0.053 [0.041, 0.065]** |
| STRATEGIC | — | **UNAVAILABLE** in NAVSIM; **PAUSED** (PI ruling 2026-09-19) | same |

- On every longitudinal and lateral metric, the tracker executes Tc20 closer to the teacher's intent than it executes
  T. The tactical decisions stay the same.
- **Against the human, Tc20 is not closer.** Goal-point error to H is 2.719 for Tc20 and 2.592 for T, with heavily
  overlapping intervals (full table in the appendix). Compensation makes the car do what the teacher meant. It does
  not make the teacher more human. The ~2.6 m teacher-to-human gap is a target-quality question, which is REVIEW_7's
  P1/P2, not P5.

## 7 · Next lever (RULE ZERO)

1. **P5-S, now, 0-GPU.** It is a smoothed-label amendment to arm (ii) with the same rows, controls and estimator.
   - The pre-registration is the prereg's §9 entry dated 2026-10-11, landed with this file.
   - It replaces only the correction's interpolant: natural cubic spline (primary), PCHIP (secondary), and linear as
     the bit-exact control.
   - It passes if Δ sim8 stays separated below 0, no guard is adverse, AND **both** learnability ratios are ≤ 2×.
2. **Then the P5-C and P5-D prereg and config, with launch-gate entries.**
   - P5-C trains the WTA on the compensated label (cubic if P5-S passes).
   - P5-D (drop-if-sim8) is its control.
   - The descriptive "repair only where T fails" selection (registered in §9 as Tsel) is an input to that design.
3. **Blocked, and named:**
   - **P5-C on GPU needs the PI's go, A40-days and a pod.** There is no pod today: SFT-4R is waiting on the same three
     items (`HANDOFF_2026-10-11_SFT4R_READY.md`).
   - **PI decision (a)** now carries the §5 trade-off.
4. **Priority is unchanged.** P1/P2 (target quality) stay first. The student is ~9.6 PDMS below the human, and the
   student's DAC failures are mostly *not* tracker-made (29 % against the teacher's 83 %, P4). P5 is a label lever
   for the remaining part.

## 8 · What this result does NOT license

- ⛔ **Not a PDMS claim.** No model was trained or scored.
- ⛔ **Not a lever claim for the student.** CONFIRMED is necessary for P5-C, never sufficient. A student may not
  learn a pre-distorted label, and §3 says this one is measurably harder to learn.
- ⛔ **Not a ruling on PI decision (a).** §5 supplies evidence for it.
- ⛔ **Not a navtest result.** These are 1,018 navtrain held-out targets over 11 logs.

## Appendix · Full rendered tables

Rendered by `eval/p5_render.py --labels raw/2026-10-11-p5-target-labels/p5_labels.json --families
raw/2026-10-11-p5-target-labels/p5_families.json`, pasted unedited.

n = 1018 rows, 11 logs, 0 rows raised; 468 s.

**Controls (§4)**

| control | result |
|---|---|
| C1_identity | PASS |
| C2_knot_exact | PASS |
| C4_d1_join | PASS |
| C5_review7 | PASS |
| C6_margin_direction | PASS |
| C7_batch_independence | PASS |
| C3_wrong_sign | PASS |

raw: knot_exact True, knot_checks 9162, ident_label_bit_exact True, ident_p8_bit_exact True, ident_sim8_eq_T True, batch_indep_max_m 0.0, margin_neg_not_interp8 0, d1_sim8_mismatch 0, d1_missing 0, r7_dac_T 1018, r7_dac_H 1018, r7_ddc_T 1018, r7_ddc_H 1018, r7_lk_T 1018, r7_lk_H 1018, r7_n 1018

**Per reference, scored on its executed trajectory (§3)**

| ref | intended | n | sim8 fail | interp8 fail | DDC mean | DDC < 1 | lk10 violation | comfort | path-length ratio (median) |
|---|---|---|---|---|---|---|---|---|---|
| T | T | 1018 | 8.06 % | 0.59 % | 0.9818 | 2.75 % | 13.36 % | 100.00 % | 0.9972 |
| Tc20 | T | 1018 | 0.98 % | 12.87 % | 0.9882 | 2.06 % | 11.00 % | 99.90 % | 1.0009 |
| Tc8 | T | 1018 | 0.98 % | 13.16 % | 0.9887 | 1.96 % | 10.90 % | 99.90 % | 1.0009 |
| Tw | T | 1018 | 34.09 % | 35.76 % | 0.8885 | 15.72 % | 17.19 % | 88.41 % | 0.9927 |
| ident | T | 1018 | 8.06 % | 0.59 % | 0.9818 | 2.75 % | 13.36 % | 100.00 % | 0.9972 |
| H | H | 1018 | 0.00 % | 0.79 % | 0.9764 | 3.24 % | 13.95 % | 100.00 % | 1.0089 |
| Hc | H | 1018 | 0.59 % | 14.44 % | 0.9823 | 2.55 % | 11.98 % | 99.51 % | 1.0057 |

**Executed vs intended** (mean / median / p95)

| ref | intended | max_xy | along_4s | mean_abs_along | mean_abs_cross | heading_mae_deg |
|---|---|---|---|---|---|---|
| T | T | 0.473 / 0.414 / 1.021 | -0.075 / -0.131 / 0.667 | 0.170 / 0.140 / 0.411 | 0.121 / 0.065 / 0.392 | 1.645 / 0.801 / 5.792 |
| Tc20 | T | 0.075 / 0.060 / 0.186 | 0.018 / 0.016 / 0.073 | 0.022 / 0.017 / 0.054 | 0.029 / 0.020 / 0.086 | 0.662 / 0.426 / 1.920 |
| Tc8 | T | 0.072 / 0.057 / 0.184 | 0.019 / 0.016 / 0.076 | 0.022 / 0.016 / 0.053 | 0.028 / 0.019 / 0.084 | 0.654 / 0.417 / 1.904 |
| Tw | T | 3.022 / 2.676 / 6.171 | -0.668 / -1.008 / 4.567 | 1.037 / 0.858 / 2.440 | 0.620 / 0.339 / 2.056 | 7.999 / 3.518 / 30.186 |
| ident | T | 0.473 / 0.414 / 1.021 | -0.075 / -0.131 / 0.667 | 0.170 / 0.140 / 0.411 | 0.121 / 0.065 / 0.392 | 1.645 / 0.801 / 5.792 |
| H | H | 0.553 / 0.498 / 1.047 | 0.089 / 0.165 / 0.841 | 0.263 / 0.247 / 0.535 | 0.148 / 0.078 / 0.505 | 1.383 / 0.572 / 5.612 |
| Hc | H | 0.195 / 0.183 / 0.366 | 0.092 / 0.099 / 0.245 | 0.113 / 0.103 / 0.238 | 0.070 / 0.049 / 0.201 | 0.713 / 0.324 / 2.770 |

**Executed vs H (descriptive)**

| ref | max_xy | mean_abs_along | mean_abs_cross | heading_mae_deg |
|---|---|---|---|---|
| T | 2.623 | 1.043 | 0.247 | 2.771 |
| Tc20 | 2.760 | 1.099 | 0.224 | 2.562 |
| Tc8 | 2.760 | 1.099 | 0.224 | 2.557 |

**Arm (ii): Tc20 − T, paired log-cluster bootstrap (§6)** — verdict **CONFIRMED**; adverse guards: none

| metric | Δ [95 % CI] |
|---|---|
| sim8 | -7.07 pp [-13.26, -1.83] |
| ddc | +0.64 pp [+0.00, +1.79] |
| comfort | -0.10 pp [-0.41, +0.00] |
| lk10 | -2.36 pp [-5.49, +0.57] |
| mean_abs_along | -0.1477 [-0.1825, -0.1217] |
| mean_abs_cross | -0.0924 [-0.1248, -0.0529] |
| max_xy | -0.3977 [-0.4575, -0.3373] |
| heading_mae_deg | -0.9839 [-1.3651, -0.5015] |

**Control 3, wrong sign: Tw − T (same estimator)** — verdict **REFUTED**; adverse guards: ddc, comfort, mean_abs_along, mean_abs_cross, max_xy, heading_mae_deg

| metric | Δ [95 % CI] |
|---|---|
| sim8 | +26.03 pp [+15.30, +35.28] |
| ddc | -9.33 pp [-15.58, -3.42] |
| comfort | -11.59 pp [-18.51, -4.97] |
| lk10 | +3.83 pp [-0.15, +6.92] |
| mean_abs_along | +0.8677 [+0.7417, +1.0531] |
| mean_abs_cross | +0.4991 [+0.2888, +0.6678] |
| max_xy | +2.5496 [+2.2143, +2.8999] |
| heading_mae_deg | +6.3535 [+3.6080, +8.5222] |

**Learnability** (flag: **LEARNABILITY RISK**; RISK iff a Tc20 median > 2× t20's)

| label | max acc median | max acc p95 | max heading-rate median °/s | p95 °/s |
|---|---|---|---|---|
| t20 | 1.371 | 3.136 | 5.110 | 32.742 |
| Tc20 | 2.824 | 6.309 | 8.590 | 44.022 |
| Tw | 3.319 | 15.939 | 9.929 | 79.018 |

**D1 class breakdown of what Tc20 repairs and breaks**

| D1 class | n | T sim8 fail | Tc20 sim8 fail | repaired | broken |
|---|---|---|---|---|---|
| pass | 934 | 0 | 2 | 0 | 2 |
| C_navsim_tracker | 78 | 78 | 2 | 76 | 0 |
| A_teacher_path | 6 | 4 | 6 | 0 | 2 |

**Arm (i): margin < 0.6 m on T's plan footprint** — verdict **NOT VIABLE**

- at 0.6 m: flagged 469 (46.07 %), recall 85.37 % (70 of 82 sim8 failures), precision 14.93 %
- drop-if-sim8 comparison point: flagged 8.06 %, recall 100.00 %
- margin of T: median 0.632 m, p05 0.234 m; H median 0.685 m (descriptive)

| threshold m (descriptive sweep) | flagged share | recall | precision |
|---|---|---|---|
| 0.3 | 12.08 % | 29.27 % | 19.51 % |
| 0.45 | 28.00 % | 69.51 % | 20.00 % |
| 0.6 | 46.07 % | 85.37 % | 14.93 % |
| 0.75 | 61.89 % | 97.56 % | 12.70 % |

**Iteration traces** (median max-xy deviation, m, k = 0..3)

| ref | k=0 | k=1 | k=2 | k=3 |
|---|---|---|---|---|
| Tc20 | 0.4136 | 0.1225 | 0.0764 | 0.0606 |
| Tw | 0.4136 | 0.7648 | 1.4261 | 2.6760 |
| ident | 0.4136 | 0.4136 | 0.4136 | 0.4136 |
| Tc8 | 0.4136 | 0.1212 | 0.0739 | 0.0573 |

`max |to_navsim(Tc20) − Tc8|`: median 0.0195 m, max 0.2623 m.

**Dual compliance: executed path passes sim8 AND the drawn 8 poses pass interp8** (descriptive, counted from `rows`)

| ref | n | sim8 fail | interp8 fail | fail both | pass both |
|---|---|---|---|---|---|
| T | 1018 | 82 | 6 | 4 | 934 (91.75 %) |
| Tc20 | 1018 | 10 | 131 | 7 | 884 (86.84 %) |
| Tc8 | 1018 | 10 | 134 | 7 | 881 (86.54 %) |
| Tw | 1018 | 347 | 364 | 335 | 642 (63.06 %) |
| ident | 1018 | 82 | 6 | 4 | 934 (91.75 %) |
| H | 1018 | 0 | 8 | 0 | 1010 (99.21 %) |
| Hc | 1018 | 6 | 147 | 3 | 868 (85.27 %) |
| Tsel (DERIVED: Tc20 where T fails sim8, else T) | 1018 | 6 | 52 | 3 | 963 (94.60 %) |

Tc20 interp8 failures: 131; on rows where T's interp8 passes: 128; on rows where T fails sim8: 50.
Learnability on the 82 rows Tsel compensates (median per-row max): |acc| t20 2.456 vs Tc20 5.019 m/s² (2.04×); |heading rate| t20 26.867 vs Tc20 34.691 °/s (1.29×).

**Four families** (tier T1, n 1018, 11 logs, value ± 95 % half-width; descriptive only). STRATEGIC: UNAVAILABLE in NAVSIM; PAUSED (PI ruling 2026-09-19).

| pair (executed vs intended) | speed MAE m/s | along MAE m | accel MAE m/s² | heading MAE ° | yaw-rate MAE °/s | curvature MAE 1/m | cross MAE m | lat-decision κ | lon-decision κ | goal-point err m | STRATEGIC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exec_T_vs_T | 0.105 ± 0.013 | 0.167 ± 0.027 | 0.139 ± 0.013 | 1.530 ± 0.544 | 2.128 ± 0.714 | 0.0101 ± 0.0040 | 0.120 ± 0.042 | 0.971 [0.959, 0.980] | 0.971 [0.952, 0.992] | 0.429 [0.362, 0.497] | UNAVAILABLE |
| exec_Tc20_vs_T | 0.027 ± 0.003 | 0.025 ± 0.004 | 0.077 ± 0.008 | 0.582 ± 0.156 | 1.160 ± 0.297 | 0.0061 ± 0.0024 | 0.027 ± 0.008 | 0.967 [0.942, 0.977] | 0.992 [0.988, 0.998] | 0.053 [0.041, 0.065] | UNAVAILABLE |
| exec_Tc8_vs_T | 0.025 ± 0.003 | 0.024 ± 0.004 | 0.075 ± 0.008 | 0.575 ± 0.156 | 1.147 ± 0.295 | 0.0060 ± 0.0024 | 0.026 ± 0.007 | 0.967 [0.938, 0.978] | 0.992 [0.988, 0.998] | 0.052 [0.040, 0.064] | UNAVAILABLE |
| exec_Tw_vs_T | 0.606 ± 0.095 | 0.969 ± 0.153 | 0.538 ± 0.057 | 7.000 ± 2.520 | 8.802 ± 3.405 | 0.0367 ± 0.0151 | 0.670 ± 0.255 | 0.747 [0.627, 0.813] | 0.548 [0.448, 0.688] | 2.884 [2.506, 3.316] | UNAVAILABLE |
| exec_T_vs_H | 0.668 ± 0.153 | 1.014 ± 0.217 | 0.394 ± 0.058 | 2.618 ± 0.818 | 2.846 ± 0.817 | 0.0129 ± 0.0046 | 0.309 ± 0.090 | 0.885 [0.826, 0.923] | 0.483 [0.309, 0.671] | 2.592 [1.962, 3.198] | UNAVAILABLE |
| exec_Tc20_vs_H | 0.712 ± 0.154 | 1.053 ± 0.211 | 0.457 ± 0.054 | 2.345 ± 0.666 | 2.866 ± 0.788 | 0.0134 ± 0.0048 | 0.305 ± 0.087 | 0.876 [0.819, 0.914] | 0.479 [0.308, 0.668] | 2.719 [2.069, 3.351] | UNAVAILABLE |
| exec_Tc8_vs_H | 0.712 ± 0.154 | 1.053 ± 0.211 | 0.458 ± 0.054 | 2.340 ± 0.664 | 2.864 ± 0.790 | 0.0134 ± 0.0048 | 0.304 ± 0.087 | 0.872 [0.814, 0.909] | 0.479 [0.308, 0.668] | 2.719 [2.069, 3.350] | UNAVAILABLE |
| exec_H_vs_H | 0.121 ± 0.018 | 0.268 ± 0.034 | 0.128 ± 0.018 | 1.564 ± 0.542 | 1.956 ± 0.549 | 0.0104 ± 0.0035 | 0.145 ± 0.056 | 0.963 [0.941, 0.985] | 0.907 [0.882, 0.929] | 0.486 [0.437, 0.553] | UNAVAILABLE |
| exec_Hc_vs_H | 0.051 ± 0.008 | 0.123 ± 0.026 | 0.100 ± 0.013 | 0.900 ± 0.241 | 1.602 ± 0.399 | 0.0089 ± 0.0031 | 0.062 ± 0.021 | 0.978 [0.964, 0.992] | 0.933 [0.906, 0.954] | 0.156 [0.122, 0.191] | UNAVAILABLE |
