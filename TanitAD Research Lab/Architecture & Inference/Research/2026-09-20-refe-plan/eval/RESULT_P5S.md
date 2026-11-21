<title>P5-S — smoothed labels</title>

# RESULT P5-S: smoothed tracker-compensated labels (2026-10-11)

**Pre-registration:** the P5-S entry in §9 of `eval/PREREG_P5_TRACKER_AWARE_TARGETS.md`. It was landed in `da6fdb2`
with `RESULT_P5.md`, before any P5-S code or number existed.

**Tier:** T1 label test on NAVSIM v1.1's non-reactive pseudo-simulation. **No model is trained or scored.**

**Population:** the same 1,018 held-out rank-0 navtrain teacher targets over 11 logs as P5. Nothing was re-selected,
and 0 rows raised.

**Estimator:** paired log-cluster bootstrap, 10,000 resamples, seed 20260927. It answers *another draw of LOGS*. The
labels and the tracker are deterministic, so there is no training or inference variance to replicate.

**What changed:** only the interpolant that spreads the 8 knot errors onto the 0.2 s label grid.
- `linear` is P5, kept as the control.
- `cubic` (natural spline) is the primary arm; its label is called **Tc20s**.
- `pchip` is the secondary arm.

Every number is MEASURED from `raw/2026-10-11-p5s-smoothed/`. Each section names the file it comes from.
- Gates, PASS bar and paired contrasts come from `p5s_compare.json`, produced by `eval/p5s_compare.py`.
- Per-arm controls, metrics and verdicts come from `p5s_{linear,cubic,pchip}.json`.
- Families come from `p5s_{cubic,pchip}_families.json`.
- The appendix tables are rendered by script, not transcribed.

## 0 · Verdict

**P5-S PASSED on the primary arm (cubic).**

Tc20s meets all three letters of the committed bar:
- **(a)** Δ sim8 fail (Tc20s − T) is **−7.07 pp [−13.26, −1.83]**, the same interval as P5's linear label. It
  repairs the same failures: 82 → 10 rows.
- **(b)** No guard is adverse:
  - ddc +0.69 pp [+0.00, +1.99];
  - comfort −0.10 pp [−0.41, +0.00];
  - lk10 −2.46 pp [−5.58, +0.51];
  - all four executed-vs-intended deviations fall, with intervals that exclude 0.
- **(c)** Learnability is under 2× on both axes:
  - median per-row max |acc| **2.321 vs t20's 1.371 m/s², 1.69×** (P5 linear: 2.06×);
  - median max |heading rate| **8.978 vs 5.110 °/s, 1.76×** (P5 linear: 1.68×).

**The secondary arm (pchip) also passes**, at 1.82× and 1.79×. Under §9, cubic's PASS decides the outcome, so
**Tc20s (cubic) is the label P5-C pre-registers.**

## 1 · The amendment's own gates (all passed; each able to fail)

| gate | result |
|---|---|
| **C8**: the `linear` arm reproduces landed P5 exactly | **PASS.** 0 mismatching fields in the JSON (`seconds` stripped), max float diff 0. All 13 NPZ arrays are `array_equal`. |
| **C2′**: the spline arms still pass through the knots | **PASS.** Max knot deviation is 4.44e-16 for both cubic and pchip (bar ≤ 1e-9). |
| **C2′ deliberate regression**: `--mutate knot --interp cubic` | **FAILED, as it must.** Max deviation 0.257 m, and the log reads `ZZP5_CONTROLS FAIL`. |
| **C3**: the wrong-sign Tw under cubic | **REFUTED, as it must be.** Six guards are adverse. |
| P5's §4 controls (C1, C2, C4–C7), every arm | all PASS |

C8 is what makes the comparison clean. The code change cannot have moved anything except the interpolant, because
the linear path reproduces the landed result bit for bit.

## 2 · What smoothing did, and what it did not

| label | max \|acc\| median (ratio to t20) | p95 | max \|heading rate\| median °/s (ratio) | p95 | sim8 fail | interp8 fail |
|---|---|---|---|---|---|---|
| t20 (teacher) | 1.371 | 3.136 | 5.110 | 32.742 | — | — |
| Tc20, linear (P5) | 2.824 (2.06×) | 6.309 | 8.590 (1.68×) | 44.022 | 10 | 131 |
| **Tc20s, cubic** | **2.321 (1.69×)** | **4.780** | 8.978 (1.76×) | 45.574 | **10** | 133 |
| Tc20, pchip | 2.497 (1.82×) | 5.353 | 9.151 (1.79×) | 45.598 | 9 | 133 |

**The P5 mechanism hypothesis is only partly supported.**
- The hypothesis was that the excess acceleration comes from slope jumps at the knots.
- The cubic spline removes those jumps, but only **35 %** of the acceleration excess over the teacher goes away:
  1.453 → 0.950 m/s². pchip removes 23 %.
- So the knot kinks were a real part of the excess, not most of it. **The rest belongs to the compensation itself.**
  Inverting the tracker genuinely asks for larger accelerations than the teacher's plan.

**Heading rate moved the other way.** It rose 4.5 % under cubic and 6.5 % under pchip, and it stays under the bar.
- HYPOTHESIS, not tested: the spline overshoots between knots in the lateral component.
- This is the axis to watch if P5-C's student struggles.

**The tracker repair is unchanged.**
- The Tc20s − Tc20 paired contrast on sim8 is +0.00 pp [+0.00, +0.00] for cubic: the same 10 rows fail.
- interp8 is +0.20 pp [+0.00, +0.49]: two more rows fail, 133 against 131.

## 3 · The interp8 cost and the selective label (descriptive, no verdict)

Like P5's linear label, Tc20s pays for the tracker repair in interp8. NAVSIM's own interpolation of the 8 drawn poses
fails on **133 rows (13.06 %)** against T's 6. So dual compliance (sim8 AND interp8 pass) is lower for the full
label: **86.64 % against T's 91.75 %**.

The DERIVED selective label **Tsel-s** uses Tc20s only where T fails sim8, and T elsewhere.
- Its dual compliance is **94.40 %**, against linear Tsel's 94.60 % in P5.
- On its 82 compensated rows, Tsel-s is now under 2× on both axes:
  - |acc| t20 2.456 vs Tc20s 3.665 m/s² (**1.49×**, against linear Tsel's 2.04× in P5);
  - |heading rate| 26.867 vs 35.827 °/s (**1.33×**).

⛔ **Tsel-s was not a pre-registered arm.** It is an input to the P5-C design, never a verdict.

## 4 · Four families (T1, descriptive only)

The families come from `p5s_cubic_families.json`, using the same estimator as P5 with 2,000 resamples.
STRATEGIC is **UNAVAILABLE** in NAVSIM and **PAUSED** by the PI's 2026-09-19 ruling.

Executing Tc20s instead of T, measured against the teacher's intent T:

| family | metric | executing T | executing Tc20s |
|---|---|---|---|
| LONGITUDINAL | speed MAE | 0.105 ± 0.013 m/s | **0.026 ± 0.003** |
| LONGITUDINAL | along-track MAE | 0.167 ± 0.027 m | **0.024 ± 0.004** |
| LATERAL | heading MAE | 1.530 ± 0.544° | **0.577 ± 0.156** |
| LATERAL | yaw-rate MAE | 2.128 ± 0.714 °/s | **1.151 ± 0.296** |
| LATERAL | curvature MAE | 0.0101 ± 0.0040 1/m | **0.0061 ± 0.0024** |
| LATERAL | cross-track MAE | 0.120 ± 0.042 m | **0.027 ± 0.008** |
| TACTICAL | lon-decision κ | 0.971 [0.952, 0.992] | 0.992 [0.988, 0.998] |
| TACTICAL | lat-decision κ | 0.971 [0.959, 0.980] | 0.966 [0.935, 0.977] |
| TACTICAL | goal-point error | 0.429 [0.362, 0.497] m | **0.052 [0.040, 0.065]** |

**Against the human H, executing Tc20s is about the same as executing T.**
- Along-track MAE is 1.053 against 1.014 m, and heading MAE is 2.341° against 2.618°.
- This is expected. The label is faithful to the teacher's intent, so it cannot be more faithful to the human than
  the teacher is.

## 5 · Next lever (RULE ZERO)

1. **P5-C and P5-D: pre-registration and config, with launch-gate entries. This is 0-GPU and unblocked, so it is next.**
   - P5-C fine-tunes the WTA student on **Tc20s (cubic)**.
   - P5-D (drop-if-sim8) is its control.
   - The design choice the prereg must commit to is the **full Tc20s label vs the selective Tsel-s label**:
     - full Tc20s: 86.64 % dual compliance;
     - Tsel-s: 94.40 % dual compliance, 1.49× acc on its compensated rows.
2. **Blocked, and named:**
   - **P5-C on GPU needs the PI's go, A40-days and a pod.** No pod exists today. SFT-4R is waiting on the same
     three items: the old pod's state, a new pod with a write-scoped HF token, and `LANE_QUEUE_SRC`.
3. **Priority is unchanged.** P1/P2 (target quality) stay ahead of P5 (`RESULT_P5.md` §7).
   - The student sits ~9.6 PDMS below the human.
   - Only 29 % of its DAC failures are tracker-made.
   - P5 is a label lever for that part.

## 6 · What this does NOT license

- ⛔ **Not a PDMS claim, and not a student-lever claim.** Passing the learnability gate is necessary for P5-C, not
  sufficient. Whether a student can learn Tc20s is P5-C's question.
- ⛔ **The 2× learnability bar is a proxy.** It compares label smoothness with the teacher's. It is not a measured
  student-learnability limit.
- ⛔ **Not a ruling on PI decision (a).** Tc20s still embeds the tracker into the label.
- ⛔ **Not a navtest result.** These are 1,018 navtrain held-out targets over 11 logs.

## Appendix A · Gates, PASS bar and paired contrasts (`eval/p5s_compare.py`)

In this appendix, the "Tc20" rows of the cubic and pchip learnability tables are that arm's smoothed label.

n = 1018 rows, 11 logs.

**Amendment gates**

| gate | result | detail |
|---|---|---|
| C8 linear reproduces landed P5 (JSON, `seconds` stripped) | PASS | 0 mismatching fields; max float diff 0 |
| C8 linear reproduces landed P5 (NPZ, 13 arrays) | PASS | all array_equal |
| C2′ cubic knot max dev ≤ 1e-9 | PASS | 4.44e-16 |
| C2′ pchip knot max dev ≤ 1e-9 | PASS | 4.44e-16 |
| C2′ deliberate regression (`--mutate knot --interp cubic`) must FAIL | PASS (it failed) | max dev 0.257 m |

**PASS bar per arm (committed)**

| arm | §4 gates | arm (ii) verdict | (a) Δ sim8 Tc20s − T | (b) adverse guards | (c) acc median ratio | (c) heading-rate median ratio | C3 Tw | PASS |
|---|---|---|---|---|---|---|---|---|
| cubic | all PASS | CONFIRMED | -7.07 pp [-13.26, -1.83] | none | 2.321 / 1.371 = 1.69× | 8.978 / 5.110 = 1.76× | REFUTED | **PASS** |
| pchip | all PASS | CONFIRMED | -7.17 pp [-13.50, -1.83] | none | 2.497 / 1.371 = 1.82× | 9.151 / 5.110 = 1.79× | REFUTED | **PASS** |
| linear (= landed Tc20, reference) | all PASS | CONFIRMED | -7.07 pp [-13.26, -1.83] | none | 2.824 / 1.371 = 2.06× | 8.590 / 5.110 = 1.68× | REFUTED | FAIL c |

**Outcome: P5-S PASSED (primary, cubic)**

**Descriptive, cubic**

| paired contrast | Δ [95 % CI] |
|---|---|
| Tc20s-Tc20 sim8 | +0.00 pp [+0.00, +0.00] |
| Tc20s-Tc20 interp8 | +0.20 pp [+0.00, +0.49] |
| Tc20s-T interp8 | +12.48 pp [+4.37, +21.47] |

| ref | n | sim8 fail | interp8 fail | fail both | pass both |
|---|---|---|---|---|---|
| T | 1018 | 82 | 6 | 4 | 934 (91.75 %) |
| Tc20 | 1018 | 10 | 131 | 7 | 884 (86.84 %) |
| Tc20s | 1018 | 10 | 133 | 7 | 882 (86.64 %) |
| Tsel-s (DERIVED) | 1018 | 6 | 54 | 3 | 961 (94.40 %) |

Tsel-s learnability on its 82 selected rows (median per-row max): |acc| t20 2.456 vs Tc20s 3.665 m/s² (1.49×); |heading rate| t20 26.867 vs Tc20s 35.827 °/s (1.33×).

| label | max acc median | max acc p95 | max heading-rate median °/s | p95 °/s |
|---|---|---|---|---|
| t20 | 1.371 | 3.136 | 5.110 | 32.742 |
| Tc20 | 2.321 | 4.780 | 8.978 | 45.574 |
| Tw | 1.795 | 8.213 | 10.176 | 81.173 |

| arm (ii) guard, Tc20s − T | Δ [95 % CI] |
|---|---|
| sim8 | -7.07 pp [-13.26, -1.83] |
| ddc | +0.69 pp [+0.00, +1.99] |
| comfort | -0.10 pp [-0.41, +0.00] |
| lk10 | -2.46 pp [-5.58, +0.51] |
| mean_abs_along | -0.1481 [-0.1830, -0.1221] |
| mean_abs_cross | -0.0930 [-0.1256, -0.0534] |
| max_xy | -0.3998 [-0.4593, -0.3393] |
| heading_mae_deg | -0.9889 [-1.3718, -0.5051] |

**Descriptive, pchip**

| paired contrast | Δ [95 % CI] |
|---|---|
| Tc20s-Tc20 sim8 | -0.10 pp [-0.25, +0.00] |
| Tc20s-Tc20 interp8 | +0.20 pp [+0.00, +0.49] |
| Tc20s-T interp8 | +12.48 pp [+4.37, +21.47] |

| ref | n | sim8 fail | interp8 fail | fail both | pass both |
|---|---|---|---|---|---|
| T | 1018 | 82 | 6 | 4 | 934 (91.75 %) |
| Tc20 | 1018 | 10 | 131 | 7 | 884 (86.84 %) |
| Tc20s | 1018 | 9 | 133 | 6 | 882 (86.64 %) |
| Tsel-s (DERIVED) | 1018 | 5 | 54 | 2 | 961 (94.40 %) |

Tsel-s learnability on its 82 selected rows (median per-row max): |acc| t20 2.456 vs Tc20s 4.306 m/s² (1.75×); |heading rate| t20 26.867 vs Tc20s 36.018 °/s (1.34×).

| label | max acc median | max acc p95 | max heading-rate median °/s | p95 °/s |
|---|---|---|---|---|
| t20 | 1.371 | 3.136 | 5.110 | 32.742 |
| Tc20 | 2.497 | 5.353 | 9.151 | 45.598 |
| Tw | 2.284 | 11.065 | 10.543 | 84.020 |

| arm (ii) guard, Tc20s − T | Δ [95 % CI] |
|---|---|
| sim8 | -7.17 pp [-13.50, -1.83] |
| ddc | +0.69 pp [+0.00, +1.99] |
| comfort | -0.10 pp [-0.41, +0.00] |
| lk10 | -2.46 pp [-5.58, +0.51] |
| mean_abs_along | -0.1481 [-0.1830, -0.1221] |
| mean_abs_cross | -0.0930 [-0.1256, -0.0534] |
| max_xy | -0.3999 [-0.4594, -0.3393] |
| heading_mae_deg | -0.9891 [-1.3727, -0.5050] |

## Appendix B · Full per-arm renders

`raw/2026-10-11-p5s-smoothed/render_cubic.md` and `render_pchip.md` are the `p5_render.py` output for each smoothed arm, with the four families included. In those files, **"Tc20" means that arm's smoothed label** (Tc20s for cubic), and "Tw" is the wrong-sign control built with the same interpolant.
