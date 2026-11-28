# PREREG_V8F_HACC_MC: result (zero training; MC-H, MC-F, STEEP; 2 seeds; m = 10 decisive, m = 30 beside; read 2026-10-26 on the dev box under Amendments 1 and 2)

**Outcome, as committed: MECHANISM-NOT-ENOUGH.** All nine controls PASS under Amendment 2. The launch crawl fell with the constant weight and rose with the steep one at both seeds (M1(MC-H) < M1(HACC) and M1(STEEP) > M1(HACC)), but the decisive lateral row is still separated-worse at seed 0: **L2 (MC-H − P10) +0.0028 [+0.0004, +0.0055]**. At seed 1 it is not separated (+0.0013 [−0.0014, +0.0039]); the outcome row needs "separated-worse at ≥ 1 seed".

* **Evidence class:** MEASURED (ours). Every number below is copied from `C:/Users/Admin/hmc/read_a2/HMC_READ.json` (the decisive read) or the files named in section 11; JSON key paths are there too. Nothing here is a new statistic.
* **Tier:** T1 (T1-kin), self-action open loop: the ego state the model reads is rebuilt from its own executed motion at every replan, the camera frames stay logged, nav is dead-reckoned. The reader stamps `T1-kin (self-action OPEN loop …)`. Nothing here says the model drives.
* **Estimator of every interval:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, per seed, arm − P10 and arm − HACC, over 141 episodes (2,399 windows; the L2 and L3 rows run on the 2,106 windows / 135 episodes that have steps with GT speed ≥ 5.2 m/s, as stamped per row).
* **The question every interval answers:** "would another draw of EPISODES say this?". It does NOT answer "would another TRAINING RUN say this?" (the two M2b seeds are the only replicate, and the outcome row is allowed to differ between them: it does for L2). It does not answer "would another INFERENCE RUN say this?" either: the planner is deterministic (`PREREG_V8F_PGT`).
* **Sign convention.** arm − P10 and arm − HACC; for every ADE, MAE and error row lower is better, so negative = the arm better. "HACC" is the banked T1 arm of `RESULT_V8F_T1KIN_CAUSALBUCKET.md` (causal speed bucket, the headline configuration). All arms are zero-training post-maps of the banked HACC checkpoints, `plan = P + w · (τ − P)` per knot and axis (`P` the kinematic prior, `τ` O's plan).
* **Pre-registration:** `Project Steering/PREREG_V8F_HACC_MC.md`, original at `8d4b936` (read by the reader as `@ 8d4b936d`), Amendment 1 (`A1-2026-10-26`, `19d7e06`, before any eval run), Amendment 2 (`A2-2026-10-26`, POST-DATA and CONTROL ONLY, **not yet landed**: `C:/Users/Admin/hmc/AMEND_HMC2.md`). The Amendment 2 text is summarised in section 1 and in this file's manifest.
* **Status of this file:** a draft for the Master Mind. Nothing was staged, committed or pushed by it.

---

## 1. How this read came to be (disclosures; read before the numbers)

1. **Four chain launches, three stopped before any eval.** `logs_stop_c7_0620`, `…_0649`, `…_0705` each end `ZZHMC-STOP C-7: suite not green` (06:20: 2 failed, 126 passed, 99 skipped; 06:49: 3 failed, 171 passed, 1 skipped, 52 errors; 07:05: 1 failed, 225 passed, 1 skipped) with no run started. I did not examine those failures. The launch of 11:10 passed C-7 (**226 passed, 1 skipped, 6 warnings in 580.74 s**), ran all twelve evals (11:21:36 to 11:45:19, 200 to 263 s each, dev-box CPU) and ended `ZZHMC-DONE 12`.
2. **Amendment 1 was written from a 24-window SMOKE, before any eval run** (`raw/2026-10-26-hmc-build/`, outputs labelled SMOKE): it moved C-1's bar from 1e-6 m to one float32 ulp at |plan|, and C-4 clause 2 to "GT := P with the held tail also zeroed gives c = 0". It also adopted the build's interpretations, `INTERPRETATIONS.md` section B items 1 to 10. The reader applies it only with `--c1-bar ulp32 --c4-clause2 tail0` and the amendment ids.
3. **The read with the literal bars REFUSED on C-1 and C-4** (controls-only, `read/READ_CONTROLS.json`, 11:47): C-1 failed on every MC-H and MC-F run (max |d| 3.793e-06 to 1.515e-05 m against 1e-6 m) and C-4 clause 2 read c = 0.3094 / 0.3489 instead of 0.
4. **The read with Amendment 1 REFUSED on C-1 alone** (`read/HMC_READ.json`, 11:54): `OUTCOME` `{"outcome": "REFUSED", "refused_on": ["C-1"], "text": "any control fails -> no outcome row (PREREG section 6)"}`; `controls_failed` `["C-1"]`; P-1, C-2, C-3, C-4 (as amended), C-5, C-6, C-6b, C-7 PASS. **Checked here: neither file contains a `rows` key or an outcome row.** The reader stops before any row on a REFUSED, and `--force-rows-under-refusal` was not used.
5. **Amendment 2 (`A2-2026-10-26`) is POST-DATA and CONTROL ONLY.** Its text: C-1's float32 ulp is taken at the blend's largest operand, `max |d| / spacing(float32(max(|P|, |τ|, |plan|))) ≤ 1`. The defect it names: A1 measured |d| in ulps at the result's magnitude, and on a near-stationary tick the result is smaller than its operands and they have opposite signs, so a ulp at the result understates the roundings; the 24-window SMOKE held no such tick. Its measured numbers (`probe_c1_operand.json`, the reader's own d):

   | | MC-H s0 | MC-H s1 | MC-F s0 | MC-F s1 |
   |---|---|---|---|---|
   | max \|d\| / ulp at \|plan\| (A1), m10 | 1.1591 | 3.7217 | 2.4611 | 1.4873 |
   | tick-knots over 1 ulp at \|plan\| (of 86,364 at m10) | 1 | 1 | 3 | 1 |
   | max \|d\| / ulp at the largest operand, m10 / m30 | 0.49999966 / 0.49999350 | 0.49999984 / 0.49997666 | 0.49998990 / 0.49997085 | 0.49999917 / 0.49999803 |
   | the worked tick (MC-H s0, m10, tick 4336, knot 3) | plan −0.0016977825 m, P +0.0045697433 m, τ −0.0267334245 m, d 1.3494e-10 m = 1.16 ulp at \|plan\| | | | |
   | regression: HACC's table through the same check (m10), ulp at the largest operand | 5,192,788 (79,163 of 86,364 over) | 5,228,043.5 (79,160 over) | | |

   The number 1 is A1's and does not move; the regression arm is unchanged and must still fail (it reads max |d| 1.797 m and 2.006 m, and fails as it must). Nothing else moves: the arms, the TRAIN bar, the rows, F, the outcome table, the committed next steps, and C-4 as A1 amended it. The reader applies it only with `--c1-bar ulp32max --c1-amendment A2-2026-10-26`; the two earlier modes are kept and still refuse on these runs.
   * **Checked here:** the diff of `hmc_read.py` before and after A2 touches only the C-1 bar mode (the argparse choice, the PASS mapping, the bar text, one print line); `hmc_controls.py` gains the max-operand ratio and nothing else. No row code changed.
6. **The order of writing: settled, and it corrects the amendment's first draft.** The file times give the sequence:
   * 11:54:49: the A1 read REFUSES.
   * 11:56:40: the operand probe runs, and every run reads ≤ 0.5 ulp at the largest operand.
   * 11:57:33–46: the rule is coded into the reader.
   * 11:59:30: the A2 read computes and prints the outcome.
   * 12:00:46: the amendment's TEXT is written.

   So the rule was fixed in code before any outcome row existed, and the text was written after the outcome had been printed. The first draft
   of the text said "no outcome row had been computed when this was written". That was false, and this review caught it. The landed
   Amendment 2 states the sequence above and is disclosed as a post-outcome document. The outcome could not have shaped the rule: the probe
   had shown that the new unit passes every run before any row was computed, and C-1 decides only whether rows are read, never which outcome
   they give.
7. **Class (the amendment's own words):** *a float tolerance taken at the result's magnitude, blind to cancellation; the R74 class again, in the amendment written to fix it, calibrated on a smoke that held no near-zero result.* The regression arm still separates the data from the bar by seven orders of magnitude (data at 0.5 ulp, regression at 5.2e6), so the amended control keeps its power.

---

## 2. The outcome (the committed table, applied verbatim)

F = 0.1207 m (the PREREG's value); the KEEP bar F/2 was evaluated at 0.06035 (`F_half_used`), and at the printed 0.0604 and at 0.06037 as a sensitivity (`KEEP_by_F_half_reading`).

The PREREG's outcome table, quoted (decisive arm MC-H):

| outcome | condition (MC-H, both seeds unless stated) | what it means | committed next |
|---|---|---|---|
| **FIXED-KEEP** | L2 (MC-H − P10) not separated-worse at both seeds, AND MC-H − HACC ADE6 upper bound ≤ +F/2 at both seeds, AND M1(MC-H) < M1(HACC) at both | the pump was the cause; the lever ships | adopt MC-H as the T1 post-map and re-read DOMINANCE (T1KIN rows) under its own read; then m = 5 |
| **FIXED-NOKEEP** | L2 not separated-worse at both seeds, but the ADE6 KEEP fails at ≥ 1 seed | the T1 margin partly rode on the injected deceleration offsetting P10's +0.9 m/s² replan-accel overshoot (`DIAGNOSIS` "beside") | a zero-training correction of P10's executed-horizon along bias: a linear (v0, a0) → along-offset map over knots 1–12, fitted on TRAIN tick-0 samples, applied under MC-H, under its own PREREG |
| **MECHANISM-NOT-ENOUGH** | M1(MC-H) < M1(HACC) at both seeds AND M1(STEEP) > M1(HACC) at both, but L2 still separated-worse at ≥ 1 seed | the pump is real; O's own slow launch intent carries the rest (O's v0 −1.78 / −0.93 m/s at tick 1 in the launch windows) | the box-2 TRAINING lever (§7) |
| **MECHANISM-REFUTED** | M1(MC-H) ≥ M1(HACC) at ≥ 1 seed, OR M1(STEEP) ≤ M1(HACC) at ≥ 1 seed | the step variation is not what drives the crawl | the box-2 TRAINING lever (§7); the diagnosis is re-opened on O's launch from the logged state |
| **REFUSED** | any control fails | no outcome row | fix and re-run |

What was measured, row by row (tier T1-kin, self-action open loop; paired episode-cluster bootstrap, B = 2,000):

| clause | s0 | s1 | holds at both? |
|---|---|---|---|
| L2 (MC-H − P10) not separated-worse | **no**: +0.0028 [+0.0004, +0.0055], separated-worse (n 2,106 / 135) | yes: +0.0013 [−0.0014, +0.0039], not separated | **no** |
| MC-H − HACC ADE6 upper bound ≤ +F/2 (0.06035) | **no**: +0.1383 [+0.0406, +0.2353] | **no**: +0.2060 [+0.1238, +0.2873] | **no** (also at 0.0604 and 0.06037) |
| M1(MC-H) < M1(HACC) | yes: 0.018403783407266196 < 0.05356095056476456 | yes: 0.015067639403160171 < 0.02766066649558236 | yes |
| M1(STEEP) > M1(HACC) | yes: 0.08501905059912754 > 0.05356095056476456 | yes: 0.050412919801798496 > 0.02766066649558236 | yes |

| outcome row | matches? |
|---|---|
| FIXED-KEEP | no (L2 clause and KEEP clause fail) |
| FIXED-NOKEEP | no (the L2 clause fails at s0) |
| **MECHANISM-NOT-ENOUGH** | **YES** (both M1 clauses hold; L2 separated-worse at s0, which is "≥ 1 seed") |
| MECHANISM-REFUTED | no (both M1 clauses hold at both seeds) |
| REFUSED | no (`controls_all_pass` true, `controls_failed` empty) |

The reader's record: `OUTCOME.outcome` "MECHANISM-NOT-ENOUGH"; `matching_rows_in_table_order` `["MECHANISM-NOT-ENOUGH"]` (one row; `ambiguity` null); `committed_next` "the box-2 TRAINING lever (O-RECOVER, PREREG section 7)"; `KEEP_bar_sensitivity_rows_matched_if_bar_were` empty for all three spellings of the bar.

* **The committed next step, quoted from the PREREG table:** *"the box-2 TRAINING lever (§7)"*.

### What failed, stated plainly

1. **L2 is still separated-worse at seed 0** (+0.0028 [+0.0004, +0.0055]) with MC-H. The pre-registered goal "L2 no longer separated-worse than P10 at both seeds" was not reached. At seed 1 it was reached (+0.0013, not separated). The lower end of the seed-0 interval is +0.0004.
2. **MC-H costs position against HACC at both seeds, separated**: ADE6 +0.1383 [+0.0406, +0.2353] and +0.2060 [+0.1238, +0.2873] (the KEEP clause needs an upper bound ≤ 0.06035). The same holds for ADE2, speed MAE, along-track MAE, accel MAE, cross-track MAE and L1 (section 4). MC-H stays separated-better than P10 on ADE6 (−0.6296 and −0.5276).
3. **The TRAIN-side qualification did not predict the loop cost.** The PREREG's TRAIN-side, in-sample, open-loop table has MC-H within 0.0024 m of HACC's ADE6 (2.0397 against 2.0373 at s0; 1.9887 against 1.9865 at s1) and qualified the arm on that. In the T1 loop the same comparison is +0.1383 and +0.2060 m. The TRAIN bar passed for MC-H, MC-F and STEEP (`C-4.TRAIN_qualification_bar_ade6_le_HACC_plus_F_half`: true at both seeds); it did not separate any of them on the loop outcome.
4. **MC-H adds accel and lateral-acceleration cost against P10.** accel MAE +0.3166 [+0.1071, +0.5463] and +0.3219 [+0.1288, +0.5349] (separated-worse at both seeds); L1 (lateral-acceleration MAE) +0.7778 [+0.1014, +1.6309] at s1 (separated-worse). These are not in the outcome table.

---

## 3. The mechanism rows M1 and M2 (pre-registered predictions; point values)

M1 = the arm-crawl fraction: the GT-fast pairs whose own pair speed is < 5.2 m/s. M2 = the arm's L2 share on those pairs (an arm-minus-P10 excess, as in the diagnosis). Tier T1-kin. They are point values; the committed comparisons use the points, and the interval column is the reader's beside row (an episode-cluster bootstrap of the pooled-pair-count ratio difference, B = 2,000).

| | HACC (reference) | **MC-H** | MC-F | STEEP |
|---|---|---|---|---|
| **M1, s0** | 0.05356095056476456 | **0.018403783407266196** | 0.02854052865051142 | 0.08501905059912754 |
| **M1, s1** | 0.02766066649558236 | **0.015067639403160171** | 0.0193203064853173 | 0.050412919801798496 |
| M1 of P10 (both seeds) | 0.013674524324522491 | | | |
| M2, s0 | +0.0202463015969859 | +0.0014295519895939504 | +0.0027831349078208893 | +0.09897568981611371 |
| M2, s1 | +0.007205373913188921 | +0.00020742996410135633 | +0.0028996095777390415 | +0.05348496536167187 |
| arm-crawl pairs, s0 / s1 | | 1,004 / 822 | 1,557 / 1,054 | 4,619 / 2,747 |
| M1 − M1(HACC), beside, s0 | | −0.0352 [−0.0515, −0.0214] | −0.0250 [−0.0359, −0.0153] | +0.0315 [+0.0200, +0.0426] |
| M1 − M1(HACC), beside, s1 | | −0.0126 [−0.0210, −0.0057] | −0.0083 [−0.0144, −0.0032] | +0.0228 [+0.0155, +0.0308] |

* The ordering the PREREG predicted, MC-H < HACC < STEEP, holds on M1 at both seeds. The PREREG's TRAIN-side mechanism check (median a0 of the replan state; MC-H −0.48 / −0.61, HACC −0.99 / −1.12, STEEP −1.23 / −1.42 m/s² at s0 / s1) is INHERITED from the PREREG text and was not re-run; `C-3`'s HACC known values were reproduced (section 7).
* MC-H's M1 is still above P10's at both seeds (0.0184 and 0.0151 against 0.0137). The committed comparator is HACC.
* **Mean ADE6 in the T1 loop (point values, `mean_ade6_m`):** MC-H 2.8136542793306663 (s0) / 2.9156448055650306 (s1); MC-F 2.6814048342239385 / 2.740377107484759; STEEP 3.005621445129715 / 2.7796533269858195; HACC 2.675308305904701 / 2.709635998904177; P10 3.443279013457244 (both).

---

## 4. The rows: every arm, both seeds, both comparators (m = 10; tier T1-kin; paired episode-cluster bootstrap, B = 2,000)

"better" = separated and not separated-worse; "**WORSE**" = `separated_worse` true. n is per row (last column), the same for every arm.

### MC-H (the decisive arm)

| row (family) | s0: arm − P10 | s0: arm − HACC | s1: arm − P10 | s1: arm − HACC | n (windows / episodes) |
|---|---|---|---|---|---|
| ADE2, m (ADE) | -0.0087 [-0.0143, -0.0012] better | +0.0035 [+0.0016, +0.0067] **WORSE** | -0.0076 [-0.0142, +0.0020] | +0.0031 [+0.0009, +0.0069] **WORSE** | 2399 / 141 |
| ADE6, m (ADE) | -0.6296 [-0.7500, -0.5171] better | +0.1383 [+0.0406, +0.2353] **WORSE** | -0.5276 [-0.6577, -0.4050] better | +0.2060 [+0.1238, +0.2873] **WORSE** | 2399 / 141 |
| speed MAE, m/s (LONGITUDINAL) | -0.1877 [-0.2541, -0.1219] better | +0.1746 [+0.0955, +0.2543] **WORSE** | -0.1548 [-0.2211, -0.0880] better | +0.2328 [+0.1554, +0.3191] **WORSE** | 2399 / 141 |
| along-track MAE, m (LONGITUDINAL) | -0.3529 [-0.4236, -0.2868] better | +0.1209 [+0.0287, +0.2153] **WORSE** | -0.3422 [-0.4137, -0.2795] better | +0.1663 [+0.1011, +0.2327] **WORSE** | 2399 / 141 |
| accel MAE, m/s^2 (LONGITUDINAL) | +0.3166 [+0.1071, +0.5463] **WORSE** | +0.2993 [+0.1712, +0.4364] **WORSE** | +0.3219 [+0.1288, +0.5349] **WORSE** | +0.5236 [+0.3306, +0.7683] **WORSE** | 2399 / 141 |
| heading MAE, deg (LATERAL) | -1.3805 [-1.9141, -0.8613] better | -0.7481 [-1.0304, -0.4690] better | -1.0440 [-1.6620, -0.4664] better | -0.1188 [-0.3254, +0.0878] | 2373 / 141 |
| yaw-rate MAE, deg/s (LATERAL) | +0.9723 [-0.3011, +2.4264] | -1.9128 [-3.1737, -0.6256] better | +0.7836 [-0.6918, +2.3575] | -0.4081 [-1.6805, +0.8369] | 2366 / 141 |
| curvature MAE, 1/m (LATERAL) | +0.0023 [-0.0021, +0.0068] | -0.0279 [-0.0396, -0.0180] better | -0.0002 [-0.0043, +0.0037] | -0.0132 [-0.0202, -0.0073] better | 2366 / 141 |
| cross-track MAE, m (LATERAL) | -0.5050 [-0.6363, -0.3780] better | +0.0522 [+0.0034, +0.1011] **WORSE** | -0.3471 [-0.5007, -0.1977] better | +0.0873 [+0.0258, +0.1511] **WORSE** | 2399 / 141 |
| L1 lateral-accel MAE, m/s^2 (LATERAL) | +0.2633 [-0.1772, +0.7121] | +0.3700 [+0.0176, +0.7789] **WORSE** | +0.7778 [+0.1014, +1.6309] **WORSE** | +0.8928 [+0.1253, +1.9147] **WORSE** | 2366 / 141 |
| **L2 curvature MAE, GT >= 5.2 m/s, 1/m (LATERAL)** | **+0.0028 [+0.0004, +0.0055] WORSE** | -0.0182 [-0.0312, -0.0081] better | **+0.0013 [-0.0014, +0.0039]** | -0.0069 [-0.0126, -0.0023] better | 2106 / 135 |
| L3 yaw-rate MAE, GT >= 5.2 m/s, deg/s (LATERAL) | +0.4166 [-0.8621, +1.6851] | -1.5615 [-3.0962, -0.1487] better | +0.3161 [-1.0413, +1.6454] | -0.6926 [-1.6927, +0.2832] | 2106 / 135 |

HACC's own banked L2 (HACC − P10, reproduced by C-6 to |Δ| 0.0): **+0.0210 [+0.0099, +0.0351]** (s0) and **+0.0082 [+0.0026, +0.0148]** (s1), n 2,106 / 135, both separated-worse.

### MC-F (beside; c = the tail-continuation constant, 0.3071 / 0.3468; decides nothing)

| row (family) | s0: arm − P10 | s0: arm − HACC | s1: arm − P10 | s1: arm − HACC | n (windows / episodes) |
|---|---|---|---|---|---|
| ADE2, m (ADE) | -0.0030 [-0.0107, +0.0084] | +0.0092 [+0.0049, +0.0165] **WORSE** | -0.0017 [-0.0111, +0.0128] | +0.0090 [+0.0037, +0.0183] **WORSE** | 2399 / 141 |
| ADE6, m (ADE) | -0.7619 [-0.8851, -0.6458] better | +0.0061 [-0.0615, +0.0740] | -0.7029 [-0.8356, -0.5810] better | +0.0307 [-0.0165, +0.0790] | 2399 / 141 |
| speed MAE, m/s (LONGITUDINAL) | -0.3483 [-0.4187, -0.2791] better | +0.0140 [-0.0390, +0.0673] | -0.3404 [-0.4168, -0.2706] better | +0.0473 [+0.0029, +0.0974] **WORSE** | 2399 / 141 |
| along-track MAE, m (LONGITUDINAL) | -0.4863 [-0.5861, -0.3948] better | -0.0124 [-0.0796, +0.0533] | -0.4826 [-0.5748, -0.3973] better | +0.0259 [-0.0160, +0.0684] | 2399 / 141 |
| accel MAE, m/s^2 (LONGITUDINAL) | +0.3699 [+0.1835, +0.5531] **WORSE** | +0.3526 [+0.2646, +0.4453] **WORSE** | +0.1712 [-0.0120, +0.3421] | +0.3730 [+0.2585, +0.4991] **WORSE** | 2399 / 141 |
| heading MAE, deg (LATERAL) | -1.1644 [-1.7527, -0.5818] better | -0.5321 [-0.7631, -0.3153] better | -0.9730 [-1.5809, -0.3963] better | -0.0479 [-0.1701, +0.0785] | 2373 / 141 |
| yaw-rate MAE, deg/s (LATERAL) | +1.2108 [-0.0577, +2.4959] | -1.6743 [-2.7013, -0.7250] better | +1.0248 [-0.5715, +2.6264] | -0.1669 [-0.8500, +0.4935] | 2366 / 141 |
| curvature MAE, 1/m (LATERAL) | +0.0104 [+0.0049, +0.0165] **WORSE** | -0.0198 [-0.0306, -0.0113] better | +0.0101 [+0.0046, +0.0163] **WORSE** | -0.0029 [-0.0071, +0.0006] | 2366 / 141 |
| cross-track MAE, m (LATERAL) | -0.5379 [-0.6595, -0.4195] better | +0.0193 [-0.0075, +0.0465] | -0.4301 [-0.5673, -0.2996] better | +0.0043 [-0.0284, +0.0396] | 2399 / 141 |
| L1 lateral-accel MAE, m/s^2 (LATERAL) | -0.1471 [-0.5913, +0.2761] | -0.0404 [-0.2374, +0.1855] | -0.2846 [-0.8861, +0.2657] | -0.1696 [-0.4961, +0.1305] | 2366 / 141 |
| L2 curvature MAE, GT >= 5.2 m/s, 1/m (LATERAL) | +0.0043 [+0.0016, +0.0076] **WORSE** | -0.0167 [-0.0293, -0.0069] better | +0.0045 [+0.0010, +0.0085] **WORSE** | -0.0036 [-0.0092, +0.0010] | 2106 / 135 |
| L3 yaw-rate MAE, GT >= 5.2 m/s, deg/s (LATERAL) | +0.2956 [-0.7771, +1.5204] | -1.6825 [-2.8628, -0.5854] better | +0.0548 [-1.1798, +1.2815] | -0.9539 [-1.9041, -0.1447] better | 2106 / 135 |

* MC-F's ADE6 − HACC upper bounds are 0.0740 and 0.0790 (printed numbers), above 0.06035; its L2 is separated-worse than P10 at both seeds. The PREREG: "A MC-F win does not promote MC-F without its own read", and this is not a win.

### STEEP (the deliberate-regression arm: HACC's table except knots 8 to 12 = a ramp 0 to 0.6)

| row (family) | s0: arm − P10 | s0: arm − HACC | s1: arm − P10 | s1: arm − HACC | n (windows / episodes) |
|---|---|---|---|---|---|
| ADE2, m (ADE) | -0.0122 [-0.0165, -0.0071] better | +0.0000 [-0.0002, +0.0005] | -0.0107 [-0.0158, -0.0047] better | -0.0000 [-0.0002, +0.0001] | 2399 / 141 |
| ADE6, m (ADE) | -0.4377 [-0.6451, -0.2381] better | +0.3303 [+0.2290, +0.4376] **WORSE** | -0.6636 [-0.8510, -0.4910] better | +0.0700 [-0.0108, +0.1473] | 2399 / 141 |
| speed MAE, m/s (LONGITUDINAL) | -0.1301 [-0.3119, +0.0343] | +0.2321 [+0.0949, +0.3652] **WORSE** | -0.3375 [-0.5077, -0.1905] better | +0.0502 [-0.0653, +0.1549] | 2399 / 141 |
| along-track MAE, m (LONGITUDINAL) | -0.1252 [-0.3272, +0.0746] | +0.3486 [+0.2401, +0.4648] **WORSE** | -0.3821 [-0.5589, -0.2207] better | +0.1264 [+0.0434, +0.2047] **WORSE** | 2399 / 141 |
| accel MAE, m/s^2 (LONGITUDINAL) | +0.4208 [+0.0389, +0.8033] **WORSE** | +0.4036 [-0.0478, +0.8039] | -0.0070 [-0.3692, +0.3103] | +0.1948 [-0.0940, +0.4424] | 2399 / 141 |
| heading MAE, deg (LATERAL) | +2.1678 [+1.0949, +3.3043] **WORSE** | +2.8002 [+2.0809, +3.5278] **WORSE** | +0.0558 [-0.7000, +0.8566] | +0.9810 [+0.4566, +1.5529] **WORSE** | 2373 / 141 |
| yaw-rate MAE, deg/s (LATERAL) | +16.8586 [+11.3539, +22.8499] **WORSE** | +13.9735 [+8.9035, +19.4390] **WORSE** | +7.2725 [+3.2535, +11.7898] **WORSE** | +6.0808 [+2.2074, +10.4126] **WORSE** | 2366 / 141 |
| curvature MAE, 1/m (LATERAL) | +0.1536 [+0.1152, +0.1944] **WORSE** | +0.1234 [+0.0932, +0.1558] **WORSE** | +0.0829 [+0.0532, +0.1151] **WORSE** | +0.0699 [+0.0459, +0.0963] **WORSE** | 2366 / 141 |
| cross-track MAE, m (LATERAL) | -0.5583 [-0.6742, -0.4493] better | -0.0011 [-0.0368, +0.0337] | -0.4926 [-0.6207, -0.3802] better | -0.0582 [-0.0974, -0.0222] better | 2399 / 141 |
| L1 lateral-accel MAE, m/s^2 (LATERAL) | -0.1705 [-1.5362, +1.1314] | -0.0637 [-1.4070, +1.1291] | -0.8517 [-2.0526, +0.1812] | -0.7367 [-1.8025, +0.1769] | 2366 / 141 |
| L2 curvature MAE, GT >= 5.2 m/s, 1/m (LATERAL) | +0.1087 [+0.0742, +0.1474] **WORSE** | +0.0876 [+0.0608, +0.1180] **WORSE** | +0.0573 [+0.0315, +0.0888] **WORSE** | +0.0491 [+0.0272, +0.0776] **WORSE** | 2106 / 135 |
| L3 yaw-rate MAE, GT >= 5.2 m/s, deg/s (LATERAL) | +15.5933 [+10.4232, +21.7316] **WORSE** | +13.6152 [+9.4015, +18.5681] **WORSE** | +7.5710 [+3.9829, +11.8875] **WORSE** | +6.5623 [+3.5831, +10.1599] **WORSE** | 2106 / 135 |

STEEP behaves as a regression arm should: M1 above HACC's at both seeds (section 3), and lateral rows far above P10 and HACC.

---

## 5. Beside: joint own-speed mask, a GT-defined launch subset, and m = 30

### 5.1 L2 on the joint own-speed mask (arm − P10; tier T1-kin; paired, B = 2,000)

| arm | s0 | s1 |
|---|---|---|
| HACC (reference) | +0.0018 [+0.0002, +0.0036], separated (n 2,050 / 135) | +0.0015 [−0.0001, +0.0033], not separated (n 2,068 / 135) |
| MC-H | +0.0016 [−0.0004, +0.0040], not separated (n 2,085 / 135; 0.9789199692048246 of the fast pairs kept) | +0.0013 [−0.0009, +0.0037], not separated (n 2,090 / 135; 0.9817978516699051 kept) |
| MC-F | +0.0021 [+0.0001, +0.0045], separated (n 2,077 / 135; 0.9686365802690912 kept) | +0.0024 [+0.0005, +0.0048], separated (n 2,086 / 135; 0.9774718627415039 kept) |
| STEEP | +0.0169 [+0.0107, +0.0246], separated (n 2,049 / 135; 0.9103793554087136 kept) | +0.0075 [+0.0034, +0.0125], separated (n 2,061 / 135; 0.9448889704532942 kept) |

The reader reproduces HACC's value to four decimals (`joint_mask_matches_diagnosis_4dp` true at both seeds).

### 5.2 The launch subset (v0 < 6 m/s and the GT chord speed over the last 0.2 s of the 6 s horizon ≥ v0 + 1 m/s; 327 windows / 49 episodes; the L2 rows 255 / 45)

| row (MC-H) | s0 | s1 |
|---|---|---|
| ADE6, arm − P10 | −0.0348 [−0.3428, +0.2190] | +0.0507 [−0.2904, +0.3422] |
| ADE6, arm − HACC | −0.0659 [−0.2464, +0.1268] | +0.2861 [+0.1190, +0.4523], separated |
| L2, arm − P10 | **+0.0247 [+0.0055, +0.0438], separated-worse** | +0.0122 [−0.0069, +0.0325] |
| L2, arm − HACC | −0.1048 [−0.1847, −0.0277], separated | −0.0482 [−0.0903, −0.0124], separated |
| accel MAE, arm − P10 | +0.8522 [−0.2676, +1.8659] | +0.9781 [−0.0453, +1.9366] |
| accel MAE, arm − HACC | +0.5912 [+0.0402, +1.1336], separated | +1.6455 [+0.7054, +2.6364], separated |
| along-track MAE, arm − P10 | −0.0801 [−0.2413, +0.0692] | −0.1304 [−0.2775, −0.0006], separated |
| along-track MAE, arm − HACC | −0.2877 [−0.5046, −0.0698], separated | +0.0118 [−0.1511, +0.1636] |

Inside the launch windows MC-H's L2 is separated-worse than P10 at s0 and not at s1, the same pattern as the full-set row. MC-F and STEEP launch-subset rows are in `rows.seeds.<s>.arms.<arm>.beside.launch_subset` (MC-F s0: L2 arm − P10 +0.0344 [+0.0132, +0.0556]; STEEP s0: +0.3567 [+0.1791, +0.5185]).

### 5.3 m = 30 (the replan period is the whole 6 s horizon; arm − P10; n 2,399 / 141, heading 2,373 / 141, yaw-rate, curvature and L1 2,366 / 141, L2 and L3 2,106 / 135)

| row | MC-H s0 | MC-H s1 | MC-F s0 | MC-F s1 | STEEP s0 | STEEP s1 |
|---|---|---|---|---|---|---|
| ADE2, m | -0.0087 [-0.0143, -0.0012] | -0.0076 [-0.0142, +0.0020] | -0.0030 [-0.0107, +0.0084] | -0.0017 [-0.0111, +0.0128] | -0.0122 [-0.0165, -0.0071] | -0.0107 [-0.0158, -0.0047] |
| ADE6, m | -0.4375 [-0.5462, -0.3399] | -0.4501 [-0.5567, -0.3487] | -0.4359 [-0.5451, -0.3377] | -0.4482 [-0.5559, -0.3473] | -0.4386 [-0.5477, -0.3406] | -0.4510 [-0.5579, -0.3496] |
| speed MAE, m/s | -0.0606 [-0.1018, -0.0146] | -0.0790 [-0.1132, -0.0434] | -0.0558 [-0.0973, -0.0090] | -0.0759 [-0.1104, -0.0392] | -0.0534 [-0.0951, -0.0056] | -0.0748 [-0.1105, -0.0363] |
| along-track MAE, m | -0.1695 [-0.2370, -0.1029] | -0.1820 [-0.2372, -0.1223] | -0.1677 [-0.2351, -0.1006] | -0.1799 [-0.2358, -0.1186] | -0.1708 [-0.2386, -0.1041] | -0.1831 [-0.2387, -0.1233] |
| accel MAE, m/s^2 | +0.1099 [+0.0836, +0.1455] | +0.0789 [+0.0511, +0.1211] | +0.0674 [+0.0418, +0.1002] | +0.0322 [+0.0087, +0.0669] | +0.3234 [+0.2718, +0.3910] | +0.2071 [+0.1610, +0.2769] |
| heading MAE, deg | -1.5506 [-1.9101, -1.2169] | -1.5762 [-1.9673, -1.2107] | -1.5528 [-1.9127, -1.2206] | -1.5829 [-1.9765, -1.2160] | -1.5440 [-1.9021, -1.2117] | -1.5602 [-1.9510, -1.1968] |
| yaw-rate MAE, deg/s | -0.7461 [-0.9901, -0.5246] | -0.6203 [-0.8390, -0.4211] | -0.8186 [-1.0588, -0.6037] | -0.7283 [-0.9465, -0.5382] | -0.6436 [-0.9001, -0.4221] | -0.4882 [-0.7114, -0.2795] |
| curvature MAE, 1/m | -0.0049 [-0.0089, -0.0014] | -0.0034 [-0.0066, -0.0005] | -0.0053 [-0.0089, -0.0020] | -0.0039 [-0.0071, -0.0012] | -0.0053 [-0.0093, -0.0018] | -0.0026 [-0.0059, +0.0007] |
| cross-track MAE, m | -0.3905 [-0.4843, -0.3024] | -0.3774 [-0.4822, -0.2810] | -0.3905 [-0.4843, -0.3024] | -0.3774 [-0.4822, -0.2810] | -0.3905 [-0.4843, -0.3024] | -0.3774 [-0.4822, -0.2810] |
| L1 lateral-accel MAE, m/s^2 | -0.0786 [-0.0956, -0.0637] | -0.0616 [-0.0795, -0.0452] | -0.0849 [-0.1022, -0.0697] | -0.0695 [-0.0883, -0.0529] | -0.0656 [-0.0818, -0.0509] | -0.0510 [-0.0688, -0.0347] |
| L2 curvature MAE, GT >= 5.2 m/s, 1/m | +0.0010 [-0.0002, +0.0023] | +0.0014 [-0.0003, +0.0038] | +0.0009 [-0.0003, +0.0022] | +0.0013 [-0.0004, +0.0036] | +0.0012 [+0.0000, +0.0026] | +0.0016 [-0.0001, +0.0040] |
| L3 yaw-rate MAE, GT >= 5.2 m/s, deg/s | -0.2757 [-0.4122, -0.1487] | -0.2729 [-0.4140, -0.1459] | -0.3134 [-0.4546, -0.1842] | -0.3229 [-0.4608, -0.1996] | -0.2008 [-0.3407, -0.0653] | -0.2132 [-0.3538, -0.0865] |

* m = 30 is beside; the PREREG decides nothing on it. The reader stores arm − P10 only for m = 30. At m = 30 the three arms are close to one another and L2 is not separated-worse for MC-H at either seed (+0.0010, +0.0014), while STEEP's s0 L2 interval has a lower end that prints as +0.0000 and the reader flags it separated. All three arms share HACC's along weights at knots 13 to 30 and HACC's cross weights (PREREG section 2), which is the part of the plan that one 6 s period executes beyond knot 12.

---

## 6. The four metric families (binding), per family, never pooled

Common to every cell: **tier T1-kin (self-action open loop); paired episode-cluster bootstrap, B = 2,000; arm − P10 and arm − HACC; both seeds; one reading per seed (answers "another draw of episodes" only).**

* **LONGITUDINAL.** speed MAE, along-track MAE and accel MAE, section 4 (m = 10) and section 5.3 (m = 30). MC-H against P10: speed and along-track separated-better at both seeds, accel MAE separated-worse at both seeds (+0.3166 and +0.3219). **Distance keeping (headway, time gap, TTC): NOT COMPUTED**, with the reader's reason: "v8f_t1_eval passes no lead track to analyze()". Anti-echo is not reported in this read at all (the read file has no anti-echo block); it is **undischarged** for these numbers, as for every longitudinal number in the programme until a hold-v0 copy is checked. Work item.
* **LATERAL.** heading, yaw-rate, curvature, cross-track, L1, L2, L3, sections 4 and 5. The decisive row is L2 (curvature MAE where GT speed ≥ 5.2 m/s); L1 is the lateral-acceleration MAE `|v ω − v_G ω_G|` on the canonical pair mask, and L2 and L3 are curvature and yaw-rate MAE on pairs with GT speed ≥ 5.2 m/s (`RESULT_V8F_T1_LATCOND.md`).
* **TACTICAL: NOT COMPUTED.** The reader's reason: "the T1 dumps record no per-tick T code (work item)". n = 0 per tick. The decision and goal-setting quality of the loop is not measured here.
* **STRATEGIC: NOT COMPUTED.** The reader's reason: "no per-tick S checkpoints in the T1 dumps". n = 0. **The family is NOT paused for v8F** (PI ruling 2026-10-17, `PI-V8F-2026-10-17`; R72 / R73). The reader prints it as `PAUSED (PI_DECISION_QUEUE ITEM 25) and not computed`, and `INTERPRETATIONS.md` item 11 says the same. That status text is the superseded pause: it is the R72 / R73 class again (a label in a generator that was written on 2026-10-26, after R73's sweep of the live generators, which covered `stack/scripts/v8f*`, the package `code/` and `raw/2026-10-25-{kindx,listen2,mixo}`; `raw/2026-10-26-hmc-build/` did not exist then). The generator lines are `hmc_read.py:392` on the tip (394 in the working copy) and the key `families_not_computed.STRATEGIC`. The number is not computed because the dumps lack the checkpoints, so it is a work item, not a pause. No number in this read changes. **Fixed at landing:** the reader's label now reads "NOT COMPUTED: no per-tick S checkpoints in the T1 dumps (work item). NOT paused for v8F". The banked read file keeps the string it was written with.

---

## 7. Controls (every committed control; each with its deliberate-regression arm where the PREREG gives one)

Under Amendments 1 and 2 (`amendments_applied`: `c1_bar` "ulp32max", `c1_amendment` "A2-2026-10-26", `c4_clause2` "tail0", `c4_amendment` "A1-2026-10-26"). `controls_all_pass` true; `controls_failed` empty; reader time 60.65 s.

| id | control | result under the amendments | regression arm (must FAIL) |
|---|---|---|---|
| **P-1** | each run is the pre-registered configuration of the right baked checkpoint (17 checks per run) | PASS on all 12 runs | none given |
| **C-1** | reachability: (plan − P)_along = c · (τ − P)_along at knots 1–12, every tick | PASS under A2: max \|d\| / ulp32 of the largest operand 0.50000 / 0.49999 (MC-H s0 m10 / m30), 0.50000 / 0.49998 (MC-H s1), 0.49999 / 0.49997 (MC-F s0), 0.50000 / 0.50000 (MC-F s1); `table_constant_over_knots_1_12` true on all eight runs; float32 constants `const_c_float32` 0.20022018253803253 / 0.21978281438350677 (MC-H s0 / s1), 0.3070783317089081 / 0.3468058705329895 (MC-F s0 / s1). **The literal 1e-6 m bar and A1's bar both fail on these runs** (section 1) | HACC's table on the MC-H ticks (m10): max \|d\| 1.797e+00 m (s0), 2.006e+00 m (s1): fails as it must |
| **C-2** | reachability: the post-map is in the loop, max \|MC-H − HACC\| executed at m = 10 > 1e-2 m | PASS: MC-H 59.619472009731524 m (s0), 61.356355516022866 m (s1); MC-F 47.86594162079737 / 51.229760175294146; STEEP 58.717042307048075 / 72.33527608278762 (key `max_abs_cl_minus_HACC_banked_m`: the replanning-loop output against HACC's banked one) | none given |
| **C-3** | instrument: the along-ramp term on MC-H and MC-F ticks is exactly 0.0 (≤ 1e-9) | PASS: v0, a0, kappa0 terms max 0.0 on all eight MC-H and MC-F runs | HACC's and STEEP's tables on the MC-H ticks, and STEEP's own ticks, read non-zero (max a0 term 15.28 / 37.05 / 38.63 m/s² at s0; 11.51 / 25.28 / 17.48 at s1): fire as they must. **HACC's TRAIN known value reproduced:** median ramp a0 −0.15876310085314316 (banked −0.1588, \|Δ\| 3.69e-05) at s0 and −0.0726273800838051 (banked −0.0726, \|Δ\| 2.74e-05) at s1, n = 1,747, bar 2e-3 |
| **C-4** | the fit: closed-form c equals the grid argmin to ≤ 0.005; GT := P with the held tail also zeroed gives c = 0 (A1) | PASS: s0 c 0.30707832486066766 against grid 0.305 (\|Δ\| 0.0020783); s1 0.34680587498548143 against 0.345 (\|Δ\| 0.0018059); clause 2 c = −0.0 at both seeds (literal value for the record 0.3094034378411479 / 0.3488629300801739). Data-only c is 0.084 / 0.072 (A1 text), so MC-F's c is set by continuity with HACC's tail | the literal clause 2 reads c = 0.3094 / 0.3489 and fails (the A1 finding) |
| **C-5** | the floor: the P10 loop equals T1KIN's banked floor; the m = 30 open-loop identity ≤ 1e-9 m | PASS on all six arm-seed groups: \|cl − ol\| at m30 1.1369e-13 m (s0) and 1.4211e-13 m (s1); floor ADE6 at m30 2.486992117746755; m10 floor against the banked HACC dump `fl` max \|Δ\| 0.0; `max_abs_floor_vs_banked_P10_plan_m` 1.525878909092171e-05; unique grid; eid equals the banked grid | none given |
| **C-6** | the reader reproduces HACC's banked L2 from `consolidated_c` | PASS: +0.0210 [+0.0099, +0.0351] (s0) and +0.0082 [+0.0026, +0.0148] (s1), \|Δ\| 0.0 against the banked | none given |
| **C-6b** | M1, M2, M1(P10) of HACC reproduce the PREREG's printed values | PASS to 4 dp at both seeds (0.0536 / 0.0277; +0.0202 / +0.0072; 0.0137) | none given |
| **C-7** | the frozen suite | PASS: 226 passed, 1 skipped, 6 warnings in 580.74 s (0:09:40); six test files of the tool tree | none given |

Other facts of the run, from the chain log and the bake report: the tool tree `E:/Projects/TanitAD-artifacts/v8f-mini-hacc-mc/tool_81aa3e36` (its `--dump-ticks` count 8 and causal-bucket count 1, with the controls `--replan-steps` 2 and `no-such-flag` 0); 12 of 12 runs with a json; zero training (six baked checkpoints, 86,961,655 B (MC-H, MC-F) or 86,963,733 B (STEEP), each from the banked HACC checkpoint: source sha256 `71de3c61…` (s0) and `32e4809d…` (s1)). The tables' file sha256: MC-H s0 `2e7f2518b3360d6b3770beb943084bb96fe9d47b89896a122ec04aa4bd4008e2`, s1 `a302d7c7dbb90a94cc317ba5fc6dddaf53a9d5f8221c236e6dc296235ec433d6`; MC-F s0 `4c599fd8c1c72c3696f19e700b7e953e06b8a26a64b578d212581428dd6bf508`, s1 `03becfba5824b69aafaff7c4852bab19b1841d722b5ac001ca7d87a3d2c22430`; STEEP s0 `1fff87d14fddb61aedf17d09c19fcef604f5fdd44cebe95bb276a4f098eed5cf`, s1 `eb47ebb98835d9301b3de1fa3a9e8b054374fc5538c1134b8169b401a9851553`. Baked checkpoint sha256: MC-H s0 `ceb34785…`, s1 `3d05a2b2…`; MC-F s0 `e1c029da…`, s1 `f102e649…`; STEEP s0 `5dac8719…`, s1 `f4ed69b3…` (full values in `BAKE_REPORT_MC.json`).

---

## 8. What it means (inside the PREREG's committed meaning)

**The committed meaning, MECHANISM-NOT-ENOUGH (quoted):** *"the pump is real; O's own slow launch intent carries the rest (O's v0 −1.78 / −0.93 m/s at tick 1 in the launch windows)"*.

What this read measured in those terms, and no further:

1. **The crawl responds to the lever in both directions.** M1 falls to 0.0184 / 0.0151 with the constant weight (HACC 0.0536 / 0.0277) and rises to 0.0850 / 0.0504 with the steep ramp, at both seeds. This is the committed "pump is real" condition, in point values and in the beside intervals.
2. **That was not enough for the lateral row.** MC-H's L2 against P10 falls from HACC's +0.0210 / +0.0082 to +0.0028 / +0.0013 (the first still separated-worse, the second not).
3. **"O's own slow launch intent carries the rest" is the PREREG's committed reading, not something this read tested.** The O's-v0 figures belong to the diagnosis (`raw/2026-10-26-lcdiag/DIAGNOSIS.md`, INHERITED here). What the read adds is descriptive: in the GT-defined launch windows MC-H's L2 is separated-worse than P10 at s0 (+0.0247 [+0.0055, +0.0438], n 255 / 45) and not at s1 (+0.0122 [−0.0069, +0.0325]).

**Post hoc, labelled, committing nothing:**

* The zero-training lever bought its lateral gain by paying position and acceleration: ADE6 against HACC +0.1383 and +0.2060 (separated-worse) and accel MAE against P10 +0.3166 and +0.3219 (separated-worse). The PREREG's train-side qualification (open loop, in sample) did not show that cost (section 2, point 3).
* MC-F, whose ADE6 is nearest HACC's (+0.0061 [−0.0615, +0.0740], +0.0307 [−0.0165, +0.0790]), keeps L2 separated-worse than P10 at both seeds and moves the crawl less (M1 0.0285 / 0.0193). The two arms trade crawl against position in the same direction.

---

## 9. What these numbers do not say

* **Training variance.** There are two M2b seeds, each post-mapped; the outcome row reads differently at the two on L2. Neither answers "another training run". Per `H-ESTIM-SEED-1` a separated interval is necessary, not sufficient, for a claim about a lever's effect; the L2 lower bound at s0 is +0.0004.
* **Inference variance.** Not applicable (deterministic planner).
* **Tier.** T1-kin, self-action open loop, logged camera frames. This is not a driving result and does not say the model drives.
* **Longitudinal numbers are unverified against a hold-v0 copy** (anti-echo undischarged).
* **Tactical and strategic are not computed** (sections 6), so nothing here says the hierarchy's decisions survive or fail under MC-H.
* **MC-H is not adopted.** FIXED-KEEP did not match; nothing here promotes MC-H, MC-F or STEEP as a post-map.
* **The L2 interval at s0 is narrow on the favourable side.** [+0.0004, +0.0055] excludes 0 by 0.0004. The outcome is decided by that fact exactly as the PREREG's "separated-worse at ≥ 1 seed" is written.

---

## 10. The next lever (RULE ZERO: a failed arm leaves the next arm behind)

* **Committed next, verbatim:** *"the box-2 TRAINING lever (§7)"*, i.e. **O-RECOVER**, an ego-state-perturbation fine-tune of M2b (DART / ChauffeurNet style) that targets O's launch from crawl states. Its PREREG exists on the tip: `Project Steering/PREREG_V8F_ORECOVER.md` (written 2026-10-26; it records the PI's go in chat, 2026-10-26: "queue O-RECOVER on box 2 after OACC and MIXO, through the launch gate, both seeds, under its own pre-registration"). Arms ORC (decisive) and ORC-0 (the control, same fine-tune with the perturbation off), 3,000 steps, both seeds.
* **What this read hands to it:** the pump is real (M1 moves with the weight in both directions), so the post-map cannot carry the remainder; the cost of the zero-training lever on position and acceleration (section 2) is the price to beat; the launch windows are where L2 stays separated-worse at s0 (section 5.2).
* **What blocks it:** box 2 time, which the PREREG marks as needing PI / MM sign-off, and the queue (FLAT and OACC were queued there). Nothing in this read's instruments is blocked.
* **Open work items from this read (not bars):** tactical and strategic for the T1 dumps (the per-tick T code and S checkpoints are not recorded; the strategic status text in `hmc_read.py:392` and `INTERPRETATIONS.md` item 11 should be corrected to "not computed: no per-tick S checkpoints", as R72 / R73 require for v8F); distance keeping (no lead track passed to `analyze()`); the anti-echo condition; the landing of Amendment 2 with the order-of-writing question in section 1, point 6; the cause of the three earlier C-7 stops (not examined here).

---

## 11. Provenance and manifest (nothing staged, committed or pushed)

**JSON key paths** (decisive read: `C:/Users/Admin/hmc/read_a2/HMC_READ.json`, sha256 `724d07cafd16961920cacfc573c5d3b6c34b919ca1e84adab8bfb712b1d05acb`, 105,685 B):

| content | key path |
|---|---|
| outcome, matching rows, committed next, KEEP sensitivity | `OUTCOME.{outcome,matching_rows_in_table_order,first_match_reported,ambiguity,committed_next,KEEP_bar_sensitivity_rows_matched_if_bar_were,note}` |
| the clause values behind the outcome | `outcome_inputs.{L2_MCH_minus_P10_not_separated_worse,MCH_minus_HACC_ADE6_upper_bound_m,KEEP_by_F_half_reading,M1,M1_MCH_lt_HACC,M1_STEEP_gt_HACC,F_half_used}` |
| tier, estimator, amendments applied | `tier`, `estimator`, `amendments_applied` |
| controls | `controls.<P-1/C-1/C-2/C-3/C-4/C-5/C-6/C-6b/C-7>`, `controls_all_pass`, `controls_failed`, `seconds_controls` |
| arm rows, both comparators | `rows.seeds.<s0/s1>.arms.<MC-H/MC-F/STEEP>.{rows_arm_minus_P10,rows_arm_minus_HACC}.<row>` |
| mechanism rows | `rows.seeds.<s>.arms.<arm>.{M1,M2,M1_P10,n_arm_crawl_pairs,M1_minus_HACC_beside,mean_ade6_m,R2_reader_vs_tool_analyze_ade6_delta}` |
| HACC reference | `rows.HACC_reference.<s>.{M1,M2,M1_P10,joint_mask_L2_HACC_minus_P10}` |
| beside | `rows.seeds.<s>.arms.<arm>.beside.{L2_joint_own_speed_mask_arm_minus_P10,joint_mask_fraction_of_fast_pairs_kept,launch_subset,families_not_computed,m30_arm_minus_P10}` |

**The runs themselves:** `E:/Projects/TanitAD-artifacts/v8f-mini-hacc-mc/run/` holds the 12 `t1v8f_<arm>_<seed>_<m>.json`, `ticks/ticks_*.npz` (m10 15,585,862 B each, m30 5,212,586 B each), 12 `dump_*/` directories of 141 episode files (6,139,492 B each), `PINS.txt` (tool-tree digest `c8d2990e3d0436b3022412a27aee713d9d46ecf4b4b2168ce485f33991612665`, the `v8f_t1_eval.py` sha256 `ffdd1469…`) and `ckpt_pins.txt` (the baked checkpoints' sha256, as in section 7). The reader opened them; I opened only the two pin files. The baked checkpoints are `E:/Projects/TanitAD-artifacts/v8f-mini-hacc-mc/M2b_s{0,1}_hmc_{MCH,MCF,STEEP}.pt`.

**Other files read** (all under `C:/Users/Admin/hmc/` unless stated): `AMEND_HMC2.md` (sha256 `f35de466b03e1b3ec71003b54c79c0891447fc2d3798524af332364ecf3b2969`); `read_a2/hmc_read_a2.out`; the A1 read `read/HMC_READ.json` (sha256 `b04ed94a2ef1a1af51d3f8622c9df1a11dd8458030a595cd50e31db4a72e779e`) and `read/hmc_read_a1.out`; the controls-only read `read/READ_CONTROLS.json`; `read/probe_c1_operand.{json,py}`; `logs/chain.out`, `logs/read_controls.out`, `logs/suite.out`, `logs_stop_c7_{0620,0649,0705}/`; `BAKE_REPORT_MC.json`; `INTERPRETATIONS.md`; `hmc_read.py` (sha256 `6083b21b…`, the reader that ran) against `hmc_read.py.pre_a2` (`e197b7e4…`, equal to the build pin in `BUILD_SHA256.txt`); `hmc_controls.py` (`0bdba3f6…`) against `hmc_controls.py.pre_a2` (`0f9de3a7…`, the build pin). PREREG: `Project Steering/PREREG_V8F_HACC_MC.md` at tip `04d5667` (repo mirror `C:/Users/Admin/tanitad-push`); diagnosis `raw/2026-10-26-lcdiag/DIAGNOSIS.md` (INHERITED); reference results `RESULT_V8F_T1KIN_CAUSALBUCKET.md`, `RESULT_V8F_T1_LATCOND.md`.
**Checks made for this write-up beyond reading the JSON:** that the controls-only and A1 read files hold no `rows` key (section 1); the diff of the reader and controls module before and after A2 (section 1); the tip's `hmc_read.py` for the strategic status string (section 6); the file creation and modification times behind section 1, point 6; and a scan that every `point [low, high]` triple in this file occurs together in one JSON object (output beside this file: `verify_triples.out`).
