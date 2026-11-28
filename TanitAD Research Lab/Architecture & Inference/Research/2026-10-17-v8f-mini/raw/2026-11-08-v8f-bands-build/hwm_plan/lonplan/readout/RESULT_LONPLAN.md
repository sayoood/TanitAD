# RESULT LON-PLAN (AMEND_HWMPLAN_3, frozen): REG at the TRAIN-selected dose on the 994 SCORE windows, both R35 chassis

**Outcome, as committed in section 6: the chassis disagree, so the label is SEEDS-DISAGREE and the less favourable branch's next step applies.**
* Chassis s0 reads **LON-PLAN-NARROWS, INFORMATIVE**.
* Chassis s1 reads **LON-PLAN-NARROWS, SHRINKAGE-ONLY**.
* **CEILING FAILED at both chassis.** REG reaches HOLD on the deciding cell but does not beat it.

Tier T0 throughout: the HWM-ORACLE goal is the encoded true future, so this is a world-model diagnostic, never driving performance.

Estimator: paired episode-cluster bootstrap B 2,000 seed 0, over 994 windows / 71 episodes per chassis. It answers "another draw of EPISODES". "Another INFERENCE run" is the floor F_I = max(|A0 seed 1 − seed 0|, |A1 seed 1 − seed 0|). "Another TRAINING run" is the two chassis.

Evidence: MEASURED, zero training. Doses λ\*_0 = 300 and λ\*_1 = 100, selected on TRAIN before any SCORE plan.

## 1. Deciding cells (0-2 s, inference seed 0)

| cell | s0 | s1 |
|---|---|---|
| F_I (acc_mae) | 0.0073 | 0.0041 |
| **Δ_H = A1 − HOLD** acc_mae | +0.0065 [−0.0005, +0.0137] not separated | **+0.0303 [+0.0183, +0.0443] WORSE than HOLD** |
| **Δ_R = A1 − A0** acc_mae | **−0.9586 [−1.0253, −0.8880]**, beyond F, seed-1 point −0.9514 | **−0.8382 [−0.9066, −0.7662]**, beyond F, seed-1 point −0.8417 |
| **Δ_S = A1 − S** acc_mae | **−0.0112 [−0.0175, −0.0041]**, beyond F, seed-1 point −0.0113 | −0.0016 [−0.0134, +0.0112], not separated |
| G_yaw = A1 − A0 yaw_mae (guard) | −0.0911 [−0.0988, −0.0834] (better) | −0.0809 [−0.0883, −0.0733] (better) |
| label | NARROWS, INFORMATIVE | NARROWS, SHRINKAGE-ONLY |

Reading the table:
* A0 (SEED at λ 0) is far worse than HOLD on 0-2 s acceleration: A0 − HOLD is +0.9651 [+0.8949, +1.0328] at s0 and +0.8685 [+0.7929, +0.9409] at s1.
* REG closes almost all of that gap and lands on HOLD: at s0 it is level with HOLD, and at s1 slightly worse.
* At s0, REG also beats a matched pull toward HOLD of the same size (S), so its solve carries acceleration information near HOLD. At s1 it does not.

## 2. Controls and beside rows

* **Deliberate regression R (λ = −max(λ\*, 1)) reads worse than A0, as required (C-REG):** R − A0 acc_mae +0.7214 [+0.6449, +0.8002] at s0 and +0.7914 [+0.7111, +0.8714] at s1.
* **Readbacks:** all 6 arm folders reproduce bit-equal (C-READBACK rc 0, `score_chain.log`).
* **S matches A1's TRAIN distance:** alpha is recomputed and asserted against the frozen values (`*_STAMP.json`).
* **TRAIN-CEILING flag:** false at both chassis. The in-sample C(λ\*) does not beat C(HOLD).
* **DIR-ACC** (Pearson r of the plan's 0-2 s departure from HOLD against the logged one, beside, no bar):

  | arm | s0 | s1 |
  |---|---|---|
  | A0 | +0.3795 [+0.3088, +0.4475] | +0.4109 [+0.3502, +0.4682] |
  | A1 | +0.4190 [+0.3599, +0.4778] | +0.3903 [+0.3152, +0.4608] |
  | S | +0.3795 [+0.3088, +0.4475] | +0.4109 [+0.3502, +0.4682] |
  | R | +0.4306 [+0.3741, +0.4853] | +0.4431 [+0.3896, +0.4931] |

  * S equals A0 by construction, because it scales A0's departure.
  * REG does not change the direction of the departure consistently. Every arm's departure points the way the drive went, at r of about 0.4.

## 3. Four families, A1 − A0, 0-2 s unless stated (T0)

* **LONGITUDINAL.**
  * acc_mae: the deciding cell, above.
  * speed_err: −0.9326 [−1.0566, −0.8096] / −0.9067 [−1.0331, −0.7875].
  * Distance-keeping on LEAD windows (214 windows / 32 episodes; tg 182 / 31):
    * hw_err −0.9700 [−1.6528, −0.4932] / −1.0577 [−1.8693, −0.5142] m;
    * tg_err −0.1378 [−0.2292, −0.0714] / −0.1346 [−0.2349, −0.0653] s;
    * ttc_err −2.7996 [−4.1028, −1.6095] / −3.2158 [−4.4841, −2.1497] s.
* **LATERAL.**
  * yaw_mae: the guard, above.
  * head_err −0.1046 [−0.1157, −0.0935] / −0.0914 [−0.1042, −0.0791].
  * curv_err −0.0151 [−0.0199, −0.0112] / −0.0127 [−0.0168, −0.0095] (950 windows / 71 episodes, steps with logged speed ≥ 1 m/s).
  * xtrack −1.2704 [−1.5184, −1.0507] / −1.1252 [−1.3340, −0.9244].
* **TACTICAL** (plan-derived, kappa against the logged manoeuvre, 0-2 s):
  * lateral 0.738 (A1) / 0.083 (A0) / 0.723 (HOLD) at s0;
  * longitudinal 0.682 / 0.056 / 0.680.
  * Per-band tables are in `c_A1_A0_s{0,1}.json`.
  * Goal and anchor selection: n = 0, NOT COMPUTABLE (the planner has no decision output).
* **STRATEGIC** (reported in full, full 6 s band only; on oracle rows it measures the world model's use of an oracle goal):
  * s_adh A1 − A0: +0.25 [0.00, +0.50] (n 12 / 4 episodes) at s0 and +0.375 [0.00, +0.857] (n 8 / 5) at s1. Not decision-grade at this n.
  * s_adh_deg: +11.5 [+2.2, +16.8] and +13.6 [+7.1, +22.2] degrees toward the announced side.
  * s_false: +0.026 [−0.008, +0.063] (853 / 69) and +0.033 [+0.002, +0.067] (855 / 70).
  * S_CKPT, intent 6-30 s, derived goals and detected info: NOT COMPUTABLE on this planner (structural n/a, as section 5 lists).

## 4. Next lever (Rule Zero; the less favourable label's branch, SHRINKAGE-ONLY)

* **The level-2 image constraint at λ = 0.**
  * Level 1's 0-2 s cost is the distance to level-2 rollout states, which are planned under off-image macros (ratio 6.77 / 6.79, AMEND_HWMPLAN_3 section 7).
  * That is the largest measured planner defect left.
* **After it: a level-1 cost read through RA**, the 2 s readout with transition skill (Amendment 1 section 1.4, frozen weights `readout_K10_R35_s{k}.pt`). Zero training.
* Each needs its own amendment, drafted before any plan.
* REG stays available at λ\*_k. It is not the solver of record: CEILING failed, and its gain is mostly the pull toward HOLD.
