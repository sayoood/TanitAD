# MM reading: LON-PLAN AMEND_6 (P3), RA refit on 1,000 TRAIN clips (2026-11-17 ~22:10Z)

Evidence class MEASURED (ours, zero world-model training). Tier T0 (oracle goal), which is a world-model diagnostic and never driving performance. Estimator: paired episode-cluster bootstrap, B 2,000 seed 0 ("another draw of EPISODES"). F_I answers "another INFERENCE run". Card `../AMEND_HWMPLAN_6.md`, sha256 db89e128, frozen before the pull.

## Outcome as committed (section 6): **R6-SAME**

Delta_R = RA6-L1 minus RA-L1 (AMEND_5) on 0-2 s acc_mae, T0 (`readout/RESULT_AMEND6.md`):

| chassis | Delta_R | CI | floor | separated-better | worse |
|---|---|---|---|---|---|
| R35_s0 | +0.0065 | [-0.0040, +0.0163] | 0.0042 | no | no |
| R35_s1 | -0.0051 | [-0.0149, +0.0046] | 0.0104 | no | no |

* RA6-L1 under AMEND_5's label rules: NARROWS at both chassis, qualifier MEAN-ONLY. CEIL-ORACLE6 is False at both.
* Controls: `controls/CONTROLS.json`, ok True, red list empty (C-RA-SHA, C-TRAIN-ISO, C-GRAD, C-PLANT, C-YAW-ID, K-control at both chassis).
* Four families per arm pair are in `readout/RESULT_AMEND6_full.md`, 994 windows / 71 episodes per chassis:
  * LONGITUDINAL, LATERAL and TACTICAL rows are computed at all three bands.
  * STRATEGIC is computed at the full 6 s band only: 60 of 180 rows. The 0-2 s and 2-6 s bands have no strategic events (n 0), so those rows read n/a. The route-adherence rows rest on 12 windows / 4 episodes; s_false rests on 856 / 69.
  * No outcome is decided on a STRATEGIC or TACTICAL cell (card section 5).

## What this read adds beyond the label

* **The readout got more precise, and the score did not move.** RA's held-out R^2 on the nested fit sets rose from 141 to 1,000 clips (`readout/FIT_CURVE_s{0,1}.json`):
  * s0: 0.222 [0.162, 0.286] at 141, then 0.262, 0.270 and 0.281 [0.255, 0.305] at 1,000;
  * s1: 0.208 [0.145, 0.274] at 141, then 0.256, 0.265 and 0.279 [0.254, 0.303] at 1,000.
* The T0 score is unchanged within the floor at both chassis. So a 7-point R^2 gain on the readout does not reach the plan score.
* The card already flagged this (R^2 was flat from 56 to 113 clips). The read now measures it at N = 1,000.
* **Fit-set size is ELIMINATED as a lever at N = 1,000.** This is a result and is logged as one.

## Next lever (card section 6, R6-SAME row, Rule Zero)

1. **AMEND_7, a nonlinear 0-2 s acceleration readout.**
   * What: an MLP on [z0, z10] in the RY head's shape, differentiable so the same pass inverts it.
   * Data: fit on the same 1,000-clip latents, which stay on F: (`lat6/`).
   * Cost: CPU only, with no world-model training and no box 2.
   * Order: its FIT-CURVE row is read first. The SCORE read follows only if its held-out R^2 beats the linear RA's (0.281 / 0.279 at 1,000) by a separated margin on the same folds.
   * The MM pre-registers it now.
2. **R35's response along RA's direction (ACTRSP Option B).** Approved by the PI on 2026-11-17 ("4: yes"). The build is in progress (`actrsp/optb/`), and the box-2 pilot follows the build review.
   * Given this read, Option B is the stronger candidate: precision of the readout is not what holds the score.
