# PREREG_L1C_H2CN -- does the NONLINEAR progress head (fitted on the clean fresh split) clear the guard G? (Master Mind, 2026-10-26 ~02:55 CET)

Sealed BEFORE the four-family rows and the braking AUC of H2cn exist. 0 GPU (CPU fit + scoring on banked arrays).

## 0. What is already known, and what is not
* L1C (`PREREG_L1C.md` + AMENDMENTS 1-2) read **O2**: P1 and P2 pass; **G fails on one row, `LAT_cross_mae_m` +0.0012 m, worse at both
  seeds**; braking AUC H2c vs H0 AUC-SAME. AMENDMENT 2 A2.8 names the O2 lever: the L1 H2n recipe (nonlinear head) on the same split.
* That lever ran as `l1c_next.py` N1 (EXPLORATORY, no bar): **S2 ADE H2cn - H0 = -0.0130 [-0.0197, -0.0064] (s0) / -0.0136 [-0.0203,
  -0.0075] (s1), separated** -- KNOWN before this seal. **NOT computed, therefore unseen:** H2cn's four-family rows, its braking AUC, any
  S6 fit. This registration exists to read those, under the SAME G as L1C.

## 1. Arm
**H2cn** = `l1_fit_score.h2n_fit` (one-hidden-layer MLP on the scalars + standardised z_tac, L1 loss, wd and epochs by 5-fold episode CV,
seed 0) fitted on the FRESH rows of each surface (S2 for P2n / Gn rows, S6 for the AUC), applied exactly like H2c
(`PH.at_progress(O, arc(HP) + pred)`). H0 = the frozen per-frame RETIME stage; H2c / H2cs / OS refitted as in `l1c_fit_score.py`.

## 2. Population, estimator, tier
As L1C: eval139 S2 compliant (4,159 windows, 138 episodes) at inference seeds 0 and 1; S6 compliant for the AUC; paired episode-cluster
bootstrap B 2000 seed 0 (`F.cell`) -- "another draw of EPISODES" only, models fixed; ONE training seed of refcv8; the head's own fit
seed is 0. Tier T1 self-action open loop on logged eval139 frames.

## 3. Controls
* **K-R reproduction:** S2 H2cn - H0 ADE equals N1's -0.0130 / -0.0136 within 1e-4. If it does not, the run is NOT void: the difference
  is reported as the head's fit-to-fit variance and the bars read this fit.
* **K-ROC:** `l1c_roc.py` controls (GT AUC 1, constant 0.5, banked stage / os / r7 AUC and FD/MD reproduce, L1C_OS == panel os) pass.
* **K-H2c:** the refitted H2c reproduces L1C's S2 H2c - H0 ADE (-0.0094 / -0.0099) within 1e-4.

## 4. Bars (L1C's definitions, applied to H2cn)
* **P2n** H2cn - H0 S2 ADE separated below at both seeds (already known to hold; re-read).
* **Gn** no S2 four-family row (H2cn - H0) worse at both seeds (TAC_* and LON_speed_within_0p5 higher-is-better; every other row
  lower-is-better), AND the braking-AUC reading of H2cn vs H0 is not AUC-DOWN.
* **PASS = P2n and Gn.**

## 5. Committed next step
* **PASS** -> H2cn is a refcv8 STAGE-UPGRADE candidate: the registered SHIP-R8 battery on the new stage is the next step; because it
  changes the shipped deliverable, the request goes to the PI with the numbers (O1's A2.8 route).
* **Gn fails on the same row (`LAT_cross_mae_m`)** -> the cross-track cost belongs to the PROGRESS mechanism, not the head class: read
  its magnitude against the inference-seed floor and carry the head to refcv9 as a trained, clean-split stage (the O3-style closing).
* **Gn fails on other rows** -> name them; the next lever is a fit that penalises them (constrained objective), pre-registered first.
