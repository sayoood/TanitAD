# PREREG_SELCOR -- a TRAIN-fitted selection correction over refcv8's own fan, emitted through the factored (native-path) composition (Master Mind, 2026-10-26 ~04:47 CET)

Sealed BEFORE the fresh-clip locus roll ran and before any fit, pick, rate or bar exists. The committed next lever of L2-B
(`refcv9_l2/result/l2b_final/VERDICT.md`): *"TRAIN-fitted selector / score correction over the fan (the L1-type lever)"*.

## 0. What is known (disclosed)
* L2-B (PREREG_L2B 089d7ab6) read **SELECTION** at both inference seeds: on the 259 seed-0 false-brake (FB) windows a cruise
  candidate exists in 100 % and a GOOD one (staged ADE_6s <= tau) in 0.981 [0.956, 0.997]; FD 0.1395 (n GT-cruise 1,856); the floor if
  every SELECTION window were fixed is 0.0027. The refcv6 pre-blend pick on FB is decel 85 %.
* SSGUARD (PREREG_SSGUARD 2b9d1827) PASSED at seed 0: the native path with a re-chosen PROGRESS profile, composed per window, carries a
  lon lever without the lateral cost of a fan re-selection. The seed-1 replicate is in flight (PREREG_SSGUARD_S1 68fc8a07).
* L1C (PREREG_L1C): progress heads fitted on the 300-clip FRESH split read O2 -- vision is real on a clean fit; G fails at that head class.

## 1. Data (no eval row is used for any fit or choice)
* **FIT** = the L1C FRESH split (300 clips, never in train4572 nor eval139), S6 surface, **compliant rows** as L1C loads them
  (`l1c_fit_score.load_fresh`, tree `step48852ema_fresh`; L1C population: 7,147 rows), joined by (clip index, ws) to the fresh LOCUS roll:
  `code/l2c_locus_roll_fresh.py` = `l2b_locus_roll.py` + L1C's input swap only (diff `code/l2c_locus_roll_fresh.DIFF.txt`), shipped
  EMA-48,852, inference seed 0, os-only, per-frame feed, kit `F:/l1c/kit_fresh`, remap `refcv9_l1c/code/remap_l1c_fresh.json`, v9 md5
  `cc06d03a453f87f778e7555d5607a5aa`, windows `F:/l1c/kit_fresh/windows_s0.json` (10,456).
* **SCORE** = eval139 S6, the banked LOCUS traces `F:/refcv9_l2_locus/locus_s{0,1}.npz` (L2-B; os-only native roll, bit-equal to L1 BASE-os),
  seed 0 primary, seed 1 replicate (never pooled).

## 2. The arm SELCOR-G
Per window, `E` = the eligible set (exact numpy replica `l2b_score.eligible` of refc_v3.e9_rank's finite mask over `sel_score_v3`).
Corrected score `s'(c) = a * sel_score_v3(c) + w . phi(c)`, pick `c* = argmax_{c in E} s'(c)`, emitted plan
`P = at_progress(P_native, arc(fan[c*]))` (the SSGUARD composition: native PATH, progress of the picked candidate), staged by the
REGISTERED per-frame RETIME table exactly as the native plan. With `w = 0` the pick is the native `sel` (K-ID).
* **phi(c)** -- inference-available only (vision outputs + v0 at t0, admissible per the PI 09-02 ruling; no GT, no future ego, no situation-
  classifier output): lon6 one-hot of c (6); lon6 one-hot x v0/10 (6); `d = (arc_8(fan[c]) - arc_8(P_native)) / max(arc_8(P_native), 1 m)` and
  `d x v0/10` (2); candidate kind base/allocated (1).  **Variant Z** adds lon6 one-hot x PCA_k(standardised z_tac) (6k; z_tac from the L1C
  fresh bank on FIT rows and the L1 eval banks on SCORE rows; PCA fitted on FIT rows only), k in {4, 8, 16}.
* **Label** (FIT rows only): `y = argmin_{c in E} ADE_S6(stage(at_progress(P_native, arc(fan[c]))), G)` -- mean Euclidean error over the six
  S6 slots against the human future.
* **Fit**: conditional (multinomial) logit over E, cross-entropy to y, L2 on w (not a), float64, LBFGS, seed 0. lambda in
  {1e-3, 1e-2, 1e-1, 1, 10, 100} and the variant (S = scalars, Z with k) chosen by **5-fold episode-grouped CV on FIT rows**, criterion = mean
  staged ADE_S6 of the argmax pick on held-out folds; the final fit uses all FIT rows. Exactly ONE configuration is scored as SELCOR-G.
* **Beside (descriptive, never a bar):** PRIOR-ONLY (phi = the 6 lon6 intercepts: a class-prior shift; the calibration control); DER (the same
  recipe fitted on labels deranged across FIT episodes); ORACLE (argmin staged ADE on eval = the selection ceiling of this composition).

## 3. Population, estimator, tier
eval139 S6 **compliant** windows as registered; paired episode-cluster bootstrap B 2000 seed 0 (`pair_read_r8.paired_cell` -> `taniteval.ci`)
= "another draw of EPISODES" only; one training seed; inference seed 0 primary, 1 replicate. Tier: **T1 self-action OPEN LOOP on logged
eval139 frames (not driving performance).** Families: LONGITUDINAL (MD, FD, speed rows), LATERAL and TACTICAL rows of `PR.METRICS` + `ade_m`;
distance-keeping (no lead track, n = 0) and STRATEGIC (refcv8 runs --no-strategic, n = 0) not computable -- stated per family.

## 4. Controls (a failed control voids the bars)
* **K-ID:** `w = 0, a = 1` picks `sel` on every SCORE window and reproduces BASE-os|stage MD / FD to 5 dp: s0 (0.46948, 0.13955),
  s1 (0.47105, 0.13739).
* **K-FRESH-ROLL:** the fresh locus trace is FINAL with 10,456 windows, every FIELD present on every window, fan size 149, batch 1; its
  emitted plan equals `fan[sel]` bit-for-bit; and it equals the L1C fresh bank's `traj_own` (`F:/l1c/bank/ztac_fresh.npz`) bit-for-bit on every
  shared window (same code path, same seed).
* **K-JOIN:** every FIT row joins exactly one fresh locus window; **K-PRIOR:** the trace's `prior_path` at the S6 slots equals the loader's
  `HP` within 1e-5 m on every FIT row.
* **K-ADE:** the label's ADE_S6 of the staged NATIVE plan equals the battery's per-window `ade_m` (`arm_cells(...)['stage']`) within 1e-5 m on
  every eval139 compliant window at s0 (the label is the battery's metric).
* **K-ELIG:** `sel` is the argmax of `sel_score_v3` over E on every window (FIT and SCORE).

## 5. Bars (SELCOR-G vs BASE-os|stage, seed 0; PASS = P1 & P2 & G)
* **P1** FD separated BELOW (CI hi < 0).  **P2** MD NOT separated above (CI lo <= 0).  **G-strict** no S6 row (`PR.METRICS` + `ade_m`)
  separated worse.
* **Replicate (seed 1, same fitted correction, same bars):** REPLICATED iff P1 and G-strict hold at s1; the verdict carries both words.

## 6. Committed next steps
* **PASS + REPLICATED** -> SELCOR-G is a refcv8 deliverable candidate (a <= 6k+16-parameter score correction, no retraining): a SHIP-R8
  amendment needs a PI go first; then compose with the SELFSHARP progress lever as ONE sealed arm (FD and MD act on the same progress
  profile, so the composition is never assumed).
* **FAIL P1** with ORACLE FD far below BASE -> the correction cannot see the brake/cruise distinction from these features: the next lever is a
  per-candidate vision feature (the candidate's own decoder cues), pre-registered first; with ORACLE also near BASE -> this composition has no
  FD headroom: name it and move to the lon decoder (the refcv9 design input).
* **FAIL P2** -> FD is traded for MD (an operating-point move): read the FD/MD frontier over lambda on FIT rows; next = an MD-constrained fit.
* **FAIL G** -> name the rows; next = a cross-fitted margin gate (SSGUARD's GUARD-CF recipe, FIT-side tau).
* **PASS, NOT REPLICATED** -> the seed-0 gain is inside the inference-seed floor: report both words; no SHIP-R8 request.
