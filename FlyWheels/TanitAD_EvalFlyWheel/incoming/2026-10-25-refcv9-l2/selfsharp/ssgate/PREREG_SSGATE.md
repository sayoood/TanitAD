# PREREG_SSGATE -- sharpen the lon head only where it is CONFIDENT (Master Mind, 2026-10-26 ~03:02 CET)

Sealed BEFORE any gated-arm number exists. 0 GPU, 0 training: the arm is COMPOSED per window from two banked rolls.

## 0. What is known
* SELFSHARP (`../PREREG_SELFSHARP.md` cb2dc8ea, `../result/SS_RESULT.json`) read **FAIL (G_strict)**: P1 MD 0.4695 -> 0.3897
  (-0.0798 [-0.1317, -0.0357], separated), P2 FD +0.0108 [-0.0165, +0.0358] not separated; G-strict fails on six S6 rows
  (LAT_cross_abs_final_m, LAT_cross_mae_m, LAT_curvature_mae_1pm_valid, LAT_heading_mae_deg, LAT_yaw_rate_mae_radps_valid,
  TAC_traj_lat_correct); ORACLE also degrades only curvature / heading / yaw. Window flips: MD fixed 68 / broken 17,
  FD fixed 83 / broken 103.
* The scorer's committed FAIL_G lever (in `ss_score.py`, hashed at start): *"the cost is in the head's wrong choices, not the
  channel -- gate the sharpen on head margin."* This registration runs that lever.
* NOT computed, therefore unseen: any per-margin split of the SELFSHARP deltas, any gated arm.

## 1. Arm
**SSGATE(tau):** window w is sharpened iff `m_w >= tau`, where `m_w` = top-1 minus top-2 of `raw_lon_logits` over the six LON6 v7
columns (0,1,3,4,5,7) -- the head's own logits, no label (deployable). Plan = SELFSHARP's banked plan8 if sharpened, else BASE-os's
banked plan8 (L1 os-only `traj_own`).

**Why composition is exact (checked, not assumed):** the T1 roll is per-window -- in the ORACLE roll, windows the force did not touch
equal BASE-os bit-for-bit (ORACLE_ATTR identity 0.0). Control K-IND re-checks it here. Under that property a gated GPU roll would
produce exactly the composed plans; the GPU roll stays the confirmation step (section 5).

## 2. tau by episode cross-fit (never tuned on the windows it scores)
* Population: S6 compliant (as SELFSHARP), stage on. Episodes split into 5 folds: unique `eid` sorted, permuted with
  `numpy.random.default_rng(0)`, fold = rank mod 5.
* For held-out fold k, candidates on the other four folds: `tau` = +inf (no sharpen = BASE-os), -inf (all = SELFSHARP), and the
  in-training-fold margin quantiles 0.1, 0.2, ..., 0.9 of the S6 windows.
* **Admissible** on the training folds: G-strict holds (no `PR.METRICS` row and not `ade_m` separated WORSE than BASE-os, same
  `l2_score.compare`, B 2000 seed 0) AND FD not separated above BASE-os. tau = +inf is admissible by identity, so the set is never empty.
* **Pick** the admissible tau with the lowest training-fold missed-decel rate; ties -> the larger tau (fewer sharpened windows).
* Apply tau_k to fold k. The out-of-fold composed arm is SSGATE-CF.

## 3. Estimator, tier
Paired episode-cluster bootstrap over EPISODES (`l2_score.compare` -> `pair_read_r8.paired_cell`), B 2000, seed 0 -- "another draw of
episodes" only; ONE refcv8 training seed; ONE inference seed (0). Tier T1 self-action open loop on logged eval139 frames.
S2 rows NOT computed (stated PENDING, as SELFSHARP).

## 4. Controls (a failed control voids the bars)
* **K-IND:** ORACLE trace windows with no force (`oracle_labels` not in LON6) have plan8 bit-equal to BASE-os.
* **K-ALL / K-NONE:** the composer at tau = -inf reproduces SELFSHARP's MD / FD (0.38967 / 0.15032) and at +inf BASE-os's
  (0.46948 / 0.13955), to 5 dp.
* **K-SS:** the composer's SELFSHARP-vs-BASE `missed_decel` cell equals `SS_RESULT.json`'s exactly.

## 5. Bars and outcomes (SELFSHARP's definitions, on SSGATE-CF vs BASE-os)
* **P1** MD separated BELOW BASE-os. **P2** FD NOT separated above. **G-strict** no S6 row separated worse.
* **PASS** = P1 + P2 + G-strict -> the margin gate is the deployable lever: confirm with a GPU roll of the gated labels at the
  full-data tau (seed 0) AND an inference-seed-1 roll, then take it to refcv9 (a confidence-gated sharpen / temperature in the lon
  head) and to the PI for SHIP-R8.
* **FAIL P1** (the gate keeps too little of the gain) -> margin does not separate right from wrong choices: the lever is head
  ACCURACY (train the lon head toward the z_tac probe on the clean L1C split).
* **FAIL G only** -> the lateral cost rides on the sharpened windows whatever the margin: the cost is the sharpen channel, so the
  lever is a SOFTER sharpen (temperature on the LON6 logits) with a lateral guard, one GPU roll, pre-registered first.
* Descriptive beside (no bar): tau_k per fold; per-margin-decile MD fixed/broken, FD fixed/broken and the six G rows' mean delta;
  the full-data tau.
