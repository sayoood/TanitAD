# PREREG_MAPSOFT_G -- MAPSOFT with a NEAR-FIELD GATE on the own-map features (Master Mind, 2026-10-26 ~01:55 CET)

Sealed BEFORE any MAPSOFT-G pick is computed. Identical to `PREREG_MAPSOFT.md` (sha256 `38c77b17…`) except sec. 1-2 below.

## 0. Why, and the honesty clause
* MAPSOFT (FAIL M1-M5): the cross-fit chose `(0, 0)` in 3 of 5 folds -- any own-map penalty cost ADE-to-human on the fitting folds.
* `RANGE_DIAG.json` (MEASURED, DESCRIPTIVE, 3,198 S6c GT windows, 136 episodes, raw candidates, written AFTER MAPSOFT's verdict):
  the own sidewalk rule's false alarms on HUMAN paths concentrate at **0-10 m ahead (8.3 % of windows)** vs 1.6-4.8 % in every farther
  band, while BASE's TRUE GT-sidewalk violations sit mostly **beyond 20 m** (7.2 / 7.3 / 7.2 % in 20-40 / 40-60 / 60+ m vs 1.3 % at 0-10 m).
  Own-flag precision on BASE plans (derived): ~3 % at 0-10 m, ~54-67 % beyond 20 m.
* **Honesty clause:** the gate FAMILY (exclude the near field) was suggested by that descriptive read of ALL eval episodes. The gate
  VALUE is cross-fitted (held-out episodes never tune it), but the family choice saw them. A PASS here is therefore labelled
  *"family chosen on eval, parameters cross-fitted"* and must be RE-READ on clips refcv8 and this analysis never saw (the L1C fresh
  split, once map rasters exist for it) before it is promoted to an inference row.

## 1. The arm (deployable: own scores + own map only)
`u(c) = z(c) - lam_s * pswmax_r(c) - lam_e * bnd_r(c)`, where `pswmax_r` / `bnd_r` are MAPSOFT's features computed ONLY over the
candidate's in-range FP5 points with **x >= r0** (x = forward distance in the t0 ego frame, `ix * 0.1 m`), and **r0 in {0, 10, 20} m**
(r0 = 0 reproduces MAPSOFT's features exactly -- control S-K3). Everything else as MAPSOFT sec. 1.

## 2. Fitting
As MAPSOFT sec. 2 over the grid `{0, 10, 20} x {0, 0.25, 0.5, 1, 2, 4, 8}^2` (147 points); ties -> smaller `lam_s + lam_e`, then smaller
`lam_s`, then smaller `r0`. **DER-MAPSOFT-G-XF** uses the donor's planes, gated and fitted the same way.

## 3-5. Evaluation, controls, bars: exactly MAPSOFT's (S-K0, S-K1, S-K2; M1-M5; PASS = M1-M5), plus
* **S-K3** the r0 = 0 gated features equal MAPSOFT's banked features (`pswmax`, `ownbnd_edge`, `dpswmax`, `derbnd_edge`) exactly.

## 6. Committed next step
* **PASS** -> label as above; re-read on the fresh split (map rasters for the L1C clips: a GPU roll -- queued, not a PI decision since it
  is the same eval tooling); only then an opt-in inference row.
* **FAIL with r0 = 0 or (0, 0) chosen in >= 3 folds** -> the gate does not rescue the own map: map-head training for sidewalk / boundary
  precision is the lever (compute -> PI decision), with the near-field false alarm (8.3 % of human paths at 0-10 m) as its first target.
* **FAIL otherwise** -> as MAPSOFT sec. 6.
