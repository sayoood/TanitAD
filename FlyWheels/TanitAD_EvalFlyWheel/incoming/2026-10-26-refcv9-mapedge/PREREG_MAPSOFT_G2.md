# PREREG_MAPSOFT_G2 -- MAPSOFT-G with the near-field gate grid extended to the ego-body zone (Master Mind, 2026-10-26 ~03:58 CET)

Sealed BEFORE any MAPSOFT-G2 pick or bar exists. 0 GPU: the banked seed-0 MAPSEL trace + rasters (`F:/refcv9_mapsel/trace_s0.npz`,
`F:/refcv9_mapsel/rast_s0`), exactly MAPSOFT-G's inputs.

## 0. What is known (disclosed -- computed on the SAME eval windows, which is why the gate is cross-fitted, not fixed)
* MAPSOFT-G (`PREREG_MAPSOFT_G.md` 1687688b, `result/mapsoft_g/`): FAIL (M4, M5); the cross-fit chose r0 = 10 m in 4/5 folds.
* New descriptive profile (scratch `nearfield_rows.py` / `nearfield_prec.py`, seed-0 rasters, every 6th window, 783 with GT, corridor
  |y| < 3 m): own-sidewalk (psw >= 0.5) precision on GT-seen cells rises from **0.10 at x 0-1 m to ~0.6 at 3.5-4 m and 0.65-0.75
  beyond**; mean psw ramps 0.37 -> 0.05 over 0-4 m while GT sidewalk is flat at ~0.05; GT coverage there is 97-100 % seen; 63 % of the
  0-5 m false flags sit on GT-drivable. Reading: x < ~4 m is the ego body / hood zone -- the current camera frame cannot see it.
  The r0 = 10 m gate removed that zone AND the ~70 %-precise 4-10 m band. No gate between 0 and 10 m was ever tried.

## 1. The only change
`R0 = (0, 2, 4, 6, 10, 20)` (was `(0, 10, 20)`): the gate grid of `pass1_gate` / `picks_for_gate`; the cross-fit grid becomes
6 x 7 x 7 = 294 points. Everything else is `mapsoft_g.py` byte-for-byte: the objective (GT violations, V = gt_off | gt_edge) under the
ADE-to-human constraint, the 5 episode folds (`default_rng(0)`), the deranged-map control arm, controls S-K0..S-K3, bars M1-M5.

## 2. Bars (verbatim MAPSOFT-G; PASS = M1-M5)
M1 sidewalk vs BASE separated below; M2 edge vs BASE separated below; M3 no four-family row worse; **M4 the DERANGED-map arm is NOT
separated below BASE on sidewalk or edge** (window-specificity); M5 the STOP / progress guard.
Beside, descriptive (no bar): OWN minus DER point estimates on sidewalk / edge; the fitted (r0, lam_s, lam_e) per fold; GT-T1 share.

## 3. Committed next steps
* **PASS** -> the own map is usable by the planner WITHOUT training (the PI's 10-25 directive, first delivery): roll MAPSEL at
  inference seed 1 (one GPU roll) and read the same instrument there; the authorised map-head fine-tune is then aimed at precision
  beyond 4 m only.
* **FAIL M4 again** (the deranged map does as well) -> excluding the ego zone is not enough: the own map beyond 4 m is not
  window-specific at ~70 % precision. The authorised fine-tune (map-only modules, planner bit-identical) is the lever, read with this
  same instrument; the 0-4 m zone is then also a training target.
* **FAIL M5 (only)** -> MAPSOFT-G's committed lever: the multi-objective fit constraining all four families.
* **FAIL M1/M2** -> name them; the gain is not there at any gate.
