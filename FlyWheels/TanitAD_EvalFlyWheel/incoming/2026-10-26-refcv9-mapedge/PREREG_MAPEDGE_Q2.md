# PREREG_MAPEDGE_Q2 -- select refcv8's fan away from its OWN drivable boundary (the "red edges"), 0 training (Master Mind, 2026-10-26 ~00:45 CET)

Sealed BEFORE any outcome exists: MAPEDGE Q1 (`PREREG_MAPEDGE.md`, sha256 `31a53010…`) has not run, and MAPSEL's s0 roll was still
rolling (no `MAPSEL_RESULT.json` for s0). Runs only under the conditions Q1 committed (sec. 4 there): on **INSTRUMENT-PASS** as the
headline arm; on **B1 pass / B2 or B3 fail** labelled *"geometric prior, not perception"*; **not at all on B1 fail**.

## 1. Arms (added to MAPSEL's machinery; everything else IDENTICAL to `PREREG_MAPSEL.md` + `AMENDMENT1`)
`mapedge_q2.py` imports `refcv9_mapsel/code/mapsel_score.py` UNCHANGED (its sealed code pins are re-checked by its own controls) and
adds two `(name, source, tier)` arms by wrapping `pass1` -- no MAPSEL definition changes:
* **OWN-BND-T12** (source `ownbnd`, tier 2): T1 = MAPSEL's OWN off-drivable flag (own P^(sidewalk) >= 128, AMENDMENT1 A1.1);
  T2 edge flag = any in-range FP5 point within 0.3 m (`EDGE_DIL` 3, square) of an **OWN-BND** cell, where OWN-BND =
  `mapedge.boundary(pdrv >= 128)` (inner boundary of the own drivable plane, outermost raster row/col cleared -- the SAME function
  Q1 scored). Lexicographic, each tier skipped if it would empty the eligible set; pick = argmax `sel_score_v3`; shipped stage.
* **DER-BND-T12** (source `derbnd`): the same with the DERANGED donor's `psw` / `pdrv` (MAPSEL's donor table).
* Every MAPSEL arm is recomputed in the same run.

## 2. Population, estimator, tier
S6-compliant windows with a GT map (MAPSEL's `S6c`), plus `S2c` for the four families. Compliance of the EMITTED staged plan vs the GT
raster. Paired episode-cluster bootstrap (`pair_read_r8.paired_cell`), B 2000, seed 0 -- "another draw of EPISODES" only; one training
seed; one inference seed (0). Tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).

## 3. Controls (any failure VOIDS)
* **Q2-K1 reproduction**: every MAPSEL arm's S6c compliance rates and `vs_BASE` cells in this run equal MAPSEL's own s0
  `MAPSEL_RESULT.json` exactly (the wrapper changes nothing it does not name).
* **Q2-K2**: every MAPSEL hard control in this run passes (as in MAPSEL's own reading).
* **Q2-K3 no-op identity**: an arm with the OWN-BND edge flag forced all-False equals OWN-T1's picks exactly.

## 4. Bars (for OWN-BND-T12)
* **E1** GT edge-contact rate separated BELOW BASE (paired CI upper < 0).
* **E2** = MAPSEL C2 (no four-family row, S6c and S2c, separated worse than BASE).
* **E3** DER-BND-T12 NOT separated below BASE on GT edge contact (else **INCONCLUSIVE-DER**; OWN minus DER printed).
* **E4** = MAPSEL C4 (STOP guard).
* **PASS = E1 & E2 & E3 & E4.** Reported beside: edge-contact minus the HUMAN floor (AMENDMENT1 A1.5), off-drivable and lane-crossing
  rates, OWN-T12 (decided-class edge) and GT-T12 (bound) for the same tier, fallback share at T2, kept-set size.

## 5. Committed next step
* **PASS** -> register OWN-BND-T12 as an opt-in inference row (T1 stamp) and carry the boundary edge into refcv9's map wiring
  (critic feature + calibrated rejection), then a NavSim read only with a PI go.
* **E1 fails, GT-T12 clears it** -> perception is the bottleneck at the curb: train the edge with a distance-transform / boundary loss
  (P3), GT-T12 = the bound.
* **E1 fails and GT-T12 fails too** -> the fan holds no edge-clear candidate the selector can reach: a fan-availability audit
  (share of windows with >= 1 GT-edge-clear eligible candidate) decides between a fan change and a cost term.
* **E1 passes, E2 or E4 fails** -> the edge rule costs driving quality: a soft edge cost (rank penalty) instead of a hard filter.
