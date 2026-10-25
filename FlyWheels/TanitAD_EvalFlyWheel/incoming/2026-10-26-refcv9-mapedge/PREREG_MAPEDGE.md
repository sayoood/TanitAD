# PREREG_MAPEDGE -- can refcv8's OWN map find the "red edges" through its drivable BOUNDARY? (Master Mind, 2026-10-26 ~00:25 CET)

Sealed BEFORE any s0 raster of the MAPSEL roll is read (that roll was at 880 / 4,754 windows when this was written; no MAPSEL s0 outcome
exists). 0 training, 0 GPU: reads the rasters MAPSEL banks per window (`F:/refcv9_mapsel/rast_s0/e*_w*.npz`: `pdrv`, `pedge`, `psw`,
`code`, `gt`; 1000 x 600 cells, 10 cm, ego frame at t0, x forward 0..100 m, y -30..+30 m; layout per `PREREG_MAPSEL.md` sec. 3.1).

## 0. Why
PI 2026-10-25: *"the trajectory must be in the drivable area, should avoid red edges and preferably within the lane."* The red edges are
class 5 (`edge`) of the 10 cm head. MEASURED (perc8, MAPSEL prereg sec. 3.4): refcv8's own `edge` IoU is ~0.001, so tier 2 cannot use
the decided `edge` class. But the own DRIVABLE plane is far better (IoU 0.605 at 0-20 m), and a road edge is, geometrically, where
drivable stops. Hypothesis: **the inner boundary of the own drivable plane recovers the GT edge cells far better than the own edge
class does, and the recovery is window-specific (not a generic "edges sit at +-y" prior).**

## 1. Definitions (all fixed)
* **Region**: cells with x in [0, 60) m (rows 0..599) -- the band where MAPSEL's IoU is reported and where 6 s plans live; bands
  0-20 / 20-40 / 40-60 m reported beside, not barred. Cells in the outermost row/column of the raster are never boundary cells.
* **GT edge cells** `E_gt` = `gt == 5`. **Judged GT cells** `J` = `gt` not in {0 (nocls), 255 (not seen)}.
* **Edge candidates X** (bool planes):
  * `OWN-CODE5` = `code == 5` (the decided own class, MAPSEL's OWN edge);
  * `OWN-BND` = inner boundary of own drivable `D = pdrv >= 128`: `D & dilate(~D, 1)` (square, 8-neighbourhood);
  * `DER-BND` = the same boundary on the `pdrv` of a DERANGED donor window: cyclic NEXT episode (sorted by episode index), position-
    matched window `min(p, n_donor - 1)` (MAPSEL's derangement rule);
  * `GT-BND` (reference ceiling, not an arm) = inner boundary of `G = gt in {1,2,3,4,6}` (DRIVABLE codes) restricted to `J`.
* **Tolerance** 0.3 m = 3 cells (MAPSEL's `EDGE_DIL`), square dilation (`mapsel_lib.dilate`).
* **Per-window recall** `R(X)` = |E_gt & dilate(X, 3)| / |E_gt| over the region (windows with |E_gt| >= 20 cells only);
  **per-window precision** `P(X)` = |X & J & dilate(E_gt, 3)| / |X & J| (windows with |X & J| >= 20 only; else that window is
  missing for that X). Population = every banked window that has the cell counts above; **not** restricted to S6 (this is a
  perception-instrument question, not a planning one; stated).
* **Estimator**: per-window values averaged with window weights within episode then over windows (pooled window mean); paired
  differences on the windows where both are defined; episode-cluster bootstrap over episodes, B 2000, seed 0 -- "another draw of
  EPISODES" only; one training seed; the map head is deterministic given the frames (no inference seed involved).

## 2. Controls (a failing one VOIDS the run)
* **K1 analytic world**: a synthetic straight road `pdrv = 255` for |y| <= 3.5 m else 0, GT edge cells at the two first non-drivable
  columns (GT `1` inside the road, `5` on those columns, `7` beyond): R(OWN-BND) = 1.0 and P(OWN-BND) = 1.0 exactly.
* **K2 shifted world**: the same road shifted 1.0 m laterally against the GT: R = 0.0 exactly.
* **K3 identity**: the paired bootstrap of OWN-BND minus itself reads delta 0, CI [0, 0].
* **K4 donor**: no window is its own donor; every donor is from a different episode.
* **K5 decode**: every raster has the five planes at [1000, 600] uint8, and `pdrv` is non-constant on >= 95 % of windows (an all-zero
  bank must not score).

## 3. Bars (INSTRUMENT, Q1)
* **B1** R(OWN-BND) - R(OWN-CODE5) separated > 0.
* **B2** R(OWN-BND) - R(DER-BND) separated > 0 (window-specific, not geometry prior).
* **B3** P(OWN-BND) - P(DER-BND) separated > 0.
* **INSTRUMENT-PASS** = B1 & B2 & B3. Reported beside: R and P for every X per band, GT-BND as the ceiling, F at tolerance
  `2PR/(P+R)` on window means, the share of `OWN-BND` cells that fall on GT `nocls` / not-seen (unjudgeable).
* Prediction before data (not a bar): R(OWN-CODE5) < 0.05; R(OWN-BND) 0.3-0.6; GT-BND recall < 0.9 (edges are not only drivable
  boundaries).

## 4. Committed next step (both outcomes)
* **INSTRUMENT-PASS** -> Q2, a 0-GPU selection arm through MAPSEL's own scorer machinery (same fan, eligibility, `sel_score_v3`,
  shipped stage, S6-compliant GT windows): `OWN-BND-T12` = MAPSEL's OWN-T1 then keep candidates with no FP5 point within 0.3 m of an
  OWN-BND cell (skipped if it empties the set), with `DER-BND-T12` as its deranged control. Bars: **E1** GT edge-contact rate of the
  EMITTED staged plan separated BELOW BASE; **E2** = MAPSEL C2 (no four-family row separated worse); **E3** = DER-BND-T12 NOT separated
  below BASE on GT edge contact (else INCONCLUSIVE-DER); **E4** = MAPSEL C4 stop guard. Edge contact is read relative to the human floor
  (MAPSEL A1.5). Q2's code and its seal come in a separate file written before Q2 is run.
* **B1 fails** -> the boundary does not beat the decided class: the edge needs a TRAINED signal -> P3 thin-class recall (class-
  balanced / distance-transform edge loss) is the lever; Q2 does not run on OWN-BND.
* **B1 passes, B2 or B3 fails** -> the boundary is a geometric prior, not perception: Q2 runs with the PRIOR as the arm (honestly
  labelled), and perception work (drivable precision at the curb) is the lever.
