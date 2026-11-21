# PREREG_MAPSEL -- AMENDMENT 1 (sealed separately; supersedes the named parts of PREREG_MAPSEL.md, sha256 `e9be326e692a6402e70bf79c98097770a5cec24259f27d1d6a01d179c64a3bc9`)

*Research Lab agent, 2026-10-25 evening, on the Master Mind's APPROVAL of `PREREG_MAPSEL_AMENDMENT1_DRAFT.md` (~23:30 CET) with four additions (items 1-4 below). Sealed (sha256 in `PREREG_MAPSEL_AMENDMENT1.sha256`) BEFORE any per-candidate map feature, any roll, any arm pick and
the amended human / sensitivity control exist on eval data. Computed on eval data BEFORE this seal (all DESCRIPTIVE, human-path only, no candidate): the sealed-class-set human control (0.0926, FAILED), its per-class decomposition
(nocls 0.0735, edge 0.0291, sidewalk 0.0125) and the per-episode nocls list. Unit tests and a PLANTED synthetic world (no eval map data) were run on the amended code.*

## A1.0 Why and the honesty clause
The sealed K_HUMAN gate (human off-drivable <= 2 %) FAILED at **0.0926**; the experiment STOPPED as written (no GPU taken). The cause was the class set, not the geometry (a +-1 window GT shift does not lower the rate; centre-only 0.044): the GT raster has UNLABELLED
`nocls` holes (7.35 % of windows, near field, 8 of 136 episodes) and `edge` cells that humans legitimately touch. **The amended class set was chosen AFTER seeing those numbers**, so the 2 % human gate is demoted to a SANITY BOUND (it can no longer be read as a blind
control) and a BLIND sensitivity control (A1.4) is added and sealed here, to be computed after the seal.

## A1.1 T1 "off-drivable" (replaces PREREG sec. 3.2 `off_drv`)
* **GT:** a check point (FP5, any path sample, in range) on a SEEN **SIDEWALK cell (code 7)** = a violation. **nocls (code 0) and not-seen (255) are UNKNOWN: never a violation.** Edge (code 5) is NOT a T1 violation (it stays Tier 2).
* **COVERAGE (new, reported per arm and for HUMAN):** the share of in-range check points of the EMITTED staged plan that land on nocls cells and on not-seen cells (GT raster), so the reader sees how much of the plan the T1 check could not judge.
* **OWN mirror:** the model's OWN class posterior of **sidewalk (code 7)** under the declared `prior_corrected` rule: `psw_u8 = round(255 * p^_sidewalk)`, `p^ = softmax(z - log w)`. Violation iff **`psw_u8 >= 128` (P^(sidewalk) >= 0.5) at any check point**.
  The roll banks, per window: `pdrv` (P^ of the DRIVABLE set; K_LIVE liveness only), `pedge`, **`psw`**, `code` (decided class), and the GT code plane.
* Edge contact (0.3 m margin) and lane crossing (centre line, +-0.2 m band, outside lane-change windows) are UNCHANGED (GT: codes 5 / 2; OWN: decided codes 5 / 2). The lane-change exception is unchanged.

## A1.2 OWN-CAL-T1 (replaces PREREG sec. 3.3's `pmin` definition)
Score of a candidate `s(c)` = the MAXIMUM over its in-range FP5 samples of `psw_u8 / 255`. **tau = the 99th-percentile (`numpy.quantile(..., 0.99, method='higher')`) of the HUMAN paths' `s`** (S6-valid human windows, same window's own raster), so that >= 99 % of
human paths pass (`s <= tau`). A candidate is rejected iff `s(c) > tau`, unless none survive (fallback). Computed once, before any candidate is scored, from privileged GT trajectories and the model's own map, **no outcome metric**; **calibrated ON THE EVAL human paths
(privileged GT trajectories); a train-side calibration is not available in this roll** -> labelled privileged-calibrated. `DER-CAL-T1` mirrors it with the deranged donor's `psw`. The hard rule's plan-level false-alarm rate on human paths
(`s >= 0.502`) is reported beside tau. (If the human 99th percentile of `s` is 1.0 the CAL arm is inert; that is a finding, not a defect.)

## A1.3 K_HUMAN is now a SANITY BOUND (replaces PREREG sec. 5 K_HUMAN)
Human off-drivable (amended definition) <= 2.0 % on S6-compliant GT windows and parity with the banked S6 `g` < 1e-3 m. Descriptively measured before the seal: 0.0125 (sidewalk only).

## A1.4 K_SENS -- BLIND sensitivity control (new; computed AFTER this seal, 0 GPU, before any GPU is taken)
The human path shifted **laterally by +3.5 m and by -3.5 m** (one lane; a rigid translation of all 8 slots in the t0 ego frame, y LEFT) is scored by the SAME amended off-drivable metric on the S6-compliant GT windows.
**Pass iff, for at least one side, the shifted paths' off-drivable rate is SEPARATED ABOVE the unshifted human's** (`pair_read_r8.paired_cell`, paired episode-cluster bootstrap B 2000 seed 0, new = shifted, ref = unshifted; delta CI lower bound > 0).
**If neither side is separated above, the metric cannot see violations and the run is VOID** (the chain stops before taking the GPU). Both sides' rates and CIs are reported either way.

## A1.5 T2 / T3 are judged RELATIVE to the human floors
The human floors measured on the sealed-set control (edge-contact 0.1485, lane-crossing 0.1986; edge and lane definitions are unchanged by this amendment) are reported beside every arm as **arm rate minus HUMAN rate**; T2 / T3 get NO absolute bar.
The verdict clauses C1-C4 stay exactly as sealed; C1 now reads the amended (sidewalk) off-drivable rate.

## A1.6 Prediction update (not a bar)
Under the amended rule the human reads ~1.3 % and BASE's off-drivable rate is expected to be low, so T1 may have LITTLE HEADROOM. If BASE's rate on the S6-compliant GT windows is <= ~3 %, a FAIL on C1 means "no headroom on this metric", and the mechanical next-lever template
(sec. 6.3, unchanged) reads the GT-T1 bound to say whether the map is the bottleneck.

## A1.7 Code changes (re-pinned in `CODE_SHA256_AT_AMENDMENT1.txt`; no definition outside A1.1-A1.5 changes)
`mapsel_lib.py` (BAD_CODES = (7,), `masks_own` on `psw`, `Geom.coverage`, `own_cont` on `psw`), `mapsel_roll.py` (+`psw` plane; both seals required), `mapsel_score.py` (psw / tau / coverage / minus-HUMAN / K_SENS in the VOID list / both seals),
`mapsel_human_control.py` (amended metric + K_SENS), `test_mapsel_lib.py` (amended analytic targets; the sealed class set `(0,5,7)` is a THIRD deliberate-regression arm), `chain_mapsel.sh` (both seals; STOP on K_SENS),
`smoke/roll_capture_test.py` (planted `psw` plane). Everything else (arms, E9 eligibility, shipped stage, C1-C4, DER control, estimator, populations, budget) stays as sealed.

## A1.8 Disclosure
No per-candidate map feature, no roll output and no arm pick exist on eval data at this seal.
