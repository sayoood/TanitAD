# AMENDMENT_LATSEM2B_1: a permutation scheme that the fold sizes allow, and the details the card left open (Master Mind, 2026-11-07, written before any LATSEM2B statistic exists)

## Why
* `PREREG_V8F_LATSEM2B.md` (`bf6b3717`) asks for R = 199 draws plus 20 calibration draws, each "a distinct cyclic shift of the episode order". That is 219 distinct shifts. A fixed order of n episodes has only n - 1 non-identity shifts.
* The shift must act inside the fit fold, as LATSEM2's `perm_labels` does, so that the evaluation fold's labels never reach the fit.
* The fit folds hold few labelled episodes: lead_gap 29 / 30 on TRAIN. That allows at most 28 or 29 shifts.
* The implementer stopped before computing anything (`theory/latsem2b/HANDOFF_LATSEM2B.md`). No statistic of this card exists.

## The scheme (replaces "a distinct cyclic shift of the episode order")
* Draw s puts the fit fold's labelled episodes in a random order (numpy `default_rng(s)`). Each episode then receives the labels of the next episode in that order, the last receiving the first's.
  * No episode keeps its own labels.
  * Labels move between episodes as LATSEM2's `perm_labels` moves them (run 1's code, unchanged).
* The reference uses seeds 1 to 199; the calibration uses seeds 1001 to 1020.

## Points fixed now
1. Probes are refitted per fold and per (seed, level, target) on the permuted fit-fold labels. They are scored on the other fold against its TRUE labels, with the baseline taken from the TRUE fit-fold mean or majority, as in run 1's extra draws.
   * The S2 reference passes the permuted probes through the S2 queries and scores them against the true future labels.
2. **Claimed side:** above 0 for S1 and S2, below 0 for S3-PARTIAL.
   * A calibration draw is declared separated when both hold:
     * its own bootstrap interval (probes fixed) excludes 0 on that side;
     * its point lies beyond that cell's 97.5th percentile of the R draws (2.5th for S3-PARTIAL).
   * Control 3's families pool seeds, levels and all 9 targets, as run 1 did.
3. **S3-PARTIAL reference:** as the card states, plus the residualisation step. The fit fold's lead-frame gaps are permuted across frames (run 1's control 4). They are then residualised on each frame's own ego speed, and the probe is refitted.
4. **Ego speed:** `native_tracks.pt` column 3 at raw row 2g, the field `latsem2/beside_gap_speed.py` used.
5. **eval-141 (beside, post hoc)** gets the same instrument, with its own reference.
* Disclosed, deciding nothing:
  * On TRAIN, the label-alignment check peaks at a -1 raw-frame offset (r 0.682, against 0.652 at 0). Eval peaked at 0.
  * The L1 shrink factor was fitted on these same TRAIN windows, so it is in-sample on the deciding set.

## Log (dated, append-only after the freeze)
* 2026-11-07 13:37:58 (dev-box clock, from `date`), Master Mind. **Frozen:** the first 2778 bytes above this log, sha256 `120072ffb46081bec267fc4f515e5704848272792790eb2577f357c8739ece53`, before any LATSEM2B statistic exists.
