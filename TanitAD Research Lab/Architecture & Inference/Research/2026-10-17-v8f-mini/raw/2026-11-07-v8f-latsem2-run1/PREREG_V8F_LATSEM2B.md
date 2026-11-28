# PREREG_V8F_LATSEM2B: the scene read with a permutation-calibrated instrument (Master Mind, 2026-11-07, written AFTER LATSEM2 run 1 and before any LATSEM2B code or statistic exists)

## Why
* LATSEM2 (`PREREG_V8F_LATSEM2.md`, `1325f21c`; IMPL_CHOICES `b17688dc`) run 1 is **INVALID**. Control 3 failed on 1 of 54 S1 rows, and control 4 failed on 3 of 6 S3 rows. The mechanism was measured in run 1's beside draws (`theory/latsem2/RESULT_LATSEM2.json`):
  * A single permuted-label fit judged by its bootstrap interval is "separated" on 18 of 360 classification draws (5.0 %) and 0 of 720 continuous draws. The control cannot pass reliably as committed.
  * Accelerate and brake move the imagined latent along an almost fixed direction, so a probe on shuffled labels still reads a nonzero S3. The episode bootstrap holds the probe fixed, so it cannot see this.
  * In the labels, lead gap and ego speed correlate at r +0.52. A gap probe can therefore read speed, and S3 then follows the imagined speed rather than the imagined gap.
* Run 1's numbers have been seen. This card's eval-141 rows are therefore labelled **post hoc**. The deciding read uses windows no LATSEM2 statistic has touched.

## Data
* **Deciding set:** WMACT's TRAIN windows (`theory/wmact/latents_train.pt`; 1,833 windows with 16 real frames, 141 clips, clips sha256 `e9e0e2c4`). These are M2b's own training clips, which is optimistic for S2. A target is decided here only if its label source exists for these clips. The sources and n are fixed in `IMPL_CHOICES_LATSEM2B.md`, frozen with its sha256 before any statistic.
* **Beside, post hoc:** eval-141, LATSEM2's frame bank and windows.
* Models M2b s0 / s1, zero training. Targets, folds, probes (ridge lambda 10, logistic C 0.1), the L1 shrink and the bootstrap (episode-cluster, B 2,000, seed 0) are LATSEM2's, unchanged.

## The instrument (the only changes from LATSEM2)
1. **Two-part separation.** A row is separated on the claimed side only if both hold:
   * its bootstrap 95 % interval excludes 0 on that side; AND
   * its point estimate lies beyond the 97.5th percentile (or below the 2.5th) of a permutation reference. The reference has R = 199 draws. Each draw moves the target labels across episodes by a distinct cyclic shift of the episode order (no fixed points), then refits the probe and recomputes the statistic. For S3 the lead gaps are moved across windows instead, as in control 4, and the gap probe is refitted.
2. **S3-PARTIAL decides ACTION-SEMANTIC-LEAD.** The gap probe is fitted, within the fit fold, on the residual of the lead gap after a linear regression on ego speed. Then S3-PARTIAL = imagined residual gap under "accelerate" minus "brake", at the longest horizon, on windows with a lead. LATSEM2's raw S3 is reported beside it.

## Controls (each must read its known value, or the read is INVALID)
1. The start-state query reads SKILL exactly 0.0 (LATSEM2 control 1).
2. Two identical action sequences give S3 and S3-PARTIAL exactly 0.0 (LATSEM2 control 2).
3. **Calibration:** 20 further permuted draws, with shifts outside the R draws, are each run through the full two-part rule as if real. Pooled over the 20 draws, the share of rows declared separated is at most 5 % in each family: S1 classification, S1 continuous, S2 and S3-PARTIAL.
4. **Speed removed:** on the fit fold, the correlation between the residual gap target and ego speed is 0 to 1e-6.

## Outcomes, committed now (per level and target; deciding set first, eval-141 beside)
* **SCENE-AWARE at level l:** ENCODES lead gap or agents within 30 m at both seeds on the deciding set, and the eval-141 estimate has the same sign.
* **SCENE-IMAGINING at level l:** IMAGINES lead gap at the longest horizon at both seeds on the deciding set, and the eval-141 estimate has the same sign.
* **ACTION-SEMANTIC-LEAD at level l:** S3-PARTIAL < 0, separated, at both seeds on the deciding set, and the eval-141 estimate has the same sign.
* A target without TRAIN labels is reported on eval-141 only, as post-hoc exploration, and decides nothing.
* **Next levers (LATSEM2's, unchanged):**
  * no level SCENE-AWARE ⇒ MRATE's W1, then a scene-target auxiliary under its own card;
  * SCENE-AWARE but not SCENE-IMAGINING ⇒ the reliance arm on that level.
* **INVALID:** a control fails.

## Scope
* Tier T0, a world-model diagnostic, never a driving claim. The estimator answers "another draw of episodes"; the two seeds answer "another training run". The permutation reference answers "would this appear with no relation between latent and label?"

## Log (dated, append-only after the freeze)
* 2026-11-07 13:21:36 (dev-box clock, from `date`), Master Mind. **Frozen:** the first 4611 bytes above this log, sha256 `bf6b37178818f778cf9220c4abac1a44de11a954d0e3a2b57f39e8e8b1bc48a8`, before any LATSEM2B code or statistic exists. Designed after LATSEM2 run 1 (INVALID, numbers seen); its eval-141 rows are post hoc.
