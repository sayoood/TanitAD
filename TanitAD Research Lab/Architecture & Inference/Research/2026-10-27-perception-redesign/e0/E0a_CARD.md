# E0a card -- box over-fit gap, train vs held-out AP@2 m (REFCV9_DESIGN §4.1 / §2.5; gates augmentation arm A1)

**Question.** Does refcv8's box head fit TRAIN windows better than held-out ones (over-fit), so that augmentation can help?
**Frozen design.** refcv8 EMA-48,852 (ckpt md5 `b4364029…`), banked PERC8 packs produced on Thor (no new GPU pass): TRAIN = `perc8/out/T_nosync/train_diag.packs.pkl` (139 train episodes × 8 windows, 1,074 box3d packs); HELD = `perc8/out/A_nosync/perc_det.pkl` (eval139, 4,532 packs).
AP@2 m = `pbn_q3_box.py`'s definition (`detection_metrics.greedy_rows(pk, 2.0)` + all-point AP, class-agnostic), pooled and by band 0-20 / 20-40 / 40-60 m. **Equal counts:** HELD is cut to exactly 1,074 packs (8 evenly spaced windows per episode by `ws`, then a seeded drop; 20 draws for sensitivity).
gap = AP_train − AP_heldout; 95 % interval = TWO-SAMPLE episode-cluster bootstrap (each group's episodes resampled independently), B 1000, seed 0. **Decision rule: A1 RUNS iff the pooled gap's lower bound > 0** (inside the half-width ⇒ only photometric augmentation enters the full run).
**Controls.** K_REDERIVE: HELD pooled AP@2 m over all 4,532 packs re-derives the banked 0.328 (±0.0015). K_NULL: 20 random half-splits of the HELD episodes, same code and counts, the gap interval excludes 0 in ≤ 2 of 20. Tests (`test_e0a.py`, 4 PASS): perfect detector AP = 1.0; detector 6 m off AP = 0.0; one of two GT found at top confidence AP = 0.5; the equal-count subset is exact and seed-stable.
**Disclosed limits.** Train packs come from `run_diag_r8.py`, HELD packs from `perc_pass_r8.py`: both read the same EMA weights with `R8_NO_POSE_SYNC=1` and the tree's `detection_metrics`, but harness equivalence beyond that is UNVERIFIED. The TRAIN windows are 8 per episode on a different window rule than the eval grid. One training seed (a gap is a property of this run, not of the recipe).
**Result (MEASURED, `raw/E0A_RESULT.json`; T0 label level; one training seed; ran 32 min on a shared CPU).** Controls PASS: pooled held-out AP@2 m re-derives 0.3282 (banked 0.328); K_NULL 0 of 20 held-out half-splits exclude 0.

| AP@2 m | train (1,074 packs) | held-out, equal count (draw 0) | gap | 95 % interval (two-sample episode bootstrap, B 1000) |
|---|---|---|---|---|
| pooled | 0.3462 | 0.3329 | **+0.0134** | [-0.0244, +0.0500] |
| 0-20 m | 0.5694 | 0.5508 | +0.0186 | [-0.0323, +0.0699] |
| 20-40 m | 0.2402 | 0.2315 | +0.0087 | [-0.0281, +0.0451] |
| 40-60 m | 0.0950 | 0.0926 | +0.0024 | [-0.0212, +0.0259] |

Over the 20 equal-count draws the pooled point gap is +0.011 .. +0.015. **Decision: the gap is inside the bootstrap half-width (lower bound -0.024 < 0), so A1 does NOT run; only photometric augmentation enters the full run, disclosed (design §2.5).** Read it as "no detectable over-fit at this n", not "no over-fit": the interval admits a gap up to +0.05, and the point estimate is positive at every range band and every draw. Harness-equivalence caveat above still applies.
