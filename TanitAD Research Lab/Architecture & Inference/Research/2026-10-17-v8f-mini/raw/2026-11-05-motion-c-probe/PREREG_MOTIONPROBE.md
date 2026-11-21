# PREREG_MOTIONPROBE — do cached frozen DINOv3-L token grids, stacked over frames, carry ego motion?

Written 2026-11-05, BEFORE any fit on real data. Its sha256 is in `FROZEN_SHA.txt`.
Evidence class of everything this probe will report: MEASURED (ours, artifact paths in RESULT.md).

## 1. Question and what it gates

Do the cached frozen DINOv3-L token grids (4 x 10 grid, 40 tokens x 1024 per frame, frames 0.2 s apart),
stacked over four consecutive frames (t-3..t, a 0.6 s span), carry EGO MOTION beyond what the single frame t carries?

The answer gates the cheap perception change "a motion adapter over cached tokens". The earlier SPEEDREAD
probe averaged each frame's 40 tokens into one vector (so it could not see displacement) and targeted
future speed cells (not current motion). This probe fixes both.

## 2. Data (read only) and sample

* Cache: `F:/TanitAD-scratch/speedread/cache/train/` (4,572 TRAIN clips, `manifest.jsonl`, `clips/<clip>.pt`).
  Keys used: `pool4x10` fp8 `[T,40,1024]`, `pool8x20` fp8 `[T,160,1024]`, `v` `[T]` (logged speed, m/s), `a`.
* Sample: 600 clips drawn without replacement from the 4,572, `numpy.random.default_rng(20261105)`, from the
  manifest order. If a clip file fails to load or its `T` differs from the manifest it is dropped and the drop is
  logged (no replacement draw).
* eval-141 isolation: the script asserts the sampled clip ids are disjoint from the ids in
  `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/manifest.jsonl` and aborts otherwise.
* Windows: per clip, every 2nd frame `t = 3, 5, 7, ...` with `t < T`, keeping a window only if `v[t]` and `v[t-3]`
  are both finite. Frames of a window are t-3, t-2, t-1, t (0.2 s apart).
* Folds: the 600 sampled clips are shuffled with seed 20261105 and assigned fold = rank mod 5 (120 clips per
  fold). 5-fold cross-validation BY CLIP: no clip contributes windows to both train and test of any fold.
* Second grid (`pool8x20`): a 300-clip subset (the first 60 clips of each fold) because the 8 x 20 files are 4x
  larger; stated in RESULT.md. Everything below that says "4 x 10" is the main analysis; the 8 x 20 grid is
  supplementary (section 8).

## 3. Targets

* `v` = v(t), current logged speed.
* `dv` = v(t) - v(t-3), speed change over 0.6 s.
Labels use ego signals (binding rule: labels may use ego); the INPUTS are vision tokens only. No ego state, no
pose, no `a`, no `steer`/`yaw_rate`, and no `kin3`/`p10` enter any feature set.

## 4. Feature sets (identical windows for all)

Let f_j be frame j's token grid `[40,1024]` (fp8 cast to float32).

| name | content | dims |
|---|---|---|
| F1mean | mean over the 40 tokens of frame t (SPEEDREAD's form), raw 1024 | 1,024 |
| F1 | frame t, per-cell PCA(32) | 1,280 |
| F4 | frames t-3..t, each per-cell PCA(32), stacked oldest to newest | 5,120 |
| D3 | [f_t - f_{t-1}, f_{t-1} - f_{t-2}, f_{t-2} - f_{t-3}] per cell, projected with the DIFFERENCE basis (below) | 3,840 |
| D3f (sensitivity) | the same three differences, projected with the FRAME basis | 3,840 |

Reduction: per-cell PCA (cell = one of the 40 grid positions; 1024 -> 32 dims per cell, 1,280 per frame), fitted on
the TRAIN folds only, separately for every outer fold.
* Frame basis: covariance of the frames of the train-fold clips (all frames, not only window frames), per cell.
  Used for F1, F4, D3f.
* Difference basis: covariance of consecutive-frame differences f_j - f_{j-1} of the train-fold clips, per cell.
  Used for D3. Reason: a basis fitted to scene content can discard exactly the low-variance directions in which a
  displacement lives; D3 is therefore given a basis fitted to what it will read. D3 is the PRIMARY difference
  feature; D3f is a sensitivity row and CANNOT change the verdict (if D3f alone would have flipped it, RESULT.md
  says so).
* PCA is exact (eigendecomposition of the accumulated per-cell covariance in float32, top 32 eigenvectors), not a
  random projection. The fraction of variance the 32 kept dims retain (mean over cells, per fold) is reported.
* Every feature column is standardised with train-fold mean/SD before any readout.

## 5. Readouts

* Ridge, closed form. Alpha = g x n_train with g in 10^{-2, -1.5, ..., 2.5} (10 values), chosen per target by an
  inner split BY CLIP of the train clips (20 % of train clips held out, seed 20261105 + 100 + fold); the final fit
  uses all train windows of the fold at the chosen alpha. If the chosen g sits on a grid edge it is logged.
* MLP, 2 x 256 ReLU hidden layers, dropout 0.1, joint two-output head on standardised (v, dv); Adam lr 1e-3, weight
  decay 1e-4, batch 256, at most 25 epochs, early stopping (patience 4) on the same inner-split validation MSE, the
  best-epoch weights are used; one seed (20261105 + fold). The MLP is run for F1mean, F1, F4, D3 and the K2/K3
  controls of F4 and D3, and the K1 control on F4. It is NOT run on D3f.
* No temporal-conv readout (optional in the brief; skipped to stay inside the compute budget, stated here).

## 6. Metrics and estimator

* R² = 1 - SSE / SST on the TEST windows of each fold, SST taken about that fold's own subset mean, for the subset
  "all windows" and the subset "band" (3 <= v(t) <= 8 m/s). The model is always trained on all train windows; the
  band only restricts which test windows are scored. RMSE (m/s) is reported beside R².
* Reported: mean and SD (ddof 1) over the 5 folds, plus n windows per cell (per fold and total).
* Estimator: fold mean +/- SD by clip. It answers "would another draw of CLIPS say this?" ONLY. It does not
  answer "would another training run say this?" (the ridge fit is deterministic; the MLP has one seed) and not
  "would another inference run say this?" (nothing stochastic at inference). A paired difference between feature
  sets is taken fold by fold on identical test windows.
* Band R² caveat (stated before the data): the band is selected on the target `v` itself, so SST_band is small and
  any model that predicts near the global mean scores a strongly NEGATIVE band R² on `v`. Band R² gains therefore
  read as RMSE reductions relative to the band's own spread. K1's validity check is on the ALL-windows R².

## 7. Controls (they must read their known values or the verdict is INVALID)

* K1 permuted labels (AMENDED, see section 12): the (v, dv) rows are permuted ONCE across ALL windows (seed
  20261105 + 200) before any split. Each feature set runs through the standard pipeline on the permuted targets
  (ridge chooses its own alpha on the inner split, the MLP early-stops on the inner split) and is scored on the
  permuted test targets. A window shared by train and test (a leak) would carry the same permuted label on both
  sides and read positive; a metric or alignment defect reads the same way. Known value: ALL-windows R² within
  [-0.05, +0.03]. Run for every ridge cell; for the MLP on F4. A K1 cell outside the range makes the whole verdict
  INVALID (pipeline defect), whatever the other numbers say.
* K2 foreign history: frames t-3..t-1 are taken from a DIFFERENT clip of the same fold (cyclic donor = the next
  clip in that fold's list; donor frame index = same offset, clamped to the donor's length), frame t is kept. F4 and
  D3 are rebuilt with the same fold bases. Known value: the gain over F1 must vanish.
* K3 shuffled order: for every window independently, the four frames are put in a random order (seed
  20261105 + 7); F4 stacks them in that order, D3 differences consecutive slots of that order. Known value: a
  motion-borne gain must shrink. (A single fixed permutation for all windows is NOT used: it would only relabel the
  columns of F4 and change nothing.)
* Positive/negative plumbing check on synthetic data (planted displacement vs none) is run before the real run and
  banked in `selftest/`; it uses no real token.

## 8. Second grid (supplementary, not rule-bearing)

`pool8x20` (160 cells), 300-clip subset, RIDGE only, feature sets F1 and D3 (differences of frame-projected
tokens), K1 included, K2/K3 NOT repeated. Reduction: a PCA basis POOLED across the 160 cells (one 1024 x 1024
covariance from every 2nd frame and every 2nd cell of the train-fold clips), 8 dims per cell so the per-frame
dimension stays 1,280. A 4 x 10 matched reference on the same 300 clips (F1 and D3f, ridge) is computed so the two
grids are compared on the same clips. Because the 8 x 20 run has no K2/K3, it cannot produce PRESENT; it reports
the gain only.

## 9. DECISION RULE (committed; applies to the band 3-8 m/s, target v, and separately to dv)

Per feature set and target, take the BEST readout = the readout (ridge or MLP) with the higher mean band R² over
the 5 folds. Define
* R1 = best-readout band R² of F1.
* s* = the feature set in {F4, D3} with the higher best-readout band R²; g = R(s*) - R1.
* gK3 = (best-readout band R² of s*_K3) - R1, and gK2 = (best-readout band R² of s*_K2) - R1. The control is
  given its best readout too (the more conservative choice).

Verdict:
* INVALID if any K1 cell breaks its range (section 7).
* **ABSENT** if g < 0.05.
* **PRESENT** if g >= 0.15 AND gK3 <= 0.5 g (the gain at least halves under K3) AND gK2 <= max(0.02, 0.10 g)
  ("vanishes" under K2 = at least 90 % of the gain gone, with a 0.02 floor; AMENDED, see section 12).
* **WEAK** otherwise (including 0.05 <= g < 0.15, and g >= 0.15 with a control that fails: such a gain is not
  shown to be motion-borne).
Margins: no goalpost moves after the data. Anything computed beyond this section is labelled post-hoc in RESULT.md
and cannot change the verdict.

## 10. What each outcome implies (written before the data)

* PRESENT (on v): the cached token grids carry motion that a single frame does not; a cheap motion adapter that
  reads stacked or differenced cached tokens is viable as a perception change. Next arm: a small trained adapter
  (temporal conv or attention over the 4 x 40 x 1024 cache) with an in-loop speed readout, validated on the
  ladder rig, scored on the 3-8 m/s band.
* WEAK: something is readable but not shown to be motion-borne, or smaller than the bar. Next arm: the cheapest
  discriminator, either a stronger readout (unreduced 1024-d differences, temporal conv) to see whether the PCA
  bottleneck is hiding signal, or a pixel-difference CNN control on the same windows.
* ABSENT: under per-cell PCA(32) plus ridge/MLP, stacking or differencing cached tokens adds less than 0.05 band
  R² for current speed over frame t alone; motion has to come from pixels (a small CNN on frame differences,
  fused with the frozen tokens) or from encoder adaptation (fine-tuned or temporally trained encoder).
In every case the result is a statement about THIS reduction and THESE readouts; "absent" is not a statement about
what a trained adapter over the full 1024-d tokens could extract. The PCA bottleneck is the stated limitation.

## 11. Compute and hygiene

CPU only (`CUDA_VISIBLE_DEVICES=-1`), threads <= 8, RAM target below 5 GB (about 6 GB were free on the shared
box at launch). Scratch on `F:/TanitAD-scratch/motion_c/probe/`. No ssh, no box 2, no driving-model training.
Code: `code/motion_probe.py`; the run records the sha256 of the code file and of this document in
`raw/MOTIONPROBE.json`. Any change to the code after this freeze is listed in RESULT.md.

## 12. Amendment 1 (2026-11-05, made BEFORE any fit on real data)

Why: the synthetic plumbing check (planted displacement, 200 synthetic clips, no real token) showed that K1 as first
written (train labels permuted inside the train windows, alpha reused from the real target) read -0.24 (v) and -0.15 (dv)
for D3. There the real signal is strong enough that the alpha chosen is the smallest on the grid, and a model at a low
alpha overfits permuted noise: that is overfit, not a pipeline defect, and it would have issued INVALID on a clean
pipeline. Changes, and only these: (a) K1 as written in section 7; (b) a sensitivity row F1d (frame t projected with the
DIFFERENCE basis, 1,280 dims; ridge and MLP) added beside F1, so that a D3 gain that is only basis content can be told
apart from motion; F1d is NOT in the decision rule. No threshold in section 9 changed. The initial text is kept as
`PREREG_MOTIONPROBE_v0.md`; both shas are in `FROZEN_SHA.txt`.

## 13. Amendment 2 (2026-11-05, made BEFORE any fit on real data)

Why: in the synthetic plumbing check (planted displacement, 120 clips, no real token) the verdict machinery returned WEAK
for a planted gain of +2.27 band R² because the K2 control sat +0.074 above F1. That is the size of the band-R² noise
between two models that have no signal on `v` (their fold SD was 0.3-0.6: the band is selected on `v`, so the score of a
no-signal model is dominated by its mean offset), not a motion gain. Change, and only this: the K2 criterion in section 9
reads gK2 <= max(0.02, 0.10 g) instead of gK2 <= 0.02. The thresholds on g (0.05, 0.15) and on gK3 (0.5 g) are unchanged.
Initial texts kept as `PREREG_MOTIONPROBE_v0.md` and `PREREG_MOTIONPROBE_v1.md`; all shas are in `FROZEN_SHA.txt`.
