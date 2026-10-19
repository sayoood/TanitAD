# Amendment 3 to PREREG_HEADROW_PROBE: C2y and the proposal-index diagnostic, POST-HOC, UNREGISTERED, DESCRIPTIVE (they decide nothing)

**Parent:** `PREREG_HEADROW_PROBE.md` sha256 `1800705ef49333dffe0162ea5e51eb72fa1fe8fba57a0ea2f2869bf156762c55`; Amendment 1 `f98e24dea...`; Amendment 2 `662d228c...`.

**Written 2026-10-19 ~09:41, after the outcome tables AND the C2x numbers were read, BEFORE any C2y or index-diagnostic number exists.** Sections 2 to 7 of the parent are unchanged. Nothing here may change, rescue or void a per-row call, the outcome row or the next arm; it exists so the report can say honestly whether the registered C2 RED (8 of 8 cells) and the C2x RED (8 of 8 cells, a cross-set label permutation, points between +0.10 and +0.33 above 0.5) mean "the pipeline manufactures concordance" or "labels carry a scene-independent structure a probe can learn from the features".

## What C2x showed, and what Amendment 2's own rule says

Amendment 2 committed: reading (a) (the within-set permutation is not a null, the pipeline is clean) is supported iff at most 1 of 8 C2x cells is RED and all points are within +-0.01. **C2x: 8 of 8 RED, points +0.10 to +0.33. Reading (a) is NOT supported; by the committed rule the registered C2 REDs must be read as a possible pipeline artefact that all FEATURE-LIMITED calls inherit as a caveat, until the next test says otherwise.**

## Reading (c), the one this amendment tests

A cross-set permutation keeps the association between a proposal's POSITION in its set (index 0 to 63; the decoder emits ordered proposals, and the features of a position can carry that order) and its label distribution. If the labels depend on the proposal index (or any scene-independent proposal type that the features encode), a probe trained on set-permuted labels still learns that dependence and scores above 0.5 against the real labels, with no leak in the pipeline.

## C2y (exact)

Same fits as C2x (cold start, skip = 0, Adam lr 1e-3, weight decay 1e-4, 20 epochs, 256 sets per batch, 5 folds grouped by log, training seed 0, 1 replicate per (row, arm)), but the training labels are an ENTRY-LEVEL permutation: the 15,808 x 64 labels of the row are flattened and randomly permuted (`numpy default_rng(20261019 + 177 + col)`), so a proposal's label is independent of its set, its position and its features; the valid mask is kept as the entry's own. Scoring as C2x (OOF logit vs the REAL labels and real valid mask, all-64 concordance, the same estimator).

**Reading rule, committed now:** reading (c) is SUPPORTED, and the pipeline-artefact reading is REJECTED, iff at most 1 of the 8 C2y cells is RED (lower bound of concordance - 0.5 > 0) AND every cell's point lies within +-0.01 of 0.5. If C2y is also RED in 2 or more cells, the pipeline-artefact reading stands, the FEATURE calls carry it as a stated caveat, and no number of this probe is quotable without it.

## Proposal-index diagnostic (no training)

The all-64 concordance of the score `z = proposal index` (0..63) against the real labels of each row (EP among safe pairs), with the same estimator, per row. A value far from 0.5 says the labels depend on the proposal's position. Reported, not thresholded.

Files: `code/c2y.py` (new), `raw/f10/c2y.json`, `raw/f10/index_diag.json`.
