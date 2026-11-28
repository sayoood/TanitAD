# RBSTACK: the stacking-combiner check (pre-registration, frozen by the MM before any RBSTACK number exists)

Trigger: NAVRBD read FINAL `RBD-NO-GAIN` + `RBD-FLOOR-HIGH` (card `PREREG_V8F_NAVRBD.md` sha256 881d7f3d, section 7 row RBD-NO-GAIN, first lever). Caveat 1 of `MM_READING_RBJOINT.md`: "Not JOINT" said that one combiner (vision probe with the ED features concatenated) failed; it did not test whether vision and text together carry more than text. This check is the fair combiner. Zero model training of the driving model; CPU; tier **T0 decodability, never driving performance**.

## 1. Population, target, folds (unchanged from RBJK)
* Primary target: the 3-class kinematic lat label (lane_keep / turn_left / turn_right) of the GT 6 s track, exactly as `navrbd/rbjk/rbjk.py` builds it (RBJK population: 9,236 strided windows / 395 episodes with `gt_ok`; fold training on the dense fold-train windows that have a label, as RBJK).
* Folds: `fold_of(clip, salt="rbjoint-fold", k=5)` (clip-grouped). Tree: `RD.fit_predict` with `RD.FEATS["ED"]` (rbdist_zt HGB, unchanged).
* Beside, no bar: the 5-way v9 `lat_cls` target on RB-JOINT's population (12,201 / 426).

## 2. Arms
* **ED**: the RBJK ED tree (reference).
* **V**: the vision probe alone. Probe = RB-JOINT's **RAW8** arm (`rbjoint.py` `make_net` / `fit_arm`, cached DINOv3 pool8x20 frames {t-3, t}, no ED input, 6 epochs, its own epoch selection), trained on the primary target.
* **EDV**: the ED tree with the probe's 3 class probabilities appended as features. **Nested**: for outer fold k, the test rows carry probabilities from a probe fit on the 4 other folds; each training row (fold j != k) carries probabilities from a probe fit on the 3 folds other than j and k. No probability on any row of the stack's fit set for fold k comes from a probe that saw fold k (5 outer + 20 inner fits per seed).
* **EDV_SHUF** (control): EDV with the probability rows permuted across windows (seed 0) inside every fit set and every test fold.
* Probe seeds: **0 and 1** (the probe is a trained net: the episode bootstrap is blind to its training variance).

## 3. Estimator and readings (committed)
Paired episode-cluster bootstrap (`taniteval.ci`), B 2,000, seed 0, on per-window correctness; answers "another draw of EPISODES" only. Margin M = 0.0383.

| reading | condition (at BOTH probe seeds; if the seeds disagree the lower reading holds) | next |
|---|---|---|
| **STACK-CARRIES** | EDV - ED lower bound above 0 and delta >= 0.0383 | the card's vision lever (RB window weight in the loss and/or the 8x20 grid input) is carded for the PI with this delta as its expected T0 ceiling |
| **STACK-WEAK** | EDV - ED separated, delta below 0.0383 | the vision lever is still carded, and the PI is told its expected size is below the bar |
| **STACK-SILENT** | EDV - ED not separated | at T0, vision adds nothing over the route text on this label; the vision lever is not carded on decodability grounds; the decision rests on the FLOOR-HIGH lever (more seeds of the N1D / N1R pair, PI go) |

## 4. Controls (each must hold or the reading is NOT-READ, with the reason)
* **K-REPRO**: the ED and S arms reproduce RBJK exactly (ED 0.6406..., S 0.4075..., `RESULT_RBJK.json`, |diff| <= 1e-9).
* **K-SHUF**: EDV_SHUF - ED below 0.0383 at both seeds.
* **K-NEST**: an assertion over the recorded fit-set folds of every probe proves section 2's nesting; a deliberate-regression mutant that feeds plain (non-nested) out-of-fold probabilities must turn it RED.
* **K-V**: V alone above the majority rate (0.4075), separated (the probe reads the frames at all).

## 5. Held-out windows (stage B, beside, no bar)
The same arms fit on all train RB windows, scored on the NAVRBD D1-HO windows (5,736 / 497) if the pooled 8x20 features and the ED columns for those windows can be built without re-selecting any episode; otherwise **NOT-YET with the reason**. Stage B never changes the section 3 reading.

## Log
* 2026-11-19 ~00:35Z MM: frozen before any RBSTACK code or number exists. Inputs: RBJK (`navrbd/rbjk/RESULT_RBJK.json`), RB-JOINT (`rbjoint/out/RESULT_RBJOINT.json`, card sha256 499860a2).
