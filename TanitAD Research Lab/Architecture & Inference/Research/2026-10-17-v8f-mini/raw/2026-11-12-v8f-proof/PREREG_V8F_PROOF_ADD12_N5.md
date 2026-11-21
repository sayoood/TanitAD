# Note N5 to PREREG_V8F_PROOF_ADD12 (SPATIAL): stage 0c, the read of the CHANGE of the lead gap (2026-11-13)

Written before any number of stage 0c exists. Written AFTER stage 0b was read (`spatial/s0/VERDICT_S0.md`): this is the next lever that stage 0b leaves, not a re-read of it. Stage 0b stands as quoted.

## 1. Why (MEASURED, stage 0b, T0)

The r-b read asked for the LEVEL of the lead gap 2.0 s ahead. From the grid state the true future state reads it at R2 0.366 and a copy of the present at 0.357: +0.009 of room. A predictor cannot show imagination on a target that barely moves against the probe error.
Theorem T6 names the admissible yardstick: skill over persistence, the predictable fraction of the CHANGE. Group P adds the mechanism: a moving object changes a state that keeps the cell layout and changes nothing in a mean over cells (VC-P8).

## 2. Stage 0c (dev box, CPU, T0, no model trained)

* Pairs: every pair of bank frames (clip, g) and (clip, g+10) with a finite lead gap at both and a bank frame at (clip, g-5). All such pairs are used, not only the 1,809 eval windows. n is printed. Episodes, folds, states (S0g, S0m), the probe family and the bootstrap are those of stage 0 (`S0_DEFS.md` R4, R5, R7, R11).
* Target: d = gap(g+10) - gap(g), metres.
* Ego block (given to EVERY arm, and alone to the floor): the ego vector at g, at g-5 and at g+10. The true future ego vector is more than an ego-only predictor could know; the floor is generous on purpose.
* Arms: H-g = [S0g(g), S0g(g) - S0g(g-5), ego block]; H-m = the same with S0m; N-g = [S0g(g), ego block] (no history); E = [ego block].
* Number: skill over persistence = 1 - sum (d - d_hat)^2 / sum d^2, pooled over the out-of-fold predictions (persistence predicts no change). The plain R2 is printed beside. Intervals: paired episode-cluster bootstrap, B 2,000.

## 3. Outcome names (fixed now)

* Per state: `S0-CHANGE-READ` if skill(H) minus skill(E) is above 0 and its interval excludes 0; else `S0-CHANGE-EGO` if skill(E) is above 0 and its interval excludes 0 (the change is predictable, from ego motion); else `S0-CHANGE-FLAT`.
* Pair: `GRID-ADDS-CHANGE` if skill(H-g) minus skill(H-m) is above 0 and its interval excludes 0; else `GRID-EQUAL-CHANGE`.
* History: `HISTORY-ADDS` if skill(H-g) minus skill(N-g) is above 0 and its interval excludes 0; else `HISTORY-EQUAL`.

## 4. Controls (each must read its known value, or no outcome name is printed)

* k1: targets permuted within fold: every skill at or below 0.02.
* k2, planted motion (label free; the prediction of VC-P8, frozen in `fv/FV_SPEC_V5_POOL.md` before this note): in every pair the planted vector (f5 size, the median token norm) sits in one fixed cell at frame g; at frame g-5 it sits in the left neighbour cell for a random half of the pairs and in the right neighbour for the others. Target = which. Bars: AUROC above 0.95 from [S0g(g), S0g(g) - S0g(g-5)]; at or below 0.60 from [S0m(g), S0m(g) - S0m(g-5)] (the lemma predicts 0.50); at or below 0.60 from S0g(g) alone (one frame carries no motion).
* k3, alignment: for every pair the two gaps and the three ego vectors equal the banked values at their rows (exact).
* k4: the stage-0b numbers of r-a are reproduced by the same loaded data (grid gap R2 0.3981 and ego 0.1327 to 4 decimals), so that 0c reads the same rows, states and folds.

## 5. Both outcomes, committed

* `S0-CHANGE-READ` for the grid state: the closing motion of the lead is linearly present in two frames of a state that keeps location. The lead read of Q2 in stage 1 then takes this form on the trained states with the learned predictor.
* `S0-CHANGE-EGO` or `S0-CHANGE-FLAT`: two pooled 4x10 frames 1.0 s apart do not expose the closing motion to a linear read beyond ego motion. Stage 1 still runs (a trained state and predictor are not linear). The lever that rises in rank is then the cue itself: finer tokens on the lead (a crop at native resolution), own card.
* For stage 1 of the card, before any of its data: the lead read of Q2 is computed in both forms (level, as the card writes it, and change, as here). The outcome name `IMAGINE-LEAD` / `IMAGINE-EGO` / `IMAGINE-FLAT` is assigned from the change form, because the level form has no room (section 1); the level form is printed beside it.

## 6. Limits

139 episodes; overlapping pairs inside a clip (the episode-cluster bootstrap carries the dependence); linear reads on untrained states; T0, a world-model diagnostic, never driving performance.
