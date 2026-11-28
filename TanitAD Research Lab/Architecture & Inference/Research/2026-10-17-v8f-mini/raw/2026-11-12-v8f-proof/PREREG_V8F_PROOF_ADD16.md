# PREREG_V8F_PROOF_ADD16: CELLPRED, a level-1 predictor that attends across cells and time on the state that keeps the cell layout (2026-11-13)

Frozen before any data of this card is read, and before the outcome names Q1, Q2, Q3 of stage 1 of SPATIAL (addendum 12) are assigned
(when frozen: W_s1 is training; the pixel read and the r-b read on the trained states of G and W are not run; the T1-kin family read of G is preliminary).
Runs on the pod on the slot the SPATIAL chain frees, each arm under its own selfcheck gate token.
Theorems served: T5 (pooling lemma, group P: a mean over cells cannot hold where an object is), T7 (the value of an imagined rollout is bounded by the step error of the model).

## 1. What is known when this card is frozen (MEASURED, T0, `spatial/s1/`, rows H-V8F-PROOF-23 and the G_s1 read)

* The trained state that keeps the cell layout (G: 40 cells x 16 numbers) reads WHERE the lead is: lead gap R2 out of sample 0.552 / 0.541 at seeds 0 / 1, against 0.237 / 0.212 for the mean state.
* No state reads HOW the gap changes over 2.0 s beyond ego motion (state with history minus the ego block: -0.050 / -0.057, intervals below 0).
* In G the level-1 predictor is unchanged from the mean arm: one linear map takes the whole 640-number state into ONE token of width 256 per time step.
  The cells are not addressable inside the predictor, and whatever moves in the scene has to pass that one token.

## 2. Questions

C1 (imagination). Does the rollout of a predictor that attends across cells and time read the lead gap at +2.0 s better than the copy of the present state and than the ego-only floor, and better than the rollout of the flat predictor of G?
C2 (retention). Does the state trained with this predictor keep the position read of G?
C3 (cost). Does the arm cost any of the four metric families at T1-kin (kinematic rollout on the model's own actions)?
C4 (step error, the premise of T7; reported, no name). The latent error of the rollout at 0.8 s and 2.0 s as a skill over the copy, beside G.

## 3. The lever (fixed now)

Flag `--wm-l1-pred cell` (default `flat` = every earlier arm). Only with `--wm-state grid`; with a mean state the flag is refused.
The level-1 predictor becomes a module with the SAME interface as the flat one (`forward(z_ctx [B, L, D], a_ctx [B, L, E]) -> [B, L, D]`, attribute `context`), so the loss, the rollout and the planner calls of the stack are unchanged:
* tokens: every cell of every time step is one token; one shared linear map takes the 16 numbers of a cell to width 128; a learned cell-position term (40) and a learned time-position term (context) are added;
* depth 2; each block = (a) self-attention across the 40 cells of ONE time step together with one token that carries the embedded action of that step, then (b) causal attention across time for each cell, conditioned on the action in the way the flat predictor is (the existing block); 4 heads;
* output: one shared two-layer map per cell back to 16 numbers; no skip from the input state (as the flat predictor);
* levels 2 and 3, the encoders, the losses and every weight of the loss are unchanged. The driving path (readers, heads, planner) is unchanged: the world-model auxiliary is a training-time auxiliary.
Design constraint: the level-1 predictor of this arm has no more parameters than the flat predictor of G, so a gain cannot be assigned to capacity.

## 4. Arms and reads

* Arms, from scratch, 19,164 steps, the argv of G plus the lever: P = `--wm-state grid --wm-cell-dim 16 --wm-l1-pred cell`, seeds 0 and 1: 2 new arms.
* Baselines without new training: G_s0, G_s1 (same state, flat predictor), W and the HIER seeds of HIERREP.
* Reads, with the tools and splits of stage 1 of SPATIAL unchanged:
  * r-b of stage 0 on the trained model (lead gap at +2.0 s, windows with a lead at both times, read from (i) the rolled-out state, (ii) the present state, (iii) the true future state, (iv) the ego-only floor), for P and for G on the same windows; T0;
  * the eval-bank and train-probe reads of the state (`s1_eval_read.py`, `s1_read.py`) for P, paired with G; T0;
  * the pixel read of addendum 10 in its form for the grid arms (reported; its names as that card defines them); T0;
  * the four families at T1-kin strict (`--limit causal`), eval-141, the read tool of HIERREP (`spatial/q3/q3_read.py` on its engine); STRATEGIC reported PAUSED (PI item 25).
* Noise floor: for T1-kin cells the six-seed spread of the HIER arms (15 pairs); for the T0 reads the largest difference between HIER seeds of the same number. One training run per arm per seed: a separated interval is necessary and not sufficient.

## 5. Outcome names (each needs BOTH seeds)

* C1: `CELL-IMAGINES` if on r-b the rollout of P beats the copy AND the ego-only floor (both intervals excluding 0);
  `CELL-BEATS-FLAT` if it does not, but the rollout of P minus the rollout of G, paired on the same windows, is above 0 with the interval excluding 0;
  else `CELL-FLAT`.
* C2: `LOSES-POSITION` if the lead gap R2 of P minus G on the train probe set (paired over clips) is below 0 with the interval excluding 0 and larger in size than the largest difference between HIER seeds on that number; else `KEEPS-POSITION`.
* C3: the rule of addendum 12, Q3, for P minus G and for P minus HIER: `NO-COST`, `COST-<family>`, `GAIN-<family>`.

## 6. Committed next steps

* `CELL-IMAGINES` with `KEEPS-POSITION` and `NO-COST`: the grid state with this predictor is the proposed world-model default of the next flagship arm; next lever = imagination rollouts for planning on it (T7): candidate plans scored by the imagined lead gap, and the go / stay cue at rest (addendum 15) read from the imagined state; own card.
* `CELL-BEATS-FLAT`: the shape of the predictor helps and the one-step training signal is the limit; next lever = a rollout loss over 2.0 s (the predictor trained on its own rollouts); own card.
* `CELL-FLAT`: the limit is the training signal, not the shape of the predictor; next lever = an explicit target of object motion (predict the frozen token grid of the future frame cell by cell); own card.
* `LOSES-POSITION` or any `COST-<family>`: the arm is not promoted; it is reported as written, and the next lever of its C1 outcome still runs.
* If addendum 12 names `IMAGINE-LEAD` for G (the flat predictor already imagines the lead), this card is still read as written and the planning lever of addendum 12 proceeds on the better of the two predictors.

## 7. Controls

* c1 identity: the patched tree with the default `--wm-l1-pred flat` reproduces the tree of stage 1 bit for bit, for the argv of G and for the mean default (same state-dict keys; same loss trace over 200 steps, same seed, same device).
* c2 parameters: the parameter count of the level-1 predictor of P is at or below that of G; both counts are written to the selfcheck.
* c3 causality: the output at time step i does not move when the input at a later time step is changed (exact 0), and moves when the input at step i is changed.
* c4 cells: changing ONE cell of the input at the last time step changes the prediction of other cells (the spatial attention is live), and a predictor built `flat` is reported `flat` (the probe reads the BUILT module, not the flag).
* c5 reachability: after one real-data step every new parameter group has a non-zero gradient, and a parameter outside the graph reads exactly 0.
* c6 gate: the selfcheck of each arm reports `wm_l1_pred = cell`, the state width 640 and the clip count; an arm that declares `cell` and builds `flat` is refused (deliberate regression, once per arm, as the chain does for every extra expectation).
* c7 evaluation: the T1-kin eval loader builds the identical model (strict loading), and the checkpoint round trip is exact.

## 8. Compute and authority

Pod L40: 2 arms of 19,164 steps on one of the three slots (about 66 min each beside two other arms, ESTIMATED from the stage-1 arms), plus the T1-kin dump of each; reads on the dev-box CPU and one dev-box GPU read behind the shared lock. About 2.5 L40-h (ESTIMATED).
The arm is the lever addendum 12 (section 5) committed in advance for the pattern "the state keeps objects, the predictor does not move them"; it runs on this fork's pod queue under the PI's three-at-once allowance; it changes no default and nothing in the driving path.
