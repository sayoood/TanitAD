# PREREG_V8F_PROOF_ADD20: OBJSLOT, stage 0. Can an object-level slot with a kinematic transition read the lead gap 2.0 s ahead better than a copy? Zero training.

Frozen before any number of this card is computed. Addendum to `PREREG_V8F_PROOF.md`; the committed next lever of `PREREG_V8F_PROOF_ADD19.md` section 6 in every outcome that does not imagine. Tier T0: a world-model diagnostic on logged frames, never driving performance. Evidence class of every result: MEASURED once run. LONGITUDINAL (distance keeping) is the family this lever serves and it is not measured by this read; LATERAL and TACTICAL: not measured; STRATEGIC: PAUSED (PI item 25).

## 1. Why this card, and what is known before it

* Known (MEASURED, rows H-V8F-PROOF-34 to -40): the grid state reads the present lead gap (R2 0.54 / 0.55 on held-out clips); three learned predictors (flat, cell, cell with a rollout loss at seed 0, cell with the ego speed at seed 0) rolled out 2.0 s read the future gap no better than a copy of the present state; the true future state holds a part of the change of the gap (skill over the copy +0.16 to +0.21 through the level probe).
* Known (MEASURED, note N8): on the train probe clips the ego motion alone explains 29 % of the squared change of the gap over 2.0 s.
* Hypothesis: the motion of the lead is not a property a latent one-step predictor picks up, and it does not need to be learned. Two numbers per lead vehicle (gap, closing speed) and one kinematic identity predict the gap: with a constant lead speed, gap(t + T) = gap(t) + T * (lead speed) - (ego travel over T), and the ego travel is given by the actions of the rollout.
* This card measures, without training anything, how much of the change that rule delivers (a) when its inputs come from the logged labels (a privileged ceiling) and (b) when they come from the trained grid state through the existing level probe (vision only, the deployable form).

## 2. Data and definitions (fixed)

* Windows, start frame and horizon are those of `spatial/q2/ctx_roll.py` and `rb_trained.py`: the eval-141 windows of the r-b read (1,809 used), frames 0.2 s apart, start frame S0 = 4, K = 10 steps, target frame 14. Scored set: the windows with a lead at frame S0 and at frame 14 under the continuity rule of note N6 (the same lead; a window whose lead label jumps to another vehicle between S0 and 14 is dropped and counted).
* History: the frames 0 to 4 of the window (5 frames, 0.8 s). A window enters a rule only if the gap source of that rule has a value at frame 4 and at least two more history frames (H_MIN = 3); the count per rule is reported.
* Gap sources.
  * LABEL: the logged lead gap of each history frame (the label source of the bank of the r-b read). If the labels exist only on a subset of the frames, the rule uses the labelled history frames and the card's H_MIN; the available spacing is reported before any outcome.
  * VISION: the level probe of the r-b read (closed-form ridge on the bank frames of the training folds, cross-fitted by episode, the same probe object that reads (ii) there), applied to the grid state of the arm at each history frame. Arms: P_s0, P_s1 (cell predictor), G_s0, G_s1 (grid state, flat predictor). No state of a future frame enters.
* Closing speed c: the least-squares slope of the gap over the history frames of the source, in m/s.
* Ego speed: the logged speed of the frames S0 to 14 (`SPEED_WINDOWS.npz` of `collect_speed.py`; in a rollout these follow from the actions, which are inputs; here they are read teacher-forced, as in every T0 read of this programme).
* Rules, each giving a predicted gap at frame 14:
  * R0 copy: gap(S0) of the source.
  * R1 constant closing speed: gap(S0) + 2.0 * c.
  * R2 constant lead speed with the known ego motion: lead speed vL = ego speed(S0) + c; predicted gap = gap(S0) + sum over the ten steps of 0.2 * (vL - ego speed at that step), ego speed at a step = the mean of the logged speeds at its two frames.
  * R3 rule plus a learned residual, the smallest form: gap(S0) + a * (R1 term) + b * (R2 ego term) + d, the three coefficients fitted by least squares on the training folds only (5 folds by episode, the fold map of the r-b read) and applied to the held-out fold. R3 is the PRIMARY rule of the vision source; R2 is the primary rule of the label source.
* Target and metric: the logged gap at frame 14. Change form (primary): skill = 1 - SSE(rule) / SSE(R0 of the same source), so the copy of the same source reads 0. Level form (secondary, for comparison with the banked reads): R2 of each rule against the logged gap at frame 14.
* Estimator: episode-cluster bootstrap over the episodes of the scored set, B 2,000, rng 0, paired between a rule and R0 and between a rule and the banked rollout read (i) of the same arm on the common windows. A separated interval from one training run answers another draw of episodes only; the vision outcome needs both seeds.
* Secondary, deciding nothing: the same skills at horizons 0.4, 0.8, 1.2 and 1.6 s (target frames 6, 8, 10, 12) with the rule scaled to the horizon.

## 3. Controls (all must pass before any outcome is named; a failed control is reported and no outcome is named)

* o1 planted series: a synthetic gap with a constant lead speed and the logged ego speeds, fed through the LABEL path, must give R2 skill 1.000 within 1e-9, and R1 below 1 whenever the ego speed varies.
* o2 shuffled target: with the future gap permuted across windows (fixed seed), every rule reads skill at most +0.02.
* o3 identity: the copy reads exactly 0; a rule that returns the logged future gap reads exactly 1.
* o4 probe fidelity: the level R2 of VISION R0 equals the banked value (ii) of the r-b read of the same arm within 0.005 on the common windows.
* o5 leak detector (deliberate-regression arm): R1 with the history moved one frame into the future (frames 1 to 5) must read above R1 on the LABEL source. If it does not, the read cannot see a time-alignment error and no outcome is named.
* o6 no future input: the script asserts that no rule of the VISION source reads a state or a label of a frame after S0, other than the logged ego speeds of rule R2 and R3.

## 4. Outcomes (both committed)

Label source (one result, it does not depend on an arm):
* `KIN-CEILING-HOLDS`: R2 skill over the label copy is above 0 with an interval that excludes 0.
* `KIN-CEILING-FLAT`: otherwise.

Vision source (per arm type, both seeds needed for a name):
* `SLOT-IMAGINES`: R3 skill over the vision copy is above 0 with an interval that excludes 0 at both seeds.
* `SLOT-BEATS-ROLLOUT` (in addition): the level R2 of R3 minus the banked rollout read (i) of the same arm is above 0 with an interval that excludes 0 at both seeds.
* `SLOT-FLAT`: otherwise. Reported as written.

Reported beside the names, deciding nothing: the share of the label ceiling that the vision source delivers (vision R3 skill over label R2 skill), and the horizon profile.

## 5. Committed next steps

* `SLOT-IMAGINES`: stage 1, own card: the slot becomes a part of the level-1 state of v8F-mini (gap and closing speed from a supervised lead head on the grid state, the fixed transition of rule R2, a learned residual), two seeds, read at T0 with this script and at T1-kin with the four families; then planning by imagination on the slot (candidate plans scored by the imagined gap: distance keeping and the go or stay cue at rest).
* `KIN-CEILING-HOLDS` with `SLOT-FLAT`: the rule works and the precision of the vision gap is the limit. Next lever, own card: a lead head trained on the grid state with the gap and the closing speed as supervised targets (labels may use privileged sources, inference stays vision only), re-read with this script.
* `KIN-CEILING-FLAT`: a constant lead speed does not carry 2.0 s. The horizon profile locates where it does; the imagination horizon of the slot is set there, and the next lever is the lead head with an acceleration term, own card.

## 6. Cost and limits

* CPU only, the banked states and labels, no pod time. Estimated under one hour.
* The eval bank has 139 episodes; the scored set is small (441 windows with a lead at both times before the continuity rule), so a small effect cannot be separated on it. If the label ceiling is separated and the vision read is not, the same read on the train probe clips (in-sample for the model, reported as such) is the committed follow-up of the script, deciding nothing here.
* The probe is the level probe of the r-b read, fitted on levels and not on changes.
* Rule R2 uses the logged ego speeds of the future frames. That is admissible as the action input of a rollout and it is stated with every number; it is not a goal input and it carries nothing of the situation classifier.
