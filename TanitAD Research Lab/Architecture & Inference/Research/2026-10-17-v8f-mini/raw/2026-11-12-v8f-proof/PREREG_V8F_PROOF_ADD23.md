# PREREG_V8F_PROOF_ADD23: LEADHEAD, stage 0b. Is the flat read of card ADD21 a limit of the state or of the 1,700 to 1,900 labelled fit rows? The same heads, fitted on the train probe set (1,200 training clips, about 121,000 frames), scored on the same held-out windows. No training of the model.

Frozen before any number of this card is computed. Addendum to `PREREG_V8F_PROOF.md` and to `PREREG_V8F_PROOF_ADD21.md`. Tier T0: a world-model diagnostic on logged frames, never driving performance. Evidence class of every result: MEASURED once run. LONGITUDINAL (distance keeping) is the family this lever serves and it is not measured by this read; LATERAL and TACTICAL: not measured; STRATEGIC: PAUSED (PI item 25).

## 1. What is known before it (MEASURED, card ADD21, `spatial/leadhead/VERDICT_LEADHEAD_S0.md`, all 26 controls PASS)

* Heads fitted on the eval bank, cross-fitted by episode (1,689 to 1,926 fit rows with a closing-speed label per fold, 3,200 inputs): held-out R2 of the closing speed -0.094, -0.108, -0.098, -0.060 (P_s0, P_s1, G_s0, G_s1) for the temporal ridge, lower for the small network; the ridge chose the largest penalty of its grid in 19 of 20 fits; in-sample R2 0.26 to 0.29. `HEAD-CLOSING-FLAT`, `HEAD-SLOT-FLAT`, `STATE-LACKS-FUTURE` for both arm types.
* The temporal head reads the present gap no better than the one-frame probe (R2 0.52 to 0.66, RMSE 6.0 to 7.1 m), which is what too few rows for 3,200 inputs looks like. Card ADD21 section 6 names this limit. The committed next step of that card (heads on the camera tokens before pooling) has more inputs and the same few rows, so the row count is settled first; the token read stays committed (section 5).

## 2. Data and definitions (fixed; everything not named here is as in cards ADD20 and ADD21, `OBJSLOT_DEFS.md`, `LEADHEAD_DEFS.md`)

* Scored windows, target, rules R1 to R3, copy, ego speeds, estimator: those of card ADD21 (429 scored windows of 48 eval episodes, episode-cluster bootstrap B 2,000, rng 0). Arms: P_s0, P_s1, G_s0, G_s1.
* Fit rows: the frames of the train probe set of the trained-state read (`<ARM>_train.npz`: 1,200 training clips, 100 to 104 consecutive frames per clip, the row layout and the labels of `spatial/s1/s1_read.py`), with the grid state of the frame and of the four frames before it in the same clip. The frame step of the probe set is read from its definition and asserted equal to the 0.2 s of the eval windows; if it differs, the history is built at 0.2 s by the stride that gives it, and the read stops if no stride does.
* Labels of a fit row: gap(t) from the join of the probe set; c(t) = the least-squares slope of the gap over the trailing run of labelled frames t-4 to t under the continuity rule of card ADD20 (a step above 3.0 m breaks the run), H_MIN = 3. Direct label: gap(t+10) minus gap(t) with the continuity rule over t to t+10; the ego term X needs the speeds of the probe set, and if the probe set holds no speed the direct read is fitted without X and that is stated.
* The train probe clips and the eval episodes must be disjoint (asserted by id). The model has been trained on the train probe clips, so their states are in-sample for the model; the scored windows are held out for the model and for the heads.
* Heads: HT and HM exactly as in card ADD21 (inputs standardised on the fit rows; penalty of HT by 4 inner folds by clip from the grid of card ADD21, extended upward by 1e6 and 1e7 because the earlier fits sat on its edge; HM as defined there, with the deviations recorded in `LEADHEAD_DEFS.md`). The head for the names is chosen by the inner-fold error on the fit rows, never by the scored windows. No cross-fitting over the eval folds is needed for the heads; rule R3 keeps its cross-fitted residual over the eval folds.
* Secondary, deciding nothing: (a) the learning curve of the held-out R2 of the closing speed and of the gap for HT fitted on 2,000, 8,000, 32,000 and all labelled fit rows (clips drawn with rng 0); (b) the same heads fitted on the train probe rows and the eval-bank rows of the training folds together (cross-fitted).

## 3. Controls (all must pass before any outcome is named)

* j1 disjoint sets: no clip id of the fit rows is an eval episode id.
* j2 alignment of the probe set: the row order, clip list and frame index of the state file equal those of the label loader (the t1 check of `s1_read.py`), for every arm.
* j3 shuffled labels: with c permuted across the fit rows, the held-out R2 of the head closing speed is at most +0.02 (HT and HM).
* j4 in-sample ceiling (deliberate-regression arm): HT applied to its own fit rows must read a higher R2 of the closing speed than on a held-out tenth of the train clips; if not, the read cannot see over-fitting and no outcome is named.
* j5 planted state on the train layout: the control h1 of card ADD21 run on the row layout of the train probe set, held-out R2 above 0.9.
* j6 label path: the label gap and closing speed of the scored windows through the rule path reproduce 0.750 within 1e-6 (control h6 of card ADD21).
* j7 no future input: replacing every state after frame S0 of the scored windows by noise changes no output.
* j8 transfer check, reported and not a gate: the held-out R2 of the gap head on a held-out tenth of the train clips beside its R2 on the eval windows; the difference is the size of the shift between states the model was trained on and states it was not.

## 4. Outcomes (both committed), per arm type, both seeds needed for a name

* `HEAD-READS-CLOSING-B`: the held-out R2 of the closing speed on the scored windows is above 0 with an interval that excludes 0 at both seeds. Else `HEAD-CLOSING-FLAT-B`.
* `HEAD-SLOT-IMAGINES-B`: R3 skill over the copy of the same head is above 0 with an interval that excludes 0 at both seeds; in addition `HEAD-SLOT-BEATS-ROLLOUT-B` if the level R2 of R3 minus the banked rollout read is above 0 with an interval that excludes 0 at both seeds. Else `HEAD-SLOT-FLAT-B`.
* Direct read: `STATE-HOLDS-FUTURE-B` or `STATE-LACKS-FUTURE-B`, as in card ADD21.
* Reading of the pair of cards: flat in ADD21 and reading here = the earlier flat read was the row count; flat in both with a flat learning curve = the pooled grid state does not hold the motion of the lead at this precision.

## 5. Committed next steps

* `HEAD-SLOT-IMAGINES-B`: stage 1 of card ADD20, own card: the head (fitted on training clips) and the fixed transition become a part of the level-1 state, two seeds on the pod, read at T0 with this script and at T1-kin with the four families; then planning by imagination on the slot.
* `HEAD-READS-CLOSING-B` with `HEAD-SLOT-FLAT-B`: a supervised auxiliary loss on gap and closing speed during training of the grid-state arm, own card, two seeds on the pod.
* Flat in both cards: two levers, each its own card: (i) the committed token read of card ADD21 (the same heads on the frozen camera tokens before pooling, fitted on the training clips), zero training; (ii) the supervised auxiliary loss during training, which lets the state allocate capacity to the lead. Run (i) first if the token cache of the probe clips exists, else (ii).
* A rising learning curve that has not flattened at all rows: the fit set is extended to more training clips before any name is read as a property of the state.

## 6. Cost and limits

* CPU only, the banked states and labels, no pod time. Estimated under one hour; the Gram matrix of HT is accumulated in blocks so that the read needs under 3 GB of memory.
* The heads are fitted on states of clips the model was trained on and applied to states of clips it was not; control j8 measures that shift, and a flat read with a large shift is reported as not decisive.
* 429 scored windows of 48 episodes; one training run per seed, so the names need both seeds; the rules use the logged ego speeds of the future frames as the action input of a rollout.
