# PREREG_V8F_PROOF_ADD21: LEADHEAD, stage 0. Does a supervised lead head on the history of the grid state read the gap and the closing speed well enough for the kinematic slot to imagine 2.0 s ahead? No training of the model.

Frozen before any number of this card is computed. Addendum to `PREREG_V8F_PROOF.md`; the committed next lever of `PREREG_V8F_PROOF_ADD20.md` section 5 for the outcome it read (`KIN-CEILING-HOLDS` with `SLOT-FLAT`). Tier T0: a world-model diagnostic on logged frames, never driving performance. Evidence class of every result: MEASURED once run. LONGITUDINAL (distance keeping) is the family this lever serves and it is not measured by this read; LATERAL and TACTICAL: not measured; STRATEGIC: PAUSED (PI item 25).

## 1. What is known before it (MEASURED, card ADD20, `spatial/objslot/VERDICT_OBJSLOT_S0.md`)

* With the logged gap as input, the kinematic rule R2 (constant lead speed, known ego motion) reads the gap 2.0 s ahead with skill +0.750 [+0.664, +0.811] over the copy (413 windows; RMSE 4.36 m for the copy, 2.18 m for the rule; recomputed independently by the lead: 0.7499). `KIN-CEILING-HOLDS`.
* With the gap read from the grid state by the level probe (one frame, ridge on levels), the same rule is below the copy (R2 skill -0.24 to -0.41) and the rule with a learned residual reads 0 (R3 skill -0.007 to +0.027) at both seeds of both arm types. `SLOT-FLAT`.
* Open question this card answers: is the limit the READ (a head fitted on levels of one frame cannot give a closing speed, and a head supervised on the closing speed over the history of the state can), or the STATE (the pooled grid state does not hold the motion of the lead at the needed precision)?

## 2. Data and definitions (fixed; everything not named here is as in card ADD20 and `spatial/objslot/OBJSLOT_DEFS.md`)

* Windows, scored set, continuity rule, history rule, ego speeds, target, estimator, folds: those of card ADD20 (the S0 = 4 file, 1,948 windows, 429 scored, 5 folds by episode, episode-cluster bootstrap B 2,000, rng 0). Arms: P_s0, P_s1, G_s0, G_s1.
* Fit rows: the bank frames of the training folds of each fold (the rows the level probe of the r-b read is fitted on) that have the grid state of the frame and of the four frames before it in the same episode. Rows without the five frames are dropped and counted.
* Labels of a fit row at frame t: gap(t) = the logged lead gap; c(t) = the least-squares slope of the logged gap over the trailing run of labelled frames t-4 to t under the continuity rule, H_MIN = 3 (the label closing speed of card ADD20). Labels use the privileged source; inference uses the grid states only.
* Head input: the grid states of the frames t-4 to t (5 x 640 values). No state after t.
* Heads, each cross-fitted by episode with the fold map of card ADD20:
  * HL (reference): the level probe of card ADD20 (one frame, gap only).
  * HT, temporal ridge: closed-form ridge from the 3,200 inputs (standardised on the fit rows) to gap(t) and, separately, to c(t). The penalty is chosen on the fit rows only, by 4 inner folds by episode, from the grid 1e0, 1e1, 1e2, 1e3, 1e4, 1e5 times the number of fit rows divided by 1,000.
  * HM, small network (so that a negative of a linear read is not taken for a negative about the state): the inputs projected on the first 64 principal components of the fit rows, then 64 - 64 - 1 with GELU, Adam, learning rate 1e-3, weight decay 1e-3, 300 epochs, full batch, 3 initialisations averaged (seeds 0, 1, 2), one network per target. CPU.
* Slot read (the deployable form): for a scored window the head gives the gap and the closing speed at frame S0 from the states of the frames 0 to 4; the rules R1, R2 and R3 of card ADD20 run on these two numbers and the logged ego speeds. The copy is the gap of the same head at S0. R3 is the PRIMARY rule, as in card ADD20; R2 is reported beside it.
* Direct read (secondary, the ceiling of what a supervised read of the history can give without the kinematic rule): the heads HT and HM fitted to the change of the gap over 2.0 s (gap(t+10) minus gap(t), fit rows with the same lead at t+10) from the same inputs plus the ego term X of card ADD20 as a 3,201st input; applied at S0 of the scored windows; skill over the copy of the head gap.
* Metrics: (a) fidelity on the held-out folds: R2 of the head gap against the logged gap at S0 and R2 of the head closing speed against the label closing speed, on the scored windows with the label history rule; (b) skill of each rule over the copy of the same head, change form, as in card ADD20; (c) level R2 of R3 against the logged gap at frame 14, and its difference to the banked rollout read (i) of the same arm, paired on the common windows.

## 3. Controls (all must pass before any outcome is named)

* h1 planted state: a synthetic state history in which a fixed random direction carries a planted gap with a constant closing speed, plus noise of the measured size of the state; HT must read the planted closing speed with held-out R2 above 0.9.
* h2 shuffled labels: with c permuted across the fit rows (fixed seed), the held-out R2 of the head closing speed is at most +0.02, for HT and HM.
* h3 fidelity of the reference: the gap of HL equals the banked values `VG_<arm>.npz` of card ADD20 within 1e-6.
* h4 leak detector (deliberate-regression arm): the same HT fitted on all folds and applied in-sample must read a higher R2 of the closing speed than the cross-fitted head. If it does not, the read cannot see a fold leak and no outcome is named.
* h5 no future input: replacing every state after frame S0 by noise changes no head output and no rule output.
* h6 label path: the label gap and the label closing speed pushed through the rule path of this script reproduce the R2 skill of card ADD20 (0.750) within 1e-6.

## 4. Outcomes (both committed), per arm type, both seeds needed for a name

* `HEAD-READS-CLOSING`: for the better of HT and HM (chosen by the inner folds of the fit rows, never by the scored windows), the held-out R2 of the closing speed is above 0 with an interval that excludes 0 at both seeds. Else `HEAD-CLOSING-FLAT`.
* `HEAD-SLOT-IMAGINES`: R3 skill over the copy of the same head is above 0 with an interval that excludes 0 at both seeds. In addition `HEAD-SLOT-BEATS-ROLLOUT` if the level R2 of R3 minus the banked rollout read (i) is above 0 with an interval that excludes 0 at both seeds. Else `HEAD-SLOT-FLAT`.
* Direct read: `STATE-HOLDS-FUTURE` if the skill of the direct head over the copy is above 0 with an interval that excludes 0 at both seeds; else `STATE-LACKS-FUTURE`.
* Reported beside the names, deciding nothing: the share of the label ceiling (R3 skill of the head over 0.750), the RMSE of the head gap and of the head closing speed in metres and metres per second, and the horizon profile at 0.4, 0.8, 1.2, 1.6 s.

## 5. Committed next steps

* `HEAD-SLOT-IMAGINES`: stage 1 of card ADD20, own card: the head and the fixed transition become a part of the level-1 state of v8F-mini, two seeds, read at T0 with this script and at T1-kin with the four families; then planning by imagination on the slot.
* `HEAD-READS-CLOSING` with `HEAD-SLOT-FLAT`: the closing speed is in the state and too imprecise for 2.0 s. Next lever, own card: the head trained with the model (a supervised auxiliary loss on gap and closing speed during training of the grid-state arm), which lets the state allocate capacity to the lead; two seeds on the pod.
* `HEAD-CLOSING-FLAT` with `STATE-LACKS-FUTURE`: the pooled grid state does not hold the motion of the lead. Next lever, own card: the same heads on the frozen camera tokens before pooling (the full token grid of the lead region), zero training of the model; if they read the closing speed, the level-1 state gets an object read at token resolution.
* `STATE-HOLDS-FUTURE` with `HEAD-SLOT-FLAT`: a direct supervised read works where the rule does not; the slot then carries the direct head and the rule is kept as its prior, own card.

## 6. Cost and limits

* CPU only, the banked states and labels, no pod time. Estimated under one hour.
* The fit rows are few (about 1,900 to 2,150 per fold, fewer with five frames); a negative of HT and HM is a negative for heads of this size on this bank, and the third next step tests the state before pooling.
* The scored set is small (429 windows, 48 episodes); one training run per seed, so an interval answers another draw of episodes and the names need both seeds.
* Rule R2 and R3 and the direct read use the logged ego speeds of the future frames, the action input of a rollout; no goal input, nothing of the situation classifier.
