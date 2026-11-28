# PREREG_V8F_PROOF_ADD34_N1 : RULEOWN, the own-plan term in a linear form

Note to card `PREREG_V8F_PROOF_ADD34.md`. Frozen before the trainer change for the linear form is built, before any gate, step or read of its arms. The card and its names stay as written. Tier of every read: T1-kin (kinematic rollout on the model's own actions, logged frames, a lead that does not react).

## 1. What was seen (MEASURED, row H-V8F-PROOF-64)

* The squared form of the term (mean over knots of the squared distance in m2) at weight 1: U against the control -0.0276 [-0.0647, +0.0089] (`OWN-RULE-FLAT`), `OWN-RULE-COSTS` (LONGITUDINAL, TACTICAL); position error over 6 s on all windows +2.67 m; share of windows more than 10 m behind 0.19 to 0.54. At weight 4: U -0.0525 [-0.0973, -0.0116], position error +3.86 m.
* The first-tick share of lead windows the label-fed rule still corrects fell from 0.240 / 0.246 to 0.170 / 0.189 (weight 1).
* For comparison, the price of a similar fall of U with a rule at inference: label-fed +0.18 m; vision-fed with the calibrated margin +0.86 to +0.95 m (row H-V8F-PROOF-58: U -0.0712 and -0.0827).
* In the squared form the pull of a sample grows with the size of its violation, so one plan with a violation of 10 m pulls about twenty times as hard as one imitation sample (the position terms of the trainer are linear in metres). Hypothesis of this note: this is what turns a term that is active on about 6 percent of the samples into a slowdown everywhere.

## 2. The lever

A flag for the form of the own-plan term, default the squared form of the card (bit for bit): in the linear form the per-sample term is the mean over the knots of the Euclidean distance in metres between the predicted knot and its cap (0 where the plan keeps the rule; the gradient of a knot at distance 0 is defined as 0). Everything else as section 2 of the card (mask, inputs, detached plan, float64 rule call, cap-only, no rule and no lead input at inference).

## 3. Arms

Fine-tunes of 2,000 steps from G_s0 and G_s1, the argv of GOW with the linear form: GOL (weight 1, the named arm of this note), GOL4 (weight 4, dose arm, names nothing); the label-fed rule at inference on them (GOLRL, GOL4RL). Control GFT, floor GFTb (card ADD32). Cost (ESTIMATED from GOW at about 10 min per arm): about 0.8 L40-h.

## 4. Read and controls

Read as section 4 of the card with GOL in place of GOW. Controls: o1 (default form bit for bit against the tree of card ADD34), o2 and o3 for the linear form (cap by hand, exact zeros, three-part reach with a mutation test, finite gradient when a knot sits exactly on its cap), o4 (gate entry for the form: declared equals built, wrong expectation refused; the live step and the train log carry the term and both counts), o5, o6, o7 as the card.

## 5. Outcomes (both bases needed; named for GOL)

* `OWN-RULE-L1-CUTS-UNSAFE`, `-WORSE`, `-FLAT`, `-REACHES-ORACLE`: the rules of section 6 of the card for U.
* Cost against GFT by the rule of card ADD11: `OWN-RULE-L1-NO-COST` or `OWN-RULE-L1-COSTS` with the cells.
* The bar of this note, `OWN-RULE-L1-BEATS-MARGIN`: named only with `OWN-RULE-L1-CUTS-UNSAFE`, if in both bases the position error over 6 s on all windows, GOL minus GFT, is below +0.86 m (the lower of the two prices of the calibrated margin rule) and the share reached is at least 0.5. Else not named.
* Reported beside: the same cells for GOL4; GOL against GOW on U and on the position error (does the linear form buy the same U at a lower price); the first-tick share; the fall of U per metre of mean gap shift at 2 s.

## 6. What follows (committed before the data)

* `OWN-RULE-L1-CUTS-UNSAFE` with `-BEATS-MARGIN`: the trained rule is cheaper than the checked vision-fed rule; six bases (G_s2 to G_s5), the checked rule on top, and an arm trained from the start.
* `OWN-RULE-L1-CUTS-UNSAFE` without `-BEATS-MARGIN`, or `-FLAT` with a price at or above that of the squared form: the form of the term is not the cause of the price; the cause is that the plan head cannot place its braking (the lead is in the state too coarsely or the head does not use it). Next is then the lead read as an explicit input of the plan head (the presence and gap outputs of the lead head of arm LS fed to the planner, vision only, no ego state), with the term on top; the finer lead read stays the decision of the PI named in row H-V8F-PROOF-56.
* `-FLAT` with a low price: the dose is too small; weight 4 is the next read (it is trained in the same chain).
* In every case the term on own-state samples (owed by the card for `OWN-RULE-FLAT` with the first-tick share fallen) stays owed and is built with whichever form reads better here.

## 7. Limits

Two bases with one replicate of the control; 2,000 steps; label lead in the term at training, vision only at inference; T1-kin with logged frames and a lead that does not react; one corpus. The bar of section 5 compares a price measured on fine-tunes of arm G with a price measured on the arms LS and LSD of row H-V8F-PROOF-58 (other checkpoints, same windows, same reader).
