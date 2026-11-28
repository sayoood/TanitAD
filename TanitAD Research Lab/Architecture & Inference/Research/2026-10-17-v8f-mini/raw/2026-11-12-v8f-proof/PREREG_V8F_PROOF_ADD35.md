# PREREG_V8F_PROOF_ADD35 : LEADCOND, the lead read as an input of the plan head, alone and under the own-plan rule term

Frozen before the pod tree of these arms is built, before any gate, step or read of them, and before the read of the linear arms of note `PREREG_V8F_PROOF_ADD34_N1.md` (their chain is training while this card is written; nothing of it has been pulled or read). Tier of every read: T1-kin (kinematic rollout on the model's own actions, logged frames, a lead that does not react). No number of this card is a driving-performance number.

## 1. What is known (MEASURED, rows H-V8F-PROOF-54, 58, 59, 61, 63, 64, 65)

* A distance rule at inference lowers the unsafe rate U: label-fed -0.068 at +0.18 m position error over 6 s; vision-fed with a calibrated margin to the same U at +0.86 to +0.95 m.
* The rule in the logged target (relabel) moves the plan and not U, at every dose.
* The rule on the model's own plan as a training term (squared form) lowers U by slowing the model everywhere: weight 1 -0.0276 [-0.0647, +0.0089] at +2.67 m, weight 4 -0.0525 [-0.0973, -0.0116] at +3.86 m.
* Hypothesis (section 6 of note ADD34_N1): the plan head cannot place its braking, because the lead is not in its input precisely enough. The plan head reads a pooled scene and an ego token; the supervised lead head of arm LS reads presence and gap from the level-1 grid states, and its outputs go nowhere.

## 2. The lever

`--lead-cond on` (definitions with file and line: `leadcond/LEADCOND_DEFS.md`; trainer `v8f_mini_train.py` sha256 d45621201f4a5bc95df0d5ef2be7c07587d208a752e81f9abb1e935a84e22f9e, model `v8f_mini.py` sha256 2e6622081a2cbbe50cd92367afb92f29beeda1f674c3144779953018a0713001; default off, bit for bit the tree of note ADD34_N1). At the current frame the outputs of the LEADSUP head become u = (p, p * g) with p the presence probability and g the gap output in the unit of the head (gap over 50 m, clamped to the label range); u is detached; a linear layer with weight and bias zero at the start maps u to the width of the ego token and is added to the ego token of the tactical and of the operative plan head. Vision only: u depends on the frames up to the current one and on nothing else of the batch. At step 0 a model loaded from an LS checkpoint is the LS model exactly.

## 3. Arms

Fine-tunes of 6,000 steps at learning rate 1e-4 (warmup 100) from the banked checkpoints of arm LS (lead head with gradient, 19,164 steps), bases LS_s0 and LS_s1, argv of LS plus:

* LFT: nothing (control). LFTb: the same argv again (floor of a rerun).
* LC: `--lead-cond on`.
* LO: the own-plan rule term, form and weight by section 4.
* LCO: both.
* Not trained: the label-fed rule at inference on LFT, LO, LCO (LFTRL, LORL, LCORL); the checkpoints of LC and LCO with the two tensors of the new layer set to zero (LCZ, LCOZ), rolled with the same pinned tool.

6,000 steps and not 2,000, because the new input starts at zero and has to be learned; the term then acts three times as long as in row H-V8F-PROOF-64, and every pair of this card is within the same step count. Chain A (LFT, LC, LFTb) does not depend on section 4 and starts first.

## 4. Form and weight of the term in LO and LCO (rule fixed here, before the read of the linear arms)

Candidates are the four configurations measured on the G bases against GFT: squared at weight 1 and 4 (row H-V8F-PROOF-64) and linear at weight 1 and 4 (note ADD34_N1, pending). Admitted are those with U below 0 in both bases and a pooled fall of U of at least 0.02. Of the admitted, the one with the lowest ratio of pooled position error over 6 s on all windows (m) to pooled fall of U is taken. If two ratios differ by less than 5 percent: the linear form before the squared, then the lower weight. If the read of the linear arms does not exist when chain B is built, squared at weight 4 (ratio 73.5 m per unit of U; weight 1: 96.7).

## 5. Read

Reader, windows and cells as rows H-V8F-PROOF-61 to 65: U and contact rate on 642 lead windows of 56 episodes, cost cells in four families and the position error over 6 s on 2,399 windows of 141 episodes, paired on windows, two-level bootstrap (bases, then episodes) B 2,000, spread of the six seeds of arm G beside. Pairs: LC, LO, LCO and LFTb against LFT; LCO against LO; LCO against LC; LFTRL against LFT; LCO, LO, LC against LFTRL; LC against LCZ and LCO against LCOZ. Share reached of an arm, per base: (U of LFT minus U of the arm) over (U of LFT minus U of LFTRL). First-tick share: how often the label-fed rule still changes the plan of the arm at the first tick of a lead window.

## 6. Controls (first; a failed control names nothing)

* q1: the build controls c1 to c7 of `leadcond/checks/` PASS (flag off bit for bit on four argv shapes; zero start is an identity for `forward` and `forward_infer`; the plan outputs move with the layer, gradient on the layer above 0 and exactly 0 with u zero, no gradient into the head or trunk through u, both mutations red; u unchanged exactly under changes of every non-vision key and of frames after the current one; train and inference agree; gate accepts the right expectation and refuses the wrong one; fine-tune and eval smoke).
* q2: pod gate of every arm: declared equals built, live step finite, 6,000 steps, start checkpoint applied with exactly the two fresh keys for the arms with the input and none for the others; arms with the input: live probe gradient on the layer above 0 and its zero-u control exactly 0; arms with the term: the probe and the train log carry the term and both counts.
* q3: identity at step 0 on real data: the live total of the dry gate of LC_s0 equals that of LFT_s0 (same seed, same batch; relative difference at most 1e-6).
* q4: LFTb minus LFT is not named different on U.
* q5: LFTRL minus LFT is named as cutting U (the known positive of the reader).
* q6: reader controls ALL_PASS in every read. q7: no plan moved ahead in the rule logs.

## 7. Outcomes (both bases needed; rules for U as section 6 of card ADD34 with control LFT and floor LFTb minus LFT; cost by the rule of card ADD11)

* Input alone, LC against LFT: `LEAD-INPUT-CUTS-UNSAFE`, `-FLAT` or `-WORSE`; cost `LEAD-INPUT-NO-COST` or `LEAD-INPUT-COSTS` with the cells.
* Term without the input, LO against LFT: `TERM-LS-CUTS-UNSAFE`, `-FLAT` or `-WORSE`, with its cost cells (the result of row H-V8F-PROOF-64 again on the LS base at 6,000 steps).
* Term with the input, LCO against LFT: `LEAD-INPUT-TERM-CUTS-UNSAFE`, `-FLAT` or `-WORSE`; `LEAD-INPUT-TERM-REACHES-ORACLE` by the rule of card ADD34; cost `-NO-COST` or `-COSTS` with the cells.
* The claim of this card, LCO against LO: `LEAD-INPUT-PLACES-BRAKING`, named only with `LEAD-INPUT-TERM-CUTS-UNSAFE`, if the position error over 6 s on all windows of LCO minus LO is below 0 in both bases with the two-level interval excluding 0, while U of LCO minus LO is not above 0 in both bases with an interval excluding 0. Else `LEAD-INPUT-NO-PLACEMENT`.
* The bar, `LEAD-INPUT-BEATS-MARGIN`: named only with `LEAD-INPUT-TERM-CUTS-UNSAFE`, if in both bases the position error over 6 s on all windows of LCO minus LFT is below +0.86 m and the share reached is at least 0.5. Else not named.
* Reported beside, naming nothing: use of the input at inference (LC minus LCZ and LCO minus LCOZ on U, on the gap bias at 2 s and on the position error); the input counts as used if one of these pairs is different on U or on the gap bias at 2 s with the same sign in both bases and a two-level interval excluding 0. First-tick shares. Fall of U per metre of mean gap shift at 2 s. The train-log sums of the lead head terms of LC against LFT (does the fine-tune keep the lead read).

## 8. What follows (committed before the data)

* `LEAD-INPUT-PLACES-BRAKING` with `-BEATS-MARGIN`: training with the rule is then a route of its own beside the rule at inference. Next: an arm trained from the start with the input and the term (two seeds), the checked vision-fed rule on top of it, and the own-state samples for the term.
* `LEAD-INPUT-PLACES-BRAKING` without the bar: the input helps and the price is still above the rule at inference; next the term on own-state samples (owed since card ADD34) with the input on.
* `LEAD-INPUT-NO-PLACEMENT` with the input used: the limit is the precision of the lead read itself; the finer lead read is the decision of the PI named in row H-V8F-PROOF-56 and is put to the PI with this result; until then the own-state term.
* `LEAD-INPUT-NO-PLACEMENT` with the input not used: 6,000 steps from a zero start do not teach the head to read the input; next an arm trained from the start with the input (two seeds), then this card's pairs on it.
* `LEAD-INPUT-CUTS-UNSAFE` for LC alone with no cost is reported as a result of its own (the plan reacts to a lead it is told about, with no rule anywhere).
* In every case the rule at inference stays the checked way to keep the distance (rows H-V8F-PROOF-54, 58, 59).

## 9. Limits

Two bases with one rerun of the control; 6,000 steps; label lead in the term at training, vision only at inference; one corpus. T1-kin: the frames are the logged ones, so the lead read sees the gap from the logged position and not from the position the model has rolled to; a model that has fallen behind is told a gap that is too short. The bar of section 7 compares with a price measured on arms LS and LSD trained from the start (row H-V8F-PROOF-58), same windows and reader. The input is added to the ego token; another place of entry is not tested here.

## 10. Cost (ESTIMATED)

Ten fine-tunes of 6,000 steps at 27 to 45 min each (27 min MEASURED on the G shape; the LS shape also runs the world-model part) and nine short rollouts: 6 to 9 L40-h on the pod, at most three GPU jobs at once.
