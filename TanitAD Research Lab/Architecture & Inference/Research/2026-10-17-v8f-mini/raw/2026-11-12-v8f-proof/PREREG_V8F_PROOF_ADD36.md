# PREREG_V8F_PROOF_ADD36 : IMAGLEAD, the supervised lead head on imagined states, and whether the imagined gap follows the ego's own actions

Card of the v8F proof programme, MOTION line. Frozen before any arm of this card is trained and before the read tool exists; no rollout of any model has been read with a lead head. Tier of the reads H1 and H2: T0 (world-model diagnostics on logged frames, never driving performance). Tier of the family read F: T1-kin (kinematic rollout on the model's own actions, logged frames, a lead that does not react). STRATEGIC: PAUSED (PI item 25).

## 1. Why (MEASURED, with the artifact)

* Row H-V8F-PROOF-67: the residual form makes the level-1 predictor beat the copy one step ahead (0.829 and 0.844), the rollout loss needs the residual form to fit, and only the two together (arm RR) add lead information 2 s ahead to the present state in the calibrated read on the two known seeds (+0.043 [+0.003, +0.086], about half of the headroom of a linear probe; the two added seeds are training and decide the names of note ADD18_N3).
* Every lead read of a rollout so far used a linear probe fitted after training; its headroom is about 0.08 of the label variance, so an absent effect is weak evidence (note ADD18_N3, section 6).
* The supervised lead head (card ADD24, rows H-V8F-PROOF-52 to 58) reads presence and gap from the level-1 states of five frames, and a distance rule fed by it reaches the unsafe rate of a rule fed by the label at a price of +0.86 to +0.95 m (rows 54, 58). That rule extrapolates the lead with a constant-velocity assumption from the present read; it does not use the world model's rollout.
* Planning by imagination (section 6 of card ADD18) needs two things that no read has shown: (a) the head, applied to imagined states, reads the gap ahead; (b) the imagined gap moves when the ego's own planned actions change, by the amount the kinematics give. (b) is the precondition for comparing candidate plans in imagination; a world model whose rollout ignores the action input cannot rank plans.
* This card is frozen while RR_s2 and RR_s3 train. Its reads do not depend on the names of note ADD18_N3; the note's names are reported as that note rules.

## 2. Arms

* LS_s0, LS_s1 (exist; card ADD24): the base predictor with the lead head (gradient into the encoder path). Read first.
* RL_s0, RL_s1 (new): the argv of LS plus `--wm-l1-residual on --wm-l1-roll-k 10 --w-wm-roll 1.0 --wm-l1-diag on`, in the pinned tree of the LS arms (every flag exists there; defaults checked bit for bit by the controls of cards ADD18 and ADD24), from scratch, 19,164 steps, seeds 0 and 1. The head is trained on real states only, as in LS.
* Budget (ESTIMATED from the RR arms): about 90 min per arm on one slot of the pod (L40), after chain `motion_e` (one arm with a rollout loss at a time; container memory limit 50 GB). Reads: minutes per arm on the pod GPU.

## 3. Reads (tool `imaghead`, built after this freeze; anything the build forces to change goes into a note before data)

The rollout is that of the stack at read time: from the true states up to frame t, 10 steps of 0.2 s, the context growing by its own predictions, with a given sequence of ego actions. The head reads five consecutive states; at 2.0 s ahead all five are imagined (steps 6 to 10), at 1.0 s ahead steps 1 to 5.

**H1, the head on imagined states.** Windows of the eval bank with a lead at t and at t + 2.0 s (the 441 windows of 139 episodes of the r-b read). Label: the lead gap at t + 2.0 s. Reads of the arm's own head: (i) from the rolled-out states (logged ego actions), (ii) from the present states (the head at t), (iii) from the true future states (the head at t + 2.0 s, the most the head can show), and the ego-only floor read of the r-b bank. Calibrated statistics as in note ADD18_N3 (five folds by episode, linear fit of the label on the named reads out of fold): H1 = R2 of (ii, i) minus R2 of (ii); H1e = R2 of (ii, floor, i) minus R2 of (ii, floor); H1c and H1ec with (iii) in place of (i). Beside, in metres and without calibration: the mean absolute error of (i), (ii), (iii) against the label. Two-level bootstrap (seeds, then episodes), B 2,000, seed 0.

**H2, the response of the imagined gap to the ego's own actions.** Windows of the eval bank with a lead at t, a label gap between 5 and 50 m and an ego speed of at least 5 m/s at t (count reported). For each window the rollout is run four times: with the logged ego actions and with a constant offset d of -2, -1 and +1 m/s2 added to the acceleration of every step (yaw rate unchanged, in the units the predictor was trained with). Response r(d, T) = imagined gap under the offset minus imagined gap under the logged actions, read by the head at T = 2.0 s (named) and T = 1.0 s (reported). Kinematic expectation for a lead that keeps its motion: e(d, T) = -0.5 d T^2 (braking by 2 m/s2 opens the gap by 4.0 m at 2.0 s). Statistic: slope b of r on e through the origin over windows and the three offsets, per seed; two-level bootstrap as above. Beside: the mean response per offset in metres, the response of the presence output, the same slope on the windows with a gap below 20 m.

**F, families at T1-kin** for RL against LS and against G by the rule of ADD12 Q3 (`NO-COST`, `COST-<family>`, `GAIN-<family>`), from the T1-kin dump the chain writes; and the reads S (one step) and the train-side ratios, reported.

## 4. Names (each needs both seeds of the arm; given per arm, for LS and for RL)

* H1: `IMAG-HEAD-ADDS` if H1 is above 0 at both seeds and the two-level interval lies above 0; with it `IMAG-HEAD-ADDS-LEAD` if the same holds for H1e, else `IMAG-HEAD-ADDS-EGO-ONLY`. Else `IMAG-HEAD-FLAT`.
* H2: `IMAG-RESPONDS` if b is at least 0.5 at both seeds and the two-level interval lies above 0. `IMAG-RESPONDS-WEAKLY` if b is above 0 at both seeds with the two-level interval above 0 and below 0.5 at a seed. Else `IMAG-DEAF`. A slope above 1.5 at a seed is stated as an overshoot beside the name.
* RL against LS (paired on the same windows, two-level): `RL-IMAGINES-BETTER` if H1 of RL minus H1 of LS is above 0 at both seeds with the interval excluding 0; `RL-RESPONDS-BETTER` if the absolute distance of b from 1 is smaller for RL at both seeds with the interval of the difference excluding 0. Else nothing is named for the pair.

## 5. Controls (all must pass for an arm; else none of its names is given)

* c1: an offset of 0 gives a response of exactly 0.0 (same code path).
* c2: on the logged actions the rollout of the tool equals the rollout of the stack (the banked rollout files of the r-b read) and the head on true states equals the trainer's own head output, each to 1e-5.
* c3 (known positive, CPU test): with a planted predictor that moves one state coordinate by the kinematic displacement and a planted head that reads that coordinate, the tool reads a slope of 1.0 to within 1e-6.
* c4 (the response comes through the action input): on a copy of the model with the weights of the predictor's action input set to zero, the response to every offset is exactly 0.0.
* c5 (calibrated read): with the labels permuted across windows inside each seed H1 and H1e do not lie above 0; with the present read given as the rollout they are 0 to within 1e-9; with the true future read given as the rollout H1 lies above 0.
* c6: pod gate per new arm on real data (declared equals built for the levers of both sets, the rollout term and the lead term finite, a deliberately wrong expectation refused); no training without the PASS token.
* c7: the family read keeps its controls q1 to q5.

## 6. What follows (committed before the data)

* `IMAG-RESPONDS` (for RL or for LS), own card: planning by imagination at T1-kin. At each replan the candidate plans are the model's plan and speed-scaled forms of it; each is rolled out by the world model with its own actions; the head reads the imagined gap along the rollout; the distance rule is applied to the imagined gap in place of the constant-velocity extrapolation of the present read. It is compared with the vision-fed rule of rows H-V8F-PROOF-54 and 58 on the unsafe rate and on the price in position error. With `IMAG-HEAD-FLAT` or `IMAG-HEAD-ADDS-EGO-ONLY` beside it the claim is limited to the ego part (a lead that keeps its motion).
* `IMAG-RESPONDS-WEAKLY` or `IMAG-DEAF`: the rollout does not move the lead gap with the ego's own actions as the kinematics demand. Next lever, own card: an action-consistency term at training on arm RL, the head's read of a rollout under an offset action sequence must move by the kinematic amount against the rollout under the logged actions (a physical rule injected through imagination; it needs the ego kinematics only, no new label). The response read of this card is its bar.
* `IMAG-HEAD-FLAT` for both arms with `IMAG-DEAF` for both: the lead head is trained on rolled-out states as well (labels of the future frame), own card, before any planning by imagination.
* Any `COST-<family>` of RL against LS: RL is not promoted; the next lever of its H1 and H2 outcome still runs on the arm without the cost.

## 7. Limits

Two seeds per arm, one training run per seed. T0 reads on logged frames; the offsets produce action sequences the predictor has rarely or never seen in training; the kinematic expectation assumes a lead that keeps its motion and a straight relative geometry. The head was trained on real states only. The linear calibration of H1 measures what a linear map of the head reads explains. One corpus. A response in imagination is a property of the world model and says nothing yet about driving; the T1-kin card of section 6 is where a driving-side number can appear.
