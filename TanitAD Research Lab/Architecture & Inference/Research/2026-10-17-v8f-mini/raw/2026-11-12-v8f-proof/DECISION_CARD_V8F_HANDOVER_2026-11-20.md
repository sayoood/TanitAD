# Decision card D16 (coordinator fork, for the PI and the other session): the inference-side candidate, what the planner reads of its own state, the rate of the lead rule, a strategic read along the rollout, and the heading of the kinematic executor

Written 2026-11-20 by the coordinator fork after the reads of cards ADD62, ADD63, ADD65 and ADD66 (rows 118 to 121 of the register). Section 2e, recommendation D and its cost were completed the same day after the read of card ADD67 (row 122); sections 1 to 2d and recommendations A to C are as first written. Nothing in this card is adopted by this card. Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; the lead does not react; logged camera frames); never driving performance. Evidence class MEASURED (ours) with the row beside each number; the two 10k models `UE_s0`, `UE_s1` are INHERITED from the other session (code cdbff971). 141 evaluation episodes, 2,399 windows, 562 windows scored for the unsafe rate; differences are pooled over the two model seeds with a paired episode-cluster bootstrap interval. The bootstrap answers another draw of episodes; the two model seeds are the only replicate for training variance; the rollout does not sample.

## 1. The decisions asked

A. The inference-side candidate as one proposal per rule form (label-fed, reader-fed): readout K at the replan, the cap-only lead rule, and the rule's second look after 1.0 s. Asked: whether the integrated 10k candidate of the common plan is specified with it, and whether it is reported under the programme's convention and under the candidate side by side (rows 119 to 121).
B. The strategic read along the rollout (card ADD66): whether this study instrument becomes the programme's STRATEGIC read at T1-kin (row 120). Until then this fork reports it as a study instrument in every row.
C. The training lever (the other session's line): train the planner and the strategic head on their own rolled history at the tick. Rows 119 and 120 are the measured case; this fork trains no model for it.
D. The heading of the kinematic executor (the other session's tool): the pinned tool holds the executed heading only below 1 mm of path per step; row 120 measures what that does near a standstill. Asked: whether the tier gets a bound a car obeys. Card ADD67 measured one as a study arm (section 2e, row 122).

Not asked, unchanged: the 2.0 s cycle for every capability number (coordinator decision D12; rows 114 to 116), the braking limit 6.0 of the lead rule, the pinned tool and the evaluation convention of the programme.

## 2. Measured basis

### 2a. What the planner reads at the replan tick (no rule; row 119, card ADD63)

The pinned tool fills the state at the last executed step from a spline through the old plan, so the planner is handed a speed, an acceleration and a heading that contain the part of the old plan it is about to discard.

| readout | against the pinned readout | unsafe rate per model seed |
|---|---|---|
| pinned (the convention) | reference | 0.1459, 0.1139 |
| causal, all of the own acceleration (what a car would measure) | the worst of the dose on every family (against half of it: speed error +1.38 m/s, yaw rate +25.5 deg/s, tactical ADE +1.28 m) | 0.1530, 0.1548 |
| no own acceleration, pinned path | speed error -0.612 m/s [-0.715, -0.521], yaw rate -12.7 deg/s, tactical ADE -0.364 m; lateral cells open | 0.0996, 0.0872 (not separated) |
| readout K: no own acceleration, path from executed steps only | all ten family cells better: speed error -0.613 m/s [-0.722, -0.515], heading -2.83 deg, yaw rate -17.1 deg/s [-21.1, -13.7], tactical ADE -0.728 m [-0.948, -0.545] | 0.1121, 0.0890 (-0.0294 [-0.0668, +0.0064], not separated) |

The family errors rise with the share of its own acceleration the planner reads, at both steps of the dose (0, 0.5, 1) and on all three families. The models were trained on logged histories; on their own rollout they continue the acceleration they read.

### 2b. The rate of the lead rule (row 118, card ADD62)

With the pinned readout, the rule applied again after 1.0 s to the rest of the 2.0 s plan: label-fed the unsafe rate falls from 0.0463, 0.0302 to 0.0338, 0.0249 (-0.0089 [-0.0176, -0.0011]), contact stays at 0, no family cell separated worse. Reader-fed the same size, not separated, contact lower by 0.0101 [-0.0235, -0.0022]. A look every 0.4 s adds nothing. Shortening the model's own cycle to 1.0 s raises the unsafe rate under every readout (rows 114 to 116).

### 2c. The candidate (row 121, card ADD65): readout K, the rule, the second look

| | label-fed | reader-fed |
|---|---|---|
| the rule under K (against no rule under K) | unsafe rate -0.0507 [-0.0739, -0.0310] | -0.0196 [-0.0409, -0.0055] |
| the second look under K | -0.0133 [-0.0235, -0.0049] | -0.0071 [-0.0128, -0.0020] |
| candidate against the convention with the rule: unsafe rate | -0.0018 [-0.0183, +0.0169], not separated | -0.0062 [-0.0372, +0.0269], not separated |
| contact | -0.0016, not separated (0 in both seeds) | +0.0031, not separated |
| falling behind | -0.0678 [-0.1149, -0.0249] | -0.0935 [-0.1539, -0.0354] |
| LONGITUDINAL: speed error / along-track | -0.635 m/s / -0.460 m | -0.635 m/s / -0.512 m |
| LATERAL: heading / yaw rate | -2.83 deg / -16.7 deg/s | -2.99 deg / -16.3 deg/s |
| TACTICAL: tactical ADE | -0.842 m [-1.112, -0.627] | -0.917 m [-1.154, -0.699] |
| STRATEGIC: route goal at 30 / 50 / 80 m, 3.6 s into the rollout | -3.48 / -7.31 / -12.32 m | -3.74 / -7.77 / -12.98 m |
| levels of the candidate: unsafe / contact / falling behind per seed | 0.0391, 0.0338 / 0, 0 / 0.1308, 0.1698 | 0.0765, 0.0712 / 0.0312, 0.0280 / 0.1667, 0.1651 |

All ten family cells and the strategic cells read better than the convention with the rule in both forms. Against the convention without a rule the candidate lowers the unsafe rate (-0.0934 [-0.1440, -0.0463] label-fed, -0.0560 [-0.1064, -0.0098] reader-fed) and contact, with falling behind not separated and every family cell better.

What it costs or leaves open: the gain over the convention is the readout's, and readout K is a study device (it withholds an acceleration a car measures). With the same rule rate the reader-fed form has more contact under K than under the pinned readout (+0.0132 [+0.0002, +0.0285]); label-fed the readout alone reads +0.0116 and +0.0071 on the unsafe rate at both seeds, not separated. The reader-fed unsafe-rate differences lie at or inside the reader floor (0.0196). Two model seeds of one training recipe.

### 2d. STRATEGIC along the rollout (row 120, card ADD66)

The pinned tool writes the strategic head's route checkpoints (30, 50, 80 m) at every replan tick. Scored against the programme's own route label they give a read whose first tick is exactly the other session's T0 checkpoint error. In the convention (no rule) the error rises over its T0 reference by +1.84, +2.99, +4.50 m for an output computed 1.2 s into the rollout and by +8.38, +14.77, +22.95 m at 3.6 s (T0 level 0.96, 2.3, 5.8 m; the means carry heavy tails, median at 30 m 0.25 m at T0 and 4.64 m at 3.6 s). 60 to 73 % of that is in the head's own output scored in the camera's frame (descriptive split): on the same camera frames its raw checkpoints move by 5 m at the 30 m arc between two readouts. Readout K: better than pinned at every arc (-3.42, -7.05, -11.75 m); causal with all of the own acceleration: worse (+3.21, +6.25, +9.28 m); label-fed rule: not separated; reader-fed rule: slightly worse (+0.45, +0.95, +1.44 m). A strategic decision between routes is not scored (n = 0).

### 2e. The heading of the kinematic executor (row 120, note N1 of card ADD66; card ADD67 on the pod)

The pinned tool takes the heading of every executed frame from the tangent of the executed path and holds it only while the tangent is shorter than 1 mm per 0.2 s. In the convention, 3.6 s into the rollout, the executed heading is off the logged one by 52.9 degrees on average where the speed is below 2 m/s (median 39.8 degrees, n 310), 13.3 degrees at 2 to 8 m/s, 3.4 degrees above 8 m/s. The planner's next plan is laid in that frame and the model reads that yaw history.

Card ADD67 (row 122) bounds the heading of every executed frame: it may turn by at most 0.2 1/m times the distance travelled (a turning radius of 5 m); positions and speeds are not changed; the pinned tool is not edited (a study wrapper). Controls e1 and h1 to h9 pass. Differences are the run with the bound minus the same run without it.

| setting | LONGITUDINAL | LATERAL | TACTICAL | STRATEGIC (route goal at 30 / 50 / 80 m) | unsafe rate |
|---|---|---|---|---|---|
| the convention (pinned readout, no rule) | better: speed error -0.240 m/s [-0.317, -0.173], along-track -0.063 m | better: heading -1.70 deg [-2.49, -1.05], yaw rate -4.58 deg/s [-6.55, -2.97] | better: tactical ADE -0.224 m [-0.312, -0.152] | better: -2.20 / -3.71 / -5.42 m | lower: -0.0383 [-0.0691, -0.0134], both seeds (0.1459, 0.1139 to 0.1121, 0.0712) |
| readout K (no rule) | not separated | mixed: heading -0.95 deg better; yaw rate +1.36 deg/s, inside the seed spread | better: -0.088 m [-0.127, -0.055] | better: -1.67 / -2.91 / -4.36 m | not separated (-0.0036) |
| the candidate, label-fed | not separated | mixed: heading -0.92 deg better; yaw rate +1.26 deg/s, inside the seed spread | not separated (-0.066 m, inside the seed spread) | better: -1.83 / -3.11 / -4.60 m | equal at both seeds |
| the candidate, reader-fed | not separated | mixed: heading -0.98 deg better; yaw rate +1.11 deg/s, inside the seed spread | better: -0.072 m [-0.105, -0.043] | better: -1.89 / -3.21 / -4.65 m | not separated (+0.0000) |

No family reads worse in any setting. The bound acts where the offset was: on 92 % of the executed frames below 2 m/s, 17 % at 2 to 8 m/s and 1.5 % above 8 m/s in the convention; the heading offset below 2 m/s falls from 52.9 to 6.6 degrees (under readout K from 27.4 to 6.8 degrees) and is unchanged above 8 m/s. The tighter bound 0.1 1/m reads no worse than 0.2 on any family (strategic slightly better), so this dose does not show where a bound starts to cut turns a car makes. Against the convention with the rule, the candidate with the bound reads: unsafe rate not separated (-0.0018 label-fed, -0.0062 reader-fed), falling behind lower (-0.0631, -0.0911), all ten family cells better (tactical ADE -0.908 m and -0.989 m), strategic -5.33 / -10.42 / -16.94 m and -5.64 / -11.00 / -17.64 m.

What this measures: the unbounded heading of the kinematic executor is a term of the convention's T1-kin errors on all four families and of its unsafe rate. For scale: the strategic error rises on the rollout by +8.38, +14.77, +22.95 m at 3.6 s (section 2d); at that tick it is lower by 2.20, 3.71, 5.42 m with the bound alone and by 5.10, 9.96, 16.13 m with readout K and the bound (differences between runs on the same windows, not shares of the rise). What it leaves open: the bound is one constraint and not a vehicle model (positions still follow the plan's steps exactly); one value and one dose; under readout K the yaw-rate error reads higher by 1.1 to 1.4 deg/s in all three settings (inside the seed spread); the convention with the rule and the causal readout a car would measure have not been run with the bound (card ADD68, next on the pod).

## 3. What the coordinator recommends

A. Specify the integrated 10k candidate with the lead rule and its second look (both forms), and report it under the programme's convention and under readout K side by side. Keep the convention (2.0 s, pinned readout) as the convention of record. Do not adopt readout K as the deployed handover by itself: it shows what the planner does when it is not handed its own rollout's acceleration, and the repair at the source is C.
B. Yes, as the STRATEGIC read at T1-kin beside the first-tick read, with its limits stated (logged camera frames; route goal only). It costs no rollout: it reads dumps the pinned tool already writes.
C. Yes. The check after such a training is built: on the new checkpoint the causal readout with all of the own acceleration should stop being the worst (the runs of card ADD63), and the strategic read should stop degrading (card ADD66).
D. Yes: give the T1-kin tier a heading bound a car obeys. The measured case is section 2e: it lowers the convention's errors on all four families and its unsafe rate, acts almost only near a standstill, and no family reads worse in any of the four settings. Proposed form: the curvature bound of card ADD67 (0.2 1/m) or a vehicle constraint of the other session's choice that does the same near a standstill; the value is not tuned (0.1 reads no worse), and card ADD68 adds doses in the convention. Proposed way: one overlap read in which the reference runs of the programme are stated under the unbounded and the bounded executor side by side, and from then on the bounded one as the tier of record. This changes the evaluation convention, so it is the PI's and the other session's decision; until then every row of this line carries the limit that its executor has no heading bound. With D the recommendation A stays as written: the candidate still reads better than the convention with the rule when the candidate has the bound, and whether that holds when both have it is one of the reads of card ADD68.

## 4. What each choice costs

* A: one more application of the rule per plan; side-by-side reporting is one more rollout per arm and seed (about 4 to 5 minutes each on the A40).
* B: one CPU read of about a minute per set of runs.
* C: a training change in the other session's tree and its gate; not started by this fork.
* D: a change of the pinned tool or a wrapper around it (the bound is one short function applied to the heading column after the readout; the study wrapper and its tool check with planted faults are landed), and every T1-kin number of the programme would have to be stated under the old and the new executor for one overlap read (one rollout per reference run and seed, about 4 minutes each on the A40). Earlier T1-kin rows stand as measured under the unbounded executor.

## 5. Where the numbers are

Rows H-V8F-PROOF-118 to 122 of `Project Steering/GOALS_AND_CLAIMS.md` on the branch `agent/arch-inf-20260803` (commits d6bb0416 to bb061af6); cards `PREREG_V8F_PROOF_ADD62`, `ADD63`, `ADD65`, `ADD66` (with Amendment 1 and note N1), `ADD67`; result files under `integ10k/read_rulefast/`, `read_accgain/`, `read_cand/`, `read_stratroll/`, `read_headbound/` of the v8F proof folder.
