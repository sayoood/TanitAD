# VERDICT_PT3M (Master Mind, 2026-11-09)
Card `PREREG_V8F_PT3M.md` (one folder up in the repo), frozen prefix sha256 `70797004...1f1d0eec`. Tier **T0**, an oracle-goal diagnostic, never a driving result. MEASURED: `RESULT_PT3M.md`, `peek_ceil.log` (ADE and speed error of the ceilings); arrays PRIVATE on HF, `HF_BANK_PT3M.json`. Estimator on every interval: paired episode-cluster bootstrap over the 71 SCORE episodes (994 windows), B 2,000. It answers "another draw of episodes"; both training seeds are required, and the planner-seed replicate reads at most 0.0004 on every row. HOLD is 0.3348 m/s^2 and 0.0256 rad/s.

## Validity
All six controls pass. Reader tie to DECN difference 0.00000 on all four rows; reachability non-zero in both channels (1.4e-2 to 2.1e-2) and exactly 0.0 under SHRINK(0); harness tie relative cost difference at most 2.7e-14; split, clips, isolation pass; run errors 0.

## Outcome: NO-GAIN in both families, both forms, both training seeds
| family | form | s0: G-TRUE minus HOLD | s1: G-TRUE minus HOLD | label |
|---|---|---|---|---|
| LONGITUDINAL (m/s^2) | plain | +0.0151 [+0.0030, +0.0328] worse | +0.0287 [+0.0132, +0.0486] worse | NO-GAIN, HOLD-COLLAPSE at s0 |
| LONGITUDINAL (m/s^2) | shrink | +0.0065 [-0.0032, +0.0200] none | +0.0091 [-0.0000, +0.0214] none | NO-GAIN, HOLD-COLLAPSE at both |
| LATERAL (rad/s) | plain | +0.0028 [+0.0009, +0.0052] worse | +0.0031 [+0.0017, +0.0048] worse | NO-GAIN |
| LATERAL (rad/s) | shrink | +0.0023 [+0.0006, +0.0045] worse | +0.0021 [+0.0007, +0.0039] worse | NO-GAIN |

TACTICAL: not computed (n = 0, this read has no decision output). STRATEGIC: PAUSED (PI item 25).

* **The goal does reach the plan.** G-TRUE beats G-RAND, separated-better, on all eight rows (-0.0196 to -0.0383 m/s^2; -0.0041 to -0.0070 rad/s).
* **The four-frame reader did not help.** PT3M minus PT3N is separated-worse on all four LONGITUDINAL rows (+0.0106 to +0.0240 m/s^2) and not separated on the four LATERAL rows.
* **Lambda sits at the grid edge (1) on all eight cells.** On TUNE the unregularised plan (lambda 0) reads 1.28 to 1.38 m/s^2 and 0.135 to 0.156 rad/s, about four times (acceleration) and six to eight times (yaw rate) the lambda = 1 values on the same windows. The cost alone pulls the plan away from the logged actions; the prior holds it back.
* **Decoded distance to the goal does not fall along the chosen plan** (monotonicity -0.05 to -0.19), while the planner lowers its own cost on at least 99 % of windows. The optimiser works; the cost it lowers is not the quantity that matters.
* **DECODER-WEAK is raised** at both seeds (reader on encoded latents: speed 0.695 / 0.686 against 0.70; yaw rate 0.430 / 0.396 against 0.50). KIN-CEILING does not beat HOLD (acceleration 1.40 m/s^2; yaw rate 0.030 / 0.031 rad/s).

## The ceilings (no world model, no planner)
| family | HOLD | ORACLE-END | ORACLE minus HOLD | READ-END s0 / s1 | READ minus HOLD s0 / s1 |
|---|---|---|---|---|---|
| LONGITUDINAL (m/s^2) | 0.3348 | 0.2161 | -0.1187 [-0.1364, -0.1014] | 1.151 / 1.115 | +0.816 worse / +0.780 worse |
| LATERAL (rad/s) | 0.0256 | 0.0142 | -0.0114 [-0.0151, -0.0082] | 0.0255 / 0.0261 | -0.0001 none / +0.0005 none |

**An end-point goal known exactly beats HOLD by 35 % in acceleration and 45 % in yaw rate, both seeds, separated.** ADE at 2.4 s reads 0.2066 m against HOLD's 0.2284 m. So the end-point form is not the limit. The same formula with the two values read from the encoded future does not beat HOLD in either family.

## The learning-rate rule, as frozen
The frozen rule picks the MLP learning rate by early-stopping loss. It chose 1e-4 at both seeds, where the three losses differ by about 1 % to 6 %. At that rate the yaw-rate reader reads 0.430 / 0.396 on the held-out episodes, against 0.570 / 0.441 at 1e-3. The result above is the card's arm as written. The reader at 1e-3 is carried into the next arm as its own labelled row.

## Beside, written after the READ-END read, decides nothing (`diag_response.py`, `DIAG_RESPONSE.json`; T0, SCORE)
* **Differencing two reads does not rescue the speed.** The difference of the reads at knots 12 and 3 has mean absolute error 1.32 / 1.31 m/s, against a true change of 0.78 m/s on average. Acceleration from it reads 0.779 / 0.770 m/s^2, worse than HOLD.
* **The world model's action response, seen through this reader, is weak.** Between the logged actions and HOLD, the read speed at knot 12 moves 0.32 / 0.35 m/s per m/s of integrated difference (line R2 0.61 / 0.50); read yaw rate moves 0.06 / 0.07 per rad/s (R2 0.08 / 0.10). Under SHRINK these are 0.18 / 0.19 and 0.03 / 0.04.
* ESTIMATED, assuming error scales with the square root of 1 - R2: matching HOLD's 0.69 m/s speed error at 2.4 s needs a speed R2 near 0.98.

## Trigger applied
The card's row for this case, in both families: *no PLANNABLE row, ORACLE-END beats HOLD and READ-END does not.* The reader's precision is the limit. Committed next: the N4565 readers of the full-n probe, and the measured-speed world-model card (`egowm/PREREG_V8F_EGOWM_DRAFT.md`). Every outcome: the harness is read on the ladder world models when they finish; PLANTEST and PLANTEST2 stay required gates.

## Limits
* SCORE is one val-side subset (71 episodes); the interval is blind to training variance beyond the two seeds and to the choice of world model (M2b only).
* The goal is the encoded true future, so this says nothing about a goal the tactical layer would predict.
* HOLD is hard to beat on these 2.4 s windows: its speed error is 0.69 m/s.
* A failing read through an N1053 reader leaves open what a reader of a different form would do. The READ-END ceiling bounds that cheaply and goes first.

## Next (Rule Zero)
1. **N4565 READ-END ceiling** on z1, under its own frozen card, with the yaw-rate reader at 1e-3 as a labelled row. It needs no world model, no planner and no GPU. If READ-END beats HOLD in a family at both seeds, the planner run with those readers follows. If not, the reader lever on z1 is spent at this scale.
2. **Ladder harness read** (about 18:00, ESTIMATED): this cost on the wider predictors, whose action response is the second half of the problem.
3. **Measured-speed world model:** card frozen after the ladder's T0 read; runs on box 2 after the band retrain (PI, 2026-11-09).
