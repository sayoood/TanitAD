# Note N7 to PREREG_V8F_PROOF_ADD12 (SPATIAL): stage 0d is read (FLAT, no power); stage 0e, the same reads on the train corpus (2026-11-13)

Written before any number of stage 0e exists and before its data were extracted.

## 1. Stage 0d, as read (T0, `spatial/s0c/VERDICT_S0C.md`; continuous-lead pairs: 1,586 pairs of 50 episodes, 166 pairs removed by the rule; controls k1 to k4 pass)

Skill over persistence for the change of the lead gap over 2.0 s: ego block alone +0.111 [-0.105, +0.253]; grid state with history -0.087 [-0.299, +0.079]; mean state with history +0.035 [-0.143, +0.183]; grid state without history +0.041 [-0.101, +0.151].
Grid with history minus grid without history -0.129 [-0.256, -0.030]: adding 640 more numbers to a ridge fitted on 40 episodes makes it worse.
Outcomes as notes N5 and N6 write them, stage 0d (continuous lead): `S0-CHANGE-FLAT` for both states, `GRID-EQUAL-CHANGE`, `HISTORY-EQUAL`. Control k2 (planted motion): 1.000 from two grid frames, 0.493 from two mean frames, 0.511 from one grid frame.
Reading committed in N6: 50 episodes give no linear read of the change any power, the ego block included. The read moves to the train corpus.

## 2. Stage 0e (T0, no model trained): the reads of stage 0b and 0d on train clips

* Why it has power (MEASURED on labels, `join_train.npz`): the training join carries a lead gap on 36.9 % of its 461,518 frames, against 2,485 frames with a lead in the eval bank.
* Probe set, fixed by rule: the train clips at positions 600 to 1,799 of the ascending hash order of the train manifest (1,200 clips; the block 540 to 600 that every arm excludes is outside it). No eval clip is in it.
* Tokens: the pod cache, key `pool4x10`. To move them, each cell is projected once from 1,024 to 64 numbers by one fixed Gaussian matrix (seed 0); the states are then built as in stage 0 (per cell 16 principal directions fitted on training folds; the mean token is the mean of the projected cells, 128 directions are replaced by all 64). Extraction is a streaming CPU job on the pod at idle priority, one clip in memory at a time, no GPU.
* Ego vector on train frames: speed, acceleration, yaw rate (the cache fields `v`, `a`, `yaw_rate`). Lead labels: `lead_valid`, `lead_gap` of the join, gaps kept in 0 to 60 m.
* Folds: 5, by clip. Probes, bootstrap (clusters = clips, B 2,000), outcome rules and names: those of stage 0b (r-a: `S0-LEAD-KEPT` / `-LOST`, `GRID-ADDS` / `GRID-EQUAL`) and of stage 0d (change read on continuous-lead pairs: `S0-CHANGE-READ` / `-EGO` / `-FLAT`, `GRID-ADDS-CHANGE` / `GRID-EQUAL-CHANGE`, `HISTORY-ADDS` / `HISTORY-EQUAL`), quoted as "stage 0e (train corpus)". The r-b level read is not repeated (no room, stage 0b).
* To bound the cost, a probe may be fitted on a fixed random subsample of the training rows (at most 20,000, seed 0); the size is printed.

## 3. Controls (each must read its known value, or no outcome name is printed)

* e1: labels permuted within fold: every R2 and skill at or below 0.02.
* e2: planted position (stage 0 s2p) and planted motion (stage 0c k2) on the projected tokens: grid above 0.95, mean at or below 0.60, one grid frame at or below 0.60 for motion.
* e3, the projection keeps what the read needs: on the eval bank (dev-box eval cache, the same projection matrix) the r-a lead gap R2 of the projected grid state is within 0.03 of the stage-0b value 0.3981, and of the projected mean state within 0.05 of 0.2577.
* e4: alignment: for 30 random (clip, frame) rows the extracted kinematics equal the cache values and the join labels equal the join values (exact); clip count 1,200; no eval clip (count 0).

## 4. Both outcomes, committed

* `GRID-ADDS` and / or `S0-CHANGE-READ` for the grid state on the train corpus: the claim "a state that keeps location carries the lead and its motion beyond a mean state and beyond ego motion" stands at the level of untrained linear reads, with power; stage 1 tests whether training realises it.
* `GRID-EQUAL` and `S0-CHANGE-FLAT` with power: pooled 4x10 tokens do not carry more about the lead than their mean plus ego motion to a linear read. The lever that rises is the cue (finer tokens on the lead), own card; stage 1 still runs, since a trained state is not linear.
* In both cases this tool, with the trained states in place of the principal-component states, is the lead read of stage 1 (Q1 retention and Q2 in the change form).

## 5. Limits

Train clips are in-sample for trained models; for stage 0e no model is involved. One fixed projection; 1,200 clips; linear and kernel reads on untrained states; T0, a world-model diagnostic, never driving performance.
