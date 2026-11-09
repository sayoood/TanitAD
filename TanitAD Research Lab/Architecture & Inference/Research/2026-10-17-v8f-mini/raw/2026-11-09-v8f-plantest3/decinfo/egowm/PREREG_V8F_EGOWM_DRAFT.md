# PREREG_V8F_EGOWM (DRAFT, not frozen; the Master Mind freezes it after the ladder's T0 read): does giving the world-model predictor the measured speed make the planner use the world model? (Master Mind, 2026-11-09, written before any EGOWM code or run)

## Why
`PREREG_V8F_PLANTEST3.md` read NO-GAIN at both M2b seeds with DECODER-WEAK: the level-1 latent reads speed at held-out R2 0.271 / 0.314 (`VERDICT_PLANTEST3.md`). The measured ego state reaches S, T and O; the world model sees only the mean-pooled vision token and the actions (`v8f_mini.py:666-669`). HYPOTHESIS: **the predictor cannot tell how fast the scene will change.** The same acceleration or yaw-rate action moves the image differently at 3 m/s and at 25 m/s, and a state that carries speed weakly cannot resolve it. If this is the mechanism, adding speed to the predictor's input raises k-step skill at the band ends and the latent's response to actions, with no change to the encoder.

Admissibility: measured ego motion is an input at training and inference (PI ruling D-PI-YAWRATE-1; R-EGO: all three layers read the current measured ego state, and none reads a state planned by another layer). Under `kin_mode yaw`, channel 1 of the action already is the measured yaw rate, so the yaw rate adds nothing new. **Speed is the one new scalar.**

## Echo hazard (why this is an input and not a part of the latent)
Published precedent: DINO-WM concatenates proprioception to the latent and predicts it. Here the arm of record does not put speed in the latent or in the prediction target. Two reasons. (1) A JEPA target containing a nearly deterministic channel invites the encoder to favour it. (2) An oracle goal latent containing future speed would hand the planner the action sequence by differencing, the action-echo failure already measured in the programme (EVAL_DOCTRINE section 1.12). So the goal latents and the target latents stay vision-only.

## Arms (world-model only, the ladder rig: adapter, readers and planner frozen at M2b s{k}; driving cannot change, so T0)
| arm | predictor input | seeds |
|---|---|---|
| **Control (already trained by the ladder): R35H or R35, the arm named world model of record** | latent context, actions | 0, 1 |
| **E-KIN** | control + speed at each input step, from a kinematic integration of the step's actions starting at the measured speed at NOW | 0, 1 |
| **E-NOW** | control + the measured speed at NOW, held constant over the rollout | 0, 1 |
| **E-KIN_rep** | E-KIN's flags with seed 2, for the run-to-run floor (H-ESTIM-SEED-1). The seed differs because a same-seed replicate on this rig is bit-identical (`ladder_read/R35B_CKPT_IDENTITY.json`) | 2 |

* Speed enters every level's predictor through its action embedding, as an added channel, zero-initialised so a zero speed weight reproduces the control bit for bit (control 1).
* E-NOW is the strict reading of R-EGO. E-KIN integrates the world model's own candidate actions inside the rollout, which is a simulator step and no reading of another layer's plan; the PI's point on "taken as actions" is open and this arm pair measures whether the distinction matters.
* Every lever is named in each argv, read back from the BUILT model, and each run passes the per-run token route (the mini trainer has no `launch_gate.py` profile, AUDIT F12). Default off.

## Read (T0, the ladder's instruments on the same windows, both training seeds required)
* Deciding per level: IA-0 skill against Copy-Last at the band end (L1 2.0 s, L2 6.0 s, L3 12 s), by IMAG1B's two-part rule, and IA-0 health.
* Deciding for planning: the PLANTEST3 harness on FLAT1 and TD3 with the latent cost and DEC-END, G-TRUE (vision-only goal) minus HOLD and minus G-RAND, LONGITUDINAL and LATERAL, paired EC-BOOT over the SCORE episodes, B 2,000. TACTICAL: n = 0, no decision output. STRATEGIC: PAUSED (PI item 25).
* Named beside row, no bar: the IA-3 lateral gain at L1 2.0 s, a measured defect of every ladder arm (-0.074 to -0.010, gain 1.0 = right size).
* Beside, no bar: WMACT's action-sensitivity audit (latent response to true against shuffled actions), the speed-probe R2 of a rolled-out latent, a baseline where speed is decoded by persistence.

## Committed outcomes (per arm, both seeds)
* **SPEED-HELPS:** IA-0 skill over the control exceeds the E-KIN_rep floor at the L1 band end, and the planner read is PLANNABLE (separated-better than HOLD and than G-RAND) in at least one family. Next: this arm becomes the world model of record, and HWM-PLAN is read on it.
* **SKILL-ONLY:** skill rises, planner stays NO-GAIN. The predictor was missing speed, and the planner's limit is elsewhere (the goal, the cost). Next: the predicted goal path (HWM-PLAN's G4, G5).
* **NO-EFFECT:** skill does not exceed the floor. The mechanism above is refuted. Next: the lever `PREREG_V8F_DECINFO.md` names (spatial pool or adapter).

## To settle before freezing
1. Settled: DECINFO, DECN, PT3N, PT3M and PT3F spent the reader lever on z1 (`VERDICT_PT3F.md`). A goal read from z1 stays above HOLD (speed read error 2.30 / 2.57 m/s at 2.4 s), so the planner read here may land on SKILL-ONLY, which the outcomes above commit.
2. Settled: the ladder read fixes the control's numbers (`ladder_read/VERDICT_LADDER.md`: R35 L1 0.178 / 0.190, R35H L1 0.259 / 0.267).
3. Settled: the runs follow the band retrain on box 2 (PI, 2026-11-09). ESTIMATED training cost: the control's own time per run, 3.3 h for R35b's chain time, 5 new runs (E-KIN and E-NOW at two seeds, plus the replicate).
4. Open: which arm is the control (R35H or R35), the PI's choice after the ladder verdict.
5. Open: the PI's reading of "taken as actions" (only if E-KIN and E-NOW differ).

## Log
