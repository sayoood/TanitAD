# PREREG_V8F_PROOF_ADD17: CTXROLL, is the rollout flat because the predictor is rolled out from ONE state while it was trained on THREE? (2026-11-14)

Written after the lead read r-b and the pixel read of the arms G and W (rows H-V8F-PROOF-26, -28: `IMAGINE-FLAT` at both seeds on both reads)
and BEFORE any number of this read exists. No training, no pod time: a read on the banked checkpoints of G_s0 and G_s1. Tier T0
(a world-model diagnostic on logged frames, never driving performance). It changes nothing in the cards ADD12, ADD16 or the note ADD16_N1.

## 1. Why (MEASURED, with the artifact)

* The level-1 predictor is trained teacher-forced in chunks of its context, 3 states (`hjepa.py`, `_pred_loss`): two of three predicted positions see earlier TRUE states.
* Every imagination read so far rolls it out from ONE state with context 1 (`wmceil.level_pack`, the rule of the banked reader; `HJEPAStack.rollout(context=1)`).
* The trained states hold the position of the lead but not its motion: on the change read no state with history adds to ego motion (row 26: G -0.050 / -0.057).
* In pixels the learned rollout is below the copy at 0.8 s and not at 2.0 s, while a ridge on the start state, the actions and the rollout is below the copy at 2.0 s (row 28).

Hypothesis (HYPOTHESIS, not measured): a single state does not carry the motion of the scene, the history does; rolled out from one state the predictor can only move the scene by the ego action.
If so, the same predictor seeded with the true history must read the lead better at +2.0 s than when rolled out from one state.

## 2. The read

* Arms: G_s0, G_s1 (card rule: an outcome needs both seeds). W_s0 is reported beside if its files are present. The arms P of ADD16 get the same read when they exist; for them it is reported beside C1 and names nothing of ADD16.
* Windows and states: the windows of the pixel read (`LATENTS_GW_<ARM>.npz`, 1,948 windows, 16 frames at 0.2 s) and the per-frame states the pixel read collected (`latents/<ARM>.pt`), matched to the window list by exact equality of the frame-0 state.
* Start frame 4 of the window (0.8 s after its first frame), target frame 14 (+2.0 s). Actions: the WM-scaled actions of the window, transitions 4 to 13.
* Three rollouts of the SAME checkpoint to frame 14:
  * R1: from the state of frame 4 alone, context 1 (the banked rule).
  * R3h: seeded with the TRUE states of frames 2, 3, 4 and their actions, then autoregressive with a context of 3 (true states are replaced by predictions as the rollout advances).
  * R3g: from the state of frame 4 alone, the context growing to 3 with its own predictions (`HJEPAStack.rollout(context=3)`). It separates the history from the context size.
* Lead read: `spatial/q2/rb_trained.py` unchanged (the r-b definitions of stage 0: probes, folds, floors, bootstrap B 2,000), once per rollout, with present state = state of frame 4, true future state = state of frame 14.
* Beside, names nothing: the change form of note ADD16_N1 (`rb_change.py`) per rollout; the one-step error of the predictor at frame 4 to 5 with context 1 and with context 3 over the error of the copy.

## 3. Controls (all must pass; else the read names nothing)

* x1 reproduce: this tool's context-1 rollout from frame 0 over 10 steps equals the banked `pred_2s` of the pixel read (max abs difference at or below 1e-4).
* x2 mapping: each of the 1,948 windows matches exactly one row of the state file; the state of frame 10 equals the banked `true_2s` (max abs difference at or below 1e-5).
* x3 identity: the seeded loop with its context cut to 1 equals R1 (max abs difference at or below 1e-5); it proves the seeded loop is the computation of the stack when no history is used.
* x4 overlap: for windows whose start + 4 is the start of another window of the same episode, the state of frame 4 equals the other window's frame-0 state and R1 equals that window's banked `pred_2s` (max abs difference at or below 1e-3).
* x5: the controls t1 to t6 of `rb_trained.py` pass in every run; k1 to k3 of `rb_change.py` pass.

## 4. Outcomes (committed before data; per seed, the name needs both seeds of G, a mixed pair reads the weaker name)

* `CTX-RECOVERS`: on the lead read, R3h minus the copy is above 0 with the interval excluding 0.
* `CTX-HELPS`: not that, but R3h minus R1, paired on the same windows, is above 0 with the interval excluding 0.
* `CTX-FLAT`: neither.
* Secondary sentence (as note ADD16_N1): if the change skill of R3h is above 0 with the interval excluding 0 at both seeds and above that of R1 (paired, interval excluding 0, both seeds), the report adds: the history moves the lead in the rolled-out state.
* R3g is reported: if R3g equals R3h within the intervals the gain is the context size, not the true history.

## 5. Committed next steps

* `CTX-RECOVERS`: every imagination read and the planning lever use the history-seeded rollout from here on (no training needed); the lever for a state that carries motion by itself (one-state rollout) stays the outcome lever of ADD16.
* `CTX-HELPS`: the mismatch is a part of the limit. Next lever, own card: a rollout loss over 2.0 s with the history-seeded context (the predictor trained on its own rollouts from the context it is given at read time), on the predictor shape that ADD16 names.
* `CTX-FLAT`: the context is not the limit; the next lever is the one ADD16 commits for its outcome, unchanged.
* Whatever the name, the outcome of ADD16 (C1 on the banked rule) is reported as written.

## 6. Limits

The start frame differs from the r-b read of rows 26 (frame 4, not 0), so the windows with a lead at both times differ and the levels are not compared across the two reads;
one checkpoint per arm and seed; a probe shows what is linearly readable; 139 episodes. LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
