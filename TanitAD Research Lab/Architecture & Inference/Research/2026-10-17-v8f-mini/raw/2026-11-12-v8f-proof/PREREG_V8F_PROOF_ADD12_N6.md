# Note N6 to PREREG_V8F_PROOF_ADD12 (SPATIAL): stage 0c is read (all FLAT); a label defect of its target; stage 0d, one change, written before its run (2026-11-13)

## 1. Stage 0c, as read (T0, `spatial/s0c/VERDICT_S0C_c.md`; 1,752 pairs of 51 episodes; controls k1 to k4 pass)

Skill over persistence for the change of the lead gap over 2.0 s: grid state with history +0.002 [-0.093, +0.102]; mean state with history +0.044 [-0.013, +0.114]; grid state without history +0.006 [-0.080, +0.104]; ego block alone -0.033 [-0.151, +0.057].
Outcomes as note N5 writes them: `S0-CHANGE-FLAT` for both states, `GRID-EQUAL-CHANGE`, `HISTORY-EQUAL`. These stand for stage 0c.
Control k2 (planted motion, the prediction of VC-P8): AUROC 1.000 from two grid frames, 0.507 from two mean frames, 0.524 from one grid frame. The instrument reads motion where the state carries it.

## 2. What the labels show (MEASURED on labels only, after 0c was read; `res/cache/frames.npz`)

Not even the ego block with the TRUE future ego vector predicts the change. A check of the target, with no state and no model:
* Between consecutive frames (0.2 s) with a lead at both, the gap moves by a median of 0.31 m and by 0.97 m at the 90th percentile; 1.28 % of the steps exceed 3 m (the 99th percentile is 11.3 m). A step of 3 m in 0.2 s is a relative speed of 15 m/s: the label has moved to another vehicle.
* Of the 1,931 pairs 2.0 s apart with a lead at both ends, 185 (9.6 %) contain such a break (a step above 3 m, or no lead on a frame in between). Their change has a standard deviation of 15.0 m against 4.44 m for the 1,746 continuous pairs, and they carry 54.7 % of the sum of squared changes.
So more than half of the target of stage 0c is the lead label switching vehicles, which nothing at time t can predict. The lead label carries no vehicle identity; continuity is the admissible stand-in.

## 3. Stage 0d: what changes, fixed now, before any 0d number

I have seen every number of stage 0c. The change is a repair of the target, made from label statistics alone, and is labelled as such wherever a 0d number is quoted.
* **One change, the pair set.** A pair (clip, g), (clip, g+10) is used only if the lead gap is finite on all 11 frames g .. g+10 and no step between consecutive frames exceeds 3.0 m in absolute value ("continuous lead"). The history frame g-5 needs a bank frame as before, not a lead.
* Everything else is stage 0c unchanged: target, ego block, arms, skill, bootstrap, outcome names of note N5, controls k1 to k4. One count is added to the output: pairs removed by the rule.
* The names are quoted as "stage 0d (continuous lead)".
* If every arm is FLAT again including the ego block, the reading is: on 51 episodes the change of the gap is not predictable by any linear read available here, and the read needs the train corpus (the pod cache) to have power; that read is then prepared for the trained states of stage 1, where it is needed in any case.
