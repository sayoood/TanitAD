**Night of 13 to 14 Nov, standing at 02:20 UTC.** Every measured value is a T0 or T1-kin diagnostic on logged frames, never driving performance. The strategic family is paused (PI item 25).
* The grid state knows where the lead vehicle is, and the gain is the cell layout, not the width (both seeds, three reads). Its predictor did not imagine: rolled out 2.0 s it reads the lead no better than a copy of the present state, in the lead read and in pixels.
* Why, measured without any training: the flat predictor is worse than a plain copy of the state at the single step it is trained on (error ratio 1.23 and 1.25, intervals wholly above 1). The rollout context is not the cause.
* First positive result on the predictor: a predictor that attends across the cells (CELLPRED, seed 0) is below the copy one step ahead (0.79), keeps the position and costs no family. Its 2.0 s rollout still does not separate from the copy; seed 1 is training.
* Next arms, pre-registered and queued behind their gates: a rollout loss over 2.0 s on the cell predictor (both seeds), and a predictor that outputs the state plus a change on the flat one.
* Data efficiency, seed 0: full, half and a quarter of the clips read alike; the first cost appears at one eighth (+0.25 m on 6 s ADE against a quarter). Seed 1 is training; names need both seeds and six baseline seeds.
* The six-seed hierarchy against flat replicate is still running on the third slot of the pod.

-----END-TOP-----

**Spatial state, stage 1 with both seeds (addendum 12), MEASURED, T0 for the state reads, T1-kin preliminary for the families.** Sources `spatial/s1/`, `spatial/q2/out/`, `spatial/latvis_gw/r2/VERDICT_LATVIS_GW.md`, `spatial/q3/out_prelim_b/VERDICT_Q3.md`; all controls pass.
* Position of the lead, held-out clips, gap R2: grid state 0.552 and 0.541 (two seeds), wide mean state 0.361 and 0.309, baseline 0.237, 0.212, 0.143. On 1,200 training clips, paired: grid minus wide mean +0.104 [+0.074, +0.134] and +0.116 [+0.087, +0.146]; wide mean minus baseline -0.008 and +0.009. The gain is assigned to LOCATION.
* Pictures decoded from the state: grid skill 0.605 and 0.555 against 0.777 for the token grid (ratio 0.78 and 0.72, bar 0.90): outcome Z-ABOVE-MEAN. The wide mean state reads 0.338 and 0.345: Z-FLAT.
* Imagination: lead gap R2 at +2.0 s from the rolled-out state, the present state, the true future state: 0.442 / 0.496 / 0.552 and 0.437 / 0.473 / 0.543. Rollout minus copy -0.054 [-0.154, +0.033] and -0.036 [-0.153, +0.064]: outcome IMAGINE-FLAT at both seeds, and the same name in pixels.
* Four families at T1-kin, both seeds of each pair: no family cell worse for the grid state or the wide mean state against the baseline; preliminary until six baseline seeds are read.

**Why the rollout is flat (addendum 17, CTXROLL), MEASURED, T0; no training, the banked checkpoints.** Source `spatial/q2/ctx/VERDICT_CTXROLL.md`, `STEP_CTX.md`; all controls pass. Card frozen before any number.
* The predictor is trained on three true states and rolled out from one. Seeded with the true history it reads the lead no better than the copy (-0.034 and -0.004), and only +0.010 [-0.005, +0.030] and +0.028 [+0.008, +0.055] better than the one-state rollout: outcome CTX-FLAT. The context is not the limit.
* The finding of the read: one step (0.2 s) ahead the error of the flat predictor is 1.233 [1.180, 1.295] and 1.254 [1.197, 1.317] times the error of copying the state. It has not learned the change of the state at the step it is trained on.

**A predictor that attends across cells and time (addendum 16, CELLPRED), seed 0 of two, MEASURED, T0 and T1-kin preliminary.** Sources `spatial/q2/out_p/`, `spatial/q2/ctx/STEP_CTX.md`, `spatial/s1/VERDICT_S1_P_s0.md`, `spatial/latvis_p/r1/VERDICT_LATVIS_GW.md`, `spatial/q3/out_prelim_p_s0/VERDICT_Q3.md`; all controls pass. One seed names nothing.
* One step ahead it is below the copy: 0.788 [0.770, 0.807]. It has fewer parameters than the flat predictor (873,488 against 2,074,496).
* Lead gap R2 at +2.0 s: rolled-out 0.501, present 0.487, true future 0.532. Rollout minus copy +0.013 [-0.037, +0.057]; against the flat rollout, paired, +0.058 [-0.030, +0.163]. No interval excludes 0, so this seed reads CELL-FLAT by the rule as written; every line moved the wanted way.
* What is missing: the true future state holds a part of the change of the gap (skill +0.190 [+0.062, +0.330]); the rollout delivers +0.007. One step of 0.2 s barely contains the motion of the lead.
* Position kept (gap R2 against the grid arm +0.010 [-0.011, +0.031]); no family cell worse at T1-kin.

**Next lever (addendum 18, MOTION), pre-registered 14 Nov, built and gated; no result.** Sources `PREREG_V8F_PROOF_ADD18.md`, `spatial/motion/`.
* Two levers: the predictor is also trained on its own 2.0 s rollouts; and it outputs the state plus a change, so that it starts as the copy. Defaults are bit-identical to the earlier arms; 55 tests pass; the gate on real data passed in a dry run.
* Queued on two slots of the pod: the rollout loss on the cell predictor at both seeds, then the residual form with and without the rollout loss on the flat predictor.

**Data efficiency (addendum 13), seed 0, MEASURED, T1-kin, preliminary.** Sources `dataeff/read/out_prelim_s0*/VERDICT_Q3.md`; controls pass.
* 6 s ADE on all windows / on selection-critical windows: full 3.193 / 3.703 m, half 3.144 / 3.566, quarter 3.152 / 3.661, eighth 3.405 / 4.005.
* One eighth against a quarter: +0.253 m [+0.075, +0.451], outside the spread of the three baseline seeds. Half and quarter against full: no cell worse.
* Clips chosen by their share of selection-critical windows, at a quarter: no gain on critical windows and a loss on common-case windows (+0.80 m). The one-eighth choice arms, where random selection starts to cost, are queued.
