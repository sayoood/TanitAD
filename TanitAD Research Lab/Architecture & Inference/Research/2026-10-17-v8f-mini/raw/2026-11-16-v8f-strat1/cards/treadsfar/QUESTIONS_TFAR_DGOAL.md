# QUESTIONS_TFAR_DGOAL: the PI's decisions for T-reads-far and derived goals

Cards: `PREREG_V8F_TFAR.md` (frozen by its author, sha256 beside; not built, not run) and `PREREG_V8F_DGOAL.md` (scoping note). Only the PI's decisions are listed here: architecture changes, training go and pod time, and definitions. Costs are ESTIMATED from the CKPT120 arms' wall time of about 15-16 min per 3,000-step arm on the LoRA pod (RTX 4000 Ada), MEASURED from the MM's pod-log timestamps.

## Why T-reads-far, and what we expect (one paragraph)
CKPT120 read H2 S120-FOLLOWS: S places its 120 m point on the side of the supplied route (T0, with the route supplied along the driven path). Today that point goes to the loss and evaluator only. This card lets T read it. The labels say where it could help: on 202 eval windows (48 episodes) where the road bends between 80 and 120 m, T's label is decelerate-or-stop on 33 %, against 14 % where it does not bend. Lateral has almost nothing to gain: at most 12 of 2,399 windows. A probe of H2's banked outputs finds the predicted far point adds little beyond what T already reads: probe AUC +0.030 at s0 and +0.004 at s1 on those windows. **We expect a small or null gain and say so in advance.** An ORACLE row (T fed the true 120 m point, diagnostic only) tells which lever comes next either way.

## Q1. Architecture change: does T read S's 120 m point? (TFAR)
* **What changes.** One new 1,280-parameter layer, zero-initialised, adds S's 120 m point to T's existing checkpoint token. O is untouched. At step 0 the model is bit-identical to H2.
* **Admissibility.** The point is computed from S's inputs only (camera, measured ego motion, nav, limit, route text). It carries no situation-classifier output (the model has none; pinned). T cannot train S through it (gradient exactly 0, pinned).
* **Recommended: YES.** It is CKPT120's committed next step, it is cheap, and the ORACLE row routes the next lever whatever the outcome.

## Q2. Training go and pod time (TFAR)
* **Arms.** 6 arms of 3,000 steps on the LoRA pod, one job at a time:
  * T-reads-far at seeds 0 and 1, plus a replicate (the training-run floor);
  * a must-fail arm (the far point scrambled across samples at train);
  * a token-off arm (the layer built, no information);
  * the attribution arm of Q3.
* **Time.** About **1.6 pod-h**, plus a 5-minute pilot and the G-SUITE on pod CPU. Conditional arms run only on their committed outcomes:
  * a 4-seed confirmation, about 0.5 pod-h;
  * a matched continuation pair, about 1.1 pod-h;
  * branches of about 0.5 pod-h each.
  * A projected chain above 4 h stops and returns to the MM.
* **Queue.** After DSCALE block 3 (to about 05:00Z, E) and NAVRB2.
* **Recommended: YES**, slotted after NAVRB2.

## Q3. Attribution arm: may T read the route text directly for one arm? (TN)
* **What changes.** For one arm only, T reads the route text's next turn through the same zero-init form. This lifts CKPT120's rule "T never reads the route-text token" for that arm alone.
* **Why.** It separates "S's processed far point helps" (the hierarchy) from "the route text reaching T helps" (flat). Without it, a gain cannot be credited to the hierarchy.
* **Cost.** 16 min.
* **Recommended: YES.**

## Q4. H2 at seeds 10 and 11
* **What.** CKPT120 read H2 driving SEEDS-DISAGREE, and its card commits H2_s10 / H2_s11. The TFAR 4-seed confirmation needs them as comparators.
* **Cost.** About 0.5 pod-h.
* **Recommended:** confirm they run under CKPT120's go, first in the TFAR chain. If your CKPT120 go did not cover conditional arms, this is the go.

## Q5. Definition: what are the derived goals? (DGOAL)
* **G1, the geometric goal** (S's route checkpoints incl. 120 m), exists, and T reading it is Q1.
* **G2, a 6-30 s speed goal**, was measured on 4,350 training clips. Weather, surface, illumination, period of day and country together predict only **+0.3 % of variance** beyond the current speed (R² 0.801 → 0.805; a scrambled-label control reads −0.1 %). Drivers in rain or snow are already slower now. Only "dark" shows a signal.
* **G3, a time-gap goal**, is not measured yet. The lead table is on the pod.
* **G4, stop goals**, depends on the stop-line card.
* **Recommended:**
  * name G1 as derived goal number one (it is TFAR);
  * let G3's labels-only measurement run (CPU, no go needed);
  * tie G4 to the stop-line card;
  * keep G2 as a candidate only if G3 or the PI's intent says otherwise.

## Q6. DGOAL base rung D0 (only after Q5)
* **What.** H2 plus the detected-information heads in their detached form (the S1d form), which removed S1's lane-keeping cost and still detects 4 of 5 variables. It has never been trained on H2.
* **Cost.** 2 arms + replicate, about 0.8 pod-h.
* **Recommended: hold until Q5.**

## What runs with no go
* Nothing is built or trained.
* Done tonight (CPU, labels and banked outputs): the TFAR label measurement, the H2 far-point probe, and the G2 information measurement.
* Next without a go: the G3 measurement (CPU).
