# THEORY_CRAWL: why v8F crawls, what bounds every lever, and what weight-space reading should find (Master Mind, 2026-11-06, before any measurement named here)

The PI, 2026-11-06, verbatim: *"Incorporate also in your strategy theoretic analysis and reasoning and not only experiments. Try to analyze our mini mdel with methods like weight space programming"*

Evidence classes: **[M]** MEASURED (artifact named), **[P]** PUBLISHED, **[H]** HYPOTHESIS. Every prediction below is written before its measurement and can fail.

## 1. The facts the theory must explain (all [M], `V8F_SCIENTIFIC_REPORT.md` v41 unless named)
* **F1, T2 read of record** (`t2v8f/stage4/score/DOMINANCE_T2.md`). C0C − P10 at T2:
  * M1 crawl share +0.4073 / +0.4041; along-track +1.42 / +1.30 m; ADE6 +0.79 / +0.64 m; TAC-LON separated-worse;
  * LATERAL separated-better (yaw-rate, curvature, cross-track);
  * **speed MAE separated-better** (−0.566 / −0.449 m/s).
* **F2.** The frozen DINOv3-L field caps the go/stay cue (§7.31, §7.33):
  * AUC at V+E: P-STOP 0.680, P-MOVE 0.785; ego alone 0.626 and 0.732;
  * the model's own four frames add nothing;
  * finer grids and learned attention readouts do not help.
* **F3.** The crawl gap sits mostly in MOVING windows at v0 3–8 m/s, not in the go/stay trigger windows (15 % of windows) (§7.34):
  * 9.0 / 11.4 % of plan steps fall below 0.5 m/s, against 0.7 % for P10;
  * about half the crawl sits behind a lead (§7.38).
* **F4.** O is insensitive to a supplied speed level: +3 m/s moves the plan +0.0160 / +0.0049 m/s (§7.39).
  * The exact logged level, forced through the operator, closes 0.738 of M1 and fails the guards (§7.41, A5).
  * Bounded lifts and floors trade crawl for guard failures (A6, LF, lonfloor).
* **F5.** History at the EXECUTED speed (HCV) made M1 worse (§7.35).

## 2. Theory
**T1. A mean-regressing planner, iterated on its own speed, contracts to a crawl. [H]**
* **The trained planner is a conditional mean.** Trained under squared error, O's longitudinal output approximates E[v_{t+h} | z_t], where z_t holds the frozen tokens, the ego history and T's code.
  * Wherever the features do not resolve the driver's intent (F2), that mean mixes "keep going" with "slow or stop".
  * So, at speed v, the plan's speed satisfies g(v) = E[v⁺ | v, z] = v − δ(v, z). Here δ > 0 on the share of states where the logged future holds a deceleration that the features cannot predict.
* **Open loop, this is harmless.** The history is the human's, so one step of bias does not compound.
* **Under T1 and T2 the planner's own executed speed becomes its history.** The speed then follows the return map v_{k+1} = g(v_k):
  * it converges to the fixed point v* where δ(v*) = 0;
  * the convergence rate is |g′(v*)| < 1.
* **The prediction:** a fixed point well below the human's speed in the 3–8 m/s band, which is F3. F5 is this mechanism observed directly: feeding the executed speed back made the crawl worse.

**T2. No downstream lever can beat the information frontier (the data-processing inequality). [H on the bound's tightness; the inequality itself is a theorem]**
* **The bound.** Let I = intent (keep or slow), and z the features O and T see. Any function of z, whether a retrained head, a re-weighting, DAgger on the same features, a floor, a level, or a weight edit of O or T, has go/stay skill at most that of the Bayes classifier on z.
  * That classifier's ROC curve is the frontier.
  * Every downstream lever moves along it: fewer crawl states cost more stop violations.
  * This is F4's pattern. Floors and forced levels close the crawl and fail the guards.
* **⇒ A DOMINANT needs a point on the frontier with BOTH the M1 closure and intact guards.** At AUC 0.68–0.79, that point may not exist. If it does not, no amount of downstream training finds it.
* **⇒ The lever ranking that follows.** Only a lever that changes z can move the frontier: LoRA, motion features, more frames, a new sensor channel. Downstream levers can only choose a point on it. This is the formal form of "the lever is perception".

**T3. What LoRA on q/v, with a fixed 4×10 pool and a per-frame encoder, can and cannot add. [H]**
* **Can:** re-weight what each frame's pooled tokens keep. Light state, lead-vehicle state, and lane-occupancy detail near the ego are single-frame cues. LoRA can raise their share where the frozen pool averages them away.
* **Cannot:** create motion inside one frame. Motion must come from O and T differencing tokens across the held ages, and F2 measured that the frozen readout carries none.
* **⇒ Prediction for the LoRA arm.** It moves the frontier on the single-frame cue classes, such as lights and leads, and not on pure self-motion.
  * Its mechanism read should therefore include the go/stay AUC on L's features against V+E, per cue class.
  * This sits beside the card's M-LAT and decides nothing.

**T4. What weight-space reading (WSP) can and cannot do here. [P] + [H]**
* **[P]** Takehi, Teoh, Zhang and Khattab (2026-11-02, *Sparks of Weight-Space Programming*, a blog post, not peer-reviewed). An LLM read a 7.16M-parameter net's circuits from its weights with Python analysis and no forward passes, and its claims matched ablation ground truth. Its strongest edit subtracted one direction from 64 output columns in two MLPs and removed an obstacle (reasoning loops). The authors state that an edit removes an obstacle and does not add an ability (a Gemma 3 270M math edit failed).
* **[H] Mapping to v8F.** By T2, an edit of O cannot add the intent information F2 says is missing. What it can remove is the contraction bias δ, an obstacle, where it is not information-bearing:
  * the share of δ that is unconditional "regress to the mean" rather than a cue-driven brake.
  * By T2 any such edit still lands on the frontier. Its value is diagnostic (where δ lives in the weights), and it adds a frontier point at zero training cost.

## 3. Predictions, written before their measurement (each can fail)
* **P1 (T1, return map).** From the stage-4 T2 rollouts (C0C, P10) and the T1-kin dumps:
  * C0C's measured g(v) lies below the identity over 3–8 m/s, with a fixed point v* below the human's median speed in matched windows;
  * P10's map lies nearer the identity.
  * **Fails if** C0C's g(v) ≥ v over the band. Then the crawl is not contraction, and T1 is wrong.
* **P2 (T1, inertia path in weights).** Reading O's weights, with claims stated from the weights alone and then checked by input ablation on held-out windows:
  * the plan-speed output's gain on the ego-speed history exceeds its gain on the vision tokens by ≥ 5×, as the norm of the Jacobian blocks at the 3–8 m/s windows;
  * the T-code path to plan speed is weak, consistent with F4.
  * **Fails if** the vision gain is comparable (≤ 2×).
* **P3 (T2, frontier).** Every banked downstream lever with a measured (M1 closure, guard) pair lies on or below one monotone frontier. The candidates are LF, A4–A6, lonfloor and the boxes.
  * **Fails if** one dominates another on both axes by more than its interval. That would mean a downstream lever moved the frontier, contradicting T2.
* **P4 (T3, LoRA).** When L exists: the go/stay AUC on L's features rises over V+E on light and lead classes before it rises on the pure self-motion class.

## 4. What runs where (acceleration)
* **Dev box, CPU, now (one Sonnet agent, WSP-READ):** P1, P2 and P3 from banked artifacts, with zero training and no GPU.
* **Pod, now:** frames → cache_pod → smoke → B′ s0/s1 → the LoRA seed 0 (`AMENDMENT_V8F_LORA_4.md`, chain prep running).
* **Box 2 (idle; D: 13.5 GB free, too small for the 94 GB cache_pod, so B′ cannot move there):** its next job is T2 evaluation.
  * **(i)** If P1 and P2 hold: the contraction-debias edit arm, under its own card written from the P2 reading, before any edit is scored. About 1.6–2.5 h per stage, MEASURED stage-4 wall.
  * **(ii)** Then L's T2 stage (`DRAFT_PREREG_T2_DOM_LORA.md`).
* **Theory to settle by measurement, not deference:** if P3 holds and the frontier passes below the dominance corner, only perception can deliver a DOMINANT. The run order already reflects that, and the LoRA mechanism read gains P4's AUC row.
