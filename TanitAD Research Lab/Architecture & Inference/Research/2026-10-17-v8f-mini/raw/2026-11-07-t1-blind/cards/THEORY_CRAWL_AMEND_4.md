# THEORY_CRAWL, Amendment 4: T5 fell at P5; the T2 facts give T6 (render-domain information loss), tested by P6 = PREREG_T1_BLIND (Master Mind, 2026-11-07 04:53 dev-box clock; dated amendment beside the frozen note `6e83451c` and Amendments 1 `9d39ac53`, 2 `0427c325`, 3 `dcb82ae9`; written before any T1-kin BLIND rollout exists)

## 1. Where the theory stands ([M] unless named)
* **T1** fell at P1, **T1′** at P1′b, and **T5** at P5. On T5: the partial reliability of the a0 input is λ = 0.968 [0.966, 0.970], above the 0.90 bar, so input noise is too small to explain the shrunken gain (`p5/`). A note on a0: it is a one-frame (0.1 s) backward difference of measured speed. Amendment 3 wrote 0.2 s, and the retraction log records it with its class.
* **T2** (the data-processing frontier) has survived three tests.
  * Both weight edits read HARMS: WSP-GAIN, aimed at the base's gains (`D-V8F-WSPGAIN-1`), and WSP-GAIN-D, aimed at the data's own gains (`D-V8F-WSPGAIN-D-1`). Each closes part of M1 (share_closed 0.25–0.33) and pays in 9–10 guard rows.
  * AUCMAP reads outcome (iii). No gate on the forced-level and floor levers reaches the dominance corner, even at AUC 1.0, and the binding predicate is the lever operator itself (`aucmap/RESULTS_AUCMAP.md`).
* So every admissible lever downstream of the features, and every weight edit tried, sits on or inside the frontier. That is what T2 predicts. A lever that moves the frontier must change what z carries.

## 2. The T2 facts this amendment must explain
* At T2 the crawl gap to P10 is +0.34 / +0.43 on 35 launch scenes (`D-V8F-T2-CRAWL-READ-1`). It is +0.41 / +0.40 for the composite C0C over 48 scenes, and +0.35 / +0.29 for MC, the composite of plain M2b (the attribution read). At T1-kin on eval-141 it is +0.053 / +0.060, on a different population.
* **X2 ≈ 0:** rendering at the logged pose crawls as much as rendering at the executed pose.
* **BLIND ≈ C0 at T2:** replacing every frame's scene tokens by one fixed mean does not move M1 (+0.011 [−0.070, +0.092] / −0.076 [−0.160, +0.014]).
* **Post-training adds 0.06 / 0.11** to the T2 crawl (PT-ACTIVE). Most of the crawl is already in the base.
* **The render gap, descriptive** (`t2v8f/gapscale/`, 2 scenes): a render sits at cosine 0.837 / 0.740 from its own real frame. Two real frames 5 s apart in one clip sit at 0.834, two real frames of different clips at 0.640, and a real frame and the eval mean at 0.806.

## 3. T6. Render-domain information loss [H]
* Let x = E(I) be the scene tokens of a real frame I, and x_r = E(R(s, p)) those of a NuRec render of scene s at pose p. The planner is π(x, h), with h the measured ego history.
* Define the planner's **token dependence** on a population D as Δ(D) = E_D[ d(π(x, h), π(x̄, h)) ], where x̄ is the BLIND mean and d is a driving row (ADE6, or the M1 crawl indicator).
* **T6:** Δ(real) > 0 and Δ(render) ≈ 0. On real frames the plan depends on what the camera sees. Renders land where π's response to x is flat: as far from their own real frame as x̄ is from any real frame. So π treats a render as an uninformative image.
* **Consequence, by data processing:**
  * At T2, the scene information reaching the plan is bounded by I(s; x_r). A lever downstream of x_r cannot raise that bound, and the bound is low exactly where Δ(render) ≈ 0.
  * A perception lever trained on real frames, such as the LoRA arm L, changes E. Whether it moves Δ(render) is not determined by its effect on real frames: that is a separate empirical question [H].
  * So under T6, a T2 dominance read of L can fail for a reason that says nothing about L's driving on real frames.
* **The rival, T6′ (population):** Δ(real) ≈ 0 as well. The planner's driving does not depend on its scene tokens even on real frames. Then the T2 crawl is a function of h and the prior, its larger size is a launch-population effect, and BLIND ≈ C0 at T2 is not a render effect.

## 4. P6 is `PREREG_T1_BLIND.md` (frozen `9619414a`)
* It measures B = BLIND1 − C0 at T1-kin on eval-141, on real frames, at both seeds.
* **VISION-USED** ⇒ T6 survives P6. It is not proven: T2's Δ(render) ≈ 0 rests on the stage-2 BLIND arm, 35 scenes.
* **VISION-UNUSED** ⇒ T6 falls and T6′ stands.
* **MIXED** ⇒ neither is decided, and both gates enter the LoRA card.
* What each outcome sends next is committed in the card's section 5. This amendment adds no outcome, and it changes no bar.

## 5. What T6 predicts beyond P6, to be pre-registered before it is read
* **(a)** On the 48 read scenes, render tokens sit at about the same cosine from the BLIND mean as from their own real frame. GAP48 measures this on box 2 (`t2v8f/gap48/`, descriptive).
* **(b)** An inference-time map from render tokens toward real-token statistics should raise Δ(render) and move the T2 crawl. It must be fitted on scenes disjoint from the ones it is read on, and applied to every arm including P10, which comes from the same forward.
* **(c)** For L: L must pass a T1-kin BLIND gate (it uses its tokens) and a T2 BLIND gate (it keeps that use on renders) before any T2 number of LC can say anything about L's perception.
