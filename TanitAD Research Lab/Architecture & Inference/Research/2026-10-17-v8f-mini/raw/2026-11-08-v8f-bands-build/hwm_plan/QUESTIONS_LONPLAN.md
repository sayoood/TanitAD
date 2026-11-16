# QUESTIONS_LONPLAN: PI decisions raised by AMEND_HWMPLAN_3 (LON-PLAN)

Draft, 2026-11-16, Arch+Inference agent, for the Master Mind (MM) to put to the PI. Each question has a recommendation. Everything else in the amendment is the MM's to freeze (its section 10).

## Q1. Should LON-PLAN run now, before the PI chooses between Option A and Option B of ACTRSP Amendment 1?

**Context.**
* Amendment 1 (frozen at `4e122aa5…`, commit `76a8012f`) holds the ACTRSP arms. It recommends Option A, which runs two zero-training levers, LAT-TOK and LON-PLAN.
* LAT-TOK has run: FAIL, admissible.
* LON-PLAN is the second lever. It costs about 8-9.5 h of dev-box CPU wall time, ESTIMATED from the MEASURED phase-1 rates (amendment section 9). It needs no box 2, no GPU and no training. The ACTRSP arms stay held either way.

**Why it bears on the A/B choice.** Option B trains a calibrated 2 s acceleration term into the world model. LON-PLAN asks whether the planner, not the world model, is what keeps 0-2 s acceleration above HOLD. Its result can be read either way:
* If REG on R35 beats HOLD (LON-PLAN-CEILING), Option B's reason is gone.
* If REG only shrinks the plan toward HOLD (SHRINKAGE-ONLY), Option B stays a candidate. The image-constraint amendment comes first, as the next zero-training lever.

**Recommendation: yes, run LON-PLAN now.** It is zero training, it does not use box 2, and it does not pre-empt Option B. It also informs the A/B choice, at about one dev-box CPU day.

## Q2. Should the training-gated next levers be pre-authorised, so that a LON-PLAN outcome does not stall?

**(a) On LON-PLAN-CEILING at both chassis:** the T1-kin read, HWM-PLAN with a predicted goal at λ\*.
* It needs a goal head trained under its own card and gate (HWM-PLAN levers G4 / G5, box 2).
* **Recommendation:** approve it conditionally. It runs only if CEILING holds at both chassis and the goal-head card passes its gate, and the box-2 hours come back to the PI with that card's estimate.

**(b) On LON-PLAN-NULL or SHRINKAGE-ONLY, if the level-2 image constraint (AMEND_HWMPLAN_4) then also fails:** world-model action-sensitivity training on box 2, the lever named in the L1-FULL verdict.
* **Recommendation:** do not pre-approve it. Ask again with the image-constraint result.
* Reason: Amendment 1 section 1.5 measured R35's 2 s acceleration response at 1.20-1.24× the real one through a readout with transition skill. So that lever's case on acceleration is weaker now than when the L1-FULL verdict named it.
